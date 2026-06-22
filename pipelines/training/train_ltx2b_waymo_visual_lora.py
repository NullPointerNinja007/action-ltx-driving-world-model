from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx2b-waymo24-visual-lora-r16-train")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
MODELS_VOLUME_NAME = "models"
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dev-waymo24fps-visual-lora-r16-checkpoints",
)
BASELINE_CHECKPOINT_VOLUME_NAME = os.environ.get(
    "LTX_BASELINE_CHECKPOINT_VOLUME_NAME",
    "waymo24-unused-baseline-ckpts",
)

DATA_ROOT = Path("/data")
MODELS_ROOT = Path("/models")
CKPT_ROOT = Path("/checkpoints")
BASELINE_CKPT_ROOT = Path("/baseline_checkpoints")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = os.environ.get("LTX_CKPT_2B", "ltxv-2b-0.9.6-dev-04-25.safetensors")
DEFAULT_LATENT_PREFIX = os.environ.get("WAYMO24_LATENT_PREFIX", "latents")
DEFAULT_ACTION_CONDITIONING = os.environ.get("WAYMO24_ACTION_CONDITIONING", "0") == "1"
DEFAULT_ACTION_ENCODER_TYPE = os.environ.get("WAYMO24_ACTION_ENCODER_TYPE", "global_mlp")
DEFAULT_BASELINE_LORA_RUN_NAME = os.environ.get("WAYMO24_BASELINE_LORA_RUN_NAME", "")
DEFAULT_BASELINE_LORA_STEP = os.environ.get("WAYMO24_BASELINE_LORA_STEP", "")
DEFAULT_FREEZE_TRANSFORMER_LORA = os.environ.get("WAYMO24_FREEZE_TRANSFORMER_LORA", "0") == "1"
DEFAULT_EXPAND_BASELINE_LORA_TO_RANK = os.environ.get("WAYMO24_EXPAND_BASELINE_LORA_TO_RANK", "0") == "1"
DEFAULT_DISABLE_TEXT_CONDITIONING = os.environ.get("WAYMO24_DISABLE_TEXT_CONDITIONING", "0") == "1"
FRAME_ACTION_STATS_RELPATH = "manifests/frame_action_24fps_normalization_stats.json"
FULL112_FRAME_ACTION_STATS_RELPATH = "manifests/frame_action_24fps_full112_normalization_stats.json"
DEFAULT_FRAME_ACTION_FEATURE_KEY = os.environ.get("WAYMO24_FRAME_ACTION_FEATURE_KEY", "actions")
DEFAULT_FRAME_ACTION_STATS_RELPATH = os.environ.get("WAYMO24_FRAME_ACTION_STATS_RELPATH", FRAME_ACTION_STATS_RELPATH)
IMAGE_COND_NOISE_SCALE = float(os.environ.get("LTX_IMAGE_COND_NOISE_SCALE", "0.0"))
DEFAULT_TRAIN_GPU = os.environ.get("LTX_MODAL_TRAIN_GPU", "H100")
MIDBLOCK_GATED_XATTN_MODE = "frame_midblock_gated_xattn"
TEMPORAL_BOTTLENECK_HF_TEACHER_MODE = "frame_temporal_bottleneck_hf_teacher"
TEMPORAL_BOTTLENECK_LOWFREQ_V3_MODE = "frame_temporal_bottleneck_lowfreq_v3"
TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE = "frame_temporal_bottleneck_fullaction_motion_v4"
TEMPORAL_BOTTLENECK_ACTION_MODES = {
    TEMPORAL_BOTTLENECK_HF_TEACHER_MODE,
    TEMPORAL_BOTTLENECK_LOWFREQ_V3_MODE,
    TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE,
}
BOUNDED_TEMPORAL_BOTTLENECK_ACTION_MODES = {
    TEMPORAL_BOTTLENECK_LOWFREQ_V3_MODE,
    TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE,
}

FPS = 24
WIDTH = 512
HEIGHT = 512
TOTAL_FRAMES = 121
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
CONTEXT_LATENT_FRAMES = CONTEXT_FRAMES // 8 + 1
LORA_RANK = 16

DEFAULT_PROMPT = (
    "Forward-facing autonomous driving video from a real Waymo-style car-mounted front camera. "
    "Preserve realistic road geometry, lane markings, traffic lights, vehicles, sidewalks, buildings, "
    "lighting, weather, and stable ego-vehicle motion."
)
DEFAULT_NEGATIVE_PROMPT = (
    "scene restart, camera cut, new location, wrong viewpoint, rear camera, side camera, blurry, jittery, "
    "distorted, impossible motion, duplicated cars"
)

CHECKPOINT_STEPS = [0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    max_steps: int
    max_train_hours: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    save_steps: list[int]
    sample_steps: list[int]
    lora_rank: int
    lora_alpha: int
    prompt: str
    negative_prompt: str
    disable_text_conditioning: bool
    train_manifest: str
    val_manifest: str
    latent_prefix: str
    base_checkpoint: str
    num_val_samples: int
    action_conditioning: bool = False
    action_encoder_type: str = "global_mlp"
    action_dim: int = 112
    action_token_count: int = 24
    action_hidden_dim: int = 1024
    action_dropout: float = 0.0
    action_learning_rate: float = 1e-4
    action_injector_learning_rate: float = 1e-4
    action_gate_learning_rate: float = 1e-3
    action_injector_heads: int = 8
    action_midblock_start: int = -1
    action_midblock_end: int = -1
    action_transformer_layers: int = 2
    action_transformer_heads: int = 8
    action_frame_count: int = TOTAL_FRAMES
    frame_action_stats_relpath: str = FRAME_ACTION_STATS_RELPATH
    frame_action_feature_key: str = "actions"
    baseline_lora_run_name: str = ""
    baseline_lora_step: str = ""
    freeze_transformer_lora: bool = False
    expand_baseline_lora_to_rank: bool = False
    skip_missing_latents: bool = False
    timestep_sampling: str = "uniform"
    timestep_lognormal_mean: float = 0.0
    timestep_lognormal_std: float = 1.0
    hf_teacher_loss_weight: float = 0.0
    diffusion_loss_weight: float = 1.0
    lowfreq_target_loss_weight: float = 0.0
    lowfreq_delta_loss_weight: float = 0.0
    action_motion_aux_loss_weight: float = 0.0
    action_residual_loss_weight: float = 0.0
    action_gate_loss_weight: float = 0.0
    action_gate_scale: float = 1.0
    action_gate_bound: float = 0.25
    resume_from_run_name: str = ""
    resume_from_checkpoint: str = ""
    external_stop_relpath: str = ""
    external_stop_check_steps: int = 100


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
baseline_checkpoint_volume = modal.Volume.from_name(BASELINE_CHECKPOINT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "huggingface_hub",
        "av",
        "imageio",
        "imageio-ffmpeg",
        "imageio[ffmpeg]",
        "safetensors",
        "peft",
        "accelerate",
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video",
        "cd /workspace/LTX-Video && python -m pip install -e '.[inference-script]'",
    )
)


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_checkpoint_symlink(ckpt_name: str = CKPT_2B) -> Path:
    src = MODELS_ROOT / "ltx" / ckpt_name
    dst = REPO / ckpt_name
    if not src.exists():
        raise FileNotFoundError(f"Missing checkpoint in Modal volume: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return dst


def load_manifest(path: Path, limit: int = 0) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit > 0 else rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def replace_latent_prefix(latent_relpath: str, latent_prefix: str) -> str:
    parts = Path(latent_relpath).parts
    if not parts:
        raise ValueError("Empty latent_relpath")
    return str(Path(latent_prefix, *parts[1:]))


class LatentRowsDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        latent_prefix: str,
        frame_action_stats_relpath: str = "",
        frame_action_feature_key: str = "actions",
    ):
        self.rows = rows
        self.latent_prefix = latent_prefix
        self.frame_action_feature_key = frame_action_feature_key
        self.frame_action_stats = self._load_frame_action_stats(frame_action_stats_relpath)

    def _load_frame_action_stats(self, frame_action_stats_relpath: str) -> dict[str, Any] | None:
        if not frame_action_stats_relpath:
            return None
        stats_path = DATA_ROOT / frame_action_stats_relpath
        if not stats_path.exists():
            return None
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return {
            "mean": stats["mean"],
            "std": stats["std"],
            "p01": stats["p01"],
            "p99": stats["p99"],
            "feature_order": stats.get("feature_order", []),
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file

        row = self.rows[idx]
        latent_path = DATA_ROOT / replace_latent_prefix(row["latent_relpath"], self.latent_prefix)
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent cache: {latent_path}")
        latents = load_file(str(latent_path))["latents"].to(torch.bfloat16)
        item = {"latents": latents, "row": row}
        if "frame_action_relpath" in row and row["frame_action_relpath"]:
            import numpy as np

            action_path = DATA_ROOT / row["frame_action_relpath"]
            if not action_path.exists():
                raise FileNotFoundError(f"Missing frame-action cache: {action_path}")
            with np.load(action_path) as payload:
                if self.frame_action_feature_key not in payload:
                    raise KeyError(
                        f"Missing frame-action array key {self.frame_action_feature_key!r} in {action_path}; "
                        f"available keys={list(payload.keys())}"
                    )
                actions_np = payload[self.frame_action_feature_key].astype("float32")
            action_frames = torch.from_numpy(actions_np)
            if self.frame_action_stats is not None:
                mean = torch.tensor(self.frame_action_stats["mean"], dtype=torch.float32)
                std = torch.tensor(self.frame_action_stats["std"], dtype=torch.float32).clamp_min(1e-6)
                p01 = torch.tensor(self.frame_action_stats["p01"], dtype=torch.float32)
                p99 = torch.tensor(self.frame_action_stats["p99"], dtype=torch.float32)
                if mean.numel() != action_frames.shape[-1]:
                    raise ValueError(
                        f"Frame-action stats dim {mean.numel()} does not match "
                        f"{self.frame_action_feature_key!r} dim {action_frames.shape[-1]} for {action_path}."
                    )
                action_frames = torch.minimum(torch.maximum(action_frames, p01), p99)
                action_frames = ((action_frames - mean) / std).clamp(-5.0, 5.0)
            item["action_vector"] = action_frames
        if "action_vector_json" in row and row["action_vector_json"]:
            action_vector = torch.tensor(json.loads(row["action_vector_json"]), dtype=torch.float32)
            item["action_vector"] = action_vector
        return item


def make_batch(dataset: LatentRowsDataset, indices: list[int]) -> dict[str, Any]:
    import torch

    items = [dataset[i] for i in indices]
    batch = {
        "latents": torch.stack([item["latents"] for item in items], dim=0),
        "rows": [item["row"] for item in items],
    }
    if all("action_vector" in item for item in items):
        batch["action_vectors"] = torch.stack([item["action_vector"] for item in items], dim=0)
    return batch


def filter_rows_with_existing_latents(rows: list[dict[str, str]], latent_prefix: str) -> tuple[list[dict[str, str]], int]:
    kept = []
    skipped = 0
    for row in rows:
        latent_path = DATA_ROOT / replace_latent_prefix(row["latent_relpath"], latent_prefix)
        if latent_path.exists():
            kept.append(row)
        else:
            skipped += 1
    return kept, skipped


def get_transformer_core(transformer):
    return transformer.base_model.model if hasattr(transformer, "base_model") else transformer


def get_transformer_inner_dim(transformer) -> int:
    core = get_transformer_core(transformer)
    if not hasattr(core, "inner_dim"):
        raise AttributeError("Expected LTX transformer core to expose `inner_dim`.")
    return int(core.inner_dim)


def install_action_adaln_hook(transformer) -> None:
    """Add a trainable action delta to LTX's existing per-block AdaLN timestep embedding."""
    core = get_transformer_core(transformer)
    if getattr(core, "_waymo_action_adaln_hook_installed", False):
        return
    original_forward = core.adaln_single.forward

    def forward_with_action_adaln(self_adaln, *args, **kwargs):
        timestep, embedded_timestep = original_forward(*args, **kwargs)
        action_delta = getattr(core, "_waymo_action_adaln_delta", None)
        if action_delta is None:
            return timestep, embedded_timestep

        batch_size = kwargs.get("batch_size")
        if batch_size is None:
            batch_size = action_delta.shape[0]
        timestep_view = timestep.view(batch_size, -1, timestep.shape[-1])
        delta = action_delta.to(device=timestep.device, dtype=timestep.dtype)
        if delta.shape[-1] != timestep_view.shape[-1]:
            raise ValueError(f"AdaLN action dim {delta.shape[-1]} != timestep dim {timestep_view.shape[-1]}")
        if delta.shape[0] != batch_size:
            if batch_size % delta.shape[0] != 0:
                raise ValueError(f"Cannot broadcast AdaLN action batch {delta.shape[0]} to timestep batch {batch_size}")
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        timestep = (timestep_view + delta).reshape_as(timestep)
        return timestep, embedded_timestep

    core.adaln_single.forward = types.MethodType(forward_with_action_adaln, core.adaln_single)
    core._waymo_action_adaln_delta = None
    core._waymo_action_adaln_hook_installed = True


def set_action_adaln_delta(transformer, action_delta) -> None:
    core = get_transformer_core(transformer)
    core._waymo_action_adaln_delta = action_delta


def clear_action_adaln_delta(transformer) -> None:
    core = get_transformer_core(transformer)
    if hasattr(core, "_waymo_action_adaln_delta"):
        core._waymo_action_adaln_delta = None


def get_transformer_blocks(transformer):
    core = get_transformer_core(transformer)
    blocks = getattr(core, "transformer_blocks", None)
    if blocks is None:
        raise AttributeError("Expected LTX transformer core to expose `transformer_blocks`.")
    if len(blocks) < 3:
        raise ValueError(f"Expected at least 3 transformer blocks for middle-block routing, got {len(blocks)}.")
    return core, blocks


def resolve_midblock_indices(num_blocks: int, start: int = -1, end: int = -1) -> list[int]:
    resolved_start = num_blocks // 3 if start < 0 else start
    resolved_end = math.ceil((2 * num_blocks) / 3) if end < 0 else end
    if not (0 <= resolved_start < resolved_end <= num_blocks):
        raise ValueError(
            f"Invalid middle-block range start={resolved_start}, end={resolved_end}, num_blocks={num_blocks}."
        )
    return list(range(resolved_start, resolved_end))


def make_midblock_action_injector(config: TrainConfig, transformer):
    import torch

    _, blocks = get_transformer_blocks(transformer)
    hidden_dim = get_transformer_inner_dim(transformer)
    num_heads = int(config.action_injector_heads)
    if num_heads <= 0 or hidden_dim % num_heads != 0:
        raise ValueError(f"action_injector_heads={num_heads} must divide transformer hidden_dim={hidden_dim}.")
    block_indices = resolve_midblock_indices(
        len(blocks),
        start=int(config.action_midblock_start),
        end=int(config.action_midblock_end),
    )

    class MidBlockActionInjector(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            block_indices: list[int],
            num_heads: int,
            dropout: float,
        ):
            super().__init__()
            self.conditioning_mode = "midblock_gated_xattn"
            self.hidden_dim = hidden_dim
            self.block_indices = list(block_indices)
            self.num_heads = num_heads
            self.dropout = dropout
            self.layers = torch.nn.ModuleDict()
            self.gates = torch.nn.ParameterDict()
            for block_idx in self.block_indices:
                key = str(block_idx)
                self.layers[key] = torch.nn.ModuleDict(
                    {
                        "norm": torch.nn.LayerNorm(hidden_dim),
                        "attn": torch.nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "dropout": torch.nn.Dropout(dropout),
                    }
                )
                # Gate zero means checkpoint 0 is exactly the corrected no-action path.
                self.gates[key] = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward_block(self, block_idx: int, hidden_states, action_tokens):
            key = str(block_idx)
            if action_tokens is None or key not in self.layers:
                return hidden_states
            if hidden_states.ndim != 3:
                raise ValueError(f"Expected block hidden states [B,N,D], got shape={tuple(hidden_states.shape)}")
            if action_tokens.ndim != 3:
                raise ValueError(f"Expected action tokens [B,T,D], got shape={tuple(action_tokens.shape)}")
            if action_tokens.shape[0] != hidden_states.shape[0]:
                raise ValueError(f"Action batch {action_tokens.shape[0]} != hidden batch {hidden_states.shape[0]}")
            if action_tokens.shape[-1] != self.hidden_dim or hidden_states.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"Hidden/action dims must equal {self.hidden_dim}; "
                    f"got hidden={hidden_states.shape[-1]}, action={action_tokens.shape[-1]}"
                )
            layer = self.layers[key]
            original_dtype = hidden_states.dtype
            query = layer["norm"](hidden_states.float())
            key_value = action_tokens.float()
            attn_out, _ = layer["attn"](query, key_value, key_value, need_weights=False)
            gated = self.gates[key].float() * layer["dropout"](attn_out)
            return hidden_states + gated.to(dtype=original_dtype)

        def gate_parameters(self):
            return list(self.gates.parameters())

        def xattn_parameters(self):
            gate_ids = {id(param) for param in self.gates.parameters()}
            return [param for param in self.parameters() if id(param) not in gate_ids]

        def metadata(self) -> dict[str, Any]:
            return {
                "conditioning_mode": self.conditioning_mode,
                "hidden_dim": self.hidden_dim,
                "block_indices": self.block_indices,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }

    return MidBlockActionInjector(
        hidden_dim=hidden_dim,
        block_indices=block_indices,
        num_heads=num_heads,
        dropout=float(config.action_dropout),
    ).to("cuda")


def remove_midblock_action_injector(transformer) -> None:
    core = get_transformer_core(transformer)
    for handle in getattr(core, "_waymo_midblock_action_handles", []) or []:
        handle.remove()
    core._waymo_midblock_action_handles = []
    core._waymo_midblock_action_tokens = None


def install_midblock_action_injector(transformer, action_injector) -> None:
    core, blocks = get_transformer_blocks(transformer)
    if getattr(core, "_waymo_midblock_action_injector", None) is action_injector:
        return
    remove_midblock_action_injector(transformer)
    core._waymo_midblock_action_injector = action_injector
    core._waymo_midblock_action_tokens = None
    handles = []

    def make_hook(block_idx: int):
        def hook(_module, _inputs, output):
            action_tokens = getattr(core, "_waymo_midblock_action_tokens", None)
            if action_tokens is None:
                return output
            if isinstance(output, tuple):
                hidden = action_injector.forward_block(block_idx, output[0], action_tokens)
                return (hidden, *output[1:])
            return action_injector.forward_block(block_idx, output, action_tokens)

        return hook

    for block_idx in action_injector.block_indices:
        if block_idx < 0 or block_idx >= len(blocks):
            raise ValueError(f"Action injector block index {block_idx} is outside [0, {len(blocks)}).")
        handles.append(blocks[block_idx].register_forward_hook(make_hook(block_idx)))
    core._waymo_midblock_action_handles = handles
    print(
        json.dumps(
            {
                "installed_midblock_action_injector": True,
                "num_transformer_blocks": len(blocks),
                **action_injector.metadata(),
            },
            sort_keys=True,
        )
    )


def set_midblock_action_tokens(transformer, action_tokens) -> None:
    core = get_transformer_core(transformer)
    if getattr(core, "_waymo_midblock_action_injector", None) is None:
        raise RuntimeError("Mid-block action tokens were set before installing the action injector.")
    core._waymo_midblock_action_tokens = action_tokens


def clear_midblock_action_tokens(transformer) -> None:
    core = get_transformer_core(transformer)
    if hasattr(core, "_waymo_midblock_action_tokens"):
        core._waymo_midblock_action_tokens = None


def latent_time_ids_from_indices_grid(indices_grid):
    import torch

    if indices_grid is None:
        raise RuntimeError("Temporal bottleneck action injection requires transformer indices_grid.")
    coords = indices_grid
    if coords.ndim == 3:
        coords = coords[0]
    if coords.ndim != 2 or coords.shape[-1] < 1:
        raise ValueError(f"Expected indices_grid shape [N,C] or [B,N,C], got {tuple(indices_grid.shape)}")
    time_values = coords[0] if coords.shape[0] <= 4 and coords.shape[1] > coords.shape[0] else coords[:, 0]
    _, inverse = torch.unique(time_values.to(torch.float32), sorted=True, return_inverse=True)
    return inverse.to(device=coords.device, dtype=torch.long)


def make_temporal_bottleneck_action_injector(config: TrainConfig, transformer):
    import torch
    import torch.nn.functional as F

    _, blocks = get_transformer_blocks(transformer)
    hidden_dim = get_transformer_inner_dim(transformer)
    block_indices = [10, 14, 18]
    if config.action_midblock_start >= 0 or config.action_midblock_end >= 0:
        block_indices = resolve_midblock_indices(
            len(blocks),
            start=int(config.action_midblock_start),
            end=int(config.action_midblock_end),
        )
    invalid = [idx for idx in block_indices if idx < 0 or idx >= len(blocks)]
    if invalid:
        raise ValueError(f"Temporal bottleneck block indices {invalid} are outside [0, {len(blocks)}).")

    class TemporalBottleneckActionInjector(torch.nn.Module):
        def __init__(
            self,
            action_hidden_dim: int,
            hidden_dim: int,
            block_indices: list[int],
            gate_scale: float,
            *,
            bounded_gates: bool,
            gate_bound: float,
            projector_final_std: float,
        ):
            super().__init__()
            self.conditioning_mode = "temporal_bottleneck"
            self.action_hidden_dim = action_hidden_dim
            self.hidden_dim = hidden_dim
            self.block_indices = list(block_indices)
            self.gate_scale = float(gate_scale)
            self.bounded_gates = bool(bounded_gates)
            self.gate_bound = float(gate_bound)
            self.projector_final_std = float(projector_final_std)
            self.projectors = torch.nn.ModuleDict()
            self.gates = torch.nn.ParameterDict()
            self._residual_norm_terms = []
            for block_idx in self.block_indices:
                key = str(block_idx)
                self.projectors[key] = torch.nn.Sequential(
                    torch.nn.LayerNorm(action_hidden_dim),
                    torch.nn.Linear(action_hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
                torch.nn.init.normal_(self.projectors[key][-1].weight, mean=0.0, std=self.projector_final_std)
                torch.nn.init.zeros_(self.projectors[key][-1].bias)
                self.gates[key] = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def reset_regularization_state(self) -> None:
            self._residual_norm_terms = []

        def _pool_frame_controls(self, frame_controls, latent_time_ids):
            if frame_controls.ndim != 3:
                raise ValueError(f"Expected frame controls [B,T,D], got shape={tuple(frame_controls.shape)}")
            num_latent_times = int(latent_time_ids.max().detach().cpu().item()) + 1
            if num_latent_times <= 0:
                raise ValueError("Expected at least one latent time bin.")
            pooled = F.adaptive_avg_pool1d(frame_controls.float().transpose(1, 2), num_latent_times).transpose(1, 2)
            return pooled

        def forward_block(self, block_idx: int, hidden_states, frame_controls, latent_time_ids):
            key = str(block_idx)
            if frame_controls is None or key not in self.projectors:
                return hidden_states
            if hidden_states.ndim != 3:
                raise ValueError(f"Expected block hidden states [B,N,D], got shape={tuple(hidden_states.shape)}")
            if latent_time_ids.ndim != 1 or latent_time_ids.shape[0] != hidden_states.shape[1]:
                raise ValueError(
                    f"Expected latent_time_ids [N] matching hidden tokens; got ids={tuple(latent_time_ids.shape)} "
                    f"hidden={tuple(hidden_states.shape)}"
                )
            if frame_controls.shape[0] != hidden_states.shape[0]:
                raise ValueError(f"Action batch {frame_controls.shape[0]} != hidden batch {hidden_states.shape[0]}")
            original_dtype = hidden_states.dtype
            latent_time_ids = latent_time_ids.to(device=hidden_states.device, dtype=torch.long)
            controls_by_time = self._pool_frame_controls(frame_controls.to(hidden_states.device), latent_time_ids)
            residual_by_time = self.projectors[key](controls_by_time)
            self._residual_norm_terms.append(residual_by_time.float().pow(2).mean())
            residual = residual_by_time[:, latent_time_ids, :]
            raw_gate = self.gates[key].float()
            if self.bounded_gates:
                effective_gate = float(self.gate_scale) * float(self.gate_bound) * torch.tanh(raw_gate)
            else:
                effective_gate = float(self.gate_scale) * raw_gate
            gated = effective_gate * residual
            return hidden_states + gated.to(dtype=original_dtype)

        def gate_parameters(self):
            return list(self.gates.parameters())

        def xattn_parameters(self):
            gate_ids = {id(param) for param in self.gates.parameters()}
            return [param for param in self.parameters() if id(param) not in gate_ids]

        def residual_norm_loss(self, device):
            if not self._residual_norm_terms:
                return torch.zeros((), device=device, dtype=torch.float32)
            return torch.stack([term.to(device=device, dtype=torch.float32) for term in self._residual_norm_terms]).mean()

        def gate_loss(self, device):
            if len(self.gates) == 0:
                return torch.zeros((), device=device, dtype=torch.float32)
            if self.bounded_gates:
                values = [float(self.gate_bound) * torch.tanh(gate.float()).abs().to(device) for gate in self.gates.values()]
            else:
                values = [gate.float().abs().to(device) for gate in self.gates.values()]
            return torch.stack(values).mean()

        def gate_stats(self) -> dict[str, float]:
            if len(self.gates) == 0:
                return {"mean_abs_gate": 0.0, "max_abs_gate": 0.0, "mean_abs_raw_gate": 0.0, "max_abs_raw_gate": 0.0}
            raw_values = [float(gate.detach().float().abs().cpu()) for gate in self.gates.values()]
            if self.bounded_gates:
                values = [
                    float(self.gate_bound) * float(torch.tanh(gate.detach().float()).abs().cpu())
                    for gate in self.gates.values()
                ]
            else:
                values = raw_values
            return {
                "mean_abs_gate": sum(values) / len(values),
                "max_abs_gate": max(values),
                "mean_abs_raw_gate": sum(raw_values) / len(raw_values),
                "max_abs_raw_gate": max(raw_values),
            }

        def metadata(self) -> dict[str, Any]:
            return {
                "conditioning_mode": self.conditioning_mode,
                "action_hidden_dim": self.action_hidden_dim,
                "hidden_dim": self.hidden_dim,
                "block_indices": self.block_indices,
                "gate_scale": self.gate_scale,
                "bounded_gates": self.bounded_gates,
                "gate_bound": self.gate_bound,
                "projector_final_std": self.projector_final_std,
            }

    has_bounded_gates = config.action_encoder_type in BOUNDED_TEMPORAL_BOTTLENECK_ACTION_MODES
    return TemporalBottleneckActionInjector(
        action_hidden_dim=int(config.action_hidden_dim),
        hidden_dim=hidden_dim,
        block_indices=block_indices,
        gate_scale=float(config.action_gate_scale),
        bounded_gates=has_bounded_gates,
        gate_bound=float(config.action_gate_bound),
        projector_final_std=1e-3,
    ).to("cuda")


def remove_temporal_bottleneck_action_injector(transformer) -> None:
    core = get_transformer_core(transformer)
    for handle in getattr(core, "_waymo_temporal_bottleneck_action_handles", []) or []:
        handle.remove()
    core._waymo_temporal_bottleneck_action_handles = []
    core._waymo_temporal_bottleneck_frame_controls = None
    core._waymo_temporal_bottleneck_latent_time_ids = None


def install_temporal_bottleneck_action_injector(transformer, action_injector) -> None:
    core, blocks = get_transformer_blocks(transformer)
    if getattr(core, "_waymo_temporal_bottleneck_action_injector", None) is action_injector:
        return
    remove_temporal_bottleneck_action_injector(transformer)
    core._waymo_temporal_bottleneck_action_injector = action_injector
    core._waymo_temporal_bottleneck_frame_controls = None
    core._waymo_temporal_bottleneck_latent_time_ids = None
    handles = []

    if not getattr(core, "_waymo_temporal_bottleneck_forward_wrapped", False):
        original_forward = core.forward

        def forward_with_temporal_bottleneck(*args, **kwargs):
            frame_controls = getattr(core, "_waymo_temporal_bottleneck_frame_controls", None)
            if frame_controls is not None:
                core._waymo_temporal_bottleneck_latent_time_ids = latent_time_ids_from_indices_grid(
                    kwargs.get("indices_grid")
                )
            try:
                return original_forward(*args, **kwargs)
            finally:
                core._waymo_temporal_bottleneck_latent_time_ids = None

        core.forward = forward_with_temporal_bottleneck
        core._waymo_temporal_bottleneck_forward_wrapped = True

    def make_hook(block_idx: int):
        def hook(_module, _inputs, output):
            frame_controls = getattr(core, "_waymo_temporal_bottleneck_frame_controls", None)
            latent_time_ids = getattr(core, "_waymo_temporal_bottleneck_latent_time_ids", None)
            if frame_controls is None or latent_time_ids is None:
                return output
            if isinstance(output, tuple):
                hidden = action_injector.forward_block(block_idx, output[0], frame_controls, latent_time_ids)
                return (hidden, *output[1:])
            return action_injector.forward_block(block_idx, output, frame_controls, latent_time_ids)

        return hook

    for block_idx in action_injector.block_indices:
        if block_idx < 0 or block_idx >= len(blocks):
            raise ValueError(f"Action injector block index {block_idx} is outside [0, {len(blocks)}).")
        handles.append(blocks[block_idx].register_forward_hook(make_hook(block_idx)))
    core._waymo_temporal_bottleneck_action_handles = handles
    print(
        json.dumps(
            {
                "installed_temporal_bottleneck_action_injector": True,
                "num_transformer_blocks": len(blocks),
                **action_injector.metadata(),
            },
            sort_keys=True,
        )
    )


def set_temporal_bottleneck_action_controls(transformer, frame_controls) -> None:
    core = get_transformer_core(transformer)
    if getattr(core, "_waymo_temporal_bottleneck_action_injector", None) is None:
        raise RuntimeError("Temporal bottleneck action controls were set before installing the injector.")
    core._waymo_temporal_bottleneck_frame_controls = frame_controls


def clear_temporal_bottleneck_action_controls(transformer) -> None:
    core = get_transformer_core(transformer)
    if hasattr(core, "_waymo_temporal_bottleneck_frame_controls"):
        core._waymo_temporal_bottleneck_frame_controls = None


def make_global_action_token_encoder(config: TrainConfig, output_dim: int):
    import torch

    class GlobalActionTokenEncoder(torch.nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            token_count: int,
            output_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.conditioning_mode = "tokens"
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.token_count = token_count
            self.output_dim = output_dim
            self.input_norm = torch.nn.LayerNorm(input_dim)
            self.state_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
            )
            self.token_embeddings = torch.nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            state = self.state_mlp(self.input_norm(action_vectors.float()))
            token_hidden = self.token_embeddings + state[:, None, :]
            return self.token_proj(token_hidden)

    return GlobalActionTokenEncoder(
        input_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        token_count=config.action_token_count,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_adaln_action_encoder(config: TrainConfig, inner_dim: int):
    import torch

    output_dim = 6 * inner_dim

    class AdaLNActionEncoder(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.conditioning_mode = "adaln"
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.input_norm = torch.nn.LayerNorm(input_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            # AdaLN-zero: checkpoint 0 is exactly the unconditioned transformer path.
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            return self.net(self.input_norm(action_vectors.float())).unsqueeze(1)

    return AdaLNActionEncoder(
        input_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_temporal_per_point_action_encoder(config: TrainConfig, output_dim: int):
    import torch

    if config.action_dim != 112:
        raise ValueError(f"Temporal per-point encoder expects action_dim=112, got {config.action_dim}")
    if config.action_token_count != 24:
        raise ValueError(
            f"Temporal per-point encoder emits exactly 24 tokens; got action_token_count={config.action_token_count}"
        )

    class TemporalPerPointActionEncoder(torch.nn.Module):
        def __init__(self, hidden_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.conditioning_mode = "tokens"
            self.input_dim = 112
            self.hidden_dim = hidden_dim
            self.token_count = 24
            self.output_dim = output_dim
            self.future_point_norm = torch.nn.LayerNorm(5)
            self.summary_norm = torch.nn.LayerNorm(16)
            self.future_mlp = torch.nn.Sequential(
                torch.nn.Linear(5, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
            )
            self.summary_mlp = torch.nn.Sequential(
                torch.nn.Linear(16, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
            )
            self.future_time_embeddings = torch.nn.Parameter(torch.randn(1, 20, hidden_dim) * 0.02)
            self.summary_token_embeddings = torch.nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            action_vectors = action_vectors.float()
            future_x = action_vectors[:, 0:20]
            future_y = action_vectors[:, 20:40]
            past_vel_x = action_vectors[:, 40:56]
            past_vel_y = action_vectors[:, 56:72]
            past_accel_x = action_vectors[:, 72:88]
            past_accel_y = action_vectors[:, 88:104]
            current_state = action_vectors[:, 104:112]

            future_xy = torch.stack([future_x, future_y], dim=-1)
            first_delta = future_xy[:, :1]
            future_delta = torch.cat([first_delta, future_xy[:, 1:] - future_xy[:, :-1]], dim=1)
            time = torch.linspace(0.0, 1.0, 20, device=action_vectors.device, dtype=action_vectors.dtype)
            time = time.view(1, 20, 1).expand(action_vectors.shape[0], -1, -1)
            future_features = torch.cat([future_xy, future_delta, time], dim=-1)
            future_hidden = self.future_mlp(self.future_point_norm(future_features)) + self.future_time_embeddings

            past_vel = torch.stack([past_vel_x, past_vel_y], dim=-1)
            past_accel = torch.stack([past_accel_x, past_accel_y], dim=-1)
            vel_stats = torch.cat(
                [
                    past_vel.mean(dim=1),
                    past_vel.std(dim=1, unbiased=False),
                    past_vel[:, 0],
                    past_vel[:, -1],
                ],
                dim=-1,
            )
            accel_stats = torch.cat(
                [
                    past_accel.mean(dim=1),
                    past_accel.std(dim=1, unbiased=False),
                    past_accel[:, 0],
                    past_accel[:, -1],
                ],
                dim=-1,
            )
            current_summary = torch.cat(
                [
                    current_state,
                    past_vel[:, -1],
                    past_accel[:, -1],
                    future_xy[:, 0],
                    future_delta[:, 0],
                ],
                dim=-1,
            )
            future_summary = torch.cat(
                [
                    current_state,
                    future_xy[:, 0],
                    future_xy[:, -1],
                    future_delta.mean(dim=1),
                    future_delta[:, -1],
                ],
                dim=-1,
            )
            summary_features = torch.stack(
                [
                    torch.cat([current_state, vel_stats], dim=-1),
                    torch.cat([current_state, accel_stats], dim=-1),
                    current_summary,
                    future_summary,
                ],
                dim=1,
            )
            summary_hidden = self.summary_mlp(self.summary_norm(summary_features)) + self.summary_token_embeddings
            token_hidden = torch.cat([future_hidden, summary_hidden], dim=1)
            return self.token_proj(token_hidden)

    return TemporalPerPointActionEncoder(
        hidden_dim=config.action_hidden_dim,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_tiny_transformer_action_encoder(config: TrainConfig, output_dim: int):
    import torch

    if config.action_dim != 112:
        raise ValueError(f"Tiny transformer encoder expects action_dim=112, got {config.action_dim}")

    class TinyTransformerActionEncoder(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            token_count: int,
            dropout: float,
            num_layers: int,
            num_heads: int,
        ):
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
            self.conditioning_mode = "tokens"
            self.input_dim = 112
            self.hidden_dim = hidden_dim
            self.token_count = token_count
            self.output_dim = output_dim
            self.future_norm = torch.nn.LayerNorm(5)
            self.past_norm = torch.nn.LayerNorm(5)
            self.current_norm = torch.nn.LayerNorm(8)
            self.future_proj = torch.nn.Linear(5, hidden_dim)
            self.past_proj = torch.nn.Linear(5, hidden_dim)
            self.current_proj = torch.nn.Linear(8, hidden_dim)
            self.query_embeddings = torch.nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
            self.future_time_embeddings = torch.nn.Parameter(torch.randn(1, 20, hidden_dim) * 0.02)
            self.past_time_embeddings = torch.nn.Parameter(torch.randn(1, 16, hidden_dim) * 0.02)
            self.current_token_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            action_vectors = action_vectors.float()
            future_x = action_vectors[:, 0:20]
            future_y = action_vectors[:, 20:40]
            past_vel_x = action_vectors[:, 40:56]
            past_vel_y = action_vectors[:, 56:72]
            past_accel_x = action_vectors[:, 72:88]
            past_accel_y = action_vectors[:, 88:104]
            current_state = action_vectors[:, 104:112]

            future_xy = torch.stack([future_x, future_y], dim=-1)
            first_delta = future_xy[:, :1]
            future_delta = torch.cat([first_delta, future_xy[:, 1:] - future_xy[:, :-1]], dim=1)
            future_time = torch.linspace(0.0, 1.0, 20, device=action_vectors.device, dtype=action_vectors.dtype)
            future_time = future_time.view(1, 20, 1).expand(action_vectors.shape[0], -1, -1)
            future_features = torch.cat([future_xy, future_delta, future_time], dim=-1)
            future_tokens = self.future_proj(self.future_norm(future_features)) + self.future_time_embeddings

            past_time = torch.linspace(-1.0, 0.0, 16, device=action_vectors.device, dtype=action_vectors.dtype)
            past_time = past_time.view(1, 16, 1).expand(action_vectors.shape[0], -1, -1)
            past_features = torch.stack([past_vel_x, past_vel_y, past_accel_x, past_accel_y], dim=-1)
            past_features = torch.cat([past_features, past_time], dim=-1)
            past_tokens = self.past_proj(self.past_norm(past_features)) + self.past_time_embeddings

            current_token = self.current_proj(self.current_norm(current_state)).unsqueeze(1)
            current_token = current_token + self.current_token_embedding

            source_tokens = torch.cat([future_tokens, past_tokens, current_token], dim=1)
            query_tokens = self.query_embeddings.expand(action_vectors.shape[0], -1, -1)
            hidden = self.transformer(torch.cat([query_tokens, source_tokens], dim=1))
            return self.token_proj(hidden[:, : self.token_count])

    return TinyTransformerActionEncoder(
        hidden_dim=config.action_hidden_dim,
        output_dim=output_dim,
        token_count=config.action_token_count,
        dropout=config.action_dropout,
        num_layers=config.action_transformer_layers,
        num_heads=config.action_transformer_heads,
    ).to("cuda")


def frame_segment_ids(frame_count: int, context_frames: int):
    import torch

    ids = torch.zeros(frame_count, dtype=torch.long)
    ids[context_frames:] = 1
    return ids


def make_frame_global_mlp_action_encoder(config: TrainConfig, output_dim: int):
    import torch

    class FrameGlobalMLPActionEncoder(torch.nn.Module):
        def __init__(
            self,
            frame_count: int,
            action_dim: int,
            hidden_dim: int,
            token_count: int,
            output_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.conditioning_mode = "tokens"
            self.input_dim = action_dim
            self.frame_count = frame_count
            self.hidden_dim = hidden_dim
            self.token_count = token_count
            self.output_dim = output_dim
            flat_dim = frame_count * action_dim
            self.input_norm = torch.nn.LayerNorm(flat_dim)
            self.state_mlp = torch.nn.Sequential(
                torch.nn.Linear(flat_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
            )
            self.token_embeddings = torch.nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.ndim != 3:
                raise ValueError(f"Expected frame actions [B,T,D], got shape={tuple(action_vectors.shape)}")
            if action_vectors.shape[1] != self.frame_count or action_vectors.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected frame actions [B,{self.frame_count},{self.input_dim}], "
                    f"got shape={tuple(action_vectors.shape)}"
                )
            flat = action_vectors.float().reshape(action_vectors.shape[0], -1)
            state = self.state_mlp(self.input_norm(flat))
            token_hidden = self.token_embeddings + state[:, None, :]
            return self.token_proj(token_hidden)

    return FrameGlobalMLPActionEncoder(
        frame_count=config.action_frame_count,
        action_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        token_count=config.action_token_count,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_frame_temporal_pool_action_encoder(config: TrainConfig, output_dim: int):
    import torch

    class FrameTemporalPoolActionEncoder(torch.nn.Module):
        def __init__(
            self,
            frame_count: int,
            action_dim: int,
            hidden_dim: int,
            token_count: int,
            output_dim: int,
            dropout: float,
        ):
            super().__init__()
            self.conditioning_mode = "tokens"
            self.input_dim = action_dim
            self.frame_count = frame_count
            self.hidden_dim = hidden_dim
            self.token_count = token_count
            self.output_dim = output_dim
            self.input_norm = torch.nn.LayerNorm(action_dim)
            self.input_proj = torch.nn.Linear(action_dim, hidden_dim)
            self.frame_embeddings = torch.nn.Parameter(torch.randn(1, frame_count, hidden_dim) * 0.02)
            self.segment_embeddings = torch.nn.Embedding(2, hidden_dim)
            self.register_buffer("segment_ids", frame_segment_ids(frame_count, CONTEXT_FRAMES), persistent=False)
            self.temporal_mlp = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            import torch.nn.functional as F

            if action_vectors.ndim != 3:
                raise ValueError(f"Expected frame actions [B,T,D], got shape={tuple(action_vectors.shape)}")
            if action_vectors.shape[1] != self.frame_count or action_vectors.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected frame actions [B,{self.frame_count},{self.input_dim}], "
                    f"got shape={tuple(action_vectors.shape)}"
                )
            hidden = self.input_proj(self.input_norm(action_vectors.float()))
            hidden = hidden + self.frame_embeddings + self.segment_embeddings(self.segment_ids)[None, :, :]
            hidden = hidden + self.temporal_mlp(hidden)
            pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.token_count).transpose(1, 2)
            return self.token_proj(pooled)

    return FrameTemporalPoolActionEncoder(
        frame_count=config.action_frame_count,
        action_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        token_count=config.action_token_count,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_frame_transformer_action_encoder(config: TrainConfig, output_dim: int):
    import torch

    class FrameTransformerActionEncoder(torch.nn.Module):
        def __init__(
            self,
            frame_count: int,
            action_dim: int,
            hidden_dim: int,
            token_count: int,
            output_dim: int,
            dropout: float,
            num_layers: int,
            num_heads: int,
        ):
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
            self.conditioning_mode = "tokens"
            self.input_dim = action_dim
            self.frame_count = frame_count
            self.hidden_dim = hidden_dim
            self.token_count = token_count
            self.output_dim = output_dim
            self.input_norm = torch.nn.LayerNorm(action_dim)
            self.input_proj = torch.nn.Linear(action_dim, hidden_dim)
            self.frame_embeddings = torch.nn.Parameter(torch.randn(1, frame_count, hidden_dim) * 0.02)
            self.segment_embeddings = torch.nn.Embedding(2, hidden_dim)
            self.query_embeddings = torch.nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
            self.register_buffer("segment_ids", frame_segment_ids(frame_count, CONTEXT_FRAMES), persistent=False)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.token_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.normal_(self.token_proj[-1].weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.token_proj[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.ndim != 3:
                raise ValueError(f"Expected frame actions [B,T,D], got shape={tuple(action_vectors.shape)}")
            if action_vectors.shape[1] != self.frame_count or action_vectors.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected frame actions [B,{self.frame_count},{self.input_dim}], "
                    f"got shape={tuple(action_vectors.shape)}"
                )
            action_tokens = self.input_proj(self.input_norm(action_vectors.float()))
            action_tokens = action_tokens + self.frame_embeddings + self.segment_embeddings(self.segment_ids)[None, :, :]
            queries = self.query_embeddings.expand(action_vectors.shape[0], -1, -1)
            hidden = self.transformer(torch.cat([queries, action_tokens], dim=1))
            return self.token_proj(hidden[:, : self.token_count])

    return FrameTransformerActionEncoder(
        frame_count=config.action_frame_count,
        action_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        token_count=config.action_token_count,
        output_dim=output_dim,
        dropout=config.action_dropout,
        num_layers=config.action_transformer_layers,
        num_heads=config.action_transformer_heads,
    ).to("cuda")


def make_frame_temporal_bottleneck_action_encoder(config: TrainConfig):
    import torch

    class FrameTemporalBottleneckActionEncoder(torch.nn.Module):
        def __init__(
            self,
            frame_count: int,
            action_dim: int,
            hidden_dim: int,
            dropout: float,
            num_layers: int,
            num_heads: int,
            use_motion_aux_head: bool,
        ):
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
            self.conditioning_mode = "temporal_bottleneck"
            self.input_dim = action_dim
            self.frame_count = frame_count
            self.hidden_dim = hidden_dim
            self.output_dim = hidden_dim
            self.token_count = frame_count
            self.input_norm = torch.nn.LayerNorm(action_dim)
            self.input_proj = torch.nn.Linear(action_dim, hidden_dim)
            self.frame_embeddings = torch.nn.Parameter(torch.randn(1, frame_count, hidden_dim) * 0.02)
            self.segment_embeddings = torch.nn.Embedding(2, hidden_dim)
            self.register_buffer("segment_ids", frame_segment_ids(frame_count, CONTEXT_FRAMES), persistent=False)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.motion_aux_head = (
                torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                if use_motion_aux_head
                else None
            )
            if self.motion_aux_head is not None:
                torch.nn.init.normal_(self.motion_aux_head[-1].weight, mean=0.0, std=1e-3)
                torch.nn.init.zeros_(self.motion_aux_head[-1].bias)
            self.last_motion_aux = None

        def forward(self, action_vectors):
            if action_vectors.ndim != 3:
                raise ValueError(f"Expected frame actions [B,T,D], got shape={tuple(action_vectors.shape)}")
            if action_vectors.shape[1] != self.frame_count or action_vectors.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected frame actions [B,{self.frame_count},{self.input_dim}], "
                    f"got shape={tuple(action_vectors.shape)}"
                )
            hidden = self.input_proj(self.input_norm(action_vectors.float()))
            hidden = hidden + self.frame_embeddings + self.segment_embeddings(self.segment_ids)[None, :, :]
            output = self.output_norm(self.transformer(hidden))
            self.last_motion_aux = (
                self.motion_aux_head(output.float()).squeeze(-1) if self.motion_aux_head is not None else None
            )
            return output

    return FrameTemporalBottleneckActionEncoder(
        frame_count=config.action_frame_count,
        action_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        dropout=config.action_dropout,
        num_layers=config.action_transformer_layers,
        num_heads=config.action_transformer_heads,
        use_motion_aux_head=(
            config.action_encoder_type == TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE
            or float(config.action_motion_aux_loss_weight) > 0.0
        ),
    ).to("cuda")


def make_frame_adaln_action_encoder(config: TrainConfig, inner_dim: int):
    import torch

    output_dim = 6 * inner_dim

    class FrameAdaLNActionEncoder(torch.nn.Module):
        def __init__(self, frame_count: int, action_dim: int, hidden_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.conditioning_mode = "adaln"
            self.input_dim = action_dim
            self.frame_count = frame_count
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            flat_dim = frame_count * action_dim
            self.input_norm = torch.nn.LayerNorm(flat_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(flat_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, output_dim),
            )
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

        def forward(self, action_vectors):
            if action_vectors.ndim != 3:
                raise ValueError(f"Expected frame actions [B,T,D], got shape={tuple(action_vectors.shape)}")
            if action_vectors.shape[1] != self.frame_count or action_vectors.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected frame actions [B,{self.frame_count},{self.input_dim}], "
                    f"got shape={tuple(action_vectors.shape)}"
                )
            flat = action_vectors.float().reshape(action_vectors.shape[0], -1)
            return self.net(self.input_norm(flat)).unsqueeze(1)

    return FrameAdaLNActionEncoder(
        frame_count=config.action_frame_count,
        action_dim=config.action_dim,
        hidden_dim=config.action_hidden_dim,
        output_dim=output_dim,
        dropout=config.action_dropout,
    ).to("cuda")


def make_action_token_encoder(config: TrainConfig, output_dim: int):
    if config.action_encoder_type == "global_mlp":
        return make_global_action_token_encoder(config, output_dim=output_dim)
    if config.action_encoder_type == "temporal_per_point":
        return make_temporal_per_point_action_encoder(config, output_dim=output_dim)
    if config.action_encoder_type == "tiny_transformer":
        return make_tiny_transformer_action_encoder(config, output_dim=output_dim)
    if config.action_encoder_type == "frame_global_mlp":
        return make_frame_global_mlp_action_encoder(config, output_dim=output_dim)
    if config.action_encoder_type == "frame_temporal_pool":
        return make_frame_temporal_pool_action_encoder(config, output_dim=output_dim)
    if config.action_encoder_type == "frame_transformer":
        return make_frame_transformer_action_encoder(config, output_dim=output_dim)
    raise ValueError(f"Unknown action_encoder_type: {config.action_encoder_type}")


def make_action_encoder(config: TrainConfig, *, prompt_hidden_dim: int, transformer_inner_dim: int):
    if config.action_encoder_type == "adaln":
        return make_adaln_action_encoder(config, inner_dim=transformer_inner_dim)
    if config.action_encoder_type == "frame_adaln":
        return make_frame_adaln_action_encoder(config, inner_dim=transformer_inner_dim)
    if config.action_encoder_type == MIDBLOCK_GATED_XATTN_MODE:
        encoder = make_frame_transformer_action_encoder(config, output_dim=transformer_inner_dim)
        encoder.conditioning_mode = "midblock_gated_xattn"
        return encoder
    if config.action_encoder_type in TEMPORAL_BOTTLENECK_ACTION_MODES:
        return make_frame_temporal_bottleneck_action_encoder(config)
    return make_action_token_encoder(config, output_dim=prompt_hidden_dim)


def cycle_indices(n: int, batch_size: int, seed: int):
    rng = random.Random(seed)
    order = list(range(n))
    while True:
        rng.shuffle(order)
        for i in range(0, n, batch_size):
            batch = order[i : i + batch_size]
            if len(batch) == batch_size:
                yield batch


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parameter_grad_norm(parameters: list[Any]) -> float:
    import torch

    grads = [param.grad.detach().float().norm(2) for param in parameters if getattr(param, "grad", None) is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).norm(2).detach().cpu())


def parameter_count(parameters: list[Any]) -> int:
    return int(sum(param.numel() for param in parameters))


def parse_step_list(value: str, max_steps: int) -> list[int]:
    if not value.strip():
        return [step for step in CHECKPOINT_STEPS if step <= max_steps]
    steps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    invalid = [step for step in steps if step < 0 or step > max_steps]
    if invalid:
        raise ValueError(f"Checkpoint steps must be in [0, {max_steps}], got {invalid}")
    return steps


def setup_pipeline_and_lora(config: TrainConfig):
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from peft import LoraConfig, get_peft_model
    from ltx_video.inference import create_ltx_video_pipeline

    ckpt = ensure_checkpoint_symlink(config.base_checkpoint)
    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(ckpt),
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        sampler="from_checkpoint",
        device="cuda",
        enhance_prompt=False,
    )
    pipeline.vae.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline.transformer.train()
    for param in pipeline.transformer.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_out",
            "patchify_proj",
            "net.2",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    # LTX pipeline code reads `transformer.config` directly during sampling.
    # PEFT wraps the module, so mirror the underlying config for compatibility.
    pipeline.transformer.config = pipeline.transformer.base_model.model.config
    if config.baseline_lora_run_name and config.baseline_lora_step:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_path = (
            BASELINE_CKPT_ROOT
            / config.baseline_lora_run_name
            / config.baseline_lora_step
            / "lora_adapter"
            / "adapter_model.safetensors"
        )
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing baseline LoRA adapter: {adapter_path}")
        state_dict = load_file(str(adapter_path), device="cpu")
        if config.expand_baseline_lora_to_rank:
            load_result = load_lora_state_dict_expanding_to_current_rank(pipeline.transformer, state_dict)
        else:
            load_result = set_peft_model_state_dict(pipeline.transformer, state_dict, adapter_name="default")
            load_result = {
                "missing_keys": len(getattr(load_result, "missing_keys", []) or []),
                "unexpected_keys": len(getattr(load_result, "unexpected_keys", []) or []),
                "expanded_keys": 0,
                "loaded_keys": len(state_dict),
            }
        print(
            json.dumps(
                {
                    "loaded_baseline_lora": str(adapter_path),
                    "freeze_transformer_lora": config.freeze_transformer_lora,
                    "expand_baseline_lora_to_rank": config.expand_baseline_lora_to_rank,
                    **load_result,
                },
                sort_keys=True,
            )
        )
        if config.freeze_transformer_lora:
            for param in pipeline.transformer.parameters():
                param.requires_grad_(False)
    pipeline.transformer.train()
    pipeline.transformer.to("cuda", dtype=torch.bfloat16)
    return pipeline


def candidate_lora_state_keys(key: str) -> list[str]:
    candidates = [key]
    for pattern in (".lora_A.weight", ".lora_B.weight"):
        if pattern in key:
            candidates.append(key.replace(pattern, pattern.replace(".weight", ".default.weight")))
    for pattern in (".lora_A.default.weight", ".lora_B.default.weight"):
        if pattern in key:
            candidates.append(key.replace(pattern, pattern.replace(".default.weight", ".weight")))
    return list(dict.fromkeys(candidates))


def expand_tensor_top_left(source, target_shape: tuple[int, ...]):
    import torch

    if tuple(source.shape) == tuple(target_shape):
        return source
    if source.ndim != len(target_shape):
        raise ValueError(f"Cannot expand tensor rank {tuple(source.shape)} to {target_shape}")
    if any(src > dst for src, dst in zip(source.shape, target_shape)):
        raise ValueError(f"Cannot expand tensor shape {tuple(source.shape)} to smaller shape {target_shape}")
    target = torch.zeros(target_shape, dtype=source.dtype)
    slices = tuple(slice(0, int(size)) for size in source.shape)
    target[slices] = source
    return target


def load_lora_state_dict_expanding_to_current_rank(transformer, state_dict: dict[str, Any]) -> dict[str, int]:
    current_state = transformer.state_dict()
    adapted: dict[str, Any] = {}
    expanded_keys = 0
    skipped_keys = 0
    for key, tensor in state_dict.items():
        model_key = next((candidate for candidate in candidate_lora_state_keys(key) if candidate in current_state), "")
        if not model_key:
            skipped_keys += 1
            continue
        target_shape = tuple(current_state[model_key].shape)
        adapted_tensor = expand_tensor_top_left(tensor, target_shape)
        if tuple(adapted_tensor.shape) != tuple(tensor.shape):
            expanded_keys += 1
        adapted[model_key] = adapted_tensor.to(dtype=current_state[model_key].dtype)
    if not adapted:
        raise RuntimeError("Failed to map any baseline LoRA tensors into the current-rank adapter.")
    load_result = transformer.load_state_dict(adapted, strict=False)
    return {
        "loaded_keys": len(adapted),
        "expanded_keys": expanded_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": len(getattr(load_result, "missing_keys", []) or []),
        "unexpected_keys": len(getattr(load_result, "unexpected_keys", []) or []),
    }


def load_lora_adapter_into_transformer(transformer, adapter_dir: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_path}")
    state_dict = load_file(str(adapter_path), device="cpu")
    load_result = set_peft_model_state_dict(transformer, state_dict, adapter_name="default")
    print(
        json.dumps(
            {
                "loaded_lora_adapter": str(adapter_path),
                "missing_keys": len(getattr(load_result, "missing_keys", []) or []),
                "unexpected_keys": len(getattr(load_result, "unexpected_keys", []) or []),
            },
            sort_keys=True,
        )
    )


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import torch

    if "python_random" in state:
        random.setstate(state["python_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def load_resume_checkpoint(
    pipeline,
    action_encoder,
    action_injector,
    optimizer,
    config: TrainConfig,
) -> tuple[int, list[dict[str, float]], bool]:
    import torch

    if not config.resume_from_checkpoint:
        return 0, [], False

    resume_run = config.resume_from_run_name or config.run_name
    ckpt_dir = CKPT_ROOT / resume_run / config.resume_from_checkpoint
    trainer_state_path = ckpt_dir / "trainer_state.pt"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {trainer_state_path}")

    load_lora_adapter_into_transformer(pipeline.transformer, ckpt_dir / "lora_adapter")
    if action_encoder is not None:
        action_path = ckpt_dir / "action_encoder.pt"
        if not action_path.exists():
            raise FileNotFoundError(f"Missing action encoder checkpoint: {action_path}")
        action_payload = torch.load(action_path, map_location="cpu", weights_only=False)
        action_encoder.load_state_dict(action_payload["state_dict"])
    if action_injector is not None:
        injector_path = ckpt_dir / "action_injector.pt"
        if not injector_path.exists():
            raise FileNotFoundError(f"Missing action injector checkpoint: {injector_path}")
        injector_payload = torch.load(injector_path, map_location="cpu", weights_only=False)
        action_injector.load_state_dict(injector_payload["state_dict"])

    trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(trainer_state["optimizer"])
    restore_rng_state(trainer_state.get("rng_state"))
    start_step = int(trainer_state["step"])
    if start_step >= config.max_steps:
        raise ValueError(
            f"Resume checkpoint step {start_step} is already >= requested max_steps {config.max_steps}."
        )
    loss_history = list(trainer_state.get("loss_history", []))
    print(
        json.dumps(
            {
                "resumed_from": str(ckpt_dir),
                "start_step": start_step,
                "loss_history_rows": len(loss_history),
                "restored_rng_state": bool(trainer_state.get("rng_state")),
            },
            sort_keys=True,
        )
    )
    return start_step, loss_history, True


def external_stop_requested(config: TrainConfig, step: int) -> bool:
    if not config.external_stop_relpath:
        return False
    check_every = max(int(config.external_stop_check_steps), 1)
    if step % check_every != 0:
        return False
    checkpoint_volume.reload()
    return (CKPT_ROOT / config.external_stop_relpath).exists()


def encode_fixed_prompt(pipeline, prompt: str):
    import torch

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=torch.device("cuda"),
            text_encoder_max_tokens=256,
        )
    return prompt_embeds, prompt_attention_mask


def maybe_disable_text_conditioning(prompt_embeds, prompt_attention_mask, config: TrainConfig):
    if not config.disable_text_conditioning:
        return prompt_embeds, prompt_attention_mask
    return prompt_embeds * 0.0, prompt_attention_mask * 0


def sample_training_timesteps(pipeline, clean_latents, config: TrainConfig):
    import torch

    bsz = clean_latents.shape[0]
    if config.timestep_sampling == "uniform":
        timesteps = torch.rand(bsz, device=clean_latents.device, dtype=torch.float32)
    elif config.timestep_sampling == "shifted_lognormal":
        logits = (
            torch.randn(bsz, device=clean_latents.device, dtype=torch.float32)
            * float(config.timestep_lognormal_std)
            + float(config.timestep_lognormal_mean)
        )
        timesteps = torch.sigmoid(logits)
        if hasattr(pipeline.scheduler, "shift_timesteps"):
            timesteps = pipeline.scheduler.shift_timesteps(clean_latents.shape, timesteps)
    else:
        raise ValueError(f"Unknown timestep_sampling={config.timestep_sampling!r}")
    return timesteps.clamp(1e-3, 1.0)


def high_frequency_token_loss(student_tokens, teacher_tokens, latent_time_ids, future_token_mask):
    import torch
    import torch.nn.functional as F

    time_ids = latent_time_ids.to(device=student_tokens.device, dtype=torch.long)
    future_mask = future_token_mask.to(device=student_tokens.device, dtype=torch.bool)
    if time_ids.ndim > 1:
        time_ids = time_ids[0]
    if future_mask.ndim > 1:
        future_mask = future_mask[0]
    losses = []
    for time_id in torch.unique(time_ids[future_mask], sorted=True):
        token_mask = future_mask & (time_ids == time_id)
        if int(token_mask.sum().detach().cpu()) <= 1:
            continue
        student = student_tokens[:, token_mask, :].float()
        teacher = teacher_tokens[:, token_mask, :].float()
        student_hf = student - student.mean(dim=1, keepdim=True)
        teacher_hf = teacher - teacher.mean(dim=1, keepdim=True)
        losses.append(F.mse_loss(student_hf, teacher_hf.detach()))
    if not losses:
        return torch.zeros((), device=student_tokens.device, dtype=torch.float32)
    return torch.stack(losses).mean()


def low_frequency_token_losses(pred_tokens, target_tokens, latent_time_ids, future_token_mask):
    import torch
    import torch.nn.functional as F

    time_ids = latent_time_ids.to(device=pred_tokens.device, dtype=torch.long)
    future_mask = future_token_mask.to(device=pred_tokens.device, dtype=torch.bool)
    if time_ids.ndim > 1:
        time_ids = time_ids[0]
    if future_mask.ndim > 1:
        future_mask = future_mask[0]

    pred_means = []
    target_means = []
    for time_id in torch.unique(time_ids[future_mask], sorted=True):
        token_mask = future_mask & (time_ids == time_id)
        if int(token_mask.sum().detach().cpu()) <= 0:
            continue
        pred_means.append(pred_tokens[:, token_mask, :].float().mean(dim=1))
        target_means.append(target_tokens[:, token_mask, :].float().mean(dim=1))

    if not pred_means:
        zero = torch.zeros((), device=pred_tokens.device, dtype=torch.float32)
        return zero, zero

    pred_low = torch.stack(pred_means, dim=1)
    target_low = torch.stack(target_means, dim=1)
    target_loss = F.mse_loss(pred_low, target_low)
    if pred_low.shape[1] <= 1:
        delta_loss = torch.zeros((), device=pred_tokens.device, dtype=torch.float32)
    else:
        delta_loss = F.mse_loss(pred_low[:, 1:] - pred_low[:, :-1], target_low[:, 1:] - target_low[:, :-1])
    return target_loss, delta_loss


def future_low_frequency_means(tokens, latent_time_ids, future_token_mask):
    import torch

    time_ids = latent_time_ids.to(device=tokens.device, dtype=torch.long)
    future_mask = future_token_mask.to(device=tokens.device, dtype=torch.bool)
    if time_ids.ndim > 1:
        time_ids = time_ids[0]
    if future_mask.ndim > 1:
        future_mask = future_mask[0]

    means = []
    for time_id in torch.unique(time_ids[future_mask], sorted=True):
        token_mask = future_mask & (time_ids == time_id)
        if int(token_mask.sum().detach().cpu()) <= 0:
            continue
        means.append(tokens[:, token_mask, :].float().mean(dim=1))
    if not means:
        return None
    return torch.stack(means, dim=1)


def normalize_temporal_profile(profile):
    return (profile - profile.mean(dim=1, keepdim=True)) / profile.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)


def action_motion_auxiliary_loss(action_encoder, target_tokens, latent_time_ids, future_token_mask):
    import torch
    import torch.nn.functional as F

    aux = getattr(action_encoder, "last_motion_aux", None)
    if aux is None:
        return torch.zeros((), device=target_tokens.device, dtype=torch.float32)
    if aux.ndim != 2:
        raise ValueError(f"Expected action motion aux logits [B,T], got shape={tuple(aux.shape)}")
    target_low = future_low_frequency_means(target_tokens, latent_time_ids, future_token_mask)
    if target_low is None or target_low.shape[1] <= 1:
        return torch.zeros((), device=target_tokens.device, dtype=torch.float32)
    aux_by_time = F.adaptive_avg_pool1d(aux.float().unsqueeze(1), target_low.shape[1]).squeeze(1)
    target_motion = (target_low[:, 1:] - target_low[:, :-1]).abs().mean(dim=-1)
    aux_motion = (aux_by_time[:, 1:] - aux_by_time[:, :-1]).abs()
    return F.mse_loss(normalize_temporal_profile(aux_motion), normalize_temporal_profile(target_motion).detach())


def training_step(
    pipeline,
    clean_latents,
    prompt_embeds,
    prompt_attention_mask,
    action_encoder=None,
    action_injector=None,
    action_vectors=None,
    config: TrainConfig | None = None,
):
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn.functional as F
    from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords

    clean_latents = clean_latents.to("cuda", dtype=torch.bfloat16)
    bsz = clean_latents.shape[0]
    noise = torch.randn_like(clean_latents)
    if config is None:
        timesteps = torch.rand(bsz, device=clean_latents.device, dtype=torch.float32).clamp(1e-3, 1.0)
    else:
        timesteps = sample_training_timesteps(pipeline, clean_latents, config)
    noisy_latents = pipeline.scheduler.add_noise(clean_latents.float(), noise.float(), timesteps).to(torch.bfloat16)

    # Hard condition on the latent prefix corresponding to 49 observed frames.
    noisy_latents[:, :, :CONTEXT_LATENT_FRAMES] = clean_latents[:, :, :CONTEXT_LATENT_FRAMES]

    noisy_tokens, latent_coords = pipeline.patchifier.patchify(noisy_latents)
    target_tokens, _ = pipeline.patchifier.patchify((noise - clean_latents).to(torch.bfloat16))
    pixel_coords = latent_to_pixel_coords(
        latent_coords,
        pipeline.vae,
        causal_fix=pipeline.transformer.base_model.model.config.causal_temporal_positioning
        if hasattr(pipeline.transformer, "base_model")
        else pipeline.transformer.config.causal_temporal_positioning,
    )
    fractional_coords = pixel_coords.to(torch.float32)
    fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / FPS)

    future_token_mask = latent_coords[:, 0] >= CONTEXT_LATENT_FRAMES
    token_timesteps = torch.where(
        future_token_mask,
        timesteps[:, None].expand_as(future_token_mask).to(torch.float32),
        torch.zeros_like(future_token_mask, dtype=torch.float32),
    )

    prompt_batch = prompt_embeds.expand(bsz, -1, -1)
    mask_batch = prompt_attention_mask.expand(bsz, -1)
    transformer_dtype = next(pipeline.transformer.parameters()).dtype
    encoder_hidden_states = prompt_batch.to(transformer_dtype)
    encoder_attention_mask = mask_batch
    action_conditioning_mode = getattr(action_encoder, "conditioning_mode", "tokens") if action_encoder is not None else ""
    if action_encoder is not None:
        if action_vectors is None:
            raise RuntimeError("Action conditioning is enabled but the batch has no action vectors.")
        if action_conditioning_mode == "adaln":
            action_delta = action_encoder(action_vectors.to(clean_latents.device)).to(transformer_dtype)
            set_action_adaln_delta(pipeline.transformer, action_delta)
        elif action_conditioning_mode == "midblock_gated_xattn":
            if action_injector is None:
                raise RuntimeError("Mid-block gated action conditioning requires an action injector.")
            action_tokens = action_encoder(action_vectors.to(clean_latents.device))
            set_midblock_action_tokens(pipeline.transformer, action_tokens)
        elif action_conditioning_mode == "temporal_bottleneck":
            if action_injector is None:
                raise RuntimeError("Temporal bottleneck action conditioning requires an action injector.")
            action_injector.reset_regularization_state()
            action_controls = action_encoder(action_vectors.to(clean_latents.device))
            set_temporal_bottleneck_action_controls(pipeline.transformer, action_controls)
        else:
            action_tokens = action_encoder(action_vectors.to(clean_latents.device)).to(transformer_dtype)
            encoder_hidden_states = torch.cat([encoder_hidden_states, action_tokens], dim=1)
            action_mask = torch.ones(
                bsz,
                action_tokens.shape[1],
                device=mask_batch.device,
                dtype=mask_batch.dtype,
            )
            encoder_attention_mask = torch.cat([mask_batch, action_mask], dim=1)
    if (
        config is not None
        and config.disable_text_conditioning
        and action_conditioning_mode in {"adaln", "midblock_gated_xattn", "temporal_bottleneck"}
    ):
        encoder_attention_mask = mask_batch.clone()
        encoder_attention_mask[:, :1] = 1

    try:
        pred = pipeline.transformer(
            noisy_tokens.to(transformer_dtype),
            indices_grid=fractional_coords,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=token_timesteps,
            skip_layer_mask=None,
            skip_layer_strategy=None,
            return_dict=False,
        )[0]
    finally:
        if action_encoder is not None:
            if action_conditioning_mode == "adaln":
                clear_action_adaln_delta(pipeline.transformer)
            elif action_conditioning_mode == "midblock_gated_xattn":
                clear_midblock_action_tokens(pipeline.transformer)
            elif action_conditioning_mode == "temporal_bottleneck":
                clear_temporal_bottleneck_action_controls(pipeline.transformer)
    out_channels = (
        pipeline.transformer.base_model.model.config.out_channels
        if hasattr(pipeline.transformer, "base_model")
        else pipeline.transformer.config.out_channels
    )
    in_channels = (
        pipeline.transformer.base_model.model.config.in_channels
        if hasattr(pipeline.transformer, "base_model")
        else pipeline.transformer.config.in_channels
    )
    if out_channels // 2 == in_channels:
        pred = pred.chunk(2, dim=-1)[0]

    mask = future_token_mask.unsqueeze(-1).expand_as(pred)
    loss_diffusion = F.mse_loss(pred[mask].float(), target_tokens[mask].float())
    loss_hf_teacher = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss_lowfreq_target = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss_lowfreq_delta = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss_action_motion_aux = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss_residual_norm = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss_gate = torch.zeros((), device=pred.device, dtype=torch.float32)
    if (
        action_conditioning_mode == "temporal_bottleneck"
        and action_injector is not None
        and config is not None
    ):
        loss_residual_norm = action_injector.residual_norm_loss(pred.device)
        loss_gate = action_injector.gate_loss(pred.device)
        if float(config.hf_teacher_loss_weight) > 0:
            old_gate_scale = float(getattr(action_injector, "gate_scale", 1.0))
            with torch.no_grad():
                action_injector.gate_scale = 0.0
                try:
                    teacher_pred = pipeline.transformer(
                        noisy_tokens.to(transformer_dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=token_timesteps,
                        skip_layer_mask=None,
                        skip_layer_strategy=None,
                        return_dict=False,
                    )[0]
                finally:
                    action_injector.gate_scale = old_gate_scale
                if out_channels // 2 == in_channels:
                    teacher_pred = teacher_pred.chunk(2, dim=-1)[0]
            loss_hf_teacher = high_frequency_token_loss(
                pred,
                teacher_pred,
                latent_coords[:, 0],
                future_token_mask,
            )
        if (
            float(config.lowfreq_target_loss_weight) > 0
            or float(config.lowfreq_delta_loss_weight) > 0
        ):
            loss_lowfreq_target, loss_lowfreq_delta = low_frequency_token_losses(
                pred,
                target_tokens,
                latent_coords[:, 0],
                future_token_mask,
            )
        if float(config.action_motion_aux_loss_weight) > 0:
            loss_action_motion_aux = action_motion_auxiliary_loss(
                action_encoder,
                target_tokens,
                latent_coords[:, 0],
                future_token_mask,
            )
    loss = (
        float(config.diffusion_loss_weight if config is not None else 1.0) * loss_diffusion
        + float(config.lowfreq_target_loss_weight if config is not None else 0.0) * loss_lowfreq_target
        + float(config.lowfreq_delta_loss_weight if config is not None else 0.0) * loss_lowfreq_delta
        + float(config.hf_teacher_loss_weight if config is not None else 0.0) * loss_hf_teacher
        + float(config.action_motion_aux_loss_weight if config is not None else 0.0) * loss_action_motion_aux
        + float(config.action_residual_loss_weight if config is not None else 0.0) * loss_residual_norm
        + float(config.action_gate_loss_weight if config is not None else 0.0) * loss_gate
    )
    metrics = {
        "loss": loss,
        "loss_diffusion": loss_diffusion.detach(),
        "loss_lowfreq_target": loss_lowfreq_target.detach(),
        "loss_lowfreq_delta": loss_lowfreq_delta.detach(),
        "loss_hf_teacher": loss_hf_teacher.detach(),
        "loss_action_motion_aux": loss_action_motion_aux.detach(),
        "loss_residual_norm": loss_residual_norm.detach(),
        "loss_gate": loss_gate.detach(),
        "weighted_loss_diffusion": (
            float(config.diffusion_loss_weight if config is not None else 1.0) * loss_diffusion
        ).detach(),
        "weighted_loss_lowfreq_target": (
            float(config.lowfreq_target_loss_weight if config is not None else 0.0) * loss_lowfreq_target
        ).detach(),
        "weighted_loss_lowfreq_delta": (
            float(config.lowfreq_delta_loss_weight if config is not None else 0.0) * loss_lowfreq_delta
        ).detach(),
        "weighted_loss_hf_teacher": (
            float(config.hf_teacher_loss_weight if config is not None else 0.0) * loss_hf_teacher
        ).detach(),
        "weighted_loss_action_motion_aux": (
            float(config.action_motion_aux_loss_weight if config is not None else 0.0) * loss_action_motion_aux
        ).detach(),
    }
    if action_injector is not None and hasattr(action_injector, "gate_stats"):
        metrics.update(action_injector.gate_stats())
    return metrics


def write_video(path: Path, video_tensor, fps: int = FPS) -> None:
    import imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in video_np:
            writer.append_data(frame)


def sample_validation(pipeline, val_rows: list[dict[str, str]], config: TrainConfig, step: int, out_dir: Path) -> None:
    if config.action_conditioning:
        save_json(
            out_dir / "action_conditioned_sampling_not_implemented.json",
            {
                "step": step,
                "reason": (
                    "The high-level LTX pipeline call does not expose extra encoder_hidden_states. "
                    "Action-conditioned training is implemented at the transformer call; "
                    "action-conditioned inference needs a custom sampler/wrapper."
                ),
                "action_token_count": config.action_token_count,
                "action_dim": config.action_dim,
            },
        )
        return

    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from ltx_video.inference import load_media_file
    from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem

    pipeline.transformer.eval()
    sample_rows = val_rows[: config.num_val_samples]
    for sample_idx, row in enumerate(sample_rows):
        mp4_path = DATA_ROOT / row["mp4_relpath"]
        media = load_media_file(
            media_path=str(mp4_path),
            height=HEIGHT,
            width=WIDTH,
            max_frames=CONTEXT_FRAMES,
            padding=(0, 0, 0, 0),
            just_crop=True,
        )
        conditioning_items = [ConditioningItem(media, 0, 1.0)]
        generator = torch.Generator(device="cuda").manual_seed(config.seed + sample_idx)
        with torch.no_grad():
            result = pipeline(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=TOTAL_FRAMES,
                frame_rate=FPS,
                timesteps=[1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250, 0.4219],
                guidance_scale=1,
                stg_scale=0,
                rescaling_scale=1,
                output_type="pt",
                conditioning_items=conditioning_items,
                is_video=True,
                vae_per_channel_normalize=True,
                image_cond_noise_scale=IMAGE_COND_NOISE_SCALE,
                mixed_precision=False,
                offload_to_cpu=False,
                enhance_prompt=False,
                generator=generator,
            ).images[0]
        write_video(out_dir / f"step_{step:06d}_{sample_idx:02d}_{row['window_id']}.mp4", result, fps=FPS)
    pipeline.transformer.train()


def save_checkpoint(
    pipeline,
    action_encoder,
    action_injector,
    optimizer,
    config: TrainConfig,
    step: int,
    loss_history: list[dict[str, float]],
    label: str,
) -> Path:
    ckpt_dir = CKPT_ROOT / config.run_name / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pipeline.transformer.save_pretrained(str(ckpt_dir / "lora_adapter"))
    import torch

    if action_encoder is not None:
        torch.save(
            {
                "state_dict": action_encoder.state_dict(),
                "action_encoder_type": config.action_encoder_type,
                "conditioning_mode": getattr(action_encoder, "conditioning_mode", "tokens"),
                "action_dim": config.action_dim,
                "action_frame_count": config.action_frame_count,
                "action_token_count": config.action_token_count,
                "action_hidden_dim": config.action_hidden_dim,
                "action_dropout": config.action_dropout,
                "action_transformer_layers": config.action_transformer_layers,
                "action_transformer_heads": config.action_transformer_heads,
                "frame_action_stats_relpath": config.frame_action_stats_relpath,
                "frame_action_feature_key": config.frame_action_feature_key,
                "action_motion_aux_loss_weight": config.action_motion_aux_loss_weight,
                "baseline_lora_run_name": config.baseline_lora_run_name,
                "baseline_lora_step": config.baseline_lora_step,
                "freeze_transformer_lora": config.freeze_transformer_lora,
                "disable_text_conditioning": config.disable_text_conditioning,
                "prompt_hidden_dim": getattr(action_encoder, "output_dim", None),
                "transformer_inner_dim": get_transformer_inner_dim(pipeline.transformer),
                "action_adaln_dim": getattr(action_encoder, "output_dim", None)
                if getattr(action_encoder, "conditioning_mode", "tokens") == "adaln"
                else None,
            },
            ckpt_dir / "action_encoder.pt",
        )

    if action_injector is not None:
        torch.save(
            {
                "state_dict": action_injector.state_dict(),
                "conditioning_mode": getattr(action_injector, "conditioning_mode", "midblock_gated_xattn"),
                "action_encoder_type": config.action_encoder_type,
                "action_injector_learning_rate": config.action_injector_learning_rate,
                "action_gate_learning_rate": config.action_gate_learning_rate,
                "action_injector_heads": config.action_injector_heads,
                "action_midblock_start": config.action_midblock_start,
                "action_midblock_end": config.action_midblock_end,
                "action_gate_scale": config.action_gate_scale,
                "action_gate_bound": config.action_gate_bound,
                "diffusion_loss_weight": config.diffusion_loss_weight,
                "lowfreq_target_loss_weight": config.lowfreq_target_loss_weight,
                "lowfreq_delta_loss_weight": config.lowfreq_delta_loss_weight,
                "hf_teacher_loss_weight": config.hf_teacher_loss_weight,
                "action_motion_aux_loss_weight": config.action_motion_aux_loss_weight,
                "action_residual_loss_weight": config.action_residual_loss_weight,
                "action_gate_loss_weight": config.action_gate_loss_weight,
                "metadata": action_injector.metadata(),
            },
            ckpt_dir / "action_injector.pt",
        )

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": asdict(config),
            "rng_state": capture_rng_state(),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        ckpt_dir / "trainer_state.pt",
    )
    save_json(ckpt_dir / "training_config.json", asdict(config))
    save_json(ckpt_dir / "loss_history.json", {"loss_history": loss_history})
    checkpoint_volume.commit()
    return ckpt_dir


@app.function(
    image=image,
    gpu=DEFAULT_TRAIN_GPU,
    cpu=8,
    memory=65536,
    timeout=11 * 60 * 60,
    volumes={
        str(DATA_ROOT): data_volume,
        str(MODELS_ROOT): models_volume,
        str(CKPT_ROOT): checkpoint_volume,
        str(BASELINE_CKPT_ROOT): baseline_checkpoint_volume,
    },
)
def train(
    config_payload: dict[str, Any],
    train_limit: int = 0,
    val_limit: int = 32,
) -> dict[str, Any]:
    import torch

    data_volume.reload()
    checkpoint_volume.reload()
    baseline_checkpoint_volume.reload()
    config = TrainConfig(**config_payload)
    seed_everything(config.seed)

    train_rows = load_manifest(DATA_ROOT / config.train_manifest, limit=train_limit)
    val_rows = load_manifest(DATA_ROOT / config.val_manifest, limit=val_limit)
    skipped_train_missing_latents = 0
    skipped_val_missing_latents = 0
    if config.skip_missing_latents:
        train_rows, skipped_train_missing_latents = filter_rows_with_existing_latents(train_rows, config.latent_prefix)
        val_rows, skipped_val_missing_latents = filter_rows_with_existing_latents(val_rows, config.latent_prefix)
    if not train_rows:
        raise RuntimeError("No training rows loaded.")
    if not val_rows:
        raise RuntimeError("No validation rows loaded.")

    pipeline = setup_pipeline_and_lora(config)
    if config.action_conditioning and config.action_encoder_type in {"adaln", "frame_adaln"}:
        install_action_adaln_hook(pipeline.transformer)
    prompt_embeds, prompt_attention_mask = encode_fixed_prompt(pipeline, config.prompt)
    prompt_embeds, prompt_attention_mask = maybe_disable_text_conditioning(
        prompt_embeds,
        prompt_attention_mask,
        config,
    )
    action_encoder = (
        make_action_encoder(
            config,
            prompt_hidden_dim=prompt_embeds.shape[-1],
            transformer_inner_dim=get_transformer_inner_dim(pipeline.transformer),
        )
        if config.action_conditioning
        else None
    )
    action_injector = None
    if config.action_conditioning and config.action_encoder_type == MIDBLOCK_GATED_XATTN_MODE:
        action_injector = make_midblock_action_injector(config, pipeline.transformer)
        install_midblock_action_injector(pipeline.transformer, action_injector)
    if config.action_conditioning and config.action_encoder_type in TEMPORAL_BOTTLENECK_ACTION_MODES:
        action_injector = make_temporal_bottleneck_action_injector(config, pipeline.transformer)
        install_temporal_bottleneck_action_injector(pipeline.transformer, action_injector)

    lora_trainable = [
        param
        for name, param in pipeline.transformer.named_parameters()
        if param.requires_grad
        and "_waymo_midblock_action_injector" not in name
        and "_waymo_temporal_bottleneck_action_injector" not in name
    ]
    trainable = list(lora_trainable)
    action_trainable = []
    injector_xattn_trainable = []
    injector_gate_trainable = []
    if action_encoder is not None or action_injector is not None:
        if action_encoder is not None:
            action_encoder.train()
        if action_injector is not None:
            action_injector.train()
        action_trainable = list(action_encoder.parameters()) if action_encoder is not None else []
        trainable.extend(action_trainable)
        injector_xattn_trainable = action_injector.xattn_parameters() if action_injector is not None else []
        injector_gate_trainable = action_injector.gate_parameters() if action_injector is not None else []
        trainable.extend(injector_xattn_trainable)
        trainable.extend(injector_gate_trainable)
        param_groups = []
        if lora_trainable:
            param_groups.append({"params": lora_trainable, "lr": config.learning_rate})
        if action_trainable:
            param_groups.append({"params": action_trainable, "lr": config.action_learning_rate})
        if injector_xattn_trainable:
            param_groups.append({"params": injector_xattn_trainable, "lr": config.action_injector_learning_rate})
        if injector_gate_trainable:
            param_groups.append({"params": injector_gate_trainable, "lr": config.action_gate_learning_rate})
        if not param_groups:
            raise RuntimeError("No trainable parameters were found.")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    trainable_parameter_counts = {
        "lora": parameter_count(lora_trainable),
        "action_encoder": parameter_count(action_trainable),
        "action_injector": parameter_count(injector_xattn_trainable),
        "action_gate": parameter_count(injector_gate_trainable),
        "total": parameter_count(trainable),
    }
    print(
        json.dumps(
            {
                "trainable_parameter_audit": trainable_parameter_counts,
                "freeze_transformer_lora": config.freeze_transformer_lora,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "expand_baseline_lora_to_rank": config.expand_baseline_lora_to_rank,
            },
            sort_keys=True,
        )
    )
    start_step, loss_history, resumed = load_resume_checkpoint(pipeline, action_encoder, action_injector, optimizer, config)
    dataset = LatentRowsDataset(
        train_rows,
        config.latent_prefix,
        config.frame_action_stats_relpath,
        config.frame_action_feature_key,
    )
    batch_iter = cycle_indices(len(dataset), config.batch_size, config.seed)
    for _ in range(start_step):
        next(batch_iter)

    run_dir = CKPT_ROOT / config.run_name
    save_json(run_dir / "run_config.json", asdict(config))

    if not resumed:
        sample_validation(pipeline, val_rows, config, 0, run_dir / "validation_samples" / "step_000000_base_reference")
        save_checkpoint(
            pipeline,
            action_encoder,
            action_injector,
            optimizer,
            config,
            0,
            loss_history,
            "step_000000_base_reference",
        )

    start = time.monotonic()
    last_log = start
    completed_step = start_step
    stop_reason = "max_steps"
    for step in range(start_step + 1, config.max_steps + 1):
        elapsed_hours = (time.monotonic() - start) / 3600.0
        if elapsed_hours >= config.max_train_hours:
            completed_step = step - 1
            stop_reason = "time_limit"
            break

        batch = make_batch(dataset, next(batch_iter))
        optimizer.zero_grad(set_to_none=True)
        step_metrics = training_step(
            pipeline,
            batch["latents"],
            prompt_embeds,
            prompt_attention_mask,
            action_encoder=action_encoder,
            action_injector=action_injector,
            action_vectors=batch.get("action_vectors"),
            config=config,
        )
        loss = step_metrics["loss"] if isinstance(step_metrics, dict) else step_metrics
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norms = {
            "grad_norm_total": parameter_grad_norm(trainable),
            "grad_norm_lora": parameter_grad_norm(lora_trainable),
            "grad_norm_action_encoder": parameter_grad_norm(action_trainable),
            "grad_norm_action_injector": parameter_grad_norm(injector_xattn_trainable),
            "grad_norm_action_gate": parameter_grad_norm(injector_gate_trainable),
        }
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        completed_step = step
        loss_value = float(loss.detach().cpu())
        loss_row = {
            "step": step,
            "loss": loss_value,
            "elapsed_hours": elapsed_hours,
            **grad_norms,
            "grad_norm_total_preclip": float(clipped_grad_norm.detach().cpu())
            if hasattr(clipped_grad_norm, "detach")
            else float(clipped_grad_norm),
        }
        if isinstance(step_metrics, dict):
            for key, value in step_metrics.items():
                if key == "loss":
                    continue
                if hasattr(value, "detach"):
                    loss_row[key] = float(value.detach().cpu())
                else:
                    loss_row[key] = float(value)
        loss_history.append(loss_row)

        now = time.monotonic()
        if now - last_log > 30 or step == 1:
            sec_per_step = (now - start) / max(step, 1)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "sec_per_step": sec_per_step,
                        "elapsed_hours": elapsed_hours,
                        **{k: v for k, v in loss_row.items() if k.startswith("loss_") or k.endswith("_gate")},
                    }
                )
            )
            last_log = now

        if step in config.save_steps:
            ckpt = save_checkpoint(
                pipeline,
                action_encoder,
                action_injector,
                optimizer,
                config,
                step,
                loss_history,
                f"step_{step:06d}",
            )
            if step in config.sample_steps:
                sample_validation(pipeline, val_rows, config, step, ckpt / "validation_samples")
                checkpoint_volume.commit()

        if external_stop_requested(config, step):
            completed_step = step
            stop_reason = "external_stop"
            break

    if completed_step < config.max_steps:
        if stop_reason == "time_limit":
            label = "final_before_timeout"
        elif stop_reason == "external_stop":
            label = f"final_external_stop_step_{completed_step:06d}"
        else:
            label = f"final_step_{completed_step:06d}"
        ckpt = save_checkpoint(
            pipeline,
            action_encoder,
            action_injector,
            optimizer,
            config,
            completed_step,
            loss_history,
            label,
        )
        sample_validation(pipeline, val_rows, config, completed_step, ckpt / "validation_samples")
        checkpoint_volume.commit()

    summary = {
        "run_name": config.run_name,
        "completed_step": completed_step,
        "stop_reason": stop_reason,
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "skipped_train_missing_latents": skipped_train_missing_latents,
        "skipped_val_missing_latents": skipped_val_missing_latents,
        "trainable_parameter_counts": trainable_parameter_counts,
        "loss_history_tail": loss_history[-20:],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_dir / "run_summary.json", summary)
    checkpoint_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "",
    max_steps: int = 3000,
    max_train_hours: float = 8.0,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    seed: int = 231,
    train_limit: int = 0,
    val_limit: int = 32,
    num_val_samples: int = 4,
    latent_prefix: str = DEFAULT_LATENT_PREFIX,
    base_checkpoint: str = CKPT_2B,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = 0,
    disable_text_conditioning: bool = DEFAULT_DISABLE_TEXT_CONDITIONING,
    action_conditioning: bool = DEFAULT_ACTION_CONDITIONING,
    action_encoder_type: str = DEFAULT_ACTION_ENCODER_TYPE,
    action_dim: int = 0,
    action_token_count: int = 24,
    action_hidden_dim: int = 1024,
    action_dropout: float = 0.0,
    action_learning_rate: float = 1e-4,
    action_injector_learning_rate: float = 1e-4,
    action_gate_learning_rate: float = 1e-3,
    action_injector_heads: int = 8,
    action_midblock_start: int = -1,
    action_midblock_end: int = -1,
    action_transformer_layers: int = 2,
    action_transformer_heads: int = 8,
    action_frame_count: int = TOTAL_FRAMES,
    frame_action_stats_relpath: str = DEFAULT_FRAME_ACTION_STATS_RELPATH,
    frame_action_feature_key: str = DEFAULT_FRAME_ACTION_FEATURE_KEY,
    baseline_lora_run_name: str = DEFAULT_BASELINE_LORA_RUN_NAME,
    baseline_lora_step: str = DEFAULT_BASELINE_LORA_STEP,
    freeze_transformer_lora: bool = DEFAULT_FREEZE_TRANSFORMER_LORA,
    expand_baseline_lora_to_rank: bool = DEFAULT_EXPAND_BASELINE_LORA_TO_RANK,
    skip_missing_latents: bool = False,
    checkpoint_steps: str = "",
    timestep_sampling: str = "uniform",
    timestep_lognormal_mean: float = 0.0,
    timestep_lognormal_std: float = 1.0,
    hf_teacher_loss_weight: float = 0.0,
    diffusion_loss_weight: float = 1.0,
    lowfreq_target_loss_weight: float = 0.0,
    lowfreq_delta_loss_weight: float = 0.0,
    action_motion_aux_loss_weight: float = 0.0,
    action_residual_loss_weight: float = 0.0,
    action_gate_loss_weight: float = 0.0,
    action_gate_scale: float = 1.0,
    action_gate_bound: float = 0.25,
    resume_from_run_name: str = "",
    resume_from_checkpoint: str = "",
    external_stop_relpath: str = "",
    external_stop_check_steps: int = 100,
) -> None:
    if not run_name:
        suffix = "action_lora_r16" if action_conditioning else "visual_lora_r16"
        run_name = datetime.now(timezone.utc).strftime(f"ltx2b_waymo24_{suffix}_%Y%m%d_%H%M%S")
    save_steps = parse_step_list(checkpoint_steps, max_steps)
    uses_frame_actions = action_conditioning and action_encoder_type.startswith("frame_")
    resolved_action_dim = (
        action_dim
        if action_dim > 0
        else (112 if action_encoder_type == TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE else (18 if uses_frame_actions else 112))
    )
    resolved_action_token_count = (
        action_frame_count
        if action_conditioning
        and action_encoder_type in TEMPORAL_BOTTLENECK_ACTION_MODES
        else action_token_count
    )
    train_manifest = "manifests/train_windows_24fps_121f.csv"
    val_manifest = "manifests/val_windows_24fps_121f.csv"
    if action_conditioning:
        if uses_frame_actions:
            train_manifest = "manifests/train_windows_24fps_121f_frame_action_conditions.csv"
            val_manifest = "manifests/val_windows_24fps_121f_frame_action_conditions.csv"
        else:
            train_manifest = "manifests/train_windows_24fps_121f_action_conditions.csv"
            val_manifest = "manifests/val_windows_24fps_121f_action_conditions.csv"
    config = TrainConfig(
        run_name=run_name,
        max_steps=max_steps,
        max_train_hours=max_train_hours,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        save_steps=save_steps,
        sample_steps=save_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha if lora_alpha > 0 else lora_rank,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        disable_text_conditioning=disable_text_conditioning,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        latent_prefix=latent_prefix,
        base_checkpoint=base_checkpoint,
        num_val_samples=num_val_samples,
        action_conditioning=action_conditioning,
        action_encoder_type=action_encoder_type,
        action_dim=resolved_action_dim,
        action_token_count=resolved_action_token_count,
        action_hidden_dim=action_hidden_dim,
        action_dropout=action_dropout,
        action_learning_rate=action_learning_rate,
        action_injector_learning_rate=action_injector_learning_rate,
        action_gate_learning_rate=action_gate_learning_rate,
        action_injector_heads=action_injector_heads,
        action_midblock_start=action_midblock_start,
        action_midblock_end=action_midblock_end,
        action_transformer_layers=action_transformer_layers,
        action_transformer_heads=action_transformer_heads,
        action_frame_count=action_frame_count,
        frame_action_stats_relpath=frame_action_stats_relpath,
        frame_action_feature_key=frame_action_feature_key,
        baseline_lora_run_name=baseline_lora_run_name,
        baseline_lora_step=baseline_lora_step,
        freeze_transformer_lora=freeze_transformer_lora,
        expand_baseline_lora_to_rank=expand_baseline_lora_to_rank,
        skip_missing_latents=skip_missing_latents,
        timestep_sampling=timestep_sampling,
        timestep_lognormal_mean=timestep_lognormal_mean,
        timestep_lognormal_std=timestep_lognormal_std,
        hf_teacher_loss_weight=hf_teacher_loss_weight,
        diffusion_loss_weight=diffusion_loss_weight,
        lowfreq_target_loss_weight=lowfreq_target_loss_weight,
        lowfreq_delta_loss_weight=lowfreq_delta_loss_weight,
        action_motion_aux_loss_weight=action_motion_aux_loss_weight,
        action_residual_loss_weight=action_residual_loss_weight,
        action_gate_loss_weight=action_gate_loss_weight,
        action_gate_scale=action_gate_scale,
        action_gate_bound=action_gate_bound,
        resume_from_run_name=resume_from_run_name,
        resume_from_checkpoint=resume_from_checkpoint,
        external_stop_relpath=external_stop_relpath,
        external_stop_check_steps=external_stop_check_steps,
    )
    result = train.remote(asdict(config), train_limit=train_limit, val_limit=val_limit)
    print(json.dumps(result, indent=2, sort_keys=True))
