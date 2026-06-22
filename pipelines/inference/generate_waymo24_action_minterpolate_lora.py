from __future__ import annotations

import csv
import json
import os
import re
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx2b-waymo24-action-lora-local-minterpolate-infer")
MODELS_VOLUME_NAME = "models"
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-waymo24fps-action-lora-r16-checkpoints",
)
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-waymo24-action-lora-local-minterpolate-inference",
)

MODELS_ROOT = Path("/models")
DATA_ROOT = Path("/data")
CHECKPOINT_ROOT = Path("/checkpoints")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")
HF_CACHE_ROOT = MODELS_ROOT / "hf_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_ROOT / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_CKPT = os.environ.get("LTX_BASE_CKPT", "ltxv-2b-0.9.6-dev-04-25.safetensors")
LORA_RUN_NAME = os.environ.get("LTX_LORA_RUN_NAME", "ltx2b_waymo24_action_lora_r16_seed231")
DEFAULT_LORA_STEP = os.environ.get("LTX_DEFAULT_LORA_STEP", "step_003000")
BASE_ONLY_STEP = "base"
FRAME_ACTION_STATS_RELPATH = "manifests/frame_action_24fps_normalization_stats.json"
FULL112_FRAME_ACTION_STATS_RELPATH = "manifests/frame_action_24fps_full112_normalization_stats.json"
MIDBLOCK_GATED_XATTN_MODE = "frame_midblock_gated_xattn"
TEMPORAL_BOTTLENECK_HF_TEACHER_MODE = "frame_temporal_bottleneck_hf_teacher"
TEMPORAL_BOTTLENECK_LOWFREQ_V3_MODE = "frame_temporal_bottleneck_lowfreq_v3"
TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE = "frame_temporal_bottleneck_fullaction_motion_v4"
TEMPORAL_BOTTLENECK_ACTION_MODES = {
    TEMPORAL_BOTTLENECK_HF_TEACHER_MODE,
    TEMPORAL_BOTTLENECK_LOWFREQ_V3_MODE,
    TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE,
}

FPS = 24
WIDTH = 512
HEIGHT = 512
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = CONTEXT_FRAMES + FUTURE_FRAMES
SEED = 231

LOCAL_SOURCE_DIR = Path("data/inference_input_clips/interpolated_24fps_waymo_full20s")
RUNS_ROOT = PurePosixPath(
    os.environ.get("LTX_RUNS_ROOT", "action_lora_24fps_minterpolate_seed231_runs")
)
IMAGE_COND_NOISE_SCALE = float(os.environ.get("LTX_IMAGE_COND_NOISE_SCALE", "0.0"))
DEFAULT_ACTION_GATE_SCALE = float(os.environ.get("WAYMO24_ACTION_GATE_SCALE", "1.0"))
DEFAULT_ACTION_VECTOR_SCALE = float(os.environ.get("WAYMO24_ACTION_VECTOR_SCALE", "1.0"))
DEFAULT_COUNTERFACTUAL_ACTION_MODE = os.environ.get("WAYMO24_COUNTERFACTUAL_ACTION_MODE", "correct")
DEFAULT_COUNTERFACTUAL_ROTATION = int(os.environ.get("WAYMO24_COUNTERFACTUAL_ROTATION", "1"))
DEFAULT_DISABLE_TEXT_CONDITIONING = os.environ.get("WAYMO24_DISABLE_TEXT_CONDITIONING", "0") == "1"
COUNTERFACTUAL_ACTION_MODES = {"correct", "zero", "shuffled", "reversed_future"}

DEFAULT_PROMPT = (
    "Forward-facing autonomous driving video from a real Waymo-style car-mounted front camera. "
    "Use the observed 49-frame 24 FPS context as fixed history. Generate only the natural future "
    "continuation after the final observed frame. Preserve the same camera viewpoint, road layout, "
    "lane geometry, nearby vehicles, traffic lights, sidewalks, buildings, lighting, and weather. "
    "Follow the provided numeric ego-motion plan. Do not restart the scene, do not copy the observed "
    "clip again, do not jump to a new location, and do not introduce a camera cut."
)
DEFAULT_NEGATIVE_PROMPT = (
    "repeated input, scene restart, camera cut, new location, wrong viewpoint, rear camera, side camera, "
    "blurry, jittery, distorted, impossible vehicle motion, teleporting cars, duplicated cars"
)


@dataclass(frozen=True)
class SourceClip:
    scene_token: str
    source_filename: str
    source_relpath: str
    source_volume: str = "artifacts"
    window_id: str = ""
    window_idx: int = -1


app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .env(
        {
            "HF_HOME": str(HF_CACHE_ROOT),
            "HF_HUB_CACHE": str(HF_CACHE_ROOT / "hub"),
            "TRANSFORMERS_CACHE": str(HF_CACHE_ROOT / "transformers"),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "huggingface_hub",
        "av",
        "imageio",
        "imageio-ffmpeg",
        "imageio[ffmpeg]",
        "peft",
        "safetensors",
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video",
        "cd /workspace/LTX-Video && python -m pip install -e '.[inference-script]'",
    )
)


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def scene_token_from_path(path: Path) -> str:
    match = re.search(r"context_([0-9a-f]{12})_frames", path.name)
    if not match:
        raise ValueError(f"Could not parse scene token from {path.name}")
    return match.group(1)


def discover_sources(limit: int = 0) -> list[Path]:
    paths = sorted(LOCAL_SOURCE_DIR.glob("*_minterpolate_24fps.mp4"))
    if not paths:
        raise FileNotFoundError(f"No minterpolate 24 FPS clips found in {LOCAL_SOURCE_DIR}")
    return paths[:limit] if limit > 0 else paths


def upload_sources(source_paths: list[Path], run_root_relpath: PurePosixPath) -> list[dict[str, Any]]:
    uploaded: list[dict[str, Any]] = []
    remote_root = run_root_relpath / "source_minterpolate_24fps_full20s"
    with artifacts_volume.batch_upload(force=True) as batch:
        for path in source_paths:
            relpath = remote_root / path.name
            batch.put_file(path, relpath.as_posix())
            uploaded.append(
                asdict(
                    SourceClip(
                        scene_token=scene_token_from_path(path),
                        source_filename=path.name,
                        source_relpath=relpath.as_posix(),
                    )
                )
            )
    return uploaded


def load_sources_payload(path: str) -> list[dict[str, Any]]:
    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("sources", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Sources JSON must contain a non-empty list: {payload_path}")
    return payload


def source_clip_path(source: SourceClip) -> Path:
    if source.source_volume == "data":
        return DATA_ROOT / source.source_relpath
    if source.source_volume == "artifacts":
        return ARTIFACTS_ROOT / source.source_relpath
    raise ValueError(f"Unsupported source_volume={source.source_volume!r}")


def use_lora_adapter(lora_step: str) -> bool:
    return lora_step.lower() not in {"", BASE_ONLY_STEP, "none", "no_lora", "base_only"}


def checkpoint_label(lora_step: str, base_label: str = "base_no_lora") -> str:
    return lora_step.replace("step_", "step") if use_lora_adapter(lora_step) else base_label


def ensure_base_checkpoint(base_ckpt_name: str = BASE_CKPT) -> Path:
    src = MODELS_ROOT / "ltx" / base_ckpt_name
    dst = REPO / base_ckpt_name
    if not src.exists():
        raise FileNotFoundError(f"Missing base checkpoint in Modal volume: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return dst


def write_video(path: Path, video_tensor, fps: int = FPS) -> None:
    import imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in video_np:
            writer.append_data(frame)


def load_action_rows(split: str, window_idx: int, *, use_frame_actions: bool = False) -> dict[str, dict[str, str]]:
    suffix = "frame_action_conditions" if use_frame_actions else "action_conditions"
    manifest_path = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f_{suffix}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing action-condition manifest: {manifest_path}")
    rows_by_token: dict[str, dict[str, str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["window_idx"]) != window_idx:
                continue
            token = row["scenario_id"][:12]
            rows_by_token[token] = row
    return rows_by_token


def load_frame_action_stats(stats_relpath: str = FRAME_ACTION_STATS_RELPATH) -> dict[str, Any]:
    stats_path = DATA_ROOT / stats_relpath
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing frame-action normalization stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "p01": stats["p01"],
        "p99": stats["p99"],
        "feature_order": stats.get("feature_order", []),
    }


def load_frame_action_vector(
    row: dict[str, str],
    stats: dict[str, Any],
    *,
    feature_key: str = "actions",
) -> list[list[float]]:
    import numpy as np

    action_path = DATA_ROOT / row["frame_action_relpath"]
    if not action_path.exists():
        raise FileNotFoundError(f"Missing frame-action cache: {action_path}")
    with np.load(action_path) as payload:
        if feature_key not in payload:
            raise KeyError(
                f"Missing frame-action array key {feature_key!r} in {action_path}; "
                f"available keys={list(payload.keys())}"
            )
        actions = payload[feature_key].astype("float32")
    mean = np.asarray(stats["mean"], dtype="float32")
    std = np.maximum(np.asarray(stats["std"], dtype="float32"), 1e-6)
    p01 = np.asarray(stats["p01"], dtype="float32")
    p99 = np.asarray(stats["p99"], dtype="float32")
    if mean.shape[-1] != actions.shape[-1]:
        raise ValueError(
            f"Frame-action stats dim {mean.shape[-1]} does not match {feature_key!r} dim {actions.shape[-1]} "
            f"for {action_path}."
        )
    actions = np.minimum(np.maximum(actions, p01), p99)
    actions = np.clip((actions - mean) / std, -5.0, 5.0)
    return actions.tolist()


def validate_counterfactual_action_mode(mode: str) -> str:
    if mode not in COUNTERFACTUAL_ACTION_MODES:
        raise ValueError(f"Unknown counterfactual_action_mode={mode}; expected one of {sorted(COUNTERFACTUAL_ACTION_MODES)}")
    return mode


def select_counterfactual_source(
    source: SourceClip,
    source_index: int,
    sources_payload: list[dict[str, Any]],
    *,
    mode: str,
    rotation: int,
) -> SourceClip:
    if mode != "shuffled":
        return source
    if len(sources_payload) < 2:
        raise ValueError("counterfactual_action_mode=shuffled requires at least 2 source clips.")
    shifted = (source_index + rotation) % len(sources_payload)
    if shifted == source_index:
        shifted = (shifted + 1) % len(sources_payload)
    return SourceClip(**sources_payload[shifted])


def transform_counterfactual_action_vector(
    action_vector: list[Any],
    *,
    mode: str,
    use_frame_actions: bool,
) -> list[Any]:
    if mode in {"correct", "shuffled"}:
        return action_vector
    if mode == "zero":
        if use_frame_actions:
            return [[0.0 for _ in frame] for frame in action_vector]
        return [0.0 for _ in action_vector]
    if mode == "reversed_future":
        if not use_frame_actions:
            raise ValueError("counterfactual_action_mode=reversed_future requires frame-aligned actions.")
        return list(action_vector[:CONTEXT_FRAMES]) + list(reversed(action_vector[CONTEXT_FRAMES:]))
    raise ValueError(f"Unhandled counterfactual action mode: {mode}")


def get_transformer_core(transformer):
    return transformer.base_model.model if hasattr(transformer, "base_model") else transformer


def get_transformer_inner_dim(transformer) -> int:
    core = get_transformer_core(transformer)
    if not hasattr(core, "inner_dim"):
        raise AttributeError("Expected LTX transformer core to expose `inner_dim`.")
    return int(core.inner_dim)


def install_action_adaln_hook(transformer) -> None:
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
    get_transformer_core(transformer)._waymo_action_adaln_delta = action_delta


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
    import math

    resolved_start = num_blocks // 3 if start < 0 else start
    resolved_end = math.ceil((2 * num_blocks) / 3) if end < 0 else end
    if not (0 <= resolved_start < resolved_end <= num_blocks):
        raise ValueError(
            f"Invalid middle-block range start={resolved_start}, end={resolved_end}, num_blocks={num_blocks}."
        )
    return list(range(resolved_start, resolved_end))


def make_midblock_action_injector_from_payload(payload: dict[str, Any], transformer):
    import torch

    _, blocks = get_transformer_blocks(transformer)
    metadata = dict(payload.get("metadata") or {})
    hidden_dim = int(metadata.get("hidden_dim", get_transformer_inner_dim(transformer)))
    num_heads = int(metadata.get("num_heads", payload.get("action_injector_heads", 8)))
    if hidden_dim % num_heads != 0:
        raise ValueError(f"action_injector_heads={num_heads} must divide transformer hidden_dim={hidden_dim}.")
    block_indices = metadata.get("block_indices")
    if not block_indices:
        block_indices = resolve_midblock_indices(
            len(blocks),
            start=int(payload.get("action_midblock_start", -1)),
            end=int(payload.get("action_midblock_end", -1)),
        )
    block_indices = [int(idx) for idx in block_indices]
    dropout = float(metadata.get("dropout", 0.0))

    class MidBlockActionInjector(torch.nn.Module):
        def __init__(self, hidden_dim: int, block_indices: list[int], num_heads: int, dropout: float):
            super().__init__()
            self.conditioning_mode = "midblock_gated_xattn"
            self.hidden_dim = hidden_dim
            self.block_indices = list(block_indices)
            self.num_heads = num_heads
            self.dropout = dropout
            self.gate_scale = 1.0
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
                self.gates[key] = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward_block(self, block_idx: int, hidden_states, action_tokens):
            key = str(block_idx)
            if action_tokens is None or key not in self.layers:
                return hidden_states
            layer = self.layers[key]
            original_dtype = hidden_states.dtype
            query = layer["norm"](hidden_states.float())
            key_value = action_tokens.float()
            attn_out, _ = layer["attn"](query, key_value, key_value, need_weights=False)
            gated = float(self.gate_scale) * self.gates[key].float() * layer["dropout"](attn_out)
            return hidden_states + gated.to(dtype=original_dtype)

        def metadata(self) -> dict[str, Any]:
            return {
                "conditioning_mode": self.conditioning_mode,
                "hidden_dim": self.hidden_dim,
                "block_indices": self.block_indices,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }

    return MidBlockActionInjector(hidden_dim, block_indices, num_heads, dropout).to("cuda")


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


def set_midblock_action_tokens(transformer, action_tokens) -> None:
    core = get_transformer_core(transformer)
    if getattr(core, "_waymo_midblock_action_injector", None) is None:
        raise RuntimeError("Mid-block action tokens were set before installing the action injector.")
    core._waymo_midblock_action_tokens = action_tokens


def set_midblock_action_gate_scale(transformer, gate_scale: float) -> None:
    core = get_transformer_core(transformer)
    injector = getattr(core, "_waymo_midblock_action_injector", None)
    if injector is None:
        raise RuntimeError("Mid-block action gate scale was set before installing the action injector.")
    injector.gate_scale = float(gate_scale)


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


def make_temporal_bottleneck_action_injector_from_payload(payload: dict[str, Any], transformer):
    import torch
    import torch.nn.functional as F

    _, blocks = get_transformer_blocks(transformer)
    metadata = dict(payload.get("metadata") or {})
    hidden_dim = int(metadata.get("hidden_dim", get_transformer_inner_dim(transformer)))
    action_hidden_dim = int(metadata.get("action_hidden_dim", payload.get("action_hidden_dim", 256)))
    block_indices = metadata.get("block_indices") or [10, 14, 18]
    block_indices = [int(idx) for idx in block_indices]
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
        ):
            super().__init__()
            self.conditioning_mode = "temporal_bottleneck"
            self.action_hidden_dim = action_hidden_dim
            self.hidden_dim = hidden_dim
            self.block_indices = list(block_indices)
            self.gate_scale = float(gate_scale)
            self.bounded_gates = bool(bounded_gates)
            self.gate_bound = float(gate_bound)
            self.projectors = torch.nn.ModuleDict()
            self.gates = torch.nn.ParameterDict()
            for block_idx in self.block_indices:
                key = str(block_idx)
                self.projectors[key] = torch.nn.Sequential(
                    torch.nn.LayerNorm(action_hidden_dim),
                    torch.nn.Linear(action_hidden_dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                )
                self.gates[key] = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def _pool_frame_controls(self, frame_controls, latent_time_ids):
            num_latent_times = int(latent_time_ids.max().detach().cpu().item()) + 1
            return F.adaptive_avg_pool1d(frame_controls.float().transpose(1, 2), num_latent_times).transpose(1, 2)

        def forward_block(self, block_idx: int, hidden_states, frame_controls, latent_time_ids):
            key = str(block_idx)
            if frame_controls is None or key not in self.projectors:
                return hidden_states
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
            residual = residual_by_time[:, latent_time_ids, :]
            raw_gate = self.gates[key].float()
            if self.bounded_gates:
                effective_gate = float(self.gate_scale) * float(self.gate_bound) * torch.tanh(raw_gate)
            else:
                effective_gate = float(self.gate_scale) * raw_gate
            gated = effective_gate * residual
            return hidden_states + gated.to(dtype=original_dtype)

        def metadata(self) -> dict[str, Any]:
            return {
                "conditioning_mode": self.conditioning_mode,
                "action_hidden_dim": self.action_hidden_dim,
                "hidden_dim": self.hidden_dim,
                "block_indices": self.block_indices,
                "gate_scale": self.gate_scale,
                "bounded_gates": self.bounded_gates,
                "gate_bound": self.gate_bound,
            }

    return TemporalBottleneckActionInjector(
        action_hidden_dim,
        hidden_dim,
        block_indices,
        float(metadata.get("gate_scale", payload.get("action_gate_scale", 1.0))),
        bounded_gates=bool(metadata.get("bounded_gates", False)),
        gate_bound=float(metadata.get("gate_bound", payload.get("action_gate_bound", 0.25))),
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
        handles.append(blocks[block_idx].register_forward_hook(make_hook(block_idx)))
    core._waymo_temporal_bottleneck_action_handles = handles


def set_temporal_bottleneck_action_controls(transformer, frame_controls) -> None:
    core = get_transformer_core(transformer)
    if getattr(core, "_waymo_temporal_bottleneck_action_injector", None) is None:
        raise RuntimeError("Temporal bottleneck action controls were set before installing the injector.")
    core._waymo_temporal_bottleneck_frame_controls = frame_controls


def set_temporal_bottleneck_action_gate_scale(transformer, gate_scale: float) -> None:
    core = get_transformer_core(transformer)
    injector = getattr(core, "_waymo_temporal_bottleneck_action_injector", None)
    if injector is None:
        raise RuntimeError("Temporal bottleneck gate scale was set before installing the injector.")
    injector.gate_scale = float(gate_scale)


def clear_temporal_bottleneck_action_controls(transformer) -> None:
    core = get_transformer_core(transformer)
    if hasattr(core, "_waymo_temporal_bottleneck_frame_controls"):
        core._waymo_temporal_bottleneck_frame_controls = None


def make_global_action_token_encoder(config: dict[str, Any], output_dim: int):
    import torch

    class GlobalActionTokenEncoder(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, token_count: int, output_dim: int, dropout: float):
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

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            state = self.state_mlp(self.input_norm(action_vectors.float()))
            token_hidden = self.token_embeddings + state[:, None, :]
            return self.token_proj(token_hidden)

    return GlobalActionTokenEncoder(
        input_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        token_count=int(config["action_token_count"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_adaln_action_encoder(config: dict[str, Any], inner_dim: int):
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

        def forward(self, action_vectors):
            if action_vectors.shape[-1] != self.input_dim:
                raise ValueError(f"Expected action dim {self.input_dim}, got {action_vectors.shape[-1]}")
            return self.net(self.input_norm(action_vectors.float())).unsqueeze(1)

    return AdaLNActionEncoder(
        input_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_temporal_per_point_action_encoder(config: dict[str, Any], output_dim: int):
    import torch

    if int(config["action_dim"]) != 112:
        raise ValueError(f"Temporal per-point encoder expects action_dim=112, got {config['action_dim']}")
    if int(config["action_token_count"]) != 24:
        raise ValueError(
            "Temporal per-point encoder emits exactly 24 tokens; "
            f"got action_token_count={config['action_token_count']}"
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
                [past_vel.mean(dim=1), past_vel.std(dim=1, unbiased=False), past_vel[:, 0], past_vel[:, -1]],
                dim=-1,
            )
            accel_stats = torch.cat(
                [past_accel.mean(dim=1), past_accel.std(dim=1, unbiased=False), past_accel[:, 0], past_accel[:, -1]],
                dim=-1,
            )
            current_summary = torch.cat(
                [current_state, past_vel[:, -1], past_accel[:, -1], future_xy[:, 0], future_delta[:, 0]],
                dim=-1,
            )
            future_summary = torch.cat(
                [current_state, future_xy[:, 0], future_xy[:, -1], future_delta.mean(dim=1), future_delta[:, -1]],
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
            return self.token_proj(torch.cat([future_hidden, summary_hidden], dim=1))

    return TemporalPerPointActionEncoder(
        hidden_dim=int(config["action_hidden_dim"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_tiny_transformer_action_encoder(config: dict[str, Any], output_dim: int):
    import torch

    if int(config["action_dim"]) != 112:
        raise ValueError(f"Tiny transformer encoder expects action_dim=112, got {config['action_dim']}")

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
        hidden_dim=int(config["action_hidden_dim"]),
        output_dim=output_dim,
        token_count=int(config["action_token_count"]),
        dropout=float(config.get("action_dropout", 0.0)),
        num_layers=int(config.get("action_transformer_layers", 2)),
        num_heads=int(config.get("action_transformer_heads", 8)),
    )


def frame_segment_ids(frame_count: int, context_frames: int):
    import torch

    ids = torch.zeros(frame_count, dtype=torch.long)
    ids[context_frames:] = 1
    return ids


def make_frame_global_mlp_action_encoder(config: dict[str, Any], output_dim: int):
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
        frame_count=int(config.get("action_frame_count", TOTAL_FRAMES)),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        token_count=int(config["action_token_count"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_frame_temporal_pool_action_encoder(config: dict[str, Any], output_dim: int):
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
        frame_count=int(config.get("action_frame_count", TOTAL_FRAMES)),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        token_count=int(config["action_token_count"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_frame_transformer_action_encoder(config: dict[str, Any], output_dim: int):
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
        frame_count=int(config.get("action_frame_count", TOTAL_FRAMES)),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        token_count=int(config["action_token_count"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
        num_layers=int(config.get("action_transformer_layers", 2)),
        num_heads=int(config.get("action_transformer_heads", 8)),
    )


def make_frame_temporal_bottleneck_action_encoder(config: dict[str, Any]):
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
        frame_count=int(config.get("action_frame_count", TOTAL_FRAMES)),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        dropout=float(config.get("action_dropout", 0.0)),
        num_layers=int(config.get("action_transformer_layers", 4)),
        num_heads=int(config.get("action_transformer_heads", 8)),
        use_motion_aux_head=(
            str(config.get("action_encoder_type", "")) == TEMPORAL_BOTTLENECK_FULLACTION_MOTION_V4_MODE
            or float(config.get("action_motion_aux_loss_weight", 0.0)) > 0.0
        ),
    )


def make_frame_adaln_action_encoder(config: dict[str, Any], inner_dim: int):
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
        frame_count=int(config.get("action_frame_count", TOTAL_FRAMES)),
        action_dim=int(config["action_dim"]),
        hidden_dim=int(config["action_hidden_dim"]),
        output_dim=output_dim,
        dropout=float(config.get("action_dropout", 0.0)),
    )


def make_action_token_encoder(config: dict[str, Any], output_dim: int):
    encoder_type = str(config.get("action_encoder_type", "global_mlp"))
    if encoder_type == "global_mlp":
        return make_global_action_token_encoder(config, output_dim=output_dim)
    if encoder_type == "temporal_per_point":
        return make_temporal_per_point_action_encoder(config, output_dim=output_dim)
    if encoder_type == "tiny_transformer":
        return make_tiny_transformer_action_encoder(config, output_dim=output_dim)
    if encoder_type == "frame_global_mlp":
        return make_frame_global_mlp_action_encoder(config, output_dim=output_dim)
    if encoder_type == "frame_temporal_pool":
        return make_frame_temporal_pool_action_encoder(config, output_dim=output_dim)
    if encoder_type == "frame_transformer":
        return make_frame_transformer_action_encoder(config, output_dim=output_dim)
    raise ValueError(f"Unknown action_encoder_type: {encoder_type}")


def make_action_encoder(config: dict[str, Any], *, prompt_hidden_dim: int, transformer_inner_dim: int):
    encoder_type = str(config.get("action_encoder_type", "global_mlp"))
    if encoder_type == "adaln":
        return make_adaln_action_encoder(config, inner_dim=transformer_inner_dim)
    if encoder_type == "frame_adaln":
        return make_frame_adaln_action_encoder(config, inner_dim=transformer_inner_dim)
    if encoder_type == MIDBLOCK_GATED_XATTN_MODE:
        encoder = make_frame_transformer_action_encoder(config, output_dim=transformer_inner_dim)
        encoder.conditioning_mode = "midblock_gated_xattn"
        return encoder
    if encoder_type in TEMPORAL_BOTTLENECK_ACTION_MODES:
        return make_frame_temporal_bottleneck_action_encoder(config)
    return make_action_token_encoder(config, output_dim=prompt_hidden_dim)


def encode_action_conditioned_prompt(
    pipeline,
    action_encoder,
    action_vector,
    prompt: str,
    negative_prompt: str,
    action_vector_scale: float = 1.0,
    disable_text_conditioning: bool = False,
):
    import torch

    device = torch.device("cuda")
    with torch.no_grad():
        text_embeds, text_mask, _, _ = pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=device,
            text_encoder_max_tokens=256,
        )
        negative_text_embeds, negative_text_mask, _, _ = pipeline.encode_prompt(
            negative_prompt,
            do_classifier_free_guidance=False,
            negative_prompt="",
            num_images_per_prompt=1,
            device=device,
            text_encoder_max_tokens=256,
        )
        if disable_text_conditioning:
            text_embeds = text_embeds * 0.0
            text_mask = text_mask * 0
            negative_text_embeds = negative_text_embeds * 0.0
            negative_text_mask = negative_text_mask * 0
        action_tensor = torch.tensor(action_vector, device=device, dtype=torch.float32).unsqueeze(0)
        action_tensor = action_tensor * float(action_vector_scale)
        conditioning_mode = getattr(action_encoder, "conditioning_mode", "tokens")
        if conditioning_mode == "adaln":
            action_delta = action_encoder(action_tensor).to(dtype=text_embeds.dtype, device=device)
            if disable_text_conditioning:
                text_mask = text_mask.clone()
                negative_text_mask = negative_text_mask.clone()
                text_mask[:, :1] = 1
                negative_text_mask[:, :1] = 1
            return text_embeds, text_mask, negative_text_embeds, negative_text_mask, action_delta, None, None
        if conditioning_mode == "midblock_gated_xattn":
            action_tokens = action_encoder(action_tensor).to(device=device)
            if disable_text_conditioning:
                text_mask = text_mask.clone()
                negative_text_mask = negative_text_mask.clone()
                text_mask[:, :1] = 1
                negative_text_mask[:, :1] = 1
            return text_embeds, text_mask, negative_text_embeds, negative_text_mask, None, action_tokens, None
        if conditioning_mode == "temporal_bottleneck":
            action_controls = action_encoder(action_tensor).to(device=device)
            if disable_text_conditioning:
                text_mask = text_mask.clone()
                negative_text_mask = negative_text_mask.clone()
                text_mask[:, :1] = 1
                negative_text_mask[:, :1] = 1
            return text_embeds, text_mask, negative_text_embeds, negative_text_mask, None, None, action_controls
        action_tokens = action_encoder(action_tensor).to(dtype=text_embeds.dtype, device=device)
        action_mask = torch.ones(action_tokens.shape[:2], dtype=text_mask.dtype, device=device)
        prompt_embeds = torch.cat([text_embeds, action_tokens], dim=1)
        prompt_attention_mask = torch.cat([text_mask, action_mask], dim=1)
        negative_prompt_embeds = torch.cat([negative_text_embeds, action_tokens], dim=1)
        negative_prompt_attention_mask = torch.cat([negative_text_mask, action_mask], dim=1)
    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, None, None, None


@app.function(
    image=image,
    gpu=os.environ.get("LTX_MODAL_GPU", "A100"),
    cpu=8,
    memory=49152,
    timeout=2 * 60 * 60,
    volumes={
        str(MODELS_ROOT): models_volume,
        str(DATA_ROOT): data_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
        str(ARTIFACTS_ROOT): artifacts_volume,
    },
)
def generate_all(
    sources_payload: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    lora_step: str,
    base_ckpt_name: str,
    lora_run_name: str,
    base_label: str,
    artifact_volume_name: str,
    prompt: str,
    negative_prompt: str,
    disable_text_conditioning: bool,
    seed: int,
    action_manifest_split: str,
    action_window_idx: int,
    action_gate_scale: float,
    action_vector_scale: float,
    counterfactual_action_mode: str,
    counterfactual_rotation: int,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from ltx_video.inference import create_ltx_video_pipeline, load_media_file
    from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem

    models_volume.reload()
    data_volume.reload()
    checkpoint_volume.reload()
    artifacts_volume.reload()

    base_ckpt = ensure_base_checkpoint(base_ckpt_name)
    ckpt_dir = CHECKPOINT_ROOT / lora_run_name / lora_step
    adapter_dir = ckpt_dir / "lora_adapter"
    action_encoder_path = ckpt_dir / "action_encoder.pt"
    action_injector_path = ckpt_dir / "action_injector.pt"
    using_lora = use_lora_adapter(lora_step)
    if not using_lora:
        raise ValueError("Action-conditioned inference requires a trained action LoRA checkpoint and action_encoder.pt.")
    if using_lora and not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_dir}")
    if using_lora and not action_encoder_path.exists():
        raise FileNotFoundError(f"Missing action encoder checkpoint: {action_encoder_path}")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(base_ckpt),
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        sampler="from_checkpoint",
        device="cuda",
        enhance_prompt=False,
    )
    models_volume.commit()
    if using_lora:
        from peft import PeftModel

        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, str(adapter_dir))
        pipeline.transformer.config = pipeline.transformer.base_model.model.config
    pipeline.transformer.eval()
    pipeline.transformer.to("cuda", dtype=torch.bfloat16)

    # Build the action encoder after text encoding so the output dim exactly matches T5 embeddings.
    prompt_probe, _, _, _ = pipeline.encode_prompt(
        prompt,
        do_classifier_free_guidance=False,
        negative_prompt="",
        num_images_per_prompt=1,
        device=torch.device("cuda"),
        text_encoder_max_tokens=256,
    )
    action_ckpt = torch.load(action_encoder_path, map_location="cuda")
    action_encoder_type = str(action_ckpt.get("action_encoder_type", "global_mlp"))
    if action_encoder_type in {MIDBLOCK_GATED_XATTN_MODE, *TEMPORAL_BOTTLENECK_ACTION_MODES} and not action_injector_path.exists():
        raise FileNotFoundError(f"Missing action injector checkpoint: {action_injector_path}")
    use_frame_actions = action_encoder_type.startswith("frame_")
    action_rows = load_action_rows(
        action_manifest_split,
        action_window_idx,
        use_frame_actions=use_frame_actions,
    )
    frame_action_stats_relpath = str(action_ckpt.get("frame_action_stats_relpath", FRAME_ACTION_STATS_RELPATH))
    frame_action_feature_key = str(action_ckpt.get("frame_action_feature_key", "actions"))
    frame_action_stats = load_frame_action_stats(frame_action_stats_relpath) if use_frame_actions else None
    action_encoder = make_action_encoder(
        action_ckpt,
        prompt_hidden_dim=prompt_probe.shape[-1],
        transformer_inner_dim=get_transformer_inner_dim(pipeline.transformer),
    ).to("cuda")
    action_encoder.load_state_dict(action_ckpt["state_dict"])
    action_encoder.eval()
    action_injector = None
    if getattr(action_encoder, "conditioning_mode", "tokens") == "adaln":
        install_action_adaln_hook(pipeline.transformer)
    elif getattr(action_encoder, "conditioning_mode", "tokens") == "midblock_gated_xattn":
        injector_ckpt = torch.load(action_injector_path, map_location="cuda")
        action_injector = make_midblock_action_injector_from_payload(injector_ckpt, pipeline.transformer)
        action_injector.load_state_dict(injector_ckpt["state_dict"])
        action_injector.eval()
        install_midblock_action_injector(pipeline.transformer, action_injector)
    elif getattr(action_encoder, "conditioning_mode", "tokens") == "temporal_bottleneck":
        injector_ckpt = torch.load(action_injector_path, map_location="cuda")
        action_injector = make_temporal_bottleneck_action_injector_from_payload(injector_ckpt, pipeline.transformer)
        action_injector.load_state_dict(injector_ckpt["state_dict"])
        action_injector.eval()
        install_temporal_bottleneck_action_injector(pipeline.transformer, action_injector)

    counterfactual_action_mode = validate_counterfactual_action_mode(counterfactual_action_mode)
    if counterfactual_action_mode != "correct" and not use_frame_actions:
        raise ValueError("Phase-1 counterfactual diagnostics require frame-aligned action conditions.")
    out_root = ARTIFACTS_ROOT / run_root_relpath / "generated_action_lora"
    results: list[dict[str, Any]] = []
    for source_index, source_payload in enumerate(sources_payload):
        source = SourceClip(**source_payload)
        action_source = select_counterfactual_source(
            source,
            source_index,
            sources_payload,
            mode=counterfactual_action_mode,
            rotation=counterfactual_rotation,
        )
        action_row = action_rows.get(action_source.scene_token)
        if action_row is None:
            raise KeyError(
                f"No action row for source token {action_source.scene_token} in "
                f"{action_manifest_split} window_idx={action_window_idx}"
            )
        if use_frame_actions:
            action_vector = load_frame_action_vector(
                action_row,
                frame_action_stats or {},
                feature_key=frame_action_feature_key,
            )
            if len(action_vector) != int(action_ckpt.get("action_frame_count", TOTAL_FRAMES)):
                raise ValueError(
                    f"Frame action count {len(action_vector)} != "
                    f"{action_ckpt.get('action_frame_count', TOTAL_FRAMES)}"
                )
            if action_vector and len(action_vector[0]) != int(action_ckpt["action_dim"]):
                raise ValueError(f"Frame action dim {len(action_vector[0])} != {action_ckpt['action_dim']}")
        else:
            action_vector = json.loads(action_row["action_vector_json"])
            if len(action_vector) != int(action_ckpt["action_dim"]):
                raise ValueError(f"Action vector length {len(action_vector)} != {action_ckpt['action_dim']}")
        action_vector = transform_counterfactual_action_vector(
            action_vector,
            mode=counterfactual_action_mode,
            use_frame_actions=use_frame_actions,
        )

        source_path = source_clip_path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source clip: {source_path}")

        step_label = checkpoint_label(lora_step, base_label=base_label)
        gate_label = f"g{float(action_gate_scale):.3f}".replace(".", "p").replace("-", "m")
        vector_label = f"v{float(action_vector_scale):.3f}".replace(".", "p").replace("-", "m")
        clip_id = (
            f"scene_{source.scene_token}_49ctx_72future_action_lora_{step_label}_"
            f"{counterfactual_action_mode}_{gate_label}_{vector_label}_seed{seed}"
        )
        output_dir = out_root / lora_step / clip_id
        output_path = output_dir / f"{clip_id}_24fps_121f.mp4"
        result_path = output_dir / "result.json"
        if output_path.exists() and result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        media = load_media_file(
            media_path=str(source_path),
            height=HEIGHT,
            width=WIDTH,
            max_frames=CONTEXT_FRAMES,
            padding=(0, 0, 0, 0),
            just_crop=True,
        )
        conditioning_items = [ConditioningItem(media, 0, 1.0)]
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            action_adaln_delta,
            action_midblock_tokens,
            action_temporal_controls,
        ) = encode_action_conditioned_prompt(
            pipeline,
            action_encoder,
            action_vector,
            prompt=prompt,
            negative_prompt=negative_prompt,
            action_vector_scale=action_vector_scale,
            disable_text_conditioning=disable_text_conditioning,
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if action_adaln_delta is not None:
            set_action_adaln_delta(pipeline.transformer, action_adaln_delta)
        if action_midblock_tokens is not None:
            set_midblock_action_gate_scale(pipeline.transformer, action_gate_scale)
            set_midblock_action_tokens(pipeline.transformer, action_midblock_tokens)
        if action_temporal_controls is not None:
            set_temporal_bottleneck_action_gate_scale(pipeline.transformer, action_gate_scale)
            set_temporal_bottleneck_action_controls(pipeline.transformer, action_temporal_controls)
        with torch.no_grad():
            try:
                video = pipeline(
                    prompt=None,
                    negative_prompt=None,
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
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                ).images[0]
            finally:
                if action_adaln_delta is not None:
                    clear_action_adaln_delta(pipeline.transformer)
                if action_midblock_tokens is not None:
                    clear_midblock_action_tokens(pipeline.transformer)
                if action_temporal_controls is not None:
                    clear_temporal_bottleneck_action_controls(pipeline.transformer)
        write_video(output_path, video, fps=FPS)

        record = {
            "scene_token": source.scene_token,
            "model_mode": f"{action_encoder_type}_{checkpoint_label(lora_step, base_label=base_label)}",
            "source_filename": source.source_filename,
            "source_relpath": source.source_relpath,
            "generated_video_relpath": output_path.relative_to(ARTIFACTS_ROOT).as_posix(),
            "output_dir_relpath": output_dir.relative_to(ARTIFACTS_ROOT).as_posix(),
            "base_checkpoint": base_ckpt_name,
            "lora_run_name": lora_run_name if using_lora else "",
            "lora_step": lora_step,
            "using_lora": using_lora,
            "adapter_relpath": adapter_dir.relative_to(CHECKPOINT_ROOT).as_posix() if using_lora else "",
            "action_encoder_relpath": action_encoder_path.relative_to(CHECKPOINT_ROOT).as_posix() if using_lora else "",
            "action_injector_relpath": action_injector_path.relative_to(CHECKPOINT_ROOT).as_posix()
            if action_injector is not None
            else "",
            "action_manifest_split": action_manifest_split,
            "action_window_idx": action_window_idx,
            "action_window_id": action_row["window_id"],
            "action_source_frame_id_10fps": action_row.get("action_source_frame_id_10fps", ""),
            "frame_action_relpath": action_row.get("frame_action_relpath", ""),
            "frame_action_feature_key": frame_action_feature_key if use_frame_actions else "",
            "frame_action_stats_relpath": frame_action_stats_relpath if use_frame_actions else "",
            "use_frame_actions": use_frame_actions,
            "action_gate_scale": float(action_gate_scale),
            "action_vector_scale": float(action_vector_scale),
            "counterfactual_action_mode": counterfactual_action_mode,
            "counterfactual_source_scene_token": action_source.scene_token,
            "counterfactual_rotation": int(counterfactual_rotation),
            "image_cond_noise_scale": IMAGE_COND_NOISE_SCALE,
            "intent_name": action_row.get("intent_name", ""),
            "intent_value": action_row.get("intent_value", ""),
            "action_dim": len(action_vector[0]) if use_frame_actions and action_vector else len(action_vector),
            "action_frame_count": len(action_vector) if use_frame_actions else 0,
            "action_token_count": int(action_ckpt["action_token_count"]),
            "action_encoder_type": action_encoder_type,
            "seed": seed,
            "fps": FPS,
            "width": WIDTH,
            "height": HEIGHT,
            "context_frames": CONTEXT_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "total_frames": TOTAL_FRAMES,
            "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
            "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "disable_text_conditioning": bool(disable_text_conditioning),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        artifacts_volume.commit()
        results.append(record)

    summary = {
        "run_root_relpath": run_root_relpath,
        "artifact_volume": artifact_volume_name,
        "base_checkpoint": base_ckpt_name,
        "lora_run_name": lora_run_name if using_lora else "",
        "lora_step": lora_step,
        "seed": seed,
        "action_gate_scale": float(action_gate_scale),
        "action_vector_scale": float(action_vector_scale),
        "counterfactual_action_mode": counterfactual_action_mode,
        "counterfactual_rotation": int(counterfactual_rotation),
        "disable_text_conditioning": bool(disable_text_conditioning),
        "num_outputs": len(results),
        "results": results,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = ARTIFACTS_ROOT / run_root_relpath / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    limit: int = 0,
    sources_json: str = "",
    seed: int = SEED,
    lora_step: str = DEFAULT_LORA_STEP,
    run_label: str = "",
    base_ckpt_name: str = BASE_CKPT,
    lora_run_name: str = LORA_RUN_NAME,
    base_label: str = "base_no_lora",
    action_manifest_split: str = "val",
    action_window_idx: int = 0,
    action_gate_scale: float = DEFAULT_ACTION_GATE_SCALE,
    action_vector_scale: float = DEFAULT_ACTION_VECTOR_SCALE,
    counterfactual_action_mode: str = DEFAULT_COUNTERFACTUAL_ACTION_MODE,
    counterfactual_rotation: int = DEFAULT_COUNTERFACTUAL_ROTATION,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    disable_text_conditioning: bool = DEFAULT_DISABLE_TEXT_CONDITIONING,
) -> None:
    counterfactual_action_mode = validate_counterfactual_action_mode(counterfactual_action_mode)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    step_label = checkpoint_label(lora_step, base_label=base_label)
    gate_label = f"g{float(action_gate_scale):.3f}".replace(".", "p").replace("-", "m")
    vector_label = f"v{float(action_vector_scale):.3f}".replace(".", "p").replace("-", "m")
    run_name = (
        run_label
        or f"action_lora_{step_label}_{counterfactual_action_mode}_{gate_label}_{vector_label}_49ctx_72future_seed{seed}_{timestamp}"
    )
    run_root_relpath = RUNS_ROOT / run_name
    uploaded_sources = load_sources_payload(sources_json) if sources_json else upload_sources(
        discover_sources(limit=limit),
        run_root_relpath,
    )
    summary = generate_all.remote(
        uploaded_sources,
        run_root_relpath=run_root_relpath.as_posix(),
        lora_step=lora_step,
        base_ckpt_name=base_ckpt_name,
        lora_run_name=lora_run_name,
        base_label=base_label,
        artifact_volume_name=ARTIFACTS_VOLUME_NAME,
        prompt=prompt,
        negative_prompt=negative_prompt,
        disable_text_conditioning=disable_text_conditioning,
        seed=seed,
        action_manifest_split=action_manifest_split,
        action_window_idx=action_window_idx,
        action_gate_scale=action_gate_scale,
        action_vector_scale=action_vector_scale,
        counterfactual_action_mode=counterfactual_action_mode,
        counterfactual_rotation=counterfactual_rotation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
