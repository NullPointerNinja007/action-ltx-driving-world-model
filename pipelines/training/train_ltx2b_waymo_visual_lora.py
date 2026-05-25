from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx2b-waymo24-visual-lora-r16-train")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
MODELS_VOLUME_NAME = "models"
CHECKPOINT_VOLUME_NAME = "ltx2b-waymo24fps-visual-lora-r16-checkpoints"

DATA_ROOT = Path("/data")
MODELS_ROOT = Path("/models")
CKPT_ROOT = Path("/checkpoints")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"

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

CHECKPOINT_STEPS = [0, 250, 500, 1000, 1500, 2000, 2500, 3000]


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
    train_manifest: str
    val_manifest: str
    num_val_samples: int


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

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


def ensure_checkpoint_symlink() -> Path:
    src = MODELS_ROOT / "ltx" / CKPT_2B
    dst = REPO / CKPT_2B
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


class LatentRowsDataset:
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file

        row = self.rows[idx]
        latent_path = DATA_ROOT / row["latent_relpath"]
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent cache: {latent_path}")
        latents = load_file(str(latent_path))["latents"].to(torch.bfloat16)
        return {"latents": latents, "row": row}


def make_batch(dataset: LatentRowsDataset, indices: list[int]) -> dict[str, Any]:
    import torch

    items = [dataset[i] for i in indices]
    return {
        "latents": torch.stack([item["latents"] for item in items], dim=0),
        "rows": [item["row"] for item in items],
    }


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


def setup_pipeline_and_lora(config: TrainConfig):
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from peft import LoraConfig, get_peft_model
    from ltx_video.inference import create_ltx_video_pipeline

    ckpt = ensure_checkpoint_symlink()
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
    pipeline.transformer.train()
    pipeline.transformer.to("cuda", dtype=torch.bfloat16)
    return pipeline


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


def training_step(pipeline, clean_latents, prompt_embeds, prompt_attention_mask):
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    import torch.nn.functional as F
    from ltx_video.models.autoencoders.vae_encode import latent_to_pixel_coords

    clean_latents = clean_latents.to("cuda", dtype=torch.bfloat16)
    bsz = clean_latents.shape[0]
    noise = torch.randn_like(clean_latents)
    timesteps = torch.rand(bsz, device=clean_latents.device, dtype=torch.float32).clamp(1e-3, 1.0)
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
    pred = pipeline.transformer(
        noisy_tokens.to(transformer_dtype),
        indices_grid=fractional_coords,
        encoder_hidden_states=prompt_batch.to(transformer_dtype),
        encoder_attention_mask=mask_batch,
        timestep=token_timesteps,
        skip_layer_mask=None,
        skip_layer_strategy=None,
        return_dict=False,
    )[0]
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
    loss = F.mse_loss(pred[mask].float(), target_tokens[mask].float())
    return loss


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
                image_cond_noise_scale=0.15,
                mixed_precision=False,
                offload_to_cpu=False,
                enhance_prompt=False,
                generator=generator,
            ).images[0]
        write_video(out_dir / f"step_{step:06d}_{sample_idx:02d}_{row['window_id']}.mp4", result, fps=FPS)
    pipeline.transformer.train()


def save_checkpoint(pipeline, optimizer, config: TrainConfig, step: int, loss_history: list[dict[str, float]], label: str) -> Path:
    ckpt_dir = CKPT_ROOT / config.run_name / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pipeline.transformer.save_pretrained(str(ckpt_dir / "lora_adapter"))
    import torch

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": asdict(config),
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
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=9 * 60 * 60,
    volumes={
        str(DATA_ROOT): data_volume,
        str(MODELS_ROOT): models_volume,
        str(CKPT_ROOT): checkpoint_volume,
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
    config = TrainConfig(**config_payload)
    seed_everything(config.seed)

    train_rows = load_manifest(DATA_ROOT / config.train_manifest, limit=train_limit)
    val_rows = load_manifest(DATA_ROOT / config.val_manifest, limit=val_limit)
    if not train_rows:
        raise RuntimeError("No training rows loaded.")
    if not val_rows:
        raise RuntimeError("No validation rows loaded.")

    pipeline = setup_pipeline_and_lora(config)
    prompt_embeds, prompt_attention_mask = encode_fixed_prompt(pipeline, config.prompt)

    trainable = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    dataset = LatentRowsDataset(train_rows)
    batch_iter = cycle_indices(len(dataset), config.batch_size, config.seed)

    loss_history: list[dict[str, float]] = []
    run_dir = CKPT_ROOT / config.run_name
    save_json(run_dir / "run_config.json", asdict(config))

    sample_validation(pipeline, val_rows, config, 0, run_dir / "validation_samples" / "step_000000_base_reference")
    save_checkpoint(pipeline, optimizer, config, 0, loss_history, "step_000000_base_reference")

    start = time.monotonic()
    last_log = start
    completed_step = 0
    stop_reason = "max_steps"
    for step in range(1, config.max_steps + 1):
        elapsed_hours = (time.monotonic() - start) / 3600.0
        if elapsed_hours >= config.max_train_hours:
            completed_step = step - 1
            stop_reason = "time_limit"
            break

        batch = make_batch(dataset, next(batch_iter))
        optimizer.zero_grad(set_to_none=True)
        loss = training_step(pipeline, batch["latents"], prompt_embeds, prompt_attention_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        completed_step = step
        loss_value = float(loss.detach().cpu())
        loss_history.append({"step": step, "loss": loss_value, "elapsed_hours": elapsed_hours})

        now = time.monotonic()
        if now - last_log > 30 or step == 1:
            sec_per_step = (now - start) / max(step, 1)
            print(json.dumps({"step": step, "loss": loss_value, "sec_per_step": sec_per_step, "elapsed_hours": elapsed_hours}))
            last_log = now

        if step in config.save_steps:
            ckpt = save_checkpoint(pipeline, optimizer, config, step, loss_history, f"step_{step:06d}")
            if step in config.sample_steps:
                sample_validation(pipeline, val_rows, config, step, ckpt / "validation_samples")
                checkpoint_volume.commit()

    if completed_step < config.max_steps:
        label = "final_before_timeout" if stop_reason == "time_limit" else f"final_step_{completed_step:06d}"
        ckpt = save_checkpoint(pipeline, optimizer, config, completed_step, loss_history, label)
        sample_validation(pipeline, val_rows, config, completed_step, ckpt / "validation_samples")
        checkpoint_volume.commit()

    summary = {
        "run_name": config.run_name,
        "completed_step": completed_step,
        "stop_reason": stop_reason,
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
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
) -> None:
    if not run_name:
        run_name = datetime.now(timezone.utc).strftime("ltx2b_waymo24_visual_lora_r16_%Y%m%d_%H%M%S")
    save_steps = [step for step in CHECKPOINT_STEPS if step <= max_steps]
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
        lora_rank=LORA_RANK,
        lora_alpha=LORA_RANK,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        train_manifest="manifests/train_windows_24fps_121f.csv",
        val_manifest="manifests/val_windows_24fps_121f.csv",
        num_val_samples=num_val_samples,
    )
    result = train.remote(asdict(config), train_limit=train_limit, val_limit=val_limit)
    print(json.dumps(result, indent=2, sort_keys=True))
