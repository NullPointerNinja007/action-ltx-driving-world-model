from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx2b-dev-waymo24-lora-local-minterpolate-infer")
MODELS_VOLUME_NAME = "models"
DATA_VOLUME_NAME = os.environ.get(
    "LTX_DATA_VOLUME_NAME",
    "waymo-e2e-24fps-121f-visual-continuation-data",
)
CHECKPOINT_VOLUME_NAME = os.environ.get(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dev-waymo24fps-visual-lora-r16-checkpoints",
)
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dev-waymo24-lora-local-minterpolate-inference",
)

MODELS_ROOT = Path("/models")
DATA_ROOT = Path("/data")
CHECKPOINT_ROOT = Path("/checkpoints")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

BASE_CKPT = os.environ.get("LTX_BASE_CKPT", "ltxv-2b-0.9.6-dev-04-25.safetensors")
LORA_RUN_NAME = os.environ.get("LTX_LORA_RUN_NAME", "ltx2b_dev_waymo24_visual_lora_r16_seed231_main")
DEFAULT_LORA_STEP = os.environ.get("LTX_DEFAULT_LORA_STEP", "step_003000")
BASE_ONLY_STEP = "base"

FPS = 24
WIDTH = 512
HEIGHT = 512
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = CONTEXT_FRAMES + FUTURE_FRAMES
SEED = 231

LOCAL_SOURCE_DIR = Path("data/inference_input_clips/interpolated_24fps_waymo_full20s")
LOCAL_OUTPUT_DIR = Path(
    os.environ.get("LTX_LOCAL_OUTPUT_DIR", "data/finetuned_ltx2b_dev_lora_24fps_minterpolate_seed231")
)
RUNS_ROOT = PurePosixPath(
    os.environ.get("LTX_RUNS_ROOT", "finetuned_ltx2b_dev_lora_24fps_minterpolate_seed231_runs")
)
IMAGE_COND_NOISE_SCALE = float(os.environ.get("LTX_IMAGE_COND_NOISE_SCALE", "0.0"))

DEFAULT_PROMPT = (
    "Forward-facing autonomous driving video from a real Waymo-style car-mounted front camera. "
    "Use the observed 49-frame 24 FPS context as fixed history. Generate only the natural future "
    "continuation after the final observed frame. Preserve the same camera viewpoint, road layout, "
    "lane geometry, nearby vehicles, traffic lights, sidewalks, buildings, lighting, and weather. "
    "Do not restart the scene, do not copy the observed clip again, do not jump to a new location, "
    "and do not introduce a camera cut. Continue with physically plausible ego-vehicle motion and "
    "stable object identities."
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


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.replace("_minterpolate_24fps", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def use_lora_adapter(lora_step: str) -> bool:
    return lora_step.lower() not in {"", BASE_ONLY_STEP, "none", "no_lora", "base_only"}


def checkpoint_label(lora_step: str, base_label: str = "base_dev_no_lora") -> str:
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
    seed: int,
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
    adapter_dir = CHECKPOINT_ROOT / lora_run_name / lora_step / "lora_adapter"
    using_lora = use_lora_adapter(lora_step)
    if using_lora and not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_dir}")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(base_ckpt),
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        sampler="from_checkpoint",
        device="cuda",
        enhance_prompt=False,
    )
    if using_lora:
        from peft import PeftModel

        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, str(adapter_dir))
        pipeline.transformer.config = pipeline.transformer.base_model.model.config
    pipeline.transformer.eval()
    pipeline.transformer.to("cuda", dtype=torch.bfloat16)

    out_root = ARTIFACTS_ROOT / run_root_relpath / "generated_lora_checkpoints"
    results: list[dict[str, Any]] = []
    for idx, source_payload in enumerate(sources_payload):
        source = SourceClip(**source_payload)
        source_path = source_clip_path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source clip: {source_path}")

        step_label = checkpoint_label(lora_step, base_label=base_label)
        model_label = "lora" if using_lora else "base"
        clip_id = f"scene_{source.scene_token}_49ctx_72future_{model_label}_{step_label}_seed{seed}"
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
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            video = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
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
        write_video(output_path, video, fps=FPS)

        record = {
            "scene_token": source.scene_token,
            "source_filename": source.source_filename,
            "source_relpath": source.source_relpath,
            "generated_video_relpath": output_path.relative_to(ARTIFACTS_ROOT).as_posix(),
            "output_dir_relpath": output_dir.relative_to(ARTIFACTS_ROOT).as_posix(),
            "base_checkpoint": base_ckpt_name,
            "lora_run_name": lora_run_name if using_lora else "",
            "lora_step": lora_step,
            "adapter_relpath": adapter_dir.relative_to(CHECKPOINT_ROOT).as_posix() if using_lora else "",
            "using_lora": using_lora,
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
        "using_lora": using_lora,
        "seed": seed,
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
    base_label: str = "base_dev_no_lora",
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    step_label = checkpoint_label(lora_step, base_label=base_label)
    prefix = "lora" if use_lora_adapter(lora_step) else "base"
    run_name = run_label or f"{prefix}_{step_label}_49ctx_72future_seed{seed}_{timestamp}"
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
        seed=seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
