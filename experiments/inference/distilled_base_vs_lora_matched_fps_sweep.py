from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx2b-distilled-base-vs-lora-matched-fps")
MODELS_VOLUME_NAME = "models"
CHECKPOINT_VOLUME_NAME = "ltx2b-distilled098-waymo24fps-visual-lora-r16-checkpoints"
ARTIFACTS_VOLUME_NAME = "ltx2b-distilled098-base-vs-lora-matched-fps-sweep"
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
CHECKPOINT_ROOT = Path("/checkpoints")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

BASE_CKPT = "ltxv-2b-0.9.8-distilled.safetensors"
LORA_RUN_NAME = "ltx2b_distilled098_waymo24_visual_lora_r16_seed231_subset512_lr2e5_steps500"
LORA_STEP = "step_000500"

WIDTH = 512
HEIGHT = 512
OUTPUT_FPS = 24
SEED = 231

SOURCE_SCENE_TOKENS = [
    "0081c4821701",
    "00c73f2a3515",
    "00f755ca8954",
    "01d758be9706",
    "023de465f73a",
]
RUNS_ROOT = PurePosixPath("distilled098_base_vs_lora_matched_fps_go_forward_runs")

FPS_CONTEXTS: dict[int, list[int]] = {
    10: [33, 49, 81, 105],
    24: [81, 121, 193, 257],
    30: [105, 153, 241, 321],
}
FUTURE_FRAMES_BY_FPS: dict[int, int] = {
    10: 40,
    24: 96,
    30: 120,
}
DURATION_BUCKETS = ["about_3p4s", "about_5p0s", "about_8p0s", "about_10p7s"]

DEFAULT_BASE_PROMPT = (
    "Forward-facing autonomous driving dashcam video from a real Waymo-style car-mounted front camera. "
    "Preserve the observed scene identity, camera viewpoint, road layout, lane geometry, traffic lights, "
    "nearby vehicles, sidewalks, buildings, weather, and lighting."
)
DEFAULT_NEGATIVE_PROMPT = (
    "wrong action, braking, turning left, turning right, repeated input, scene restart, camera cut, "
    "new location, blurry, jittery, distorted, impossible vehicle motion, teleporting cars, duplicated cars"
)


@dataclass(frozen=True)
class SourceSpec:
    scene_token: str
    fps: int
    local_path: str
    source_filename: str
    source_relpath: str


@dataclass(frozen=True)
class InferenceJob:
    scene_token: str
    duration_bucket: str
    fps: int
    context_num_frames: int
    future_num_frames: int
    target_num_frames: int
    source_filename: str
    source_relpath: str
    clip_id: str


app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
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


def ltx_compatible_frame_count(frame_count: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % 8 == 0


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", "p")


def safe_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_minterpolate_24fps", "").replace("_minterpolate_30fps", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def build_go_forward_prompt(
    *,
    duration_bucket: str,
    fps: int,
    context_num_frames: int,
    target_num_frames: int,
    base_prompt: str,
) -> str:
    future_frames = target_num_frames - context_num_frames
    return (
        "ACTION TO PERFORM: GO FORWARD. The ego vehicle must continue straight after the observed context. "
        "Do not brake as the main action. Do not turn left. Do not turn right. "
        f"Matched-duration FPS bucket: {duration_bucket}. "
        f"Observed context: {context_num_frames} frames at {fps} FPS "
        f"({seconds_for_frames(context_num_frames, fps):.2f} seconds). "
        f"Generate future continuation: {future_frames} frames at {fps} FPS "
        f"({future_frames / fps:.2f} seconds). "
        "Do not restart the scene. Do not copy the observed clip. Do not switch locations. "
        f"{base_prompt} Maintain stable ego motion, consistent road geometry, and stable object identities."
    )


def find_single_source(local_dir: Path, pattern: str, scene_token: str) -> Path:
    matches = sorted(path for path in local_dir.glob(pattern) if scene_token in path.name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source in {local_dir} matching {pattern} and {scene_token}, found {len(matches)}."
        )
    return matches[0]


def collect_sources(scene_tokens: list[str]) -> dict[str, dict[int, Path]]:
    root = Path.cwd()
    return {
        scene_token: {
            10: find_single_source(
                root / "data" / "inference_input_clips",
                "waymo_full20s_*_10fps_20p0s_context_*.mp4",
                scene_token,
            ),
            24: find_single_source(
                root / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s",
                "*_minterpolate_24fps.mp4",
                scene_token,
            ),
            30: find_single_source(
                root / "data" / "interpolated_30fps_waymo_full20s",
                "*_minterpolate_30fps.mp4",
                scene_token,
            ),
        }
        for scene_token in scene_tokens
    }


def upload_sources(
    *,
    sources: dict[str, dict[int, Path]],
    run_root_relpath: PurePosixPath,
) -> dict[str, dict[int, SourceSpec]]:
    uploaded: dict[str, dict[int, SourceSpec]] = {}
    with artifacts_volume.batch_upload(force=True) as batch:
        for scene_token, fps_sources in sorted(sources.items()):
            uploaded[scene_token] = {}
            for fps, local_path in sorted(fps_sources.items()):
                remote_path = run_root_relpath / "source_scene_inputs" / scene_token / f"{fps}fps" / local_path.name
                batch.put_file(local_path, remote_path.as_posix())
                uploaded[scene_token][fps] = SourceSpec(
                    scene_token=scene_token,
                    fps=fps,
                    local_path=str(local_path),
                    source_filename=local_path.name,
                    source_relpath=remote_path.as_posix(),
                )
    return uploaded


def build_jobs(source_specs: dict[str, dict[int, SourceSpec]], scene_tokens: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scene_token in scene_tokens:
        for bucket_idx, duration_bucket in enumerate(DURATION_BUCKETS):
            for fps in [10, 24, 30]:
                context_num_frames = FPS_CONTEXTS[fps][bucket_idx]
                future_num_frames = FUTURE_FRAMES_BY_FPS[fps]
                target_num_frames = context_num_frames + future_num_frames
                if not ltx_compatible_frame_count(context_num_frames):
                    raise ValueError(f"{fps} FPS context {context_num_frames} is not 8N+1.")
                if not ltx_compatible_frame_count(target_num_frames):
                    raise ValueError(f"{fps} FPS target {target_num_frames} is not 8N+1.")

                source = source_specs[scene_token][fps]
                clip_id = (
                    f"scene_{scene_token}_{duration_bucket}_{context_num_frames}ctx_"
                    f"{future_num_frames}future_{fps}fps_{safe_stem(Path(source.source_filename))}"
                )
                jobs.append(
                    asdict(
                        InferenceJob(
                            scene_token=scene_token,
                            duration_bucket=duration_bucket,
                            fps=fps,
                            context_num_frames=context_num_frames,
                            future_num_frames=future_num_frames,
                            target_num_frames=target_num_frames,
                            source_filename=source.source_filename,
                            source_relpath=source.source_relpath,
                            clip_id=clip_id,
                        )
                    )
                )
    return jobs


def ensure_base_checkpoint() -> Path:
    src = MODELS_ROOT / "ltx" / BASE_CKPT
    dst = REPO / BASE_CKPT
    if not src.exists():
        raise FileNotFoundError(f"Missing base checkpoint in Modal volume: {src}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return dst


def lora_adapter_dir() -> Path:
    adapter_dir = CHECKPOINT_ROOT / LORA_RUN_NAME / LORA_STEP / "lora_adapter"
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_dir}")
    return adapter_dir


def write_video(path: Path, video_tensor, fps: int) -> None:
    import imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().float().numpy()
    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in video_np:
            writer.append_data(frame)


def setup_pipeline(*, use_lora: bool):
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from ltx_video.inference import create_ltx_video_pipeline

    base_ckpt = ensure_base_checkpoint()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=str(base_ckpt),
        precision="bfloat16",
        text_encoder_model_name_or_path="PixArt-alpha/PixArt-XL-2-1024-MS",
        sampler="from_checkpoint",
        device="cuda",
        enhance_prompt=False,
    )
    if use_lora:
        from peft import PeftModel

        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, str(lora_adapter_dir()))
        pipeline.transformer.config = pipeline.transformer.base_model.model.config
    pipeline.transformer.eval()
    pipeline.transformer.to("cuda", dtype=torch.bfloat16)
    return pipeline


def generate_one(
    pipeline,
    job: dict[str, Any],
    *,
    run_root_relpath: str,
    model_mode: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO))
    import torch
    from ltx_video.inference import load_media_file
    from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem

    source_path = ARTIFACTS_ROOT / job["source_relpath"]
    if not source_path.exists():
        raise FileNotFoundError(f"Missing uploaded source clip: {source_path}")

    fps = int(job["fps"])
    context_num_frames = int(job["context_num_frames"])
    target_num_frames = int(job["target_num_frames"])
    future_num_frames = int(job["future_num_frames"])
    duration_bucket = job["duration_bucket"]
    clip_id = job["clip_id"]

    output_dir = (
        ARTIFACTS_ROOT
        / run_root_relpath
        / "generated_outputs"
        / model_mode
        / job["scene_token"]
        / duration_bucket
        / f"{fps}fps"
        / clip_id
    )
    output_path = output_dir / (
        f"{model_mode}_{clip_id}_target{target_num_frames}f_{fps}fps_seed{seed}.mp4"
    )
    result_path = output_dir / "result.json"
    if output_path.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    media = load_media_file(
        media_path=str(source_path),
        height=HEIGHT,
        width=WIDTH,
        max_frames=context_num_frames,
        padding=(0, 0, 0, 0),
        just_crop=True,
    )
    conditioning_items = [ConditioningItem(media, 0, 1.0)]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    action_prompt = build_go_forward_prompt(
        duration_bucket=duration_bucket,
        fps=fps,
        context_num_frames=context_num_frames,
        target_num_frames=target_num_frames,
        base_prompt=prompt,
    )

    with torch.no_grad():
        video = pipeline(
            prompt=action_prompt,
            negative_prompt=negative_prompt,
            height=HEIGHT,
            width=WIDTH,
            num_frames=target_num_frames,
            frame_rate=fps,
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
    write_video(output_path, video, fps=fps)

    record = {
        "scene_token": job["scene_token"],
        "duration_bucket": duration_bucket,
        "fps": fps,
        "context_num_frames": context_num_frames,
        "future_num_frames": future_num_frames,
        "target_num_frames": target_num_frames,
        "context_seconds": seconds_for_frames(context_num_frames, fps),
        "future_seconds": future_num_frames / fps,
        "target_seconds": seconds_for_frames(target_num_frames, fps),
        "source_filename": job["source_filename"],
        "source_relpath": job["source_relpath"],
        "clip_id": clip_id,
        "model_mode": model_mode,
        "base_checkpoint": BASE_CKPT,
        "lora_run_name": LORA_RUN_NAME if model_mode.startswith("distilled_lora") else "",
        "lora_step": LORA_STEP if model_mode.startswith("distilled_lora") else "",
        "seed": seed,
        "prompt": action_prompt,
        "negative_prompt": negative_prompt,
        "generated_video_relpath": output_path.relative_to(ARTIFACTS_ROOT).as_posix(),
        "output_dir_relpath": output_dir.relative_to(ARTIFACTS_ROOT).as_posix(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return record


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, text=True, capture_output=True)


def comparison_label(model_mode: str, fps: int, context_frames: int, future_frames: int) -> str:
    model = "BASE DISTILLED" if model_mode == "base_distilled_no_lora" else "DISTILLED LORA 500"
    return f"{model} | {fps} FPS | {context_frames}ctx + {future_frames}fut"


def make_comparison_video(
    grouped_records: dict[str, dict[int, dict[str, Any]]],
    *,
    run_root_relpath: str,
    scene_token: str,
    duration_bucket: str,
) -> dict[str, Any]:
    ordered: list[tuple[str, int, dict[str, Any]]] = []
    for model_mode in ["base_distilled_no_lora", "distilled_lora_step000500"]:
        for fps in [10, 24, 30]:
            ordered.append((model_mode, fps, grouped_records[model_mode][fps]))

    out_dir = ARTIFACTS_ROOT / run_root_relpath / "side_by_side_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = (
        f"scene_{scene_token}_{duration_bucket}_BASE-DISTILLED-vs-DISTILLED-LORA-step000500_"
        "TOP-base_BOTTOM-lora_LEFT-10fps_MIDDLE-24fps_RIGHT-30fps_go_forward.mp4"
    )
    out_path = out_dir / out_name
    if not out_path.exists():
        cmd = ["ffmpeg", "-y"]
        labels = []
        for model_mode, fps, record in ordered:
            cmd.extend(["-i", str(ARTIFACTS_ROOT / record["generated_video_relpath"])])
            labels.append(
                comparison_label(
                    model_mode,
                    fps,
                    int(record["context_num_frames"]),
                    int(record["future_num_frames"]),
                )
            )

        filter_parts = []
        for idx, label in enumerate(labels):
            safe_label = label.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            filter_parts.append(
                f"[{idx}:v]fps={OUTPUT_FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},setsar=1,"
                "drawbox=x=0:y=0:w=iw:h=44:color=black@0.62:t=fill,"
                f"drawtext=fontcolor=white:fontsize=19:text='{safe_label}':x=12:y=13[v{idx}]"
            )
        filter_complex = (
            ";".join(filter_parts)
            + ";[v0][v1][v2][v3][v4][v5]"
            + f"xstack=inputs=6:layout=0_0|{WIDTH}_0|{WIDTH * 2}_0|0_{HEIGHT}|{WIDTH}_{HEIGHT}|{WIDTH * 2}_{HEIGHT}:fill=black[v]"
        )
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-an",
                "-r",
                str(OUTPUT_FPS),
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )
        try:
            run_checked(cmd)
        except subprocess.CalledProcessError:
            # Fallback for ffmpeg builds without drawtext support.
            cmd = ["ffmpeg", "-y"]
            for _, _, record in ordered:
                cmd.extend(["-i", str(ARTIFACTS_ROOT / record["generated_video_relpath"])])
            filter_parts = [
                f"[{idx}:v]fps={OUTPUT_FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={WIDTH}:{HEIGHT},setsar=1[v{idx}]"
                for idx in range(6)
            ]
            filter_complex = (
                ";".join(filter_parts)
                + ";[v0][v1][v2][v3][v4][v5]"
                + f"xstack=inputs=6:layout=0_0|{WIDTH}_0|{WIDTH * 2}_0|0_{HEIGHT}|{WIDTH}_{HEIGHT}|{WIDTH * 2}_{HEIGHT}:fill=black[v]"
            )
            cmd.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v]",
                    "-an",
                    "-r",
                    str(OUTPUT_FPS),
                    "-shortest",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_path),
                ]
            )
            run_checked(cmd)

    return {
        "scene_token": scene_token,
        "duration_bucket": duration_bucket,
        "file": out_path.name,
        "comparison_video_relpath": out_path.relative_to(ARTIFACTS_ROOT).as_posix(),
        "layout": "TOP=base distilled no LoRA, BOTTOM=distilled LoRA step 500; LEFT=10 FPS, MIDDLE=24 FPS, RIGHT=30 FPS",
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
    volumes={
        str(MODELS_ROOT): models_volume,
        str(CHECKPOINT_ROOT): checkpoint_volume,
        str(ARTIFACTS_ROOT): artifacts_volume,
    },
)
def generate_and_compare(
    jobs: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    import gc
    import torch

    models_volume.reload()
    checkpoint_volume.reload()
    artifacts_volume.reload()

    all_records: list[dict[str, Any]] = []
    for model_mode, use_lora in [
        ("base_distilled_no_lora", False),
        ("distilled_lora_step000500", True),
    ]:
        pipeline = setup_pipeline(use_lora=use_lora)
        for job in jobs:
            record = generate_one(
                pipeline,
                job,
                run_root_relpath=run_root_relpath,
                model_mode=model_mode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )
            all_records.append(record)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    records_by_key: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for record in all_records:
        records_by_key[
            (record["scene_token"], record["duration_bucket"], record["model_mode"], int(record["fps"]))
        ] = record

    comparisons: list[dict[str, Any]] = []
    scene_tokens = sorted({record["scene_token"] for record in all_records})
    for scene_token in scene_tokens:
        for duration_bucket in DURATION_BUCKETS:
            grouped = {
                model_mode: {
                    fps: records_by_key[(scene_token, duration_bucket, model_mode, fps)]
                    for fps in [10, 24, 30]
                }
                for model_mode in ["base_distilled_no_lora", "distilled_lora_step000500"]
            }
            comparisons.append(
                make_comparison_video(
                    grouped,
                    run_root_relpath=run_root_relpath,
                    scene_token=scene_token,
                    duration_bucket=duration_bucket,
                )
            )
            artifacts_volume.commit()

    manifest = {
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "run_root_relpath": run_root_relpath,
        "base_checkpoint": BASE_CKPT,
        "lora_run_name": LORA_RUN_NAME,
        "lora_step": LORA_STEP,
        "seed": seed,
        "scene_tokens": scene_tokens,
        "duration_buckets": DURATION_BUCKETS,
        "fps_contexts": FPS_CONTEXTS,
        "future_frames_by_fps": FUTURE_FRAMES_BY_FPS,
        "future_seconds_by_fps": {fps: frames / fps for fps, frames in FUTURE_FRAMES_BY_FPS.items()},
        "num_generated_videos": len(all_records),
        "num_comparison_videos": len(comparisons),
        "comparison_layout": "2x3 grid: top row base distilled, bottom row distilled LoRA; columns 10/24/30 FPS",
        "generated_records": all_records,
        "comparisons": comparisons,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = ARTIFACTS_ROOT / run_root_relpath / "manifest_base_vs_lora_matched_fps_comparisons.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    scene_tokens_csv: str = ",".join(SOURCE_SCENE_TOKENS),
    seed: int = SEED,
    run_label: str = "all5_scenes_base_distilled_vs_lora_step000500_matched_10_24_30fps_4s_future_go_forward",
    prompt: str = DEFAULT_BASE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> None:
    scene_tokens = [token.strip() for token in scene_tokens_csv.split(",") if token.strip()]
    if not scene_tokens:
        raise ValueError("scene_tokens_csv must contain at least one scene token.")

    sources = collect_sources(scene_tokens)
    run_root_relpath = RUNS_ROOT / run_label
    uploaded_sources = upload_sources(sources=sources, run_root_relpath=run_root_relpath)
    jobs = build_jobs(
        {
            scene_token: {fps: spec for fps, spec in fps_specs.items()}
            for scene_token, fps_specs in uploaded_sources.items()
        },
        scene_tokens,
    )
    manifest = generate_and_compare.remote(
        jobs,
        run_root_relpath=run_root_relpath.as_posix(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
