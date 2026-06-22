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


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-waymo-30fps-interpolated-go-forward-2b")
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME",
    "waymo-ltx2b-30fps-go-forward-sweep",
)
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"

DEFAULT_BASE_PROMPT = (
    "Forward-facing autonomous driving dashcam video from a real Waymo-style car-mounted camera. "
    "Preserve the same camera viewpoint, road layout, lane geometry, nearby vehicles, traffic lights, "
    "sidewalks, buildings, weather, lighting, and scene identity from the observed 30 FPS context clip."
)
DEFAULT_NEGATIVE_PROMPT = (
    "wrong action, braking, turning left, turning right, unchanged future, repeated input, scene restart, "
    "camera cut, new location, worst quality, inconsistent motion, blurry, jittery, distorted"
)


@dataclass(frozen=True)
class LocalWaymoClipSpec:
    clip_id: str
    source_filename: str
    source_relpath: str
    num_frames: int
    fps: int
    width: int
    height: int
    source_mode: str = "local_waymo_30fps_interpolated_context"


app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
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


def ltx_compatible_frame_count(frame_count: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % 8 == 0


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def format_seconds_for_id(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", "p")


def context_tag(num_frames: int, fps: int) -> str:
    return f"{num_frames}f_{fps}fps_{format_seconds_for_id(seconds_for_frames(num_frames, fps))}s_context"


def safe_stem(path: Path) -> str:
    stem = path.stem.replace("_minterpolate_30fps", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def build_go_forward_prompt(
    *,
    observed_frames: int,
    target_num_frames: int,
    fps: int,
    base_prompt: str,
) -> str:
    future_frames = target_num_frames - observed_frames
    if future_frames <= 0:
        raise ValueError("target_num_frames must exceed observed_frames.")
    observed_seconds = seconds_for_frames(observed_frames, fps)
    future_seconds = future_frames / fps
    return (
        "ACTION TO PERFORM: GO FORWARD. GO FORWARD is the required future behavior. "
        "The ego vehicle must continue straight immediately after the observed context clip. "
        "Keep the vehicle on the current forward lane or natural forward road trajectory. "
        "Do not brake as the main action. Do not turn left. Do not turn right. "
        f"The observed input is a {observed_seconds:.1f}-second, {observed_frames}-frame, {fps}-FPS "
        "Waymo-derived 30 FPS interpolated driving context clip. Use that full context as fixed history and "
        f"generate the next {future_seconds:.1f} seconds ({future_frames} future frames) as a continuation. "
        "Do not restart the scene. Do not copy the observed clip again. Do not switch locations. "
        "The generated future must begin exactly where the observed clip ends and must follow GO FORWARD. "
        f"{base_prompt} Maintain realistic ego motion, consistent 3D road geometry, stable object identities, "
        "no camera cuts, no sudden viewpoint changes, and no impossible vehicle motion."
    )


def upload_local_sources(
    *,
    local_input_dir: Path,
    run_root_relpath: PurePosixPath,
) -> list[dict[str, str]]:
    input_paths = sorted(local_input_dir.glob("*_minterpolate_30fps.mp4"))
    if len(input_paths) != 5:
        raise ValueError(f"Expected exactly 5 interpolated 30 FPS clips under {local_input_dir}, found {len(input_paths)}.")

    uploaded: list[dict[str, str]] = []
    upload_root = run_root_relpath / "source_30fps_interpolated_full20s_mp4"
    with artifacts_volume.batch_upload(force=True) as batch:
        for path in input_paths:
            remote_path = upload_root / path.name
            batch.put_file(path, remote_path.as_posix())
            uploaded.append(
                {
                    "source_filename": path.name,
                    "source_relpath": remote_path.as_posix(),
                }
            )
    return uploaded


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(ARTIFACTS_ROOT): artifacts_volume},
)
def prepare_context_clips(
    uploaded_sources: list[dict[str, str]],
    *,
    run_root_relpath: str,
    context_num_frames: int,
    frame_rate: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    artifacts_volume.reload()
    run_root = ARTIFACTS_ROOT / run_root_relpath
    prepared: list[dict[str, Any]] = []
    tag = context_tag(context_num_frames, frame_rate)

    for source in uploaded_sources:
        source_path = ARTIFACTS_ROOT / source["source_relpath"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing uploaded source clip: {source_path}")

        source_stem = safe_stem(Path(source["source_filename"]))
        clip_id = f"local30fps_{tag}_{source_stem}"
        output_dir = run_root / f"inputs_{tag}" / clip_id
        output_dir.mkdir(parents=True, exist_ok=True)
        context_video_path = output_dir / f"{clip_id}_observed_{tag}_input.mp4"
        manifest_path = output_dir / "observed_context_clip_spec.json"

        if not context_video_path.exists() or not manifest_path.exists():
            run_checked(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source_path),
                    "-an",
                    "-vf",
                    f"fps={frame_rate},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
                    "-frames:v",
                    str(context_num_frames),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(context_video_path),
                ]
            )
            clip = LocalWaymoClipSpec(
                clip_id=clip_id,
                source_filename=source["source_filename"],
                source_relpath=source["source_relpath"],
                num_frames=context_num_frames,
                fps=frame_rate,
                width=width,
                height=height,
            )
            manifest = {
                **asdict(clip),
                "context_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            artifacts_volume.commit()
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            clip = LocalWaymoClipSpec(
                clip_id=manifest["clip_id"],
                source_filename=manifest["source_filename"],
                source_relpath=manifest["source_relpath"],
                num_frames=int(manifest["num_frames"]),
                fps=int(manifest["fps"]),
                width=int(manifest["width"]),
                height=int(manifest["height"]),
            )

        prepared.append(
            {
                "clip": asdict(clip),
                "input_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                "input_manifest_relpath": manifest_path.relative_to(ARTIFACTS_ROOT).as_posix(),
            }
        )

    return prepared


def ensure_checkpoints() -> None:
    for filename in [CKPT_2B, UPSCALER]:
        src = MODELS_ROOT / "ltx" / filename
        dst = REPO / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint in Modal volume: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def build_inference_cmd(
    *,
    prompt: str,
    negative_prompt: str,
    input_video_path: Path,
    output_dir: Path,
    target_num_frames: int,
    frame_rate: int,
    seed: int,
    conditioning_start_frame: int,
    pipeline_config_path: Path,
    width: int,
    height: int,
) -> list[str]:
    return [
        "python",
        "inference.py",
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_frames",
        str(target_num_frames),
        "--frame_rate",
        str(frame_rate),
        "--seed",
        str(seed),
        "--pipeline_config",
        str(pipeline_config_path),
        "--output_path",
        str(output_dir),
        "--conditioning_media_paths",
        str(input_video_path),
        "--conditioning_start_frames",
        str(conditioning_start_frame),
    ]


def run_ltx_for_local_waymo_impl(
    job: dict[str, Any],
    *,
    run_root_relpath: str,
    negative_prompt: str,
    target_num_frames: int,
    frame_rate: int,
    seed: int,
    conditioning_start_frame: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    clip = LocalWaymoClipSpec(**job["clip"])
    input_video_path = ARTIFACTS_ROOT / job["input_video_relpath"]
    if not input_video_path.exists():
        raise FileNotFoundError(f"Missing staged input video: {input_video_path}")

    ensure_checkpoints()

    tag = context_tag(clip.num_frames, frame_rate)
    output_dir = ARTIFACTS_ROOT / run_root_relpath / "generated_future_outputs" / "required_action_go_forward" / clip.clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / (
        f"{clip.clip_id}_ltx2b_local30fps_waymo_{tag}_REQUIRED_ACTION_go_forward_"
        f"future_continuation_{target_num_frames}f_{frame_rate}fps.mp4"
    )
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    prompt = build_go_forward_prompt(
        observed_frames=clip.num_frames,
        target_num_frames=target_num_frames,
        fps=frame_rate,
        base_prompt=job["base_prompt"],
    )
    cmd = build_inference_cmd(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_video_path=input_video_path,
        output_dir=output_dir,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
        seed=seed,
        conditioning_start_frame=conditioning_start_frame,
        pipeline_config_path=REPO / "configs" / "ltxv-2b-0.9.8-distilled.yaml",
        width=width,
        height=height,
    )
    subprocess.run(cmd, cwd=str(REPO), check=True)

    generated_candidates = sorted(
        path
        for path in output_dir.rglob("*.mp4")
        if path.name != Path(job["input_video_relpath"]).name and path.name != canonical_output.name
    )
    if not generated_candidates:
        raise RuntimeError(f"No generated mp4 found under {output_dir}")

    shutil.copy2(generated_candidates[0], canonical_output)

    record = {
        "clip": asdict(clip),
        "action_name": "go_forward",
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_num_frames": clip.num_frames,
        "context_seconds": seconds_for_frames(clip.num_frames, frame_rate),
        "target_num_frames": target_num_frames,
        "future_num_frames": target_num_frames - clip.num_frames,
        "future_seconds": (target_num_frames - clip.num_frames) / frame_rate,
        "frame_rate": frame_rate,
        "input_video_relpath": job["input_video_relpath"],
        "input_manifest_relpath": job["input_manifest_relpath"],
        "generated_video_relpath": canonical_output.relative_to(ARTIFACTS_ROOT).as_posix(),
        "raw_generated_video_relpath": generated_candidates[0].relative_to(ARTIFACTS_ROOT).as_posix(),
        "output_dir_relpath": output_dir.relative_to(ARTIFACTS_ROOT).as_posix(),
        "model_variant": "2b",
        "model_label": "ltx2b",
        "model_checkpoint": CKPT_2B,
        "pipeline_config": "ltxv-2b-0.9.8-distilled.yaml",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return record


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={
        str(MODELS_ROOT): models_volume,
        str(ARTIFACTS_ROOT): artifacts_volume,
    },
)
def run_ltx_local_waymo_batch(
    jobs: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    negative_prompt: str,
    target_num_frames: int,
    frame_rate: int,
    seed_base: int,
    conditioning_start_frame: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    artifacts_volume.reload()
    results = []
    for offset, job in enumerate(jobs):
        results.append(
            run_ltx_for_local_waymo_impl(
                job,
                run_root_relpath=run_root_relpath,
                negative_prompt=negative_prompt,
                target_num_frames=target_num_frames,
                frame_rate=frame_rate,
                seed=seed_base + offset,
                conditioning_start_frame=conditioning_start_frame,
                width=width,
                height=height,
            )
        )
    return results


def upload_run_summary(summary: dict[str, Any], *, run_root_relpath: PurePosixPath) -> str:
    tmp_dir = Path("/tmp/local_waymo_30fps_summary")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / "run_summary.json"
    local_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    remote_path = run_root_relpath / "run_summary.json"
    with artifacts_volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path.as_posix())
    return remote_path.as_posix()


@app.local_entrypoint()
def main(
    local_input_dir: str = "data/interpolated_30fps_waymo_full20s",
    context_num_frames: int = 601,
    target_num_frames: int = 753,
    frame_rate: int = 30,
    width: int = 512,
    height: int = 512,
    seed_base: int = 7000,
    prompt: str = DEFAULT_BASE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    run_label: str = "",
    conditioning_start_frame: int = 0,
) -> None:
    if not ltx_compatible_frame_count(context_num_frames):
        raise ValueError("context_num_frames must be of the form 8N+1 for LTX-compatible conditioning.")
    if not ltx_compatible_frame_count(target_num_frames):
        raise ValueError("target_num_frames must be of the form 8N+1 for LTX video outputs.")
    if target_num_frames <= context_num_frames:
        raise ValueError("target_num_frames must exceed context_num_frames so the model generates a future continuation.")
    if conditioning_start_frame % 8 != 0:
        raise ValueError("conditioning_start_frame must be a multiple of 8 for LTX conditioning.")
    if frame_rate != 30:
        raise ValueError("This script is for the 30 FPS interpolated Waymo context experiment.")

    local_dir = Path(local_input_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = context_tag(context_num_frames, frame_rate)
    run_name = run_label or (
        f"local_waymo_30fps_5clips_ltx2b_{tag}_to_{target_num_frames}f_go_forward_{timestamp}"
    )
    run_root_relpath = PurePosixPath("waymo_30fps_interpolated_go_forward_runs") / run_name

    uploaded_sources = upload_local_sources(local_input_dir=local_dir, run_root_relpath=run_root_relpath)
    clip_jobs = prepare_context_clips.remote(
        uploaded_sources,
        run_root_relpath=run_root_relpath.as_posix(),
        context_num_frames=context_num_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
    )
    jobs = [{**job, "base_prompt": prompt} for job in clip_jobs]
    results = run_ltx_local_waymo_batch.remote(
        jobs,
        run_root_relpath=run_root_relpath.as_posix(),
        negative_prompt=negative_prompt,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
        seed_base=seed_base,
        conditioning_start_frame=conditioning_start_frame,
        width=width,
        height=height,
    )

    summary = {
        "app_name": APP_NAME,
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "gpu_parallelism": 1,
        "num_source_clips": len(clip_jobs),
        "num_generated_variants": len(jobs),
        "action": "go_forward",
        "context_num_frames": context_num_frames,
        "context_seconds": seconds_for_frames(context_num_frames, frame_rate),
        "target_num_frames": target_num_frames,
        "future_num_frames": target_num_frames - context_num_frames,
        "future_seconds": (target_num_frames - context_num_frames) / frame_rate,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "uploaded_sources": uploaded_sources,
        "run_root_relpath": run_root_relpath.as_posix(),
        "summary_relpath": (run_root_relpath / "run_summary.json").as_posix(),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    upload_run_summary(summary, run_root_relpath=run_root_relpath)
    print(json.dumps(summary, indent=2, sort_keys=True))
