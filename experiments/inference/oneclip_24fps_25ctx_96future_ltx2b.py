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


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-waymo-oneclip-24fps-25ctx-96future-2b")
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME",
    "waymo-ltx2b-oneclip-24fps-25ctx-96future",
)
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "A100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"

DEFAULT_SCENE_TOKEN = "0081c4821701"
RUNS_ROOT = PurePosixPath("oneclip_24fps_25ctx_96future_ltx2b_runs")

FPS = 24
CONTEXT_FRAMES = 25
FUTURE_FRAMES = 96
TARGET_FRAMES = CONTEXT_FRAMES + FUTURE_FRAMES

DEFAULT_PROMPT = (
    "Forward-facing autonomous driving video from a real Waymo-style car-mounted front camera. "
    "Use the observed 25-frame context as fixed history. Generate only the natural future continuation "
    "after the final observed frame. Preserve the same camera viewpoint, road layout, lane geometry, "
    "traffic lights, nearby vehicles, sidewalks, buildings, lighting, and weather. Do not restart the scene, "
    "do not copy the observed clip again, do not jump to a new location, and do not introduce a camera cut. "
    "Continue with physically plausible ego-vehicle motion and stable object identities."
)
DEFAULT_NEGATIVE_PROMPT = (
    "repeated input, scene restart, camera cut, new location, wrong viewpoint, rear camera, side camera, "
    "blurry, jittery, distorted, impossible vehicle motion, teleporting cars, duplicated cars"
)


@dataclass(frozen=True)
class SourceSpec:
    scene_token: str
    fps: int
    local_path: str
    source_filename: str
    source_relpath: str


@dataclass(frozen=True)
class ContextJob:
    scene_token: str
    fps: int
    context_num_frames: int
    future_num_frames: int
    target_num_frames: int
    source_filename: str
    source_relpath: str
    clip_id: str
    width: int
    height: int
    source_mode: str = "oneclip_24fps_25ctx_96future"


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


def safe_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_minterpolate_24fps", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def find_source(scene_token: str) -> Path:
    source_dir = Path.cwd() / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
    matches = sorted(path for path in source_dir.glob("*_minterpolate_24fps.mp4") if scene_token in path.name)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one 24 FPS source for {scene_token}, found {len(matches)} in {source_dir}.")
    return matches[0]


def upload_source(*, source_path: Path, scene_token: str, run_root_relpath: PurePosixPath) -> SourceSpec:
    remote_path = run_root_relpath / "source_24fps_full20s_input" / source_path.name
    with artifacts_volume.batch_upload(force=True) as batch:
        batch.put_file(source_path, remote_path.as_posix())
    return SourceSpec(
        scene_token=scene_token,
        fps=FPS,
        local_path=str(source_path),
        source_filename=source_path.name,
        source_relpath=remote_path.as_posix(),
    )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(ARTIFACTS_ROOT): artifacts_volume},
)
def prepare_context_clip(
    source_spec_payload: dict[str, Any],
    *,
    run_root_relpath: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    artifacts_volume.reload()
    source = SourceSpec(**source_spec_payload)
    source_path = ARTIFACTS_ROOT / source.source_relpath
    if not source_path.exists():
        raise FileNotFoundError(f"Missing uploaded source clip: {source_path}")

    if not ltx_compatible_frame_count(CONTEXT_FRAMES):
        raise ValueError(f"Invalid LTX context frame count: {CONTEXT_FRAMES}")
    if not ltx_compatible_frame_count(TARGET_FRAMES):
        raise ValueError(f"Invalid LTX target frame count: {TARGET_FRAMES}")

    run_root = ARTIFACTS_ROOT / run_root_relpath
    source_stem = safe_stem(Path(source.source_filename))
    clip_id = f"scene_{source.scene_token}_24fps_25ctx_96future_{source_stem}"
    output_dir = run_root / "observed_context_input_25frames" / clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    context_video_path = output_dir / f"{clip_id}_observed_25f_24fps_input.mp4"
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
                f"fps={FPS},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
                "-frames:v",
                str(CONTEXT_FRAMES),
                "-r",
                str(FPS),
                "-fps_mode",
                "cfr",
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
        clip = ContextJob(
            scene_token=source.scene_token,
            fps=FPS,
            context_num_frames=CONTEXT_FRAMES,
            future_num_frames=FUTURE_FRAMES,
            target_num_frames=TARGET_FRAMES,
            source_filename=source.source_filename,
            source_relpath=source.source_relpath,
            clip_id=clip_id,
            width=width,
            height=height,
        )
        manifest = {
            **asdict(clip),
            "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
            "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
            "target_seconds_frames_minus_one": seconds_for_frames(TARGET_FRAMES, FPS),
            "context_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        artifacts_volume.commit()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "clip": manifest,
        "input_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
        "input_manifest_relpath": manifest_path.relative_to(ARTIFACTS_ROOT).as_posix(),
    }


def ensure_checkpoints() -> None:
    for filename in [CKPT_2B, UPSCALER]:
        src = MODELS_ROOT / "ltx" / filename
        dst = REPO / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint in Modal volume: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


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
def run_ltx(job: dict[str, Any], *, run_root_relpath: str, prompt: str, negative_prompt: str, seed: int) -> dict[str, Any]:
    artifacts_volume.reload()
    ensure_checkpoints()

    clip = job["clip"]
    input_video_path = ARTIFACTS_ROOT / job["input_video_relpath"]
    if not input_video_path.exists():
        raise FileNotFoundError(f"Missing staged input video: {input_video_path}")

    output_dir = (
        ARTIFACTS_ROOT
        / run_root_relpath
        / "generated_output_ltx2b_25ctx_96future"
        / clip["clip_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / (
        f"{clip['clip_id']}_ltx2b_24fps_25_observed_frames_96_future_frames_121_total_frames.mp4"
    )
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    cmd = [
        "python",
        "inference.py",
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
        "--height",
        str(int(clip["height"])),
        "--width",
        str(int(clip["width"])),
        "--num_frames",
        str(TARGET_FRAMES),
        "--frame_rate",
        str(FPS),
        "--seed",
        str(seed),
        "--pipeline_config",
        str(REPO / "configs" / "ltxv-2b-0.9.8-distilled.yaml"),
        "--output_path",
        str(output_dir),
        "--conditioning_media_paths",
        str(input_video_path),
        "--conditioning_start_frames",
        "0",
    ]
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
        "clip": clip,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_num_frames": CONTEXT_FRAMES,
        "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
        "future_num_frames": FUTURE_FRAMES,
        "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
        "target_num_frames": TARGET_FRAMES,
        "target_seconds_frames_minus_one": seconds_for_frames(TARGET_FRAMES, FPS),
        "frame_rate": FPS,
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


def upload_run_summary(summary: dict[str, Any], *, run_root_relpath: PurePosixPath) -> str:
    tmp_dir = Path("/tmp/oneclip_24fps_25ctx_96future_summary")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / "run_summary.json"
    local_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    remote_path = run_root_relpath / "run_summary.json"
    with artifacts_volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path.as_posix())
    return remote_path.as_posix()


@app.local_entrypoint()
def main(
    scene_token: str = DEFAULT_SCENE_TOKEN,
    width: int = 512,
    height: int = 512,
    seed: int = 25096,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    run_label: str = "",
) -> None:
    source_path = find_source(scene_token)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = run_label or f"scene_{scene_token}_24fps_25ctx_96future_ltx2b_{timestamp}"
    run_root_relpath = RUNS_ROOT / run_name

    source_spec = upload_source(
        source_path=source_path,
        scene_token=scene_token,
        run_root_relpath=run_root_relpath,
    )
    job = prepare_context_clip.remote(
        asdict(source_spec),
        run_root_relpath=run_root_relpath.as_posix(),
        width=width,
        height=height,
    )
    result = run_ltx.remote(
        job,
        run_root_relpath=run_root_relpath.as_posix(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    summary = {
        "app_name": APP_NAME,
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "scene_token": scene_token,
        "source_path": str(source_path),
        "fps": FPS,
        "context_frames": CONTEXT_FRAMES,
        "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
        "future_frames": FUTURE_FRAMES,
        "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
        "target_frames": TARGET_FRAMES,
        "target_seconds_frames_minus_one": seconds_for_frames(TARGET_FRAMES, FPS),
        "width": width,
        "height": height,
        "run_root_relpath": run_root_relpath.as_posix(),
        "summary_relpath": (run_root_relpath / "run_summary.json").as_posix(),
        "result": result,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    upload_run_summary(summary, run_root_relpath=run_root_relpath)
    print(json.dumps(summary, indent=2, sort_keys=True))
