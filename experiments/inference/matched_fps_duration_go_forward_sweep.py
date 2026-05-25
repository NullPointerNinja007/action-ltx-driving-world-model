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


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-waymo-all5-matched-fps-go-forward-2b")
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME",
    "waymo-ltx2b-all5-fps-4s-go-forward-sweep",
)
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"

SOURCE_SCENE_TOKENS = [
    "0081c4821701",
    "00c73f2a3515",
    "00f755ca8954",
    "01d758be9706",
    "023de465f73a",
]
RUNS_ROOT = PurePosixPath("matched_fps_duration_go_forward_runs")

DEFAULT_BASE_PROMPT = (
    "Forward-facing autonomous driving dashcam video from a real Waymo-style car-mounted camera. "
    "Preserve the observed scene identity, camera viewpoint, road layout, lane geometry, traffic lights, "
    "nearby vehicles, sidewalks, buildings, weather, and lighting."
)
DEFAULT_NEGATIVE_PROMPT = (
    "wrong action, braking, turning left, turning right, repeated input, scene restart, camera cut, "
    "new location, blurry, jittery, distorted, impossible vehicle motion"
)

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
    duration_bucket: str
    fps: int
    context_num_frames: int
    target_num_frames: int
    source_filename: str
    source_relpath: str
    clip_id: str
    width: int
    height: int
    source_mode: str = "matched_fps_all5_waymo_scenes"


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


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", "p")


def context_tag(num_frames: int, fps: int) -> str:
    return f"{num_frames}f_{fps}fps_{format_seconds(seconds_for_frames(num_frames, fps))}s_context"


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
        "ACTION TO PERFORM: GO FORWARD. GO FORWARD is the required future behavior. "
        "The ego vehicle must continue straight immediately after the observed context clip. "
        "Do not brake as the main action. Do not turn left. Do not turn right. "
        f"This is the matched-duration FPS sweep bucket {duration_bucket}. "
        f"The observed input is {context_num_frames} frames at {fps} FPS "
        f"({seconds_for_frames(context_num_frames, fps):.2f} seconds of fixed history). "
        f"Generate the next {future_frames} frames at {fps} FPS "
        f"({future_frames / fps:.2f} seconds) as a future continuation. "
        "Do not restart the scene. Do not copy the observed clip again. Do not switch locations. "
        "The generated future must begin exactly where the observed clip ends and must follow GO FORWARD. "
        f"{base_prompt} Maintain stable ego motion, consistent road geometry, and stable object identities."
    )


def find_single_source(local_dir: Path, pattern: str, scene_token: str) -> Path:
    matches = sorted(path for path in local_dir.glob(pattern) if scene_token in path.name)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source in {local_dir} matching {pattern} and {scene_token}, found {len(matches)}.")
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


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(ARTIFACTS_ROOT): artifacts_volume},
)
def prepare_context_clips(
    source_specs: dict[str, dict[int, dict[str, Any]]],
    *,
    run_root_relpath: str,
    scene_tokens: list[str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    artifacts_volume.reload()
    run_root = ARTIFACTS_ROOT / run_root_relpath
    jobs: list[dict[str, Any]] = []

    for scene_token in scene_tokens:
        scene_source_specs = source_specs.get(scene_token)
        if scene_source_specs is None:
            raise KeyError(f"Missing uploaded source specs for scene {scene_token}")

        for bucket_idx, duration_bucket in enumerate(DURATION_BUCKETS):
            for fps in [10, 24, 30]:
                context_num_frames = FPS_CONTEXTS[fps][bucket_idx]
                target_num_frames = context_num_frames + FUTURE_FRAMES_BY_FPS[fps]
                if not ltx_compatible_frame_count(context_num_frames):
                    raise ValueError(f"Invalid context frame count: {context_num_frames}")
                if not ltx_compatible_frame_count(target_num_frames):
                    raise ValueError(f"Invalid target frame count: {target_num_frames}")

                source_payload = scene_source_specs.get(fps) or scene_source_specs.get(str(fps))
                if source_payload is None:
                    raise KeyError(f"Missing uploaded source spec for scene {scene_token}, {fps} FPS")
                source = SourceSpec(**source_payload)
                source_path = ARTIFACTS_ROOT / source.source_relpath
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing uploaded source clip: {source_path}")

                tag = context_tag(context_num_frames, fps)
                source_stem = safe_stem(Path(source.source_filename))
                clip_id = f"scene_{scene_token}_{duration_bucket}_{tag}_{source_stem}"
                output_dir = run_root / "observed_context_inputs" / scene_token / duration_bucket / f"{fps}fps" / clip_id
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
                            f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
                            "-frames:v",
                            str(context_num_frames),
                            "-r",
                            str(fps),
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
                        scene_token=scene_token,
                        duration_bucket=duration_bucket,
                        fps=fps,
                        context_num_frames=context_num_frames,
                        target_num_frames=target_num_frames,
                        source_filename=source.source_filename,
                        source_relpath=source.source_relpath,
                        clip_id=clip_id,
                        width=width,
                        height=height,
                    )
                    manifest = {
                        **asdict(clip),
                        "context_seconds": seconds_for_frames(context_num_frames, fps),
                        "target_seconds": seconds_for_frames(target_num_frames, fps),
                        "future_num_frames": target_num_frames - context_num_frames,
                        "future_seconds": (target_num_frames - context_num_frames) / fps,
                        "context_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                    }
                    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                    artifacts_volume.commit()
                else:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

                jobs.append(
                    {
                        "clip": manifest,
                        "input_video_relpath": context_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                        "input_manifest_relpath": manifest_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                    }
                )

    return jobs


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
        "0",
    ]


def run_ltx_impl(
    job: dict[str, Any],
    *,
    run_root_relpath: str,
    base_prompt: str,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    clip = job["clip"]
    input_video_path = ARTIFACTS_ROOT / job["input_video_relpath"]
    if not input_video_path.exists():
        raise FileNotFoundError(f"Missing staged input video: {input_video_path}")

    ensure_checkpoints()

    fps = int(clip["fps"])
    context_num_frames = int(clip["context_num_frames"])
    target_num_frames = int(clip["target_num_frames"])
    duration_bucket = clip["duration_bucket"]
    tag = context_tag(context_num_frames, fps)
    output_dir = (
        ARTIFACTS_ROOT
        / run_root_relpath
        / "generated_future_outputs"
        / clip["scene_token"]
        / duration_bucket
        / f"{fps}fps"
        / "required_action_go_forward"
        / clip["clip_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / (
        f"{clip['clip_id']}_ltx2b_matched_fps_{tag}_REQUIRED_ACTION_go_forward_"
        f"future_continuation_{target_num_frames}f_{fps}fps.mp4"
    )
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    prompt = build_go_forward_prompt(
        duration_bucket=duration_bucket,
        fps=fps,
        context_num_frames=context_num_frames,
        target_num_frames=target_num_frames,
        base_prompt=base_prompt,
    )
    cmd = build_inference_cmd(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_video_path=input_video_path,
        output_dir=output_dir,
        target_num_frames=target_num_frames,
        frame_rate=fps,
        seed=seed,
        pipeline_config_path=REPO / "configs" / "ltxv-2b-0.9.8-distilled.yaml",
        width=int(clip["width"]),
        height=int(clip["height"]),
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
        "clip": clip,
        "action_name": "go_forward",
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_num_frames": context_num_frames,
        "context_seconds": seconds_for_frames(context_num_frames, fps),
        "target_num_frames": target_num_frames,
        "target_seconds": seconds_for_frames(target_num_frames, fps),
        "future_num_frames": target_num_frames - context_num_frames,
        "future_seconds": (target_num_frames - context_num_frames) / fps,
        "frame_rate": fps,
        "duration_bucket": duration_bucket,
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
def run_ltx_batch(
    jobs: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    base_prompt: str,
    negative_prompt: str,
    seed_base: int,
) -> list[dict[str, Any]]:
    artifacts_volume.reload()
    results = []
    for offset, job in enumerate(jobs):
        results.append(
            run_ltx_impl(
                job,
                run_root_relpath=run_root_relpath,
                base_prompt=base_prompt,
                negative_prompt=negative_prompt,
                seed=seed_base + offset,
            )
        )
    return results


def upload_run_summary(summary: dict[str, Any], *, run_root_relpath: PurePosixPath) -> str:
    tmp_dir = Path("/tmp/matched_fps_duration_summary")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / "run_summary.json"
    local_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    remote_path = run_root_relpath / "run_summary.json"
    with artifacts_volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path.as_posix())
    return remote_path.as_posix()


@app.local_entrypoint()
def main(
    scene_tokens_csv: str = ",".join(SOURCE_SCENE_TOKENS),
    width: int = 512,
    height: int = 512,
    seed_base: int = 9000,
    prompt: str = DEFAULT_BASE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    run_label: str = "",
) -> None:
    scene_tokens = [token.strip() for token in scene_tokens_csv.split(",") if token.strip()]
    if not scene_tokens:
        raise ValueError("scene_tokens_csv must contain at least one scene token.")

    for fps, context_counts in FPS_CONTEXTS.items():
        for context_num_frames in context_counts:
            if not ltx_compatible_frame_count(context_num_frames):
                raise ValueError(f"{fps} FPS context {context_num_frames} is not 8N+1.")
            target_num_frames = context_num_frames + FUTURE_FRAMES_BY_FPS[fps]
            if not ltx_compatible_frame_count(target_num_frames):
                raise ValueError(f"{fps} FPS target {target_num_frames} is not 8N+1.")

    sources = collect_sources(scene_tokens)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = run_label or f"{len(scene_tokens)}scenes_matched_10_24_30fps_4s_future_go_forward_{timestamp}"
    run_root_relpath = RUNS_ROOT / run_name

    uploaded_sources = upload_sources(sources=sources, run_root_relpath=run_root_relpath)
    clip_jobs = prepare_context_clips.remote(
        {
            scene_token: {fps: asdict(spec) for fps, spec in fps_specs.items()}
            for scene_token, fps_specs in uploaded_sources.items()
        },
        run_root_relpath=run_root_relpath.as_posix(),
        scene_tokens=scene_tokens,
        width=width,
        height=height,
    )
    results = run_ltx_batch.remote(
        clip_jobs,
        run_root_relpath=run_root_relpath.as_posix(),
        base_prompt=prompt,
        negative_prompt=negative_prompt,
        seed_base=seed_base,
    )

    summary = {
        "app_name": APP_NAME,
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "gpu_parallelism": 1,
        "scene_tokens": scene_tokens,
        "action": "go_forward",
        "fps_contexts": FPS_CONTEXTS,
        "future_frames_by_fps": FUTURE_FRAMES_BY_FPS,
        "future_seconds_by_fps": {fps: frames / fps for fps, frames in FUTURE_FRAMES_BY_FPS.items()},
        "duration_buckets": DURATION_BUCKETS,
        "width": width,
        "height": height,
        "num_source_scenes": len(uploaded_sources),
        "num_source_clips": sum(len(fps_specs) for fps_specs in uploaded_sources.values()),
        "num_generated_variants": len(results),
        "uploaded_sources": {
            scene_token: {fps: asdict(spec) for fps, spec in fps_specs.items()}
            for scene_token, fps_specs in uploaded_sources.items()
        },
        "run_root_relpath": run_root_relpath.as_posix(),
        "summary_relpath": (run_root_relpath / "run_summary.json").as_posix(),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    upload_run_summary(summary, run_root_relpath=run_root_relpath)
    print(json.dumps(summary, indent=2, sort_keys=True))
