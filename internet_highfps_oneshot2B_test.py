from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = "ltx-internet-dashcam-highfps-action-future-2b"
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = "internet-dashcam-ltx2b-highfps-action-future-baselines"
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"

DEFAULT_ACTIONS = "brake,turn_left,turn_right,go_forward"
DEFAULT_BASE_PROMPT = (
    "Forward-facing real driving dashcam video from a car-mounted camera, similar to Waymo front camera 1. "
    "Preserve the same camera viewpoint, road layout, lane geometry, nearby vehicles, buildings, vegetation, "
    "weather, lighting, and scene identity from the observed high-FPS context clip."
)
DEFAULT_NEGATIVE_PROMPT = (
    "wrong action, unchanged future, repeated input, scene restart, camera cut, new location, "
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)

WIKIMEDIA_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "wikimedia_us33_west_virginia_front_dashcam",
        "title": "US 33 in West Virginia",
        "source_page_url": "https://commons.wikimedia.org/wiki/File:US_33_in_West_Virginia_1-ftOd1hgDi5c.webm",
        "source_video_url": "https://upload.wikimedia.org/wikipedia/commons/d/d8/US_33_in_West_Virginia_1-ftOd1hgDi5c.webm",
        "author": "Andy Arthur",
        "license": "CC BY 3.0",
        "verified_source_fps": 30.0,
        "verified_source_duration_seconds": 60.03,
        "segment_start_seconds": 12.0,
        "description": "Rural front-facing dashcam road driving on US 33 in West Virginia.",
    },
    {
        "source_id": "wikimedia_dashcam_film_front_test_drive",
        "title": "Dashcam film",
        "source_page_url": "https://commons.wikimedia.org/wiki/File:Dashcam_film.webm",
        "source_video_url": "https://upload.wikimedia.org/wikipedia/commons/3/33/Dashcam_film.webm",
        "author": "Frukko",
        "license": "CC BY-SA 3.0",
        "verified_source_fps": 29.97,
        "verified_source_duration_seconds": 209.48,
        "segment_start_seconds": 40.0,
        "description": "Front-facing dashcam test drive from an automobile camera.",
    },
    {
        "source_id": "wikimedia_kranj_slovenia_front_dashcam",
        "title": "On the Road: Driving through Kranj",
        "source_page_url": "https://commons.wikimedia.org/wiki/File:On_the_Road-_Driving_through_Kranj.webm",
        "source_video_url": "https://upload.wikimedia.org/wikipedia/commons/7/71/On_the_Road-_Driving_through_Kranj.webm",
        "author": "Renato Lozinsek",
        "license": "CC BY 3.0",
        "verified_source_fps": 30.0,
        "verified_source_duration_seconds": 354.39,
        "segment_start_seconds": 75.0,
        "description": "Urban front-facing full-HD dashcam driving through Kranj.",
    },
)


@dataclass(frozen=True)
class InternetClipSpec:
    clip_id: str
    source_id: str
    title: str
    source_page_url: str
    source_video_url: str
    author: str
    license: str
    source_fps: float
    source_duration_seconds: float
    segment_start_seconds: float
    segment_duration_seconds: float
    num_frames: int
    fps: int
    width: int
    height: int
    description: str
    source_mode: str = "internet_highfps_front_dashcam_context"


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


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ltx_compatible_frame_count(frame_count: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % 8 == 0


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def format_seconds_for_id(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", "p")


def rational_to_float(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def build_action_prompt(
    action_name: str,
    *,
    observed_frames: int,
    target_num_frames: int,
    fps: int,
    base_prompt: str,
) -> str:
    future_frames = target_num_frames - observed_frames
    if future_frames <= 0:
        raise ValueError("target_num_frames must be greater than observed_frames for future generation.")

    observed_seconds = seconds_for_frames(observed_frames, fps)
    future_seconds = future_frames / fps
    action_instructions = {
        "brake": (
            "ACTION TO PERFORM: BRAKE. BRAKE is the required future behavior. The ego vehicle must slow down "
            "clearly and visibly immediately after the observed high-FPS context clip. Show decreasing forward "
            "speed, stable lane alignment, realistic braking distance, and physically coherent surrounding traffic."
        ),
        "turn_left": (
            "ACTION TO PERFORM: TURN LEFT. TURN LEFT is the required future behavior. The ego vehicle must steer "
            "left immediately after the observed high-FPS context clip and enter the natural leftward road path. "
            "Show clear leftward motion, correct turn geometry, lane-consistent motion, and coherent traffic."
        ),
        "turn_right": (
            "ACTION TO PERFORM: TURN RIGHT. TURN RIGHT is the required future behavior. The ego vehicle must steer "
            "right immediately after the observed high-FPS context clip and enter the natural rightward road path. "
            "Show clear rightward motion, correct turn geometry, lane-consistent motion, and coherent traffic."
        ),
        "go_forward": (
            "ACTION TO PERFORM: GO FORWARD. GO FORWARD is the required future behavior. The ego vehicle must "
            "continue straight immediately after the observed high-FPS context clip. Keep the vehicle on the "
            "current forward lane or natural forward road trajectory and avoid left or right turning."
        ),
    }
    if action_name not in action_instructions:
        raise ValueError(f"Unsupported action: {action_name}")

    return (
        f"{action_instructions[action_name]} "
        f"The observed input is a {observed_seconds:.1f}-second, {observed_frames}-frame, {fps}-FPS "
        "external front dashcam context clip. Use that full high-FPS context as fixed history and generate "
        f"the next {future_seconds:.1f} seconds ({future_frames} future frames) as a continuation. "
        "Do not restart the scene. Do not copy the observed clip again. Do not switch locations. "
        "The generated future must begin exactly where the observed clip ends and must follow the required action. "
        f"{base_prompt} Maintain realistic ego motion, consistent 3D road geometry, stable object identities, "
        "no camera cuts, no sudden viewpoint changes, and no impossible vehicle motion."
    )


def select_sources(num_clips: int, requested_source_ids: list[str]) -> list[dict[str, Any]]:
    sources_by_id = {source["source_id"]: source for source in WIKIMEDIA_SOURCES}
    if requested_source_ids:
        missing = [source_id for source_id in requested_source_ids if source_id not in sources_by_id]
        if missing:
            raise ValueError(f"Unknown source ids: {missing}")
        return [sources_by_id[source_id] for source_id in requested_source_ids]
    if num_clips > len(WIKIMEDIA_SOURCES):
        raise ValueError(f"Requested {num_clips} clips but only {len(WIKIMEDIA_SOURCES)} vetted sources are configured.")
    return list(WIKIMEDIA_SOURCES[:num_clips])


def download_url(url: str, dst_path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "action-ltx-driving-world-model/0.1 high-fps-internet-baseline"},
    )
    with urllib.request.urlopen(request, timeout=10 * 60) as response:
        with dst_path.open("wb") as output:
            shutil.copyfileobj(response, output)


def probe_video(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd.extend(
        [
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames,nb_read_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    result = run_checked(cmd)
    return json.loads(result.stdout)


def normalize_context_video(
    *,
    source_path: Path,
    output_path: Path,
    segment_start_seconds: float,
    context_num_frames: int,
    frame_rate: int,
    width: int,
    height: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={frame_rate},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(segment_start_seconds),
            "-i",
            str(source_path),
            "-an",
            "-vf",
            vf,
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
            str(output_path),
        ]
    )


def build_clip_spec(
    source: dict[str, Any],
    *,
    context_num_frames: int,
    frame_rate: int,
    width: int,
    height: int,
) -> InternetClipSpec:
    segment_duration_seconds = seconds_for_frames(context_num_frames, frame_rate)
    clip_id = (
        f"external_dashcam_{source['source_id']}_"
        f"{context_num_frames}f_{frame_rate}fps_{format_seconds_for_id(segment_duration_seconds)}s_context"
    )
    return InternetClipSpec(
        clip_id=clip_id,
        source_id=str(source["source_id"]),
        title=str(source["title"]),
        source_page_url=str(source["source_page_url"]),
        source_video_url=str(source["source_video_url"]),
        author=str(source["author"]),
        license=str(source["license"]),
        source_fps=float(source["verified_source_fps"]),
        source_duration_seconds=float(source["verified_source_duration_seconds"]),
        segment_start_seconds=float(source["segment_start_seconds"]),
        segment_duration_seconds=segment_duration_seconds,
        num_frames=context_num_frames,
        fps=frame_rate,
        width=width,
        height=height,
        description=str(source["description"]),
    )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=90 * 60,
    volumes={str(ARTIFACTS_ROOT): artifacts_volume},
)
def prepare_highfps_dashcam_contexts(
    sources: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    context_num_frames: int,
    frame_rate: int,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    run_root = ARTIFACTS_ROOT / run_root_relpath
    with tempfile.TemporaryDirectory(prefix="internet_dashcam_sources_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for source in sources:
            clip = build_clip_spec(
                source,
                context_num_frames=context_num_frames,
                frame_rate=frame_rate,
                width=width,
                height=height,
            )
            remote_clip_root = run_root / "input_highfps_dashcam_contexts" / clip.clip_id
            input_video_path = remote_clip_root / (
                f"{clip.clip_id}_observed_{context_num_frames}f_{frame_rate}fps_"
                f"{format_seconds_for_id(clip.segment_duration_seconds)}s_context.mp4"
            )
            manifest_path = remote_clip_root / "observed_context_clip_spec.json"
            if not input_video_path.exists() or not manifest_path.exists():
                source_path = tmp_root / f"{clip.source_id}.webm"
                download_url(clip.source_video_url, source_path)
                source_probe = probe_video(source_path)
                source_stream = source_probe["streams"][0]
                measured_fps = rational_to_float(source_stream["avg_frame_rate"])
                if not (25.0 <= measured_fps <= 33.0):
                    raise ValueError(f"Source {clip.source_id} measured FPS {measured_fps:.3f}, outside 25-33 FPS.")
                normalize_context_video(
                    source_path=source_path,
                    output_path=input_video_path,
                    segment_start_seconds=clip.segment_start_seconds,
                    context_num_frames=context_num_frames,
                    frame_rate=frame_rate,
                    width=width,
                    height=height,
                )
                normalized_probe = probe_video(input_video_path, count_frames=True)
                normalized_stream = normalized_probe["streams"][0]
                observed_frame_count = int(normalized_stream.get("nb_read_frames") or 0)
                if observed_frame_count != context_num_frames:
                    raise RuntimeError(
                        f"Expected {context_num_frames} frames for {clip.clip_id}, got {observed_frame_count}."
                    )
                manifest = {
                    **asdict(clip),
                    "measured_source_fps": measured_fps,
                    "source_probe": source_probe,
                    "normalized_probe": normalized_probe,
                    "input_video_relpath": input_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                artifacts_volume.commit()

            jobs.append(
                {
                    "clip": asdict(clip),
                    "input_video_relpath": input_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                    "input_manifest_relpath": manifest_path.relative_to(ARTIFACTS_ROOT).as_posix(),
                }
            )
    return jobs


def expand_action_jobs(
    clip_jobs: list[dict[str, Any]],
    *,
    actions: list[str],
    base_prompt: str,
    target_num_frames: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for clip_job in clip_jobs:
        clip = InternetClipSpec(**clip_job["clip"])
        for action_name in actions:
            expanded.append(
                {
                    **clip_job,
                    "action_name": action_name,
                    "action_output_dir": f"required_action_{action_name}",
                    "variant_name": f"{clip.clip_id}__required_action_{action_name}",
                    "prompt": build_action_prompt(
                        action_name,
                        observed_frames=clip.num_frames,
                        target_num_frames=target_num_frames,
                        fps=clip.fps,
                        base_prompt=base_prompt,
                    ),
                }
            )
    return expanded


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


def run_ltx_for_highfps_action_future_impl(
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
    clip = InternetClipSpec(**job["clip"])
    action_name = str(job["action_name"])
    action_output_dir = str(job["action_output_dir"])
    prompt = str(job["prompt"])
    input_video_path = ARTIFACTS_ROOT / job["input_video_relpath"]
    if not input_video_path.exists():
        raise FileNotFoundError(f"Missing staged input video: {input_video_path}")

    ensure_checkpoints()

    output_dir = ARTIFACTS_ROOT / run_root_relpath / "generated_future_outputs" / action_output_dir / clip.clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / (
        f"{clip.clip_id}_ltx2b_highfps_dashcam_context_REQUIRED_ACTION_{action_name}_"
        f"future_continuation_{target_num_frames}f_{frame_rate}fps.mp4"
    )
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

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
        if path.name != input_video_path.name and path.name != canonical_output.name
    )
    if not generated_candidates:
        raise RuntimeError(f"No generated mp4 found under {output_dir}")

    shutil.copy2(generated_candidates[0], canonical_output)

    record = {
        "clip": asdict(clip),
        "action_name": action_name,
        "variant_name": job["variant_name"],
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
    timeout=8 * 60 * 60,
    volumes={
        str(MODELS_ROOT): models_volume,
        str(ARTIFACTS_ROOT): artifacts_volume,
    },
)
def run_ltx_highfps_action_future_batch(
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
            run_ltx_for_highfps_action_future_impl(
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
    with tempfile.TemporaryDirectory(prefix="internet_dashcam_highfps_summary_") as tmp_dir:
        local_path = Path(tmp_dir) / "run_summary.json"
        local_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        remote_path = run_root_relpath / "run_summary.json"
        with artifacts_volume.batch_upload(force=True) as batch:
            batch.put_file(local_path, remote_path.as_posix())
        return remote_path.as_posix()


@app.local_entrypoint()
def main(
    num_clips: int = 3,
    context_num_frames: int = 201,
    target_num_frames: int = 257,
    frame_rate: int = 30,
    width: int = 512,
    height: int = 512,
    seed_base: int = 900,
    prompt: str = DEFAULT_BASE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    actions: str = DEFAULT_ACTIONS,
    source_ids: str = "",
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
    if frame_rate < 25 or frame_rate > 33:
        raise ValueError("frame_rate must stay in the high-FPS 25-33 FPS band for this experiment.")

    requested_source_ids = parse_csv_list(source_ids)
    requested_actions = parse_csv_list(actions)
    sources = select_sources(num_clips, requested_source_ids)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = run_label or (
        f"wikimedia_dashcam_{len(sources)}clips_{context_num_frames}f_{frame_rate}fps_"
        f"to_{target_num_frames}f_required_actions_{timestamp}"
    )
    run_root_relpath = PurePosixPath("internet_dashcam_highfps_action_future_runs") / run_name

    clip_jobs = prepare_highfps_dashcam_contexts.remote(
        sources,
        run_root_relpath=run_root_relpath.as_posix(),
        context_num_frames=context_num_frames,
        frame_rate=frame_rate,
        width=width,
        height=height,
    )
    jobs = expand_action_jobs(
        clip_jobs,
        actions=requested_actions,
        base_prompt=prompt,
        target_num_frames=target_num_frames,
    )

    results = run_ltx_highfps_action_future_batch.remote(
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
        "context_num_frames": context_num_frames,
        "context_seconds": seconds_for_frames(context_num_frames, frame_rate),
        "target_num_frames": target_num_frames,
        "future_num_frames": target_num_frames - context_num_frames,
        "future_seconds": (target_num_frames - context_num_frames) / frame_rate,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "actions": requested_actions,
        "source_ids": [source["source_id"] for source in sources],
        "sources": sources,
        "run_root_relpath": run_root_relpath.as_posix(),
        "summary_relpath": (run_root_relpath / "run_summary.json").as_posix(),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    upload_run_summary(summary, run_root_relpath=run_root_relpath)

    print(json.dumps(summary, indent=2, sort_keys=True))
