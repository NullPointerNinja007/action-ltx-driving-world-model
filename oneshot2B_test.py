from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-waymo-action-future-length-sweep")
PROCESSED_V2_ROOT = "gs://maleeka-waymo-e2e-1778967814/waymo-e2e/processed_v2"
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME",
    "waymo-ltx2b-full20s-action-future-baselines",
)
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
REPO = Path("/workspace/LTX-Video")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
CKPT_13B = "ltxv-13b-0.9.8-distilled.safetensors"
CKPT_13B_FP8 = "ltxv-13b-0.9.8-distilled-fp8.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"
HF_MODEL_REPO = "Lightricks/LTX-Video"

MODEL_CONFIGS = {
    "2b": {
        "label": "ltx2b",
        "checkpoint": CKPT_2B,
        "pipeline_config": "ltxv-2b-0.9.8-distilled.yaml",
        "download_if_missing": False,
    },
    "13b_fp8": {
        "label": "ltx13b_fp8",
        "checkpoint": CKPT_13B_FP8,
        "pipeline_config": "ltxv-13b-0.9.8-distilled-fp8.yaml",
        "download_if_missing": True,
    },
    "13b": {
        "label": "ltx13b",
        "checkpoint": CKPT_13B,
        "pipeline_config": "ltxv-13b-0.9.8-distilled.yaml",
        "download_if_missing": True,
    },
}

DEFAULT_BASE_PROMPT = (
    "Forward-facing autonomous driving dashcam video from a real Waymo-style car-mounted camera. "
    "Preserve the same camera viewpoint, road layout, lane geometry, nearby vehicles, traffic lights, "
    "sidewalks, buildings, weather, lighting, and scene identity from the observed context clip."
)
DEFAULT_NEGATIVE_PROMPT = (
    "wrong action, unchanged future, repeated input, scene restart, camera cut, new location, "
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)
DEFAULT_ACTIONS = "brake,turn_left,turn_right,go_forward"


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    split: str
    scenario_id: str
    start_frame: int
    end_frame: int
    num_frames: int
    fps: int
    source_tfrecord: str
    source_mode: str = "full_20_second_scenario_context"


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


def gcs_join(base: str, *parts: str) -> str:
    cleaned = [base.rstrip("/")]
    cleaned.extend(p.strip("/") for p in parts if p.strip("/"))
    return "/".join(cleaned)


def default_scenarios_uri(split: str) -> str:
    return gcs_join(
        PROCESSED_V2_ROOT,
        f"front_512_{split}",
        "metadata_clean",
        f"scenarios_{split}.csv",
    )


def default_frames_prefix(split: str) -> str:
    return gcs_join(
        PROCESSED_V2_ROOT,
        f"front_512_{split}",
        "frames_front_512",
    )


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def ltx_compatible_frame_count(frame_count: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % 8 == 0


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def context_tag(num_frames: int, fps: int) -> str:
    seconds = seconds_for_frames(num_frames, fps)
    seconds_tag = f"{seconds:.1f}".replace(".", "p")
    return f"{num_frames}f_{fps}fps_{seconds_tag}s_context"


def model_info(model_variant: str) -> dict[str, Any]:
    if model_variant not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_variant={model_variant!r}; choose one of {sorted(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[model_variant]


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
            "clearly and visibly after the observed context clip. Show decreasing forward speed, stable lane "
            "alignment, realistic braking distance, and surrounding traffic that remains physically coherent."
        ),
        "turn_left": (
            "ACTION TO PERFORM: TURN LEFT. TURN LEFT is the required future behavior. The ego vehicle must steer "
            "left after the observed context clip and enter the natural leftward road continuation or left turn "
            "path. Show clear leftward motion, correct turn geometry, lane-consistent motion, and coherent traffic."
        ),
        "turn_right": (
            "ACTION TO PERFORM: TURN RIGHT. TURN RIGHT is the required future behavior. The ego vehicle must steer "
            "right after the observed context clip and enter the natural rightward road continuation or right turn "
            "path. Show clear rightward motion, correct turn geometry, lane-consistent motion, and coherent traffic."
        ),
        "go_forward": (
            "ACTION TO PERFORM: GO FORWARD. GO FORWARD is the required future behavior. The ego vehicle must continue "
            "straight after the observed context clip. Keep the vehicle on the current forward lane or natural "
            "forward road trajectory, maintain realistic forward motion, and avoid left or right turning."
        ),
    }
    if action_name not in action_instructions:
        raise ValueError(f"Unsupported action: {action_name}")

    return (
        f"{action_instructions[action_name]} "
        f"The observed input is a {observed_seconds:.1f}-second, {observed_frames}-frame Waymo driving context clip. "
        f"Use that full context as fixed history and generate the next {future_seconds:.1f} seconds "
        f"({future_frames} future frames) as a continuation. "
        "Do not restart the scene. Do not copy the observed clip again. Do not switch locations. "
        "The future must begin where the observed clip ends and must follow the required action. "
        f"{base_prompt} Maintain realistic ego motion, consistent 3D road geometry, stable object identities, "
        "no camera cuts, no sudden viewpoint changes, and no impossible vehicle motion."
    )


def load_csv_rows(uri: str) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="waymo_scenarios_") as tmp_dir:
        local_path = Path(tmp_dir) / Path(uri).name
        run_checked(["gsutil", "cp", uri, str(local_path)])
        with local_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def build_full_context_clip(row: dict[str, str], *, context_num_frames: int) -> ClipSpec:
    min_frame_id = int(row["min_frame_id"])
    max_frame_id = int(row["max_frame_id"])
    fps = int(row.get("fps") or 10)
    end_frame = min_frame_id + context_num_frames - 1
    if end_frame > max_frame_id:
        raise ValueError(f"Scenario {row['scenario_id']} does not have {context_num_frames} contiguous frames.")

    scenario_short = row["scenario_id"][:12]
    clip_id = (
        f"{row['split']}_{context_tag(context_num_frames, fps)}_{scenario_short}_"
        f"frames_{min_frame_id:06d}_{end_frame:06d}"
    )
    return ClipSpec(
        clip_id=clip_id,
        split=row["split"],
        scenario_id=row["scenario_id"],
        start_frame=min_frame_id,
        end_frame=end_frame,
        num_frames=context_num_frames,
        fps=fps,
        source_tfrecord=row.get("source_tfrecord", ""),
    )


def select_full_context_clips(
    rows: Iterable[dict[str, str]],
    *,
    num_clips: int,
    requested_scenario_ids: list[str],
    context_num_frames: int,
) -> list[ClipSpec]:
    eligible_rows = [
        row
        for row in rows
        if row.get("is_contiguous") == "True"
        and int(row.get("num_front_frames") or 0) >= context_num_frames
    ]
    if requested_scenario_ids:
        row_by_id = {row["scenario_id"]: row for row in eligible_rows}
        missing = [scenario_id for scenario_id in requested_scenario_ids if scenario_id not in row_by_id]
        if missing:
            raise ValueError(f"Scenario ids missing or too short for full context: {missing}")
        return [build_full_context_clip(row_by_id[scenario_id], context_num_frames=context_num_frames) for scenario_id in requested_scenario_ids]

    selected = eligible_rows[:num_clips]
    if len(selected) < num_clips:
        raise ValueError(f"Requested {num_clips} full-context clips but found only {len(selected)} eligible scenarios.")
    return [build_full_context_clip(row, context_num_frames=context_num_frames) for row in selected]


def download_clip_frames(clip: ClipSpec, frames_prefix: str, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    frame_uris = [
        gcs_join(frames_prefix, clip.scenario_id, f"{frame_id:06d}.jpg")
        for frame_id in range(clip.start_frame, clip.end_frame + 1)
    ]
    run_checked(["gsutil", "-m", "cp", *frame_uris, str(dst_dir)])

    missing = [
        frame_id
        for frame_id in range(clip.start_frame, clip.end_frame + 1)
        if not (dst_dir / f"{frame_id:06d}.jpg").exists()
    ]
    if missing:
        raise RuntimeError(f"Missing downloaded frames for {clip.clip_id}: {missing[:5]}")


def stage_clips_to_volume(
    clips: list[ClipSpec],
    *,
    frames_prefix: str,
    run_root_relpath: PurePosixPath,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="waymo_context_stage_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        with artifacts_volume.batch_upload(force=True) as batch:
            for clip in clips:
                local_clip_root = tmp_root / clip.clip_id
                local_frames_dir = local_clip_root / f"observed_{context_tag(clip.num_frames, clip.fps)}_frames"
                download_clip_frames(clip, frames_prefix, local_frames_dir)

                local_manifest_path = local_clip_root / "observed_context_clip_spec.json"
                local_manifest_path.write_text(
                    json.dumps(asdict(clip), indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                remote_clip_root = run_root_relpath / f"inputs_{context_tag(clip.num_frames, clip.fps)}" / clip.clip_id
                batch.put_directory(local_clip_root, remote_clip_root.as_posix())

                jobs.append(
                    {
                        "clip": asdict(clip),
                        "input_frames_relpath": (
                            remote_clip_root / f"observed_{context_tag(clip.num_frames, clip.fps)}_frames"
                        ).as_posix(),
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
        clip = ClipSpec(**clip_job["clip"])
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


def ensure_model_file(filename: str, *, download_if_missing: bool) -> None:
    path = MODELS_ROOT / "ltx" / filename
    if path.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(f"Missing checkpoint in Modal volume: {path}")

    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename))
    shutil.copy2(downloaded, path)
    models_volume.commit()


def ensure_checkpoints(model_variant: str) -> dict[str, Any]:
    info = model_info(model_variant)
    ensure_model_file(str(info["checkpoint"]), download_if_missing=bool(info["download_if_missing"]))
    ensure_model_file(UPSCALER, download_if_missing=False)

    for filename in [str(info["checkpoint"]), UPSCALER]:
        src = MODELS_ROOT / "ltx" / filename
        dst = REPO / filename
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    return info


def build_input_video(frames_dir: Path, clip: ClipSpec, *, frame_rate: int, output_path: Path) -> None:
    pattern = frames_dir / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(frame_rate),
        "-start_number",
        str(clip.start_frame),
        "-i",
        str(pattern),
        "-frames:v",
        str(clip.num_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(frames_dir))


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
) -> list[str]:
    return [
        "python",
        "inference.py",
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
        "--height",
        "512",
        "--width",
        "512",
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


def run_ltx_for_action_future_impl(
    job: dict[str, Any],
    *,
    run_root_relpath: str,
    model_variant: str,
    negative_prompt: str,
    target_num_frames: int,
    frame_rate: int,
    seed: int,
    conditioning_start_frame: int,
) -> dict[str, Any]:
    clip = ClipSpec(**job["clip"])
    action_name = str(job["action_name"])
    action_output_dir = str(job["action_output_dir"])
    prompt = str(job["prompt"])
    input_frames_dir = ARTIFACTS_ROOT / job["input_frames_relpath"]
    if not input_frames_dir.exists():
        raise FileNotFoundError(f"Missing staged frames: {input_frames_dir}")

    info = ensure_checkpoints(model_variant)
    model_label = str(info["label"])

    output_dir = ARTIFACTS_ROOT / run_root_relpath / "generated_future_outputs" / action_output_dir / clip.clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output = output_dir / (
        f"{clip.clip_id}_{model_label}_{context_tag(clip.num_frames, frame_rate)}_REQUIRED_ACTION_{action_name}_"
        f"future_continuation_{target_num_frames}f_{frame_rate}fps.mp4"
    )
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    input_video_path = output_dir / f"{clip.clip_id}_observed_{context_tag(clip.num_frames, frame_rate)}_input.mp4"
    build_input_video(input_frames_dir, clip, frame_rate=frame_rate, output_path=input_video_path)

    cmd = build_inference_cmd(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_video_path=input_video_path,
        output_dir=output_dir,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
        seed=seed,
        conditioning_start_frame=conditioning_start_frame,
        pipeline_config_path=REPO / "configs" / str(info["pipeline_config"]),
    )
    subprocess.run(cmd, cwd=str(REPO), check=True)

    generated_candidates = sorted(
        path
        for path in output_dir.rglob("*.mp4")
        if path.name != input_video_path.name
    )
    if not generated_candidates:
        raise RuntimeError(f"No generated mp4 found under {output_dir}")

    shutil.copy2(generated_candidates[0], canonical_output)

    record = {
        "clip": asdict(clip),
        "action_name": action_name,
        "variant_name": job["variant_name"],
        "seed": seed,
        "model_variant": model_variant,
        "model_label": model_label,
        "model_checkpoint": info["checkpoint"],
        "pipeline_config": info["pipeline_config"],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_num_frames": clip.num_frames,
        "context_seconds": seconds_for_frames(clip.num_frames, frame_rate),
        "target_num_frames": target_num_frames,
        "future_num_frames": target_num_frames - clip.num_frames,
        "future_seconds": (target_num_frames - clip.num_frames) / frame_rate,
        "frame_rate": frame_rate,
        "input_frames_relpath": job["input_frames_relpath"],
        "input_video_relpath": input_video_path.relative_to(ARTIFACTS_ROOT).as_posix(),
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
    timeout=3 * 60 * 60,
    volumes={
        str(MODELS_ROOT): models_volume,
        str(ARTIFACTS_ROOT): artifacts_volume,
    },
)
def run_ltx_for_action_future(
    job: dict[str, Any],
    *,
    run_root_relpath: str,
    model_variant: str,
    negative_prompt: str,
    target_num_frames: int,
    frame_rate: int,
    seed: int,
    conditioning_start_frame: int,
) -> dict[str, Any]:
    return run_ltx_for_action_future_impl(
        job,
        run_root_relpath=run_root_relpath,
        model_variant=model_variant,
        negative_prompt=negative_prompt,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
        seed=seed,
        conditioning_start_frame=conditioning_start_frame,
    )


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
def run_ltx_action_future_batch(
    jobs: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    model_variant: str,
    negative_prompt: str,
    target_num_frames: int,
    frame_rate: int,
    seed_base: int,
    conditioning_start_frame: int,
) -> list[dict[str, Any]]:
    results = []
    for offset, job in enumerate(jobs):
        results.append(
            run_ltx_for_action_future_impl(
                job,
                run_root_relpath=run_root_relpath,
                model_variant=model_variant,
                negative_prompt=negative_prompt,
                target_num_frames=target_num_frames,
                frame_rate=frame_rate,
                seed=seed_base + offset,
                conditioning_start_frame=conditioning_start_frame,
            )
        )
    return results


def upload_run_summary(summary: dict[str, Any], *, run_root_relpath: PurePosixPath) -> str:
    with tempfile.TemporaryDirectory(prefix="waymo_full20s_summary_") as tmp_dir:
        local_path = Path(tmp_dir) / "run_summary.json"
        local_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        remote_path = run_root_relpath / "run_summary.json"
        with artifacts_volume.batch_upload(force=True) as batch:
            batch.put_file(local_path, remote_path.as_posix())
        return remote_path.as_posix()


@app.local_entrypoint()
def main(
    split: str = "val",
    num_clips: int = 5,
    context_num_frames: int = 201,
    target_num_frames: int = 257,
    frame_rate: int = 10,
    model_variant: str = "2b",
    seed_base: int = 0,
    prompt: str = DEFAULT_BASE_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    actions: str = DEFAULT_ACTIONS,
    scenarios_uri: str = "",
    frames_prefix: str = "",
    scenario_ids: str = "",
    run_label: str = "",
    conditioning_start_frame: int = 0,
) -> None:
    scenarios_uri = scenarios_uri or default_scenarios_uri(split)
    frames_prefix = frames_prefix or default_frames_prefix(split)
    info = model_info(model_variant)

    if not ltx_compatible_frame_count(context_num_frames):
        raise ValueError("context_num_frames must be of the form 8N+1 for LTX-compatible conditioning.")
    if not ltx_compatible_frame_count(target_num_frames):
        raise ValueError("target_num_frames must be of the form 8N+1 for LTX video outputs.")
    if target_num_frames <= context_num_frames:
        raise ValueError("target_num_frames must exceed context_num_frames so the model generates a future continuation.")
    if conditioning_start_frame % 8 != 0:
        raise ValueError("conditioning_start_frame must be a multiple of 8 for LTX conditioning.")

    scenario_rows = load_csv_rows(scenarios_uri)
    requested_scenario_ids = parse_csv_list(scenario_ids)
    requested_actions = parse_csv_list(actions)
    clips = select_full_context_clips(
        scenario_rows,
        num_clips=num_clips,
        requested_scenario_ids=requested_scenario_ids,
        context_num_frames=context_num_frames,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = run_label or (
        f"{split}_{info['label']}_{context_num_frames}f_{frame_rate}fps_context_to_"
        f"{target_num_frames}f_future_{timestamp}"
    )
    run_root_relpath = PurePosixPath("waymo_context_length_action_future_runs") / run_name

    clip_jobs = stage_clips_to_volume(clips, frames_prefix=frames_prefix, run_root_relpath=run_root_relpath)
    jobs = expand_action_jobs(
        clip_jobs,
        actions=requested_actions,
        base_prompt=prompt,
        target_num_frames=target_num_frames,
    )

    results = run_ltx_action_future_batch.remote(
        jobs,
        run_root_relpath=run_root_relpath.as_posix(),
        model_variant=model_variant,
        negative_prompt=negative_prompt,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
        seed_base=seed_base,
        conditioning_start_frame=conditioning_start_frame,
    )

    summary = {
        "app_name": APP_NAME,
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "model_variant": model_variant,
        "model_label": info["label"],
        "model_checkpoint": info["checkpoint"],
        "pipeline_config": info["pipeline_config"],
        "gpu_type": GPU_TYPE,
        "gpu_parallelism": 1,
        "split": split,
        "num_source_clips": len(clips),
        "num_generated_variants": len(jobs),
        "context_num_frames": context_num_frames,
        "context_seconds": seconds_for_frames(context_num_frames, frame_rate),
        "target_num_frames": target_num_frames,
        "future_num_frames": target_num_frames - context_num_frames,
        "future_seconds": (target_num_frames - context_num_frames) / frame_rate,
        "frame_rate": frame_rate,
        "actions": requested_actions,
        "scenarios_uri": scenarios_uri,
        "frames_prefix": frames_prefix,
        "run_root_relpath": run_root_relpath.as_posix(),
        "summary_relpath": (run_root_relpath / "run_summary.json").as_posix(),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    upload_run_summary(summary, run_root_relpath=run_root_relpath)

    print(json.dumps(summary, indent=2, sort_keys=True))
