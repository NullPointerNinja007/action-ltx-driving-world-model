from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-waymo-rife-vs-minterpolate-24fps-seed231")
MODELS_VOLUME_NAME = "models"
ARTIFACTS_VOLUME_NAME = os.environ.get(
    "LTX_ARTIFACTS_VOLUME",
    "waymo-ltx2b-rife-vs-minterpolate-24fps-seed231",
)
GPU_TYPE = os.environ.get("LTX_MODAL_GPU", "H100")

MODELS_ROOT = Path("/models")
ARTIFACTS_ROOT = Path("/artifacts")
LTX_REPO = Path("/workspace/LTX-Video")
RIFE_REPO = Path("/workspace/Practical-RIFE")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"

FPS = 24
SOURCE_FPS = 10
RIFE_MULTI = 3
RIFE_INTERMEDIATE_FPS = SOURCE_FPS * RIFE_MULTI
WIDTH = 512
HEIGHT = 512
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TARGET_FRAMES = CONTEXT_FRAMES + FUTURE_FRAMES
SEED = 231

RUNS_ROOT = PurePosixPath("rife_vs_minterpolate_24fps_ltx2b_seed231_runs")

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

RIFE_MODEL_SOURCES = [
    {
        "label": "rife_v4_25_lite",
        "kind": "gdown",
        "url": "https://drive.google.com/file/d/1zlKblGuKNatulJNFf5jdB-emp9AqGK05/view?usp=share_link",
    },
    {
        "label": "rife_v3_6_hf_fallback",
        "kind": "url",
        "url": "https://huggingface.co/aka7774/ECCV2022-RIFE/resolve/main/RIFE_trained_model_v3.6.zip",
    },
]


@dataclass(frozen=True)
class SourcePair:
    scene_token: str
    raw_10fps_filename: str
    raw_10fps_relpath: str
    minterpolate_24fps_filename: str
    minterpolate_24fps_relpath: str


app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "fonts-dejavu-core", "libgl1", "libglib2.0-0", "unzip", "wget")
    .pip_install(
        "torch",
        "torchvision",
        "huggingface_hub",
        "av",
        "gdown",
        "imageio",
        "imageio-ffmpeg",
        "imageio[ffmpeg]",
        "moviepy",
        "numpy<2",
        "opencv-python-headless",
        "scikit-video",
        "tqdm",
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video",
        "cd /workspace/LTX-Video && python -m pip install -e '.[inference-script]'",
        "git clone https://github.com/hzwer/Practical-RIFE.git /workspace/Practical-RIFE",
        "python -m pip install 'numpy<2'",
    )
)


def run_checked(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print("FAILED COMMAND:", " ".join(cmd))
        if result.stdout:
            print("STDOUT:\n" + result.stdout)
        if result.stderr:
            print("STDERR:\n" + result.stderr)
        result.check_returncode()
    return result


def seconds_for_frames(num_frames: int, fps: int) -> float:
    return (num_frames - 1) / fps


def ltx_compatible_frame_count(frame_count: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % 8 == 0


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def scene_token_from_path(path: Path) -> str:
    match = re.search(r"context_([0-9a-f]{12})_frames", path.name)
    if not match:
        raise ValueError(f"Could not parse scene token from {path.name}")
    return match.group(1)


def discover_source_pairs(raw_dir: Path, minterpolate_dir: Path) -> list[tuple[str, Path, Path]]:
    raw_paths = sorted(raw_dir.glob("waymo_full20s_201f_context_val_201f_10fps_20p0s_context_*.mp4"))
    if len(raw_paths) != 5:
        raise ValueError(f"Expected exactly 5 local raw 10 FPS Waymo clips in {raw_dir}, found {len(raw_paths)}")

    pairs: list[tuple[str, Path, Path]] = []
    for raw_path in raw_paths:
        scene_token = scene_token_from_path(raw_path)
        matches = sorted(minterpolate_dir.glob(f"*{scene_token}*_minterpolate_24fps.mp4"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one minterpolate 24 FPS clip for scene {scene_token}, "
                f"found {len(matches)} in {minterpolate_dir}"
            )
        pairs.append((scene_token, raw_path, matches[0]))
    return pairs


def upload_sources(pairs: list[tuple[str, Path, Path]], run_root_relpath: PurePosixPath) -> list[dict[str, Any]]:
    source_specs: list[dict[str, Any]] = []
    raw_upload_root = run_root_relpath / "source_raw_10fps_full20s"
    minterpolate_upload_root = run_root_relpath / "source_minterpolate_24fps_full20s"
    with artifacts_volume.batch_upload(force=True) as batch:
        for scene_token, raw_path, minterpolate_path in pairs:
            raw_relpath = raw_upload_root / raw_path.name
            minterpolate_relpath = minterpolate_upload_root / minterpolate_path.name
            batch.put_file(raw_path, raw_relpath.as_posix())
            batch.put_file(minterpolate_path, minterpolate_relpath.as_posix())
            source_specs.append(
                asdict(
                    SourcePair(
                        scene_token=scene_token,
                        raw_10fps_filename=raw_path.name,
                        raw_10fps_relpath=raw_relpath.as_posix(),
                        minterpolate_24fps_filename=minterpolate_path.name,
                        minterpolate_24fps_relpath=minterpolate_relpath.as_posix(),
                    )
                )
            )
    return source_specs


def ensure_ltx_checkpoints() -> None:
    for filename in [CKPT_2B, UPSCALER]:
        src = MODELS_ROOT / "ltx" / filename
        dst = LTX_REPO / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint in Modal volume: {src}")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)


def find_file(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def install_rife_model() -> dict[str, str]:
    model_cache_root = ARTIFACTS_ROOT / "_rife_model_cache"
    model_cache_root.mkdir(parents=True, exist_ok=True)
    active_model_dir = model_cache_root / "active_train_log"

    if (active_model_dir / "flownet.pkl").exists() and (active_model_dir / "RIFE_HDv3.py").exists():
        shutil.copytree(active_model_dir, RIFE_REPO / "train_log", dirs_exist_ok=True)
        return {"model_dir": active_model_dir.relative_to(ARTIFACTS_ROOT).as_posix(), "source": "cached"}

    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rife_model_download_") as tmp:
        tmp_dir = Path(tmp)
        for source in RIFE_MODEL_SOURCES:
            try:
                archive = tmp_dir / f"{source['label']}.zip"
                if source["kind"] == "gdown":
                    run_checked(["python", "-m", "gdown", "--fuzzy", source["url"], "-O", str(archive)])
                else:
                    run_checked(["wget", "-O", str(archive), source["url"]])

                extract_root = tmp_dir / f"extract_{source['label']}"
                extract_root.mkdir(parents=True, exist_ok=True)
                run_checked(["unzip", "-q", str(archive), "-d", str(extract_root)])

                flownet = find_file(extract_root, "flownet.pkl")
                rife_py = find_file(extract_root, "RIFE_HDv3.py")
                if not flownet or not rife_py:
                    raise RuntimeError(f"{source['label']} did not contain flownet.pkl and RIFE_HDv3.py")

                active_model_dir.mkdir(parents=True, exist_ok=True)
                for child in flownet.parent.iterdir():
                    dst = active_model_dir / child.name
                    if child.is_dir():
                        shutil.copytree(child, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(child, dst)
                shutil.copytree(active_model_dir, RIFE_REPO / "train_log", dirs_exist_ok=True)
                (active_model_dir / "model_source.json").write_text(
                    json.dumps(source, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                artifacts_volume.commit()
                return {"model_dir": active_model_dir.relative_to(ARTIFACTS_ROOT).as_posix(), "source": source["label"]}
            except Exception as exc:  # noqa: BLE001 - keep fallback errors visible in the final failure.
                errors.append(f"{source['label']}: {exc}")

    raise RuntimeError("Failed to install any RIFE model:\n" + "\n".join(errors))


def patch_skvideo_numpy_aliases() -> None:
    import skvideo.io.abstract

    path = Path(skvideo.io.abstract.__file__)
    text = path.read_text(encoding="utf-8")
    patched = (
        text.replace("np.float", "float")
        .replace("np.int", "int")
        .replace("np.bool", "bool")
    )
    if patched != text:
        path.write_text(patched, encoding="utf-8")


def count_video_frames(path: Path) -> int:
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        return int(reader.count_frames())
    finally:
        reader.close()


def make_rife_24fps_clip(raw_10fps_path: Path, out_24fps_path: Path) -> dict[str, Any]:
    if out_24fps_path.exists():
        return {"relpath": out_24fps_path.relative_to(ARTIFACTS_ROOT).as_posix(), "frames": count_video_frames(out_24fps_path)}

    out_24fps_path.parent.mkdir(parents=True, exist_ok=True)
    patch_skvideo_numpy_aliases()
    with tempfile.TemporaryDirectory(prefix="rife_10_to_24_") as tmp:
        tmp_dir = Path(tmp)
        rife_30fps = tmp_dir / "rife_3x_30fps.mp4"
        run_checked(
            [
                "python",
                "inference_video.py",
                "--video",
                str(raw_10fps_path),
                "--output",
                str(rife_30fps),
                "--multi",
                str(RIFE_MULTI),
                "--fps",
                str(RIFE_INTERMEDIATE_FPS),
                "--fp16",
                "--scale",
                "1.0",
            ],
            cwd=RIFE_REPO,
        )
        run_checked(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(rife_30fps),
                "-an",
                "-vf",
                f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1",
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
                str(out_24fps_path),
            ]
        )
    return {"relpath": out_24fps_path.relative_to(ARTIFACTS_ROOT).as_posix(), "frames": count_video_frames(out_24fps_path)}


def make_context_clip(source_24fps_path: Path, context_path: Path) -> dict[str, Any]:
    if not context_path.exists():
        context_path.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_24fps_path),
                "-an",
                "-vf",
                f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1",
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
                str(context_path),
            ]
        )
    frames = count_video_frames(context_path)
    if frames != CONTEXT_FRAMES:
        raise RuntimeError(f"{context_path} has {frames} frames, expected {CONTEXT_FRAMES}")
    return {"relpath": context_path.relative_to(ARTIFACTS_ROOT).as_posix(), "frames": frames}


def run_ltx_generation(
    *,
    context_path: Path,
    output_dir: Path,
    canonical_output: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if canonical_output.exists() and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    run_checked(
        [
            "python",
            "inference.py",
            "--prompt",
            prompt,
            "--negative_prompt",
            negative_prompt,
            "--height",
            str(HEIGHT),
            "--width",
            str(WIDTH),
            "--num_frames",
            str(TARGET_FRAMES),
            "--frame_rate",
            str(FPS),
            "--seed",
            str(seed),
            "--pipeline_config",
            str(LTX_REPO / "configs" / "ltxv-2b-0.9.8-distilled.yaml"),
            "--output_path",
            str(output_dir),
            "--conditioning_media_paths",
            str(context_path),
            "--conditioning_start_frames",
            "0",
        ],
        cwd=LTX_REPO,
    )

    generated_candidates = sorted(
        path
        for path in output_dir.rglob("*.mp4")
        if path.name != context_path.name and path.name != canonical_output.name
    )
    if not generated_candidates:
        raise RuntimeError(f"No generated mp4 found under {output_dir}")

    shutil.copy2(generated_candidates[0], canonical_output)
    result = {
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "target_frames": TARGET_FRAMES,
        "fps": FPS,
        "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
        "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
        "generated_video_relpath": canonical_output.relative_to(ARTIFACTS_ROOT).as_posix(),
        "raw_generated_video_relpath": generated_candidates[0].relative_to(ARTIFACTS_ROOT).as_posix(),
        "model_checkpoint": CKPT_2B,
        "pipeline_config": "ltxv-2b-0.9.8-distilled.yaml",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def make_side_by_side(left: Path, right: Path, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            (
                "[0:v]scale=512:512,setsar=1,"
                "drawtext=text='LEFT RIFE 24fps conditioning -> LTX seed 231':"
                "fontcolor=white:fontsize=18:box=1:boxcolor=black@0.65:x=12:y=12[l];"
                "[1:v]scale=512:512,setsar=1,"
                "drawtext=text='RIGHT minterpolate 24fps conditioning -> LTX seed 231':"
                "fontcolor=white:fontsize=18:box=1:boxcolor=black@0.65:x=12:y=12[r];"
                "[l][r]hstack=inputs=2[v]"
            ),
            "-map",
            "[v]",
            "-an",
            "-r",
            str(FPS),
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
    return {"relpath": out_path.relative_to(ARTIFACTS_ROOT).as_posix(), "frames": count_video_frames(out_path)}


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=49152,
    timeout=12 * 60 * 60,
    volumes={str(MODELS_ROOT): models_volume, str(ARTIFACTS_ROOT): artifacts_volume},
)
def run_comparison_batch(
    source_pair_payloads: list[dict[str, Any]],
    *,
    run_root_relpath: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
) -> dict[str, Any]:
    artifacts_volume.reload()
    ensure_ltx_checkpoints()
    rife_model_info = install_rife_model()

    run_root = ARTIFACTS_ROOT / run_root_relpath
    results: list[dict[str, Any]] = []

    for payload in source_pair_payloads:
        pair = SourcePair(**payload)
        scene = safe_name(pair.scene_token)
        raw_10fps_path = ARTIFACTS_ROOT / pair.raw_10fps_relpath
        minterpolate_24fps_path = ARTIFACTS_ROOT / pair.minterpolate_24fps_relpath

        rife_24fps_path = run_root / "source_rife_24fps_full20s" / f"scene_{scene}_rife_from_10fps_to_24fps.mp4"
        rife_source = make_rife_24fps_clip(raw_10fps_path, rife_24fps_path)

        context_paths = {
            "rife": run_root / "observed_context_inputs_49f" / "rife" / f"scene_{scene}_rife_24fps_49f_context.mp4",
            "minterpolate": run_root
            / "observed_context_inputs_49f"
            / "minterpolate"
            / f"scene_{scene}_minterpolate_24fps_49f_context.mp4",
        }
        contexts = {
            "rife": make_context_clip(rife_24fps_path, context_paths["rife"]),
            "minterpolate": make_context_clip(minterpolate_24fps_path, context_paths["minterpolate"]),
        }

        generations: dict[str, Any] = {}
        for method in ["rife", "minterpolate"]:
            output_dir = run_root / "generated_ltx2b_49ctx_72future_seed231" / method / f"scene_{scene}"
            canonical = output_dir / f"scene_{scene}_{method}_ltx2b_49ctx_72future_121total_24fps_seed231.mp4"
            generations[method] = run_ltx_generation(
                context_path=context_paths[method],
                output_dir=output_dir,
                canonical_output=canonical,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
            )

        side_by_side_path = (
            run_root
            / "side_by_side_comparisons_local_download"
            / f"scene_{scene}_LEFT-rife_RIGHT-minterpolate_ltx2b_49ctx_72future_seed231.mp4"
        )
        side_by_side = make_side_by_side(
            ARTIFACTS_ROOT / generations["rife"]["generated_video_relpath"],
            ARTIFACTS_ROOT / generations["minterpolate"]["generated_video_relpath"],
            side_by_side_path,
        )

        result = {
            "scene_token": pair.scene_token,
            "seed": seed,
            "raw_10fps_relpath": pair.raw_10fps_relpath,
            "minterpolate_24fps_relpath": pair.minterpolate_24fps_relpath,
            "rife_24fps": rife_source,
            "contexts": contexts,
            "generations": generations,
            "side_by_side": side_by_side,
        }
        result_path = run_root / "per_scene_results" / f"scene_{scene}_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        results.append(result)
        artifacts_volume.commit()

    summary = {
        "app_name": APP_NAME,
        "artifact_volume": ARTIFACTS_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "gpu_type": GPU_TYPE,
        "seed": seed,
        "rife_model": rife_model_info,
        "source_fps": SOURCE_FPS,
        "fps": FPS,
        "rife_intermediate_fps": RIFE_INTERMEDIATE_FPS,
        "rife_multi": RIFE_MULTI,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "target_frames": TARGET_FRAMES,
        "context_seconds_frames_minus_one": seconds_for_frames(CONTEXT_FRAMES, FPS),
        "future_seconds_frames_over_fps": FUTURE_FRAMES / FPS,
        "num_scenes": len(results),
        "run_root_relpath": run_root_relpath,
        "results": results,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    artifacts_volume.commit()
    return summary


def download_volume_file(relpath: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as handle:
        artifacts_volume.read_file_into_fileobj(relpath, handle)


@app.local_entrypoint()
def main(
    raw_input_dir: str = "data/inference_input_clips",
    minterpolate_input_dir: str = "data/inference_input_clips/interpolated_24fps_waymo_full20s",
    local_output_dir: str = "data/rife_vs_minterpolate_24fps_ltx2b_seed231_side_by_side",
    seed: int = SEED,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    run_label: str = "",
) -> None:
    if seed != SEED:
        raise ValueError("This project sidequest is fixed to seed 231.")
    if not ltx_compatible_frame_count(CONTEXT_FRAMES):
        raise ValueError("CONTEXT_FRAMES must be 8N+1.")
    if not ltx_compatible_frame_count(TARGET_FRAMES):
        raise ValueError("TARGET_FRAMES must be 8N+1.")

    raw_dir = Path(raw_input_dir)
    minterpolate_dir = Path(minterpolate_input_dir)
    output_dir = Path(local_output_dir)

    pairs = discover_source_pairs(raw_dir, minterpolate_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = run_label or f"waymo_5clips_rife_vs_minterpolate_24fps_ltx2b_49ctx_72future_seed231_{timestamp}"
    run_root_relpath = RUNS_ROOT / run_name
    source_pair_payloads = upload_sources(pairs, run_root_relpath)

    summary = run_comparison_batch.remote(
        source_pair_payloads,
        run_root_relpath=run_root_relpath.as_posix(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    downloaded: list[dict[str, str]] = []
    for result in summary["results"]:
        relpath = result["side_by_side"]["relpath"]
        local_path = output_dir / Path(relpath).name
        download_volume_file(relpath, local_path)
        downloaded.append({"scene_token": result["scene_token"], "local_path": str(local_path), "volume_relpath": relpath})

    summary_local = {
        **summary,
        "local_output_dir": str(output_dir),
        "downloaded_side_by_side": downloaded,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(json.dumps(summary_local, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary_local, indent=2, sort_keys=True))
