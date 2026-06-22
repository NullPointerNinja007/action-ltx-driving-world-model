from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "action-conditioning-fvd-benchmark"
VOLUME_NAME = os.environ.get("ACTION_FVD_VOLUME_NAME", "action-conditioning-fvd-benchmark-clips")
REMOTE_ROOT = Path("/fvd_data")

app = modal.App(APP_NAME)
fvd_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg", "torch", "torchvision")
)


def parse_step(model_mode: str) -> int:
    if "base" in model_mode and "step" not in model_mode:
        return 0
    match = re.search(r"step0*([0-9]+)$", model_mode)
    if not match:
        match = re.search(r"step0*([0-9]+)", model_mode)
    if not match:
        raise ValueError(f"Could not parse checkpoint step from model_mode={model_mode!r}")
    return int(match.group(1))


def encoder_name(model_mode: str) -> str:
    if model_mode.startswith("global_mlp"):
        return "Global MLP"
    if model_mode.startswith("temporal"):
        return "Temporal Per-Point"
    if model_mode.startswith("tiny_transformer"):
        return "Transformer Action"
    if model_mode.startswith("adaln"):
        return "AdaLN Action"
    if model_mode.startswith("noaction"):
        return "No-Action Visual LoRA"
    return model_mode


def safe_relpath(path: Path, index: int) -> str:
    suffix = path.suffix or ".mp4"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:160]
    return f"{index:05d}_{stem}{suffix}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@app.function(
    image=image,
    gpu=os.environ.get("ACTION_FVD_GPU", "A10G"),
    cpu=8,
    memory=49152,
    timeout=4 * 60 * 60,
    volumes={str(REMOTE_ROOT): fvd_volume},
)
def compute_fvd(
    manifest_relpath: str,
    *,
    context_frames: int,
    total_frames: int,
    fps: int,
    width: int,
    height: int,
    fvd_num_frames: int,
    fvd_size: int,
    batch_size: int,
) -> dict[str, Any]:
    import subprocess

    import imageio_ffmpeg
    import numpy as np
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    def decode_video_frames(path: Path, frame_count: int) -> np.ndarray:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vf = (
            f"fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            vf,
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {path}:\n{proc.stderr.decode('utf-8', errors='replace')}")
        frame_size = width * height * 3
        actual_frames = len(proc.stdout) // frame_size
        if actual_frames < frame_count:
            raise ValueError(f"{path} yielded {actual_frames} frames, expected {frame_count}")
        raw = np.frombuffer(proc.stdout[: frame_count * frame_size], dtype=np.uint8)
        return raw.reshape(frame_count, height, width, 3)

    def sample_video_frames(frames: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, len(frames) - 1, fvd_num_frames).round().astype(np.int64)
        return frames[indices]

    def preprocess_batch(videos: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frames in videos:
            sampled = sample_video_frames(frames)
            tensor = torch.from_numpy(sampled).to(dtype=torch.float32) / 255.0
            tensors.append(tensor.permute(3, 0, 1, 2))
        batch = torch.stack(tensors, dim=0).to(device)
        batch = torch.nn.functional.interpolate(
            batch,
            size=(fvd_num_frames, fvd_size, fvd_size),
            mode="trilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
        return (batch - mean) / std

    def extract_features(videos: list[np.ndarray]) -> np.ndarray:
        features: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(videos), batch_size):
                batch = preprocess_batch(videos[start : start + batch_size])
                output = model(batch).flatten(start_dim=1).detach().float().cpu().numpy()
                features.append(output)
        return np.concatenate(features, axis=0)

    def symmetric_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        matrix = (matrix + matrix.T) * 0.5
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 0.0, None)
        return (vectors * np.sqrt(values)[None, :]) @ vectors.T

    def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
        if len(features_a) < 2 or len(features_b) < 2:
            return float("nan")
        a = features_a.astype(np.float64)
        b = features_b.astype(np.float64)
        mu_a = np.mean(a, axis=0)
        mu_b = np.mean(b, axis=0)
        sigma_a = np.atleast_2d(np.cov(a, rowvar=False))
        sigma_b = np.atleast_2d(np.cov(b, rowvar=False))
        eps = 1e-6
        sigma_a = sigma_a + np.eye(sigma_a.shape[0], dtype=np.float64) * eps
        sigma_b = sigma_b + np.eye(sigma_b.shape[0], dtype=np.float64) * eps
        diff = mu_a - mu_b
        sqrt_sigma_a = symmetric_matrix_sqrt(sigma_a)
        covmean = symmetric_matrix_sqrt(sqrt_sigma_a @ sigma_b @ sqrt_sigma_a)
        value = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean)
        return float(max(value, 0.0))

    fvd_volume.reload()
    manifest = json.loads((REMOTE_ROOT / manifest_relpath).read_text(encoding="utf-8"))
    records = manifest["records"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["model_mode"]].append(record)

    source_future_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for model_mode in sorted(grouped):
        generated_futures: list[np.ndarray] = []
        reference_futures: list[np.ndarray] = []
        for record in grouped[model_mode]:
            generated = decode_video_frames(REMOTE_ROOT / record["remote_generated_relpath"], total_frames)
            generated_futures.append(generated[context_frames:])

            source_relpath = record["remote_source_relpath"]
            if source_relpath not in source_future_cache:
                source = decode_video_frames(REMOTE_ROOT / source_relpath, total_frames)
                source_future_cache[source_relpath] = source[context_frames:]
            reference_futures.append(source_future_cache[source_relpath])

        generated_features = extract_features(generated_futures)
        reference_features = extract_features(reference_futures)
        fvd_future = frechet_distance(generated_features, reference_features)
        rows.append(
            {
                "model_mode": model_mode,
                "encoder": encoder_name(model_mode),
                "step": parse_step(model_mode),
                "fvd_future": fvd_future,
                "fvd_backend": "torchvision_r3d18_kinetics400",
                "fvd_num_videos": len(generated_futures),
                "fvd_num_frames": fvd_num_frames,
                "fvd_size": fvd_size,
                "fvd_feature_dim": int(generated_features.shape[1]),
                "fvd_device": str(device),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True))

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_records": len(records),
        "num_models": len(rows),
        "rows": rows,
        "note": (
            "FVD is computed with torchvision R3D-18 Kinetics features over the 72 generated future frames. "
            "This is an internal relative metric, not canonical I3D FVD."
        ),
    }


@app.local_entrypoint()
def main(
    manifest: str,
    source_dir: str = "data/inference_input_clips/interpolated_24fps_waymo_full20s",
    output_dir: str = "data/benchmarks/action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5",
    run_id: str = "",
    context_frames: int = 49,
    total_frames: int = 121,
    fps: int = 24,
    width: int = 512,
    height: int = 512,
    fvd_num_frames: int = 16,
    fvd_size: int = 112,
    batch_size: int = 4,
) -> None:
    manifest_path = Path(manifest)
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in {manifest_path}")

    if not run_id:
        run_id = "action_fvd_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    remote_records: list[dict[str, Any]] = []
    uploaded_sources: dict[str, str] = {}
    with fvd_volume.batch_upload(force=True) as batch:
        for idx, record in enumerate(records):
            generated_path = Path(record["local_file"])
            if not generated_path.exists():
                raise FileNotFoundError(generated_path)
            generated_relpath = f"{run_id}/generated/{safe_relpath(generated_path, idx)}"
            batch.put_file(generated_path, generated_relpath)

            source_filename = record["source_filename"]
            if source_filename not in uploaded_sources:
                source_path = source_root / source_filename
                if not source_path.exists():
                    raise FileNotFoundError(source_path)
                source_relpath = f"{run_id}/source/{safe_relpath(source_path, len(uploaded_sources))}"
                batch.put_file(source_path, source_relpath)
                uploaded_sources[source_filename] = source_relpath

            remote = dict(record)
            remote["remote_generated_relpath"] = generated_relpath
            remote["remote_source_relpath"] = uploaded_sources[source_filename]
            remote_records.append(remote)

        remote_manifest = {**payload, "records": remote_records}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(remote_manifest, handle, indent=2, sort_keys=True)
            temp_manifest_path = Path(handle.name)
        manifest_relpath = f"{run_id}/manifest.json"
        batch.put_file(temp_manifest_path, manifest_relpath)

    result = compute_fvd.remote(
        manifest_relpath,
        context_frames=context_frames,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        fvd_num_frames=fvd_num_frames,
        fvd_size=fvd_size,
        batch_size=batch_size,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    fvd_csv = output_root / "fvd_summary.csv"
    report_json = output_root / "fvd_report.json"
    write_csv(fvd_csv, result["rows"])
    report_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fvd_summary_csv": str(fvd_csv), "fvd_report_json": str(report_json)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    main(args.manifest)
