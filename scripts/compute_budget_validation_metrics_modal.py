from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "budget-validation-remote-metrics"

DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
NOACTION_ARTIFACT_VOLUME_NAME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer"
V4_ARTIFACT_VOLUME_NAME = "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-infer"
ADALN_ARTIFACT_VOLUME_NAME = "ltx2b-dist098-waymo24-frameadaln-action-lora-infer"
HFTV2_ARTIFACT_VOLUME_NAME = "ltx2b-dist098-waymo24-framebneck-hft-v2-infer"

DATA_ROOT = Path("/data")
NOACTION_ROOT = Path("/artifacts_noaction")
V4_ROOT = Path("/artifacts_v4")
ADALN_ROOT = Path("/artifacts_adaln")
HFTV2_ROOT = Path("/artifacts_hftv2")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
noaction_volume = modal.Volume.from_name(NOACTION_ARTIFACT_VOLUME_NAME)
v4_volume = modal.Volume.from_name(V4_ARTIFACT_VOLUME_NAME)
adaln_volume = modal.Volume.from_name(ADALN_ARTIFACT_VOLUME_NAME)
hftv2_volume = modal.Volume.from_name(HFTV2_ARTIFACT_VOLUME_NAME)

cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg", "torch", "torchvision")
)

MOUNTED_VOLUMES = {
    str(DATA_ROOT): data_volume,
    str(NOACTION_ROOT): noaction_volume,
    str(V4_ROOT): v4_volume,
    str(ADALN_ROOT): adaln_volume,
    str(HFTV2_ROOT): hftv2_volume,
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def volume_root(volume_key: str) -> Path:
    roots = {
        "noaction": NOACTION_ROOT,
        "v4": V4_ROOT,
        "adaln": ADALN_ROOT,
        "hftv2": HFTV2_ROOT,
    }
    if volume_key not in roots:
        raise ValueError(f"Unknown generated_volume_key={volume_key!r}")
    return roots[volume_key]


def summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "future_mse",
        "future_mae",
        "future_psnr",
        "future_global_ssim",
        "context_mse",
        "context_psnr",
        "generated_future_laplacian_sharpness",
        "sharpness_ratio_generated_over_reference",
        "fft_high_frequency_energy_ratio_generated_over_reference",
        "motion_ratio_generated_over_reference",
        "low_frequency_motion_ratio_generated_over_reference",
        "temporal_delta_error_mae",
        "low_frequency_temporal_delta_error_mae",
        "boundary_mae_ratio_generated_over_reference",
        "copy_leakage_ratio_min_context_over_future",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_key"])].append(row)

    summaries: list[dict[str, Any]] = []
    for model_key, model_rows in sorted(grouped.items()):
        first = model_rows[0]
        summary: dict[str, Any] = {
            "model_key": model_key,
            "model_label": first.get("model_label", model_key),
            "checkpoint_step": first.get("checkpoint_step", ""),
            "num_clips": len(model_rows),
        }
        for name in metric_names:
            vals = [float(row[name]) for row in model_rows if row.get(name, "") != ""]
            summary[f"mean_{name}"] = mean(vals)
        summaries.append(summary)
    return summaries


@app.function(
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
)
def compute_quality_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    import subprocess

    import imageio_ffmpeg
    import numpy as np

    records = payload["records"]
    fps = int(payload["fps"])
    width = int(payload["width"])
    height = int(payload["height"])
    context_frames = int(payload["context_frames"])
    future_frames = int(payload["future_frames"])
    total_frames = int(payload["total_frames"])

    def decode_video_frames(path: Path, *, frame_count: int) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
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
            raise ValueError(f"{path} yielded {actual_frames} frames, expected {frame_count}.")
        raw = np.frombuffer(proc.stdout[: frame_count * frame_size], dtype=np.uint8)
        return raw.reshape(frame_count, height, width, 3)

    def rgb_to_gray(frames: np.ndarray) -> np.ndarray:
        frames_f = frames.astype(np.float32)
        return frames_f[..., 0] * 0.299 + frames_f[..., 1] * 0.587 + frames_f[..., 2] * 0.114

    def mse(a: np.ndarray, b: np.ndarray) -> float:
        diff = a.astype(np.float32) - b.astype(np.float32)
        return float(np.mean(diff * diff))

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    def psnr_from_mse(value: float) -> float:
        if value <= 1e-12:
            return float("inf")
        return float(20.0 * math.log10(255.0) - 10.0 * math.log10(value))

    def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
        x = rgb_to_gray(a).astype(np.float64)
        y = rgb_to_gray(b).astype(np.float64)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_x = np.mean(x, axis=(1, 2))
        mu_y = np.mean(y, axis=(1, 2))
        x_centered = x - mu_x[:, None, None]
        y_centered = y - mu_y[:, None, None]
        sigma_x = np.mean(x_centered * x_centered, axis=(1, 2))
        sigma_y = np.mean(y_centered * y_centered, axis=(1, 2))
        sigma_xy = np.mean(x_centered * y_centered, axis=(1, 2))
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        return float(np.mean(ssim))

    def laplacian_sharpness(frames: np.ndarray) -> float:
        gray = rgb_to_gray(frames)
        center = gray[:, 1:-1, 1:-1]
        lap = (
            -4.0 * center
            + gray[:, :-2, 1:-1]
            + gray[:, 2:, 1:-1]
            + gray[:, 1:-1, :-2]
            + gray[:, 1:-1, 2:]
        )
        return float(np.mean(np.var(lap, axis=(1, 2))))

    def temporal_delta_mae(frames: np.ndarray) -> float:
        if len(frames) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(frames.astype(np.float32), axis=0))))

    def temporal_delta_error(generated: np.ndarray, reference: np.ndarray) -> float:
        if len(generated) < 2 or len(reference) < 2:
            return 0.0
        gen_d = np.diff(generated.astype(np.float32), axis=0)
        ref_d = np.diff(reference.astype(np.float32), axis=0)
        return float(np.mean(np.abs(gen_d - ref_d)))

    def fft_high_frequency_energy(frames: np.ndarray, cutoff_fraction: float = 0.25, stride: int = 4) -> float:
        gray = rgb_to_gray(frames[::2, ::stride, ::stride, :]).astype(np.float32)
        spectrum = np.fft.fftshift(np.fft.fft2(gray, axes=(1, 2)), axes=(1, 2))
        power = np.abs(spectrum) ** 2
        h, w = gray.shape[1:3]
        yy, xx = np.ogrid[:h, :w]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        high_mask = radius >= cutoff_fraction * max(float(radius.max()), 1e-6)
        return float(np.mean(power[:, high_mask]))

    def low_frequency_frames(frames: np.ndarray, stride: int = 8) -> np.ndarray:
        return frames[:, ::stride, ::stride, :].astype(np.float32)

    def copy_leakage_metrics(generated_future: np.ndarray, reference_context: np.ndarray, reference_future: np.ndarray) -> dict[str, float]:
        gen_future = low_frequency_frames(generated_future)
        ref_context = low_frequency_frames(reference_context)
        ref_future = low_frequency_frames(reference_future)
        min_context_mses = []
        for future_frame in gen_future:
            diff = ref_context - future_frame[None, ...]
            min_context_mses.append(float(np.min(np.mean(diff * diff, axis=(1, 2, 3)))))
        future_mse = float(np.mean((gen_future - ref_future) ** 2))
        min_context_mse = float(np.mean(min_context_mses))
        return {
            "copy_leakage_min_context_mse_downsampled": min_context_mse,
            "copy_leakage_future_mse_downsampled": future_mse,
            "copy_leakage_ratio_min_context_over_future": min_context_mse / max(future_mse, 1e-12),
        }

    data_volume.reload()
    noaction_volume.reload()
    v4_volume.reload()
    adaln_volume.reload()
    hftv2_volume.reload()

    rows: list[dict[str, Any]] = []
    reference_cache: dict[str, Any] = {}
    for record in records:
        generated_path = volume_root(record["generated_volume_key"]) / record["generated_video_relpath"]
        reference_relpath = record["source_relpath"]
        reference_path = DATA_ROOT / reference_relpath
        generated = decode_video_frames(generated_path, frame_count=total_frames)
        if reference_relpath not in reference_cache:
            reference_cache[reference_relpath] = decode_video_frames(reference_path, frame_count=total_frames)
        reference = reference_cache[reference_relpath]

        c = context_frames
        generated_context = generated[:c]
        generated_future = generated[c:]
        reference_context = reference[:c]
        reference_future = reference[c:]

        context_mse = mse(generated_context, reference_context)
        future_mse = mse(generated_future, reference_future)
        gen_sharpness = laplacian_sharpness(generated_future)
        ref_sharpness = laplacian_sharpness(reference_future)
        gen_high_freq = fft_high_frequency_energy(generated_future)
        ref_high_freq = fft_high_frequency_energy(reference_future)
        gen_temporal = temporal_delta_mae(generated_future)
        ref_temporal = temporal_delta_mae(reference_future)
        gen_low_freq_temporal = temporal_delta_mae(low_frequency_frames(generated_future))
        ref_low_freq_temporal = temporal_delta_mae(low_frequency_frames(reference_future))
        boundary_generated = mae(generated[c], generated[c - 1])
        boundary_reference = mae(reference[c], reference[c - 1])

        metrics = {
            "model_key": record["model_key"],
            "model_label": record.get("model_label", record["model_key"]),
            "checkpoint_step": record.get("checkpoint_step", ""),
            "scene_token": record["scene_token"],
            "window_id": record.get("window_id", ""),
            "window_idx": record.get("window_idx", ""),
            "generated_video_relpath": record["generated_video_relpath"],
            "source_relpath": reference_relpath,
            "fps": fps,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
            "seed": record.get("seed", ""),
            "lora_step": record.get("lora_step", ""),
            "future_mse": future_mse,
            "future_mae": mae(generated_future, reference_future),
            "future_psnr": psnr_from_mse(future_mse),
            "future_global_ssim": global_ssim(generated_future, reference_future),
            "context_mse": context_mse,
            "context_mae": mae(generated_context, reference_context),
            "context_psnr": psnr_from_mse(context_mse),
            "context_global_ssim": global_ssim(generated_context, reference_context),
            "generated_future_laplacian_sharpness": gen_sharpness,
            "reference_future_laplacian_sharpness": ref_sharpness,
            "sharpness_ratio_generated_over_reference": gen_sharpness / max(ref_sharpness, 1e-12),
            "generated_future_fft_high_frequency_energy": gen_high_freq,
            "reference_future_fft_high_frequency_energy": ref_high_freq,
            "fft_high_frequency_energy_ratio_generated_over_reference": gen_high_freq / max(ref_high_freq, 1e-12),
            "generated_future_temporal_delta_mae": gen_temporal,
            "reference_future_temporal_delta_mae": ref_temporal,
            "motion_ratio_generated_over_reference": gen_temporal / max(ref_temporal, 1e-12),
            "temporal_delta_error_mae": temporal_delta_error(generated_future, reference_future),
            "generated_future_low_frequency_temporal_delta_mae": gen_low_freq_temporal,
            "reference_future_low_frequency_temporal_delta_mae": ref_low_freq_temporal,
            "low_frequency_motion_ratio_generated_over_reference": gen_low_freq_temporal / max(ref_low_freq_temporal, 1e-12),
            "low_frequency_temporal_delta_error_mae": temporal_delta_error(
                low_frequency_frames(generated_future),
                low_frequency_frames(reference_future),
            ),
            "context_to_future_boundary_mae_generated": boundary_generated,
            "context_to_future_boundary_mae_reference": boundary_reference,
            "boundary_mae_ratio_generated_over_reference": boundary_generated / max(boundary_reference, 1e-12),
        }
        metrics.update(copy_leakage_metrics(generated_future, reference_context, reference_future))
        rows.append(metrics)
    return rows


@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    memory=49152,
    timeout=4 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
)
def compute_fvd_for_model(payload: dict[str, Any]) -> dict[str, Any]:
    import subprocess

    import imageio_ffmpeg
    import numpy as np
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18

    records = payload["records"]
    fps = int(payload["fps"])
    width = int(payload["width"])
    height = int(payload["height"])
    context_frames = int(payload["context_frames"])
    total_frames = int(payload["total_frames"])
    fvd_num_frames = int(payload["fvd_num_frames"])
    fvd_size = int(payload["fvd_size"])
    batch_size = int(payload["batch_size"])

    def decode_video_frames(path: Path, frame_count: int) -> np.ndarray:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vf = f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
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
            raise ValueError(f"{path} yielded {actual_frames} frames, expected {frame_count}.")
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
        features = []
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

    data_volume.reload()
    noaction_volume.reload()
    v4_volume.reload()
    adaln_volume.reload()
    hftv2_volume.reload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    generated_futures: list[np.ndarray] = []
    reference_futures: list[np.ndarray] = []
    reference_cache: dict[str, np.ndarray] = {}
    for record in records:
        generated_path = volume_root(record["generated_volume_key"]) / record["generated_video_relpath"]
        generated = decode_video_frames(generated_path, total_frames)
        generated_futures.append(generated[context_frames:])

        reference_relpath = record["source_relpath"]
        if reference_relpath not in reference_cache:
            reference = decode_video_frames(DATA_ROOT / reference_relpath, total_frames)
            reference_cache[reference_relpath] = reference[context_frames:]
        reference_futures.append(reference_cache[reference_relpath])

    generated_features = extract_features(generated_futures)
    reference_features = extract_features(reference_futures)
    first = records[0]
    return {
        "model_key": first["model_key"],
        "model_label": first.get("model_label", first["model_key"]),
        "checkpoint_step": first.get("checkpoint_step", ""),
        "fvd_future": frechet_distance(generated_features, reference_features),
        "fvd_backend": "torchvision_r3d18_kinetics400",
        "fvd_num_videos": len(records),
        "fvd_num_frames": fvd_num_frames,
        "fvd_size": fvd_size,
        "fvd_feature_dim": int(generated_features.shape[1]),
        "fvd_device": str(device),
    }


def chunked(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]


@app.local_entrypoint()
def main(
    manifest: str,
    output_dir: str,
    fps: int = 24,
    width: int = 512,
    height: int = 512,
    context_frames: int = 49,
    future_frames: int = 72,
    total_frames: int = 121,
    chunk_size: int = 8,
    fvd_num_frames: int = 16,
    fvd_size: int = 112,
    fvd_batch_size: int = 8,
) -> None:
    manifest_path = Path(manifest)
    output_root = Path(output_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    if context_frames + future_frames != total_frames:
        raise ValueError("context_frames + future_frames must equal total_frames.")

    quality_payloads = [
        {
            "records": shard,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
        }
        for shard in chunked(records, chunk_size)
    ]
    shard_rows = list(compute_quality_shard.map(quality_payloads))
    rows = [row for shard in shard_rows for row in shard]
    rows.sort(key=lambda row: (str(row["model_key"]), str(row["window_id"]), str(row["scene_token"])))
    summaries = summarize_by_model(rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model_key"])].append(record)
    fvd_payloads = [
        {
            "records": model_records,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "total_frames": total_frames,
            "fvd_num_frames": fvd_num_frames,
            "fvd_size": fvd_size,
            "batch_size": fvd_batch_size,
        }
        for _, model_records in sorted(grouped.items())
    ]
    fvd_rows = list(compute_fvd_for_model.map(fvd_payloads))
    fvd_rows.sort(key=lambda row: str(row["model_key"]))

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_clip_metrics.csv", rows)
    write_csv(output_root / "model_summary.csv", summaries)
    write_csv(output_root / "fvd_summary.csv", fvd_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "num_records": len(records),
        "num_models": len(summaries),
        "num_quality_shards": len(quality_payloads),
        "chunk_size": chunk_size,
        "fvd_num_frames": fvd_num_frames,
        "fvd_size": fvd_size,
        "note": "FVD-style uses torchvision R3D-18 Kinetics features over future frames; it is for relative comparison only.",
    }
    (output_root / "metrics_modal_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.manifest, output_dir=args.output_dir)
