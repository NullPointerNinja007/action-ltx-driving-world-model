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


APP_NAME = "video-quality-metrics-benchmark"
VOLUME_NAME = os.environ.get("VIDEO_QUALITY_VOLUME_NAME", "video-quality-metrics-benchmark-clips")
REMOTE_ROOT = Path("/quality_data")

app = modal.App(APP_NAME)
quality_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg")
)


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


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


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
        grouped[str(row["model_mode"])].append(row)
    summaries: list[dict[str, Any]] = []
    for model_mode, model_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {"model_mode": model_mode, "num_clips": len(model_rows)}
        for name in metric_names:
            summary[f"mean_{name}"] = mean([float(row[name]) for row in model_rows])
        summaries.append(summary)
    return summaries


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(REMOTE_ROOT): quality_volume},
)
def compute_quality_shard(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
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
        if gray.shape[1] < 3 or gray.shape[2] < 3:
            return 0.0
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
        diffs = np.diff(frames.astype(np.float32), axis=0)
        return float(np.mean(np.abs(diffs)))

    def temporal_delta_error(generated: np.ndarray, reference: np.ndarray) -> float:
        if len(generated) < 2 or len(reference) < 2:
            return 0.0
        gen_d = np.diff(generated.astype(np.float32), axis=0)
        ref_d = np.diff(reference.astype(np.float32), axis=0)
        return float(np.mean(np.abs(gen_d - ref_d)))

    def fft_high_frequency_energy(frames: np.ndarray, cutoff_fraction: float = 0.25, stride: int = 4) -> float:
        gray = rgb_to_gray(frames[::2, ::stride, ::stride, :]).astype(np.float32)
        if gray.shape[1] < 4 or gray.shape[2] < 4:
            return 0.0
        spectrum = np.fft.fftshift(np.fft.fft2(gray, axes=(1, 2)), axes=(1, 2))
        power = np.abs(spectrum) ** 2
        h, w = gray.shape[1:3]
        yy, xx = np.ogrid[:h, :w]
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_radius = max(float(radius.max()), 1e-6)
        high_mask = radius >= cutoff_fraction * max_radius
        return float(np.mean(power[:, high_mask]))

    def low_frequency_frames(frames: np.ndarray, stride: int = 8) -> np.ndarray:
        return frames[:, ::stride, ::stride, :].astype(np.float32)

    def copy_leakage_metrics(generated_future: np.ndarray, reference_context: np.ndarray, reference_future: np.ndarray) -> dict[str, float]:
        gen_future = low_frequency_frames(generated_future).astype(np.float32)
        ref_context = low_frequency_frames(reference_context).astype(np.float32)
        ref_future = low_frequency_frames(reference_future).astype(np.float32)
        min_context_mses: list[float] = []
        for future_frame in gen_future:
            diff = ref_context - future_frame[None, ...]
            frame_mses = np.mean(diff * diff, axis=(1, 2, 3))
            min_context_mses.append(float(np.min(frame_mses)))
        future_mse = float(np.mean((gen_future - ref_future) ** 2))
        min_context_mse = float(np.mean(min_context_mses))
        return {
            "copy_leakage_min_context_mse_downsampled": min_context_mse,
            "copy_leakage_future_mse_downsampled": future_mse,
            "copy_leakage_ratio_min_context_over_future": min_context_mse / max(future_mse, 1e-12),
        }

    quality_volume.reload()
    rows: list[dict[str, Any]] = []
    source_cache: dict[str, Any] = {}
    for record in records:
        generated = decode_video_frames(REMOTE_ROOT / record["remote_generated_relpath"], frame_count=total_frames)
        source_relpath = record["remote_source_relpath"]
        if source_relpath not in source_cache:
            source_cache[source_relpath] = decode_video_frames(REMOTE_ROOT / source_relpath, frame_count=total_frames)
        reference = source_cache[source_relpath]

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
            "scene_token": record["scene_token"],
            "model_mode": record["model_mode"],
            "generated_file": record.get("local_file", ""),
            "source_file": record.get("source_filename", ""),
            "fps": fps,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
            "seed": record.get("seed", ""),
            "using_lora": record.get("using_lora", ""),
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


def chunked(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]


@app.local_entrypoint()
def main(
    manifest: str,
    source_dir: str = "data/inference_input_clips/interpolated_24fps_waymo_full20s",
    output_dir: str = "data/benchmarks/video_quality_modal",
    run_id: str = "",
    fps: int = 24,
    width: int = 512,
    height: int = 512,
    context_frames: int = 49,
    future_frames: int = 72,
    total_frames: int = 121,
    chunk_size: int = 8,
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
    if context_frames + future_frames != total_frames:
        raise ValueError("context_frames + future_frames must equal total_frames.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not run_id:
        run_id = "video_quality_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    remote_records: list[dict[str, Any]] = []
    uploaded_sources: dict[str, str] = {}
    with quality_volume.batch_upload(force=True) as batch:
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

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({**payload, "records": remote_records}, handle, indent=2, sort_keys=True)
            temp_manifest_path = Path(handle.name)
        batch.put_file(temp_manifest_path, f"{run_id}/manifest.json")

    shards = chunked(remote_records, chunk_size)
    shard_payloads = [
        {
            "records": shard,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
        }
        for shard in shards
    ]
    shard_rows = list(
        compute_quality_shard.map(
            shard_payloads,
        )
    )
    rows = [row for shard in shard_rows for row in shard]
    rows.sort(key=lambda row: (str(row["model_mode"]), str(row["scene_token"])))
    summaries = summarize_by_model(rows)

    output_root.mkdir(parents=True, exist_ok=True)
    per_clip_csv = output_root / "per_clip_metrics.csv"
    summary_csv = output_root / "model_summary.csv"
    report_json = output_root / "video_quality_modal_report.json"
    write_csv(per_clip_csv, rows)
    write_csv(summary_csv, summaries)
    report_json.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": str(manifest_path),
                "num_records": len(records),
                "num_models": len(summaries),
                "num_shards": len(shards),
                "chunk_size": chunk_size,
                "per_clip_metrics_csv": str(per_clip_csv),
                "model_summary_csv": str(summary_csv),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"per_clip_metrics_csv": str(per_clip_csv), "model_summary_csv": str(summary_csv)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    main(args.manifest)
