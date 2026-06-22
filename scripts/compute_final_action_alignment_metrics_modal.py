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


APP_NAME = "final-action-alignment-remote-metrics"

DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
NOACTION_ARTIFACT_VOLUME_NAME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer"
FINAL_ACTION_ARTIFACT_VOLUME_NAME = "ltx2b-final-action-alignment-validation-infer"

DATA_ROOT = Path("/data")
NOACTION_ROOT = Path("/artifacts_noaction")
FINAL_ACTION_ROOT = Path("/artifacts_final_action")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
noaction_volume = modal.Volume.from_name(NOACTION_ARTIFACT_VOLUME_NAME)
final_action_volume = modal.Volume.from_name(FINAL_ACTION_ARTIFACT_VOLUME_NAME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "imageio-ffmpeg", "opencv-python-headless")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg", "torch", "torchvision")
)

MOUNTED_VOLUMES = {
    str(DATA_ROOT): data_volume,
    str(NOACTION_ROOT): noaction_volume,
    str(FINAL_ACTION_ROOT): final_action_volume,
}

QUALITY_METRICS = [
    "future_mse",
    "future_mae",
    "future_psnr",
    "future_global_ssim",
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

STRATA = ["all", "low_action", "brake", "accelerate", "turning", "high_action"]
WRONG_MODES = ["zero", "shuffled", "reversed_future"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def volume_root(volume_key: str) -> Path:
    roots = {
        "noaction": NOACTION_ROOT,
        "final_action_alignment": FINAL_ACTION_ROOT,
    }
    if volume_key not in roots:
        raise ValueError(f"Unknown generated_volume_key={volume_key!r}")
    return roots[volume_key]


def chunked(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def summarize(rows: list[dict[str, Any]], group_keys: list[str], metric_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items()):
        out: dict[str, Any] = {key: group[idx] for idx, key in enumerate(group_keys)}
        out["num_clips"] = len(group_rows)
        for metric in metric_names:
            vals = [float(row[metric]) for row in group_rows if row.get(metric, "") != ""]
            out[f"mean_{metric}"] = mean(vals)
        summaries.append(out)
    return summaries


def attach_strata(rows: list[dict[str, Any]], strata_by_window: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        flags = strata_by_window.get(str(row.get("window_id", "")), {})
        active = ["all"] + [name for name in STRATA if name != "all" and flags.get(f"is_{name}", False)]
        for stratum in active:
            out = dict(row)
            out["stratum"] = stratum
            out_rows.append(out)
    return out_rows


@app.function(image=cpu_image, cpu=8, memory=32768, timeout=2 * 60 * 60, volumes=MOUNTED_VOLUMES)
def compute_action_strata(records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    data_volume.reload()
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        window_id = str(record.get("window_id", ""))
        relpath = str(record.get("frame_action_relpath", ""))
        if window_id and relpath:
            unique[window_id] = record

    rows: list[dict[str, Any]] = []
    for window_id, record in sorted(unique.items()):
        npz_path = DATA_ROOT / str(record["frame_action_relpath"])
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path)
        actions = np.asarray(data["actions"], dtype=np.float32)
        future = actions[49:121]
        speed = future[:, 0]
        yaw_rate = future[:, 1]
        accel_x = future[:, 8]
        future_y = future[:, 17]
        speed_delta = float(np.mean(speed[-24:]) - np.mean(speed[:24]))
        mean_yaw_rate = float(np.mean(yaw_rate))
        mean_accel_x = float(np.mean(accel_x))
        future_y_displacement = float(np.mean(future_y))
        combined_score = abs(speed_delta) + abs(mean_yaw_rate) + abs(mean_accel_x)
        rows.append(
            {
                "window_id": window_id,
                "scenario_id": record.get("scenario_id", ""),
                "window_idx": record.get("window_idx", ""),
                "frame_action_relpath": record["frame_action_relpath"],
                "speed_delta": speed_delta,
                "mean_yaw_rate": mean_yaw_rate,
                "mean_abs_yaw_rate": abs(mean_yaw_rate),
                "mean_accel_x": mean_accel_x,
                "future_y_displacement": future_y_displacement,
                "abs_future_y_displacement": abs(future_y_displacement),
                "combined_action_score": combined_score,
            }
        )

    thresholds = {
        "combined_action_score_p30": percentile([float(row["combined_action_score"]) for row in rows], 0.30),
        "speed_delta_p20": percentile([float(row["speed_delta"]) for row in rows], 0.20),
        "speed_delta_p80": percentile([float(row["speed_delta"]) for row in rows], 0.80),
        "mean_accel_x_p20": percentile([float(row["mean_accel_x"]) for row in rows], 0.20),
        "mean_accel_x_p80": percentile([float(row["mean_accel_x"]) for row in rows], 0.80),
        "mean_abs_yaw_rate_p80": percentile([float(row["mean_abs_yaw_rate"]) for row in rows], 0.80),
        "abs_future_y_displacement_p80": percentile([float(row["abs_future_y_displacement"]) for row in rows], 0.80),
    }

    for row in rows:
        is_low = float(row["combined_action_score"]) <= thresholds["combined_action_score_p30"]
        is_brake = (
            float(row["speed_delta"]) <= thresholds["speed_delta_p20"]
            or float(row["mean_accel_x"]) <= thresholds["mean_accel_x_p20"]
        )
        is_accel = (
            float(row["speed_delta"]) >= thresholds["speed_delta_p80"]
            or float(row["mean_accel_x"]) >= thresholds["mean_accel_x_p80"]
        )
        is_turn = (
            float(row["mean_abs_yaw_rate"]) >= thresholds["mean_abs_yaw_rate_p80"]
            or float(row["abs_future_y_displacement"]) >= thresholds["abs_future_y_displacement_p80"]
        )
        row["is_low_action"] = is_low
        row["is_brake"] = is_brake
        row["is_accelerate"] = is_accel
        row["is_turning"] = is_turn
        row["is_high_action"] = bool(is_brake or is_accel or is_turn)
    return {"thresholds": thresholds, "rows": rows}


def remote_decode_helpers() -> str:
    # Kept as a separate string-free helper marker so remote functions below stay self-contained.
    return ""


@app.function(
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
    max_containers=100,
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

    def decode_video_frames(path: Path, frame_count: int) -> np.ndarray:
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
        lap = -4.0 * center + gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1] + gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:]
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

    def low_frequency_frames(frames: np.ndarray, stride: int = 8) -> np.ndarray:
        return frames[:, ::stride, ::stride, :].astype(np.float32)

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
    final_action_volume.reload()

    rows: list[dict[str, Any]] = []
    reference_cache: dict[str, Any] = {}
    for record in records:
        generated_path = volume_root(record["generated_volume_key"]) / record["generated_video_relpath"]
        reference_relpath = record["source_relpath"]
        reference_path = DATA_ROOT / reference_relpath
        generated = decode_video_frames(generated_path, total_frames)
        if reference_relpath not in reference_cache:
            reference_cache[reference_relpath] = decode_video_frames(reference_path, total_frames)
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
            "counterfactual_action_mode": record.get("counterfactual_action_mode", "correct"),
            "scene_token": record["scene_token"],
            "scenario_id": record.get("scenario_id", ""),
            "window_id": record.get("window_id", ""),
            "window_idx": record.get("window_idx", ""),
            "generated_video_relpath": record["generated_video_relpath"],
            "source_relpath": reference_relpath,
            "generated_volume_key": record["generated_volume_key"],
            "fps": fps,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
            "seed": record.get("seed", ""),
            "lora_step": record.get("lora_step", ""),
            "image_cond_noise_scale": record.get("image_cond_noise_scale", ""),
            "frame_action_feature_key": record.get("frame_action_feature_key", ""),
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
    image=cpu_image,
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
    max_containers=100,
)
def compute_alignment_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    import subprocess

    import cv2
    import imageio_ffmpeg
    import numpy as np

    groups = payload["groups"]
    fps = int(payload["fps"])
    width = int(payload["width"])
    height = int(payload["height"])
    context_frames = int(payload["context_frames"])
    total_frames = int(payload["total_frames"])

    def decode_video_frames(path: Path, frame_count: int) -> np.ndarray:
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

    def temporal_delta_error(generated: np.ndarray, reference: np.ndarray) -> float:
        if len(generated) < 2 or len(reference) < 2:
            return 0.0
        gen_d = np.diff(generated.astype(np.float32), axis=0)
        ref_d = np.diff(reference.astype(np.float32), axis=0)
        return float(np.mean(np.abs(gen_d - ref_d)))

    def temporal_delta_mae(frames: np.ndarray) -> float:
        if len(frames) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(frames.astype(np.float32), axis=0))))

    def low_frequency_frames(frames: np.ndarray, stride: int = 8) -> np.ndarray:
        return frames[:, ::stride, ::stride, :].astype(np.float32)

    def optical_flow_summary(frames: np.ndarray) -> dict[str, float]:
        # Secondary diagnostic only: downsample and sparsely sample the future.
        sampled = frames[::6]
        if len(sampled) < 2:
            return {"flow_magnitude": float("nan"), "flow_x": float("nan"), "flow_y": float("nan")}
        gray = rgb_to_gray(sampled).astype(np.uint8)
        mags: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        for idx in range(len(gray) - 1):
            prev = cv2.resize(gray[idx], (128, 128), interpolation=cv2.INTER_AREA)
            nxt = cv2.resize(gray[idx + 1], (128, 128), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(np.mean(mag)))
            xs.append(float(np.mean(flow[..., 0])))
            ys.append(float(np.mean(flow[..., 1])))
        return {"flow_magnitude": mean(mags), "flow_x": mean(xs), "flow_y": mean(ys)}

    def quality(frames: np.ndarray, ref: np.ndarray) -> dict[str, float]:
        value_mse = mse(frames, ref)
        gen_temporal = temporal_delta_mae(frames)
        ref_temporal = temporal_delta_mae(ref)
        gen_lf = temporal_delta_mae(low_frequency_frames(frames))
        ref_lf = temporal_delta_mae(low_frequency_frames(ref))
        return {
            "future_psnr": psnr_from_mse(value_mse),
            "future_global_ssim": global_ssim(frames, ref),
            "temporal_delta_error_mae": temporal_delta_error(frames, ref),
            "low_frequency_temporal_delta_error_mae": temporal_delta_error(low_frequency_frames(frames), low_frequency_frames(ref)),
            "motion_ratio_generated_over_reference": gen_temporal / max(ref_temporal, 1e-12),
            "low_frequency_motion_ratio_generated_over_reference": gen_lf / max(ref_lf, 1e-12),
        }

    data_volume.reload()
    noaction_volume.reload()
    final_action_volume.reload()

    out_rows: list[dict[str, Any]] = []
    reference_cache: dict[str, Any] = {}
    for group in groups:
        by_mode = {record["counterfactual_action_mode"]: record for record in group["records"]}
        if "correct" not in by_mode or not all(mode in by_mode for mode in WRONG_MODES):
            continue
        reference_relpath = by_mode["correct"]["source_relpath"]
        if reference_relpath not in reference_cache:
            reference_cache[reference_relpath] = decode_video_frames(DATA_ROOT / reference_relpath, total_frames)[context_frames:]
        reference_future = reference_cache[reference_relpath]

        futures: dict[str, np.ndarray] = {}
        qualities: dict[str, dict[str, float]] = {}
        flows: dict[str, dict[str, float]] = {}
        for mode, record in by_mode.items():
            generated = decode_video_frames(volume_root(record["generated_volume_key"]) / record["generated_video_relpath"], total_frames)
            future = generated[context_frames:]
            futures[mode] = future
            qualities[mode] = quality(future, reference_future)
            flows[mode] = optical_flow_summary(future)
        gt_flow = optical_flow_summary(reference_future)

        correct_future = futures["correct"]
        s_rgb_by_mode = {mode: mae(correct_future, futures[mode]) for mode in WRONG_MODES}
        s_delta_by_mode = {mode: temporal_delta_error(correct_future, futures[mode]) for mode in WRONG_MODES}
        correct_q = qualities["correct"]
        wrong_qs = [qualities[mode] for mode in WRONG_MODES]
        flow_correct = flows["correct"]
        flow_wrong = [flows[mode] for mode in WRONG_MODES]

        row: dict[str, Any] = {
            "model_key": by_mode["correct"]["model_key"],
            "model_label": by_mode["correct"].get("model_label", by_mode["correct"]["model_key"]),
            "checkpoint_step": by_mode["correct"].get("checkpoint_step", ""),
            "scene_token": by_mode["correct"].get("scene_token", ""),
            "scenario_id": by_mode["correct"].get("scenario_id", ""),
            "window_id": by_mode["correct"].get("window_id", ""),
            "window_idx": by_mode["correct"].get("window_idx", ""),
            "source_relpath": reference_relpath,
            "S_rgb": mean(list(s_rgb_by_mode.values())),
            "S_delta": mean(list(s_delta_by_mode.values())),
            "S_rgb_correct_vs_zero": s_rgb_by_mode["zero"],
            "S_rgb_correct_vs_shuffled": s_rgb_by_mode["shuffled"],
            "S_rgb_correct_vs_reversed_future": s_rgb_by_mode["reversed_future"],
            "S_delta_correct_vs_zero": s_delta_by_mode["zero"],
            "S_delta_correct_vs_shuffled": s_delta_by_mode["shuffled"],
            "S_delta_correct_vs_reversed_future": s_delta_by_mode["reversed_future"],
            "delta_psnr": correct_q["future_psnr"] - mean([q["future_psnr"] for q in wrong_qs]),
            "delta_ssim": correct_q["future_global_ssim"] - mean([q["future_global_ssim"] for q in wrong_qs]),
            "delta_temporal_error": mean([q["temporal_delta_error_mae"] for q in wrong_qs]) - correct_q["temporal_delta_error_mae"],
            "delta_lowfreq_temporal_error": mean([q["low_frequency_temporal_delta_error_mae"] for q in wrong_qs])
            - correct_q["low_frequency_temporal_delta_error_mae"],
            "delta_motion_ratio_error": mean([abs(q["motion_ratio_generated_over_reference"] - 1.0) for q in wrong_qs])
            - abs(correct_q["motion_ratio_generated_over_reference"] - 1.0),
            "delta_lowfreq_motion_ratio_error": mean(
                [abs(q["low_frequency_motion_ratio_generated_over_reference"] - 1.0) for q in wrong_qs]
            )
            - abs(correct_q["low_frequency_motion_ratio_generated_over_reference"] - 1.0),
            "flow_magnitude_error_correct": abs(flow_correct["flow_magnitude"] - gt_flow["flow_magnitude"]),
            "flow_x_error_correct": abs(flow_correct["flow_x"] - gt_flow["flow_x"]),
            "flow_y_error_correct": abs(flow_correct["flow_y"] - gt_flow["flow_y"]),
            "flow_magnitude_error_wrong_mean": mean([abs(flow["flow_magnitude"] - gt_flow["flow_magnitude"]) for flow in flow_wrong]),
            "flow_x_error_wrong_mean": mean([abs(flow["flow_x"] - gt_flow["flow_x"]) for flow in flow_wrong]),
            "flow_y_error_wrong_mean": mean([abs(flow["flow_y"] - gt_flow["flow_y"]) for flow in flow_wrong]),
        }
        row["delta_flow_magnitude_error"] = row["flow_magnitude_error_wrong_mean"] - row["flow_magnitude_error_correct"]
        row["delta_flow_x_error"] = row["flow_x_error_wrong_mean"] - row["flow_x_error_correct"]
        row["delta_flow_y_error"] = row["flow_y_error_wrong_mean"] - row["flow_y_error_correct"]
        out_rows.append(row)
    return out_rows


@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    memory=49152,
    timeout=4 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
    max_containers=100,
)
def compute_fvd_for_group(payload: dict[str, Any]) -> dict[str, Any]:
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
        batch = torch.nn.functional.interpolate(batch, size=(fvd_num_frames, fvd_size, fvd_size), mode="trilinear", align_corners=False)
        mean_tensor = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
        std_tensor = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
        return (batch - mean_tensor) / std_tensor

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
    final_action_volume.reload()

    global _FVD_FEATURE_MODEL_CACHE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = str(device)
    try:
        cached = _FVD_FEATURE_MODEL_CACHE
    except NameError:
        cached = {}
        _FVD_FEATURE_MODEL_CACHE = cached
    if cache_key not in cached:
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()
        cached[cache_key] = model.to(device).eval()
    model = cached[cache_key]

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
    return {
        "model_key": payload["model_key"],
        "model_label": payload.get("model_label", payload["model_key"]),
        "counterfactual_action_mode": payload["counterfactual_action_mode"],
        "checkpoint_step": payload.get("checkpoint_step", ""),
        "stratum": payload.get("stratum", "all"),
        "fvd_future": frechet_distance(generated_features, reference_features),
        "fvd_backend": "torchvision_r3d18_kinetics400",
        "fvd_num_videos": len(records),
        "fvd_num_frames": fvd_num_frames,
        "fvd_size": fvd_size,
        "fvd_feature_dim": int(generated_features.shape[1]),
        "fvd_device": str(device),
    }


@app.function(
    image=gpu_image,
    gpu="A10G",
    cpu=8,
    memory=49152,
    timeout=2 * 60 * 60,
    volumes=MOUNTED_VOLUMES,
    max_containers=100,
)
def compute_fvd_feature_shard(payload: dict[str, Any]) -> dict[str, Any]:
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
        batch = torch.nn.functional.interpolate(batch, size=(fvd_num_frames, fvd_size, fvd_size), mode="trilinear", align_corners=False)
        mean_tensor = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
        std_tensor = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
        return (batch - mean_tensor) / std_tensor

    def extract_features(videos: list[np.ndarray]) -> np.ndarray:
        features = []
        with torch.no_grad():
            for start in range(0, len(videos), batch_size):
                batch = preprocess_batch(videos[start : start + batch_size])
                output = model(batch).flatten(start_dim=1).detach().float().cpu().numpy()
                features.append(output)
        return np.concatenate(features, axis=0).astype(np.float32)

    data_volume.reload()
    noaction_volume.reload()
    final_action_volume.reload()

    global _FVD_FEATURE_SHARD_MODEL_CACHE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = str(device)
    try:
        cached = _FVD_FEATURE_SHARD_MODEL_CACHE
    except NameError:
        cached = {}
        _FVD_FEATURE_SHARD_MODEL_CACHE = cached
    if cache_key not in cached:
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()
        cached[cache_key] = model.to(device).eval()
    model = cached[cache_key]

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

    return {
        "output_kind": payload["output_kind"],
        "group_id": payload["group_id"],
        "model_key": payload["model_key"],
        "model_label": payload.get("model_label", payload["model_key"]),
        "counterfactual_action_mode": payload["counterfactual_action_mode"],
        "checkpoint_step": payload.get("checkpoint_step", ""),
        "stratum": payload.get("stratum", "all"),
        "num_records": len(records),
        "fvd_num_frames": fvd_num_frames,
        "fvd_size": fvd_size,
        "fvd_feature_dim": 512,
        "fvd_device": str(device),
        "generated_features": extract_features(generated_futures),
        "reference_features": extract_features(reference_futures),
    }


def build_alignment_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("use_frame_actions", False):
            continue
        grouped[(str(record["model_key"]), str(record["window_id"]))].append(record)
    groups = []
    for key, group_records in sorted(grouped.items()):
        modes = {record["counterfactual_action_mode"] for record in group_records}
        if {"correct", "zero", "shuffled", "reversed_future"}.issubset(modes):
            groups.append({"group_key": "|".join(key), "records": group_records})
    return groups


def build_fvd_payloads(
    records: list[dict[str, Any]],
    strata_by_window: dict[str, dict[str, Any]],
    *,
    fps: int,
    width: int,
    height: int,
    context_frames: int,
    total_frames: int,
    fvd_num_frames: int,
    fvd_size: int,
    fvd_batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    stratum_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["model_key"]), str(record.get("counterfactual_action_mode", "correct")))
        all_grouped[key].append(record)
        flags = strata_by_window.get(str(record.get("window_id", "")), {})
        # Stratum FVD is most useful for the generated operating point, not for
        # every counterfactual. Keeping this to correct mode avoids 100+ large
        # redundant FVD jobs in the full-val run.
        if key[1] == "correct":
            active = ["all"] + [name for name in STRATA if name != "all" and flags.get(f"is_{name}", False)]
            for stratum in active:
                stratum_grouped[(key[0], key[1], stratum)].append(record)

    def payload_base(group_records: list[dict[str, Any]]) -> dict[str, Any]:
        first = group_records[0]
        return {
            "records": group_records,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "total_frames": total_frames,
            "fvd_num_frames": fvd_num_frames,
            "fvd_size": fvd_size,
            "batch_size": fvd_batch_size,
            "model_key": first["model_key"],
            "model_label": first.get("model_label", first["model_key"]),
            "counterfactual_action_mode": first.get("counterfactual_action_mode", "correct"),
            "checkpoint_step": first.get("checkpoint_step", ""),
        }

    fvd_summary_payloads = [payload_base(group_records) for _, group_records in sorted(all_grouped.items())]
    fvd_by_stratum_payloads = []
    for (_, _, stratum), group_records in sorted(stratum_grouped.items()):
        if len(group_records) < 2:
            continue
        payload = payload_base(group_records)
        payload["stratum"] = stratum
        fvd_by_stratum_payloads.append(payload)
    return fvd_summary_payloads, fvd_by_stratum_payloads


def build_fvd_feature_payloads(
    group_payloads: list[dict[str, Any]],
    *,
    output_kind: str,
    fvd_feature_chunk_size: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for group_idx, group_payload in enumerate(group_payloads):
        records = group_payload["records"]
        stratum = group_payload.get("stratum", "all")
        group_id = "|".join(
            [
                output_kind,
                str(group_payload["model_key"]),
                str(group_payload["counterfactual_action_mode"]),
                str(stratum),
                str(group_idx),
            ]
        )
        for shard_idx, shard in enumerate(chunked(records, fvd_feature_chunk_size)):
            payload = dict(group_payload)
            payload["records"] = shard
            payload["output_kind"] = output_kind
            payload["group_id"] = group_id
            payload["feature_shard_idx"] = shard_idx
            payloads.append(payload)
    return payloads


def fvd_from_features(features_a: Any, features_b: Any) -> float:
    import numpy as np

    def symmetric_matrix_sqrt(matrix: Any) -> Any:
        matrix = (matrix + matrix.T) * 0.5
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 0.0, None)
        return (vectors * np.sqrt(values)[None, :]) @ vectors.T

    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
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


def aggregate_fvd_feature_shards(feature_shards: list[dict[str, Any]], *, output_kind: str) -> list[dict[str, Any]]:
    import numpy as np

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in feature_shards:
        if shard["output_kind"] == output_kind:
            grouped[str(shard["group_id"])].append(shard)

    rows: list[dict[str, Any]] = []
    for _, shards in sorted(grouped.items()):
        first = shards[0]
        generated = np.concatenate([np.asarray(shard["generated_features"], dtype=np.float32) for shard in shards], axis=0)
        reference = np.concatenate([np.asarray(shard["reference_features"], dtype=np.float32) for shard in shards], axis=0)
        rows.append(
            {
                "model_key": first["model_key"],
                "model_label": first.get("model_label", first["model_key"]),
                "counterfactual_action_mode": first["counterfactual_action_mode"],
                "checkpoint_step": first.get("checkpoint_step", ""),
                "stratum": first.get("stratum", "all"),
                "fvd_future": fvd_from_features(generated, reference),
                "fvd_backend": "torchvision_r3d18_kinetics400",
                "fvd_num_videos": int(generated.shape[0]),
                "fvd_num_frames": first["fvd_num_frames"],
                "fvd_size": first["fvd_size"],
                "fvd_feature_dim": int(generated.shape[1]),
                "fvd_device": "feature_sharded_a10g",
                "num_feature_shards": len(shards),
            }
        )
    return rows


def write_plots(output_root: Path) -> None:
    import matplotlib.pyplot as plt

    model_mode = read_csv(output_root / "model_mode_summary.csv")
    action_summary = read_csv(output_root / "correct_vs_wrong_advantage_summary.csv")
    fvd_rows = read_csv(output_root / "fvd_summary.csv")

    def label(row: dict[str, str]) -> str:
        return f"{row['model_key']}\\n{row.get('counterfactual_action_mode', '')}".strip()

    correct_rows = [row for row in model_mode if row.get("counterfactual_action_mode", "correct") == "correct"]
    xs = list(range(len(correct_rows)))
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 5))
    plt.bar(xs, [float(row["mean_future_psnr"]) for row in correct_rows], label="PSNR")
    plt.xticks(xs, [row["model_key"] for row in correct_rows], rotation=30, ha="right")
    plt.ylabel("PSNR (higher better)")
    plt.tight_layout()
    plt.savefig(plot_dir / "correct_psnr_by_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 5))
    width = 0.35
    plt.bar([x - width / 2 for x in xs], [float(row["mean_sharpness_ratio_generated_over_reference"]) for row in correct_rows], width, label="sharpness ratio")
    plt.bar([x + width / 2 for x in xs], [float(row["mean_motion_ratio_generated_over_reference"]) for row in correct_rows], width, label="motion ratio")
    plt.xticks(xs, [row["model_key"] for row in correct_rows], rotation=30, ha="right")
    plt.ylabel("ratio to reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "sharpness_motion_by_model.png", dpi=180)
    plt.close()

    high = [row for row in action_summary if row.get("stratum") == "high_action"]
    all_rows = [row for row in action_summary if row.get("stratum") == "all"]
    for rows, name in [(all_rows, "all"), (high, "high_action")]:
        if not rows:
            continue
        xvals = list(range(len(rows)))
        plt.figure(figsize=(13, 5))
        plt.bar(xvals, [float(row["mean_delta_psnr"]) for row in rows], label="Delta PSNR")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(xvals, [row["model_key"] for row in rows], rotation=30, ha="right")
        plt.ylabel("correct - wrong PSNR")
        plt.tight_layout()
        plt.savefig(plot_dir / f"correct_vs_wrong_delta_psnr_{name}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(13, 5))
        plt.bar(xvals, [float(row["mean_delta_temporal_error"]) for row in rows], label="Delta temporal error")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(xvals, [row["model_key"] for row in rows], rotation=30, ha="right")
        plt.ylabel("wrong error - correct error")
        plt.tight_layout()
        plt.savefig(plot_dir / f"correct_vs_wrong_delta_temporal_error_{name}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(13, 5))
        plt.bar(xvals, [float(row["mean_S_rgb"]) for row in rows])
        plt.xticks(xvals, [row["model_key"] for row in rows], rotation=30, ha="right")
        plt.ylabel("S_rgb")
        plt.tight_layout()
        plt.savefig(plot_dir / f"action_sensitivity_{name}.png", dpi=180)
        plt.close()

    fvd_correct = [row for row in fvd_rows if row.get("counterfactual_action_mode") == "correct"]
    if fvd_correct:
        xvals = list(range(len(fvd_correct)))
        plt.figure(figsize=(13, 5))
        plt.bar(xvals, [float(row["fvd_future"]) for row in fvd_correct])
        plt.xticks(xvals, [row["model_key"] for row in fvd_correct], rotation=30, ha="right")
        plt.ylabel("FVD-style future distance (lower better)")
        plt.tight_layout()
        plt.savefig(plot_dir / "fvd_by_model_correct.png", dpi=180)
        plt.close()

    # Required aliases with clearer names.
    alias_map = {
        "correct_vs_wrong_delta_psnr_high_action.png": "correct_vs_wrong_delta_psnr_by_stratum.png",
        "correct_vs_wrong_delta_temporal_error_high_action.png": "correct_vs_wrong_delta_temporal_error_by_stratum.png",
        "action_sensitivity_high_action.png": "action_sensitivity_by_stratum.png",
        "fvd_by_model_correct.png": "fvd_by_model_mode.png",
        "sharpness_motion_by_model.png": "high_action_summary_barplot.png",
    }
    for src, dst in alias_map.items():
        src_path = plot_dir / src
        if src_path.exists():
            (plot_dir / dst).write_bytes(src_path.read_bytes())

    # Pareto: quality vs action alignment.
    if high and fvd_correct:
        fvd_by_model = {row["model_key"]: float(row["fvd_future"]) for row in fvd_correct}
        plt.figure(figsize=(8, 6))
        for row in high:
            model = row["model_key"]
            if model in fvd_by_model:
                plt.scatter(float(row["mean_S_rgb"]), fvd_by_model[model])
                plt.text(float(row["mean_S_rgb"]), fvd_by_model[model], model, fontsize=8)
        plt.xlabel("High-action S_rgb")
        plt.ylabel("FVD-style future distance")
        plt.tight_layout()
        plt.savefig(plot_dir / "quality_vs_action_alignment_pareto.png", dpi=180)
        plt.close()

        correct_quality = {row["model_key"]: row for row in correct_rows}
        plt.figure(figsize=(8, 6))
        for row in high:
            model = row["model_key"]
            if model in correct_quality:
                plt.scatter(float(row["mean_delta_temporal_error"]), float(correct_quality[model]["mean_sharpness_ratio_generated_over_reference"]))
                plt.text(
                    float(row["mean_delta_temporal_error"]),
                    float(correct_quality[model]["mean_sharpness_ratio_generated_over_reference"]),
                    model,
                    fontsize=8,
                )
        plt.xlabel("High-action delta temporal error (wrong - correct)")
        plt.ylabel("Sharpness ratio")
        plt.tight_layout()
        plt.savefig(plot_dir / "final_budget_pareto_fvd_vs_sharpness.png", dpi=180)
        plt.close()

    # SSIM plot alias required by plan.
    if all_rows:
        xvals = list(range(len(all_rows)))
        plt.figure(figsize=(13, 5))
        plt.bar(xvals, [float(row["mean_delta_ssim"]) for row in all_rows])
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(xvals, [row["model_key"] for row in all_rows], rotation=30, ha="right")
        plt.ylabel("correct - wrong SSIM")
        plt.tight_layout()
        plt.savefig(plot_dir / "correct_vs_wrong_delta_ssim_by_stratum.png", dpi=180)
        plt.close()


def write_markdown_report(output_root: Path) -> None:
    action_summary = read_csv(output_root / "correct_vs_wrong_advantage_summary.csv")
    model_summary = read_csv(output_root / "model_mode_summary.csv")
    fvd_rows = read_csv(output_root / "fvd_summary.csv")

    high = [row for row in action_summary if row.get("stratum") == "high_action"]
    all_rows = [row for row in action_summary if row.get("stratum") == "all"]
    correct_quality = [row for row in model_summary if row.get("counterfactual_action_mode") == "correct"]
    fvd_correct = [row for row in fvd_rows if row.get("counterfactual_action_mode") == "correct"]

    lines = [
        "# Final Action-Alignment Validation Report",
        "",
        f"Generated at `{utc_now()}`.",
        "",
        "This campaign evaluates whether correct actions beat wrong actions on the full validation set, not just whether the model changes under action perturbation.",
        "",
        "## Correct-vs-Wrong Advantage",
        "",
        "| Stratum | Model | clips | S_rgb | delta PSNR | delta SSIM | delta temporal error | delta LF temporal error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows + high:
        lines.append(
            "| {stratum} | {model_key} | {num_clips} | {s:.3f} | {dpsnr:.4f} | {dssim:.5f} | {dterr:.4f} | {dlfterr:.4f} |".format(
                stratum=row["stratum"],
                model_key=row["model_key"],
                num_clips=row["num_clips"],
                s=float(row["mean_S_rgb"]),
                dpsnr=float(row["mean_delta_psnr"]),
                dssim=float(row["mean_delta_ssim"]),
                dterr=float(row["mean_delta_temporal_error"]),
                dlfterr=float(row["mean_delta_lowfreq_temporal_error"]),
            )
        )
    lines.extend(
        [
            "",
            "Positive delta PSNR/SSIM means correct actions are closer to ground truth than wrong actions. Positive delta temporal error means wrong actions have larger temporal error than correct actions.",
            "",
            "## Correct-Mode Visual Quality",
            "",
            "| Model | clips | PSNR | SSIM | sharpness | motion | FVD-style |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    fvd_by_model = {row["model_key"]: row for row in fvd_correct}
    for row in correct_quality:
        fvd = fvd_by_model.get(row["model_key"], {})
        fvd_value = float(fvd.get("fvd_future", "nan")) if fvd else float("nan")
        lines.append(
            "| {model_key} | {num_clips} | {psnr:.3f} | {ssim:.3f} | {sharp:.3f} | {motion:.3f} | {fvd:.3f} |".format(
                model_key=row["model_key"],
                num_clips=row["num_clips"],
                psnr=float(row["mean_future_psnr"]),
                ssim=float(row["mean_future_global_ssim"]),
                sharp=float(row["mean_sharpness_ratio_generated_over_reference"]),
                motion=float(row["mean_motion_ratio_generated_over_reference"]),
                fvd=fvd_value,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If high-action delta metrics are positive for V4, we have semantic action-alignment evidence.",
            "- If S_rgb is high but delta metrics are not positive, the model is action-sensitive but not proven semantically correct.",
            "- If no-action has stronger PSNR/SSIM but V4 has positive high-action advantage, average-case fidelity metrics understate controllability.",
        ]
    )
    (output_root / "final_action_alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    quality_chunk_size: int = 32,
    alignment_chunk_size: int = 12,
    fvd_num_frames: int = 16,
    fvd_size: int = 112,
    fvd_batch_size: int = 8,
    fvd_feature_chunk_size: int = 32,
) -> None:
    manifest_path = Path(manifest)
    output_root = Path(output_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in {manifest_path}")
    if context_frames + future_frames != total_frames:
        raise ValueError("context_frames + future_frames must equal total_frames.")

    output_root.mkdir(parents=True, exist_ok=True)

    strata = compute_action_strata.remote(records)
    strata_rows = strata["rows"]
    thresholds = strata["thresholds"]
    write_csv(output_root / "action_strata_manifest.csv", strata_rows)
    (output_root / "action_strata_thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    strata_by_window = {str(row["window_id"]): row for row in strata_rows}

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
        for shard in chunked(records, quality_chunk_size)
    ]
    quality_shards = list(compute_quality_shard.map(quality_payloads))
    quality_rows = [row for shard in quality_shards for row in shard]
    quality_rows.sort(
        key=lambda row: (
            str(row["model_key"]),
            str(row["counterfactual_action_mode"]),
            int(row.get("window_idx", -1)),
            str(row.get("window_id", "")),
        )
    )
    write_csv(output_root / "per_clip_mode_metrics.csv", quality_rows)

    model_summary = summarize(quality_rows, ["model_key", "counterfactual_action_mode"], QUALITY_METRICS)
    write_csv(output_root / "model_mode_summary.csv", model_summary)

    stratum_quality_rows = attach_strata(quality_rows, strata_by_window)
    stratum_summary = summarize(stratum_quality_rows, ["stratum", "model_key", "counterfactual_action_mode"], QUALITY_METRICS)
    write_csv(output_root / "stratum_summary.csv", stratum_summary)

    alignment_groups = build_alignment_groups(records)
    alignment_payloads = [
        {
            "groups": shard,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "future_frames": future_frames,
            "total_frames": total_frames,
        }
        for shard in chunked(alignment_groups, alignment_chunk_size)
    ]
    alignment_shards = list(compute_alignment_shard.map(alignment_payloads))
    alignment_rows = [row for shard in alignment_shards for row in shard]
    alignment_rows.sort(key=lambda row: (str(row["model_key"]), int(row.get("window_idx", -1)), str(row.get("window_id", ""))))
    write_csv(output_root / "per_clip_action_alignment.csv", alignment_rows)

    stratum_alignment_rows = attach_strata(alignment_rows, strata_by_window)
    alignment_metrics = [
        "S_rgb",
        "S_delta",
        "delta_psnr",
        "delta_ssim",
        "delta_temporal_error",
        "delta_lowfreq_temporal_error",
        "delta_motion_ratio_error",
        "delta_lowfreq_motion_ratio_error",
        "delta_flow_magnitude_error",
        "delta_flow_x_error",
        "delta_flow_y_error",
    ]
    advantage_summary = summarize(stratum_alignment_rows, ["stratum", "model_key"], alignment_metrics)
    write_csv(output_root / "correct_vs_wrong_advantage_summary.csv", advantage_summary)

    fvd_payloads, fvd_by_stratum_payloads = build_fvd_payloads(
        records,
        strata_by_window,
        fps=fps,
        width=width,
        height=height,
        context_frames=context_frames,
        total_frames=total_frames,
        fvd_num_frames=fvd_num_frames,
        fvd_size=fvd_size,
        fvd_batch_size=fvd_batch_size,
    )
    fvd_feature_payloads = build_fvd_feature_payloads(
        fvd_payloads,
        output_kind="summary",
        fvd_feature_chunk_size=fvd_feature_chunk_size,
    )
    fvd_feature_payloads.extend(
        build_fvd_feature_payloads(
            fvd_by_stratum_payloads,
            output_kind="stratum",
            fvd_feature_chunk_size=fvd_feature_chunk_size,
        )
    )
    fvd_feature_shards = list(compute_fvd_feature_shard.map(fvd_feature_payloads))

    fvd_rows = aggregate_fvd_feature_shards(fvd_feature_shards, output_kind="summary")
    fvd_rows.sort(key=lambda row: (str(row["model_key"]), str(row["counterfactual_action_mode"])))
    write_csv(output_root / "fvd_summary.csv", fvd_rows)

    fvd_by_stratum_rows = aggregate_fvd_feature_shards(fvd_feature_shards, output_kind="stratum")
    fvd_by_stratum_rows.sort(key=lambda row: (str(row["stratum"]), str(row["model_key"]), str(row["counterfactual_action_mode"])))
    write_csv(output_root / "fvd_by_stratum.csv", fvd_by_stratum_rows)

    write_plots(output_root)
    write_markdown_report(output_root)

    report = {
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "num_records": len(records),
        "num_quality_shards": len(quality_payloads),
        "num_alignment_groups": len(alignment_groups),
        "num_alignment_shards": len(alignment_payloads),
        "num_fvd_groups": len(fvd_payloads),
        "num_fvd_stratum_groups": len(fvd_by_stratum_payloads),
        "num_fvd_feature_shards": len(fvd_feature_payloads),
        "quality_chunk_size": quality_chunk_size,
        "alignment_chunk_size": alignment_chunk_size,
        "fvd_num_frames": fvd_num_frames,
        "fvd_size": fvd_size,
        "fvd_batch_size": fvd_batch_size,
        "fvd_feature_chunk_size": fvd_feature_chunk_size,
        "note": "FVD-style uses torchvision R3D-18 Kinetics features over future frames; it is for relative comparison only.",
    }
    (output_root / "metrics_modal_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-chunk-size", type=int, default=32)
    parser.add_argument("--alignment-chunk-size", type=int, default=12)
    parser.add_argument("--fvd-num-frames", type=int, default=16)
    parser.add_argument("--fvd-size", type=int, default=112)
    parser.add_argument("--fvd-batch-size", type=int, default=8)
    parser.add_argument("--fvd-feature-chunk-size", type=int, default=32)
    args = parser.parse_args()
    main(
        manifest=args.manifest,
        output_dir=args.output_dir,
        quality_chunk_size=args.quality_chunk_size,
        alignment_chunk_size=args.alignment_chunk_size,
        fvd_num_frames=args.fvd_num_frames,
        fvd_size=args.fvd_size,
        fvd_batch_size=args.fvd_batch_size,
        fvd_feature_chunk_size=args.fvd_feature_chunk_size,
    )
