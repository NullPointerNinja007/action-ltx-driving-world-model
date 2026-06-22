from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import imageio_ffmpeg
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit("Missing dependency: imageio-ffmpeg. Install with `pip install imageio-ffmpeg`.") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit("Missing dependency: numpy. Install with `pip install numpy`.") from exc


DEFAULT_MANIFEST = Path(
    "data/distilled098_24fps_49ctx_72future_base_vs_lora_seed231_all5/"
    "manifest_24fps_49ctx_72future_base_vs_lora_seed231_all5.json"
)
DEFAULT_SOURCE_DIR = Path("data/inference_input_clips/interpolated_24fps_waymo_full20s")
DEFAULT_OUTPUT_DIR = Path("data/benchmarks/distilled098_24fps_49ctx_72future_seed231_all5")


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest_path: Path
    source_dir: Path
    output_dir: Path
    fps: int
    width: int
    height: int
    context_frames: int
    future_frames: int
    total_frames: int
    max_records: int
    compute_fvd: bool
    fvd_backend: str
    fvd_torchscript_path: Path | None
    fvd_device: str
    fvd_batch_size: int
    fvd_num_frames: int
    fvd_size: int


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark generated driving videos against the real future frames. "
            "Metrics are computed separately on context and future regions; the main scores use future frames only."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--context-frames", type=int, default=49)
    parser.add_argument("--future-frames", type=int, default=72)
    parser.add_argument("--total-frames", type=int, default=121)
    parser.add_argument("--max-records", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument(
        "--compute-fvd",
        action="store_true",
        help=(
            "Compute Frechet video distance over generated/reference future-frame feature distributions. "
            "Requires torch. The default backend uses torchvision R3D-18 Kinetics features; use "
            "--fvd-backend torchscript with --fvd-torchscript-path for a canonical I3D feature extractor."
        ),
    )
    parser.add_argument(
        "--fvd-backend",
        choices=("torchvision_r3d18", "torchscript"),
        default="torchvision_r3d18",
        help="Feature extractor backend for FVD. Lower FVD is better.",
    )
    parser.add_argument(
        "--fvd-torchscript-path",
        type=Path,
        default=None,
        help="TorchScript video feature extractor path for --fvd-backend torchscript.",
    )
    parser.add_argument("--fvd-device", default="auto", help="FVD device: auto, cuda, or cpu.")
    parser.add_argument("--fvd-batch-size", type=int, default=2)
    parser.add_argument("--fvd-num-frames", type=int, default=16)
    parser.add_argument("--fvd-size", type=int, default=112, help="Spatial resize used before FVD feature extraction.")
    args = parser.parse_args()

    if args.context_frames + args.future_frames != args.total_frames:
        raise ValueError("context_frames + future_frames must equal total_frames.")
    if args.compute_fvd and args.fvd_backend == "torchscript" and args.fvd_torchscript_path is None:
        raise ValueError("--fvd-torchscript-path is required when --fvd-backend=torchscript.")
    if args.fvd_batch_size <= 0:
        raise ValueError("--fvd-batch-size must be positive.")
    if args.fvd_num_frames <= 1:
        raise ValueError("--fvd-num-frames must be greater than 1.")
    if args.fvd_size <= 0:
        raise ValueError("--fvd-size must be positive.")
    return BenchmarkConfig(
        manifest_path=args.manifest,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        total_frames=args.total_frames,
        max_records=args.max_records,
        compute_fvd=args.compute_fvd,
        fvd_backend=args.fvd_backend,
        fvd_torchscript_path=args.fvd_torchscript_path,
        fvd_device=args.fvd_device,
        fvd_batch_size=args.fvd_batch_size,
        fvd_num_frames=args.fvd_num_frames,
        fvd_size=args.fvd_size,
    )


def decode_video_frames(path: Path, *, frame_count: int, fps: int, width: int, height: int) -> np.ndarray:
    """Decode the first `frame_count` frames as RGB uint8 using the bundled ffmpeg binary."""
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
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {path}:\n{stderr}")

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
    """Frame-averaged global grayscale SSIM. This is lighter than windowed SSIM."""
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
    """Mean grayscale FFT energy outside a radial low-frequency cutoff.

    The diagnostic is computed on every other frame after spatial downsampling.
    This keeps checkpoint sweeps practical while still measuring retained detail.
    """
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


def low_frequency_temporal_delta_error(generated: np.ndarray, reference: np.ndarray) -> float:
    return temporal_delta_error(low_frequency_frames(generated), low_frequency_frames(reference))


def downsample_for_copy_metric(frames: np.ndarray) -> np.ndarray:
    return frames[:, ::8, ::8, :].astype(np.float32)


def copy_leakage_metrics(generated_future: np.ndarray, reference_context: np.ndarray, reference_future: np.ndarray) -> dict[str, float]:
    gen_future = downsample_for_copy_metric(generated_future)
    ref_context = downsample_for_copy_metric(reference_context)
    ref_future = downsample_for_copy_metric(reference_future)

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


def compute_clip_metrics(record: dict[str, Any], cfg: BenchmarkConfig) -> dict[str, Any]:
    generated_path = Path(record["local_file"])
    source_path = cfg.source_dir / record["source_filename"]
    generated = decode_video_frames(
        generated_path,
        frame_count=cfg.total_frames,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
    )
    reference = decode_video_frames(
        source_path,
        frame_count=cfg.total_frames,
        fps=cfg.fps,
        width=cfg.width,
        height=cfg.height,
    )

    c = cfg.context_frames
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
        "generated_file": record["local_file"],
        "source_file": str(source_path),
        "fps": cfg.fps,
        "context_frames": cfg.context_frames,
        "future_frames": cfg.future_frames,
        "total_frames": cfg.total_frames,
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
        "low_frequency_temporal_delta_error_mae": low_frequency_temporal_delta_error(
            generated_future,
            reference_future,
        ),
        "context_to_future_boundary_mae_generated": boundary_generated,
        "context_to_future_boundary_mae_reference": boundary_reference,
        "boundary_mae_ratio_generated_over_reference": boundary_generated / max(boundary_reference, 1e-12),
    }
    metrics.update(copy_leakage_metrics(generated_future, reference_context, reference_future))
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def sample_video_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniformly sample a fixed-length clip from an RGB uint8 video array."""
    if len(frames) < 1:
        raise ValueError("Cannot sample frames from an empty video.")
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    return frames[indices]


def symmetric_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root for symmetric positive semidefinite matrices."""
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)[None, :]) @ vectors.T


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compute Frechet distance between two feature distributions.

    FVD uses this formula over video features. This implementation uses the symmetric
    PSD formulation, avoiding a scipy dependency for sqrtm.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Frechet inputs must be 2D arrays of shape [num_videos, feature_dim].")
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError(f"Feature dimensions differ: {features_a.shape[1]} vs {features_b.shape[1]}.")
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
    middle = sqrt_sigma_a @ sigma_b @ sqrt_sigma_a
    covmean = symmetric_matrix_sqrt(middle)
    value = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean)
    return float(max(value, 0.0))


class FvdFeatureExtractor:
    def __init__(self, cfg: BenchmarkConfig):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional FVD env.
            raise SystemExit(
                "FVD requires torch. Install torch/torchvision, or run without --compute-fvd. "
                "For canonical I3D FVD, pass --fvd-backend torchscript --fvd-torchscript-path <i3d_features.pt>."
            ) from exc

        self.torch = torch
        if cfg.fvd_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.fvd_device)
        self.backend = cfg.fvd_backend
        self.num_frames = cfg.fvd_num_frames
        self.size = cfg.fvd_size
        self.batch_size = cfg.fvd_batch_size
        self.model = self._load_model(cfg).to(self.device).eval()

    def _load_model(self, cfg: BenchmarkConfig):
        torch = self.torch
        if cfg.fvd_backend == "torchscript":
            if cfg.fvd_torchscript_path is None:
                raise ValueError("--fvd-torchscript-path is required for torchscript FVD.")
            return torch.jit.load(str(cfg.fvd_torchscript_path), map_location=self.device)

        try:
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError as exc:  # pragma: no cover - depends on optional FVD env.
            raise SystemExit(
                "The default FVD backend requires torchvision. Install torchvision, or use "
                "--fvd-backend torchscript with an I3D TorchScript feature extractor."
            ) from exc

        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        model.fc = torch.nn.Identity()
        return model

    def _preprocess_batch(self, videos: list[np.ndarray]):
        torch = self.torch
        tensors = []
        for frames in videos:
            sampled = sample_video_frames(frames, self.num_frames)
            tensor = torch.from_numpy(sampled).to(dtype=torch.float32) / 255.0
            tensor = tensor.permute(3, 0, 1, 2)  # C, T, H, W
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(self.device)
        batch = torch.nn.functional.interpolate(
            batch,
            size=(self.num_frames, self.size, self.size),
            mode="trilinear",
            align_corners=False,
        )
        if self.backend == "torchvision_r3d18":
            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
            batch = (batch - mean) / std
        return batch

    def extract(self, videos: list[np.ndarray]) -> np.ndarray:
        if not videos:
            return np.empty((0, 0), dtype=np.float32)
        torch = self.torch
        features: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(videos), self.batch_size):
                batch = self._preprocess_batch(videos[start : start + self.batch_size])
                output = self.model(batch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.flatten(start_dim=1).detach().float().cpu().numpy()
                features.append(output)
        return np.concatenate(features, axis=0)


def fvd_safe_model_key(model_mode: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_mode)


def compute_fvd_by_model(
    records: list[dict[str, Any]],
    cfg: BenchmarkConfig,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    extractor = FvdFeatureExtractor(cfg)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model_mode"])].append(record)

    fvd_rows: list[dict[str, Any]] = []
    feature_payload: dict[str, np.ndarray] = {}
    for model_mode, model_records in sorted(grouped.items()):
        generated_futures: list[np.ndarray] = []
        reference_futures: list[np.ndarray] = []
        for record in model_records:
            generated_path = Path(record["local_file"])
            source_path = cfg.source_dir / record["source_filename"]
            generated = decode_video_frames(
                generated_path,
                frame_count=cfg.total_frames,
                fps=cfg.fps,
                width=cfg.width,
                height=cfg.height,
            )
            reference = decode_video_frames(
                source_path,
                frame_count=cfg.total_frames,
                fps=cfg.fps,
                width=cfg.width,
                height=cfg.height,
            )
            generated_futures.append(generated[cfg.context_frames :])
            reference_futures.append(reference[cfg.context_frames :])

        generated_features = extractor.extract(generated_futures)
        reference_features = extractor.extract(reference_futures)
        fvd_value = frechet_distance(generated_features, reference_features)
        model_key = fvd_safe_model_key(model_mode)
        feature_payload[f"{model_key}_generated"] = generated_features
        feature_payload[f"{model_key}_reference"] = reference_features
        fvd_rows.append(
            {
                "model_mode": model_mode,
                "fvd_future": fvd_value,
                "fvd_backend": cfg.fvd_backend,
                "fvd_num_videos": len(model_records),
                "fvd_num_frames": cfg.fvd_num_frames,
                "fvd_size": cfg.fvd_size,
                "fvd_feature_dim": int(generated_features.shape[1]) if generated_features.ndim == 2 else 0,
                "fvd_device": str(extractor.device),
                "fvd_note": (
                    "Canonical FVD requires I3D features; torchvision_r3d18 is an FVD-style Kinetics feature "
                    "distance useful for internal comparisons only."
                    if cfg.fvd_backend == "torchvision_r3d18"
                    else "TorchScript feature extractor supplied by caller."
                ),
            }
        )

    feature_path = cfg.output_dir / "fvd_features.npz"
    np.savez_compressed(feature_path, **feature_payload)
    fvd_csv = cfg.output_dir / "fvd_summary.csv"
    write_csv(fvd_csv, fvd_rows)
    return fvd_rows, {
        "fvd_summary_csv": str(fvd_csv),
        "fvd_features_npz": str(feature_path),
    }


def merge_fvd_into_summary(summary_rows: list[dict[str, Any]], fvd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {str(row["model_mode"]): row for row in fvd_rows}
    fvd_keys = [
        "fvd_future",
        "fvd_backend",
        "fvd_num_videos",
        "fvd_num_frames",
        "fvd_size",
        "fvd_feature_dim",
        "fvd_device",
        "fvd_note",
    ]
    merged: list[dict[str, Any]] = []
    for summary in summary_rows:
        row = dict(summary)
        fvd = by_model.get(str(row["model_mode"]))
        for key in fvd_keys:
            row[key] = fvd.get(key, "") if fvd is not None else ""
        merged.append(row)
    return merged


def paired_model_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scene[str(row["scene_token"])][str(row["model_mode"])] = row

    comparisons: list[dict[str, Any]] = []
    for scene_token, scene_rows in sorted(by_scene.items()):
        if len(scene_rows) < 2:
            continue
        base = scene_rows.get("base_distilled_no_lora")
        lora_mode = next((mode for mode in sorted(scene_rows) if mode != "base_distilled_no_lora"), "")
        lora = scene_rows.get(lora_mode) if lora_mode else None
        if base is None or lora is None:
            continue
        comparisons.append(
            {
                "scene_token": scene_token,
                "lora_model_mode": lora_mode,
                "lora_minus_base_future_psnr": float(lora["future_psnr"]) - float(base["future_psnr"]),
                "lora_minus_base_future_global_ssim": float(lora["future_global_ssim"]) - float(base["future_global_ssim"]),
                "lora_minus_base_future_mse": float(lora["future_mse"]) - float(base["future_mse"]),
                "lora_minus_base_sharpness_ratio": float(lora["sharpness_ratio_generated_over_reference"])
                - float(base["sharpness_ratio_generated_over_reference"]),
                "lora_minus_base_temporal_delta_error_mae": float(lora["temporal_delta_error_mae"])
                - float(base["temporal_delta_error_mae"]),
                "winner_by_future_psnr": lora_mode
                if float(lora["future_psnr"]) > float(base["future_psnr"])
                else "base_distilled_no_lora",
                "winner_by_future_global_ssim": lora_mode
                if float(lora["future_global_ssim"]) > float(base["future_global_ssim"])
                else "base_distilled_no_lora",
                "winner_by_future_mse": lora_mode
                if float(lora["future_mse"]) < float(base["future_mse"])
                else "base_distilled_no_lora",
            }
        )
    return comparisons


def quality_gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {str(row["model_mode"]): row for row in summary_rows}
    base = by_model.get("base_distilled_no_lora")
    lora_mode = next((mode for mode in sorted(by_model) if mode != "base_distilled_no_lora"), "")
    lora = by_model.get(lora_mode) if lora_mode else None
    if base is None or lora is None:
        return {
            "status": "not_applicable",
            "reason": "Expected base_distilled_no_lora and at least one non-base model summary.",
        }

    base_sharpness = float(base["mean_sharpness_ratio_generated_over_reference"])
    lora_sharpness = float(lora["mean_sharpness_ratio_generated_over_reference"])
    base_motion = float(base["mean_motion_ratio_generated_over_reference"])
    lora_motion = float(lora["mean_motion_ratio_generated_over_reference"])
    base_ssim = float(base["mean_future_global_ssim"])
    lora_ssim = float(lora["mean_future_global_ssim"])

    sharpness_retention = lora_sharpness / max(base_sharpness, 1e-12)
    motion_retention = lora_motion / max(base_motion, 1e-12)
    ssim_delta = lora_ssim - base_ssim
    passes = sharpness_retention >= 0.8 and motion_retention >= 0.8 and ssim_delta >= -0.01
    return {
        "status": "pass" if passes else "fail",
        "criteria": {
            "lora_model_mode": lora_mode,
            "sharpness_retention_lora_over_base_must_be_at_least": 0.8,
            "motion_retention_lora_over_base_must_be_at_least": 0.8,
            "future_global_ssim_delta_lora_minus_base_must_be_at_least": -0.01,
        },
        "observed": {
            "sharpness_retention_lora_over_base": sharpness_retention,
            "motion_retention_lora_over_base": motion_retention,
            "future_global_ssim_delta_lora_minus_base": ssim_delta,
            "future_psnr_delta_lora_minus_base": float(lora["mean_future_psnr"])
            - float(base["mean_future_psnr"]),
        },
        "interpretation": (
            "LoRA passes only if it preserves sharpness and motion while not losing future SSIM. "
            "This prevents a blurry checkpoint from looking good only because PSNR improved."
        ),
    }


def load_manifest_records(path: Path, max_records: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a list at key `records`.")
    if max_records > 0:
        records = records[:max_records]
    return manifest, records


def validate_fvd_dependencies(cfg: BenchmarkConfig) -> None:
    if not cfg.compute_fvd:
        return
    try:
        __import__("torch")
    except ImportError as exc:
        raise SystemExit(
            "FVD requires torch. Install torch/torchvision, or run without --compute-fvd. "
            "For canonical I3D FVD, pass --fvd-backend torchscript --fvd-torchscript-path <i3d_features.pt>."
        ) from exc
    if cfg.fvd_backend == "torchvision_r3d18":
        try:
            __import__("torchvision")
        except ImportError as exc:
            raise SystemExit(
                "The default FVD backend requires torchvision. Install torchvision, or use "
                "--fvd-backend torchscript with an I3D TorchScript feature extractor."
            ) from exc
    if cfg.fvd_backend == "torchscript":
        if cfg.fvd_torchscript_path is None or not cfg.fvd_torchscript_path.exists():
            raise FileNotFoundError(f"Missing FVD TorchScript feature extractor: {cfg.fvd_torchscript_path}")


def main() -> None:
    cfg = parse_args()
    validate_fvd_dependencies(cfg)
    manifest, records = load_manifest_records(cfg.manifest_path, cfg.max_records)
    if not records:
        raise ValueError(f"No records to benchmark in {cfg.manifest_path}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] {record['model_mode']} {record['scene_token']}")
        rows.append(compute_clip_metrics(record, cfg))

    summary_rows = summarize_by_model(rows)
    fvd_artifacts: dict[str, str] = {}
    if cfg.compute_fvd:
        fvd_rows, fvd_artifacts = compute_fvd_by_model(records, cfg)
        summary_rows = merge_fvd_into_summary(summary_rows, fvd_rows)
    paired_rows = paired_model_comparison(rows)
    gate = quality_gate(summary_rows)

    per_clip_csv = cfg.output_dir / "per_clip_metrics.csv"
    summary_csv = cfg.output_dir / "model_summary.csv"
    paired_csv = cfg.output_dir / "paired_base_vs_lora.csv"
    write_csv(per_clip_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_csv(paired_csv, paired_rows)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(cfg.manifest_path),
        "source_dir": str(cfg.source_dir),
        "output_dir": str(cfg.output_dir),
        "fps": cfg.fps,
        "width": cfg.width,
        "height": cfg.height,
        "context_frames": cfg.context_frames,
        "future_frames": cfg.future_frames,
        "total_frames": cfg.total_frames,
        "compute_fvd": cfg.compute_fvd,
        "fvd_backend": cfg.fvd_backend if cfg.compute_fvd else "",
        "fvd_num_frames": cfg.fvd_num_frames if cfg.compute_fvd else "",
        "fvd_size": cfg.fvd_size if cfg.compute_fvd else "",
        "num_records": len(rows),
        "notes": [
            "Main reference metrics are computed only on generated future frames.",
            "global_ssim is frame-averaged global grayscale SSIM, not windowed SSIM.",
            "Lower is better for MSE, MAE, temporal_delta_error_mae, and copy_leakage_ratio_min_context_over_future.",
            "Higher is better for PSNR and global_ssim.",
            "Sharpness ratio near 1 means generated future sharpness matches the real future; much below 1 means blurrier than reference.",
            "FVD, when enabled, is a distribution metric over future-frame video features; lower is better.",
            "The torchvision_r3d18 FVD backend is an FVD-style internal comparison metric, not directly comparable to canonical I3D FVD papers.",
        ],
        "source_manifest_context": {
            "description": manifest.get("description", ""),
            "models": manifest.get("models", []),
            "seed": manifest.get("seed", ""),
        },
        "summary_by_model": summary_rows,
        "paired_base_vs_lora": paired_rows,
        "quality_gate": gate,
        "artifacts": {
            "per_clip_metrics_csv": str(per_clip_csv),
            "model_summary_csv": str(summary_csv),
            "paired_base_vs_lora_csv": str(paired_csv),
            **fvd_artifacts,
        },
    }
    report_path = cfg.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
