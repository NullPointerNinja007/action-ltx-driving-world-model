from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"
if not MODAL.exists():
    MODAL = Path("modal")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TRAIN_WRAPPER = "train_ltx2b_distilled_waymo_frame_temporal_bottleneck_fullaction_motion_v4_action_lora.py"
INFER_WRAPPER = "generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py"

CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-r16-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-infer"
RUNS_ROOT = "distilled098_full112_lowfreq_motion_v4_action_lora_24fps_minterpolate_seed231_runs"
BASE_CKPT = "ltxv-2b-0.9.8-distilled.safetensors"

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot_seed231_all5"
LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot_generated"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot_seed231_all5"

SEED = 231
FPS = 24
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121
CHECKPOINT_STEPS = [0, 100, 250, 500, 750, 1000]
COUNTERFACTUAL_STEPS = [500, 1000]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]
TRAIN_CHECKPOINT_STEPS = "0,100,250,500,750,1000"


VARIANTS: dict[str, dict[str, Any]] = {
    "main": {
        "run_name": "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_seed231_from_shifted_noaction_step003000_steps1000",
        "diffusion_loss_weight": 0.25,
        "lowfreq_target_loss_weight": 1.0,
        "lowfreq_delta_loss_weight": 1.0,
        "hf_teacher_loss_weight": 0.20,
        "action_motion_aux_loss_weight": 0.05,
        "action_residual_loss_weight": 0.002,
        "action_gate_loss_weight": 0.002,
        "action_learning_rate": 3e-5,
        "action_injector_learning_rate": 3e-5,
        "action_gate_learning_rate": 3e-5,
    },
    "conservative": {
        "run_name": "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_conservative_seed231_from_shifted_noaction_step003000_steps1000",
        "diffusion_loss_weight": 0.50,
        "lowfreq_target_loss_weight": 0.50,
        "lowfreq_delta_loss_weight": 0.50,
        "hf_teacher_loss_weight": 0.10,
        "action_motion_aux_loss_weight": 0.05,
        "action_residual_loss_weight": 0.002,
        "action_gate_loss_weight": 0.002,
        "action_learning_rate": 3e-5,
        "action_injector_learning_rate": 3e-5,
        "action_gate_learning_rate": 3e-5,
    },
}


@dataclass(frozen=True)
class EvalConfig:
    variant: str
    checkpoint_step: int
    diagnostic_group: str = "checkpoint_sweep"
    counterfactual_action_mode: str = "correct"
    action_gate_scale: float = 1.0
    action_vector_scale: float = 1.0
    counterfactual_rotation: int = 1

    @property
    def run_name(self) -> str:
        return str(VARIANTS[self.variant]["run_name"])

    @property
    def checkpoint_name(self) -> str:
        if self.checkpoint_step == 0:
            return "step_000000_base_reference"
        return f"step_{self.checkpoint_step:06d}"

    @property
    def model_mode(self) -> str:
        suffix = (
            f"_{self.counterfactual_action_mode}"
            if self.diagnostic_group == "counterfactual_suite"
            else ""
        )
        return f"v4_{self.variant}_full112_lowfreq_step{self.checkpoint_step:06d}{suffix}"

    @property
    def run_label(self) -> str:
        gate = f"{self.action_gate_scale:.3f}".replace(".", "p")
        return f"{self.model_mode}_g{gate}_seed{SEED}_all5"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=merged_env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def log_completed(log_path: Path) -> bool:
    return log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore")


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        TRAIN_WRAPPER,
        INFER_WRAPPER,
        "scripts/prepare_waymo24_full112_action_stats.py",
        "scripts/audit_waymo24_full112_action_alignment.py",
        "scripts/run_frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot.py",
        "scripts/compute_video_quality_modal.py",
        "scripts/compute_action_fvd_modal.py",
        "scripts/compute_counterfactual_sensitivity_modal.py",
    ]
    run_command([str(PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log")


def prepare_stats(force: bool = False, limit: int = 1024) -> None:
    cmd = [str(MODAL), "run", "scripts/prepare_waymo24_full112_action_stats.py"]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if force:
        cmd.append("--force")
    run_command(cmd, LOG_DIR / "preflight" / "prepare_full112_stats.log")


def audit_actions() -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/audit_waymo24_full112_action_alignment.py",
            "--train-limit",
            "512",
            "--val-limit",
            "128",
            "--latent-prefix",
            "latents",
        ],
        LOG_DIR / "preflight" / "audit_full112_actions.log",
    )


def train_variant(variant: str, *, smoke: bool = False, force: bool = False) -> str:
    weights = VARIANTS[variant]
    run_name = f"{weights['run_name']}_smoke10" if smoke else str(weights["run_name"])
    log_path = LOG_DIR / "training" / f"{run_name}.log"
    if not force and log_completed(log_path):
        return run_name

    max_steps = "10" if smoke else "1000"
    train_limit = "8" if smoke else "0"
    val_limit = "8" if smoke else "32"
    checkpoint_steps = "0,10" if smoke else TRAIN_CHECKPOINT_STEPS
    cmd = [
        str(MODAL),
        "run",
        TRAIN_WRAPPER,
        "--run-name",
        run_name,
        "--max-steps",
        max_steps,
        "--max-train-hours",
        "4.0",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        str(weights["action_learning_rate"]),
        "--action-injector-learning-rate",
        str(weights["action_injector_learning_rate"]),
        "--action-gate-learning-rate",
        str(weights["action_gate_learning_rate"]),
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        checkpoint_steps,
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--diffusion-loss-weight",
        str(weights["diffusion_loss_weight"]),
        "--lowfreq-target-loss-weight",
        str(weights["lowfreq_target_loss_weight"]),
        "--lowfreq-delta-loss-weight",
        str(weights["lowfreq_delta_loss_weight"]),
        "--hf-teacher-loss-weight",
        str(weights["hf_teacher_loss_weight"]),
        "--action-motion-aux-loss-weight",
        str(weights["action_motion_aux_loss_weight"]),
        "--action-residual-loss-weight",
        str(weights["action_residual_loss_weight"]),
        "--action-gate-loss-weight",
        str(weights["action_gate_loss_weight"]),
        "--action-gate-scale",
        "1.0",
        "--action-gate-bound",
        "0.25",
        "--action-hidden-dim",
        "384",
        "--action-transformer-layers",
        "4",
        "--action-transformer-heads",
        "8",
        "--frame-action-feature-key",
        "actions_full_112",
        "--frame-action-stats-relpath",
        "manifests/frame_action_24fps_full112_normalization_stats.json",
        "--num-val-samples",
        "0",
        "--train-limit",
        train_limit,
        "--val-limit",
        val_limit,
    ]
    run_command(cmd, log_path)
    return run_name


def train_variants_parallel(*, smoke: bool, force: bool) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(train_variant, variant, smoke=smoke, force=force) for variant in VARIANTS]
        for future in as_completed(futures):
            print(json.dumps({"trained": future.result(), "smoke": smoke}, sort_keys=True), flush=True)


def sweep_configs() -> list[EvalConfig]:
    return [EvalConfig(variant, step) for variant in VARIANTS for step in CHECKPOINT_STEPS]


def counterfactual_configs() -> list[EvalConfig]:
    return [
        EvalConfig(variant, step, diagnostic_group="counterfactual_suite", counterfactual_action_mode=mode)
        for variant in VARIANTS
        for step in COUNTERFACTUAL_STEPS
        for mode in COUNTERFACTUAL_MODES
    ]


def generate_one(config: EvalConfig, *, seed: int, limit: int) -> str:
    log_path = LOG_DIR / "generation" / f"{config.run_label}.log"
    if log_completed(log_path):
        return config.run_label
    cmd = [
        str(MODAL),
        "run",
        INFER_WRAPPER,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--lora-step",
        config.checkpoint_name,
        "--lora-run-name",
        config.run_name,
        "--run-label",
        config.run_label,
        "--base-label",
        "base_distilled_no_lora",
        "--action-gate-scale",
        str(config.action_gate_scale),
        "--action-vector-scale",
        str(config.action_vector_scale),
        "--counterfactual-action-mode",
        config.counterfactual_action_mode,
        "--counterfactual-rotation",
        str(config.counterfactual_rotation),
    ]
    run_command(cmd, log_path, env={"LTX_MODAL_GPU": "A100"})
    return config.run_label


def generate_all(configs: list[EvalConfig], *, max_workers: int, seed: int, limit: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, config, seed=seed, limit=limit) for config in configs]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True), flush=True)


def download_one(config: EvalConfig) -> Path:
    remote_path = f"{RUNS_ROOT}/{config.run_label}"
    local_dest = LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label
    local_dest.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "download" / f"{config.run_label}.log"
    summary_local = local_dest / "run_summary.json"
    if not summary_local.exists():
        run_command(
            [str(MODAL), "volume", "get", "--force", ARTIFACT_VOLUME, f"{remote_path}/run_summary.json", str(summary_local)],
            log_path,
        )
    summary = json.loads(summary_local.read_text(encoding="utf-8"))
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    for record in summary["results"]:
        generated_relpath = record["generated_video_relpath"]
        if not generated_relpath.startswith(prefix):
            raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
        video_local = local_dest / generated_relpath[len(prefix) :]
        video_local.parent.mkdir(parents=True, exist_ok=True)
        if video_local.exists():
            continue
        run_command(
            [str(MODAL), "volume", "get", "--force", ARTIFACT_VOLUME, generated_relpath, str(video_local)],
            log_path,
        )
    return local_dest


def download_all(configs: list[EvalConfig]) -> None:
    for config in configs:
        print(json.dumps({"downloaded": str(download_one(config).relative_to(ROOT))}, sort_keys=True), flush=True)


def local_file_for_record(record: dict[str, Any], config: EvalConfig) -> Path:
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label / generated_relpath[len(prefix) :]


def build_manifest(configs: list[EvalConfig], path: Path, *, seed: int, limit: int, description: str) -> Path:
    records: list[dict[str, Any]] = []
    for config in configs:
        summary_path = LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary["results"]:
            local_file = local_file_for_record(record, config)
            if not local_file.exists():
                raise FileNotFoundError(local_file)
            row = dict(record)
            row.update(
                {
                    "local_file": str(local_file),
                    "diagnostic_group": config.diagnostic_group,
                    "method_key": f"v4_{config.variant}_full112_lowfreq_motion",
                    "method_label": f"V4 {config.variant} Full112 LowFreq Motion",
                    "variant": config.variant,
                    "checkpoint_step": config.checkpoint_step,
                    "checkpoint_name": config.checkpoint_name,
                    "model_mode": config.model_mode,
                    "action_gate_scale": config.action_gate_scale,
                    "action_vector_scale": config.action_vector_scale,
                    "counterfactual_action_mode": config.counterfactual_action_mode,
                    "counterfactual_rotation": config.counterfactual_rotation,
                    "using_lora": True,
                }
            )
            records.append(row)
    write_json(
        path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "seed": seed,
            "limit": limit,
            "context_frames": CONTEXT_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "total_frames": TOTAL_FRAMES,
            "fps": FPS,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "run_names": {variant: payload["run_name"] for variant, payload in VARIANTS.items()},
            "records": records,
        },
    )
    return path


def run_quality_metrics(manifest_path: Path, *, chunk_size: int) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_video_quality_modal.py",
            "--manifest",
            str(manifest_path),
            "--source-dir",
            str(SOURCE_DIR),
            "--output-dir",
            str(BENCHMARK_DIR),
            "--run-id",
            "v4_full112_lowfreq_motion_quality",
            "--chunk-size",
            str(chunk_size),
        ],
        LOG_DIR / "metrics" / "compute_quality.log",
    )


def run_fvd(manifest_path: Path) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_action_fvd_modal.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(BENCHMARK_DIR),
            "--run-id",
            "v4_full112_lowfreq_motion_fvd",
        ],
        LOG_DIR / "metrics" / "compute_fvd.log",
        env={"ACTION_FVD_GPU": "A10G"},
    )


def run_counterfactual_sensitivity(manifest_path: Path, *, chunk_size: int) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_counterfactual_sensitivity_modal.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(BENCHMARK_DIR),
            "--run-id",
            "v4_full112_lowfreq_motion_counterfactual_sensitivity",
            "--chunk-size",
            str(chunk_size),
        ],
        LOG_DIR / "metrics" / "compute_counterfactual_sensitivity.log",
    )


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return float(raw)


def merge_summaries() -> list[dict[str, Any]]:
    quality_rows = read_csv(BENCHMARK_DIR / "model_summary.csv")
    fvd_rows = {row["model_mode"]: row for row in read_csv(BENCHMARK_DIR / "fvd_summary.csv")}
    merged: list[dict[str, Any]] = []
    for row in quality_rows:
        model_mode = row["model_mode"]
        fvd = fvd_rows.get(model_mode, {})
        step_match = re.search(r"step0*([0-9]+)", model_mode)
        variant = "main" if "_main_" in model_mode else "conservative" if "_conservative_" in model_mode else ""
        merged.append(
            {
                **row,
                "variant": variant,
                "checkpoint_step": int(step_match.group(1)) if step_match else -1,
                "future_fvd_style": fvd.get("future_fvd_style", fvd.get("fvd_future", "")),
                "mean_feature_l2": fvd.get("mean_feature_l2", ""),
            }
        )
    merged = sorted(merged, key=lambda row: (row["variant"], int(row["checkpoint_step"])))
    write_csv(BENCHMARK_DIR / "v4_pilot_model_summary_merged.csv", merged)
    return merged


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, filename: str) -> None:
    plot_dir = BENCHMARK_DIR / "metric_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant and str(row.get(metric, "")) != ""]
        if not subset:
            continue
        subset = sorted(subset, key=lambda row: int(row["checkpoint_step"]))
        x = [int(row["checkpoint_step"]) for row in subset]
        y = [as_float(row, metric) for row in subset]
        plt.plot(x, y, marker="o", linewidth=2, label=variant)
    plt.xlabel("Checkpoint step")
    plt.ylabel(ylabel)
    plt.title(ylabel + " over V4 pilot checkpoints")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=180)
    plt.close()


def make_plots() -> None:
    rows = merge_summaries()
    metrics = [
        ("mean_future_psnr", "Future PSNR", "future_psnr_over_checkpoints.png"),
        ("mean_future_global_ssim", "Future SSIM", "future_ssim_over_checkpoints.png"),
        (
            "mean_sharpness_ratio_generated_over_reference",
            "Sharpness ratio generated / GT",
            "sharpness_ratio_over_checkpoints.png",
        ),
        (
            "mean_fft_high_frequency_energy_ratio_generated_over_reference",
            "FFT high-frequency ratio generated / GT",
            "fft_high_frequency_ratio_over_checkpoints.png",
        ),
        (
            "mean_motion_ratio_generated_over_reference",
            "Motion ratio generated / GT",
            "motion_ratio_over_checkpoints.png",
        ),
        (
            "mean_low_frequency_motion_ratio_generated_over_reference",
            "Low-frequency motion ratio generated / GT",
            "low_frequency_motion_ratio_over_checkpoints.png",
        ),
        ("mean_temporal_delta_error_mae", "Temporal delta error MAE", "temporal_delta_error_over_checkpoints.png"),
        ("future_fvd_style", "FVD-style future distance", "fvd_style_over_checkpoints.png"),
    ]
    for metric, ylabel, filename in metrics:
        plot_metric(rows, metric, ylabel, filename)
    write_json(
        BENCHMARK_DIR / "metric_plots" / "README.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": "V4 pilot checkpoint plots. Metrics are computed on Modal; plots are CSV-only local postprocessing.",
            "plots": [filename for _, _, filename in metrics],
        },
    )


def write_summary(phase: str) -> None:
    write_json(
        BENCHMARK_DIR / "v4_pilot_run_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "variants": VARIANTS,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "benchmark_dir": str(BENCHMARK_DIR),
            "generated_dir": str(LOCAL_GENERATED_ROOT),
            "corrected_setup": {
                "seed": SEED,
                "fps": FPS,
                "context_frames": CONTEXT_FRAMES,
                "future_frames": FUTURE_FRAMES,
                "total_frames": TOTAL_FRAMES,
                "image_cond_noise_scale": 0.0,
                "timestep_sampling": "shifted_lognormal",
                "frame_action_feature_key": "actions_full_112",
                "frame_action_stats_relpath": "manifests/frame_action_24fps_full112_normalization_stats.json",
                "uses_upsampled_frame_actions": True,
                "uses_original_10fps_actions": False,
                "recaches_latents": False,
                "text_conditioning_enabled": True,
            },
        },
    )


def run_checkpoint_sweep(max_workers: int, seed: int, limit: int, metric_chunk_size: int) -> None:
    configs = sweep_configs()
    generate_all(configs, max_workers=max_workers, seed=seed, limit=limit)
    download_all(configs)
    manifest = build_manifest(
        configs,
        BENCHMARK_DIR / "manifest_v4_full112_lowfreq_motion_checkpoint_sweep_seed231_all5.json",
        seed=seed,
        limit=limit,
        description="V4 full112 low-frequency motion pilot checkpoint sweep.",
    )
    run_quality_metrics(manifest, chunk_size=metric_chunk_size)
    run_fvd(manifest)
    make_plots()


def run_counterfactual(max_workers: int, seed: int, limit: int, metric_chunk_size: int) -> None:
    configs = counterfactual_configs()
    generate_all(configs, max_workers=max_workers, seed=seed, limit=limit)
    download_all(configs)
    manifest = build_manifest(
        configs,
        BENCHMARK_DIR / "manifest_v4_full112_lowfreq_motion_counterfactual_seed231_all5.json",
        seed=seed,
        limit=limit,
        description="V4 full112 low-frequency motion counterfactual action sensitivity.",
    )
    run_counterfactual_sensitivity(manifest, chunk_size=metric_chunk_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V4 full112 low-frequency motion pilot.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "prep", "smoke_train", "train", "sweep", "counterfactual", "plots", "all"),
        default="all",
    )
    parser.add_argument("--max-generation-workers", type=int, default=6)
    parser.add_argument("--metric-chunk-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-stats", action="store_true")
    parser.add_argument("--stats-limit", type=int, default=1024)
    parser.add_argument("--skip-smoke-train", action="store_true")
    args = parser.parse_args()

    if args.phase in {"syntax", "all"}:
        syntax_check()
    if args.phase in {"prep", "all"}:
        prepare_stats(force=args.force_stats, limit=args.stats_limit)
        audit_actions()
    if args.phase in {"smoke_train", "all"} and not args.skip_smoke_train:
        train_variants_parallel(smoke=True, force=args.force_train)
    if args.phase in {"train", "all"}:
        train_variants_parallel(smoke=False, force=args.force_train)
    if args.phase in {"sweep", "all"}:
        run_checkpoint_sweep(args.max_generation_workers, args.seed, args.limit, args.metric_chunk_size)
    if args.phase in {"counterfactual", "all"}:
        run_counterfactual(args.max_generation_workers, args.seed, args.limit, args.metric_chunk_size)
    if args.phase in {"plots"}:
        make_plots()
    write_summary(args.phase)
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
