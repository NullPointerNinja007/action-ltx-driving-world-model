from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio
import imageio.v3 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"
SIDE_BY_SIDE_DIR = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_side_by_side_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"

RUN_NAME = "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_seed231_from_shifted_noaction_step003000_steps3000"
CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-framebottleneck-hfteacher-action-r16-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-framebottleneck-hfteacher-action-infer"
RUNS_ROOT = "distilled098_framebottleneck_hfteacher_action_lora_24fps_minterpolate_seed231_runs"
WRAPPER = "generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora.py"
METHOD_KEY = "frame_temporal_bottleneck_hfteacher"
METHOD_LABEL = "Frame Temporal Bottleneck + HF Teacher"
MANIFEST_STEM = "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"
COUNTERFACTUAL_MANIFEST_STEM = "frame_temporal_bottleneck_hfteacher_gate_scale_counterfactual_seed231_all5"
FVD_RUN_ID = "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"
PLOT_README_TITLE = "Phase 2 Gate-Scale Sweep Metric Plots"
SIDE_BY_SIDE_STEM = "phase2_gate_sweep"
SIDE_BY_SIDE_MANIFEST_NAME = "manifest_phase2_gate_scale_side_by_side.json"
FINAL_SUMMARY_NAME = "phase2_gate_scale_sweep_summary.json"

V2_RUN_NAME = "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_from_shifted_noaction_step003000_steps1000"
V2_TRAIN_WRAPPER = "train_ltx2b_distilled_waymo_frame_temporal_bottleneck_hf_teacher_action_lora_v2.py"

NOACTION_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "noaction_shifted_timestep_longer_seed231_all5"
    / "manifest_noaction_shifted_timestep_longer_seed231_all5.json"
)

CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121
FPS = 24
SEED = 231

SWEEP_CHECKPOINTS = ["step_000100", "step_000250", "step_003000"]
COUNTERFACTUAL_BASE_CHECKPOINTS = ["step_000100", "step_000250"]
GATE_SCALES = [0.0, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]
LOW_GATE_MAX = 0.25

SHARPNESS_THRESHOLD = 0.22
FFT_THRESHOLD = 0.70
MOTION_THRESHOLD = 0.70
ACTION_SENSITIVITY_THRESHOLD = 1.60

GATE_MANIFEST = BENCHMARK_DIR / f"manifest_{MANIFEST_STEM}.json"
COUNTERFACTUAL_MANIFEST = (
    BENCHMARK_DIR / f"manifest_{COUNTERFACTUAL_MANIFEST_STEM}.json"
)
SELECTION_PATH = BENCHMARK_DIR / "phase2_gate_scale_selection.json"


@dataclass(frozen=True)
class EvalConfig:
    diagnostic_group: str
    checkpoint_name: str
    counterfactual_action_mode: str = "correct"
    action_gate_scale: float = 1.0
    action_vector_scale: float = 1.0
    counterfactual_rotation: int = 1

    @property
    def checkpoint_step(self) -> int:
        return step_to_int(self.checkpoint_name)

    @property
    def model_mode(self) -> str:
        if self.diagnostic_group == "gate_scale_sweep":
            return (
                f"{METHOD_KEY}_gatesweep_"
                f"step{self.checkpoint_step:06d}_g{scale_label(self.action_gate_scale)}"
            )
        return (
            f"{METHOD_KEY}_counterfactual_"
            f"step{self.checkpoint_step:06d}_{self.counterfactual_action_mode}_"
            f"g{scale_label(self.action_gate_scale)}"
        )

    @property
    def run_label(self) -> str:
        return f"{self.model_mode}_v{scale_label(self.action_vector_scale)}_seed{SEED}_all5"


def step_to_int(step: str) -> int:
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse checkpoint step: {step}")
    return int(match.group(1))


def scale_label(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def gate_configs() -> list[EvalConfig]:
    return [
        EvalConfig("gate_scale_sweep", checkpoint, action_gate_scale=gate)
        for checkpoint in SWEEP_CHECKPOINTS
        for gate in GATE_SCALES
    ]


def smoke_configs() -> list[EvalConfig]:
    return [
        EvalConfig("smoke", "step_000250", action_gate_scale=0.0),
        EvalConfig("smoke", "step_000250", action_gate_scale=0.1),
    ]


def counterfactual_configs() -> list[EvalConfig]:
    selection = read_selection()
    checkpoints = selection["counterfactual_checkpoints"]
    gates = selection["counterfactual_gate_scales"]
    return [
        EvalConfig("counterfactual_suite", checkpoint, mode, action_gate_scale=gate)
        for checkpoint in checkpoints
        for gate in gates
        for mode in COUNTERFACTUAL_MODES
    ]


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env["LTX_MODAL_GPU"] = "A100"
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=merged_env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def generate_one(config: EvalConfig, *, seed: int, limit: int) -> str:
    log_path = LOG_DIR / "generation" / f"{config.run_label}.log"
    if log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore"):
        return config.run_label
    cmd = [
        str(MODAL),
        "run",
        WRAPPER,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--lora-step",
        config.checkpoint_name,
        "--lora-run-name",
        RUN_NAME,
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
    run_command(cmd, log_path)
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
                    "method_key": METHOD_KEY,
                    "method_label": METHOD_LABEL,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
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
                "lora_run_name": RUN_NAME,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return float(raw)


def run_quality_metrics(manifest_path: Path, *, log_name: str = "benchmark_video_quality.log") -> None:
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
            FVD_RUN_ID.replace("fvd", "quality"),
            "--chunk-size",
            "8",
        ],
        LOG_DIR / log_name,
    )


def run_fvd(manifest_path: Path, *, run_id: str, log_name: str = "compute_fvd.log") -> None:
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
            run_id,
        ],
        LOG_DIR / log_name,
    )


def run_counterfactual_sensitivity(manifest_path: Path) -> None:
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
            FVD_RUN_ID.replace("fvd", "counterfactual_sensitivity"),
            "--chunk-size",
            "8",
        ],
        LOG_DIR / "counterfactual_sensitivity.log",
    )


def merge_summaries(manifest_path: Path, *, output_csv: Path) -> list[dict[str, Any]]:
    summary_rows = read_csv(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {row["model_mode"]: row for row in read_csv(BENCHMARK_DIR / "fvd_summary.csv")}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta_by_model: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        meta_by_model[record["model_mode"]] = {
            "diagnostic_group": record["diagnostic_group"],
            "checkpoint_step": record["checkpoint_step"],
            "checkpoint_name": record["checkpoint_name"],
            "action_gate_scale": record["action_gate_scale"],
            "action_vector_scale": record["action_vector_scale"],
            "counterfactual_action_mode": record["counterfactual_action_mode"],
        }
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        model_mode = row["model_mode"]
        out = {**meta_by_model.get(model_mode, {}), **row}
        if model_mode in fvd_by_model:
            out.update(
                {
                    "fvd_future": fvd_by_model[model_mode].get("fvd_future", ""),
                    "fvd_backend": fvd_by_model[model_mode].get("fvd_backend", ""),
                    "fvd_num_videos": fvd_by_model[model_mode].get("fvd_num_videos", ""),
                    "fvd_num_frames": fvd_by_model[model_mode].get("fvd_num_frames", ""),
                    "fvd_size": fvd_by_model[model_mode].get("fvd_size", ""),
                }
            )
        rows.append(out)
    rows.sort(
        key=lambda row: (
            str(row.get("diagnostic_group", "")),
            int(row.get("checkpoint_step", 0)),
            float(row.get("action_gate_scale", 1.0)),
            str(row.get("counterfactual_action_mode", "")),
        )
    )
    write_csv(output_csv, rows)
    return rows


def write_gate_scale_summary() -> list[dict[str, Any]]:
    rows = merge_summaries(GATE_MANIFEST, output_csv=BENCHMARK_DIR / "gate_scale_summary.csv")
    return rows


def aggregate_gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("diagnostic_group") == "gate_scale_sweep":
            grouped[float(row["action_gate_scale"])].append(row)
    out: list[dict[str, Any]] = []
    for gate, selected in sorted(grouped.items()):
        out.append(
            {
                "action_gate_scale": gate,
                "num_checkpoint_rows": len(selected),
                "mean_future_psnr": float(np.mean([as_float(row, "mean_future_psnr") for row in selected])),
                "mean_future_global_ssim": float(np.mean([as_float(row, "mean_future_global_ssim") for row in selected])),
                "mean_sharpness_ratio_generated_over_reference": float(
                    np.mean([as_float(row, "mean_sharpness_ratio_generated_over_reference") for row in selected])
                ),
                "mean_fft_high_frequency_energy_ratio_generated_over_reference": float(
                    np.mean([as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference") for row in selected])
                ),
                "mean_motion_ratio_generated_over_reference": float(
                    np.mean([as_float(row, "mean_motion_ratio_generated_over_reference") for row in selected])
                ),
                "mean_low_frequency_motion_ratio_generated_over_reference": float(
                    np.mean([as_float(row, "mean_low_frequency_motion_ratio_generated_over_reference") for row in selected])
                ),
                "mean_temporal_delta_error_mae": float(np.mean([as_float(row, "mean_temporal_delta_error_mae") for row in selected])),
                "mean_fvd_future": float(np.mean([as_float(row, "fvd_future") for row in selected])),
            }
        )
    write_csv(BENCHMARK_DIR / "gate_scale_aggregate_summary.csv", out)
    return out


def score_low_gate_row(row: dict[str, Any]) -> float:
    sharp = as_float(row, "mean_sharpness_ratio_generated_over_reference", 0.0)
    fft = as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference", 0.0)
    motion = as_float(row, "mean_motion_ratio_generated_over_reference", 0.0)
    fvd = as_float(row, "fvd_future", 100.0)
    return sharp + 0.25 * fft + 0.25 * motion - 0.002 * fvd


def select_counterfactual_settings() -> dict[str, Any]:
    rows = [row for row in read_csv(BENCHMARK_DIR / "gate_scale_summary.csv") if row.get("diagnostic_group") == "gate_scale_sweep"]
    if not rows:
        raise FileNotFoundError(BENCHMARK_DIR / "gate_scale_summary.csv")

    selectable_steps = {step_to_int(step) for step in COUNTERFACTUAL_BASE_CHECKPOINTS}
    low_rows = [
        row
        for row in rows
        if int(row["checkpoint_step"]) in selectable_steps
        and 0.0 < float(row["action_gate_scale"]) <= LOW_GATE_MAX
    ]
    threshold_rows = [
        row
        for row in low_rows
        if as_float(row, "mean_sharpness_ratio_generated_over_reference") >= SHARPNESS_THRESHOLD
        and as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference") >= FFT_THRESHOLD
        and as_float(row, "mean_motion_ratio_generated_over_reference") >= MOTION_THRESHOLD
    ]
    pool = threshold_rows or low_rows
    if not pool:
        raise ValueError("No nonzero low-gate rows are available for counterfactual selection.")

    by_gate: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_gate[float(row["action_gate_scale"])].append(row)
    gate_scores = []
    for gate, selected in by_gate.items():
        gate_scores.append(
            {
                "gate": gate,
                "score": float(np.mean([score_low_gate_row(row) for row in selected])),
                "passes_thresholds": all(row in threshold_rows for row in selected) and bool(threshold_rows),
            }
        )
    gate_scores.sort(key=lambda row: (-float(row["score"]), float(row["gate"])))
    selected_gates = [float(row["gate"]) for row in gate_scores[:2]]
    if 1.0 not in selected_gates:
        selected_gates.append(1.0)
    selected_gates = sorted(set(selected_gates))

    step3000_low = [
        row
        for row in rows
        if int(row["checkpoint_step"]) == 3000
        and 0.0 < float(row["action_gate_scale"]) <= LOW_GATE_MAX
        and as_float(row, "mean_sharpness_ratio_generated_over_reference") >= SHARPNESS_THRESHOLD
        and as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference") >= FFT_THRESHOLD
        and as_float(row, "mean_motion_ratio_generated_over_reference") >= MOTION_THRESHOLD
    ]
    checkpoints = list(COUNTERFACTUAL_BASE_CHECKPOINTS)
    if step3000_low:
        checkpoints.append("step_003000")

    selection = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source": str(BENCHMARK_DIR / "gate_scale_summary.csv"),
        "selection_rule": (
            f"Best two nonzero low gate scales among steps {COUNTERFACTUAL_BASE_CHECKPOINTS}; prefer rows passing "
            f"sharpness>={SHARPNESS_THRESHOLD}, fft>={FFT_THRESHOLD}, motion>={MOTION_THRESHOLD}; always add gate=1.0."
        ),
        "gate_scores": gate_scores,
        "counterfactual_gate_scales": selected_gates,
        "counterfactual_checkpoints": checkpoints,
        "included_step3000": "step_003000" in checkpoints,
        "threshold_rows_available": bool(threshold_rows),
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection


def read_selection() -> dict[str, Any]:
    if not SELECTION_PATH.exists():
        return select_counterfactual_settings()
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def plot_gate_metric(rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for checkpoint in SWEEP_CHECKPOINTS:
        step = step_to_int(checkpoint)
        selected = [row for row in rows if int(row["checkpoint_step"]) == step]
        selected.sort(key=lambda row: float(row["action_gate_scale"]))
        x = [float(row["action_gate_scale"]) for row in selected]
        y = [as_float(row, metric) for row in selected]
        ax.plot(x, y, marker="o", linewidth=2, label=f"step {step}")
    ax.set_title(title)
    ax.set_xlabel("Inference action gate scale")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_fvd_vs_sharpness(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for checkpoint in SWEEP_CHECKPOINTS:
        step = step_to_int(checkpoint)
        selected = [row for row in rows if int(row["checkpoint_step"]) == step]
        x = [as_float(row, "mean_sharpness_ratio_generated_over_reference") for row in selected]
        y = [as_float(row, "fvd_future") for row in selected]
        ax.scatter(x, y, s=65, label=f"step {step}")
        for row, x_value, y_value in zip(selected, x, y):
            ax.annotate(f"g={float(row['action_gate_scale']):g}", (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Pareto: FVD-Style Distance vs Sharpness")
    ax.set_xlabel("Sharpness ratio")
    ax.set_ylabel("FVD-style future distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_action_sensitivity_vs_sharpness(output_path: Path) -> None:
    sensitivity = read_csv(BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv")
    gate_rows = read_csv(BENCHMARK_DIR / "gate_scale_summary.csv")
    sharp_by_key = {
        (int(row["checkpoint_step"]), float(row["action_gate_scale"])): as_float(
            row, "mean_sharpness_ratio_generated_over_reference"
        )
        for row in gate_rows
        if row.get("diagnostic_group") == "gate_scale_sweep"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mode in ("zero", "shuffled", "reversed_future"):
        selected = [row for row in sensitivity if row["comparison_mode"] == mode]
        x = [sharp_by_key.get((int(row["checkpoint_step"]), float(row["action_gate_scale"])), float("nan")) for row in selected]
        y = [as_float(row, "mean_future_rgb_mae_correct_vs_mode") for row in selected]
        ax.scatter(x, y, s=65, label=mode)
        for row, x_value, y_value in zip(selected, x, y):
            if not math.isnan(x_value) and not math.isnan(y_value):
                ax.annotate(
                    f"s{int(row['checkpoint_step'])}/g{float(row['action_gate_scale']):g}",
                    (x_value, y_value),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
    ax.axhline(ACTION_SENSITIVITY_THRESHOLD, color="tab:red", linestyle="--", linewidth=1, label="target sensitivity")
    ax.axvline(SHARPNESS_THRESHOLD, color="tab:gray", linestyle="--", linewidth=1, label="sharpness target")
    ax.set_title("Pareto: Action Sensitivity vs Sharpness")
    ax.set_xlabel("Sharpness ratio")
    ax.set_ylabel("Correct-vs-counterfactual future RGB MAE")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_plots(*, include_sensitivity: bool) -> None:
    rows = [row for row in read_csv(BENCHMARK_DIR / "gate_scale_summary.csv") if row.get("diagnostic_group") == "gate_scale_sweep"]
    metric_dir = BENCHMARK_DIR / "metric_plots"
    for metric, title, name in [
        ("mean_sharpness_ratio_generated_over_reference", "Gate Scale vs Sharpness Ratio", "gate_scale_vs_sharpness_ratio.png"),
        (
            "mean_fft_high_frequency_energy_ratio_generated_over_reference",
            "Gate Scale vs FFT High-Frequency Ratio",
            "gate_scale_vs_fft_high_frequency_ratio.png",
        ),
        ("mean_motion_ratio_generated_over_reference", "Gate Scale vs Motion Ratio", "gate_scale_vs_motion_ratio.png"),
        (
            "mean_low_frequency_motion_ratio_generated_over_reference",
            "Gate Scale vs Low-Frequency Motion Ratio",
            "gate_scale_vs_low_frequency_motion_ratio.png",
        ),
        ("fvd_future", "Gate Scale vs FVD-Style Future Distance", "gate_scale_vs_fvd.png"),
        ("mean_future_psnr", "Gate Scale vs Future PSNR", "gate_scale_vs_psnr.png"),
        ("mean_future_global_ssim", "Gate Scale vs Future SSIM", "gate_scale_vs_ssim.png"),
        ("mean_temporal_delta_error_mae", "Gate Scale vs Temporal Delta Error", "gate_scale_vs_temporal_delta_error.png"),
    ]:
        plot_gate_metric(rows, metric, title, metric_dir / name)
    plot_fvd_vs_sharpness(rows, metric_dir / "pareto_fvd_vs_sharpness.png")
    if include_sensitivity and (BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv").exists():
        plot_action_sensitivity_vs_sharpness(metric_dir / "pareto_action_sensitivity_vs_sharpness.png")
    plots = sorted(path.name for path in metric_dir.glob("*.png"))
    (metric_dir / "README.md").write_text(
        f"# {PLOT_README_TITLE}\n\n" + "\n".join(f"- [{name}]({name})" for name in plots) + "\n",
        encoding="utf-8",
    )


def records_by_scene(
    manifest_path: Path,
    *,
    method_key: str | None = None,
    checkpoint_step: int | None = None,
    gate_scale: float | None = None,
    diagnostic_group: str | None = None,
) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for row in manifest["records"]:
        if method_key and row.get("method_key") != method_key:
            continue
        if checkpoint_step is not None and int(row.get("checkpoint_step", -1)) != checkpoint_step:
            continue
        if diagnostic_group and row.get("diagnostic_group") != diagnostic_group:
            continue
        if gate_scale is not None and abs(float(row.get("action_gate_scale", 1.0)) - gate_scale) > 1e-6:
            continue
        if row.get("local_file") and Path(row["local_file"]).exists():
            out[row["scene_token"]] = row
    return out


def noaction_records() -> dict[str, dict[str, Any]]:
    manifest = json.loads(NOACTION_MANIFEST.read_text(encoding="utf-8"))
    rows = {}
    for row in manifest["records"]:
        if row.get("method_key") == "shifted_lognormal" and int(row.get("checkpoint_step", -1)) == 3000:
            if row.get("local_file") and Path(row["local_file"]).exists():
                rows[row["scene_token"]] = row
    return rows


def write_streaming_side_by_side(output_path: Path, input_paths: list[Path], labels: list[str], *, width: int = 256, height: int = 256) -> None:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterators = [iio.imiter(path) for path in input_paths]
    font = ImageFont.load_default()
    label_height = 24
    with imageio.get_writer(str(output_path), fps=FPS, codec="libx264", quality=8) as writer:
        for frame_idx, frames in enumerate(zip(*iterators)):
            if frame_idx >= TOTAL_FRAMES:
                break
            panels = []
            for frame, label in zip(frames, labels):
                resized = Image.fromarray(frame).resize((width, height), resample=Image.Resampling.BILINEAR)
                panel = Image.new("RGB", (width, height + label_height), color=(18, 18, 18))
                panel.paste(resized, (0, label_height))
                draw = ImageDraw.Draw(panel)
                draw.text((6, 6), label, fill=(245, 245, 245), font=font)
                panels.append(np.asarray(panel))
            writer.append_data(np.concatenate(panels, axis=1))


def best_low_gate_for_side_by_side() -> tuple[int, float]:
    selection = read_selection()
    rows = [row for row in read_csv(BENCHMARK_DIR / "gate_scale_summary.csv") if row.get("diagnostic_group") == "gate_scale_sweep"]
    candidate_gates = [gate for gate in selection["counterfactual_gate_scales"] if 0.0 < float(gate) < 1.0]
    selectable_steps = {step_to_int(step) for step in COUNTERFACTUAL_BASE_CHECKPOINTS}
    candidates = [
        row
        for row in rows
        if float(row["action_gate_scale"]) in set(map(float, candidate_gates))
        and int(row["checkpoint_step"]) in selectable_steps
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if 0.0 < float(row["action_gate_scale"]) <= LOW_GATE_MAX and int(row["checkpoint_step"]) in selectable_steps
        ]
    best = max(candidates, key=score_low_gate_row)
    return int(best["checkpoint_step"]), float(best["action_gate_scale"])


def build_side_by_side() -> None:
    checkpoint_step, best_gate = best_low_gate_for_side_by_side()
    gate_zero = records_by_scene(
        GATE_MANIFEST,
        method_key=METHOD_KEY,
        checkpoint_step=checkpoint_step,
        gate_scale=0.0,
        diagnostic_group="gate_scale_sweep",
    )
    gate_best = records_by_scene(
        GATE_MANIFEST,
        method_key=METHOD_KEY,
        checkpoint_step=checkpoint_step,
        gate_scale=best_gate,
        diagnostic_group="gate_scale_sweep",
    )
    gate_full = records_by_scene(
        GATE_MANIFEST,
        method_key=METHOD_KEY,
        checkpoint_step=checkpoint_step,
        gate_scale=1.0,
        diagnostic_group="gate_scale_sweep",
    )
    noaction = noaction_records()
    scenes = sorted(set(gate_zero) & set(gate_best) & set(gate_full) & set(noaction))
    SIDE_BY_SIDE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in scenes:
        source_path = SOURCE_DIR / gate_best[scene]["source_filename"]
        out_path = SIDE_BY_SIDE_DIR / f"scene_{scene}_{SIDE_BY_SIDE_STEM}_step{checkpoint_step:06d}_bestg{scale_label(best_gate)}.mp4"
        input_paths = [
            source_path,
            Path(noaction[scene]["local_file"]),
            Path(gate_zero[scene]["local_file"]),
            Path(gate_best[scene]["local_file"]),
            Path(gate_full[scene]["local_file"]),
        ]
        labels = ["GT", "no-action", "gate 0.0", f"gate {best_gate:g}", "gate 1.0"]
        write_streaming_side_by_side(out_path, input_paths, labels)
        records.append(
            {
                "scene_token": scene,
                "checkpoint_step": checkpoint_step,
                "best_gate_scale": best_gate,
                "output_path": str(out_path),
                "panels_left_to_right": labels,
            }
        )
    (SIDE_BY_SIDE_DIR / SIDE_BY_SIDE_MANIFEST_NAME).write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def decide_v2_support() -> dict[str, Any]:
    sensitivity = read_csv(BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv")
    gate_rows = read_csv(BENCHMARK_DIR / "gate_scale_summary.csv")
    metrics_by_key = {
        (int(row["checkpoint_step"]), float(row["action_gate_scale"])): row
        for row in gate_rows
        if row.get("diagnostic_group") == "gate_scale_sweep"
    }
    sensitivity_by_key: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in sensitivity:
        gate = float(row["action_gate_scale"])
        if 0.0 < gate <= LOW_GATE_MAX:
            sensitivity_by_key[(int(row["checkpoint_step"]), gate)].append(as_float(row, "mean_future_rgb_mae_correct_vs_mode"))

    candidates = []
    for key, values in sensitivity_by_key.items():
        metrics = metrics_by_key.get(key)
        if not metrics:
            continue
        mean_sensitivity = float(np.mean(values))
        candidate = {
            "checkpoint_step": key[0],
            "action_gate_scale": key[1],
            "mean_action_sensitivity_rgb_mae": mean_sensitivity,
            "sharpness_ratio": as_float(metrics, "mean_sharpness_ratio_generated_over_reference"),
            "fft_high_frequency_ratio": as_float(metrics, "mean_fft_high_frequency_energy_ratio_generated_over_reference"),
            "motion_ratio": as_float(metrics, "mean_motion_ratio_generated_over_reference"),
            "fvd_future": as_float(metrics, "fvd_future"),
        }
        candidate["passes"] = (
            candidate["mean_action_sensitivity_rgb_mae"] >= ACTION_SENSITIVITY_THRESHOLD
            and candidate["sharpness_ratio"] >= SHARPNESS_THRESHOLD
            and candidate["fft_high_frequency_ratio"] >= FFT_THRESHOLD
            and candidate["motion_ratio"] >= MOTION_THRESHOLD
        )
        candidates.append(candidate)
    candidates.sort(key=lambda row: (not row["passes"], -row["sharpness_ratio"], row["fvd_future"]))
    decision = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "action_sensitivity_rgb_mae": ACTION_SENSITIVITY_THRESHOLD,
            "sharpness_ratio": SHARPNESS_THRESHOLD,
            "fft_high_frequency_ratio": FFT_THRESHOLD,
            "motion_ratio": MOTION_THRESHOLD,
        },
        "v2_supported": any(row["passes"] for row in candidates),
        "candidates": candidates,
    }
    (BENCHMARK_DIR / "phase2_v2_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def train_v2_if_supported(*, force: bool = False) -> None:
    decision = decide_v2_support()
    if not force and not decision["v2_supported"]:
        print(json.dumps({"train_v2": "skipped", "reason": "gate sweep did not meet v2 thresholds"}, sort_keys=True))
        return
    log_path = LOG_DIR / "training" / f"{V2_RUN_NAME}.log"
    cmd = [
        str(MODAL),
        "run",
        V2_TRAIN_WRAPPER,
        "--run-name",
        V2_RUN_NAME,
        "--max-steps",
        "1000",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        "3e-5",
        "--action-injector-learning-rate",
        "3e-5",
        "--action-gate-learning-rate",
        "1e-5",
        "--hf-teacher-loss-weight",
        "0.15",
        "--action-residual-loss-weight",
        "0.001",
        "--action-gate-loss-weight",
        "0.001",
        "--weight-decay",
        "0.0",
        "--max-train-hours",
        "8.0",
        "--checkpoint-steps",
        "0,25,50,100,150,200,250,350,500,750,1000",
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--seed",
        str(SEED),
    ]
    run_command(cmd, log_path, env={"LTX_MODAL_GPU": "H100"})


def write_final_summary() -> None:
    rows = read_csv(BENCHMARK_DIR / "gate_scale_summary.csv")
    aggregate = read_csv(BENCHMARK_DIR / "gate_scale_aggregate_summary.csv")
    selection = read_selection() if SELECTION_PATH.exists() else {}
    decision = json.loads((BENCHMARK_DIR / "phase2_v2_decision.json").read_text(encoding="utf-8")) if (
        BENCHMARK_DIR / "phase2_v2_decision.json"
    ).exists() else {}
    best_sharp = max(rows, key=lambda row: as_float(row, "mean_sharpness_ratio_generated_over_reference")) if rows else {}
    best_fvd = min(rows, key=lambda row: as_float(row, "fvd_future", float("inf"))) if rows else {}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(BENCHMARK_DIR),
        "generated_dir": str(LOCAL_GENERATED_ROOT),
        "side_by_side_dir": str(SIDE_BY_SIDE_DIR),
        "gate_scale_summary_rows": len(rows),
        "gate_scale_aggregate_rows": aggregate,
        "selection": selection,
        "v2_decision": decision,
        "best_sharpness_row": best_sharp,
        "best_fvd_row": best_fvd,
    }
    (BENCHMARK_DIR / FINAL_SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 temporal bottleneck HF-teacher gate-scale diagnostics.")
    parser.add_argument(
        "--phase",
        choices=(
            "smoke",
            "generate_sweep",
            "download_sweep",
            "manifest_sweep",
            "metrics_sweep",
            "fvd_sweep",
            "summaries_sweep",
            "plots_sweep",
            "select_counterfactual",
            "generate_counterfactual",
            "download_counterfactual",
            "manifest_counterfactual",
            "sensitivity",
            "plots",
            "side_by_side",
            "decide_v2",
            "train_v2",
            "all_sweep",
            "all_counterfactual",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force-train-v2", action="store_true")
    args = parser.parse_args()

    if args.phase == "smoke":
        configs = smoke_configs()
        generate_all(configs, max_workers=min(args.max_workers, 2), seed=args.seed, limit=1)
        download_all(configs)
        manifest_path = build_manifest(
            configs,
            BENCHMARK_DIR / f"manifest_{MANIFEST_STEM}_smoke.json",
            seed=args.seed,
            limit=1,
            description="Smoke test for Phase 2 temporal bottleneck gate-scale sweep.",
        )
        print(json.dumps({"smoke_manifest": str(manifest_path.relative_to(ROOT))}, sort_keys=True))
        return

    if args.phase in {"generate_sweep", "all_sweep", "all"}:
        generate_all(gate_configs(), max_workers=args.max_workers, seed=args.seed, limit=args.limit)
    if args.phase in {"download_sweep", "all_sweep", "all"}:
        download_all(gate_configs())
    if args.phase in {"manifest_sweep", "metrics_sweep", "fvd_sweep", "summaries_sweep", "plots_sweep", "all_sweep", "all"}:
        build_manifest(
            gate_configs(),
            GATE_MANIFEST,
            seed=args.seed,
            limit=args.limit,
            description="Phase 2 temporal bottleneck HF-teacher inference-only gate-scale sweep.",
        )
    if args.phase in {"metrics_sweep", "all_sweep", "all"}:
        run_quality_metrics(GATE_MANIFEST)
    if args.phase in {"fvd_sweep", "all_sweep", "all"}:
        run_fvd(GATE_MANIFEST, run_id=FVD_RUN_ID)
    if args.phase in {"summaries_sweep", "plots_sweep", "select_counterfactual", "all_sweep", "all"}:
        rows = write_gate_scale_summary()
        aggregate_gate_rows(rows)
        select_counterfactual_settings()
    if args.phase in {"plots_sweep", "all_sweep"}:
        make_plots(include_sensitivity=False)

    if args.phase in {"select_counterfactual"}:
        print(json.dumps(read_selection(), indent=2, sort_keys=True))
        return

    if args.phase in {"generate_counterfactual", "all_counterfactual", "all"}:
        generate_all(counterfactual_configs(), max_workers=args.max_workers, seed=args.seed, limit=args.limit)
    if args.phase in {"download_counterfactual", "all_counterfactual", "all"}:
        download_all(counterfactual_configs())
    if args.phase in {"manifest_counterfactual", "sensitivity", "all_counterfactual", "all"}:
        build_manifest(
            counterfactual_configs(),
            COUNTERFACTUAL_MANIFEST,
            seed=args.seed,
            limit=args.limit,
            description="Phase 2 temporal bottleneck HF-teacher low-gate counterfactual action sensitivity suite.",
        )
    if args.phase in {"sensitivity", "all_counterfactual", "all"}:
        run_counterfactual_sensitivity(COUNTERFACTUAL_MANIFEST)
    if args.phase in {"plots", "all_counterfactual", "all"}:
        make_plots(include_sensitivity=True)
    if args.phase in {"side_by_side", "all_counterfactual", "all"}:
        build_side_by_side()
    if args.phase in {"decide_v2", "train_v2", "all_counterfactual", "all"}:
        decide_v2_support()
    if args.phase == "train_v2":
        train_v2_if_supported(force=args.force_train_v2)
    if args.phase == "all":
        train_v2_if_supported(force=args.force_train_v2)

    write_final_summary()
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
