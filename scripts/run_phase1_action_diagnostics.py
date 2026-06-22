from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
LOCAL_GENERATED_ROOT = ROOT / "data" / "phase1_action_diagnostics_midblock_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "phase1_action_diagnostics_midblock_seed231_all5"
SIDE_BY_SIDE_DIR = ROOT / "data" / "phase1_action_diagnostics_side_by_side_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "phase1_action_diagnostics_midblock_seed231_all5"

RUN_NAME = "ltx2b_dist098_waymo24_frame_midblock_gated_xattn_seed231_from_shifted_noaction_step003000_steps3000"
CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-midxattn-r16-shift-infer"
RUNS_ROOT = "distilled098_framemidxattn_action_lora_24fps_minterpolate_seed231_shifted_runs"
WRAPPER = "scripts/wrappers/generate_waymo24_distilled_frame_midblock_gated_xattn_action_minterpolate_lora.py"

CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121
FPS = 24
SEED = 231

GATE_SWEEP_CHECKPOINTS = ["step_000100", "step_000250"]
GATE_SCALES = [0.0, 0.1, 0.25, 0.5, 1.0]
COUNTERFACTUAL_CHECKPOINTS = ["step_000100", "step_000250"]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]
GATE_INSPECTION_CHECKPOINTS = [
    "step_000000_base_reference",
    "step_000100",
    "step_000250",
    "step_000500",
    "step_001000",
    "step_001500",
    "step_002000",
    "step_002500",
    "step_003000",
]


@dataclass(frozen=True)
class DiagnosticConfig:
    diagnostic_group: str
    checkpoint_name: str
    counterfactual_action_mode: str
    action_gate_scale: float
    action_vector_scale: float = 1.0
    counterfactual_rotation: int = 1

    @property
    def checkpoint_step(self) -> int:
        return step_to_int(self.checkpoint_name)

    @property
    def model_mode(self) -> str:
        return (
            f"phase1_{self.diagnostic_group}_step{self.checkpoint_step:06d}_"
            f"{self.counterfactual_action_mode}_g{scale_label(self.action_gate_scale)}"
        )

    @property
    def run_label(self) -> str:
        return f"{self.model_mode}_v{scale_label(self.action_vector_scale)}_seed{SEED}_all5"


def step_to_int(step: str) -> int:
    if step == "step_000000_base_reference":
        return 0
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse checkpoint step: {step}")
    return int(match.group(1))


def scale_label(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def full_configs() -> list[DiagnosticConfig]:
    configs: list[DiagnosticConfig] = []
    for checkpoint in GATE_SWEEP_CHECKPOINTS:
        for gate_scale in GATE_SCALES:
            configs.append(
                DiagnosticConfig(
                    diagnostic_group="gate_scale_sweep",
                    checkpoint_name=checkpoint,
                    counterfactual_action_mode="correct",
                    action_gate_scale=gate_scale,
                )
            )
    for checkpoint in COUNTERFACTUAL_CHECKPOINTS:
        for mode in COUNTERFACTUAL_MODES:
            configs.append(
                DiagnosticConfig(
                    diagnostic_group="counterfactual_suite",
                    checkpoint_name=checkpoint,
                    counterfactual_action_mode=mode,
                    action_gate_scale=1.0,
                )
            )
    return configs


def smoke_configs() -> list[DiagnosticConfig]:
    return [
        DiagnosticConfig("smoke", "step_000100", "correct", 0.0),
        DiagnosticConfig("smoke", "step_000100", "correct", 1.0),
        DiagnosticConfig("smoke", "step_000100", "zero", 0.0),
        DiagnosticConfig("smoke", "step_000100", "zero", 1.0),
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


def generate_one(config: DiagnosticConfig, *, seed: int, limit: int) -> str:
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


def generate_all(configs: list[DiagnosticConfig], *, max_workers: int, seed: int, limit: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, config, seed=seed, limit=limit) for config in configs]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True))


def download_one(config: DiagnosticConfig) -> Path:
    remote_path = f"{RUNS_ROOT}/{config.run_label}"
    local_dest = LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label
    local_dest.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "download" / f"{config.run_label}.log"

    summary_remote = f"{remote_path}/run_summary.json"
    summary_local = local_dest / "run_summary.json"
    if not summary_local.exists():
        run_command(
            [
                str(MODAL),
                "volume",
                "get",
                "--force",
                ARTIFACT_VOLUME,
                summary_remote,
                str(summary_local),
            ],
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
            [
                str(MODAL),
                "volume",
                "get",
                "--force",
                ARTIFACT_VOLUME,
                generated_relpath,
                str(video_local),
            ],
            log_path,
        )
    return local_dest


def download_all(configs: list[DiagnosticConfig]) -> None:
    for config in configs:
        path = download_one(config)
        print(json.dumps({"downloaded": str(path.relative_to(ROOT))}, sort_keys=True))


def local_file_for_record(record: dict[str, Any], config: DiagnosticConfig) -> Path:
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label / generated_relpath[len(prefix) :]


def build_manifest(configs: list[DiagnosticConfig], *, seed: int, limit: int, name_suffix: str = "") -> Path:
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
                    "method_key": "frame_midblock_gated_xattn",
                    "method_label": "Frame Mid-Block Gated XAttn",
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

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    manifest_path = BENCHMARK_DIR / f"manifest_phase1_action_diagnostics{suffix}.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Phase 1 no-training diagnostics for middle-block gated action conditioning.",
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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def run_quality_metrics(manifest_path: Path) -> None:
    run_command(
        [
            str(PYTHON),
            "pipelines/evaluation/benchmark_video_quality.py",
            "--manifest",
            str(manifest_path),
            "--source-dir",
            str(SOURCE_DIR),
            "--output-dir",
            str(BENCHMARK_DIR),
        ],
        LOG_DIR / "benchmark_video_quality.log",
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
            "phase1_action_diagnostics_midblock_seed231_all5",
        ],
        LOG_DIR / "compute_fvd.log",
    )


def run_counterfactual_sensitivity(manifest_path: Path) -> None:
    run_command(
        [
            str(PYTHON),
            "scripts/compute_phase1_counterfactual_sensitivity.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(BENCHMARK_DIR),
        ],
        LOG_DIR / "counterfactual_sensitivity.log",
    )


def run_gate_extraction() -> None:
    checkpoints_json = json.dumps(GATE_INSPECTION_CHECKPOINTS, separators=(",", ":"))
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/extract_midblock_gate_values_modal.py",
            "--run-name",
            RUN_NAME,
            "--checkpoints-json",
            checkpoints_json,
            "--output-csv",
            str(BENCHMARK_DIR / "gate_values_by_checkpoint.csv"),
            "--output-json",
            str(BENCHMARK_DIR / "gate_values_by_checkpoint.json"),
        ],
        LOG_DIR / "extract_gate_values.log",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
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


def as_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return float("nan")
    return float(raw)


def merge_fvd() -> dict[str, dict[str, str]]:
    fvd_path = BENCHMARK_DIR / "fvd_summary.csv"
    if not fvd_path.exists():
        return {}
    return {row["model_mode"]: row for row in read_csv(fvd_path)}


def build_gate_scale_summary() -> list[dict[str, Any]]:
    summary_rows = read_csv(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = merge_fvd()
    manifest = json.loads((BENCHMARK_DIR / "manifest_phase1_action_diagnostics.json").read_text(encoding="utf-8"))
    metadata_by_model = {}
    for record in manifest["records"]:
        metadata_by_model[record["model_mode"]] = {
            "diagnostic_group": record["diagnostic_group"],
            "checkpoint_step": record["checkpoint_step"],
            "checkpoint_name": record["checkpoint_name"],
            "action_gate_scale": record["action_gate_scale"],
            "action_vector_scale": record["action_vector_scale"],
            "counterfactual_action_mode": record["counterfactual_action_mode"],
        }

    merged: list[dict[str, Any]] = []
    for row in summary_rows:
        model_mode = row["model_mode"]
        meta = metadata_by_model.get(model_mode, {})
        out: dict[str, Any] = {**meta, **row}
        fvd = fvd_by_model.get(model_mode, {})
        if fvd:
            out.update(
                {
                    "fvd_future": fvd.get("fvd_future", ""),
                    "fvd_backend": fvd.get("fvd_backend", ""),
                    "fvd_num_videos": fvd.get("fvd_num_videos", ""),
                    "fvd_num_frames": fvd.get("fvd_num_frames", ""),
                    "fvd_size": fvd.get("fvd_size", ""),
                }
            )
        merged.append(out)
    merged.sort(
        key=lambda item: (
            str(item.get("diagnostic_group", "")),
            int(item.get("checkpoint_step", 0)),
            float(item.get("action_gate_scale", 0.0)),
            str(item.get("counterfactual_action_mode", "")),
        )
    )
    write_csv(BENCHMARK_DIR / "phase1_model_summary_with_fvd.csv", merged)
    gate_rows = [row for row in merged if row.get("diagnostic_group") == "gate_scale_sweep"]
    write_csv(BENCHMARK_DIR / "gate_scale_summary.csv", gate_rows)
    return merged


def build_gate_vs_metrics() -> list[dict[str, Any]]:
    gate_path = BENCHMARK_DIR / "gate_values_by_checkpoint.csv"
    if not gate_path.exists():
        return []
    gate_rows = read_csv(gate_path)
    gate_by_step: dict[int, dict[str, Any]] = {}
    for row in gate_rows:
        if row.get("missing_action_injector") == "True":
            continue
        step = int(row["checkpoint_step"])
        gate_by_step[step] = {
            "checkpoint_step": step,
            "mean_abs_gate": row["mean_abs_gate"],
            "max_abs_gate": row["max_abs_gate"],
            "enabled_block_indices": row["enabled_block_indices"],
        }

    reference_summary = (
        ROOT
        / "data"
        / "benchmarks"
        / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
        / "midblock_gated_xattn_summary_with_fvd.csv"
    )
    if not reference_summary.exists():
        return []
    metric_rows = read_csv(reference_summary)
    out_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        step = int(row["checkpoint_step"])
        gates = gate_by_step.get(step)
        if not gates:
            continue
        out_rows.append({**gates, **row})
    write_csv(BENCHMARK_DIR / "gate_vs_metrics.csv", out_rows)
    return out_rows


def plot_line(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    output_path: Path,
    xlabel: str = "Action gate scale",
    ylabel: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    steps = sorted({int(row["checkpoint_step"]) for row in rows})
    for step in steps:
        selected = [row for row in rows if int(row["checkpoint_step"]) == step]
        selected.sort(key=lambda row: float(row["action_gate_scale"]))
        x = [float(row["action_gate_scale"]) for row in selected]
        y = [as_float(row, metric) for row in selected]
        if not x or all(math.isnan(value) for value in y):
            continue
        ax.plot(x, y, marker="o", linewidth=2, label=f"step {step}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_counterfactual_sensitivity() -> None:
    summary_path = BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv"
    if not summary_path.exists():
        return
    rows = read_csv(summary_path)
    output_path = BENCHMARK_DIR / "metric_plots" / "counterfactual_sensitivity_by_mode.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modes = ["zero", "shuffled", "reversed_future"]
    steps = sorted({int(row["checkpoint_step"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title in [
        (axes[0], "mean_future_rgb_mae_correct_vs_mode", "Future RGB MAE vs Correct"),
        (axes[1], "mean_future_temporal_delta_mae_correct_vs_mode", "Future Temporal Delta MAE vs Correct"),
    ]:
        width = 0.35
        positions = np.arange(len(modes))
        for idx, step in enumerate(steps):
            selected = {row["comparison_mode"]: row for row in rows if int(row["checkpoint_step"]) == step}
            values = [as_float(selected.get(mode, {}), metric) for mode in modes]
            ax.bar(positions + (idx - (len(steps) - 1) / 2) * width, values, width=width, label=f"step {step}")
        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(modes, rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gate_vs(metric: str, title: str, output_path: Path) -> None:
    rows_path = BENCHMARK_DIR / "gate_vs_metrics.csv"
    if not rows_path.exists():
        return
    rows = read_csv(rows_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = [as_float(row, "mean_abs_gate") for row in rows]
    y = [as_float(row, metric) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=55)
    for row, x_value, y_value in zip(rows, x, y):
        if not math.isnan(x_value) and not math.isnan(y_value):
            ax.annotate(str(row["checkpoint_step"]), (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("Mean absolute learned gate")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_plots() -> None:
    gate_rows_path = BENCHMARK_DIR / "gate_scale_summary.csv"
    if not gate_rows_path.exists():
        raise FileNotFoundError(gate_rows_path)
    rows = read_csv(gate_rows_path)
    metric_dir = BENCHMARK_DIR / "metric_plots"
    plot_line(rows, metric="mean_future_psnr", title="Gate Scale vs Future PSNR", output_path=metric_dir / "gate_scale_vs_psnr.png")
    plot_line(rows, metric="mean_future_global_ssim", title="Gate Scale vs Future SSIM", output_path=metric_dir / "gate_scale_vs_ssim.png")
    plot_line(rows, metric="fvd_future", title="Gate Scale vs FVD-Style Future Distance", output_path=metric_dir / "gate_scale_vs_fvd.png")
    plot_line(
        rows,
        metric="mean_sharpness_ratio_generated_over_reference",
        title="Gate Scale vs Sharpness Ratio",
        output_path=metric_dir / "gate_scale_vs_sharpness_ratio.png",
    )
    plot_line(
        rows,
        metric="mean_motion_ratio_generated_over_reference",
        title="Gate Scale vs Motion Ratio",
        output_path=metric_dir / "gate_scale_vs_motion_ratio.png",
    )
    plot_line(
        rows,
        metric="mean_fft_high_frequency_energy_ratio_generated_over_reference",
        title="Gate Scale vs FFT High-Frequency Ratio",
        output_path=metric_dir / "gate_scale_vs_fft_high_frequency_ratio.png",
    )
    plot_counterfactual_sensitivity()
    plot_gate_vs(
        "mean_sharpness_ratio_generated_over_reference",
        "Gate Magnitude vs Sharpness Ratio",
        metric_dir / "gate_magnitude_vs_sharpness.png",
    )
    plot_gate_vs("fvd_future", "Gate Magnitude vs FVD-Style Future Distance", metric_dir / "gate_magnitude_vs_fvd.png")
    plot_gate_vs(
        "mean_motion_ratio_generated_over_reference",
        "Gate Magnitude vs Motion Ratio",
        metric_dir / "gate_magnitude_vs_motion.png",
    )
    readme = metric_dir / "README.md"
    plots = sorted(path.name for path in metric_dir.glob("*.png"))
    readme.write_text(
        "# Phase 1 Action Diagnostic Metric Plots\n\n" + "\n".join(f"- [{name}]({name})" for name in plots) + "\n",
        encoding="utf-8",
    )


def read_video(path: Path, max_frames: int = TOTAL_FRAMES) -> np.ndarray:
    frames = iio.imread(path)
    if frames.shape[0] < max_frames:
        raise ValueError(f"{path} has {frames.shape[0]} frames, expected at least {max_frames}")
    return frames[:max_frames]


def resize_panel(frames: np.ndarray, width: int = 256, height: int = 256) -> np.ndarray:
    from PIL import Image

    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), resample=Image.Resampling.BILINEAR)) for frame in frames],
        axis=0,
    )


def build_side_by_side() -> None:
    manifest = json.loads((BENCHMARK_DIR / "manifest_phase1_action_diagnostics.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    by_key = {
        (
            row["scene_token"],
            row["diagnostic_group"],
            row["checkpoint_name"],
            row["counterfactual_action_mode"],
            float(row["action_gate_scale"]),
        ): row
        for row in records
    }
    scenes = sorted({row["scene_token"] for row in records})
    panel_specs = [
        ("GT", "source", "", "correct", 0.0),
        ("gate_scale_0.0", "gate_scale_sweep", "step_000100", "correct", 0.0),
        ("gate_scale_0.25", "gate_scale_sweep", "step_000100", "correct", 0.25),
        ("gate_scale_1.0", "gate_scale_sweep", "step_000100", "correct", 1.0),
        ("correct_action", "counterfactual_suite", "step_000100", "correct", 1.0),
        ("zero_action", "counterfactual_suite", "step_000100", "zero", 1.0),
        ("shuffled_action", "counterfactual_suite", "step_000100", "shuffled", 1.0),
        ("reversed_future_action", "counterfactual_suite", "step_000100", "reversed_future", 1.0),
    ]
    SIDE_BY_SIDE_DIR.mkdir(parents=True, exist_ok=True)
    out_records: list[dict[str, Any]] = []
    for scene in scenes:
        panels = []
        labels = []
        source_record = next(row for row in records if row["scene_token"] == scene)
        for label, group, checkpoint, mode, gate_scale in panel_specs:
            labels.append(label)
            if group == "source":
                path = SOURCE_DIR / source_record["source_filename"]
            else:
                record = by_key[(scene, group, checkpoint, mode, gate_scale)]
                path = Path(record["local_file"])
            panels.append(resize_panel(read_video(path)))
        min_frames = min(panel.shape[0] for panel in panels)
        grid = np.concatenate([panel[:min_frames] for panel in panels], axis=2)
        out_path = SIDE_BY_SIDE_DIR / f"phase1_midblock_diagnostics_scene_{scene}_step000100.mp4"
        iio.imwrite(out_path, grid, fps=FPS, codec="libx264", quality=8)
        out_records.append({"scene_token": scene, "output_path": str(out_path), "panels_left_to_right": labels})
    manifest_path = SIDE_BY_SIDE_DIR / "manifest_phase1_side_by_side.json"
    manifest_path.write_text(json.dumps({"records": out_records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_final_summary() -> None:
    rows = read_csv(BENCHMARK_DIR / "phase1_model_summary_with_fvd.csv")
    gate_rows = [row for row in rows if row.get("diagnostic_group") == "gate_scale_sweep"]
    best_gate_by_metric: dict[str, Any] = {}
    for metric, higher in [
        ("mean_future_psnr", True),
        ("mean_future_global_ssim", True),
        ("fvd_future", False),
        ("mean_sharpness_ratio_generated_over_reference", True),
        ("mean_motion_ratio_generated_over_reference", True),
        ("mean_fft_high_frequency_energy_ratio_generated_over_reference", True),
    ]:
        valid = [row for row in gate_rows if str(row.get(metric, "")) not in {"", "nan"}]
        if not valid:
            continue
        selected = max(valid, key=lambda row: as_float(row, metric)) if higher else min(valid, key=lambda row: as_float(row, metric))
        best_gate_by_metric[metric] = {
            "checkpoint_step": int(selected["checkpoint_step"]),
            "action_gate_scale": float(selected["action_gate_scale"]),
            "model_mode": selected["model_mode"],
            "value": as_float(selected, metric),
        }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(BENCHMARK_DIR),
        "side_by_side_dir": str(SIDE_BY_SIDE_DIR),
        "num_diagnostic_model_modes": len(rows),
        "best_gate_scale_by_metric": best_gate_by_metric,
        "decision_rules": [
            "If gate_scale=0.0 is sharper and lower-FVD than gate_scale=1.0, action injection is directly degrading visuals.",
            "If counterfactual outputs are nearly identical, the model is not using actions meaningfully.",
            "If counterfactual outputs differ but all are blurry, the action pathway is active but harmful.",
        ],
    }
    (BENCHMARK_DIR / "phase1_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 middle-block action diagnostics without training.")
    parser.add_argument(
        "--phase",
        choices=("smoke", "generate", "download", "manifest", "metrics", "fvd", "sensitivity", "gates", "plots", "side_by_side", "summaries", "all"),
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    configs = smoke_configs() if args.phase == "smoke" else full_configs()
    manifest_name_suffix = "smoke" if args.phase == "smoke" else ""

    if args.phase in {"smoke", "generate", "all"}:
        generate_all(configs, max_workers=args.max_workers, seed=args.seed, limit=args.limit if args.phase != "smoke" else 1)
    if args.phase in {"smoke", "download", "all"}:
        download_all(configs)
    manifest_path = BENCHMARK_DIR / f"manifest_phase1_action_diagnostics{'_smoke' if args.phase == 'smoke' else ''}.json"
    if args.phase in {"smoke", "manifest", "metrics", "fvd", "sensitivity", "all"}:
        manifest_path = build_manifest(
            configs,
            seed=args.seed,
            limit=args.limit if args.phase != "smoke" else 1,
            name_suffix=manifest_name_suffix,
        )
        print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT))}, sort_keys=True))
    if args.phase in {"metrics", "all"}:
        run_quality_metrics(manifest_path)
    if args.phase in {"fvd", "all"}:
        run_fvd(manifest_path)
    if args.phase in {"sensitivity", "all"}:
        run_counterfactual_sensitivity(manifest_path)
    if args.phase in {"gates", "all"}:
        run_gate_extraction()
    if args.phase in {"summaries", "plots", "side_by_side", "all"}:
        build_gate_scale_summary()
        build_gate_vs_metrics()
    if args.phase in {"plots", "all"}:
        make_plots()
    if args.phase in {"side_by_side", "all"}:
        build_side_by_side()
    if args.phase in {"summaries", "all"}:
        write_final_summary()
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
