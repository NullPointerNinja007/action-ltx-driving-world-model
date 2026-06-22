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

import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5"
SIDE_BY_SIDE_DIR = ROOT / "data" / "frame_temporal_bottleneck_hfteacher_side_by_side_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5"

RUN_NAME = "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_seed231_from_shifted_noaction_step003000_steps3000"
CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-framebottleneck-hfteacher-action-r16-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-framebottleneck-hfteacher-action-infer"
RUNS_ROOT = "distilled098_framebottleneck_hfteacher_action_lora_24fps_minterpolate_seed231_runs"
WRAPPER = "scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora.py"

NOACTION_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "noaction_shifted_timestep_longer_seed231_all5"
    / "manifest_noaction_shifted_timestep_longer_seed231_all5.json"
)
MIDBLOCK_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
    / "manifest_midblock_gated_xattn_checkpoint_sweep_seed231_all5.json"
)

CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121
FPS = 24
SEED = 231

CHECKPOINTS = [
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
COUNTERFACTUAL_CHECKPOINTS = ["step_000100", "step_000500", "step_003000"]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]


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
        if self.diagnostic_group == "checkpoint_sweep":
            return f"frame_temporal_bottleneck_hfteacher_step{self.checkpoint_step:06d}"
        return (
            f"frame_temporal_bottleneck_hfteacher_{self.diagnostic_group}_"
            f"step{self.checkpoint_step:06d}_{self.counterfactual_action_mode}_"
            f"g{scale_label(self.action_gate_scale)}"
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


def full_configs() -> list[EvalConfig]:
    configs = [EvalConfig("checkpoint_sweep", checkpoint) for checkpoint in CHECKPOINTS]
    for checkpoint in COUNTERFACTUAL_CHECKPOINTS:
        for mode in COUNTERFACTUAL_MODES:
            configs.append(EvalConfig("counterfactual_suite", checkpoint, counterfactual_action_mode=mode))
    return configs


def smoke_configs() -> list[EvalConfig]:
    return [
        EvalConfig("smoke", "step_000100", "correct", 0.0),
        EvalConfig("smoke", "step_000100", "correct", 1.0),
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
            print(json.dumps({"generated": future.result()}, sort_keys=True))


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
        print(json.dumps({"downloaded": str(download_one(config).relative_to(ROOT))}, sort_keys=True))


def local_file_for_record(record: dict[str, Any], config: EvalConfig) -> Path:
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / RUNS_ROOT / config.run_label / generated_relpath[len(prefix) :]


def build_manifest(configs: list[EvalConfig], *, seed: int, limit: int, suffix: str = "") -> Path:
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
                    "method_key": "frame_temporal_bottleneck_hfteacher",
                    "method_label": "Frame Temporal Bottleneck + HF Teacher",
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
    suffix_part = f"_{suffix}" if suffix else ""
    manifest_path = BENCHMARK_DIR / f"manifest_frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5{suffix_part}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "description": "Phase 2 temporal bottleneck HF-teacher checkpoint sweep.",
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
    return manifest_path


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
            "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5",
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
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/extract_midblock_gate_values_modal.py",
            "--run-name",
            RUN_NAME,
            "--checkpoints-json",
            json.dumps(CHECKPOINTS, separators=(",", ":")),
            "--output-csv",
            str(BENCHMARK_DIR / "gate_values_by_checkpoint.csv"),
            "--output-json",
            str(BENCHMARK_DIR / "gate_values_by_checkpoint.json"),
        ],
        LOG_DIR / "extract_gate_values.log",
        env={"LTX_GATE_CHECKPOINT_VOLUME": CHECKPOINT_VOLUME},
    )


def merge_summaries() -> list[dict[str, Any]]:
    summary_rows = read_csv(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {}
    fvd_path = BENCHMARK_DIR / "fvd_summary.csv"
    if fvd_path.exists():
        fvd_by_model = {row["model_mode"]: row for row in read_csv(fvd_path)}
    manifest = json.loads((BENCHMARK_DIR / "manifest_frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5.json").read_text(encoding="utf-8"))
    meta_by_model = {}
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
    rows.sort(key=lambda row: (str(row.get("diagnostic_group", "")), int(row.get("checkpoint_step", 0)), str(row.get("counterfactual_action_mode", ""))))
    write_csv(BENCHMARK_DIR / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv", rows)
    return rows


def build_gate_vs_metrics(rows: list[dict[str, Any]]) -> None:
    gate_path = BENCHMARK_DIR / "gate_values_by_checkpoint.csv"
    if not gate_path.exists():
        return
    gates_by_step: dict[int, dict[str, str]] = {}
    for row in read_csv(gate_path):
        if row.get("missing_action_injector") == "True":
            continue
        gates_by_step[int(row["checkpoint_step"])] = row
    out = []
    for row in rows:
        if row.get("diagnostic_group") != "checkpoint_sweep":
            continue
        gates = gates_by_step.get(int(row["checkpoint_step"]))
        if gates:
            out.append({**gates, **row})
    write_csv(BENCHMARK_DIR / "gate_vs_metrics.csv", out)


def plot_metric_over_checkpoints(rows: list[dict[str, Any]], metric: str, title: str, output_path: Path) -> None:
    selected = [row for row in rows if row.get("diagnostic_group") == "checkpoint_sweep"]
    selected.sort(key=lambda row: int(row["checkpoint_step"]))
    x = [int(row["checkpoint_step"]) for row in selected]
    y = [as_float(row, metric) for row in selected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gate_vs(metric: str, title: str, output_path: Path) -> None:
    path = BENCHMARK_DIR / "gate_vs_metrics.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    x = [as_float(row, "mean_abs_gate") for row in rows]
    y = [as_float(row, metric) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    rows = read_csv(BENCHMARK_DIR / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv")
    metric_dir = BENCHMARK_DIR / "metric_plots"
    for metric, title, name in [
        ("mean_future_psnr", "Future PSNR", "checkpoint_vs_psnr.png"),
        ("mean_future_global_ssim", "Future SSIM", "checkpoint_vs_ssim.png"),
        ("fvd_future", "FVD-Style Future Distance", "checkpoint_vs_fvd.png"),
        ("mean_sharpness_ratio_generated_over_reference", "Sharpness Ratio", "checkpoint_vs_sharpness_ratio.png"),
        ("mean_fft_high_frequency_energy_ratio_generated_over_reference", "FFT High-Frequency Ratio", "checkpoint_vs_fft_high_frequency_ratio.png"),
        ("mean_motion_ratio_generated_over_reference", "Motion Ratio", "checkpoint_vs_motion_ratio.png"),
        ("mean_temporal_delta_error_mae", "Temporal Delta Error", "checkpoint_vs_temporal_delta_error.png"),
    ]:
        plot_metric_over_checkpoints(rows, metric, title, metric_dir / name)
    plot_gate_vs("mean_sharpness_ratio_generated_over_reference", "Gate Magnitude vs Sharpness Ratio", metric_dir / "gate_magnitude_vs_sharpness.png")
    plot_gate_vs("fvd_future", "Gate Magnitude vs FVD-Style Future Distance", metric_dir / "gate_magnitude_vs_fvd.png")
    plot_gate_vs("mean_motion_ratio_generated_over_reference", "Gate Magnitude vs Motion Ratio", metric_dir / "gate_magnitude_vs_motion.png")
    plots = sorted(path.name for path in metric_dir.glob("*.png"))
    (metric_dir / "README.md").write_text(
        "# Temporal Bottleneck HF-Teacher Metric Plots\n\n" + "\n".join(f"- [{name}]({name})" for name in plots) + "\n",
        encoding="utf-8",
    )


def best_checkpoint_step() -> int:
    rows = [row for row in read_csv(BENCHMARK_DIR / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv") if row.get("diagnostic_group") == "checkpoint_sweep"]
    valid = [row for row in rows if as_float(row, "mean_sharpness_ratio_generated_over_reference") >= 0.28]
    pool = valid or rows
    best = min(pool, key=lambda row: as_float(row, "fvd_future") if not math.isnan(as_float(row, "fvd_future")) else float("inf"))
    return int(best["checkpoint_step"])


def records_by_scene(manifest_path: Path, *, method_key: str | None = None, checkpoint_step: int | None = None) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for row in manifest["records"]:
        if method_key and row.get("method_key") != method_key:
            continue
        if checkpoint_step is not None and int(row.get("checkpoint_step", -1)) != checkpoint_step:
            continue
        if row.get("local_file") and Path(row["local_file"]).exists():
            out[row["scene_token"]] = row
    return out


def read_video(path: Path) -> np.ndarray:
    frames = iio.imread(path)
    if frames.shape[0] < TOTAL_FRAMES:
        raise ValueError(f"{path} has {frames.shape[0]} frames, expected at least {TOTAL_FRAMES}")
    return frames[:TOTAL_FRAMES]


def resize_panel(frames: np.ndarray, width: int = 256, height: int = 256) -> np.ndarray:
    from PIL import Image

    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), resample=Image.Resampling.BILINEAR)) for frame in frames],
        axis=0,
    )


def write_streaming_side_by_side(output_path: Path, input_paths: list[Path], *, width: int = 256, height: int = 256) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterators = [iio.imiter(path) for path in input_paths]
    with imageio.get_writer(str(output_path), fps=FPS, codec="libx264", quality=8) as writer:
        for frame_idx, frames in enumerate(zip(*iterators)):
            if frame_idx >= TOTAL_FRAMES:
                break
            panels = [
                np.asarray(Image.fromarray(frame).resize((width, height), resample=Image.Resampling.BILINEAR))
                for frame in frames
            ]
            writer.append_data(np.concatenate(panels, axis=1))


def build_side_by_side() -> None:
    new_manifest = BENCHMARK_DIR / "manifest_frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5.json"
    step = best_checkpoint_step()
    new_rows = records_by_scene(new_manifest, method_key="frame_temporal_bottleneck_hfteacher", checkpoint_step=step)
    noaction_rows = records_by_scene(NOACTION_MANIFEST, method_key="shifted_lognormal", checkpoint_step=3000)
    midblock_rows = records_by_scene(MIDBLOCK_MANIFEST, method_key="frame_midblock_gated_xattn", checkpoint_step=100)
    scenes = sorted(set(new_rows) & set(noaction_rows) & set(midblock_rows))
    SIDE_BY_SIDE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for scene in scenes:
        source_path = SOURCE_DIR / new_rows[scene]["source_filename"]
        out_path = SIDE_BY_SIDE_DIR / f"scene_{scene}_gt_noaction_midblock_temporal_bottleneck_step{step:06d}.mp4"
        write_streaming_side_by_side(
            out_path,
            [
                source_path,
                Path(noaction_rows[scene]["local_file"]),
                Path(midblock_rows[scene]["local_file"]),
                Path(new_rows[scene]["local_file"]),
            ],
        )
        records.append(
            {
                "scene_token": scene,
                "checkpoint_step": step,
                "output_path": str(out_path),
                "panels_left_to_right": ["ground_truth", "corrected_noaction_step3000", "midblock_step100", f"temporal_bottleneck_step{step:06d}"],
            }
        )
    (SIDE_BY_SIDE_DIR / "manifest_temporal_bottleneck_side_by_side.json").write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_final_summary() -> None:
    rows = read_csv(BENCHMARK_DIR / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv")
    best_step = best_checkpoint_step()
    best_row = next(row for row in rows if row.get("diagnostic_group") == "checkpoint_sweep" and int(row["checkpoint_step"]) == best_step)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_dir": str(BENCHMARK_DIR),
        "side_by_side_dir": str(SIDE_BY_SIDE_DIR),
        "selected_best_checkpoint_step": best_step,
        "selection_rule": "lowest FVD among checkpoints with sharpness_ratio >= 0.28; fallback lowest FVD",
        "best_checkpoint_metrics": best_row,
    }
    (BENCHMARK_DIR / "frame_temporal_bottleneck_hfteacher_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 temporal bottleneck HF-teacher checkpoint sweep.")
    parser.add_argument(
        "--phase",
        choices=("smoke", "generate", "download", "manifest", "metrics", "fvd", "sensitivity", "gates", "summaries", "plots", "side_by_side", "all"),
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    configs = smoke_configs() if args.phase == "smoke" else full_configs()
    suffix = "smoke" if args.phase == "smoke" else ""
    if args.phase in {"smoke", "generate", "all"}:
        generate_all(configs, max_workers=args.max_workers, seed=args.seed, limit=args.limit if args.phase != "smoke" else 1)
    if args.phase in {"smoke", "download", "all"}:
        download_all(configs)
    manifest_path = BENCHMARK_DIR / f"manifest_frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5{'_smoke' if suffix else ''}.json"
    if args.phase in {"smoke", "manifest", "metrics", "fvd", "sensitivity", "all"}:
        manifest_path = build_manifest(configs, seed=args.seed, limit=args.limit if args.phase != "smoke" else 1, suffix=suffix)
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
        rows = merge_summaries()
        build_gate_vs_metrics(rows)
    if args.phase in {"plots", "all"}:
        make_plots()
    if args.phase in {"side_by_side", "all"}:
        build_side_by_side()
    if args.phase in {"summaries", "all"}:
        write_final_summary()
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
