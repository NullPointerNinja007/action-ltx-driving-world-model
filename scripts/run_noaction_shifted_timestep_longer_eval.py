from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
LOCAL_GENERATED_ROOT = ROOT / "data" / "noaction_cleanctx_shifted_timestep_ablation_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "noaction_shifted_timestep_longer_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "noaction_shifted_timestep_longer_seed231_all5"

UNIFORM_RUN_NAME = "ltx2b_distilled098_waymo24_noaction_visual_lora_r16_seed231_full7992_lr5e6_2epochs_steps15984"
SHIFTED_SHORT_RUN_NAME = "ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps1000"
SHIFTED_LONG_RUN_NAME = "ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000"

UNIFORM_CHECKPOINTS = [
    "base",
    "step_000500",
    "step_001000",
    "step_002000",
    "step_003000",
    "step_004000",
    "step_006000",
]
SHIFTED_SHORT_CHECKPOINTS = [
    "base",
    "step_000100",
    "step_000250",
    "step_000500",
    "step_001000",
]
SHIFTED_LONG_CHECKPOINTS = [
    "step_001500",
    "step_002000",
    "step_002500",
    "step_003000",
    "step_004000",
    "step_005000",
    "step_006000",
]


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    wrapper: str
    checkpoint_volume: str
    artifact_volume: str
    runs_root: str
    lora_run_name: str
    checkpoints: list[str]
    generate: bool


CONDITIONS = [
    Condition(
        key="uniform_cleanctx",
        label="Uniform timestep, clean context",
        wrapper="scripts/wrappers/generate_waymo24_minterpolate_distilled_lora_2epoch.py",
        checkpoint_volume="ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-noaction-cleanctx-infer",
        runs_root="distilled098_noaction_uniform_cleanctx_24fps_minterpolate_seed231_runs",
        lora_run_name=UNIFORM_RUN_NAME,
        checkpoints=UNIFORM_CHECKPOINTS,
        generate=False,
    ),
    Condition(
        key="shifted_lognormal",
        label="Shifted log-normal timestep, clean context",
        wrapper="scripts/wrappers/generate_waymo24_minterpolate_distilled_lora_shifted_timestep_ablation.py",
        checkpoint_volume="ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer",
        runs_root="distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs",
        lora_run_name=SHIFTED_SHORT_RUN_NAME,
        checkpoints=SHIFTED_SHORT_CHECKPOINTS,
        generate=False,
    ),
    Condition(
        key="shifted_lognormal",
        label="Shifted log-normal timestep, clean context",
        wrapper="scripts/wrappers/generate_waymo24_minterpolate_distilled_lora_shifted_timestep_ablation.py",
        checkpoint_volume="ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer",
        runs_root="distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs",
        lora_run_name=SHIFTED_LONG_RUN_NAME,
        checkpoints=SHIFTED_LONG_CHECKPOINTS,
        generate=True,
    ),
]


def step_to_int(step: str) -> int:
    if step == "base":
        return 0
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse step from {step}")
    return int(match.group(1))


def step_label(step: str) -> str:
    if step == "base":
        return "base"
    return step.replace("step_", "step")


def run_label(condition: Condition, step: str, seed: int) -> str:
    return f"{condition.key}_{step_label(step)}_seed{seed}_all5_cleanctx"


def modal_env(condition: Condition) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LTX_CHECKPOINT_VOLUME_NAME": condition.checkpoint_volume,
            "LTX_ARTIFACTS_VOLUME_NAME": condition.artifact_volume,
            "LTX_RUNS_ROOT": condition.runs_root,
            "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
            "LTX_MODAL_GPU": "A100",
        }
    )
    return env


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def generate_one(condition: Condition, step: str, seed: int) -> tuple[str, str]:
    label = run_label(condition, step, seed)
    log_path = LOG_DIR / "generation" / f"{label}.log"
    if log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore"):
        return condition.key, step
    cmd = [
        str(MODAL),
        "run",
        condition.wrapper,
        "--limit",
        "5",
        "--seed",
        str(seed),
        "--lora-step",
        step,
        "--lora-run-name",
        condition.lora_run_name,
        "--run-label",
        label,
        "--base-label",
        "base_distilled_no_lora",
    ]
    run_command(cmd, log_path, env=modal_env(condition))
    return condition.key, step


def generate_all(max_workers: int, seed: int) -> None:
    jobs = [(condition, step) for condition in CONDITIONS if condition.generate for step in condition.checkpoints]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, condition, step, seed) for condition, step in jobs]
        for future in as_completed(futures):
            condition_key, step = future.result()
            print(json.dumps({"generated": condition_key, "step": step}, sort_keys=True))


def download_one(condition: Condition, step: str, seed: int) -> Path:
    label = run_label(condition, step, seed)
    remote_path = f"{condition.runs_root}/{label}"
    local_dest = LOCAL_GENERATED_ROOT / condition.runs_root / label
    local_dest.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "download" / f"{label}.log"

    summary_remote = f"{remote_path}/run_summary.json"
    summary_local = local_dest / "run_summary.json"
    run_command(
        [
            str(MODAL),
            "volume",
            "get",
            "--force",
            condition.artifact_volume,
            summary_remote,
            str(summary_local),
        ],
        log_path,
    )

    summary = json.loads(summary_local.read_text(encoding="utf-8"))
    prefix = f"{condition.runs_root}/{label}/"
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
                condition.artifact_volume,
                generated_relpath,
                str(video_local),
            ],
            log_path,
        )
    return local_dest


def download_all(seed: int) -> None:
    for condition in CONDITIONS:
        if not condition.generate:
            continue
        for step in condition.checkpoints:
            path = download_one(condition, step, seed)
            print(json.dumps({"downloaded": str(path.relative_to(ROOT))}, sort_keys=True))


def local_file_for_record(record: dict[str, Any], condition: Condition, step: str, seed: int) -> Path:
    label = run_label(condition, step, seed)
    prefix = f"{condition.runs_root}/{label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / condition.runs_root / label / generated_relpath[len(prefix) :]


def build_manifest(seed: int) -> Path:
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for step in condition.checkpoints:
            label = run_label(condition, step, seed)
            summary_path = LOCAL_GENERATED_ROOT / condition.runs_root / label / "run_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for record in summary["results"]:
                local_file = local_file_for_record(record, condition, step, seed)
                if not local_file.exists():
                    raise FileNotFoundError(local_file)
                row = dict(record)
                row["local_file"] = str(local_file)
                row["method_key"] = condition.key
                row["method_label"] = condition.label
                row["checkpoint_step"] = step_to_int(step)
                row["checkpoint_name"] = step
                row["model_mode"] = f"{condition.key}_{step_label(step)}"
                row["using_lora"] = step != "base"
                row["source_lora_run_name"] = condition.lora_run_name
                records.append(row)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Longer shifted-lognormal no-action continuation compared against clean-context uniform baseline.",
        "seed": seed,
        "context_frames": 49,
        "future_frames": 72,
        "total_frames": 121,
        "fps": 24,
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "records": records,
    }
    manifest_path = BENCHMARK_DIR / "manifest_noaction_shifted_timestep_longer_seed231_all5.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def run_quality_metrics(manifest_path: Path) -> None:
    cmd = [
        str(PYTHON),
        "pipelines/evaluation/benchmark_video_quality.py",
        "--manifest",
        str(manifest_path),
        "--source-dir",
        str(SOURCE_DIR),
        "--output-dir",
        str(BENCHMARK_DIR),
    ]
    run_command(cmd, LOG_DIR / "benchmark_video_quality.log")


def run_fvd(manifest_path: Path) -> None:
    cmd = [
        str(MODAL),
        "run",
        "scripts/compute_action_fvd_modal.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(BENCHMARK_DIR),
        "--run-id",
        "noaction_shifted_timestep_longer_seed231_all5",
    ]
    run_command(cmd, LOG_DIR / "compute_fvd.log")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def parse_model_mode(model_mode: str) -> tuple[str, int]:
    if model_mode.startswith("uniform_cleanctx_"):
        suffix = model_mode[len("uniform_cleanctx_") :]
        return "uniform_cleanctx", step_to_int("base" if suffix == "base" else suffix.replace("step", "step_"))
    if model_mode.startswith("shifted_lognormal_"):
        suffix = model_mode[len("shifted_lognormal_") :]
        return "shifted_lognormal", step_to_int("base" if suffix == "base" else suffix.replace("step", "step_"))
    raise ValueError(f"Unknown model_mode={model_mode}")


def merge_summaries() -> list[dict[str, Any]]:
    summary_rows = read_csv_rows(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {}
    fvd_path = BENCHMARK_DIR / "fvd_summary.csv"
    if fvd_path.exists():
        fvd_by_model = {row["model_mode"]: row for row in read_csv_rows(fvd_path)}

    labels = {
        "uniform_cleanctx": "Uniform timestep, clean context",
        "shifted_lognormal": "Shifted log-normal timestep, clean context",
    }
    merged: list[dict[str, Any]] = []
    for row in summary_rows:
        method_key, step = parse_model_mode(row["model_mode"])
        out: dict[str, Any] = {
            "method_key": method_key,
            "method_label": labels[method_key],
            "checkpoint_step": step,
            **row,
        }
        fvd = fvd_by_model.get(row["model_mode"], {})
        if fvd:
            out.update(
                {
                    "fvd_future": fvd["fvd_future"],
                    "fvd_backend": fvd["fvd_backend"],
                    "fvd_num_videos": fvd["fvd_num_videos"],
                    "fvd_num_frames": fvd["fvd_num_frames"],
                    "fvd_size": fvd["fvd_size"],
                }
            )
        merged.append(out)
    merged.sort(key=lambda row: (row["method_key"], int(row["checkpoint_step"])))
    write_csv(BENCHMARK_DIR / "noaction_shifted_timestep_longer_summary_with_fvd.csv", merged)
    return merged


def plot_clear_graphs(summary_csv: Path) -> None:
    cmd = [
        str(PYTHON),
        "scripts/plot_action_checkpoint_metrics_clear.py",
        "--input",
        str(summary_csv),
        "--schema",
        "frame",
        "--output-dir",
        str(BENCHMARK_DIR / "metric_plots"),
        "--title",
        "No-Action Clean-Context: Uniform vs Longer Shifted Log-Normal",
    ]
    run_command(cmd, LOG_DIR / "plot_clear_graphs.log")


def write_report(rows: list[dict[str, Any]]) -> Path:
    def as_float(row: dict[str, Any], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else float("nan")

    by_mode = {row["model_mode"]: row for row in rows}
    comparisons: dict[str, Any] = {}
    for step in [500, 1000, 2000, 3000, 4000, 6000]:
        uniform = by_mode.get(f"uniform_cleanctx_step{step:06d}")
        shifted = by_mode.get(f"shifted_lognormal_step{step:06d}")
        if not uniform or not shifted:
            continue
        comparisons[f"step_{step:06d}"] = {
            "future_psnr_shifted_minus_uniform": as_float(shifted, "mean_future_psnr")
            - as_float(uniform, "mean_future_psnr"),
            "future_ssim_shifted_minus_uniform": as_float(shifted, "mean_future_global_ssim")
            - as_float(uniform, "mean_future_global_ssim"),
            "sharpness_ratio_shifted_minus_uniform": as_float(
                shifted, "mean_sharpness_ratio_generated_over_reference"
            )
            - as_float(uniform, "mean_sharpness_ratio_generated_over_reference"),
            "motion_ratio_shifted_minus_uniform": as_float(shifted, "mean_motion_ratio_generated_over_reference")
            - as_float(uniform, "mean_motion_ratio_generated_over_reference"),
            "fvd_shifted_minus_uniform": as_float(shifted, "fvd_future") - as_float(uniform, "fvd_future"),
            "uniform": uniform,
            "shifted": shifted,
        }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_csv": str(BENCHMARK_DIR / "noaction_shifted_timestep_longer_summary_with_fvd.csv"),
        "comparisons": comparisons,
    }
    path = BENCHMARK_DIR / "noaction_shifted_timestep_longer_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-fvd", action="store_true")
    args = parser.parse_args()

    if not args.skip_generation:
        generate_all(max_workers=args.max_workers, seed=args.seed)
    if not args.skip_download:
        download_all(seed=args.seed)
    manifest_path = build_manifest(seed=args.seed)
    if not args.skip_quality:
        run_quality_metrics(manifest_path)
    if not args.skip_fvd:
        run_fvd(manifest_path)
    rows = merge_summaries()
    summary_csv = BENCHMARK_DIR / "noaction_shifted_timestep_longer_summary_with_fvd.csv"
    plot_clear_graphs(summary_csv)
    report_path = write_report(rows)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary_csv": str(summary_csv),
                "report_json": str(report_path),
                "metric_plots": str(BENCHMARK_DIR / "metric_plots"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
