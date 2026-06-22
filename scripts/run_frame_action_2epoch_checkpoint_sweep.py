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
LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_action_2epoch_checkpoint_sweep_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_action_2epoch_checkpoint_sweep_seed231_all5"

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
    "step_004000",
    "step_005000",
    "step_006000",
    "step_007000",
    "step_007992",
    "step_010000",
    "step_012000",
    "step_014000",
    "step_015984",
]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    wrapper: str
    checkpoint_volume: str
    artifact_volume: str
    runs_root: str
    lora_run_name: str


METHODS = [
    Method(
        key="frame_transformer",
        label="Frame Transformer",
        wrapper="scripts/wrappers/generate_waymo24_distilled_frame_transformer_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-framexf-action-lora-r16-2epoch-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-framexf-action-lora-2epoch-infer",
        runs_root="distilled098_framexf_action_lora_24fps_minterpolate_seed231_2epoch_runs",
        lora_run_name=(
            "ltx2b_dist098_waymo24_frame_transformer_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_2epochs_steps15984"
        ),
    ),
    Method(
        key="frame_temporal_pool",
        label="Frame Temporal Pool",
        wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_pool_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-framepool-action-lora-r16-2epoch-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-framepool-action-lora-2epoch-infer",
        runs_root="distilled098_framepool_action_lora_24fps_minterpolate_seed231_2epoch_runs",
        lora_run_name=(
            "ltx2b_dist098_waymo24_frame_temporal_pool_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_2epochs_steps15984"
        ),
    ),
    Method(
        key="frame_global_mlp",
        label="Frame Global MLP",
        wrapper="scripts/wrappers/generate_waymo24_distilled_frame_global_mlp_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-frameglobal-action-lora-r16-2epoch-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-frameglobal-action-lora-2epoch-infer",
        runs_root="distilled098_frameglobal_action_lora_24fps_minterpolate_seed231_2epoch_runs",
        lora_run_name=(
            "ltx2b_dist098_waymo24_frame_global_mlp_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_2epochs_steps15984"
        ),
    ),
    Method(
        key="frame_adaln",
        label="Frame AdaLN",
        wrapper="scripts/wrappers/generate_waymo24_distilled_frame_adaln_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-frameadaln-action-lora-r16-2epoch-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-frameadaln-action-lora-2epoch-infer",
        runs_root="distilled098_frameadaln_action_lora_24fps_minterpolate_seed231_2epoch_runs",
        lora_run_name=(
            "ltx2b_dist098_waymo24_frame_adaln_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr5e5_2epochs_steps15984"
        ),
    ),
]


def step_to_int(step: str) -> int:
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse step from {step}")
    return int(match.group(1))


def step_label(step: str) -> str:
    if step == "step_000000_base_reference":
        return "step000000"
    return step.replace("step_", "step")


def run_label(method: Method, step: str, seed: int) -> str:
    return f"{method.key}_{step_label(step)}_seed{seed}_all5_2epoch"


def modal_env(method: Method) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LTX_CHECKPOINT_VOLUME_NAME": method.checkpoint_volume,
            "LTX_ARTIFACTS_VOLUME_NAME": method.artifact_volume,
            "LTX_RUNS_ROOT": method.runs_root,
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


def generate_one(method: Method, step: str, seed: int) -> tuple[str, str]:
    label = run_label(method, step, seed)
    log_path = LOG_DIR / "generation" / f"{label}.log"
    if log_path.exists() and "✓ App completed" in log_path.read_text(encoding="utf-8", errors="ignore"):
        return method.key, step
    cmd = [
        str(MODAL),
        "run",
        method.wrapper,
        "--limit",
        "5",
        "--seed",
        str(seed),
        "--lora-step",
        step,
        "--lora-run-name",
        method.lora_run_name,
        "--run-label",
        label,
    ]
    run_command(cmd, log_path, env=modal_env(method))
    return method.key, step


def generate_all(max_workers: int, seed: int) -> None:
    jobs = [(method, step) for step in CHECKPOINTS for method in METHODS]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, method, step, seed) for method, step in jobs]
        for future in as_completed(futures):
            method_key, step = future.result()
            print(json.dumps({"generated": method_key, "step": step}, sort_keys=True))


def download_one(method: Method, step: str, seed: int) -> Path:
    label = run_label(method, step, seed)
    remote_path = f"{method.runs_root}/{label}"
    local_dest = LOCAL_GENERATED_ROOT / method.runs_root / label
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
            method.artifact_volume,
            summary_remote,
            str(summary_local),
        ],
        log_path,
    )

    summary = json.loads(summary_local.read_text(encoding="utf-8"))
    prefix = f"{method.runs_root}/{label}/"
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
                method.artifact_volume,
                generated_relpath,
                str(video_local),
            ],
            log_path,
        )
    return local_dest


def download_all(seed: int) -> None:
    for method in METHODS:
        for step in CHECKPOINTS:
            path = download_one(method, step, seed)
            print(json.dumps({"downloaded": str(path.relative_to(ROOT))}, sort_keys=True))


def local_file_for_record(record: dict[str, Any], method: Method, step: str, seed: int) -> Path:
    label = run_label(method, step, seed)
    prefix = f"{method.runs_root}/{label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / method.runs_root / label / generated_relpath[len(prefix) :]


def build_manifest(seed: int) -> Path:
    records: list[dict[str, Any]] = []
    for method in METHODS:
        for step in CHECKPOINTS:
            label = run_label(method, step, seed)
            summary_path = LOCAL_GENERATED_ROOT / method.runs_root / label / "run_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for record in summary["results"]:
                local_file = local_file_for_record(record, method, step, seed)
                if not local_file.exists():
                    raise FileNotFoundError(local_file)
                row = dict(record)
                row["local_file"] = str(local_file)
                row["method_key"] = method.key
                row["method_label"] = method.label
                row["checkpoint_step"] = step_to_int(step)
                row["checkpoint_name"] = step
                row["model_mode"] = f"{method.key}_{step_label(step)}"
                row["using_lora"] = True
                records.append(row)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "2-epoch frame-aligned action conditioning checkpoint sweep over 5 local Waymo validation clips.",
        "seed": seed,
        "context_frames": 49,
        "future_frames": 72,
        "total_frames": 121,
        "fps": 24,
        "methods": [method.__dict__ for method in METHODS],
        "checkpoints": CHECKPOINTS,
        "records": records,
    }
    manifest_path = BENCHMARK_DIR / "manifest_frame_action_2epoch_checkpoint_sweep_seed231_all5.json"
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
        "frame_action_2epoch_checkpoint_sweep_seed231_all5",
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
    for method in METHODS:
        prefix = method.key + "_"
        if model_mode.startswith(prefix):
            return method.key, step_to_int(model_mode[len(prefix) :].replace("step", "step_"))
    raise ValueError(f"Unknown model_mode={model_mode}")


def merge_summaries() -> list[dict[str, Any]]:
    summary_rows = read_csv_rows(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {}
    fvd_path = BENCHMARK_DIR / "fvd_summary.csv"
    if fvd_path.exists():
        fvd_by_model = {row["model_mode"]: row for row in read_csv_rows(fvd_path)}

    merged: list[dict[str, Any]] = []
    label_by_key = {method.key: method.label for method in METHODS}
    for row in summary_rows:
        method_key, step = parse_model_mode(row["model_mode"])
        out: dict[str, Any] = {
            "method_key": method_key,
            "method_label": label_by_key[method_key],
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
    write_csv(BENCHMARK_DIR / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv", merged)
    return merged


def write_best_report(rows: list[dict[str, Any]]) -> Path:
    def as_float(row: dict[str, Any], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else float("nan")

    def best(metric: str, *, higher: bool, target_one: bool = False) -> dict[str, Any]:
        valid = [row for row in rows if str(row.get(metric, "")) not in {"", "nan"}]
        if target_one:
            selected = min(valid, key=lambda row: abs(as_float(row, metric) - 1.0))
        else:
            selected = max(valid, key=lambda row: as_float(row, metric)) if higher else min(valid, key=lambda row: as_float(row, metric))
        return {
            "metric": metric,
            "higher_is_better": higher,
            "target_one": target_one,
            "method_key": selected["method_key"],
            "method_label": selected["method_label"],
            "checkpoint_step": int(selected["checkpoint_step"]),
            "model_mode": selected["model_mode"],
            "value": as_float(selected, metric),
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_future_psnr": best("mean_future_psnr", higher=True),
        "best_future_ssim": best("mean_future_global_ssim", higher=True),
        "best_fvd": best("fvd_future", higher=False),
        "best_sharpness_ratio_match": best("mean_sharpness_ratio_generated_over_reference", higher=True, target_one=True),
        "best_motion_ratio_match": best("mean_motion_ratio_generated_over_reference", higher=True, target_one=True),
    }
    path = BENCHMARK_DIR / "frame_action_2epoch_checkpoint_sweep_best.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
        "Upsampled Frame-Action Conditioning, 2 Epochs, Seed 231, 5 Clips",
    ]
    run_command(cmd, LOG_DIR / "plot_clear_graphs.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--max-workers", type=int, default=4)
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
    best_path = write_best_report(rows)
    summary_csv = BENCHMARK_DIR / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv"
    plot_clear_graphs(summary_csv)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary_csv": str(summary_csv),
                "best_json": str(best_path),
                "metric_plots": str(BENCHMARK_DIR / "metric_plots"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
