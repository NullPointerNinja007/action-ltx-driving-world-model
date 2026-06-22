from __future__ import annotations

import argparse
import csv
import json
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
LOCAL_GENERATED_ROOT = ROOT / "data" / "midblock_gated_xattn_checkpoint_sweep_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"

RUN_NAME = "ltx2b_dist098_waymo24_frame_midblock_gated_xattn_seed231_from_shifted_noaction_step003000_steps3000"
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


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    wrapper: str
    checkpoint_volume: str
    artifact_volume: str
    runs_root: str
    lora_run_name: str


METHOD = Method(
    key="frame_midblock_gated_xattn",
    label="Frame Mid-Block Gated XAttn",
    wrapper="generate_waymo24_distilled_frame_midblock_gated_xattn_action_minterpolate_lora.py",
    checkpoint_volume="ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts",
    artifact_volume="ltx2b-dist098-waymo24-midxattn-r16-shift-infer",
    runs_root="distilled098_framemidxattn_action_lora_24fps_minterpolate_seed231_shifted_runs",
    lora_run_name=RUN_NAME,
)


def step_to_int(step: str) -> int:
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse step from {step}")
    return int(match.group(1))


def step_label(step: str) -> str:
    if step == "step_000000_base_reference":
        return "step000000"
    return step.replace("step_", "step")


def run_label(step: str, seed: int) -> str:
    return f"{METHOD.key}_{step_label(step)}_seed{seed}_all5_shifted_cleanctx"


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def generate_one(step: str, seed: int) -> str:
    label = run_label(step, seed)
    log_path = LOG_DIR / "generation" / f"{label}.log"
    if log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore"):
        return step
    cmd = [
        str(MODAL),
        "run",
        METHOD.wrapper,
        "--limit",
        "5",
        "--seed",
        str(seed),
        "--lora-step",
        step,
        "--lora-run-name",
        METHOD.lora_run_name,
        "--run-label",
        label,
        "--base-label",
        "base_distilled_no_lora",
    ]
    run_command(cmd, log_path)
    return step


def generate_all(max_workers: int, seed: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, step, seed) for step in CHECKPOINTS]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True))


def download_one(step: str, seed: int) -> Path:
    label = run_label(step, seed)
    remote_path = f"{METHOD.runs_root}/{label}"
    local_dest = LOCAL_GENERATED_ROOT / METHOD.runs_root / label
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
            METHOD.artifact_volume,
            summary_remote,
            str(summary_local),
        ],
        log_path,
    )

    summary = json.loads(summary_local.read_text(encoding="utf-8"))
    prefix = f"{METHOD.runs_root}/{label}/"
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
                METHOD.artifact_volume,
                generated_relpath,
                str(video_local),
            ],
            log_path,
        )
    return local_dest


def download_all(seed: int) -> None:
    for step in CHECKPOINTS:
        path = download_one(step, seed)
        print(json.dumps({"downloaded": str(path.relative_to(ROOT))}, sort_keys=True))


def local_file_for_record(record: dict[str, Any], step: str, seed: int) -> Path:
    label = run_label(step, seed)
    prefix = f"{METHOD.runs_root}/{label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / METHOD.runs_root / label / generated_relpath[len(prefix) :]


def build_manifest(seed: int) -> Path:
    records: list[dict[str, Any]] = []
    for step in CHECKPOINTS:
        label = run_label(step, seed)
        summary_path = LOCAL_GENERATED_ROOT / METHOD.runs_root / label / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary["results"]:
            local_file = local_file_for_record(record, step, seed)
            if not local_file.exists():
                raise FileNotFoundError(local_file)
            row = dict(record)
            row["local_file"] = str(local_file)
            row["method_key"] = METHOD.key
            row["method_label"] = METHOD.label
            row["checkpoint_step"] = step_to_int(step)
            row["checkpoint_name"] = step
            row["model_mode"] = f"{METHOD.key}_{step_label(step)}"
            row["using_lora"] = True
            records.append(row)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Middle-block gated action cross-attention checkpoint sweep over 5 Waymo validation clips.",
        "seed": seed,
        "context_frames": 49,
        "future_frames": 72,
        "total_frames": 121,
        "fps": 24,
        "method": METHOD.__dict__,
        "checkpoints": CHECKPOINTS,
        "records": records,
    }
    manifest_path = BENCHMARK_DIR / "manifest_midblock_gated_xattn_checkpoint_sweep_seed231_all5.json"
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
            "midblock_gated_xattn_checkpoint_sweep_seed231_all5",
        ],
        LOG_DIR / "compute_fvd.log",
    )


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


def merge_summaries() -> list[dict[str, Any]]:
    summary_rows = read_csv_rows(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {}
    fvd_path = BENCHMARK_DIR / "fvd_summary.csv"
    if fvd_path.exists():
        fvd_by_model = {row["model_mode"]: row for row in read_csv_rows(fvd_path)}

    merged: list[dict[str, Any]] = []
    for row in summary_rows:
        step = step_to_int(str(row["model_mode"]).removeprefix(f"{METHOD.key}_").replace("step", "step_"))
        out: dict[str, Any] = {
            "method_key": METHOD.key,
            "method_label": METHOD.label,
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
    merged.sort(key=lambda row: int(row["checkpoint_step"]))
    write_csv(BENCHMARK_DIR / "midblock_gated_xattn_summary_with_fvd.csv", merged)
    return merged


def best_report(rows: list[dict[str, Any]]) -> None:
    def value(row: dict[str, Any], key: str) -> float:
        raw = row.get(key, "")
        return float(raw) if raw not in {"", None} else float("nan")

    def best(metric: str, *, higher: bool) -> dict[str, Any]:
        valid = [row for row in rows if str(row.get(metric, "")) not in {"", "nan"}]
        selected = max(valid, key=lambda row: value(row, metric)) if higher else min(valid, key=lambda row: value(row, metric))
        return {
            "metric": metric,
            "higher_is_better": higher,
            "checkpoint_step": int(selected["checkpoint_step"]),
            "model_mode": selected["model_mode"],
            "value": value(selected, metric),
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_future_psnr": best("mean_future_psnr", higher=True),
        "best_future_ssim": best("mean_future_global_ssim", higher=True),
        "best_fvd": best("fvd_future", higher=False) if any(row.get("fvd_future") for row in rows) else None,
        "best_sharpness_ratio": best("mean_sharpness_ratio_generated_over_reference", higher=True),
        "best_fft_high_frequency_ratio": best("mean_fft_high_frequency_energy_ratio_generated_over_reference", higher=True),
        "best_motion_ratio": best("mean_motion_ratio_generated_over_reference", higher=True),
        "best_low_frequency_temporal_delta_error": best("mean_low_frequency_temporal_delta_error_mae", higher=False),
        "note": "Sharpness/motion ratios are internal diagnostics; use visual inspection before selecting final checkpoint.",
    }
    (BENCHMARK_DIR / "midblock_gated_xattn_best_checkpoints.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and benchmark mid-block gated xattn checkpoints.")
    parser.add_argument("--phase", choices=("generate", "download", "manifest", "metrics", "fvd", "all"), default="all")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args()

    manifest_path = BENCHMARK_DIR / "manifest_midblock_gated_xattn_checkpoint_sweep_seed231_all5.json"
    if args.phase in {"generate", "all"}:
        generate_all(args.max_workers, args.seed)
    if args.phase in {"download", "all"}:
        download_all(args.seed)
    if args.phase in {"manifest", "metrics", "fvd", "all"}:
        manifest_path = build_manifest(args.seed)
        print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT))}, sort_keys=True))
    if args.phase in {"metrics", "all"}:
        run_quality_metrics(manifest_path)
    if args.phase in {"fvd", "all"}:
        run_fvd(manifest_path)
    if args.phase in {"metrics", "fvd", "all"}:
        rows = merge_summaries()
        best_report(rows)
        print(json.dumps({"summary_rows": len(rows), "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
