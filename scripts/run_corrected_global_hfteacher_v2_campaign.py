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
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "corrected_global_mlp_hfteacher_v2_seed231_all5"
LOCAL_GENERATED_ROOT = ROOT / "data" / "corrected_global_mlp_hfteacher_v2_generated"
LOG_DIR = ROOT / "data" / "modal_logs" / "corrected_global_mlp_hfteacher_v2_seed231_all5"

SHIFTED_BASELINE_VOLUME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts"
SHIFTED_BASELINE_RUN = "ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000"
SHIFTED_BASELINE_STEP = "step_003000"

GLOBAL_RUN_3000 = (
    "ltx2b_dist098_waymo24_frame_global_mlp_action_lora_r16_seed231_"
    "from_shifted_noaction_step003000_lr5e6_actionlr1e4_steps3000"
)
GLOBAL_RUN_6000 = (
    "ltx2b_dist098_waymo24_frame_global_mlp_action_lora_r16_seed231_"
    "from_shifted_noaction_step003000_lr5e6_actionlr1e4_steps6000_resume3000"
)
GLOBAL_CKPT_VOLUME = "ltx2b-dist098-waymo24-frameglobal-shifted-action-r16-ckpts"
GLOBAL_ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-frameglobal-shifted-action-infer"
GLOBAL_RUNS_ROOT = "distilled098_frameglobal_shifted_action_lora_24fps_minterpolate_seed231_runs"

HFTEACHER_V2_RUN_1000 = (
    "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_"
    "from_shifted_noaction_step003000_steps1000"
)
HFTEACHER_V2_RUN_3000 = (
    "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_"
    "from_shifted_noaction_step003000_steps3000_resume1000"
)
HFTEACHER_V2_CKPT_VOLUME = "ltx2b-dist098-waymo24-framebneck-hft-v2-r16-ckpts"
HFTEACHER_V2_ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-framebneck-hft-v2-infer"
HFTEACHER_V2_RUNS_ROOT = "distilled098_framebottleneck_hfteacher_v2_action_lora_24fps_minterpolate_seed231_runs"

FIRST_CHECKPOINTS = [
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
HFTEACHER_OLD_CHECKPOINTS = [
    "step_000000_base_reference",
    "step_000050",
    "step_000100",
    "step_000250",
    "step_000500",
    "step_001000",
]
HFTEACHER_RESUMED_CHECKPOINTS = [
    "step_001500",
    "step_002000",
    "step_002500",
    "step_003000",
]
GLOBAL_LONG_CHECKPOINTS = ["step_004000", "step_005000", "step_006000"]
COUNTERFACTUAL_STEPS = ["step_000000_base_reference", "step_000250", "step_001000", "step_003000"]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    train_wrapper: str
    infer_wrapper: str
    checkpoint_volume: str
    artifact_volume: str
    runs_root: str
    run_name: str
    checkpoints: list[str]


GLOBAL_SPEC_3000 = ModelSpec(
    key="corrected_global_mlp",
    label="Corrected Global MLP action tokens",
    train_wrapper="train_ltx2b_distilled_waymo_frame_global_mlp_action_lora.py",
    infer_wrapper="generate_waymo24_distilled_frame_global_mlp_action_minterpolate_lora.py",
    checkpoint_volume=GLOBAL_CKPT_VOLUME,
    artifact_volume=GLOBAL_ARTIFACT_VOLUME,
    runs_root=GLOBAL_RUNS_ROOT,
    run_name=GLOBAL_RUN_3000,
    checkpoints=FIRST_CHECKPOINTS,
)
GLOBAL_SPEC_6000 = ModelSpec(
    key="corrected_global_mlp",
    label="Corrected Global MLP action tokens",
    train_wrapper="train_ltx2b_distilled_waymo_frame_global_mlp_action_lora.py",
    infer_wrapper="generate_waymo24_distilled_frame_global_mlp_action_minterpolate_lora.py",
    checkpoint_volume=GLOBAL_CKPT_VOLUME,
    artifact_volume=GLOBAL_ARTIFACT_VOLUME,
    runs_root=GLOBAL_RUNS_ROOT,
    run_name=GLOBAL_RUN_6000,
    checkpoints=GLOBAL_LONG_CHECKPOINTS,
)
HFTEACHER_V2_SPEC_1000 = ModelSpec(
    key="hfteacher_v2",
    label="Temporal bottleneck HF-teacher v2",
    train_wrapper="train_ltx2b_distilled_waymo_frame_temporal_bottleneck_hf_teacher_action_lora_v2.py",
    infer_wrapper="generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora_v2.py",
    checkpoint_volume=HFTEACHER_V2_CKPT_VOLUME,
    artifact_volume=HFTEACHER_V2_ARTIFACT_VOLUME,
    runs_root=HFTEACHER_V2_RUNS_ROOT,
    run_name=HFTEACHER_V2_RUN_1000,
    checkpoints=HFTEACHER_OLD_CHECKPOINTS,
)
HFTEACHER_V2_SPEC_3000 = ModelSpec(
    key="hfteacher_v2",
    label="Temporal bottleneck HF-teacher v2",
    train_wrapper="train_ltx2b_distilled_waymo_frame_temporal_bottleneck_hf_teacher_action_lora_v2.py",
    infer_wrapper="generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora_v2.py",
    checkpoint_volume=HFTEACHER_V2_CKPT_VOLUME,
    artifact_volume=HFTEACHER_V2_ARTIFACT_VOLUME,
    runs_root=HFTEACHER_V2_RUNS_ROOT,
    run_name=HFTEACHER_V2_RUN_3000,
    checkpoints=HFTEACHER_RESUMED_CHECKPOINTS,
)


def step_to_int(step_name: str) -> int:
    if step_name == "step_000000_base_reference":
        return 0
    match = re.search(r"step_0*([0-9]+)", step_name)
    if not match:
        raise ValueError(f"Could not parse step from {step_name}")
    return int(match.group(1))


def step_label(step_name: str) -> str:
    if step_name == "step_000000_base_reference":
        return "step000000"
    return step_name.replace("step_", "step")


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None, retries: int = 1) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_code = 0
    for attempt in range(1, retries + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"attempt={attempt} started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
            last_code = proc.returncode
        if last_code == 0:
            return
    raise RuntimeError(f"Command failed with code {last_code}; see {log_path}")


def log_completed(path: Path) -> bool:
    return path.exists() and "App completed" in path.read_text(encoding="utf-8", errors="ignore")


def modal_env(spec: ModelSpec, *, train: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LTX_CHECKPOINT_VOLUME_NAME": spec.checkpoint_volume,
            "LTX_ARTIFACTS_VOLUME_NAME": spec.artifact_volume,
            "LTX_RUNS_ROOT": spec.runs_root,
            "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
            "LTX_MODAL_GPU": "H100",
            "LTX_MODAL_TRAIN_GPU": "H100",
        }
    )
    if train:
        env["LTX_BASELINE_CHECKPOINT_VOLUME_NAME"] = SHIFTED_BASELINE_VOLUME
    return env


def train_global_3000(force: bool = False) -> None:
    log_path = LOG_DIR / "training" / f"{GLOBAL_RUN_3000}.log"
    if not force and log_completed(log_path):
        print(json.dumps({"train": "skipped", "run_name": GLOBAL_RUN_3000}, sort_keys=True), flush=True)
        return
    cmd = [
        str(MODAL),
        "run",
        GLOBAL_SPEC_3000.train_wrapper,
        "--run-name",
        GLOBAL_RUN_3000,
        "--max-steps",
        "3000",
        "--max-train-hours",
        "4.0",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        "1e-4",
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        "0,100,250,500,1000,1500,2000,2500,3000",
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--baseline-lora-run-name",
        SHIFTED_BASELINE_RUN,
        "--baseline-lora-step",
        SHIFTED_BASELINE_STEP,
        "--num-val-samples",
        "0",
        "--train-limit",
        "0",
        "--val-limit",
        "32",
        "--seed",
        "231",
    ]
    run_command(cmd, log_path, env=modal_env(GLOBAL_SPEC_3000, train=True), retries=2)
    print(json.dumps({"trained": GLOBAL_RUN_3000}, sort_keys=True), flush=True)


def train_global_6000(force: bool = False) -> None:
    log_path = LOG_DIR / "training" / f"{GLOBAL_RUN_6000}.log"
    if not force and log_completed(log_path):
        print(json.dumps({"train": "skipped", "run_name": GLOBAL_RUN_6000}, sort_keys=True), flush=True)
        return
    cmd = [
        str(MODAL),
        "run",
        GLOBAL_SPEC_6000.train_wrapper,
        "--run-name",
        GLOBAL_RUN_6000,
        "--max-steps",
        "6000",
        "--max-train-hours",
        "4.0",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        "1e-4",
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        "4000,5000,6000",
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--baseline-lora-run-name",
        SHIFTED_BASELINE_RUN,
        "--baseline-lora-step",
        SHIFTED_BASELINE_STEP,
        "--resume-from-run-name",
        GLOBAL_RUN_3000,
        "--resume-from-checkpoint",
        "step_003000",
        "--num-val-samples",
        "0",
        "--train-limit",
        "0",
        "--val-limit",
        "32",
        "--seed",
        "231",
    ]
    run_command(cmd, log_path, env=modal_env(GLOBAL_SPEC_6000, train=True), retries=2)
    print(json.dumps({"trained": GLOBAL_RUN_6000}, sort_keys=True), flush=True)


def train_hfteacher_v2_3000(force: bool = False) -> None:
    log_path = LOG_DIR / "training" / f"{HFTEACHER_V2_RUN_3000}.log"
    if not force and log_completed(log_path):
        print(json.dumps({"train": "skipped", "run_name": HFTEACHER_V2_RUN_3000}, sort_keys=True), flush=True)
        return
    cmd = [
        str(MODAL),
        "run",
        HFTEACHER_V2_SPEC_3000.train_wrapper,
        "--run-name",
        HFTEACHER_V2_RUN_3000,
        "--max-steps",
        "3000",
        "--max-train-hours",
        "4.0",
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
        "--checkpoint-steps",
        "1500,2000,2500,3000",
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--resume-from-run-name",
        HFTEACHER_V2_RUN_1000,
        "--resume-from-checkpoint",
        "step_001000",
        "--num-val-samples",
        "0",
        "--train-limit",
        "0",
        "--val-limit",
        "32",
        "--seed",
        "231",
    ]
    run_command(cmd, log_path, env=modal_env(HFTEACHER_V2_SPEC_3000, train=True), retries=2)
    print(json.dumps({"trained": HFTEACHER_V2_RUN_3000}, sort_keys=True), flush=True)


def run_label(spec: ModelSpec, step: str, *, mode: str = "correct") -> str:
    suffix = "" if mode == "correct" else f"_{mode}"
    return f"{spec.key}_{step_label(step)}{suffix}_g1p000_seed231_all5"


def generate_one(spec: ModelSpec, step: str, *, mode: str = "correct", force: bool = False) -> str:
    label = run_label(spec, step, mode=mode)
    log_path = LOG_DIR / "generation" / f"{label}.log"
    if not force and log_completed(log_path):
        return label
    cmd = [
        str(MODAL),
        "run",
        spec.infer_wrapper,
        "--limit",
        "5",
        "--seed",
        "231",
        "--lora-step",
        step,
        "--lora-run-name",
        spec.run_name,
        "--run-label",
        label,
        "--base-label",
        "base_distilled_no_lora",
        "--action-gate-scale",
        "1.0",
        "--action-vector-scale",
        "1.0",
        "--counterfactual-action-mode",
        mode,
        "--counterfactual-rotation",
        "1",
    ]
    run_command(cmd, log_path, env=modal_env(spec), retries=2)
    return label


def generate_all(specs: list[ModelSpec], *, counterfactual: bool, max_workers: int, force: bool = False) -> list[tuple[ModelSpec, str, str]]:
    jobs: list[tuple[ModelSpec, str, str]] = []
    for spec in specs:
        if counterfactual:
            for step in COUNTERFACTUAL_STEPS:
                if step in spec.checkpoints:
                    for mode in COUNTERFACTUAL_MODES:
                        jobs.append((spec, step, mode))
        else:
            for step in spec.checkpoints:
                jobs.append((spec, step, "correct"))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, spec, step, mode=mode, force=force) for spec, step, mode in jobs]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True), flush=True)
    return jobs


def download_one(spec: ModelSpec, step: str, mode: str) -> str:
    label = run_label(spec, step, mode=mode)
    local_dest = LOCAL_GENERATED_ROOT / spec.runs_root / label
    local_dest.mkdir(parents=True, exist_ok=True)
    remote_path = f"{spec.runs_root}/{label}"
    summary_path = local_dest / "run_summary.json"
    log_path = LOG_DIR / "download" / f"{label}.log"
    if not summary_path.exists():
        run_command(
            [str(MODAL), "volume", "get", "--force", spec.artifact_volume, f"{remote_path}/run_summary.json", str(summary_path)],
            log_path,
            retries=3,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prefix = f"{spec.runs_root}/{label}/"
    for record in summary["results"]:
        relpath = record["generated_video_relpath"]
        if not relpath.startswith(prefix):
            raise ValueError(f"Unexpected generated relpath {relpath}; expected prefix {prefix}")
        local_file = local_dest / relpath[len(prefix) :]
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if local_file.exists():
            continue
        run_command(
            [str(MODAL), "volume", "get", "--force", spec.artifact_volume, relpath, str(local_file)],
            log_path,
            retries=3,
        )
    return label


def download_all(jobs: list[tuple[ModelSpec, str, str]], *, max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, spec, step, mode) for spec, step, mode in jobs]
        for future in as_completed(futures):
            print(json.dumps({"downloaded": future.result()}, sort_keys=True), flush=True)


def local_file_for(spec: ModelSpec, step: str, mode: str, record: dict[str, Any]) -> Path:
    label = run_label(spec, step, mode=mode)
    prefix = f"{spec.runs_root}/{label}/"
    relpath = record["generated_video_relpath"]
    if not relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated relpath {relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / spec.runs_root / label / relpath[len(prefix) :]


def build_manifest(jobs: list[tuple[ModelSpec, str, str]], path: Path, *, description: str) -> Path:
    records: list[dict[str, Any]] = []
    diagnostic_group = "counterfactual_suite" if any(mode != "correct" for _, _, mode in jobs) else "checkpoint_sweep"
    for spec, step, mode in jobs:
        label = run_label(spec, step, mode=mode)
        summary_path = LOCAL_GENERATED_ROOT / spec.runs_root / label / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary["results"]:
            local_file = local_file_for(spec, step, mode, record)
            if not local_file.exists():
                raise FileNotFoundError(local_file)
            row = dict(record)
            row.update(
                {
                    "local_file": str(local_file),
                    "diagnostic_group": diagnostic_group,
                    "method_key": spec.key,
                    "method_label": spec.label,
                    "checkpoint_step": step_to_int(step),
                    "checkpoint_name": step,
                    "counterfactual_action_mode": mode,
                    "action_gate_scale": 1.0,
                    "action_vector_scale": 1.0,
                    "model_mode": f"{spec.key}_{step_label(step)}" + ("" if mode == "correct" else f"_{mode}"),
                    "using_lora": step != "step_000000_base_reference",
                    "lora_run_name": spec.run_name,
                    "checkpoint_volume": spec.checkpoint_volume,
                    "artifact_volume": spec.artifact_volume,
                }
            )
            records.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "seed": 231,
                "fps": 24,
                "context_frames": 49,
                "future_frames": 72,
                "total_frames": 121,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_quality_and_fvd(manifest: Path, *, output_dir: Path, run_id: str) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_video_quality_modal.py",
            "--manifest",
            str(manifest),
            "--source-dir",
            str(SOURCE_DIR),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id + "_quality",
            "--chunk-size",
            "8",
        ],
        LOG_DIR / "metrics" / f"{run_id}_quality.log",
        retries=2,
    )
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_action_fvd_modal.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id + "_fvd",
        ],
        LOG_DIR / "metrics" / f"{run_id}_fvd.log",
        retries=2,
    )
    merge_quality_fvd(output_dir)


def run_counterfactual_metrics(manifest: Path, *, output_dir: Path, run_id: str) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_counterfactual_sensitivity_modal.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id + "_counterfactual",
            "--chunk-size",
            "8",
        ],
        LOG_DIR / "metrics" / f"{run_id}_counterfactual.log",
        retries=2,
    )


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


def merge_quality_fvd(output_dir: Path) -> None:
    rows = read_csv(output_dir / "model_summary.csv")
    fvd = {row["model_mode"]: row for row in read_csv(output_dir / "fvd_summary.csv")}
    merged = []
    for row in rows:
        out = dict(row)
        fvd_row = fvd.get(row["model_mode"], {})
        out["future_fvd_style"] = fvd_row.get("fvd_future", "")
        out["fvd_backend"] = fvd_row.get("fvd_backend", "")
        merged.append(out)
    write_csv(output_dir / "model_summary_with_fvd.csv", merged)


def metric(row: dict[str, str], name: str) -> float:
    aliases = {
        "psnr": "mean_future_psnr",
        "ssim": "mean_future_global_ssim",
        "sharp": "mean_sharpness_ratio_generated_over_reference",
        "fft": "mean_fft_high_frequency_energy_ratio_generated_over_reference",
        "motion": "mean_motion_ratio_generated_over_reference",
        "lowmotion": "mean_low_frequency_motion_ratio_generated_over_reference",
        "fvd": "future_fvd_style",
    }
    try:
        return float(row.get(aliases.get(name, name), "nan"))
    except ValueError:
        return float("nan")


def should_extend_global(output_dir: Path) -> dict[str, Any]:
    rows = read_csv(output_dir / "model_summary_with_fvd.csv")
    global_rows = [row for row in rows if row.get("model_mode") == "corrected_global_mlp_step003000"]
    if not global_rows:
        return {"extend": False, "reason": "missing step_003000 metrics"}
    row = global_rows[0]
    sharp = metric(row, "sharp")
    motion = metric(row, "motion")
    fvd = metric(row, "fvd")
    psnr = metric(row, "psnr")
    ssim = metric(row, "ssim")
    extend = sharp >= 0.20 and motion >= 0.62 and fvd <= 90.0
    return {
        "extend": extend,
        "reason": "passes stability thresholds" if extend else "fails stability thresholds",
        "thresholds": {"sharpness_min": 0.20, "motion_min": 0.62, "fvd_max": 90.0},
        "step_3000": {"psnr": psnr, "ssim": ssim, "sharpness": sharp, "motion": motion, "fvd": fvd},
    }


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        "train_ltx2b_distilled_waymo_frame_global_mlp_action_lora.py",
        "train_ltx2b_distilled_waymo_frame_temporal_bottleneck_hf_teacher_action_lora_v2.py",
        "generate_waymo24_distilled_frame_global_mlp_action_minterpolate_lora.py",
        "generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora_v2.py",
        "scripts/compute_video_quality_modal.py",
        "scripts/compute_action_fvd_modal.py",
        "scripts/compute_counterfactual_sensitivity_modal.py",
        "scripts/run_corrected_global_hfteacher_v2_campaign.py",
    ]
    run_command([str(PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log")


def run_eval(specs: list[ModelSpec], *, output_dir: Path, run_id: str, max_workers: int, counterfactual: bool = True) -> None:
    jobs = generate_all(specs, counterfactual=False, max_workers=max_workers)
    download_all(jobs, max_workers=max_workers)
    manifest = build_manifest(jobs, output_dir / "manifest_checkpoint_sweep.json", description=run_id + " checkpoint sweep")
    run_quality_and_fvd(manifest, output_dir=output_dir, run_id=run_id)
    if counterfactual:
        cf_jobs = generate_all(specs, counterfactual=True, max_workers=max_workers)
        download_all(cf_jobs, max_workers=max_workers)
        cf_manifest = build_manifest(cf_jobs, output_dir / "manifest_counterfactual.json", description=run_id + " counterfactual sensitivity")
        run_counterfactual_metrics(cf_manifest, output_dir=output_dir, run_id=run_id)


def write_campaign_summary(*, global_extended: dict[str, Any]) -> None:
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corrected_recipe": {
            "seed": 231,
            "fps": 24,
            "context_frames": 49,
            "future_frames": 72,
            "total_frames": 121,
            "timestep_sampling": "shifted_lognormal",
            "image_cond_noise_scale": 0.0,
            "baseline_volume": SHIFTED_BASELINE_VOLUME,
            "baseline_run": SHIFTED_BASELINE_RUN,
            "baseline_step": SHIFTED_BASELINE_STEP,
            "recached_latents": False,
            "uses_original_10fps_actions": False,
        },
        "trained_or_checked": {
            "global_mlp_corrected": GLOBAL_RUN_3000,
            "global_mlp_6000": GLOBAL_RUN_6000 if global_extended.get("extend") else "",
            "hfteacher_v2_3000": HFTEACHER_V2_RUN_3000,
            "v3_lowfreq": "already completed: ltx2b_dist098_waymo24_frame_temporal_bottleneck_lowfreq_v3b_gate1e4_proj1e3_seed231_from_shifted_noaction_step003000_steps3000",
        },
        "global_extension_decision": global_extended,
        "benchmark_dir": str(BENCHMARK_DIR),
        "generated_root": str(LOCAL_GENERATED_ROOT),
    }
    (BENCHMARK_DIR / "campaign_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "train", "eval", "global6000"], default="all")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    syntax_check()

    if args.phase in {"all", "train"}:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(train_global_3000, args.force_train),
                executor.submit(train_hfteacher_v2_3000, args.force_train),
            ]
            for future in as_completed(futures):
                future.result()

    if args.phase in {"all", "eval"}:
        run_eval(
            [GLOBAL_SPEC_3000, HFTEACHER_V2_SPEC_1000, HFTEACHER_V2_SPEC_3000],
            output_dir=BENCHMARK_DIR,
            run_id="corrected_global_hfteacher_v2_seed231_all5",
            max_workers=args.max_workers,
        )
        global_extended = should_extend_global(BENCHMARK_DIR)
        if global_extended.get("extend"):
            train_global_6000(args.force_train)
            long_dir = BENCHMARK_DIR / "global_mlp_6000"
            run_eval(
                [GLOBAL_SPEC_6000],
                output_dir=long_dir,
                run_id="corrected_global_mlp_6000_seed231_all5",
                max_workers=args.max_workers,
                counterfactual=True,
            )
        write_campaign_summary(global_extended=global_extended)

    if args.phase == "global6000":
        train_global_6000(args.force_train)
        run_eval(
            [GLOBAL_SPEC_6000],
            output_dir=BENCHMARK_DIR / "global_mlp_6000",
            run_id="corrected_global_mlp_6000_seed231_all5",
            max_workers=args.max_workers,
            counterfactual=True,
        )


if __name__ == "__main__":
    main()
