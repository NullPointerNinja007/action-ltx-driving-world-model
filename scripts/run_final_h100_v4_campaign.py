from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
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

SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "final_h100_v4_campaign_seed231_all5"
GENERATED_ROOT = ROOT / "data" / "final_h100_v4_campaign_generated"
LOG_DIR = ROOT / "data" / "modal_logs" / "final_h100_v4_campaign_seed231_all5"

SEED = 231
FPS = 24
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121
FIRST_EPOCH_TARGET = 7992
SECOND_EPOCH_TARGET = 15984
FIRST_EPOCH_STEPS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 7992]
COUNTERFACTUAL_STEPS = [0, 250, 1000, 3000, 5000, 7992]
SECOND_EPOCH_STEPS = [10000, 12000, 14000, 15984]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    run_name: str
    disable_text: bool
    diffusion: float
    lowfreq_target: float
    lowfreq_delta: float
    hf_teacher: float
    action_motion_aux: float
    residual: float
    gate: float
    action_lr: float = 3e-5
    injector_lr: float = 3e-5
    gate_lr: float = 3e-5
    resume_checkpoint: str = ""
    max_steps: int = FIRST_EPOCH_TARGET


SPECS: list[ModelSpec] = [
    ModelSpec(
        key="v4_conservative_text",
        label="V4 conservative full112 with text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_conservative_seed231_from_shifted_noaction_step003000_steps1000",
        disable_text=False,
        diffusion=0.50,
        lowfreq_target=0.50,
        lowfreq_delta=0.50,
        hf_teacher=0.10,
        action_motion_aux=0.05,
        residual=0.002,
        gate=0.002,
        resume_checkpoint="step_001000",
    ),
    ModelSpec(
        key="v4_main_text",
        label="V4 main full112 with text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_b200_seed231_from_shifted_noaction_step003000_steps3000",
        disable_text=False,
        diffusion=0.25,
        lowfreq_target=1.00,
        lowfreq_delta=1.00,
        hf_teacher=0.20,
        action_motion_aux=0.05,
        residual=0.002,
        gate=0.002,
        resume_checkpoint="step_003000",
    ),
    ModelSpec(
        key="v4_main_notext",
        label="V4 main full112 no text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_notext_h100_seed231_from_shifted_noaction_step003000_steps7992",
        disable_text=True,
        diffusion=0.25,
        lowfreq_target=1.00,
        lowfreq_delta=1.00,
        hf_teacher=0.20,
        action_motion_aux=0.05,
        residual=0.002,
        gate=0.002,
    ),
    ModelSpec(
        key="v4_conservative_notext",
        label="V4 conservative full112 no text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_conservative_notext_h100_seed231_from_shifted_noaction_step003000_steps7992",
        disable_text=True,
        diffusion=0.50,
        lowfreq_target=0.50,
        lowfreq_delta=0.50,
        hf_teacher=0.10,
        action_motion_aux=0.05,
        residual=0.002,
        gate=0.002,
    ),
    ModelSpec(
        key="v4_actionstrong_text",
        label="V4 action-strong full112 with text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_actionstrong_h100_seed231_from_shifted_noaction_step003000_steps7992",
        disable_text=False,
        diffusion=0.40,
        lowfreq_target=0.80,
        lowfreq_delta=1.20,
        hf_teacher=0.20,
        action_motion_aux=0.15,
        residual=0.002,
        gate=0.001,
        gate_lr=5e-5,
    ),
    ModelSpec(
        key="v4_qualitystrict_text",
        label="V4 quality-strict full112 with text",
        run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_qualitystrict_h100_seed231_from_shifted_noaction_step003000_steps7992",
        disable_text=False,
        diffusion=0.60,
        lowfreq_target=0.40,
        lowfreq_delta=0.50,
        hf_teacher=0.30,
        action_motion_aux=0.05,
        residual=0.005,
        gate=0.005,
        gate_lr=1e-5,
    ),
]


@dataclass(frozen=True)
class EvalConfig:
    spec: ModelSpec
    checkpoint_step: int
    diagnostic_group: str = "checkpoint_sweep"
    counterfactual_action_mode: str = "correct"

    @property
    def checkpoint_name(self) -> str:
        if self.checkpoint_step == 0:
            return "step_000000_base_reference"
        return f"step_{self.checkpoint_step:06d}"

    @property
    def model_mode(self) -> str:
        suffix = f"_{self.counterfactual_action_mode}" if self.diagnostic_group == "counterfactual_suite" else ""
        return f"{self.spec.key}_step{self.checkpoint_step:06d}{suffix}"

    @property
    def run_label(self) -> str:
        return f"{self.model_mode}_g1p000_seed{SEED}_all5"


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_completed(log_path: Path) -> bool:
    return log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore")


def run_command(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    retries: int = 2,
    retry_sleep: int = 90,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    for attempt in range(1, retries + 2):
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"attempt={attempt} started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
            log.write("$ " + " ".join(cmd) + "\n")
            env_bits = {
                key: value
                for key, value in merged_env.items()
                if key.startswith(("LTX_", "WAYMO24_", "ACTION_FVD_", "VIDEO_QUALITY_"))
            }
            if env_bits:
                log.write("$ env " + " ".join(f"{k}={env_bits[k]}" for k in sorted(env_bits)) + "\n")
            log.write("\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=merged_env)
        if proc.returncode == 0:
            return
        if attempt <= retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"Command failed after {retries + 1} attempts; see {log_path}")


def train_env(spec: ModelSpec) -> dict[str, str]:
    env = {
        "LTX_MODAL_TRAIN_GPU": "H100",
        "LTX_CHECKPOINT_VOLUME_NAME": CHECKPOINT_VOLUME,
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if spec.disable_text else "0",
    }
    return env


def infer_env(spec: ModelSpec) -> dict[str, str]:
    return {
        "LTX_MODAL_GPU": "H100",
        "LTX_CHECKPOINT_VOLUME_NAME": CHECKPOINT_VOLUME,
        "LTX_ARTIFACTS_VOLUME_NAME": ARTIFACT_VOLUME,
        "LTX_RUNS_ROOT": RUNS_ROOT,
        "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if spec.disable_text else "0",
    }


def train_command(spec: ModelSpec, *, target_steps: int, resume_checkpoint: str) -> list[str]:
    checkpoint_steps = FIRST_EPOCH_STEPS if target_steps <= FIRST_EPOCH_TARGET else SECOND_EPOCH_STEPS
    cmd = [
        str(MODAL),
        "run",
        TRAIN_WRAPPER,
        "--run-name",
        spec.run_name,
        "--max-steps",
        str(target_steps),
        "--max-train-hours",
        "3.5",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        str(spec.action_lr),
        "--action-injector-learning-rate",
        str(spec.injector_lr),
        "--action-gate-learning-rate",
        str(spec.gate_lr),
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        ",".join(str(step) for step in checkpoint_steps),
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--diffusion-loss-weight",
        str(spec.diffusion),
        "--lowfreq-target-loss-weight",
        str(spec.lowfreq_target),
        "--lowfreq-delta-loss-weight",
        str(spec.lowfreq_delta),
        "--hf-teacher-loss-weight",
        str(spec.hf_teacher),
        "--action-motion-aux-loss-weight",
        str(spec.action_motion_aux),
        "--action-residual-loss-weight",
        str(spec.residual),
        "--action-gate-loss-weight",
        str(spec.gate),
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
        "0",
        "--val-limit",
        "32",
    ]
    if resume_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
        cmd.extend(["--resume-from-run-name", spec.run_name])
    return cmd


def train_one_epoch(spec: ModelSpec, *, force: bool = False) -> str:
    log_path = LOG_DIR / "training" / f"{spec.key}_to{FIRST_EPOCH_TARGET}.log"
    if not force and command_completed(log_path):
        return spec.key
    run_command(
        train_command(spec, target_steps=FIRST_EPOCH_TARGET, resume_checkpoint=spec.resume_checkpoint),
        log_path,
        env=train_env(spec),
        retries=3,
    )
    return spec.key


def train_second_epoch(spec: ModelSpec, *, force: bool = False) -> str:
    log_path = LOG_DIR / "training_second_epoch" / f"{spec.key}_to{SECOND_EPOCH_TARGET}.log"
    if not force and command_completed(log_path):
        return spec.key
    run_command(
        train_command(spec, target_steps=SECOND_EPOCH_TARGET, resume_checkpoint=f"step_{FIRST_EPOCH_TARGET:06d}"),
        log_path,
        env=train_env(spec),
        retries=3,
    )
    return spec.key


def train_all(specs: list[ModelSpec], *, max_workers: int, second_epoch: bool = False, force: bool = False) -> None:
    fn = train_second_epoch if second_epoch else train_one_epoch
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, spec, force=force) for spec in specs]
        for future in as_completed(futures):
            print(json.dumps({"trained": future.result(), "second_epoch": second_epoch}, sort_keys=True), flush=True)


def generate_one(config: EvalConfig, *, limit: int, force: bool = False) -> str:
    log_path = LOG_DIR / "generation" / f"{config.run_label}.log"
    if not force and command_completed(log_path):
        return config.run_label
    cmd = [
        str(MODAL),
        "run",
        INFER_WRAPPER,
        "--limit",
        str(limit),
        "--seed",
        str(SEED),
        "--lora-step",
        config.checkpoint_name,
        "--lora-run-name",
        config.spec.run_name,
        "--run-label",
        config.run_label,
        "--base-label",
        "base_distilled_no_lora",
        "--action-gate-scale",
        "1.0",
        "--action-vector-scale",
        "1.0",
        "--counterfactual-action-mode",
        config.counterfactual_action_mode,
        "--counterfactual-rotation",
        "1",
    ]
    run_command(cmd, log_path, env=infer_env(config.spec), retries=3)
    return config.run_label


def generate_all(configs: list[EvalConfig], *, max_workers: int, limit: int, force: bool = False) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, config, limit=limit, force=force) for config in configs]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True), flush=True)


def download_one(config: EvalConfig) -> str:
    local_dest = GENERATED_ROOT / RUNS_ROOT / config.run_label
    local_dest.mkdir(parents=True, exist_ok=True)
    remote_path = f"{RUNS_ROOT}/{config.run_label}"
    summary_path = local_dest / "run_summary.json"
    log_path = LOG_DIR / "download" / f"{config.run_label}.log"
    if not summary_path.exists():
        run_command(
            [str(MODAL), "volume", "get", "--force", ARTIFACT_VOLUME, f"{remote_path}/run_summary.json", str(summary_path)],
            log_path,
            retries=3,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    for record in summary["results"]:
        relpath = record["generated_video_relpath"]
        if not relpath.startswith(prefix):
            raise ValueError(f"Unexpected generated relpath {relpath}; expected prefix {prefix}")
        local_file = local_dest / relpath[len(prefix) :]
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if local_file.exists():
            continue
        run_command(
            [str(MODAL), "volume", "get", "--force", ARTIFACT_VOLUME, relpath, str(local_file)],
            log_path,
            retries=3,
        )
    return config.run_label


def download_all(configs: list[EvalConfig], *, max_workers: int) -> None:
    # Modal volume downloads are independent and mostly I/O-bound. Parallelizing
    # this avoids making counterfactual sweeps spend hours in serial transfer.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, config) for config in configs]
        for future in as_completed(futures):
            print(json.dumps({"downloaded": future.result()}, sort_keys=True), flush=True)


def local_file_for_record(record: dict[str, Any], config: EvalConfig) -> Path:
    prefix = f"{RUNS_ROOT}/{config.run_label}/"
    relpath = record["generated_video_relpath"]
    if not relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated relpath {relpath}; expected prefix {prefix}")
    return GENERATED_ROOT / RUNS_ROOT / config.run_label / relpath[len(prefix) :]


def build_manifest(configs: list[EvalConfig], path: Path, *, description: str) -> Path:
    records: list[dict[str, Any]] = []
    for config in configs:
        summary_path = GENERATED_ROOT / RUNS_ROOT / config.run_label / "run_summary.json"
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
                    "method_key": config.spec.key,
                    "method_label": config.spec.label,
                    "checkpoint_step": config.checkpoint_step,
                    "checkpoint_name": config.checkpoint_name,
                    "model_mode": config.model_mode,
                    "counterfactual_action_mode": config.counterfactual_action_mode,
                    "disable_text_conditioning": config.spec.disable_text,
                    "diffusion_loss_weight": config.spec.diffusion,
                    "lowfreq_target_loss_weight": config.spec.lowfreq_target,
                    "lowfreq_delta_loss_weight": config.spec.lowfreq_delta,
                    "hf_teacher_loss_weight": config.spec.hf_teacher,
                    "action_motion_aux_loss_weight": config.spec.action_motion_aux,
                    "using_lora": True,
                }
            )
            records.append(row)
    write_json(
        path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "seed": SEED,
            "fps": FPS,
            "context_frames": CONTEXT_FRAMES,
            "future_frames": FUTURE_FRAMES,
            "total_frames": TOTAL_FRAMES,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "records": records,
        },
    )
    return path


def first_epoch_configs(specs: list[ModelSpec]) -> list[EvalConfig]:
    return [EvalConfig(spec, step) for spec in specs for step in FIRST_EPOCH_STEPS]


def second_epoch_configs(specs: list[ModelSpec]) -> list[EvalConfig]:
    return [EvalConfig(spec, step) for spec in specs for step in SECOND_EPOCH_STEPS]


def counterfactual_configs(specs: list[ModelSpec], *, steps: list[int]) -> list[EvalConfig]:
    return [
        EvalConfig(spec, step, diagnostic_group="counterfactual_suite", counterfactual_action_mode=mode)
        for spec in specs
        for step in steps
        for mode in COUNTERFACTUAL_MODES
    ]


def run_quality_metrics(manifest_path: Path, *, output_dir: Path, chunk_size: int) -> None:
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
            str(output_dir),
            "--run-id",
            output_dir.name + "_quality",
            "--chunk-size",
            str(chunk_size),
        ],
        LOG_DIR / "metrics" / f"{output_dir.name}_quality.log",
        retries=3,
    )


def run_fvd_by_spec(manifest_path: Path, specs: list[ModelSpec], *, output_dir: Path, max_workers: int) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts_dir = output_dir / "fvd_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_parts: list[tuple[ModelSpec, Path]] = []
    for spec in specs:
        records = [row for row in payload["records"] if row.get("method_key") == spec.key]
        if not records:
            continue
        part_path = parts_dir / f"manifest_{spec.key}.json"
        write_json(part_path, {**payload, "records": records})
        manifest_parts.append((spec, part_path))

    def run_part(item: tuple[ModelSpec, Path]) -> str:
        spec, part_path = item
        part_output = parts_dir / spec.key
        run_command(
            [
                str(MODAL),
                "run",
                "scripts/compute_action_fvd_modal.py",
                "--manifest",
                str(part_path),
                "--source-dir",
                str(SOURCE_DIR),
                "--output-dir",
                str(part_output),
                "--run-id",
                f"{output_dir.name}_{spec.key}_fvd",
            ],
            LOG_DIR / "metrics" / f"{output_dir.name}_fvd_{spec.key}.log",
            env={"ACTION_FVD_GPU": "H100"},
            retries=3,
        )
        return spec.key

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_part, item) for item in manifest_parts]
        for future in as_completed(futures):
            print(json.dumps({"fvd_done": future.result()}, sort_keys=True), flush=True)

    rows: list[dict[str, Any]] = []
    for spec, _ in manifest_parts:
        rows.extend(read_csv(parts_dir / spec.key / "fvd_summary.csv"))
    rows.sort(key=lambda row: (row.get("model_mode", ""), row.get("step", "")))
    write_csv(output_dir / "fvd_summary.csv", rows)


def run_counterfactual_metrics(manifest_path: Path, *, output_dir: Path, chunk_size: int) -> None:
    run_command(
        [
            str(MODAL),
            "run",
            "scripts/compute_counterfactual_sensitivity_modal.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            output_dir.name + "_counterfactual",
            "--chunk-size",
            str(chunk_size),
        ],
        LOG_DIR / "metrics" / f"{output_dir.name}_counterfactual.log",
        retries=3,
    )


def merge_quality_fvd(output_dir: Path) -> list[dict[str, Any]]:
    quality_rows = read_csv(output_dir / "model_summary.csv")
    fvd_rows = {row["model_mode"]: row for row in read_csv(output_dir / "fvd_summary.csv")}
    merged: list[dict[str, Any]] = []
    for row in quality_rows:
        model_mode = row["model_mode"]
        fvd = fvd_rows.get(model_mode, {})
        match = re.search(r"_step0*([0-9]+)", model_mode)
        spec_key = re.sub(r"_step0*[0-9]+.*$", "", model_mode)
        merged.append(
            {
                **row,
                "method_key": spec_key,
                "checkpoint_step": int(match.group(1)) if match else -1,
                "future_fvd_style": fvd.get("future_fvd_style", fvd.get("fvd_future", "")),
            }
        )
    merged.sort(key=lambda row: (row["method_key"], int(row["checkpoint_step"])))
    write_csv(output_dir / "model_summary_with_fvd.csv", merged)
    return merged


def as_float(row: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    if not row:
        return default
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return float(raw)


def plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, path: Path, *, lower_better: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    for key in sorted({row["method_key"] for row in rows}):
        subset = [row for row in rows if row["method_key"] == key and str(row.get(metric, "")) != ""]
        if not subset:
            continue
        subset.sort(key=lambda row: int(row["checkpoint_step"]))
        plt.plot(
            [int(row["checkpoint_step"]) for row in subset],
            [as_float(row, metric) for row in subset],
            marker="o",
            linewidth=2,
            label=key,
        )
    plt.xlabel("Checkpoint step")
    plt.ylabel(ylabel + (" (lower is better)" if lower_better else " (higher is better)"))
    plt.title(ylabel + " over final H100 V4 campaign checkpoints")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_plots(output_dir: Path) -> None:
    rows = merge_quality_fvd(output_dir)
    plot_dir = output_dir / "metric_plots"
    metrics = [
        ("future_fvd_style", "FVD-style future distance", "fvd_style_over_checkpoints.png", True),
        ("mean_future_psnr", "Future PSNR", "future_psnr_over_checkpoints.png", False),
        ("mean_future_global_ssim", "Future SSIM", "future_ssim_over_checkpoints.png", False),
        ("mean_sharpness_ratio_generated_over_reference", "Sharpness ratio", "sharpness_ratio_over_checkpoints.png", False),
        (
            "mean_fft_high_frequency_energy_ratio_generated_over_reference",
            "FFT high-frequency ratio",
            "fft_high_frequency_ratio_over_checkpoints.png",
            False,
        ),
        ("mean_motion_ratio_generated_over_reference", "Motion ratio", "motion_ratio_over_checkpoints.png", False),
        (
            "mean_low_frequency_motion_ratio_generated_over_reference",
            "Low-frequency motion ratio",
            "lowfreq_motion_ratio_over_checkpoints.png",
            False,
        ),
        ("mean_temporal_delta_error_mae", "Temporal delta error MAE", "temporal_delta_error_over_checkpoints.png", True),
    ]
    for metric, ylabel, filename, lower in metrics:
        plot_metric(rows, metric, ylabel, plot_dir / filename, lower_better=lower)
    write_json(
        plot_dir / "README.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plots": [filename for _, _, filename, _ in metrics],
        },
    )


def eval_checkpoint_sweep(specs: list[ModelSpec], *, steps: list[int], output_dir: Path, max_generation_workers: int) -> Path:
    configs = [EvalConfig(spec, step) for spec in specs for step in steps]
    generate_all(configs, max_workers=max_generation_workers, limit=5)
    download_all(configs, max_workers=max_generation_workers)
    manifest = build_manifest(
        configs,
        output_dir / "manifest_checkpoint_sweep.json",
        description=f"Final H100 V4 checkpoint sweep for {len(specs)} models.",
    )
    run_quality_metrics(manifest, output_dir=output_dir, chunk_size=1)
    run_fvd_by_spec(manifest, specs, output_dir=output_dir, max_workers=min(len(specs), max_generation_workers))
    make_plots(output_dir)
    return manifest


def eval_counterfactual(specs: list[ModelSpec], *, steps: list[int], output_dir: Path, max_generation_workers: int) -> Path:
    configs = counterfactual_configs(specs, steps=steps)
    generate_all(configs, max_workers=max_generation_workers, limit=5)
    download_all(configs, max_workers=max_generation_workers)
    manifest = build_manifest(
        configs,
        output_dir / "manifest_counterfactual.json",
        description=f"Final H100 V4 counterfactual sensitivity for {len(specs)} models.",
    )
    run_counterfactual_metrics(manifest, output_dir=output_dir, chunk_size=1)
    return manifest


def summarize_counterfactual_delta(output_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(output_dir / "counterfactual_sensitivity_summary.csv")
    if not rows:
        return []
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["method_key"], int(float(row["checkpoint_step"])))
        grouped.setdefault(key, []).append(row)
    by_method_step: dict[tuple[str, int], dict[str, float]] = {}
    for key, group in grouped.items():
        by_method_step[key] = {
            "mean_counterfactual_rgb_mae": sum(float(row["mean_future_rgb_mae_correct_vs_mode"]) for row in group)
            / len(group),
            "max_counterfactual_rgb_mae": max(float(row["mean_future_rgb_mae_correct_vs_mode"]) for row in group),
            "mean_counterfactual_temporal_delta_mae": sum(
                float(row["mean_future_temporal_delta_mae_correct_vs_mode"]) for row in group
            )
            / len(group),
        }
    summary: list[dict[str, Any]] = []
    for method in sorted({key[0] for key in by_method_step}):
        floor = by_method_step.get((method, 0))
        if floor is None:
            # We do not counterfactually generate step 0 by default; use the earliest available as the floor proxy.
            earliest = min(step for m, step in by_method_step if m == method)
            floor = by_method_step[(method, earliest)]
        for (m, step), values in by_method_step.items():
            if m != method:
                continue
            summary.append(
                {
                    "method_key": method,
                    "checkpoint_step": step,
                    **values,
                    "delta_vs_floor_mean_rgb_mae": values["mean_counterfactual_rgb_mae"]
                    - floor["mean_counterfactual_rgb_mae"],
                    "delta_vs_floor_max_rgb_mae": values["max_counterfactual_rgb_mae"]
                    - floor["max_counterfactual_rgb_mae"],
                    "delta_vs_floor_temporal_delta_mae": values["mean_counterfactual_temporal_delta_mae"]
                    - floor["mean_counterfactual_temporal_delta_mae"],
                }
            )
    summary.sort(key=lambda row: (row["method_key"], int(row["checkpoint_step"])))
    write_csv(output_dir / "counterfactual_sensitivity_delta_summary.csv", summary)
    return summary


def analyze_first_epoch(specs: list[ModelSpec]) -> list[ModelSpec]:
    rows = merge_quality_fvd(BENCHMARK_DIR)
    cf_rows = summarize_counterfactual_delta(BENCHMARK_DIR)
    cf_by_key_step = {(row["method_key"], int(row["checkpoint_step"])): row for row in cf_rows}
    rows_by_key_step = {(row["method_key"], int(row["checkpoint_step"])): row for row in rows}
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[float, ModelSpec]] = []
    for spec in specs:
        r3000 = rows_by_key_step.get((spec.key, 3000))
        r7992 = rows_by_key_step.get((spec.key, 7992))
        if not r7992:
            decisions.append({"method_key": spec.key, "eligible": False, "reason": "missing step_007992 metrics"})
            continue
        fvd_gain = as_float(r3000, "future_fvd_style") - as_float(r7992, "future_fvd_style") if r3000 else float("nan")
        psnr_gain = as_float(r7992, "mean_future_psnr") - as_float(r3000, "mean_future_psnr") if r3000 else float("nan")
        ssim_gain = as_float(r7992, "mean_future_global_ssim") - as_float(r3000, "mean_future_global_ssim") if r3000 else float("nan")
        sharp = as_float(r7992, "mean_sharpness_ratio_generated_over_reference")
        motion = as_float(r7992, "mean_motion_ratio_generated_over_reference")
        low_motion_gain = (
            as_float(r7992, "mean_low_frequency_motion_ratio_generated_over_reference")
            - as_float(r3000, "mean_low_frequency_motion_ratio_generated_over_reference")
            if r3000
            else float("nan")
        )
        temporal_worse = (
            as_float(r7992, "mean_temporal_delta_error_mae")
            - as_float(r3000, "mean_temporal_delta_error_mae")
            if r3000
            else float("nan")
        )
        cf3000 = cf_by_key_step.get((spec.key, 3000))
        cf7992 = cf_by_key_step.get((spec.key, 7992))
        cf_gain = (
            as_float(cf7992, "delta_vs_floor_mean_rgb_mae") - as_float(cf3000, "delta_vs_floor_mean_rgb_mae")
            if cf3000 and cf7992
            else float("nan")
        )
        hard_reject = (
            sharp < 0.20
            or motion < 0.65
            or (not (fvd_gain != fvd_gain) and fvd_gain < -3.0)
        )
        criteria = {
            "fvd_gain": fvd_gain >= 1.0 and sharp >= 0.24 and motion >= 0.75,
            "reconstruction_gain": (psnr_gain > 0 or ssim_gain > 0) and sharp >= 0.24 and motion >= 0.75,
            "counterfactual_gain": cf_gain >= 0.25,
            "lowfreq_motion_gain": low_motion_gain > 0 and temporal_worse <= 0.1,
        }
        score = (
            sum(1 for value in criteria.values() if value)
            + max(fvd_gain, 0.0) / 2.0
            + max(cf_gain, 0.0)
            + max(psnr_gain, 0.0) * 2.0
        )
        is_eligible = (not hard_reject) and any(criteria.values())
        decision = {
            "method_key": spec.key,
            "label": spec.label,
            "eligible": is_eligible,
            "score": score,
            "hard_reject": hard_reject,
            "fvd_gain_3000_to_7992": fvd_gain,
            "psnr_gain_3000_to_7992": psnr_gain,
            "ssim_gain_3000_to_7992": ssim_gain,
            "sharpness_step7992": sharp,
            "motion_step7992": motion,
            "lowfreq_motion_gain_3000_to_7992": low_motion_gain,
            "temporal_delta_worse_3000_to_7992": temporal_worse,
            "counterfactual_delta_gain_3000_to_7992": cf_gain,
            **{f"criterion_{key}": value for key, value in criteria.items()},
        }
        decisions.append(decision)
        if is_eligible:
            eligible.append((score, spec))
    decisions.sort(key=lambda row: (not row["eligible"], -float(row["score"])))
    selected_keys = {spec.key for _, spec in sorted(eligible, key=lambda item: item[0], reverse=True)[:2]}
    for row in decisions:
        row["selected_for_second_epoch"] = row["method_key"] in selected_keys
    write_csv(BENCHMARK_DIR / "second_epoch_decision_table.csv", decisions)
    write_analysis_report(decisions, rows, cf_rows)
    return [spec for _, spec in sorted(eligible, key=lambda item: item[0], reverse=True)[:2]]


def write_analysis_report(decisions: list[dict[str, Any]], rows: list[dict[str, Any]], cf_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final H100 V4 Campaign First-Epoch Analysis",
        "",
        "This analysis uses 5 validation clips, so small metric differences are treated as weak evidence. "
        "The decision gate prioritizes models that improve after step 3000 without sharpness/motion collapse.",
        "",
        "## Second-Epoch Decisions",
        "",
        "| method | selected | eligible | score | fvd gain | psnr gain | sharp@7992 | motion@7992 | cf gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            "| {method_key} | {selected} | {eligible} | {score:.3f} | {fvd:.3f} | {psnr:.4f} | {sharp:.3f} | {motion:.3f} | {cf:.3f} |".format(
                method_key=row["method_key"],
                selected=str(row["selected_for_second_epoch"]),
                eligible=str(row["eligible"]),
                score=float(row["score"]),
                fvd=float(row["fvd_gain_3000_to_7992"]) if row["fvd_gain_3000_to_7992"] == row["fvd_gain_3000_to_7992"] else float("nan"),
                psnr=float(row["psnr_gain_3000_to_7992"]) if row["psnr_gain_3000_to_7992"] == row["psnr_gain_3000_to_7992"] else float("nan"),
                sharp=float(row["sharpness_step7992"]) if row["sharpness_step7992"] == row["sharpness_step7992"] else float("nan"),
                motion=float(row["motion_step7992"]) if row["motion_step7992"] == row["motion_step7992"] else float("nan"),
                cf=float(row["counterfactual_delta_gain_3000_to_7992"]) if row["counterfactual_delta_gain_3000_to_7992"] == row["counterfactual_delta_gain_3000_to_7992"] else float("nan"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- FVD gain is `FVD(step3000) - FVD(step7992)`, so positive is better.",
            "- Counterfactual gain is action sensitivity delta relative to the stochastic floor.",
            "- A model is hard-rejected if sharpness drops below `0.20`, motion drops below `0.65`, or FVD worsens by more than `3.0` after step 3000.",
            "- At most two models are selected for a second epoch.",
            "",
        ]
    )
    (BENCHMARK_DIR / "first_epoch_deep_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_first_epoch_eval(specs: list[ModelSpec], *, max_generation_workers: int) -> None:
    eval_checkpoint_sweep(specs, steps=FIRST_EPOCH_STEPS, output_dir=BENCHMARK_DIR, max_generation_workers=max_generation_workers)
    eval_counterfactual(specs, steps=COUNTERFACTUAL_STEPS, output_dir=BENCHMARK_DIR, max_generation_workers=max_generation_workers)
    analyze_first_epoch(specs)


def run_second_epoch_eval(specs: list[ModelSpec], *, max_generation_workers: int) -> None:
    if not specs:
        write_json(BENCHMARK_DIR / "second_epoch_skipped.json", {"reason": "no eligible models"})
        return
    output_dir = BENCHMARK_DIR / "second_epoch"
    eval_checkpoint_sweep(specs, steps=SECOND_EPOCH_STEPS, output_dir=output_dir, max_generation_workers=max_generation_workers)
    eval_counterfactual(specs, steps=SECOND_EPOCH_STEPS, output_dir=output_dir, max_generation_workers=max_generation_workers)


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        TRAIN_WRAPPER,
        INFER_WRAPPER,
        "scripts/compute_video_quality_modal.py",
        "scripts/compute_action_fvd_modal.py",
        "scripts/compute_counterfactual_sensitivity_modal.py",
        "scripts/run_final_h100_v4_campaign.py",
    ]
    run_command([str(PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log", retries=0)


def write_campaign_summary(phase: str, selected: list[ModelSpec] | None = None) -> None:
    write_json(
        BENCHMARK_DIR / "campaign_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "benchmark_dir": str(BENCHMARK_DIR),
            "generated_root": str(GENERATED_ROOT),
            "first_epoch_steps": FIRST_EPOCH_STEPS,
            "counterfactual_steps": COUNTERFACTUAL_STEPS,
            "second_epoch_steps": SECOND_EPOCH_STEPS,
            "selected_second_epoch": [spec.key for spec in selected or []],
            "models": [spec.__dict__ for spec in SPECS],
            "corrected_setup": {
                "seed": SEED,
                "fps": FPS,
                "context_frames": CONTEXT_FRAMES,
                "future_frames": FUTURE_FRAMES,
                "total_frames": TOTAL_FRAMES,
                "image_cond_noise_scale": 0.0,
                "uses_24fps_full112_actions": True,
                "uses_original_10fps_actions": False,
                "recaches_latents": False,
                "train_gpu": "H100",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Final H100 V4 campaign with one-epoch decision gate.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "train1", "eval1", "analyze1", "train2", "eval2", "all"),
        default="all",
    )
    parser.add_argument("--max-train-workers", type=int, default=6)
    parser.add_argument("--max-generation-workers", type=int, default=10)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    args = parser.parse_args()

    selected: list[ModelSpec] = []
    if args.phase in {"syntax", "all"}:
        syntax_check()
    if args.phase in {"train1", "all"}:
        train_all(SPECS, max_workers=args.max_train_workers, force=args.force_train)
    if args.phase in {"eval1", "all"}:
        run_first_epoch_eval(SPECS, max_generation_workers=args.max_generation_workers)
    if args.phase in {"analyze1"}:
        selected = analyze_first_epoch(SPECS)
    if args.phase in {"train2", "all"}:
        selected = analyze_first_epoch(SPECS)
        train_all(selected, max_workers=min(len(selected), args.max_train_workers), second_epoch=True, force=args.force_train)
    if args.phase in {"eval2", "all"}:
        selected = analyze_first_epoch(SPECS)
        run_second_epoch_eval(selected, max_generation_workers=args.max_generation_workers)
    write_campaign_summary(args.phase, selected)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "benchmark_dir": str(BENCHMARK_DIR),
                "selected_second_epoch": [spec.key for spec in selected],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
