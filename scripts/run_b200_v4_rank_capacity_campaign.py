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

import run_final_h100_v4_campaign as campaign


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"
if not MODAL.exists():
    MODAL = Path("modal")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TRAIN_WRAPPER = "scripts/wrappers/train_ltx2b_distilled_waymo_frame_temporal_bottleneck_fullaction_motion_v4_action_lora.py"
INFER_WRAPPER = "scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py"
COPY_WRAPPER = "scripts/copy_modal_volume_subtree.py"

SOURCE_CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-r16-ckpts"
CHECKPOINT_VOLUME = "ltx2b-v4-b200-rank-capacity-ckpts"
ARTIFACT_VOLUME = "ltx2b-v4-b200-rank-capacity-infer"
RUNS_ROOT = "distilled098_full112_lowfreq_motion_v4_b200_rank_capacity_24fps_minterpolate_seed231_runs"

SOURCE_MAIN_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_b200_seed231_from_shifted_noaction_step003000_steps3000"
R16_CONTINUE_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r16_b200_seed231_resume015984_to023976"
R32_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_from_shifted_noaction_step003000_steps7992"
R64_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r64_b200_seed231_from_shifted_noaction_step003000_steps7992"
CALIBRATION_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_calibration100"

BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "b200_v4_rank_capacity_seed231"
GENERATED_ROOT = ROOT / "data" / "b200_v4_rank_capacity_generated"
LOG_DIR = ROOT / "data" / "modal_logs" / "b200_v4_rank_capacity_seed231"
SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"

SEED = 231
FIRST_EPOCH_TARGET = 7992
R16_TARGET = 23976
RANK_STEPS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 7992]
R16_STEPS = [15984, 18000, 20000, 22000, 23976]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]
COUNTERFACTUAL_STEPS_BY_KEY = {
    "v4_main_text_r16_continue": [15984, 22000, 23976],
    "v4_main_text_r32": [1000, 3000, 7992],
    "v4_main_text_r64": [1000, 3000, 7992],
}


@dataclass(frozen=True)
class B200Spec:
    key: str
    label: str
    run_name: str
    lora_rank: int
    max_steps: int
    checkpoint_steps: list[int]
    resume_from_checkpoint: str = ""
    resume_from_run_name: str = ""
    freeze_lora: bool = True
    expand_baseline_lora_to_rank: bool = False
    max_train_hours: float = 10.0
    disable_text: bool = False
    diffusion: float = 0.25
    lowfreq_target: float = 1.0
    lowfreq_delta: float = 1.0
    hf_teacher: float = 0.20
    action_motion_aux: float = 0.05
    residual: float = 0.002
    gate: float = 0.002
    learning_rate: float = 5e-6
    action_lr: float = 3e-5
    injector_lr: float = 3e-5
    gate_lr: float = 3e-5


CALIBRATION_SPEC = B200Spec(
    key="v4_main_text_r32_calibration100",
    label="V4 main text rank32 B200 calibration",
    run_name=CALIBRATION_RUN,
    lora_rank=32,
    max_steps=100,
    checkpoint_steps=[0, 100],
    freeze_lora=False,
    expand_baseline_lora_to_rank=True,
    max_train_hours=1.0,
)

SPECS = [
    B200Spec(
        key="v4_main_text_r16_continue",
        label="V4 main text rank16 B200 continuation from step 15984",
        run_name=R16_CONTINUE_RUN,
        lora_rank=16,
        max_steps=R16_TARGET,
        checkpoint_steps=R16_STEPS,
        resume_from_checkpoint="step_015984",
        resume_from_run_name=R16_CONTINUE_RUN,
        freeze_lora=True,
    ),
    B200Spec(
        key="v4_main_text_r32",
        label="V4 main text rank32 B200 one epoch",
        run_name=R32_RUN,
        lora_rank=32,
        max_steps=FIRST_EPOCH_TARGET,
        checkpoint_steps=RANK_STEPS,
        freeze_lora=False,
        expand_baseline_lora_to_rank=True,
    ),
    B200Spec(
        key="v4_main_text_r64",
        label="V4 main text rank64 B200 one epoch",
        run_name=R64_RUN,
        lora_rank=64,
        max_steps=FIRST_EPOCH_TARGET,
        checkpoint_steps=RANK_STEPS,
        freeze_lora=False,
        expand_baseline_lora_to_rank=True,
    ),
]


def configure_campaign_module() -> None:
    campaign.CHECKPOINT_VOLUME = CHECKPOINT_VOLUME
    campaign.ARTIFACT_VOLUME = ARTIFACT_VOLUME
    campaign.RUNS_ROOT = RUNS_ROOT
    campaign.BENCHMARK_DIR = BENCHMARK_DIR
    campaign.GENERATED_ROOT = GENERATED_ROOT
    campaign.LOG_DIR = LOG_DIR
    campaign.SOURCE_DIR = SOURCE_DIR


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
            log.write(
                "$ env "
                + " ".join(
                    f"{key}={value}"
                    for key, value in sorted(merged_env.items())
                    if key.startswith(("LTX_", "WAYMO24_", "ACTION_FVD_", "COPY_MODAL_"))
                )
                + "\n\n"
            )
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=merged_env)
        if proc.returncode == 0:
            return
        if attempt <= retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"Command failed after {retries + 1} attempts; see {log_path}")


def train_env(spec: B200Spec) -> dict[str, str]:
    return {
        "LTX_MODAL_TRAIN_GPU": "B200",
        "LTX_CHECKPOINT_VOLUME_NAME": CHECKPOINT_VOLUME,
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if spec.disable_text else "0",
        "WAYMO24_FREEZE_TRANSFORMER_LORA": "1" if spec.freeze_lora else "0",
        "WAYMO24_EXPAND_BASELINE_LORA_TO_RANK": "1" if spec.expand_baseline_lora_to_rank else "0",
    }


def infer_env(spec: B200Spec) -> dict[str, str]:
    return {
        "LTX_MODAL_GPU": "B200",
        "LTX_CHECKPOINT_VOLUME_NAME": CHECKPOINT_VOLUME,
        "LTX_ARTIFACTS_VOLUME_NAME": ARTIFACT_VOLUME,
        "LTX_RUNS_ROOT": RUNS_ROOT,
        "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if spec.disable_text else "0",
    }


def train_command(spec: B200Spec) -> list[str]:
    cmd = [
        str(MODAL),
        "run",
        TRAIN_WRAPPER,
        "--run-name",
        spec.run_name,
        "--max-steps",
        str(spec.max_steps),
        "--max-train-hours",
        str(spec.max_train_hours),
        "--batch-size",
        "1",
        "--learning-rate",
        str(spec.learning_rate),
        "--lora-rank",
        str(spec.lora_rank),
        "--lora-alpha",
        str(spec.lora_rank),
        "--action-learning-rate",
        str(spec.action_lr),
        "--action-injector-learning-rate",
        str(spec.injector_lr),
        "--action-gate-learning-rate",
        str(spec.gate_lr),
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        ",".join(str(step) for step in spec.checkpoint_steps),
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
    if spec.resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", spec.resume_from_checkpoint])
        cmd.extend(["--resume-from-run-name", spec.resume_from_run_name or spec.run_name])
    return cmd


def stage_r16_resume_checkpoint(force: bool = False) -> None:
    log_path = LOG_DIR / "preflight" / "stage_r16_step015984.log"
    if not force and command_completed(log_path):
        return
    cmd = [
        str(MODAL),
        "run",
        COPY_WRAPPER,
        "--src-relpath",
        f"{SOURCE_MAIN_RUN}/step_015984",
        "--dst-relpath",
        f"{R16_CONTINUE_RUN}/step_015984",
    ]
    if force:
        cmd.append("--force")
    run_command(
        cmd,
        log_path,
        env={
            "COPY_MODAL_SRC_VOLUME_NAME": SOURCE_CHECKPOINT_VOLUME,
            "COPY_MODAL_DST_VOLUME_NAME": CHECKPOINT_VOLUME,
        },
        retries=1,
    )


def parse_sec_per_step(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    values = [float(match) for match in re.findall(r'"sec_per_step"\s*:\s*([0-9.]+)', text)]
    return values[-1] if values else float("nan")


def train_one(spec: B200Spec, *, force: bool = False) -> str:
    log_path = LOG_DIR / "training" / f"{spec.key}_to{spec.max_steps}.log"
    if not force and command_completed(log_path):
        return spec.key
    run_command(train_command(spec), log_path, env=train_env(spec), retries=1, retry_sleep=180)
    return spec.key


def download_checkpoint_json(run_name: str, checkpoint_name: str, relpath: str, dst: Path) -> None:
    remote = f"{run_name}/{checkpoint_name}/{relpath}"
    run_command(
        [str(MODAL), "volume", "get", "--force", CHECKPOINT_VOLUME, remote, str(dst)],
        LOG_DIR / "download" / f"{run_name}_{checkpoint_name}_{relpath.replace('/', '_')}.log",
        retries=2,
    )


def run_calibration(force: bool = False) -> float:
    log_path = LOG_DIR / "training" / f"{CALIBRATION_SPEC.key}_to100.log"
    if force or not command_completed(log_path):
        run_command(train_command(CALIBRATION_SPEC), log_path, env=train_env(CALIBRATION_SPEC), retries=0)
    sec_per_step = parse_sec_per_step(log_path)
    audit = audit_rank_calibration(sec_per_step)
    write_json(BENCHMARK_DIR / "b200_calibration_runtime.json", audit)
    return float(audit["sec_per_step"])


def audit_rank_calibration(sec_per_step: float) -> dict[str, Any]:
    tmp = BENCHMARK_DIR / "tmp" / "calibration_audit"
    tmp.mkdir(parents=True, exist_ok=True)
    training_config = tmp / "training_config.json"
    adapter_config = tmp / "adapter_config.json"
    loss_history = tmp / "loss_history.json"
    run_summary = tmp / "run_summary.json"
    download_checkpoint_json(CALIBRATION_RUN, "step_000100", "training_config.json", training_config)
    download_checkpoint_json(CALIBRATION_RUN, "step_000100", "lora_adapter/adapter_config.json", adapter_config)
    download_checkpoint_json(CALIBRATION_RUN, "step_000100", "loss_history.json", loss_history)
    run_command(
        [str(MODAL), "volume", "get", "--force", CHECKPOINT_VOLUME, f"{CALIBRATION_RUN}/run_summary.json", str(run_summary)],
        LOG_DIR / "download" / "calibration_run_summary.log",
        retries=2,
    )
    config = json.loads(training_config.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
    losses = json.loads(loss_history.read_text(encoding="utf-8")).get("loss_history", [])
    summary = json.loads(run_summary.read_text(encoding="utf-8"))
    last_loss = losses[-1] if losses else {}
    adapter_rank = int(adapter.get("r", -1))
    trainable_counts = summary.get("trainable_parameter_counts", {})
    ok = (
        adapter_rank == 32
        and int(config.get("lora_rank", -1)) == 32
        and not bool(config.get("freeze_transformer_lora", True))
        and bool(config.get("expand_baseline_lora_to_rank", False))
        and int(trainable_counts.get("lora", 0)) > 0
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sec_per_step": sec_per_step,
        "calibration_run": CALIBRATION_RUN,
        "adapter_rank": adapter_rank,
        "lora_alpha": adapter.get("lora_alpha"),
        "training_config_lora_rank": config.get("lora_rank"),
        "freeze_transformer_lora": config.get("freeze_transformer_lora"),
        "expand_baseline_lora_to_rank": config.get("expand_baseline_lora_to_rank"),
        "trainable_parameter_counts": trainable_counts,
        "last_grad_norm_lora": last_loss.get("grad_norm_lora"),
        "last_loss": last_loss.get("loss"),
        "audit_passed": ok,
        "budget_decision": "run_all_three" if sec_per_step <= 5.0 else "skip_r64_after_calibration",
    }
    write_json(BENCHMARK_DIR / "trainable_parameter_audit.json", payload)
    if not ok:
        raise RuntimeError(f"Calibration trainable audit failed: {json.dumps(payload, indent=2, sort_keys=True)}")
    return payload


def selected_train_specs(sec_per_step: float) -> list[B200Spec]:
    if sec_per_step > 5.0:
        return [SPECS[0], SPECS[1]]
    return SPECS


def train_selected(specs: list[B200Spec], *, force: bool = False) -> None:
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = [executor.submit(train_one, spec, force=force) for spec in specs]
        for future in as_completed(futures):
            print(json.dumps({"trained": future.result()}, sort_keys=True), flush=True)


def checkpoint_name(step: int) -> str:
    return "step_000000_base_reference" if step == 0 else f"step_{step:06d}"


def eval_configs(specs: list[B200Spec]) -> list[campaign.EvalConfig]:
    return [campaign.EvalConfig(spec, step) for spec in specs for step in spec.checkpoint_steps]


def counterfactual_configs(specs: list[B200Spec]) -> list[campaign.EvalConfig]:
    configs: list[campaign.EvalConfig] = []
    for spec in specs:
        for step in COUNTERFACTUAL_STEPS_BY_KEY.get(spec.key, []):
            for mode in COUNTERFACTUAL_MODES:
                configs.append(
                    campaign.EvalConfig(
                        spec,
                        step,
                        diagnostic_group="counterfactual_suite",
                        counterfactual_action_mode=mode,
                    )
                )
    return configs


def generate_one(config: campaign.EvalConfig, *, limit: int, force: bool = False) -> str:
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
    run_command(cmd, log_path, env=infer_env(config.spec), retries=2)
    return config.run_label


def generate_all(configs: list[campaign.EvalConfig], *, max_workers: int, limit: int, force: bool = False) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, config, limit=limit, force=force) for config in configs]
        for future in as_completed(futures):
            print(json.dumps({"generated": future.result()}, sort_keys=True), flush=True)


def run_checkpoint_sweep(specs: list[B200Spec], *, max_generation_workers: int, force_generate: bool = False) -> Path:
    configs = eval_configs(specs)
    generate_all(configs, max_workers=max_generation_workers, limit=5, force=force_generate)
    campaign.download_all(configs, max_workers=max_generation_workers)
    manifest = campaign.build_manifest(
        configs,
        BENCHMARK_DIR / "manifest_checkpoint_sweep.json",
        description="B200 V4 rank/capacity checkpoint sweep on the fixed 5 validation clips.",
    )
    campaign.run_quality_metrics(manifest, output_dir=BENCHMARK_DIR, chunk_size=1)
    run_fvd_by_spec(manifest, specs, output_dir=BENCHMARK_DIR, max_workers=min(len(specs), max_generation_workers))
    make_plots(BENCHMARK_DIR)
    return manifest


def run_fvd_by_spec(manifest_path: Path, specs: list[B200Spec], *, output_dir: Path, max_workers: int) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts_dir = output_dir / "fvd_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_parts: list[tuple[B200Spec, Path]] = []
    for spec in specs:
        records = [row for row in payload["records"] if row.get("method_key") == spec.key]
        if not records:
            continue
        part_path = parts_dir / f"manifest_{spec.key}.json"
        write_json(part_path, {**payload, "records": records})
        manifest_parts.append((spec, part_path))

    def run_part(item: tuple[B200Spec, Path]) -> str:
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
            retries=2,
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


def run_counterfactual(specs: list[B200Spec], *, max_generation_workers: int, force_generate: bool = False) -> Path:
    configs = counterfactual_configs(specs)
    generate_all(configs, max_workers=max_generation_workers, limit=5, force=force_generate)
    campaign.download_all(configs, max_workers=max_generation_workers)
    manifest = campaign.build_manifest(
        configs,
        BENCHMARK_DIR / "manifest_counterfactual.json",
        description="B200 V4 rank/capacity counterfactual sensitivity on the fixed 5 validation clips.",
    )
    campaign.run_counterfactual_metrics(manifest, output_dir=BENCHMARK_DIR, chunk_size=1)
    return manifest


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return float(raw)


def merge_quality_fvd(output_dir: Path) -> list[dict[str, Any]]:
    rows = campaign.merge_quality_fvd(output_dir)
    for row in rows:
        row["rank"] = 16 if "r16" in row["method_key"] else (32 if "r32" in row["method_key"] else (64 if "r64" in row["method_key"] else ""))
    write_csv(output_dir / "checkpoint_metrics_by_model.csv", rows)
    return rows


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
    plt.title(ylabel + " over B200 V4 rank/capacity checkpoints")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_plots(output_dir: Path) -> None:
    rows = merge_quality_fvd(output_dir)
    plot_dir = output_dir / "metric_plots"
    plot_metric(rows, "future_fvd_style", "FVD-style future distance", plot_dir / "v4_b200_fvd_over_checkpoints.png", lower_better=True)
    plot_metric(rows, "mean_future_psnr", "Future PSNR", plot_dir / "v4_b200_psnr_over_checkpoints.png")
    plot_metric(rows, "mean_future_global_ssim", "Future SSIM", plot_dir / "v4_b200_ssim_over_checkpoints.png")
    plot_metric(
        rows,
        "mean_sharpness_ratio_generated_over_reference",
        "Sharpness ratio",
        plot_dir / "v4_b200_sharpness_over_checkpoints.png",
    )
    plot_metric(rows, "mean_motion_ratio_generated_over_reference", "Motion ratio", plot_dir / "v4_b200_motion_over_checkpoints.png")
    plot_metric(
        rows,
        "mean_fft_high_frequency_energy_ratio_generated_over_reference",
        "FFT high-frequency ratio",
        plot_dir / "v4_b200_fft_high_frequency_over_checkpoints.png",
    )

    plt.figure(figsize=(8, 6))
    for row in rows:
        if str(row.get("future_fvd_style", "")) == "":
            continue
        plt.scatter(
            as_float(row, "future_fvd_style"),
            as_float(row, "mean_sharpness_ratio_generated_over_reference"),
            s=70,
        )
        plt.annotate(row["model_mode"], (as_float(row, "future_fvd_style"), as_float(row, "mean_sharpness_ratio_generated_over_reference")), fontsize=6)
    plt.xlabel("FVD-style future distance, lower better")
    plt.ylabel("Sharpness ratio, higher better")
    plt.tight_layout()
    plt.savefig(plot_dir / "v4_b200_pareto_fvd_vs_sharpness.png", dpi=180)
    plt.close()

    write_json(
        plot_dir / "README.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Composite plots copied to benchmark root with required filenames after counterfactual metrics are available.",
        },
    )


def extract_loss_curves(specs: list[B200Spec]) -> None:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for step in spec.checkpoint_steps:
            tmp = BENCHMARK_DIR / "tmp" / "loss_curves" / spec.key / checkpoint_name(step)
            tmp.mkdir(parents=True, exist_ok=True)
            local = tmp / "loss_history.json"
            try:
                download_checkpoint_json(spec.run_name, checkpoint_name(step), "loss_history.json", local)
            except RuntimeError:
                continue
            payload = json.loads(local.read_text(encoding="utf-8"))
            for row in payload.get("loss_history", []):
                rows.append({"method_key": spec.key, "rank": spec.lora_rank, **row})
    rows.sort(key=lambda row: (row["method_key"], int(row["step"])))
    write_csv(BENCHMARK_DIR / "loss_curves_by_model.csv", rows)
    if not rows:
        return
    plt.figure(figsize=(11, 6))
    for key in sorted({row["method_key"] for row in rows}):
        subset = [row for row in rows if row["method_key"] == key]
        plt.plot([int(row["step"]) for row in subset], [float(row["loss"]) for row in subset], linewidth=1.5, label=key)
    plt.xlabel("Training step")
    plt.ylabel("Total loss")
    plt.title("B200 V4 rank/capacity loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    (BENCHMARK_DIR / "metric_plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(BENCHMARK_DIR / "metric_plots" / "v4_b200_loss_curves.png", dpi=180)
    plt.close()


def copy_required_plot_names() -> None:
    plot_dir = BENCHMARK_DIR / "metric_plots"
    mappings = {
        "v4_b200_fvd_over_checkpoints.png": "v4_b200_fvd_over_checkpoints.png",
        "v4_b200_sharpness_over_checkpoints.png": "v4_b200_sharpness_motion_over_checkpoints.png",
        "v4_b200_psnr_over_checkpoints.png": "v4_b200_psnr_ssim_over_checkpoints.png",
        "v4_b200_loss_curves.png": "v4_b200_loss_curves.png",
    }
    for src_name, dst_name in mappings.items():
        src = plot_dir / src_name
        dst = BENCHMARK_DIR / dst_name
        if src.exists():
            dst.write_bytes(src.read_bytes())
    sensitivity_csv = BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv"
    if sensitivity_csv.exists():
        rows = read_csv(sensitivity_csv)
        if rows:
            plt.figure(figsize=(10, 5))
            grouped: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                grouped.setdefault(row.get("method_key", row.get("variant", "")), []).append(row)
            for key, group in sorted(grouped.items()):
                by_step: dict[int, list[float]] = {}
                for row in group:
                    by_step.setdefault(int(float(row["checkpoint_step"])), []).append(float(row["mean_future_rgb_mae_correct_vs_mode"]))
                steps = sorted(by_step)
                vals = [sum(by_step[step]) / len(by_step[step]) for step in steps]
                plt.plot(steps, vals, marker="o", label=key)
            plt.xlabel("Checkpoint step")
            plt.ylabel("Mean counterfactual RGB MAE")
            plt.title("B200 V4 counterfactual sensitivity")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(BENCHMARK_DIR / "v4_b200_counterfactual_sensitivity.png", dpi=180)
            plt.close()
            metrics = read_csv(BENCHMARK_DIR / "checkpoint_metrics_by_model.csv")
            metric_by_key_step = {
                (row["method_key"], int(row["checkpoint_step"])): row
                for row in metrics
                if str(row.get("future_fvd_style", "")) != ""
            }
            sensitivity_by_key_step: dict[tuple[str, int], list[float]] = {}
            for row in rows:
                key = row.get("method_key", row.get("variant", ""))
                step = int(float(row["checkpoint_step"]))
                sensitivity_by_key_step.setdefault((key, step), []).append(float(row["mean_future_rgb_mae_correct_vs_mode"]))
            pareto_rows = []
            for key_step, values in sensitivity_by_key_step.items():
                metric = metric_by_key_step.get(key_step)
                if not metric:
                    continue
                pareto_rows.append(
                    {
                        "method_key": key_step[0],
                        "checkpoint_step": key_step[1],
                        "action_sensitivity": sum(values) / len(values),
                        "future_fvd_style": float(metric["future_fvd_style"]),
                        "sharpness": float(metric["mean_sharpness_ratio_generated_over_reference"]),
                    }
                )
            if pareto_rows:
                plt.figure(figsize=(8, 5.5))
                for row in pareto_rows:
                    plt.scatter(row["future_fvd_style"], row["action_sensitivity"], s=80)
                    plt.annotate(f"{row['method_key']}@{row['checkpoint_step']}", (row["future_fvd_style"], row["action_sensitivity"]), fontsize=7)
                plt.xlabel("FVD-style future distance, lower better")
                plt.ylabel("Counterfactual RGB MAE, higher means actions matter more")
                plt.tight_layout()
                plt.savefig(BENCHMARK_DIR / "v4_b200_pareto_fvd_vs_action_sensitivity.png", dpi=180)
                plt.close()

                plt.figure(figsize=(8, 5.5))
                for row in pareto_rows:
                    plt.scatter(row["sharpness"], row["action_sensitivity"], s=80)
                    plt.annotate(f"{row['method_key']}@{row['checkpoint_step']}", (row["sharpness"], row["action_sensitivity"]), fontsize=7)
                plt.xlabel("Sharpness ratio, interpret relative to Waymo reference")
                plt.ylabel("Counterfactual RGB MAE, higher means actions matter more")
                plt.tight_layout()
                plt.savefig(BENCHMARK_DIR / "v4_b200_pareto_sharpness_vs_action_sensitivity.png", dpi=180)
                plt.close()


def write_analysis(specs: list[B200Spec]) -> None:
    metrics = read_csv(BENCHMARK_DIR / "checkpoint_metrics_by_model.csv")
    cf = read_csv(BENCHMARK_DIR / "counterfactual_sensitivity_summary.csv")
    latest_by_key: dict[str, dict[str, str]] = {}
    for row in metrics:
        key = row["method_key"]
        if key not in latest_by_key or int(row["checkpoint_step"]) > int(latest_by_key[key]["checkpoint_step"]):
            latest_by_key[key] = row
    lines = [
        "# B200 V4 Rank/Capacity Campaign Analysis",
        "",
        "This run tests whether V4 main text was limited by rank/capacity or simply needed longer training.",
        "",
        "## Latest Checkpoint Summary",
        "",
        "| method | step | rank | FVD-style | PSNR | SSIM | sharpness | motion | FFT HF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        row = latest_by_key.get(spec.key)
        if not row:
            continue
        lines.append(
            "| {key} | {step} | {rank} | {fvd} | {psnr} | {ssim} | {sharp} | {motion} | {fft} |".format(
                key=spec.key,
                step=row.get("checkpoint_step", ""),
                rank=spec.lora_rank,
                fvd=row.get("future_fvd_style", ""),
                psnr=row.get("mean_future_psnr", ""),
                ssim=row.get("mean_future_global_ssim", ""),
                sharp=row.get("mean_sharpness_ratio_generated_over_reference", ""),
                motion=row.get("mean_motion_ratio_generated_over_reference", ""),
                fft=row.get("mean_fft_high_frequency_energy_ratio_generated_over_reference", ""),
            )
        )
    lines.extend(["", "## Counterfactual Sensitivity", ""])
    if cf:
        lines.extend(
            [
                "| method | step | mean RGB MAE | max RGB MAE |",
                "|---|---:|---:|---:|",
            ]
        )
        grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in cf:
            grouped.setdefault((row.get("method_key", row.get("variant", "")), int(float(row["checkpoint_step"]))), []).append(row)
        for (key, step), group in sorted(grouped.items()):
            mean_rgb = sum(float(row["mean_future_rgb_mae_correct_vs_mode"]) for row in group) / len(group)
            max_rgb = max(float(row["mean_future_rgb_mae_correct_vs_mode"]) for row in group)
            lines.append(f"| {key} | {step} | {mean_rgb:.4f} | {max_rgb:.4f} |")
    else:
        lines.append("Counterfactual metrics were not available when this report was written.")
    lines.extend(
        [
            "",
            "## Interpretation Defaults",
            "",
            "- If rank32/rank64 improve counterfactual sensitivity without hurting FVD/motion, V4 was capacity-limited.",
            "- If rank16 continuation improves sensitivity while rank32/rank64 do not, longer training mattered more than rank.",
            "- If all curves stay flat, the final story should be framed as action-conditioning interference rather than successful control.",
        ]
    )
    (BENCHMARK_DIR / "b200_rank_capacity_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        TRAIN_WRAPPER,
        INFER_WRAPPER,
        COPY_WRAPPER,
        "scripts/compute_video_quality_modal.py",
        "scripts/compute_action_fvd_modal.py",
        "scripts/compute_counterfactual_sensitivity_modal.py",
        "scripts/run_b200_v4_rank_capacity_campaign.py",
    ]
    run_command([str(PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log", retries=0)


def write_campaign_summary(phase: str, specs: list[B200Spec]) -> None:
    write_json(
        BENCHMARK_DIR / "campaign_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "source_checkpoint_volume": SOURCE_CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "runs_root": RUNS_ROOT,
            "benchmark_dir": str(BENCHMARK_DIR),
            "generated_root": str(GENERATED_ROOT),
            "gpu": "B200",
            "budget_usd": 230,
            "modal_b200_usd_per_hour": 6.25,
            "corrected_setup": {
                "seed": campaign.SEED,
                "fps": campaign.FPS,
                "context_frames": campaign.CONTEXT_FRAMES,
                "future_frames": campaign.FUTURE_FRAMES,
                "total_frames": campaign.TOTAL_FRAMES,
                "image_cond_noise_scale": 0.0,
                "uses_24fps_full112_actions": True,
                "uses_original_10fps_actions": False,
                "recaches_latents": False,
            },
            "models": [spec.__dict__ for spec in specs],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the B200 V4 rank/capacity campaign.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "stage", "calibrate", "train", "sweep", "counterfactual", "loss", "plots", "all"),
        default="all",
    )
    parser.add_argument("--max-generation-workers", type=int, default=10)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()

    configure_campaign_module()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    selected_specs: list[B200Spec] = SPECS
    if args.phase in {"syntax", "all"}:
        syntax_check()
    if args.phase in {"stage", "all"}:
        stage_r16_resume_checkpoint(force=args.force_stage)
    sec_per_step = float("nan")
    if args.phase in {"calibrate", "all"}:
        sec_per_step = run_calibration(force=args.force_train)
        selected_specs = selected_train_specs(sec_per_step)
    elif (BENCHMARK_DIR / "b200_calibration_runtime.json").exists():
        sec_per_step = float(json.loads((BENCHMARK_DIR / "b200_calibration_runtime.json").read_text(encoding="utf-8"))["sec_per_step"])
        selected_specs = selected_train_specs(sec_per_step)
    if args.phase in {"train", "all"}:
        train_selected(selected_specs, force=args.force_train)
    if args.phase in {"sweep", "all"}:
        run_checkpoint_sweep(selected_specs, max_generation_workers=args.max_generation_workers, force_generate=args.force_generate)
    if args.phase in {"counterfactual", "all"}:
        run_counterfactual(selected_specs, max_generation_workers=args.max_generation_workers, force_generate=args.force_generate)
    if args.phase in {"loss", "all"}:
        extract_loss_curves(selected_specs)
    if args.phase in {"plots", "all"}:
        make_plots(BENCHMARK_DIR)
        copy_required_plot_names()
        write_analysis(selected_specs)
    write_campaign_summary(args.phase, selected_specs)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "sec_per_step": sec_per_step,
                "selected_specs": [spec.key for spec in selected_specs],
                "benchmark_dir": str(BENCHMARK_DIR),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
