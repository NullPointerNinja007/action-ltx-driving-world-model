from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_frame_temporal_bottleneck_hfteacher_gate_scale_sweep as sweep


RUN_NAME = "ltx2b_dist098_waymo24_frame_temporal_bottleneck_lowfreq_v3b_gate1e4_proj1e3_seed231_from_shifted_noaction_step003000_steps3000"
CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-framebneck-lowfreq-v3-r16-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-framebneck-lowfreq-v3-infer"
RUNS_ROOT = "distilled098_framebneck_lowfreq_v3_action_lora_24fps_minterpolate_seed231_runs"
TRAIN_WRAPPER = "scripts/wrappers/train_ltx2b_distilled_waymo_frame_temporal_bottleneck_lowfreq_v3_action_lora.py"
INFER_WRAPPER = "scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_lowfreq_v3_action_minterpolate_lora.py"

BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_seed231_all5"
LOCAL_GENERATED_ROOT = ROOT / "data" / "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_generated"
SIDE_BY_SIDE_DIR = ROOT / "data" / "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_side_by_side_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_seed231_all5"

METHOD_KEY = "frame_temporal_bottleneck_lowfreq_v3"
METHOD_LABEL = "Frame Temporal Bottleneck Low-Frequency V3"
MANIFEST_STEM = "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_seed231_all5"
COUNTERFACTUAL_MANIFEST_STEM = "frame_temporal_bottleneck_lowfreq_v3_gate_scale_counterfactual_seed231_all5"

CHECKPOINT_STEPS = "0,100,250,500,750,1000,1500,2000,2500,3000"
SWEEP_CHECKPOINTS = [
    "step_000100",
    "step_000250",
    "step_000500",
    "step_001000",
    "step_002000",
    "step_003000",
]
COUNTERFACTUAL_BASE_CHECKPOINTS = ["step_000250", "step_000500", "step_001000"]


def configure_sweep_globals() -> None:
    sweep.LOCAL_GENERATED_ROOT = LOCAL_GENERATED_ROOT
    sweep.BENCHMARK_DIR = BENCHMARK_DIR
    sweep.SIDE_BY_SIDE_DIR = SIDE_BY_SIDE_DIR
    sweep.LOG_DIR = LOG_DIR
    sweep.RUN_NAME = RUN_NAME
    sweep.CHECKPOINT_VOLUME = CHECKPOINT_VOLUME
    sweep.ARTIFACT_VOLUME = ARTIFACT_VOLUME
    sweep.RUNS_ROOT = RUNS_ROOT
    sweep.WRAPPER = INFER_WRAPPER
    sweep.METHOD_KEY = METHOD_KEY
    sweep.METHOD_LABEL = METHOD_LABEL
    sweep.MANIFEST_STEM = MANIFEST_STEM
    sweep.COUNTERFACTUAL_MANIFEST_STEM = COUNTERFACTUAL_MANIFEST_STEM
    sweep.FVD_RUN_ID = MANIFEST_STEM
    sweep.PLOT_README_TITLE = "V3 Low-Frequency Temporal Bottleneck Gate-Scale Sweep Metric Plots"
    sweep.SIDE_BY_SIDE_STEM = "lowfreq_v3_gate_sweep"
    sweep.SIDE_BY_SIDE_MANIFEST_NAME = "manifest_lowfreq_v3_gate_scale_side_by_side.json"
    sweep.FINAL_SUMMARY_NAME = "lowfreq_v3_gate_scale_sweep_summary.json"
    sweep.GATE_MANIFEST = BENCHMARK_DIR / f"manifest_{MANIFEST_STEM}.json"
    sweep.COUNTERFACTUAL_MANIFEST = BENCHMARK_DIR / f"manifest_{COUNTERFACTUAL_MANIFEST_STEM}.json"
    sweep.SELECTION_PATH = BENCHMARK_DIR / "lowfreq_v3_gate_scale_selection.json"
    sweep.SWEEP_CHECKPOINTS = list(SWEEP_CHECKPOINTS)
    sweep.COUNTERFACTUAL_BASE_CHECKPOINTS = list(COUNTERFACTUAL_BASE_CHECKPOINTS)


def log_contains_success(path: Path) -> bool:
    return path.exists() and "App completed" in path.read_text(encoding="utf-8", errors="ignore")


def run_local(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        TRAIN_WRAPPER,
        INFER_WRAPPER,
        "scripts/run_frame_temporal_bottleneck_hfteacher_gate_scale_sweep.py",
        "scripts/run_frame_temporal_bottleneck_lowfreq_v3_campaign.py",
        "scripts/compute_counterfactual_sensitivity_modal.py",
        "scripts/compute_video_quality_modal.py",
    ]
    run_local([str(sweep.PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log")


def train_v3(*, smoke: bool = False, force: bool = False) -> None:
    if smoke:
        run_name = f"{RUN_NAME}_smoke10"
        log_path = LOG_DIR / "training" / f"{run_name}.log"
        max_steps = "10"
        train_limit = "8"
        val_limit = "8"
        checkpoint_steps = "0,10"
    else:
        run_name = RUN_NAME
        log_path = LOG_DIR / "training" / f"{run_name}.log"
        max_steps = "3000"
        train_limit = "0"
        val_limit = "32"
        checkpoint_steps = CHECKPOINT_STEPS

    if not force and log_contains_success(log_path):
        print(json.dumps({"train_v3": "skipped", "run_name": run_name, "reason": "completed log exists"}, sort_keys=True))
        return

    cmd = [
        str(sweep.MODAL),
        "run",
        TRAIN_WRAPPER,
        "--run-name",
        run_name,
        "--max-steps",
        max_steps,
        "--max-train-hours",
        "8.0",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        "3e-5",
        "--action-injector-learning-rate",
        "3e-5",
        "--action-gate-learning-rate",
        "1e-4",
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
        "0.25",
        "--lowfreq-target-loss-weight",
        "1.0",
        "--lowfreq-delta-loss-weight",
        "1.0",
        "--hf-teacher-loss-weight",
        "0.20",
        "--action-residual-loss-weight",
        "0.002",
        "--action-gate-loss-weight",
        "0.002",
        "--action-gate-scale",
        "1.0",
        "--action-gate-bound",
        "0.25",
        "--action-hidden-dim",
        "256",
        "--action-transformer-layers",
        "4",
        "--action-transformer-heads",
        "8",
        "--num-val-samples",
        "0",
        "--train-limit",
        train_limit,
        "--val-limit",
        val_limit,
    ]
    sweep.run_command(cmd, log_path)


def run_smoke_generation(max_workers: int, seed: int) -> None:
    configs = [
        sweep.EvalConfig("smoke", "step_000100", action_gate_scale=0.0),
        sweep.EvalConfig("smoke", "step_000100", action_gate_scale=0.1),
    ]
    sweep.generate_all(configs, max_workers=min(max_workers, 2), seed=seed, limit=1)
    sweep.download_all(configs)
    sweep.build_manifest(
        configs,
        BENCHMARK_DIR / f"manifest_{MANIFEST_STEM}_smoke.json",
        seed=seed,
        limit=1,
        description="Smoke test for V3 low-frequency temporal bottleneck action-conditioning.",
    )


def run_sweep(max_workers: int, seed: int, limit: int) -> None:
    configs = sweep.gate_configs()
    sweep.generate_all(configs, max_workers=max_workers, seed=seed, limit=limit)
    sweep.download_all(configs)
    sweep.build_manifest(
        configs,
        sweep.GATE_MANIFEST,
        seed=seed,
        limit=limit,
        description="V3 low-frequency temporal bottleneck gate-scale sweep.",
    )
    sweep.run_quality_metrics(sweep.GATE_MANIFEST)
    sweep.run_fvd(sweep.GATE_MANIFEST, run_id=sweep.FVD_RUN_ID)
    rows = sweep.write_gate_scale_summary()
    sweep.aggregate_gate_rows(rows)
    sweep.select_counterfactual_settings()
    sweep.make_plots(include_sensitivity=False)


def run_counterfactual(max_workers: int, seed: int, limit: int) -> None:
    configs = sweep.counterfactual_configs()
    sweep.generate_all(configs, max_workers=max_workers, seed=seed, limit=limit)
    sweep.download_all(configs)
    sweep.build_manifest(
        configs,
        sweep.COUNTERFACTUAL_MANIFEST,
        seed=seed,
        limit=limit,
        description="V3 low-frequency temporal bottleneck counterfactual action sensitivity suite.",
    )
    sweep.run_counterfactual_sensitivity(sweep.COUNTERFACTUAL_MANIFEST)
    sweep.make_plots(include_sensitivity=True)
    sweep.build_side_by_side()
    sweep.decide_v2_support()
    sweep.write_final_summary()


def write_campaign_summary(phase: str) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "run_name": RUN_NAME,
        "checkpoint_volume": CHECKPOINT_VOLUME,
        "artifact_volume": ARTIFACT_VOLUME,
        "benchmark_dir": str(BENCHMARK_DIR),
        "generated_dir": str(LOCAL_GENERATED_ROOT),
        "side_by_side_dir": str(SIDE_BY_SIDE_DIR),
        "method_key": METHOD_KEY,
        "corrected_setup": {
            "seed": 231,
            "fps": 24,
            "context_frames": 49,
            "future_frames": 72,
            "total_frames": 121,
            "image_cond_noise_scale": 0.0,
            "timestep_sampling": "shifted_lognormal",
            "latent_prefix": "latents",
            "uses_upsampled_frame_actions": True,
            "uses_original_10fps_actions": False,
            "recaches_latents": False,
        },
    }
    (BENCHMARK_DIR / "lowfreq_v3_campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    configure_sweep_globals()
    parser = argparse.ArgumentParser(description="Run the V3 low-frequency temporal bottleneck campaign.")
    parser.add_argument(
        "--phase",
        choices=(
            "syntax",
            "smoke_train",
            "train",
            "smoke_generation",
            "sweep",
            "counterfactual",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=sweep.SEED)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-smoke-train", action="store_true")
    parser.add_argument("--skip-smoke-generation", action="store_true")
    args = parser.parse_args()

    if args.phase in {"syntax", "all"}:
        syntax_check()
    if args.phase in {"smoke_train", "all"} and not args.skip_smoke_train:
        train_v3(smoke=True, force=args.force_train)
    if args.phase in {"train", "all"}:
        train_v3(smoke=False, force=args.force_train)
    if args.phase in {"smoke_generation", "all"} and not args.skip_smoke_generation:
        run_smoke_generation(max_workers=args.max_workers, seed=args.seed)
    if args.phase in {"sweep", "all"}:
        run_sweep(max_workers=args.max_workers, seed=args.seed, limit=args.limit)
    if args.phase in {"counterfactual", "all"}:
        run_counterfactual(max_workers=args.max_workers, seed=args.seed, limit=args.limit)

    write_campaign_summary(args.phase)
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
