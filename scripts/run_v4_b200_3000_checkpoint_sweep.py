from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = (
    "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_b200_seed231_"
    "from_shifted_noaction_step003000_steps3000"
)

BENCHMARK_DIR = (
    ROOT
    / "data"
    / "benchmarks"
    / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_main_b200_3000_seed231_all5"
)
LOCAL_GENERATED_ROOT = (
    ROOT / "data" / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_main_b200_3000_generated"
)
LOG_DIR = (
    ROOT
    / "data"
    / "modal_logs"
    / "frame_temporal_bottleneck_full112_lowfreq_motion_v4_main_b200_3000_seed231_all5"
)

CHECKPOINT_STEPS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]


def configure_pilot_module() -> None:
    pilot.VARIANTS = {
        "main": {
            "run_name": RUN_NAME,
            "diffusion_loss_weight": 0.25,
            "lowfreq_target_loss_weight": 1.0,
            "lowfreq_delta_loss_weight": 1.0,
            "hf_teacher_loss_weight": 0.20,
            "action_motion_aux_loss_weight": 0.05,
            "action_residual_loss_weight": 0.002,
            "action_gate_loss_weight": 0.002,
            "action_learning_rate": 3e-5,
            "action_injector_learning_rate": 3e-5,
            "action_gate_learning_rate": 3e-5,
        }
    }
    pilot.CHECKPOINT_STEPS = CHECKPOINT_STEPS
    pilot.COUNTERFACTUAL_STEPS = CHECKPOINT_STEPS
    pilot.COUNTERFACTUAL_MODES = COUNTERFACTUAL_MODES
    pilot.BENCHMARK_DIR = BENCHMARK_DIR
    pilot.LOCAL_GENERATED_ROOT = LOCAL_GENERATED_ROOT
    pilot.LOG_DIR = LOG_DIR


def write_run_note(phase: str) -> None:
    pilot.write_json(
        BENCHMARK_DIR / "v4_b200_3000_eval_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "run_name": RUN_NAME,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "counterfactual_modes": COUNTERFACTUAL_MODES,
            "benchmark_dir": str(BENCHMARK_DIR),
            "generated_dir": str(LOCAL_GENERATED_ROOT),
            "notes": [
                "Evaluation-only runner for the completed B200 V4-main 3000-step checkpoint run.",
                "Uses corrected 24 FPS upsampled frame-action conditions.",
                "Does not recache latents or mutate checkpoints.",
                "Inference uses image_cond_noise_scale=0.0 through the V4 wrapper default.",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B200 V4-main 3000-step checkpoints.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "sweep", "counterfactual", "plots", "all"),
        default="all",
    )
    parser.add_argument("--max-generation-workers", type=int, default=8)
    parser.add_argument("--metric-chunk-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=pilot.SEED)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    configure_pilot_module()
    if args.phase in {"syntax", "all"}:
        pilot.syntax_check()
    if args.phase in {"sweep", "all"}:
        pilot.run_checkpoint_sweep(
            args.max_generation_workers,
            args.seed,
            args.limit,
            args.metric_chunk_size,
        )
    if args.phase in {"counterfactual", "all"}:
        pilot.run_counterfactual(
            args.max_generation_workers,
            args.seed,
            args.limit,
            args.metric_chunk_size,
        )
    if args.phase in {"plots"}:
        pilot.make_plots()
    write_run_note(args.phase)
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
