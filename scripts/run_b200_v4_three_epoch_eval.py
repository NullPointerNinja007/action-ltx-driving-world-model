from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_b200_v4_rank_capacity_campaign as b200


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_VOLUME = "ltx2b-v4-b200-rank-capacity-ckpts"
ARTIFACT_VOLUME = "ltx2b-v4-b200-three-epoch-continuation-infer"
RUNS_ROOT = "distilled098_full112_lowfreq_motion_v4_b200_three_epoch_24fps_minterpolate_seed231_runs"

BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "b200_v4_three_epoch_continuation_seed231_eval"
GENERATED_ROOT = ROOT / "data" / "b200_v4_three_epoch_continuation_generated"
LOG_DIR = ROOT / "data" / "modal_logs" / "b200_v4_three_epoch_continuation_eval_seed231"

R32_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_resume007992_to023976"
R64_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r64_b200_seed231_resume007992_to023976"
R128_RUN = (
    "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r128_b200_seed231_"
    "from_shifted_noaction_step003000_trainuntil_r32r64done"
)

R32_R64_STEPS = [7992, 10000, 12000, 14000, 15984, 18000, 20000, 22000, 23976]
R128_STEPS = [
    0,
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
    6000,
    7000,
    7992,
    9000,
    10000,
    11000,
    12000,
    13000,
    14000,
    15000,
]
COUNTERFACTUAL_MODES = ["correct", "zero", "shuffled", "reversed_future"]
COUNTERFACTUAL_STEPS_BY_KEY = {
    # Include both final checkpoints and intermediate quality peaks so we can
    # select the best controllability/fidelity operating point, not just the
    # last checkpoint.
    "v4_main_text_r32_3epoch": [7992, 10000, 15984, 18000, 20000, 22000, 23976],
    "v4_main_text_r64_3epoch": [7992, 10000, 15984, 18000, 22000, 23976],
    "v4_main_text_r128_partial": [6000, 7992, 10000, 12000, 15000],
}


SPECS = [
    b200.B200Spec(
        key="v4_main_text_r32_3epoch",
        label="V4 main text rank32 continued to 3 epochs",
        run_name=R32_RUN,
        lora_rank=32,
        max_steps=23976,
        checkpoint_steps=R32_R64_STEPS,
        freeze_lora=False,
        expand_baseline_lora_to_rank=True,
    ),
    b200.B200Spec(
        key="v4_main_text_r64_3epoch",
        label="V4 main text rank64 continued to 3 epochs",
        run_name=R64_RUN,
        lora_rank=64,
        max_steps=23976,
        checkpoint_steps=R32_R64_STEPS,
        freeze_lora=False,
        expand_baseline_lora_to_rank=True,
    ),
    b200.B200Spec(
        key="v4_main_text_r128_partial",
        label="V4 main text rank128 trained until r32/r64 completed",
        run_name=R128_RUN,
        lora_rank=128,
        max_steps=15000,
        checkpoint_steps=R128_STEPS,
        freeze_lora=False,
        expand_baseline_lora_to_rank=True,
    ),
]


def configure_modules() -> None:
    b200.CHECKPOINT_VOLUME = CHECKPOINT_VOLUME
    b200.ARTIFACT_VOLUME = ARTIFACT_VOLUME
    b200.RUNS_ROOT = RUNS_ROOT
    b200.BENCHMARK_DIR = BENCHMARK_DIR
    b200.GENERATED_ROOT = GENERATED_ROOT
    b200.LOG_DIR = LOG_DIR
    b200.COUNTERFACTUAL_STEPS_BY_KEY.clear()
    b200.COUNTERFACTUAL_STEPS_BY_KEY.update(COUNTERFACTUAL_STEPS_BY_KEY)
    b200.COUNTERFACTUAL_MODES[:] = COUNTERFACTUAL_MODES
    b200.configure_campaign_module()


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


def merge_with_previous_baselines() -> None:
    rows: list[dict[str, Any]] = []
    sources = {
        "three_epoch": BENCHMARK_DIR / "checkpoint_metrics_by_model.csv",
        "b200_rank_capacity": ROOT / "data" / "benchmarks" / "b200_v4_rank_capacity_seed231" / "model_summary_with_fvd.csv",
        "final_h100_v4": ROOT / "data" / "benchmarks" / "final_h100_v4_campaign_seed231_all5" / "model_summary_with_fvd.csv",
        "noaction_shifted": ROOT
        / "data"
        / "benchmarks"
        / "noaction_cleanctx_shifted_timestep_ablation_seed231_all5"
        / "noaction_cleanctx_shifted_timestep_summary_with_fvd.csv",
    }
    for source_name, path in sources.items():
        for row in read_csv(path):
            row = dict(row)
            row["source_table"] = source_name
            rows.append(row)
    write_csv(BENCHMARK_DIR / "comparison_pool_with_previous_methods.csv", rows)


def write_run_summary(phase: str) -> None:
    write_json(
        BENCHMARK_DIR / "eval_campaign_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "artifact_volume": ARTIFACT_VOLUME,
            "runs_root": RUNS_ROOT,
            "benchmark_dir": str(BENCHMARK_DIR),
            "generated_root": str(GENERATED_ROOT),
            "checkpoint_protocol": {
                "r32_r64_steps": R32_R64_STEPS,
                "r128_steps": R128_STEPS,
                "counterfactual_steps_by_key": COUNTERFACTUAL_STEPS_BY_KEY,
                "counterfactual_modes": COUNTERFACTUAL_MODES,
                "limit": 5,
                "seed": 231,
                "image_cond_noise_scale": 0.0,
            },
            "specs": [spec.__dict__ for spec in SPECS],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B200 V4 three-epoch continuation checkpoints.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "sweep", "counterfactual", "loss", "plots", "all"),
        default="all",
    )
    parser.add_argument("--max-generation-workers", type=int, default=10)
    parser.add_argument("--force-generate", action="store_true")
    args = parser.parse_args()

    configure_modules()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in {"syntax", "all"}:
        b200.syntax_check()
    if args.phase in {"sweep", "all"}:
        b200.run_checkpoint_sweep(SPECS, max_generation_workers=args.max_generation_workers, force_generate=args.force_generate)
    if args.phase in {"counterfactual", "all"}:
        b200.run_counterfactual(SPECS, max_generation_workers=args.max_generation_workers, force_generate=args.force_generate)
    if args.phase in {"loss", "all"}:
        b200.extract_loss_curves(SPECS)
    if args.phase in {"plots", "all"}:
        b200.make_plots(BENCHMARK_DIR)
        b200.copy_required_plot_names()
        b200.write_analysis(SPECS)
        merge_with_previous_baselines()
    write_run_summary(args.phase)
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
