from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
LOG_DIR = ROOT / "data" / "modal_logs" / "frame_midblock_gated_xattn_shifted_seed231"

TRAIN_WRAPPER = "train_ltx2b_distilled_waymo_frame_midblock_gated_xattn_action_lora.py"
INFER_WRAPPER = "generate_waymo24_distilled_frame_midblock_gated_xattn_action_minterpolate_lora.py"

SMOKE_RUN_NAME = "ltx2b_dist098_waymo24_frame_midblock_gated_xattn_seed231_from_shifted_noaction_step003000_smoke500"
MAIN_RUN_NAME = "ltx2b_dist098_waymo24_frame_midblock_gated_xattn_seed231_from_shifted_noaction_step003000_steps3000"

CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts"
ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-midxattn-r16-shift-infer"


def run_command(cmd: list[str], log_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")
    print(json.dumps({"completed": log_name, "log": str(log_path)}, sort_keys=True))


def train(run_name: str, max_steps: int, checkpoint_steps: str, max_train_hours: float, train_limit: int) -> None:
    cmd = [
        str(MODAL),
        "run",
        TRAIN_WRAPPER,
        "--run-name",
        run_name,
        "--max-steps",
        str(max_steps),
        "--max-train-hours",
        str(max_train_hours),
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        "1e-4",
        "--action-injector-learning-rate",
        "1e-4",
        "--action-gate-learning-rate",
        "1e-3",
        "--action-injector-heads",
        "8",
        "--lora-rank",
        "16",
        "--timestep-sampling",
        "shifted_lognormal",
        "--checkpoint-steps",
        checkpoint_steps,
        "--seed",
        "231",
        "--train-limit",
        str(train_limit),
        "--val-limit",
        "32",
    ]
    run_command(cmd, f"train_{run_name}.log")


def generate(run_name: str, checkpoint: str, limit: int) -> None:
    label = f"framemidxattn_{checkpoint.replace('step_', 'step')}_seed231_all5_cleanctx"
    cmd = [
        str(MODAL),
        "run",
        INFER_WRAPPER,
        "--limit",
        str(limit),
        "--seed",
        "231",
        "--lora-step",
        checkpoint,
        "--lora-run-name",
        run_name,
        "--run-label",
        label,
        "--base-label",
        "base_distilled_no_lora",
    ]
    run_command(cmd, f"generate_{label}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrected middle-block gated action xattn experiment.")
    parser.add_argument("--phase", choices=("tiny", "smoke", "main", "generate"), required=True)
    parser.add_argument("--checkpoint", default="step_003000", help="Checkpoint for --phase generate.")
    parser.add_argument("--run-name", default=MAIN_RUN_NAME, help="Run name for --phase generate.")
    args = parser.parse_args()

    if args.phase == "tiny":
        train(
            run_name=f"{SMOKE_RUN_NAME}_tiny2",
            max_steps=2,
            checkpoint_steps="0,1,2",
            max_train_hours=1.0,
            train_limit=4,
        )
    elif args.phase == "smoke":
        train(
            run_name=SMOKE_RUN_NAME,
            max_steps=500,
            checkpoint_steps="0,100,250,500",
            max_train_hours=2.0,
            train_limit=0,
        )
    elif args.phase == "main":
        train(
            run_name=MAIN_RUN_NAME,
            max_steps=3000,
            checkpoint_steps="0,100,250,500,1000,1500,2000,2500,3000",
            max_train_hours=8.0,
            train_limit=0,
        )
    elif args.phase == "generate":
        generate(args.run_name, args.checkpoint, limit=5)


if __name__ == "__main__":
    main()
