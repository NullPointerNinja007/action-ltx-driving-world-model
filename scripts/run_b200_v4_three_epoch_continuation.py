from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
PYTHON = ROOT / ".venv" / "bin" / "python"
if not MODAL.exists():
    MODAL = Path("modal")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

TRAIN_WRAPPER = "scripts/wrappers/train_ltx2b_distilled_waymo_frame_temporal_bottleneck_fullaction_motion_v4_action_lora.py"
COPY_WRAPPER = "scripts/copy_modal_volume_within.py"

CHECKPOINT_VOLUME = "ltx2b-v4-b200-rank-capacity-ckpts"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "b200_v4_three_epoch_continuation_seed231"
LOG_DIR = ROOT / "data" / "modal_logs" / "b200_v4_three_epoch_continuation_seed231"
STOP_RELPATH = "control/stop_r128_after_r32_r64_done.json"

R32_SOURCE_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_from_shifted_noaction_step003000_steps7992"
R64_SOURCE_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r64_b200_seed231_from_shifted_noaction_step003000_steps7992"

R32_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_resume007992_to023976"
R64_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r64_b200_seed231_resume007992_to023976"
R128_RUN = "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r128_b200_seed231_from_shifted_noaction_step003000_trainuntil_r32r64done"

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
    15984,
    17000,
    18000,
    19000,
    20000,
    21000,
    22000,
    23000,
    23976,
    26000,
    28000,
    30000,
    31968,
]


@dataclass(frozen=True)
class TrainSpec:
    key: str
    run_name: str
    lora_rank: int
    max_steps: int
    checkpoint_steps: list[int]
    resume_from_checkpoint: str = ""
    resume_from_run_name: str = ""
    freeze_lora: bool = False
    expand_baseline_lora_to_rank: bool = True
    max_train_hours: float = 8.0
    external_stop_relpath: str = ""
    external_stop_check_steps: int = 100


SPECS = [
    TrainSpec(
        key="r32_continue",
        run_name=R32_RUN,
        lora_rank=32,
        max_steps=23976,
        checkpoint_steps=R32_R64_STEPS,
        resume_from_checkpoint="step_007992",
        resume_from_run_name=R32_RUN,
        max_train_hours=8.0,
    ),
    TrainSpec(
        key="r64_continue",
        run_name=R64_RUN,
        lora_rank=64,
        max_steps=23976,
        checkpoint_steps=R32_R64_STEPS,
        resume_from_checkpoint="step_007992",
        resume_from_run_name=R64_RUN,
        max_train_hours=8.0,
    ),
    TrainSpec(
        key="r128_train_until_done",
        run_name=R128_RUN,
        lora_rank=128,
        max_steps=31968,
        checkpoint_steps=R128_STEPS,
        max_train_hours=8.0,
        external_stop_relpath=STOP_RELPATH,
        external_stop_check_steps=100,
    ),
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None, retries: int = 1) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    for attempt in range(1, retries + 2):
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write(f"attempt={attempt} started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=merged_env)
        if proc.returncode == 0:
            return
        if attempt <= retries:
            time.sleep(120)
    raise RuntimeError(f"Command failed after {retries + 1} attempts; see {log_path}")


def train_env(spec: TrainSpec) -> dict[str, str]:
    return {
        "LTX_MODAL_TRAIN_GPU": "B200",
        "LTX_CHECKPOINT_VOLUME_NAME": CHECKPOINT_VOLUME,
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "0",
        "WAYMO24_FREEZE_TRANSFORMER_LORA": "1" if spec.freeze_lora else "0",
        "WAYMO24_EXPAND_BASELINE_LORA_TO_RANK": "1" if spec.expand_baseline_lora_to_rank else "0",
    }


def train_command(spec: TrainSpec) -> list[str]:
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
        "5e-6",
        "--lora-rank",
        str(spec.lora_rank),
        "--lora-alpha",
        str(spec.lora_rank),
        "--action-learning-rate",
        "3e-5",
        "--action-injector-learning-rate",
        "3e-5",
        "--action-gate-learning-rate",
        "3e-5",
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
        "0.25",
        "--lowfreq-target-loss-weight",
        "1.0",
        "--lowfreq-delta-loss-weight",
        "1.0",
        "--hf-teacher-loss-weight",
        "0.20",
        "--action-motion-aux-loss-weight",
        "0.05",
        "--action-residual-loss-weight",
        "0.002",
        "--action-gate-loss-weight",
        "0.002",
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
        cmd.extend(["--resume-from-run-name", spec.resume_from_run_name])
    if spec.external_stop_relpath:
        cmd.extend(["--external-stop-relpath", spec.external_stop_relpath])
        cmd.extend(["--external-stop-check-steps", str(spec.external_stop_check_steps)])
    return cmd


def stage_resume_checkpoint(source_run: str, dest_run: str, *, force: bool = False) -> None:
    log_path = LOG_DIR / "preflight" / f"stage_{dest_run}_step007992.log"
    if log_path.exists() and "App completed" in log_path.read_text(encoding="utf-8", errors="ignore") and not force:
        return
    if force:
        run_command(
            [str(MODAL), "volume", "rm", "-r", CHECKPOINT_VOLUME, f"{dest_run}/step_007992"],
            LOG_DIR / "preflight" / f"remove_existing_{dest_run}_step007992.log",
            retries=0,
        )
    cmd = [
        str(MODAL),
        "run",
        COPY_WRAPPER,
        "--src-relpath",
        f"{source_run}/step_007992",
        "--dst-relpath",
        f"{dest_run}/step_007992",
    ]
    run_command(cmd, log_path, env={"COPY_MODAL_VOLUME_NAME": CHECKPOINT_VOLUME}, retries=1)


def clear_stop_sentinel() -> None:
    log_path = LOG_DIR / "preflight" / "clear_stop_sentinel.log"
    cmd = [str(MODAL), "volume", "rm", CHECKPOINT_VOLUME, STOP_RELPATH]
    try:
        run_command(cmd, log_path, retries=0)
    except RuntimeError:
        # Missing sentinel is fine; `modal volume rm` exits nonzero in that case.
        pass


def write_stop_sentinel(reason: str, completed: list[str], failed: list[str]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "completed": completed,
        "failed": failed,
    }
    local = BENCHMARK_DIR / "control" / "stop_r128_after_r32_r64_done.json"
    write_json(local, payload)
    run_command(
        [str(MODAL), "volume", "put", "--force", CHECKPOINT_VOLUME, str(local), STOP_RELPATH],
        LOG_DIR / "control" / "write_stop_sentinel.log",
        retries=2,
    )


def train_one(spec: TrainSpec) -> str:
    log_path = LOG_DIR / "training" / f"{spec.key}_to{spec.max_steps}.log"
    run_command(train_command(spec), log_path, env=train_env(spec), retries=1)
    return spec.key


def run_campaign(force_stage: bool = False) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        BENCHMARK_DIR / "campaign_config.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_volume": CHECKPOINT_VOLUME,
            "stop_relpath": STOP_RELPATH,
            "specs": [asdict(spec) for spec in SPECS],
        },
    )
    clear_stop_sentinel()
    stage_resume_checkpoint(R32_SOURCE_RUN, R32_RUN, force=force_stage)
    stage_resume_checkpoint(R64_SOURCE_RUN, R64_RUN, force=force_stage)

    start_time = time.monotonic()
    completed: list[str] = []
    failed: list[str] = []
    errors: dict[str, str] = {}
    r32_r64 = [SPECS[0], SPECS[1]]
    r128 = SPECS[2]
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_by_key = {executor.submit(train_one, spec): spec.key for spec in [*r32_r64, r128]}
        pending_control = {key for key in ["r32_continue", "r64_continue"]}
        for future in as_completed([f for f, key in future_by_key.items() if key in pending_control]):
            key = future_by_key[future]
            try:
                future.result()
                completed.append(key)
            except Exception as exc:  # noqa: BLE001
                failed.append(key)
                errors[key] = repr(exc)
            pending_control.discard(key)
        reason = "r32_r64_completed" if not failed else "r32_or_r64_failed"
        write_stop_sentinel(reason, completed, failed)
        r128_future = [f for f, key in future_by_key.items() if key == "r128_train_until_done"][0]
        try:
            r128_future.result()
            completed.append("r128_train_until_done")
        except Exception as exc:  # noqa: BLE001
            failed.append("r128_train_until_done")
            errors["r128_train_until_done"] = repr(exc)

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": (time.monotonic() - start_time) / 3600.0,
        "checkpoint_volume": CHECKPOINT_VOLUME,
        "stop_relpath": STOP_RELPATH,
        "completed": completed,
        "failed": failed,
        "errors": errors,
    }
    write_json(BENCHMARK_DIR / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final B200 V4 three-epoch continuation campaign.")
    parser.add_argument("--force-stage", action="store_true", help="Re-copy r32/r64 step_007992 into continuation run dirs.")
    args = parser.parse_args()
    run_campaign(force_stage=args.force_stage)


if __name__ == "__main__":
    main()
