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
LOCAL_GENERATED_ROOT = ROOT / "data" / "action_only_no_text_checkpoint_sweep_generated"
BENCHMARK_DIR = ROOT / "data" / "benchmarks" / "action_only_no_text_ablation_seed231_all5"
LOG_DIR = ROOT / "data" / "modal_logs" / "action_only_no_text_ablation_seed231_all5"

CHECKPOINT_STEPS = "0,100,250,500,1000,1500,2000,2500,3000"
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

SHIFTED_NOACTION_VOLUME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts"
SHIFTED_NOACTION_RUN = "ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000"
SHIFTED_NOACTION_STEP = "step_003000"

TWO_EPOCH_NOACTION_VOLUME = "ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-ckpts"
TWO_EPOCH_NOACTION_RUN = "ltx2b_distilled098_waymo24_noaction_visual_lora_r16_seed231_full7992_lr5e6_2epochs_steps15984"
TWO_EPOCH_NOACTION_STEP = "step_010000"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    train_wrapper: str
    infer_wrapper: str
    checkpoint_volume: str
    artifact_volume: str
    runs_root: str
    run_name: str
    action_lr: str
    injector_lr: str = ""
    gate_lr: str = ""
    train_env: dict[str, str] | None = None
    extra_train_args: tuple[str, ...] = ()
    eval_gate_scale: float = 1.0


METHODS = [
    Method(
        key="frame_global_mlp",
        label="Frame Global MLP Tokens, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_global_mlp_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_global_mlp_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-frameglobal-action-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-frameglobal-action-infer",
        runs_root="distilled098_notext_frameglobal_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_global_mlp_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_steps3000"
        ),
        action_lr="1e-4",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": TWO_EPOCH_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": TWO_EPOCH_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": TWO_EPOCH_NOACTION_STEP,
        },
    ),
    Method(
        key="frame_temporal_pool",
        label="Frame Temporal Pool Tokens, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_temporal_pool_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_pool_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-framepool-action-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-framepool-action-infer",
        runs_root="distilled098_notext_framepool_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_temporal_pool_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_steps3000"
        ),
        action_lr="1e-4",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": TWO_EPOCH_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": TWO_EPOCH_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": TWO_EPOCH_NOACTION_STEP,
        },
    ),
    Method(
        key="frame_transformer",
        label="Frame Transformer Tokens, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_transformer_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_transformer_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-framexf-action-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-framexf-action-infer",
        runs_root="distilled098_notext_framexf_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_transformer_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr1e4_steps3000"
        ),
        action_lr="1e-4",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": TWO_EPOCH_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": TWO_EPOCH_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": TWO_EPOCH_NOACTION_STEP,
        },
    ),
    Method(
        key="frame_adaln",
        label="Frame AdaLN, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_adaln_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_adaln_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-frameadaln-action-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-frameadaln-action-infer",
        runs_root="distilled098_notext_frameadaln_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_adaln_action_lora_r16_seed231_"
            "from_noaction_step010000_lr5e6_actionlr5e5_steps3000"
        ),
        action_lr="5e-5",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": TWO_EPOCH_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": TWO_EPOCH_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": TWO_EPOCH_NOACTION_STEP,
        },
    ),
    Method(
        key="frame_midblock_gated_xattn",
        label="Middle-Block Gated XAttn, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_midblock_gated_xattn_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_midblock_gated_xattn_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-midxattn-action-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-midxattn-action-infer",
        runs_root="distilled098_notext_midxattn_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_midblock_gated_xattn_seed231_"
            "from_shifted_noaction_step003000_steps3000"
        ),
        action_lr="1e-4",
        injector_lr="1e-4",
        gate_lr="1e-3",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": SHIFTED_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": SHIFTED_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": SHIFTED_NOACTION_STEP,
            "WAYMO24_FREEZE_TRANSFORMER_LORA": "1",
        },
        extra_train_args=("--action-injector-heads", "8"),
    ),
    Method(
        key="frame_temporal_bottleneck_lowfreq_v3",
        label="Low-Frequency Temporal Bottleneck V3, No Text",
        train_wrapper="scripts/wrappers/train_ltx2b_distilled_waymo_frame_temporal_bottleneck_lowfreq_v3_action_lora.py",
        infer_wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_lowfreq_v3_action_minterpolate_lora.py",
        checkpoint_volume="ltx2b-dist098-waymo24-notext-framebneck-lowfreq-v3-r16-ckpts",
        artifact_volume="ltx2b-dist098-waymo24-notext-framebneck-lowfreq-v3-infer",
        runs_root="distilled098_notext_framebneck_lowfreq_v3_action_lora_24fps_minterpolate_seed231_runs",
        run_name=(
            "ltx2b_dist098_waymo24_notext_frame_temporal_bottleneck_lowfreq_v3_seed231_"
            "from_shifted_noaction_step003000_steps3000"
        ),
        action_lr="3e-5",
        injector_lr="3e-5",
        gate_lr="1e-4",
        train_env={
            "LTX_BASELINE_CHECKPOINT_VOLUME_NAME": SHIFTED_NOACTION_VOLUME,
            "WAYMO24_BASELINE_LORA_RUN_NAME": SHIFTED_NOACTION_RUN,
            "WAYMO24_BASELINE_LORA_STEP": SHIFTED_NOACTION_STEP,
            "WAYMO24_FREEZE_TRANSFORMER_LORA": "1",
        },
        extra_train_args=(
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
        ),
        eval_gate_scale=0.05,
    ),
]


def step_to_int(step: str) -> int:
    if step == "step_000000_base_reference":
        return 0
    match = re.search(r"step_0*([0-9]+)", step)
    if not match:
        raise ValueError(f"Could not parse step from {step!r}")
    return int(match.group(1))


def step_label(step: str) -> str:
    if step == "step_000000_base_reference":
        return "step000000"
    return step.replace("step_", "step")


def run_label(method: Method, step: str, seed: int) -> str:
    gate = f"g{method.eval_gate_scale:.3f}".replace(".", "p")
    return f"{method.key}_notext_{step_label(step)}_{gate}_seed{seed}_all5"


def modal_env(method: Method) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LTX_CHECKPOINT_VOLUME_NAME": method.checkpoint_volume,
            "LTX_ARTIFACTS_VOLUME_NAME": method.artifact_volume,
            "LTX_RUNS_ROOT": method.runs_root,
            "WAYMO24_DISABLE_TEXT_CONDITIONING": "1",
            "LTX_MODAL_GPU": "A100",
            "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
        }
    )
    if method.train_env:
        env.update(method.train_env)
    return env


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if log_path.exists() else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if mode == "a":
            log.write("\n\n" + "=" * 80 + "\n")
        log.write("$ " + " ".join(cmd) + "\n\n")
        if env:
            log.write(
                "$ env "
                + " ".join(
                    f"{key}={env[key]}"
                    for key in sorted(env)
                    if key.startswith(("LTX_", "WAYMO24_"))
                )
                + "\n\n"
            )
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


def log_completed(path: Path) -> bool:
    return path.exists() and "App completed" in path.read_text(encoding="utf-8", errors="ignore")


def train_one(
    method: Method,
    *,
    force: bool = False,
    detach: bool = False,
    resume_method: str = "",
    resume_checkpoint: str = "",
    resume_run_name: str = "",
) -> str:
    log_path = LOG_DIR / "training" / f"{method.run_name}.log"
    if not force and log_completed(log_path):
        return method.key
    cmd = [str(MODAL), "run"]
    if detach:
        cmd.append("--detach")
    cmd.extend(
        [
            method.train_wrapper,
        "--run-name",
        method.run_name,
        "--max-steps",
        "3000",
        "--max-train-hours",
        "8.0",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--action-learning-rate",
        method.action_lr,
        "--weight-decay",
        "0.0",
        "--checkpoint-steps",
        CHECKPOINT_STEPS,
        "--timestep-sampling",
        "shifted_lognormal",
        "--timestep-lognormal-mean",
        "0.0",
        "--timestep-lognormal-std",
        "1.0",
        "--lora-rank",
        "16",
        "--seed",
        "231",
        "--num-val-samples",
        "0",
        "--train-limit",
        "0",
        "--val-limit",
        "32",
        ]
    )
    if method.injector_lr:
        cmd.extend(["--action-injector-learning-rate", method.injector_lr])
    if method.gate_lr:
        cmd.extend(["--action-gate-learning-rate", method.gate_lr])
    cmd.extend(method.extra_train_args)
    if resume_checkpoint and method.key == resume_method:
        cmd.extend(["--resume-from-checkpoint", resume_checkpoint])
        cmd.extend(["--resume-from-run-name", resume_run_name or method.run_name])
    run_command(cmd, log_path, env=modal_env(method))
    return method.key


def train_all(
    max_workers: int,
    *,
    force: bool = False,
    methods: list[Method] | None = None,
    detach: bool = False,
    resume_method: str = "",
    resume_checkpoint: str = "",
    resume_run_name: str = "",
) -> None:
    selected_methods = methods or METHODS
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                train_one,
                method,
                force=force,
                detach=detach,
                resume_method=resume_method,
                resume_checkpoint=resume_checkpoint,
                resume_run_name=resume_run_name,
            )
            for method in selected_methods
        ]
        for future in as_completed(futures):
            print(json.dumps({"trained": future.result()}, sort_keys=True), flush=True)


def generate_one(method: Method, step: str, seed: int, limit: int) -> tuple[str, str]:
    label = run_label(method, step, seed)
    log_path = LOG_DIR / "generation" / f"{label}.log"
    if log_completed(log_path):
        return method.key, step
    cmd = [
        str(MODAL),
        "run",
        method.infer_wrapper,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--lora-step",
        step,
        "--lora-run-name",
        method.run_name,
        "--run-label",
        label,
        "--base-label",
        "base_distilled_no_lora",
        "--action-gate-scale",
        str(method.eval_gate_scale),
    ]
    run_command(cmd, log_path, env=modal_env(method))
    return method.key, step


def generate_all(max_workers: int, seed: int, limit: int, *, methods: list[Method] | None = None) -> None:
    selected_methods = methods or METHODS
    jobs = [(method, step) for step in CHECKPOINTS for method in selected_methods]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, method, step, seed, limit) for method, step in jobs]
        for future in as_completed(futures):
            method_key, step = future.result()
            print(json.dumps({"generated": method_key, "step": step}, sort_keys=True), flush=True)


def download_one(method: Method, step: str, seed: int) -> Path:
    label = run_label(method, step, seed)
    remote_path = f"{method.runs_root}/{label}"
    local_dest = LOCAL_GENERATED_ROOT / method.runs_root / label
    local_dest.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "download" / f"{label}.log"

    summary_local = local_dest / "run_summary.json"
    if not summary_local.exists():
        run_command(
            [
                str(MODAL),
                "volume",
                "get",
                "--force",
                method.artifact_volume,
                f"{remote_path}/run_summary.json",
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


def download_all(seed: int, *, methods: list[Method] | None = None) -> None:
    selected_methods = methods or METHODS
    for method in selected_methods:
        for step in CHECKPOINTS:
            path = download_one(method, step, seed)
            print(json.dumps({"downloaded": str(path.relative_to(ROOT))}, sort_keys=True), flush=True)


def local_file_for_record(record: dict[str, Any], method: Method, step: str, seed: int) -> Path:
    label = run_label(method, step, seed)
    prefix = f"{method.runs_root}/{label}/"
    generated_relpath = record["generated_video_relpath"]
    if not generated_relpath.startswith(prefix):
        raise ValueError(f"Unexpected generated path {generated_relpath}; expected prefix {prefix}")
    return LOCAL_GENERATED_ROOT / method.runs_root / label / generated_relpath[len(prefix) :]


def build_manifest(seed: int, limit: int, *, methods: list[Method] | None = None) -> Path:
    selected_methods = methods or METHODS
    records: list[dict[str, Any]] = []
    for method in selected_methods:
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
                row.update(
                    {
                        "local_file": str(local_file),
                        "method_key": method.key,
                        "method_label": method.label,
                        "checkpoint_step": step_to_int(step),
                        "checkpoint_name": step,
                        "model_mode": f"{method.key}_notext_{step_label(step)}",
                        "using_lora": True,
                        "disable_text_conditioning": True,
                        "eval_action_gate_scale": method.eval_gate_scale,
                    }
                )
                records.append(row)

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BENCHMARK_DIR / "manifest_action_only_no_text_ablation_seed231_all5.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "description": (
                    "Action-only/no-text checkpoint sweep. T5 prompt embeddings are zeroed and masked; "
                    "token action methods expose only action tokens to cross-attention."
                ),
                "seed": seed,
                "limit": limit,
                "context_frames": 49,
                "future_frames": 72,
                "total_frames": 121,
                "fps": 24,
                "methods": [method.__dict__ for method in selected_methods],
                "checkpoints": CHECKPOINTS,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run_quality_metrics(manifest_path: Path) -> None:
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
            str(BENCHMARK_DIR),
            "--run-id",
            "action_only_no_text_ablation_seed231_all5_quality",
            "--chunk-size",
            "8",
        ],
        LOG_DIR / "benchmark_video_quality_modal.log",
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
            "action_only_no_text_ablation_seed231_all5",
        ],
        LOG_DIR / "compute_fvd.log",
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    raw = row.get(key, "")
    if raw in {"", None}:
        return default
    return float(raw)


def merge_summaries() -> list[dict[str, Any]]:
    summary_rows = read_csv(BENCHMARK_DIR / "model_summary.csv")
    fvd_by_model = {row["model_mode"]: row for row in read_csv(BENCHMARK_DIR / "fvd_summary.csv")}
    manifest = json.loads((BENCHMARK_DIR / "manifest_action_only_no_text_ablation_seed231_all5.json").read_text())
    meta_by_model: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        meta_by_model[record["model_mode"]] = {
            "method_key": record["method_key"],
            "method_label": record["method_label"],
            "checkpoint_step": record["checkpoint_step"],
            "checkpoint_name": record["checkpoint_name"],
            "disable_text_conditioning": True,
            "eval_action_gate_scale": record.get("eval_action_gate_scale", 1.0),
        }
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        out = {**meta_by_model.get(row["model_mode"], {}), **row}
        fvd = fvd_by_model.get(row["model_mode"])
        if fvd:
            out.update(
                {
                    "fvd_future": fvd.get("fvd_future", ""),
                    "fvd_backend": fvd.get("fvd_backend", ""),
                    "fvd_num_videos": fvd.get("fvd_num_videos", ""),
                    "fvd_num_frames": fvd.get("fvd_num_frames", ""),
                    "fvd_size": fvd.get("fvd_size", ""),
                }
            )
        rows.append(out)
    rows.sort(key=lambda row: (str(row.get("method_key", "")), int(row.get("checkpoint_step", 0))))
    write_csv(BENCHMARK_DIR / "action_only_no_text_summary_with_fvd.csv", rows)
    return rows


def load_text_conditioned_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    frame_csv = ROOT / "data/benchmarks/frame_action_checkpoint_sweep_seed231_all5/frame_action_checkpoint_sweep_summary_with_fvd.csv"
    for row in read_csv(frame_csv):
        method_key = row.get("method_key", "")
        if method_key in {"frame_global_mlp", "frame_temporal_pool", "frame_transformer", "frame_adaln"}:
            rows.append({**row, "conditioning_variant": "text_plus_action"})

    midblock_csv = ROOT / "data/benchmarks/midblock_gated_xattn_checkpoint_sweep_seed231_all5/midblock_gated_xattn_summary_with_fvd.csv"
    for row in read_csv(midblock_csv):
        if row.get("method_key", "") == "frame_midblock_gated_xattn" or row.get("model_mode", "").startswith("frame_midblock_gated_xattn"):
            row = dict(row)
            row["method_key"] = "frame_midblock_gated_xattn"
            row["method_label"] = "Middle-Block Gated XAttn, Text+Action"
            row["conditioning_variant"] = "text_plus_action"
            rows.append(row)

    v3_csv = ROOT / "data/benchmarks/frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_seed231_all5/gate_scale_summary.csv"
    for row in read_csv(v3_csv):
        if row.get("diagnostic_group") == "gate_scale_sweep" and abs(as_float(row, "action_gate_scale", 999.0) - 0.05) < 1e-6:
            row = dict(row)
            row["method_key"] = "frame_temporal_bottleneck_lowfreq_v3"
            row["method_label"] = "Low-Frequency Temporal Bottleneck V3, Text+Action"
            row["conditioning_variant"] = "text_plus_action"
            rows.append(row)
    return rows


def comparison_rows(action_only_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [{**row, "conditioning_variant": "action_only_no_text"} for row in action_only_rows]
    out.extend(load_text_conditioned_rows())
    out.sort(key=lambda row: (str(row.get("method_key", "")), str(row.get("conditioning_variant", "")), int(float(row.get("checkpoint_step", 0)))))
    write_csv(BENCHMARK_DIR / "text_vs_action_only_comparison_rows.csv", out)
    return out


def plot_comparison(rows: list[dict[str, Any]]) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_dir = BENCHMARK_DIR / "metric_plots"
    metric_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_future_psnr", "Future PSNR", "higher"),
        ("mean_future_global_ssim", "Future SSIM", "higher"),
        ("fvd_future", "FVD-style Future Distance", "lower"),
        ("mean_sharpness_ratio_generated_over_reference", "Sharpness Ratio", "higher"),
        ("mean_fft_high_frequency_energy_ratio_generated_over_reference", "FFT High-Frequency Ratio", "higher"),
        ("mean_motion_ratio_generated_over_reference", "Motion Ratio", "higher"),
        ("mean_low_frequency_motion_ratio_generated_over_reference", "Low-Frequency Motion Ratio", "higher"),
        ("mean_temporal_delta_error_mae", "Temporal Delta Error", "lower"),
    ]
    method_keys = [method.key for method in METHODS]
    variants = [("text_plus_action", "Text + action"), ("action_only_no_text", "Action only")]
    colors = {"text_plus_action": "tab:blue", "action_only_no_text": "tab:orange"}

    for metric, title, direction in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
        axes_flat = axes.flatten()
        for ax, method_key in zip(axes_flat, method_keys):
            selected = [row for row in rows if row.get("method_key") == method_key]
            for variant, label in variants:
                vrows = [row for row in selected if row.get("conditioning_variant") == variant]
                vrows.sort(key=lambda row: int(float(row.get("checkpoint_step", 0))))
                x = [int(float(row.get("checkpoint_step", 0))) for row in vrows]
                y = [as_float(row, metric) for row in vrows]
                if not x or all(math.isnan(value) for value in y):
                    continue
                ax.plot(x, y, marker="o", linewidth=2, color=colors[variant], label=label)
            ax.set_title(method_key)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Checkpoint step")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{title}: Text+Action vs Action-Only/No-Text ({direction})", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(metric_dir / f"text_vs_action_only_{metric}.png", dpi=180)
        plt.close(fig)

    plots = sorted(path.name for path in metric_dir.glob("*.png"))
    (metric_dir / "README.md").write_text(
        "# Action-Only No-Text Ablation Metric Plots\n\n"
        + "\n".join(f"- [{name}]({name})" for name in plots)
        + "\n",
        encoding="utf-8",
    )


def best_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row.get("method_key") == method.key]
        if not method_rows:
            continue
        best = max(
            method_rows,
            key=lambda row: (
                as_float(row, "mean_sharpness_ratio_generated_over_reference", 0.0)
                + 0.25 * as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference", 0.0)
                + 0.25 * as_float(row, "mean_motion_ratio_generated_over_reference", 0.0)
                - 0.002 * as_float(row, "fvd_future", 100.0)
            ),
        )
        selected.append(best)
    write_csv(BENCHMARK_DIR / "action_only_no_text_best_by_method.csv", selected)
    return selected


def write_report(action_only_rows: list[dict[str, Any]], combined_rows: list[dict[str, Any]]) -> None:
    best_rows = best_by_method(action_only_rows)
    report_path = BENCHMARK_DIR / "action_only_no_text_ablation_report.md"
    lines = [
        "# Action-Only No-Text Ablation Report",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Setup",
        "",
        "- Text/T5 prompt embeddings were zeroed and masked during training and inference.",
        "- Token-action methods used action tokens as the only valid cross-attention conditioning tokens.",
        "- AdaLN, middle-block, and temporal-bottleneck methods used one zero null prompt token plus their action pathway.",
        "- All runs used upsampled 24 FPS frame-action data, 49 context frames, 72 future frames, seed 231.",
        "- Metrics were computed with Modal containers, not local video decoding.",
        "",
        "## Best Action-Only Checkpoints",
        "",
        "| Method | Step | Gate | PSNR | SSIM | FVD | Sharpness | FFT-HF | Motion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            "| {method} | {step} | {gate:.3f} | {psnr:.3f} | {ssim:.4f} | {fvd:.3f} | {sharp:.4f} | {fft:.4f} | {motion:.4f} |".format(
                method=row.get("method_label", row.get("method_key", "")),
                step=int(float(row.get("checkpoint_step", 0))),
                gate=as_float(row, "eval_action_gate_scale", 1.0),
                psnr=as_float(row, "mean_future_psnr"),
                ssim=as_float(row, "mean_future_global_ssim"),
                fvd=as_float(row, "fvd_future"),
                sharp=as_float(row, "mean_sharpness_ratio_generated_over_reference"),
                fft=as_float(row, "mean_fft_high_frequency_energy_ratio_generated_over_reference"),
                motion=as_float(row, "mean_motion_ratio_generated_over_reference"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If action-only has higher sharpness/FFT but worse PSNR/SSIM, text may help semantic alignment but can also pull generation toward text priors.",
            "- If action-only has worse sharpness and worse motion, removing T5 did not solve the action-conditioning interference problem.",
            "- If token methods improve action-only while non-token methods do not, the original issue was likely text/action token competition inside cross-attention.",
            "- If V3 remains strongest, the key result is still low-frequency gated action routing, not simply removing text.",
            "",
            "## Files",
            "",
            f"- Summary CSV: `{BENCHMARK_DIR / 'action_only_no_text_summary_with_fvd.csv'}`",
            f"- Comparison CSV: `{BENCHMARK_DIR / 'text_vs_action_only_comparison_rows.csv'}`",
            f"- Plots: `{BENCHMARK_DIR / 'metric_plots'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis() -> None:
    rows = merge_summaries()
    combined = comparison_rows(rows)
    plot_comparison(combined)
    write_report(rows, combined)


def syntax_check() -> None:
    files = [
        "pipelines/training/train_ltx2b_waymo_visual_lora.py",
        "pipelines/inference/generate_waymo24_action_minterpolate_lora.py",
        "scripts/compute_video_quality_modal.py",
        "scripts/compute_action_fvd_modal.py",
        "scripts/run_action_only_no_text_ablation.py",
    ]
    run_command([str(PYTHON), "-m", "py_compile", *files], LOG_DIR / "preflight" / "py_compile.log")


def select_methods(value: str) -> list[Method]:
    if not value:
        return METHODS
    requested = {part.strip() for part in value.split(",") if part.strip()}
    known = {method.key: method for method in METHODS}
    missing = sorted(requested - set(known))
    if missing:
        raise ValueError(f"Unknown method key(s): {missing}; known={sorted(known)}")
    return [method for method in METHODS if method.key in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run action-only/no-text ablation for frame-action LTX methods.")
    parser.add_argument(
        "--phase",
        choices=("syntax", "train", "generate", "download", "metrics", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--train-workers", type=int, default=6)
    parser.add_argument("--generate-workers", type=int, default=4)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--only-method",
        default="",
        help="Comma-separated method keys to run. Default runs every method.",
    )
    parser.add_argument(
        "--detach-train",
        action="store_true",
        help="Use modal run --detach for training so remote apps are not killed if the local client disconnects.",
    )
    parser.add_argument(
        "--resume-method",
        default="",
        help="Method key to resume. Only this method receives resume flags.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Checkpoint name to resume from, for example step_000100.",
    )
    parser.add_argument(
        "--resume-run-name",
        default="",
        help="Optional source run name for resume. Defaults to the selected method run name.",
    )
    args = parser.parse_args()
    methods = select_methods(args.only_method)

    if args.phase in {"syntax", "all"}:
        syntax_check()
    if args.phase in {"train", "all"}:
        train_all(
            max_workers=args.train_workers,
            force=args.force_train,
            methods=methods,
            detach=args.detach_train,
            resume_method=args.resume_method,
            resume_checkpoint=args.resume_checkpoint,
            resume_run_name=args.resume_run_name,
        )
    if args.phase in {"generate", "all"}:
        generate_all(max_workers=args.generate_workers, seed=args.seed, limit=args.limit, methods=methods)
    if args.phase in {"download", "all"}:
        download_all(seed=args.seed, methods=methods)
    manifest_path = BENCHMARK_DIR / "manifest_action_only_no_text_ablation_seed231_all5.json"
    if args.phase in {"metrics", "analyze", "all"}:
        manifest_path = build_manifest(seed=args.seed, limit=args.limit, methods=methods)
    if args.phase in {"metrics", "all"}:
        run_quality_metrics(manifest_path)
        run_fvd(manifest_path)
    if args.phase in {"analyze", "all"}:
        run_analysis()
    print(json.dumps({"phase": args.phase, "benchmark_dir": str(BENCHMARK_DIR)}, sort_keys=True))


if __name__ == "__main__":
    main()
