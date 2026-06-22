from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"

OUT = ROOT / "data/benchmarks/final_action_alignment_validation_seed231"
LOG_DIR = ROOT / "data/modal_logs/final_action_alignment_validation_seed231"
TMP = OUT / "tmp"
SOURCES_DIR = TMP / "sources"
SUMMARIES_DIR = TMP / "remote_summaries"

DATA_VOLUME = "waymo-e2e-24fps-121f-visual-continuation-data"
VAL_MANIFEST_RELPATH = "manifests/val_windows_24fps_121f_frame_action_conditions.csv"

V4_CHECKPOINT_VOLUME = "ltx2b-v4-b200-rank-capacity-ckpts"
V4_ARTIFACT_VOLUME = "ltx2b-final-action-alignment-validation-infer"
V4_RUNS_ROOT = "final_action_alignment_validation_seed231_runs"

NOACTION_CHECKPOINT_VOLUME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts"
NOACTION_ARTIFACT_VOLUME = "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer"
NOACTION_RUNS_ROOT = "distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs"

SEED = 231
FPS = 24
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121

ACTION_MODES = ["correct", "zero", "shuffled", "reversed_future"]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    purpose: str
    wrapper: str
    lora_run_name: str
    lora_step: str
    checkpoint_step: int
    checkpoint_volume_name: str
    artifact_volume_name: str
    generated_volume_key: str
    runs_root: str
    action: bool
    modes: tuple[str, ...]
    disable_text: bool = False
    base_label: str = "base_distilled_no_lora"
    action_gate_scale: float = 1.0
    action_vector_scale: float = 1.0


NOACTION = ModelConfig(
    key="noaction_shifted_step003000",
    label="No-action shifted LoRA step 3000",
    purpose="visual baseline; no action path, counterfactual sensitivity is structurally zero",
    wrapper="generate_waymo24_minterpolate_distilled_lora_shifted_timestep_ablation.py",
    lora_run_name="ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000",
    lora_step="step_003000",
    checkpoint_step=3000,
    checkpoint_volume_name=NOACTION_CHECKPOINT_VOLUME,
    artifact_volume_name=NOACTION_ARTIFACT_VOLUME,
    generated_volume_key="noaction",
    runs_root=NOACTION_RUNS_ROOT,
    action=False,
    modes=("correct",),
)

V4_R64_SELECTED = ModelConfig(
    key="v4_r64_selected_step018000",
    label="V4 rank 64 selected step 18000",
    purpose="primary final action model: best selected quality/sensitivity operating point",
    wrapper="generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py",
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r64_b200_seed231_resume007992_to023976",
    lora_step="step_018000",
    checkpoint_step=18000,
    checkpoint_volume_name=V4_CHECKPOINT_VOLUME,
    artifact_volume_name=V4_ARTIFACT_VOLUME,
    generated_volume_key="final_action_alignment",
    runs_root=V4_RUNS_ROOT,
    action=True,
    modes=tuple(ACTION_MODES),
)

V4_R64_QUALITY = ModelConfig(
    key="v4_r64_quality_step010000",
    label="V4 rank 64 quality step 10000",
    purpose="best observed FVD action checkpoint",
    wrapper=V4_R64_SELECTED.wrapper,
    lora_run_name=V4_R64_SELECTED.lora_run_name,
    lora_step="step_010000",
    checkpoint_step=10000,
    checkpoint_volume_name=V4_CHECKPOINT_VOLUME,
    artifact_volume_name=V4_ARTIFACT_VOLUME,
    generated_volume_key="final_action_alignment",
    runs_root=V4_RUNS_ROOT,
    action=True,
    modes=tuple(ACTION_MODES),
)

V4_R32_SENSITIVITY = ModelConfig(
    key="v4_r32_sensitivity_step010000",
    label="V4 rank 32 sensitivity step 10000",
    purpose="max observed 5-clip counterfactual sensitivity checkpoint",
    wrapper=V4_R64_SELECTED.wrapper,
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r32_b200_seed231_resume007992_to023976",
    lora_step="step_010000",
    checkpoint_step=10000,
    checkpoint_volume_name=V4_CHECKPOINT_VOLUME,
    artifact_volume_name=V4_ARTIFACT_VOLUME,
    generated_volume_key="final_action_alignment",
    runs_root=V4_RUNS_ROOT,
    action=True,
    modes=tuple(ACTION_MODES),
)

V4_R16_SENSITIVE = ModelConfig(
    key="v4_r16_sensitive_step023976",
    label="V4 rank 16 continued step 23976",
    purpose="strong old sensitivity baseline after extended training",
    wrapper=V4_R64_SELECTED.wrapper,
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_text_r16_b200_seed231_resume015984_to023976",
    lora_step="step_023976",
    checkpoint_step=23976,
    checkpoint_volume_name=V4_CHECKPOINT_VOLUME,
    artifact_volume_name=V4_ARTIFACT_VOLUME,
    generated_volume_key="final_action_alignment",
    runs_root=V4_RUNS_ROOT,
    action=True,
    modes=tuple(ACTION_MODES),
)

MODELS = [NOACTION, V4_R64_SELECTED, V4_R64_QUALITY, V4_R32_SENSITIVITY, V4_R16_SENSITIVE]


@dataclass(frozen=True)
class GenerationJob:
    phase: str
    model: ModelConfig
    mode: str
    window_idx: int | None
    chunk_idx: int
    rows: tuple[dict[str, str], ...]
    run_label: str
    sources_json: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path is None:
        return subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, stdout=handle, stderr=subprocess.STDOUT)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_val_manifest() -> Path:
    ensure_dirs()
    local = TMP / "val_windows_24fps_121f_frame_action_conditions.csv"
    if local.exists():
        return local
    cmd = [str(MODAL), "volume", "get", DATA_VOLUME, VAL_MANIFEST_RELPATH, str(local)]
    proc = run_cmd(cmd, log_path=LOG_DIR / "download_val_manifest.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download val manifest; see {LOG_DIR / 'download_val_manifest.log'}")
    return local


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["window_idx"] = str(int(row["window_idx"]))
    rows.sort(key=lambda row: (int(row["window_idx"]), row["scenario_id"], row["window_id"]))
    return rows


def safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def chunked(rows: list[dict[str, str]], chunk_size: int) -> list[list[dict[str, str]]]:
    return [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]


def select_smoke_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if int(row["window_idx"]) == 0][:2]


def select_calibration_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    # Spread calibration across all four window indices while keeping deterministic scene ordering.
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window_idx"])].append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit:
        made_progress = False
        for window_idx in sorted(grouped):
            offset = len([r for r in selected if int(r["window_idx"]) == window_idx])
            if offset < len(grouped[window_idx]):
                selected.append(grouped[window_idx][offset])
                made_progress = True
                if len(selected) >= limit:
                    break
        if not made_progress:
            break
    selected.sort(key=lambda row: (int(row["window_idx"]), row["scenario_id"], row["window_id"]))
    return selected


def source_payload(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        payload.append(
            {
                "scene_token": row["scenario_id"][:12],
                "source_filename": Path(row["mp4_relpath"]).name,
                "source_relpath": row["mp4_relpath"],
                "source_volume": "data",
                "window_id": row["window_id"],
                "window_idx": int(row["window_idx"]),
            }
        )
    return payload


def write_sources_json(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": source_payload(rows)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def group_rows_for_model(rows: list[dict[str, str]], model: ModelConfig, chunk_size: int) -> list[tuple[int | None, int, list[dict[str, str]]]]:
    if not model.action:
        return [(None, idx, shard) for idx, shard in enumerate(chunked(rows, chunk_size))]
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window_idx"])].append(row)
    jobs = []
    for window_idx in sorted(grouped):
        for chunk_idx, shard in enumerate(chunked(grouped[window_idx], chunk_size)):
            jobs.append((window_idx, chunk_idx, shard))
    return jobs


def build_generation_jobs(phase: str, rows: list[dict[str, str]], models: list[ModelConfig], chunk_size: int) -> list[GenerationJob]:
    jobs: list[GenerationJob] = []
    for model in models:
        for mode in model.modes:
            for window_idx, chunk_idx, shard in group_rows_for_model(rows, model, chunk_size):
                window_label = f"w{window_idx}" if window_idx is not None else "wall"
                run_label = safe_label(f"finalalign_{phase}_{model.key}_{mode}_{window_label}_c{chunk_idx:04d}")
                sources_json = SOURCES_DIR / phase / model.key / mode / f"{window_label}_chunk{chunk_idx:04d}.json"
                write_sources_json(shard, sources_json)
                jobs.append(
                    GenerationJob(
                        phase=phase,
                        model=model,
                        mode=mode,
                        window_idx=window_idx,
                        chunk_idx=chunk_idx,
                        rows=tuple(shard),
                        run_label=run_label,
                        sources_json=sources_json,
                    )
                )
    return jobs


def generation_env(model: ModelConfig, gpu: str) -> dict[str, str]:
    env = {
        "LTX_MODAL_GPU": gpu,
        "LTX_CHECKPOINT_VOLUME_NAME": model.checkpoint_volume_name,
        "LTX_ARTIFACTS_VOLUME_NAME": model.artifact_volume_name,
        "LTX_RUNS_ROOT": model.runs_root,
        "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
        "LTX_LORA_RUN_NAME": model.lora_run_name,
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if model.disable_text else "0",
    }
    if model.action:
        env.update(
            {
                "WAYMO24_FRAME_ACTION_FEATURE_KEY": "actions_full_112",
                "WAYMO24_FRAME_ACTION_STATS_RELPATH": "manifests/frame_action_24fps_full112_normalization_stats.json",
            }
        )
    return env


def generation_command(job: GenerationJob, gpu: str) -> tuple[list[str], dict[str, str]]:
    model = job.model
    cmd = [
        str(MODAL),
        "run",
        model.wrapper,
        "--sources-json",
        str(job.sources_json),
        "--seed",
        str(SEED),
        "--lora-step",
        model.lora_step,
        "--lora-run-name",
        model.lora_run_name,
        "--run-label",
        job.run_label,
        "--base-label",
        model.base_label,
    ]
    if model.action:
        if job.window_idx is None:
            raise ValueError("Action jobs require a window_idx.")
        cmd.extend(
            [
                "--action-manifest-split",
                "val",
                "--action-window-idx",
                str(job.window_idx),
                "--action-gate-scale",
                str(model.action_gate_scale),
                "--action-vector-scale",
                str(model.action_vector_scale),
                "--counterfactual-action-mode",
                job.mode,
                "--counterfactual-rotation",
                "1",
            ]
        )
    return cmd, generation_env(model, gpu)


def local_summary_path(job: GenerationJob) -> Path:
    return SUMMARIES_DIR / job.phase / job.model.key / job.mode / f"{job.run_label}.json"


def remote_summary_relpath(job: GenerationJob) -> str:
    return f"{job.model.runs_root}/{job.run_label}/run_summary.json"


def fetch_remote_summary(job: GenerationJob, *, force: bool = False) -> dict[str, Any] | None:
    local = local_summary_path(job)
    if local.exists() and not force:
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            local.unlink()
    local.parent.mkdir(parents=True, exist_ok=True)
    proc = run_cmd(
        [str(MODAL), "volume", "get", job.model.artifact_volume_name, remote_summary_relpath(job), str(local)],
        log_path=LOG_DIR / "summary_fetch" / job.phase / f"{job.run_label}.log",
    )
    if proc.returncode != 0:
        if local.exists():
            local.unlink()
        return None
    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def summary_complete(job: GenerationJob, summary: dict[str, Any] | None) -> bool:
    if not summary:
        return False
    results = summary.get("results", [])
    return isinstance(results, list) and len(results) == len(job.rows)


def preflight(rows: list[dict[str, str]]) -> None:
    if len(rows) != 1904:
        raise RuntimeError(f"Expected 1904 val windows, found {len(rows)}")
    required_cols = {"mp4_relpath", "frame_action_relpath", "window_idx", "window_id", "scenario_id"}
    missing = required_cols.difference(rows[0])
    if missing:
        raise RuntimeError(f"Val manifest missing required columns: {sorted(missing)}")

    checks = []
    for model in MODELS:
        checks.append(
            {
                "model_key": model.key,
                "checkpoint_volume": model.checkpoint_volume_name,
                "checkpoint_relpath": f"{model.lora_run_name}/{model.lora_step}",
            }
        )
    (OUT / "preflight_checkpoint_expectations.json").write_text(
        json.dumps({"created_at_utc": utc_now(), "checks": checks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for check in checks:
        log_name = safe_label(f"{check['model_key']}_{check['checkpoint_relpath']}.log")
        proc = run_cmd(
            [str(MODAL), "volume", "ls", check["checkpoint_volume"], check["checkpoint_relpath"]],
            log_path=LOG_DIR / "preflight" / log_name,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Missing checkpoint {check['checkpoint_volume']}/{check['checkpoint_relpath']}; "
                f"see {LOG_DIR / 'preflight' / log_name}"
            )


def run_generation_job(job: GenerationJob, *, gpu: str, force: bool) -> dict[str, Any]:
    start = time.time()
    if not force:
        existing = fetch_remote_summary(job)
        if summary_complete(job, existing):
            return {
                "job": job,
                "summary": existing,
                "status": "skipped_existing",
                "seconds": 0.0,
                "gpu": gpu,
            }

    cmd, env = generation_command(job, gpu)
    log_path = LOG_DIR / "generation" / job.phase / job.model.key / job.mode / f"{job.run_label}.log"
    proc = run_cmd(cmd, env=env, log_path=log_path)
    seconds = time.time() - start
    if proc.returncode != 0 and "billing cycle spend limit reached" in log_path.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"Modal spend limit reached while launching {job.run_label}; see {log_path}")
    if proc.returncode != 0 and gpu == "B200":
        retry_gpu = "H100"
        retry_cmd, retry_env = generation_command(job, retry_gpu)
        retry_log_path = LOG_DIR / "generation" / job.phase / job.model.key / job.mode / f"{job.run_label}_h100_retry.log"
        retry_proc = run_cmd(retry_cmd, env=retry_env, log_path=retry_log_path)
        seconds = time.time() - start
        if retry_proc.returncode != 0 and "billing cycle spend limit reached" in retry_log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ):
            raise RuntimeError(f"Modal spend limit reached while retrying {job.run_label}; see {retry_log_path}")
        if retry_proc.returncode != 0:
            raise RuntimeError(f"Generation failed on B200 and H100 retry for {job.run_label}; see {retry_log_path}")
        gpu = retry_gpu
    elif proc.returncode != 0:
        raise RuntimeError(f"Generation failed for {job.run_label}; see {log_path}")

    summary = fetch_remote_summary(job, force=True)
    if not summary_complete(job, summary):
        raise RuntimeError(f"Generation summary incomplete for {job.run_label}: {summary}")
    return {
        "job": job,
        "summary": summary,
        "status": "generated",
        "seconds": seconds,
        "gpu": gpu,
    }


def run_generation_jobs(jobs: list[GenerationJob], *, gpu: str, max_workers: int, force: bool, phase: str) -> list[dict[str, Any]]:
    phase_report_path = OUT / f"{phase}_generation_runtime.json"
    completed: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(run_generation_job, job, gpu=gpu, force=force) for job in jobs]
    try:
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            completed.append(
                {
                    "run_label": result["job"].run_label,
                    "model_key": result["job"].model.key,
                    "mode": result["job"].mode,
                    "num_rows": len(result["job"].rows),
                    "status": result["status"],
                    "seconds": result["seconds"],
                    "gpu": result["gpu"],
                    "finished_at_utc": utc_now(),
                }
            )
            if idx % 10 == 0 or idx == len(jobs):
                phase_report_path.write_text(
                    json.dumps(
                        {
                            "phase": phase,
                            "updated_at_utc": utc_now(),
                            "num_jobs": len(jobs),
                            "num_completed": len(completed),
                            "jobs": completed,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
    return completed


def enrich_record(record: dict[str, Any], model: ModelConfig, mode: str, row_by_source: dict[str, dict[str, str]]) -> dict[str, Any]:
    source_relpath = record["source_relpath"]
    source_row = row_by_source.get(source_relpath, {})
    out = dict(record)
    out.update(
        {
            "model_key": model.key,
            "model_label": model.label,
            "model_purpose": model.purpose,
            "checkpoint_step": model.checkpoint_step,
            "counterfactual_action_mode": mode,
            "generated_volume_key": model.generated_volume_key,
            "artifact_volume_name": model.artifact_volume_name,
            "checkpoint_volume_name": model.checkpoint_volume_name,
            "runs_root": model.runs_root,
            "disable_text_conditioning": bool(model.disable_text),
            "image_cond_noise_scale": float(record.get("image_cond_noise_scale", 0.0)),
            "frame_action_feature_key": record.get("frame_action_feature_key", "actions_full_112" if model.action else ""),
            "frame_action_stats_relpath": record.get(
                "frame_action_stats_relpath",
                "manifests/frame_action_24fps_full112_normalization_stats.json" if model.action else "",
            ),
            "use_frame_actions": bool(model.action),
            "window_id": source_row.get("window_id", record.get("window_id", "")),
            "window_idx": int(source_row.get("window_idx", record.get("window_idx", -1))),
            "scenario_id": source_row.get("scenario_id", record.get("scene_token", "")),
            "mp4_relpath": source_row.get("mp4_relpath", source_relpath),
            "frame_action_relpath": source_row.get("frame_action_relpath", record.get("frame_action_relpath", "")),
            "latent_relpath": source_row.get("latent_relpath", ""),
            "seed": int(record.get("seed", SEED)),
            "fps": int(record.get("fps", FPS)),
            "context_frames": int(record.get("context_frames", CONTEXT_FRAMES)),
            "future_frames": int(record.get("future_frames", FUTURE_FRAMES)),
            "total_frames": int(record.get("total_frames", TOTAL_FRAMES)),
        }
    )
    return out


def write_generation_manifest(
    path: Path,
    jobs: list[GenerationJob],
    rows: list[dict[str, str]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_by_source = {row["mp4_relpath"]: row for row in rows}
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for job in jobs:
        summary = fetch_remote_summary(job)
        if not summary_complete(job, summary):
            missing.append({"run_label": job.run_label, "model_key": job.model.key, "mode": job.mode})
            continue
        for result in summary["results"]:
            records.append(enrich_record(result, job.model, job.mode, row_by_source))
    records.sort(
        key=lambda record: (
            str(record["model_key"]),
            str(record["counterfactual_action_mode"]),
            int(record.get("window_idx", -1)),
            str(record.get("window_id", "")),
        )
    )
    payload = {
        "created_at_utc": utc_now(),
        "seed": SEED,
        "fps": FPS,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "image_cond_noise_scale": 0.0,
        "frame_action_feature_key": "actions_full_112",
        "num_source_windows": len(rows),
        "num_records": len(records),
        "expected_records": sum(len(job.rows) for job in jobs),
        "num_missing_shards": len(missing),
        "missing_shards": missing,
        "models": [asdict(model) for model in MODELS],
        "runtime_jobs": runtime_rows,
        "source_windows": rows,
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_metrics(manifest_path: Path, *, output_dir: Path, force: bool) -> None:
    report_path = output_dir / "metrics_modal_report.json"
    if report_path.exists() and not force:
        return
    cmd = [
        str(MODAL),
        "run",
        "scripts/compute_final_action_alignment_metrics_modal.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--quality-chunk-size",
        "32",
        "--alignment-chunk-size",
        "12",
        "--fvd-num-frames",
        "16",
        "--fvd-size",
        "112",
        "--fvd-batch-size",
        "8",
    ]
    proc = run_cmd(cmd, log_path=LOG_DIR / "metrics" / f"{manifest_path.stem}.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Metric run failed; see {LOG_DIR / 'metrics' / f'{manifest_path.stem}.log'}")


def write_campaign_config(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    payload = {
        "created_at_utc": utc_now(),
        "plan": "final_action_alignment_validation",
        "seed": SEED,
        "fps": FPS,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "image_cond_noise_scale": 0.0,
        "frame_action_feature_key": "actions_full_112",
        "num_val_windows": len(rows),
        "chunk_size": args.chunk_size,
        "max_generation_workers": args.max_generation_workers,
        "generation_gpu": args.gpu,
        "models": [asdict(model) for model in MODELS],
    }
    (OUT / "campaign_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def phase_rows(phase: str, rows: list[dict[str, str]], calibration_limit: int) -> list[dict[str, str]]:
    if phase == "smoke":
        return select_smoke_rows(rows)
    if phase == "calibration":
        return select_calibration_rows(rows, calibration_limit)
    if phase == "full":
        return rows
    raise ValueError(f"Unknown generation phase: {phase}")


def phase_models(phase: str) -> list[ModelConfig]:
    if phase == "smoke":
        return [NOACTION, V4_R64_SELECTED]
    return MODELS


def run_generation_phase(
    phase: str,
    rows: list[dict[str, str]],
    *,
    args: argparse.Namespace,
) -> Path:
    selected_rows = phase_rows(phase, rows, args.calibration_limit)
    jobs = build_generation_jobs(phase, selected_rows, phase_models(phase), args.chunk_size)
    expected = sum(len(job.rows) for job in jobs)
    print(f"[{phase}] rows={len(selected_rows)} jobs={len(jobs)} expected_videos={expected}")
    runtime_rows = run_generation_jobs(
        jobs,
        gpu=args.gpu,
        max_workers=args.max_generation_workers,
        force=args.force,
        phase=phase,
    )
    manifest_name = {
        "smoke": "smoke_generation_manifest.json",
        "calibration": "calibration_generation_manifest.json",
        "full": "full_val_generation_manifest.json",
    }[phase]
    manifest_path = OUT / manifest_name
    payload = write_generation_manifest(manifest_path, jobs, selected_rows, runtime_rows)
    if payload["num_records"] != expected:
        raise RuntimeError(f"{phase} generated {payload['num_records']} records, expected {expected}")
    if phase == "calibration":
        generated_jobs = [row for row in runtime_rows if row["status"] == "generated" and row["seconds"] > 0.0]
        seconds = sum(float(row["seconds"]) for row in generated_jobs)
        clips = sum(int(row["num_rows"]) for row in generated_jobs)
        calibration = {
            "created_at_utc": utc_now(),
            "num_generated_jobs": len(generated_jobs),
            "num_generated_clips": clips,
            "sum_wall_seconds_across_jobs": seconds,
            "mean_observed_job_seconds_per_clip": seconds / max(clips, 1),
            "note": "This is per-shard observed job wall time, not serialized cluster wall time.",
        }
        (OUT / "calibration_runtime.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Final full-val action-alignment validation campaign.")
    parser.add_argument(
        "--phase",
        choices=["all", "preflight", "smoke", "calibration", "full", "metrics"],
        default="all",
        help="Run a single phase or the whole campaign.",
    )
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--calibration-limit", type=int, default=20)
    parser.add_argument("--max-generation-workers", type=int, default=10)
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    rows = read_rows(ensure_val_manifest())
    write_campaign_config(args, rows)

    if args.phase in {"all", "preflight"}:
        preflight(rows)
        if args.phase == "preflight":
            return

    if args.phase in {"all", "smoke"} and not args.skip_smoke:
        smoke_manifest = run_generation_phase("smoke", rows, args=args)
        run_metrics(smoke_manifest, output_dir=OUT / "smoke_metrics", force=args.force)
        if args.phase == "smoke":
            return

    if args.phase in {"all", "calibration"} and not args.skip_calibration:
        calibration_manifest = run_generation_phase("calibration", rows, args=args)
        run_metrics(calibration_manifest, output_dir=OUT / "calibration_metrics", force=args.force)
        if args.phase == "calibration":
            return

    if args.phase in {"all", "full"}:
        full_manifest = run_generation_phase("full", rows, args=args)
        run_metrics(full_manifest, output_dir=OUT, force=args.force)
        if args.phase == "full":
            return

    if args.phase == "metrics":
        full_manifest = OUT / "full_val_generation_manifest.json"
        if not full_manifest.exists():
            raise FileNotFoundError(full_manifest)
        run_metrics(full_manifest, output_dir=OUT, force=args.force)


if __name__ == "__main__":
    main()
