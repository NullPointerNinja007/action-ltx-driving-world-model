from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODAL = ROOT / ".venv" / "bin" / "modal"
OUT = ROOT / "data/benchmarks/budget_fullval_or_512_action_conditioning_seed231"
LOG_DIR = ROOT / "data/modal_logs/budget_fullval_or_512_action_conditioning_seed231"
TMP = OUT / "tmp"

DATA_VOLUME = "waymo-e2e-24fps-121f-visual-continuation-data"
VAL_MANIFEST_RELPATH = "manifests/val_windows_24fps_121f_frame_action_conditions.csv"
SEED = 231
FPS = 24
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
TOTAL_FRAMES = 121


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    wrapper: str
    lora_run_name: str
    lora_step: str
    checkpoint_step: int
    generated_volume_key: str
    artifact_volume_name: str
    runs_root: str
    action: bool
    disable_text: bool = False
    base_label: str = "base_distilled_no_lora"
    action_gate_scale: float = 1.0
    action_vector_scale: float = 1.0


NOACTION = ModelConfig(
    key="noaction_shifted_step003000",
    label="No-action shifted LoRA step 3000",
    wrapper="scripts/wrappers/generate_waymo24_minterpolate_distilled_lora_shifted_timestep_ablation.py",
    lora_run_name="ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000",
    lora_step="step_003000",
    checkpoint_step=3000,
    generated_volume_key="noaction",
    artifact_volume_name="ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer",
    runs_root="distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs",
    action=False,
)

V4_MAIN_TEXT = ModelConfig(
    key="v4_main_text_step015984",
    label="V4 main text step 15984",
    wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py",
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_b200_seed231_from_shifted_noaction_step003000_steps3000",
    lora_step="step_015984",
    checkpoint_step=15984,
    generated_volume_key="v4",
    artifact_volume_name="ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-infer",
    runs_root="distilled098_full112_lowfreq_motion_v4_action_lora_24fps_minterpolate_seed231_runs",
    action=True,
)

V4_ACTIONSTRONG_TEXT = ModelConfig(
    key="v4_actionstrong_text_step007992",
    label="V4 action-strong text step 7992",
    wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py",
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_actionstrong_h100_seed231_from_shifted_noaction_step003000_steps7992",
    lora_step="step_007992",
    checkpoint_step=7992,
    generated_volume_key="v4",
    artifact_volume_name="ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-infer",
    runs_root="distilled098_full112_lowfreq_motion_v4_action_lora_24fps_minterpolate_seed231_runs",
    action=True,
)

ADALN = ModelConfig(
    key="frame_adaln_step002500",
    label="Frame AdaLN action step 2500",
    wrapper="scripts/wrappers/generate_waymo24_distilled_frame_adaln_action_minterpolate_lora.py",
    lora_run_name="ltx2b_dist098_waymo24_frame_adaln_action_lora_r16_seed231_from_noaction_step010000_lr5e6_actionlr5e5_steps3000",
    lora_step="step_002500",
    checkpoint_step=2500,
    generated_volume_key="adaln",
    artifact_volume_name="ltx2b-dist098-waymo24-frameadaln-action-lora-infer",
    runs_root="distilled098_frameadaln_action_lora_24fps_minterpolate_seed231_runs",
    action=True,
)

HFTV2 = ModelConfig(
    key="temporal_bottleneck_hfteacher_v2_step003000",
    label="Temporal bottleneck HF-teacher v2 step 3000",
    wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_hf_teacher_action_minterpolate_lora_v2.py",
    lora_run_name="ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_from_shifted_noaction_step003000_steps3000_resume1000",
    lora_step="step_003000",
    checkpoint_step=3000,
    generated_volume_key="hftv2",
    artifact_volume_name="ltx2b-dist098-waymo24-framebneck-hft-v2-infer",
    runs_root="distilled098_framebottleneck_hfteacher_v2_action_lora_24fps_minterpolate_seed231_runs",
    action=True,
)

V4_MAIN_NOTEXT = ModelConfig(
    key="v4_main_notext_step007992",
    label="V4 main no-text step 7992",
    wrapper="scripts/wrappers/generate_waymo24_distilled_frame_temporal_bottleneck_fullaction_motion_v4_action_minterpolate_lora.py",
    lora_run_name="ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_notext_h100_seed231_from_shifted_noaction_step003000_steps7992",
    lora_step="step_007992",
    checkpoint_step=7992,
    generated_volume_key="v4",
    artifact_volume_name="ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-infer",
    runs_root="distilled098_full112_lowfreq_motion_v4_action_lora_24fps_minterpolate_seed231_runs",
    action=True,
    disable_text=True,
)

CORE_MODELS = [NOACTION, V4_MAIN_TEXT, V4_ACTIONSTRONG_TEXT]
FALLBACK_MODELS = [NOACTION, ADALN, HFTV2, V4_MAIN_TEXT, V4_ACTIONSTRONG_TEXT, V4_MAIN_NOTEXT]


def run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_path is None:
        return subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    return proc


def ensure_val_manifest() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    local = TMP / "val_windows_24fps_121f_frame_action_conditions.csv"
    if local.exists():
        return local
    cmd = [str(MODAL), "volume", "get", DATA_VOLUME, VAL_MANIFEST_RELPATH, str(local)]
    proc = run(cmd, log_path=LOG_DIR / "download_val_manifest.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download val manifest; see {LOG_DIR / 'download_val_manifest.log'}")
    return local


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["window_idx"] = str(int(row["window_idx"]))
    return rows


def select_calibration_rows(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, str]]:
    window0 = [row for row in rows if int(row["window_idx"]) == 0]
    return sorted(window0, key=lambda row: (row["scenario_id"], row["window_id"]))[:limit]


def select_stratified_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window_idx"])].append(row)
    for window_idx in grouped:
        grouped[window_idx].sort(key=lambda row: (row["scenario_id"], row["window_id"]))
    per_window = limit // max(len(grouped), 1)
    remainder = limit % max(len(grouped), 1)
    selected: list[dict[str, str]] = []
    for offset, window_idx in enumerate(sorted(grouped)):
        take = per_window + (1 if offset < remainder else 0)
        selected.extend(grouped[window_idx][:take])
    return selected[:limit]


def group_for_generation(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window_idx"])].append(row)
    return dict(sorted(grouped.items()))


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


def safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def generation_command(model: ModelConfig, sources_json: Path, run_label: str, window_idx: int | None) -> tuple[list[str], dict[str, str]]:
    cmd = [
        str(MODAL),
        "run",
        model.wrapper,
        "--sources-json",
        str(sources_json),
        "--seed",
        str(SEED),
        "--lora-step",
        model.lora_step,
        "--lora-run-name",
        model.lora_run_name,
        "--run-label",
        run_label,
        "--base-label",
        model.base_label,
    ]
    if model.action:
        if window_idx is None:
            raise ValueError("Action models must be generated one window_idx group at a time.")
        cmd.extend(
            [
                "--action-manifest-split",
                "val",
                "--action-window-idx",
                str(window_idx),
                "--action-gate-scale",
                str(model.action_gate_scale),
                "--action-vector-scale",
                str(model.action_vector_scale),
                "--counterfactual-action-mode",
                "correct",
            ]
        )
    env = {
        "LTX_MODAL_GPU": "H100",
        "LTX_IMAGE_COND_NOISE_SCALE": "0.0",
        "LTX_LORA_RUN_NAME": model.lora_run_name,
        "WAYMO24_DISABLE_TEXT_CONDITIONING": "1" if model.disable_text else "0",
    }
    return cmd, env


def download_run_summary(model: ModelConfig, run_label: str, destination: Path) -> dict[str, Any]:
    remote = f"{model.runs_root}/{run_label}/run_summary.json"
    if destination.exists():
        destination.unlink()
    proc = run(
        [str(MODAL), "volume", "get", model.artifact_volume_name, remote, str(destination)],
        log_path=LOG_DIR / f"get_summary_{safe_label(run_label)}.log",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download summary for {run_label}; see {LOG_DIR / f'get_summary_{safe_label(run_label)}.log'}")
    return json.loads(destination.read_text(encoding="utf-8"))


def run_generation_job(args: tuple[ModelConfig, list[dict[str, str]], str, int | None]) -> dict[str, Any]:
    model, rows, phase, window_idx = args
    run_label = f"budget_{phase}_{model.key}"
    if window_idx is not None:
        run_label += f"_w{window_idx}"
    run_label = safe_label(run_label)
    sources_json = TMP / "sources" / f"{run_label}.json"
    write_sources_json(rows, sources_json)
    cmd, env = generation_command(model, sources_json, run_label, window_idx)
    log_path = LOG_DIR / f"{run_label}.log"
    started = time.monotonic()
    proc = run(cmd, env=env, log_path=log_path)
    duration = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(f"Generation failed for {run_label}; see {log_path}")

    summary_path = TMP / "summaries" / f"{run_label}.json"
    summary = download_run_summary(model, run_label, summary_path)
    results = summary.get("results", [])
    return {
        "model_key": model.key,
        "model_label": model.label,
        "checkpoint_step": model.checkpoint_step,
        "phase": phase,
        "run_label": run_label,
        "window_idx": window_idx,
        "num_requested": len(rows),
        "num_outputs": len(results),
        "duration_seconds": duration,
        "sec_per_clip": duration / max(len(results), 1),
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def generation_jobs(models: list[ModelConfig], rows: list[dict[str, str]], phase: str) -> list[tuple[ModelConfig, list[dict[str, str]], str, int | None]]:
    jobs = []
    grouped = group_for_generation(rows)
    for model in models:
        if model.action:
            for window_idx, group_rows in grouped.items():
                jobs.append((model, group_rows, phase, window_idx))
        else:
            for window_idx, group_rows in grouped.items():
                jobs.append((model, group_rows, phase, window_idx))
    return jobs


def run_generation_jobs(jobs: list[tuple[ModelConfig, list[dict[str, str]], str, int | None]], *, max_workers: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_generation_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(
                json.dumps(
                    {
                        "run_label": result["run_label"],
                        "num_outputs": result["num_outputs"],
                        "sec_per_clip": result["sec_per_clip"],
                    },
                    sort_keys=True,
                )
            )
    records.sort(key=lambda item: (item["model_key"], str(item["window_idx"])))
    return records


def flatten_generated_records(job_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in job_results:
        for result in job["summary"].get("results", []):
            record = dict(result)
            record.update(
                {
                    "model_key": job["model_key"],
                    "model_label": job["model_label"],
                    "checkpoint_step": job["checkpoint_step"],
                    "generated_volume_key": model_key_to_volume_key(job["model_key"]),
                    "generated_volume_name": model_key_to_artifact_volume(job["model_key"]),
                    "run_label": job["run_label"],
                    "phase": job["phase"],
                }
            )
            if "window_id" not in record:
                record["window_id"] = result.get("action_window_id", "")
            if "window_idx" not in record:
                record["window_idx"] = job["window_idx"] if job["window_idx"] is not None else ""
            records.append(record)
    return records


def model_key_to_volume_key(model_key: str) -> str:
    for model in [*CORE_MODELS, *FALLBACK_MODELS]:
        if model.key == model_key:
            return model.generated_volume_key
    raise KeyError(model_key)


def model_key_to_artifact_volume(model_key: str) -> str:
    for model in [*CORE_MODELS, *FALLBACK_MODELS]:
        if model.key == model_key:
            return model.artifact_volume_name
    raise KeyError(model_key)


def write_generation_manifest(path: Path, rows: list[dict[str, str]], job_results: list[dict[str, Any]], branch: str) -> list[dict[str, Any]]:
    records = flatten_generated_records(job_results)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "seed": SEED,
        "fps": FPS,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "num_source_windows": len(rows),
        "num_generated_records": len(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def sec_per_clip_from_calibration(job_results: list[dict[str, Any]]) -> float:
    total_seconds = sum(float(job["duration_seconds"]) for job in job_results)
    total_outputs = sum(int(job["num_outputs"]) for job in job_results)
    return total_seconds / max(total_outputs, 1)


def run_metrics(manifest_path: Path) -> None:
    cmd = [
        str(MODAL),
        "run",
        "scripts/compute_budget_validation_metrics_modal.py",
        str(manifest_path),
        "--output-dir",
        str(OUT),
        "--chunk-size",
        "8",
    ]
    proc = run(cmd, log_path=LOG_DIR / "budget_metrics.log")
    if proc.returncode != 0:
        raise RuntimeError(f"Budget metrics failed; see {LOG_DIR / 'budget_metrics.log'}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_results() -> None:
    import matplotlib.pyplot as plt

    summary = read_csv(OUT / "model_summary.csv")
    fvd = {row["model_key"]: row for row in read_csv(OUT / "fvd_summary.csv")}
    if not summary:
        return

    labels = [row["model_label"] for row in summary]
    x = list(range(len(labels)))
    plot_dir = OUT

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in summary]

    plt.figure(figsize=(max(8, len(labels) * 1.4), 4.5))
    fvd_vals = [float(fvd.get(row["model_key"], {}).get("fvd_future", "nan")) for row in summary]
    plt.bar(x, fvd_vals)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("FVD-style future distance, lower better")
    plt.tight_layout()
    plt.savefig(plot_dir / "fvd_vs_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.4), 4.5))
    width = 0.38
    plt.bar([i - width / 2 for i in x], values("mean_sharpness_ratio_generated_over_reference"), width=width, label="Sharpness ratio")
    plt.bar([i + width / 2 for i in x], values("mean_motion_ratio_generated_over_reference"), width=width, label="Motion ratio")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Generated / reference ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "sharpness_motion_by_model.png", dpi=180)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 1.4), 4.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.bar([i - width / 2 for i in x], values("mean_future_psnr"), width=width, color="#5177b8", label="PSNR")
    ax2.bar([i + width / 2 for i in x], values("mean_future_global_ssim"), width=width, color="#d9903d", label="SSIM")
    ax1.set_ylabel("Future PSNR, higher better")
    ax2.set_ylabel("Future SSIM, higher better")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_dir / "psnr_ssim_by_model.png", dpi=180)
    plt.close()

    gate_proxy = [0.0 if row["model_key"].startswith("noaction") else 1.0 for row in summary]
    plt.figure(figsize=(7, 4.5))
    plt.scatter(gate_proxy, values("mean_sharpness_ratio_generated_over_reference"), s=80)
    for i, label in enumerate(labels):
        plt.annotate(label, (gate_proxy[i], values("mean_sharpness_ratio_generated_over_reference")[i]), fontsize=8)
    plt.xlabel("Action path enabled proxy")
    plt.ylabel("Sharpness ratio")
    plt.tight_layout()
    plt.savefig(plot_dir / "gate_vs_quality_budget.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    sharp = values("mean_sharpness_ratio_generated_over_reference")
    plt.scatter(fvd_vals, sharp, s=90)
    for i, label in enumerate(labels):
        plt.annotate(label, (fvd_vals[i], sharp[i]), fontsize=8)
    plt.xlabel("FVD-style future distance, lower better")
    plt.ylabel("Sharpness ratio, higher better")
    plt.tight_layout()
    plt.savefig(plot_dir / "final_budget_pareto_fvd_vs_sharpness.png", dpi=180)
    plt.close()


def write_report(
    *,
    calibration_jobs: list[dict[str, Any]],
    selected_jobs: list[dict[str, Any]],
    branch: str,
    selected_rows: list[dict[str, str]],
    sec_per_clip_mean: float,
) -> None:
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_usd": 90,
        "seed": SEED,
        "calibration_sec_per_clip_mean": sec_per_clip_mean,
        "branch": branch,
        "num_selected_source_windows": len(selected_rows),
        "calibration_jobs": [
            {k: v for k, v in job.items() if k != "summary"} for job in calibration_jobs
        ],
        "selected_generation_jobs": [
            {k: v for k, v in job.items() if k != "summary"} for job in selected_jobs
        ],
        "outputs": {
            "selected_validation_manifest": str(OUT / "selected_validation_manifest.json"),
            "per_clip_metrics": str(OUT / "per_clip_metrics.csv"),
            "model_summary": str(OUT / "model_summary.csv"),
            "fvd_summary": str(OUT / "fvd_summary.csv"),
        },
    }
    (OUT / "budget_run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ensure_val_manifest()
    all_rows = read_rows(manifest_path)

    calibration_rows = select_calibration_rows(all_rows, limit=20)
    (OUT / "calibration_sources.json").write_text(
        json.dumps(source_payload(calibration_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_jobs = run_generation_jobs(generation_jobs(CORE_MODELS, calibration_rows, "calib20"), max_workers=3)
    sec_per_clip = sec_per_clip_from_calibration(calibration_jobs)
    (OUT / "calibration_runtime.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sec_per_clip_mean": sec_per_clip,
                "num_outputs": sum(int(job["num_outputs"]) for job in calibration_jobs),
                "jobs": [{k: v for k, v in job.items() if k != "summary"} for job in calibration_jobs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if sec_per_clip <= 8.0:
        branch = "full_val_3_config_core"
        selected_rows = all_rows
        selected_models = CORE_MODELS
        max_workers = 10
    elif sec_per_clip <= 12.0:
        branch = "stratified_512_6_config"
        selected_rows = select_stratified_rows(all_rows, limit=512)
        selected_models = FALLBACK_MODELS
        max_workers = 10
    else:
        branch = "stratified_256_6_config"
        selected_rows = select_stratified_rows(all_rows, limit=256)
        selected_models = FALLBACK_MODELS
        max_workers = 10

    (OUT / "selected_source_windows.json").write_text(
        json.dumps(selected_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_jobs = run_generation_jobs(generation_jobs(selected_models, selected_rows, branch), max_workers=max_workers)
    selected_manifest = OUT / "selected_validation_manifest.json"
    write_generation_manifest(selected_manifest, selected_rows, selected_jobs, branch)
    run_metrics(selected_manifest)
    plot_results()
    write_report(
        calibration_jobs=calibration_jobs,
        selected_jobs=selected_jobs,
        branch=branch,
        selected_rows=selected_rows,
        sec_per_clip_mean=sec_per_clip,
    )
    print(json.dumps({"branch": branch, "sec_per_clip_mean": sec_per_clip, "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
