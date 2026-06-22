from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "counterfactual-sensitivity-benchmark"
VOLUME_NAME = os.environ.get("VIDEO_QUALITY_VOLUME_NAME", "video-quality-metrics-benchmark-clips")
REMOTE_ROOT = Path("/quality_data")
CONTEXT_FRAMES = 49

app = modal.App(APP_NAME)
quality_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("numpy", "imageio-ffmpeg")
)


def safe_relpath(path: Path, index: int) -> str:
    suffix = path.suffix or ".mp4"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:160]
    return f"{index:05d}_{stem}{suffix}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def build_comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in records
        if row.get("diagnostic_group") == "counterfactual_suite"
        and str(row.get("counterfactual_action_mode")) in {"correct", "zero", "shuffled", "reversed_future"}
    ]
    grouped: dict[tuple[str, str, float, float, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[
            (
                row["checkpoint_name"],
                row["scene_token"],
                float(row.get("action_gate_scale", 1.0)),
                float(row.get("action_vector_scale", 1.0)),
                str(row.get("variant", "")),
                str(row.get("method_key", "")),
            )
        ][str(row["counterfactual_action_mode"])] = row

    comparisons: list[dict[str, Any]] = []
    for (
        checkpoint_name,
        scene_token,
        action_gate_scale,
        action_vector_scale,
        variant,
        method_key,
    ), by_mode in sorted(grouped.items()):
        correct = by_mode.get("correct")
        if correct is None:
            continue
        for mode in ("zero", "shuffled", "reversed_future"):
            other = by_mode.get(mode)
            if other is None:
                continue
            comparisons.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": int(correct["checkpoint_step"]),
                    "scene_token": scene_token,
                    "variant": variant,
                    "method_key": method_key,
                    "method_label": correct.get("method_label", ""),
                    "action_gate_scale": action_gate_scale,
                    "action_vector_scale": action_vector_scale,
                    "comparison_mode": mode,
                    "correct_model_mode": correct["model_mode"],
                    "comparison_model_mode": other["model_mode"],
                    "correct_local_file": correct.get("local_file", ""),
                    "comparison_local_file": other.get("local_file", ""),
                    "correct_remote_relpath": correct["remote_generated_relpath"],
                    "comparison_remote_relpath": other["remote_generated_relpath"],
                }
            )
    return comparisons


def chunked(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(REMOTE_ROOT): quality_volume},
)
def compute_sensitivity_shard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    import subprocess

    import imageio_ffmpeg
    import numpy as np

    comparisons = payload["comparisons"]
    fps = int(payload["fps"])
    width = int(payload["width"])
    height = int(payload["height"])
    context_frames = int(payload["context_frames"])
    total_frames = int(payload["total_frames"])
    spatial_stride = int(payload["spatial_stride"])

    def decode_video_frames(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vf = (
            f"fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            vf,
            "-frames:v",
            str(total_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {path}:\n{proc.stderr.decode('utf-8', errors='replace')}")
        frame_size = width * height * 3
        actual_frames = len(proc.stdout) // frame_size
        if actual_frames < total_frames:
            raise ValueError(f"{path} yielded {actual_frames} frames, expected {total_frames}.")
        raw = np.frombuffer(proc.stdout[: total_frames * frame_size], dtype=np.uint8)
        return raw.reshape(total_frames, height, width, 3)

    def compare_future(correct_path: Path, other_path: Path) -> tuple[float, float]:
        correct = decode_video_frames(correct_path)[context_frames:, ::spatial_stride, ::spatial_stride, :]
        other = decode_video_frames(other_path)[context_frames:, ::spatial_stride, ::spatial_stride, :]
        correct_i = correct.astype(np.int16, copy=False)
        other_i = other.astype(np.int16, copy=False)
        future_rgb_mae = float(np.mean(np.abs(correct_i - other_i)))
        correct_delta = np.diff(correct_i, axis=0)
        other_delta = np.diff(other_i, axis=0)
        future_temporal_delta_mae = float(np.mean(np.abs(correct_delta - other_delta)))
        return future_rgb_mae, future_temporal_delta_mae

    quality_volume.reload()
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        future_rgb_mae, future_temporal_delta_mae = compare_future(
            REMOTE_ROOT / comparison["correct_remote_relpath"],
            REMOTE_ROOT / comparison["comparison_remote_relpath"],
        )
        row = dict(comparison)
        row.update(
            {
                "future_rgb_mae_correct_vs_mode": future_rgb_mae,
                "future_temporal_delta_mae_correct_vs_mode": future_temporal_delta_mae,
            }
        )
        rows.append(row)
    return rows


@app.local_entrypoint()
def main(
    manifest: str,
    output_dir: str,
    run_id: str = "",
    fps: int = 24,
    width: int = 512,
    height: int = 512,
    context_frames: int = CONTEXT_FRAMES,
    total_frames: int = 121,
    spatial_stride: int = 2,
    chunk_size: int = 8,
) -> None:
    manifest_path = Path(manifest)
    output_root = Path(output_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not run_id:
        run_id = "counterfactual_sensitivity_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in {manifest_path}")

    remote_records: list[dict[str, Any]] = []
    uploaded: dict[str, str] = {}
    with quality_volume.batch_upload(force=True) as batch:
        for idx, record in enumerate(records):
            local_file = Path(record["local_file"])
            if not local_file.exists():
                raise FileNotFoundError(local_file)
            key = str(local_file.resolve())
            if key not in uploaded:
                remote_relpath = f"{run_id}/counterfactual/{safe_relpath(local_file, idx)}"
                batch.put_file(local_file, remote_relpath)
                uploaded[key] = remote_relpath
            remote = dict(record)
            remote["remote_generated_relpath"] = uploaded[key]
            remote_records.append(remote)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump({**payload, "records": remote_records}, handle, indent=2, sort_keys=True)
            temp_manifest_path = Path(handle.name)
        batch.put_file(temp_manifest_path, f"{run_id}/manifest.json")

    comparisons = build_comparison_rows(remote_records)
    if not comparisons:
        raise ValueError("No counterfactual comparison rows were found.")
    shards = chunked(comparisons, chunk_size)
    shard_payloads = [
        {
            "comparisons": shard,
            "fps": fps,
            "width": width,
            "height": height,
            "context_frames": context_frames,
            "total_frames": total_frames,
            "spatial_stride": spatial_stride,
        }
        for shard in shards
    ]
    shard_rows = list(compute_sensitivity_shard.map(shard_payloads))
    rows = [row for shard in shard_rows for row in shard]
    rows.sort(
        key=lambda row: (
            str(row.get("variant", "")),
            int(row["checkpoint_step"]),
            float(row["action_gate_scale"]),
            str(row["scene_token"]),
            str(row["comparison_mode"]),
        )
    )

    summary_by_key: dict[tuple[str, str, int, float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        summary_by_key[
            (
                str(row.get("variant", "")),
                str(row.get("method_key", "")),
                int(row["checkpoint_step"]),
                float(row["action_gate_scale"]),
                float(row["action_vector_scale"]),
                str(row["comparison_mode"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (variant, method_key, checkpoint_step, action_gate_scale, action_vector_scale, mode), selected in sorted(summary_by_key.items()):
        summary_rows.append(
            {
                "variant": variant,
                "method_key": method_key,
                "checkpoint_step": checkpoint_step,
                "action_gate_scale": action_gate_scale,
                "action_vector_scale": action_vector_scale,
                "comparison_mode": mode,
                "num_clips": len(selected),
                "mean_future_rgb_mae_correct_vs_mode": mean(
                    [float(row["future_rgb_mae_correct_vs_mode"]) for row in selected]
                ),
                "mean_future_temporal_delta_mae_correct_vs_mode": mean(
                    [float(row["future_temporal_delta_mae_correct_vs_mode"]) for row in selected]
                ),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "counterfactual_sensitivity.csv", rows)
    write_csv(output_root / "counterfactual_sensitivity_summary.csv", summary_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "run_id": run_id,
        "spatial_stride": spatial_stride,
        "num_records": len(records),
        "num_comparisons": len(rows),
        "num_summary_rows": len(summary_rows),
        "num_shards": len(shards),
        "chunk_size": chunk_size,
        "counterfactual_sensitivity_csv": str(output_root / "counterfactual_sensitivity.csv"),
        "counterfactual_sensitivity_summary_csv": str(output_root / "counterfactual_sensitivity_summary.csv"),
    }
    (output_root / "counterfactual_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    main(args.manifest, output_dir="data/benchmarks/counterfactual_sensitivity_modal")
