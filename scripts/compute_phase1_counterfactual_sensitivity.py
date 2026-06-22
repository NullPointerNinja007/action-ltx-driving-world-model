from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v3 as iio


CONTEXT_FRAMES = 49


def compare_future_streaming(correct_path: Path, other_path: Path, spatial_stride: int) -> tuple[float, float]:
    rgb_abs_sum = 0.0
    rgb_count = 0
    delta_abs_sum = 0.0
    delta_count = 0
    prev_correct = None
    prev_other = None
    future_frames = 0

    correct_iter = iio.imiter(correct_path)
    other_iter = iio.imiter(other_path)
    for frame_idx, (correct_frame, other_frame) in enumerate(zip(correct_iter, other_iter)):
        if frame_idx < CONTEXT_FRAMES:
            continue
        correct = correct_frame[::spatial_stride, ::spatial_stride, :].astype(np.int16, copy=False)
        other = other_frame[::spatial_stride, ::spatial_stride, :].astype(np.int16, copy=False)
        if correct.shape != other.shape:
            raise ValueError(
                f"Frame shape mismatch at frame {frame_idx}: {correct_path} has {correct.shape}, "
                f"{other_path} has {other.shape}"
            )
        rgb_abs_sum += float(np.abs(correct - other).sum())
        rgb_count += int(correct.size)
        if prev_correct is not None and prev_other is not None:
            delta_correct = correct - prev_correct
            delta_other = other - prev_other
            delta_abs_sum += float(np.abs(delta_correct - delta_other).sum())
            delta_count += int(correct.size)
        prev_correct = correct
        prev_other = other
        future_frames += 1

    if future_frames <= 0:
        raise ValueError(f"No future frames found after context for {correct_path} and {other_path}")
    if rgb_count <= 0 or delta_count <= 0:
        raise ValueError(f"Insufficient compared pixels for {correct_path} and {other_path}")
    return rgb_abs_sum / rgb_count, delta_abs_sum / delta_count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-stride", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest["records"]
        if row.get("diagnostic_group") == "counterfactual_suite"
        and str(row.get("counterfactual_action_mode")) in {"correct", "zero", "shuffled", "reversed_future"}
    ]
    grouped: dict[tuple[str, str, float, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[
            (
                row["checkpoint_name"],
                row["scene_token"],
                float(row.get("action_gate_scale", 1.0)),
                float(row.get("action_vector_scale", 1.0)),
            )
        ][row["counterfactual_action_mode"]] = row

    out_rows: list[dict[str, Any]] = []

    for (checkpoint_name, scene_token, action_gate_scale, action_vector_scale), by_mode in sorted(grouped.items()):
        correct = by_mode.get("correct")
        if correct is None:
            continue
        for mode in ("zero", "shuffled", "reversed_future"):
            other = by_mode.get(mode)
            if other is None:
                continue
            future_rgb_mae, future_temporal_delta_mae = compare_future_streaming(
                Path(correct["local_file"]),
                Path(other["local_file"]),
                args.spatial_stride,
            )
            out_rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_step": int(correct["checkpoint_step"]),
                    "scene_token": scene_token,
                    "action_gate_scale": action_gate_scale,
                    "action_vector_scale": action_vector_scale,
                    "comparison_mode": mode,
                    "correct_model_mode": correct["model_mode"],
                    "comparison_model_mode": other["model_mode"],
                    "future_rgb_mae_correct_vs_mode": future_rgb_mae,
                    "future_temporal_delta_mae_correct_vs_mode": future_temporal_delta_mae,
                    "correct_local_file": correct["local_file"],
                    "comparison_local_file": other["local_file"],
                }
            )

    write_csv(args.output_dir / "counterfactual_sensitivity.csv", out_rows)

    summary_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in out_rows:
        by_key[
            (
                int(row["checkpoint_step"]),
                float(row["action_gate_scale"]),
                float(row["action_vector_scale"]),
                row["comparison_mode"],
            )
        ].append(row)
    for (checkpoint_step, action_gate_scale, action_vector_scale, mode), selected in sorted(by_key.items()):
        summary_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "action_gate_scale": action_gate_scale,
                "action_vector_scale": action_vector_scale,
                "comparison_mode": mode,
                "num_clips": len(selected),
                "mean_future_rgb_mae_correct_vs_mode": float(
                    np.mean([float(row["future_rgb_mae_correct_vs_mode"]) for row in selected])
                ),
                "mean_future_temporal_delta_mae_correct_vs_mode": float(
                    np.mean([float(row["future_temporal_delta_mae_correct_vs_mode"]) for row in selected])
                ),
            }
        )
    write_csv(args.output_dir / "counterfactual_sensitivity_summary.csv", summary_rows)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "spatial_stride": args.spatial_stride,
        "rows": len(out_rows),
        "summary_rows": len(summary_rows),
    }
    (args.output_dir / "counterfactual_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
