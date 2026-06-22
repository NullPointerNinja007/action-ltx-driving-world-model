#!/usr/bin/env python3
"""Build top action-alignment showcase side-by-side videos.

The selection is metric-driven: high-action validation windows where the final V4
model with correct actions is closer to GT than wrong-action counterfactuals.
Only the selected MP4s are downloaded from GCS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HANDOFF_GCS = os.environ.get("FINAL_ALIGNMENT_HANDOFF_ROOT", "").rstrip("/")
DATA_GCS = os.environ.get("FINAL_ALIGNMENT_DATA_ROOT", "").rstrip("/")

ALIGNMENT_CSV = Path(
    "data/benchmarks/final_action_alignment_validation_seed231/"
    "primary_subset_metrics/per_clip_action_alignment.csv"
)
MODE_METRICS_CSV = Path(
    "data/benchmarks/final_action_alignment_validation_seed231/"
    "primary_subset_metrics/per_clip_mode_metrics.csv"
)
STRATA_CSV = Path(
    "data/benchmarks/final_action_alignment_validation_seed231/"
    "primary_subset_metrics/action_strata_manifest.csv"
)
GEN_MANIFEST = Path(
    "data/benchmarks/final_action_alignment_validation_seed231/"
    "primary_subset_generation_manifest.json"
)

V4_MODEL = "v4_r64_selected_step018000"
NOACTION_MODEL = "noaction_shifted_step003000"
MODES = ["correct", "zero", "shuffled", "reversed_future"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/final_action_alignment_showcase_top10_seed231",
    )
    parser.add_argument("--num-clips", type=int, default=10)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--panel-height", type=int, default=216)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-frames", type=int, default=121)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def require_remote_roots() -> None:
    if not DATA_GCS or not HANDOFF_GCS:
        raise RuntimeError(
            "Set FINAL_ALIGNMENT_DATA_ROOT and FINAL_ALIGNMENT_HANDOFF_ROOT before downloading showcase assets."
        )


def as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_metrics() -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    strata: dict[str, dict[str, Any]] = {}
    for row in load_csv(STRATA_CSV):
        strata[row["window_id"]] = {
            "is_high_action": as_bool(row["is_high_action"]),
            "is_accelerate": as_bool(row["is_accelerate"]),
            "is_turning": as_bool(row["is_turning"]),
            "is_brake": as_bool(row["is_brake"]),
            "is_low_action": as_bool(row["is_low_action"]),
            "combined_action_score": float(row["combined_action_score"]),
            "speed_delta": float(row["speed_delta"]),
            "mean_yaw_rate": float(row["mean_yaw_rate"]),
            "mean_accel_x": float(row["mean_accel_x"]),
            "future_y_displacement": float(row["future_y_displacement"]),
        }

    mode_metrics: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_csv(MODE_METRICS_CSV):
        key = (row["window_id"], row["model_key"], row["counterfactual_action_mode"])
        mode_metrics[key] = {
            "future_psnr": as_float(row, "future_psnr"),
            "future_ssim": as_float(row, "future_global_ssim"),
            "future_mae": as_float(row, "future_mae"),
            "temporal_error": as_float(row, "temporal_delta_error_mae"),
            "sharpness_ratio": as_float(
                row, "sharpness_ratio_generated_over_reference"
            ),
            "motion_ratio": as_float(row, "motion_ratio_generated_over_reference"),
        }

    alignment: list[dict[str, Any]] = []
    for row in load_csv(ALIGNMENT_CSV):
        wid = row["window_id"]
        st = strata.get(wid)
        v4 = mode_metrics.get((wid, V4_MODEL, "correct"))
        noaction = mode_metrics.get((wid, NOACTION_MODEL, "correct"))
        if not st or not v4 or not noaction:
            continue
        delta_psnr = as_float(row, "delta_psnr")
        delta_ssim = as_float(row, "delta_ssim")
        delta_temporal_error = as_float(row, "delta_temporal_error")
        v4_noaction_delta_psnr = v4["future_psnr"] - noaction["future_psnr"]
        v4_noaction_delta_ssim = v4["future_ssim"] - noaction["future_ssim"]
        v4_noaction_delta_temporal_error = (
            noaction["temporal_error"] - v4["temporal_error"]
        )
        score = (
            30.0 * max(delta_psnr, 0.0)
            + 700.0 * max(delta_ssim, 0.0)
            + 3.0 * max(delta_temporal_error, 0.0)
            + 8.0 * max(v4_noaction_delta_psnr, 0.0)
            + 200.0 * max(v4_noaction_delta_ssim, 0.0)
            + 2.0 * max(v4_noaction_delta_temporal_error, 0.0)
            + 0.25 * as_float(row, "S_rgb")
            + 0.25 * as_float(row, "S_delta")
            + (3.0 if st["is_high_action"] else -4.0)
            + (1.0 if st["is_accelerate"] else 0.0)
            + (1.0 if st["is_turning"] else 0.0)
            - (3.0 if st["is_low_action"] else 0.0)
        )
        # Strong showcase clips should demonstrate correct-vs-wrong advantage.
        if delta_psnr <= 0.0 or delta_ssim <= 0.0:
            score -= 50.0
        if not st["is_high_action"]:
            score -= 25.0

        alignment.append(
            {
                **row,
                **st,
                "delta_psnr": delta_psnr,
                "delta_ssim": delta_ssim,
                "delta_temporal_error": delta_temporal_error,
                "S_rgb": as_float(row, "S_rgb"),
                "S_delta": as_float(row, "S_delta"),
                "v4_noaction_delta_psnr": v4_noaction_delta_psnr,
                "v4_noaction_delta_ssim": v4_noaction_delta_ssim,
                "v4_noaction_delta_temporal_error": v4_noaction_delta_temporal_error,
                "v4_psnr": v4["future_psnr"],
                "noaction_psnr": noaction["future_psnr"],
                "v4_ssim": v4["future_ssim"],
                "noaction_ssim": noaction["future_ssim"],
                "score": score,
            }
        )

    return strata, mode_metrics, alignment


def select_windows(alignment: list[dict[str, Any]], num_clips: int) -> list[dict[str, Any]]:
    candidates = sorted(alignment, key=lambda row: row["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    for row in candidates:
        if row["scenario_id"] in seen_scenarios:
            continue
        if not row["is_high_action"]:
            continue
        if row["delta_psnr"] <= 0 or row["delta_ssim"] <= 0:
            continue
        selected.append(row)
        seen_scenarios.add(row["scenario_id"])
        if len(selected) >= num_clips:
            return selected

    # If uniqueness is too strict, fill remaining with high-scoring clips.
    selected_ids = {row["window_id"] for row in selected}
    for row in candidates:
        if row["window_id"] in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["window_id"])
        if len(selected) >= num_clips:
            return selected
    return selected


def load_manifest_records() -> dict[tuple[str, str, str], dict[str, Any]]:
    data = json.loads(GEN_MANIFEST.read_text())
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in data["records"]:
        records[(row["window_id"], row["model_key"], row["counterfactual_action_mode"])] = row
    return records


def local_rel_for_gcs(gcs_url: str, download_root: Path) -> Path:
    if gcs_url.startswith(HANDOFF_GCS + "/"):
        suffix = gcs_url[len(HANDOFF_GCS) + 1 :]
    elif gcs_url.startswith(DATA_GCS + "/"):
        suffix = "gt/" + gcs_url[len(DATA_GCS) + 1 :]
    else:
        suffix = gcs_url.split("/", 3)[-1]
    return download_root / suffix


def cp_one(gcs_url: str, local_path: Path, force: bool) -> None:
    if local_path.exists() and local_path.stat().st_size > 0 and not force:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_url, str(local_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def download_selected(
    selected: list[dict[str, Any]],
    records: dict[tuple[str, str, str], dict[str, Any]],
    download_root: Path,
    workers: int,
    force: bool,
) -> dict[tuple[str, str], Path]:
    downloads: dict[tuple[str, str], tuple[str, Path]] = {}
    for row in selected:
        wid = row["window_id"]
        noaction = records[(wid, NOACTION_MODEL, "correct")]
        gt_url = f"{DATA_GCS}/{noaction['source_relpath']}"
        downloads[(wid, "gt")] = (gt_url, local_rel_for_gcs(gt_url, download_root))

        no_url = f"{HANDOFF_GCS}/generated/noaction/{noaction['generated_video_relpath']}"
        downloads[(wid, "noaction")] = (
            no_url,
            local_rel_for_gcs(no_url, download_root),
        )

        for mode in MODES:
            rec = records[(wid, V4_MODEL, mode)]
            v4_url = (
                f"{HANDOFF_GCS}/generated/final_action_alignment/"
                f"{rec['generated_video_relpath']}"
            )
            downloads[(wid, f"v4_{mode}")] = (
                v4_url,
                local_rel_for_gcs(v4_url, download_root),
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(cp_one, gcs_url, local_path, force)
            for gcs_url, local_path in downloads.values()
        ]
        for future in as_completed(futures):
            future.result()

    return {key: path for key, (_, path) in downloads.items()}


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    regular_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if bold_path.exists() and regular_path.exists():
        return ImageFont.truetype(str(bold_path), 18), ImageFont.truetype(
            str(regular_path), 12
        )
    return ImageFont.load_default(), ImageFont.load_default()


def panel_from_frame(
    frame: np.ndarray,
    label: str,
    sublabel: str,
    frame_idx: int,
    width: int,
    height: int,
    fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont],
) -> Image.Image:
    bold, regular = fonts
    image = Image.fromarray(frame).convert("RGB").resize((width, height))
    header_height = 48
    panel = Image.new("RGB", (width, height + header_height), (248, 248, 248))
    draw = ImageDraw.Draw(panel)
    panel.paste(image, (0, header_height))
    draw.text((10, 6), label, fill=(10, 10, 10), font=bold)
    phase = "context" if frame_idx < 49 else "future"
    draw.text(
        (10, 29),
        f"frame {frame_idx:03d}/120  {phase}",
        fill=(80, 80, 80),
        font=regular,
    )
    if sublabel:
        draw.text((190, 29), sublabel, fill=(120, 30, 30), font=regular)
    return panel


def render_grid(
    paths: list[Path],
    labels: list[str],
    sublabels: list[str],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
    force: bool,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()
    readers = [imageio.get_reader(str(path), "ffmpeg") for path in paths]
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )
    try:
        for frame_idx in range(total_frames):
            panels = [
                panel_from_frame(
                    reader.get_data(frame_idx),
                    label,
                    sublabel,
                    frame_idx,
                    width,
                    height,
                    fonts,
                )
                for reader, label, sublabel in zip(readers, labels, sublabels)
            ]
            header_height = panels[0].height - height
            grid = Image.new("RGB", (3 * width, 2 * (height + header_height)), "white")
            for i, panel in enumerate(panels[:3]):
                grid.paste(panel, (i * width, 0))
            for i, panel in enumerate(panels[3:]):
                grid.paste(panel, (i * width, height + header_height))
            writer.append_data(np.asarray(grid))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def stratum_label(row: dict[str, Any]) -> str:
    labels = []
    if row["is_accelerate"]:
        labels.append("accelerate")
    if row["is_turning"]:
        labels.append("turn")
    if row["is_brake"]:
        labels.append("brake")
    if not labels:
        labels.append("high-action" if row["is_high_action"] else "other")
    return "+".join(labels)


def main() -> None:
    args = parse_args()
    require_remote_roots()
    output_dir = Path(args.output_dir)
    download_root = output_dir / "downloads"
    side_dir = output_dir / "side_by_side"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, alignment = load_metrics()
    selected = select_windows(alignment, args.num_clips)
    records = load_manifest_records()
    downloaded = download_selected(
        selected,
        records,
        download_root,
        args.download_workers,
        args.force_download,
    )

    manifest_rows = []
    for rank, row in enumerate(selected, start=1):
        wid = row["window_id"]
        paths = [
            downloaded[(wid, "gt")],
            downloaded[(wid, "noaction")],
            downloaded[(wid, "v4_correct")],
            downloaded[(wid, "v4_zero")],
            downloaded[(wid, "v4_shuffled")],
            downloaded[(wid, "v4_reversed_future")],
        ]
        labels = [
            "GT recorded",
            "No-action baseline",
            "V4 correct actions",
            "V4 zero actions",
            "V4 shuffled actions",
            "V4 reversed future",
        ]
        sublabels = [
            "",
            f"PSNR {row['noaction_psnr']:.2f}",
            f"PSNR {row['v4_psnr']:.2f}",
            "wrong action",
            "wrong action",
            "wrong action",
        ]
        output_path = side_dir / (
            f"{rank:02d}_{stratum_label(row)}_{wid}_"
            "gt_noaction_v4correct_wrongactions.mp4"
        )
        render_grid(
            paths,
            labels,
            sublabels,
            output_path,
            args.panel_width,
            args.panel_height,
            args.fps,
            args.total_frames,
            args.force_render,
        )
        manifest_rows.append(
            {
                "rank": rank,
                "window_id": wid,
                "scenario_id": row["scenario_id"],
                "scene_token": row["scene_token"],
                "stratum": stratum_label(row),
                "output_path": str(output_path),
                "S_rgb": row["S_rgb"],
                "S_delta": row["S_delta"],
                "delta_psnr_correct_vs_wrong": row["delta_psnr"],
                "delta_ssim_correct_vs_wrong": row["delta_ssim"],
                "delta_temporal_error_correct_vs_wrong": row["delta_temporal_error"],
                "v4_noaction_delta_psnr": row["v4_noaction_delta_psnr"],
                "v4_noaction_delta_ssim": row["v4_noaction_delta_ssim"],
                "v4_noaction_delta_temporal_error": row[
                    "v4_noaction_delta_temporal_error"
                ],
                "speed_delta": row["speed_delta"],
                "mean_yaw_rate": row["mean_yaw_rate"],
                "mean_accel_x": row["mean_accel_x"],
                "future_y_displacement": row["future_y_displacement"],
                "score": row["score"],
            }
        )

    with (output_dir / "selected_showcase_windows.json").open("w") as f:
        json.dump({"selected": manifest_rows}, f, indent=2)
    with (output_dir / "selected_showcase_windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Selected and rendered {len(manifest_rows)} showcase videos to {side_dir}")
    for row in manifest_rows:
        print(
            f"{row['rank']:02d} {row['window_id']} {row['stratum']} "
            f"S={row['S_rgb']:.3f} dPSNR={row['delta_psnr_correct_vs_wrong']:.4f} "
            f"dSSIM={row['delta_ssim_correct_vs_wrong']:.5f} "
            f"V4-noaction dPSNR={row['v4_noaction_delta_psnr']:.3f}"
        )


if __name__ == "__main__":
    main()
