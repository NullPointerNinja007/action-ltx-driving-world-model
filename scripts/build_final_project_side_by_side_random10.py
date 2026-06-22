#!/usr/bin/env python3
"""Build final-project side-by-side videos for random validation windows.

This script uses the final action-alignment primary-subset manifest, downloads
only the required MP4s from GCS if missing, and writes labeled 2x3 comparison
videos:

  GT | No-action | V4 correct
  V4 zero | V4 shuffled | V4 reversed future
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DATA_PREFIX = os.environ.get("FINAL_ALIGNMENT_DATA_ROOT", "").rstrip("/")
HANDOFF_PREFIX = os.environ.get("FINAL_ALIGNMENT_HANDOFF_ROOT", "").rstrip("/")

PANEL_SPECS = [
    ("gt", "GT recorded", None, None),
    ("noaction", "No-action baseline", "noaction_shifted_step003000", "correct"),
    ("v4_correct", "V4 correct actions", "v4_r64_selected_step018000", "correct"),
    ("v4_zero", "V4 zero actions", "v4_r64_selected_step018000", "zero"),
    ("v4_shuffled", "V4 shuffled actions", "v4_r64_selected_step018000", "shuffled"),
    (
        "v4_reversed",
        "V4 reversed future actions",
        "v4_r64_selected_step018000",
        "reversed_future",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/benchmarks/final_action_alignment_validation_seed231/"
        "primary_subset_generation_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/final_project_side_by_side_random10_seed231",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--panel-height", type=int, default=216)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-frames", type=int, default=121)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def require_remote_roots() -> None:
    if not DATA_PREFIX or not HANDOFF_PREFIX:
        raise RuntimeError(
            "Set FINAL_ALIGNMENT_DATA_ROOT and FINAL_ALIGNMENT_HANDOFF_ROOT before downloading showcase assets."
        )


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def index_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record["window_id"],
            record["model_key"],
            record["counterfactual_action_mode"],
        )
        out[key] = record
    return out


def select_windows(records: list[dict[str, Any]], count: int, seed: int) -> list[str]:
    available = sorted(
        {
            r["window_id"]
            for r in records
            if r["model_key"] == "v4_r64_selected_step018000"
            and r["counterfactual_action_mode"] == "correct"
        }
    )
    rng = random.Random(seed)
    rng.shuffle(available)
    return available[:count]


def gcs_uri_for_record(record: dict[str, Any]) -> str:
    rel = record["generated_video_relpath"]
    if record["generated_volume_key"] == "noaction":
        return f"{HANDOFF_PREFIX}/generated/noaction/{rel}"
    return f"{HANDOFF_PREFIX}/generated/final_action_alignment/{rel}"


def local_path_for_record(output_dir: Path, record: dict[str, Any]) -> Path:
    rel = Path(record["generated_video_relpath"])
    return output_dir / "downloads" / record["generated_volume_key"] / rel


def local_path_for_source(output_dir: Path, source_relpath: str) -> Path:
    return output_dir / "downloads" / "gt" / source_relpath


def download_one(uri: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    subprocess.run(["gcloud", "storage", "cp", uri, str(tmp)], check=True)
    tmp.replace(dst)


def download_inputs(
    jobs: list[tuple[str, Path]], workers: int, force: bool = False
) -> None:
    unique_jobs = sorted(set(jobs), key=lambda x: str(x[1]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, uri, dst, force) for uri, dst in unique_jobs]
        for future in as_completed(futures):
            future.result()


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    regular_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            bold = ImageFont.truetype(c, 18)
            break
    else:
        bold = ImageFont.load_default()
    for c in regular_candidates:
        if Path(c).exists():
            regular = ImageFont.truetype(c, 12)
            break
    else:
        regular = ImageFont.load_default()
    return bold, regular


def panel_from_frame(
    frame: np.ndarray,
    label: str,
    panel_width: int,
    panel_height: int,
    frame_idx: int,
    total_frames: int,
    fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont],
) -> Image.Image:
    bold, regular = fonts
    image = Image.fromarray(frame).convert("RGB").resize((panel_width, panel_height))
    canvas = Image.new("RGB", (panel_width, panel_height + 48), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image, (0, 48))
    draw.rectangle((0, 0, panel_width, 47), fill=(250, 250, 250))
    draw.text((10, 6), label, fill=(10, 10, 10), font=bold)
    phase = "context" if frame_idx < 49 else "future"
    draw.text(
        (10, 29),
        f"frame {frame_idx:03d}/{total_frames - 1:03d}  {phase}",
        fill=(80, 80, 80),
        font=regular,
    )
    return canvas


def render_grid_video(
    panel_paths: list[Path],
    labels: list[str],
    output_path: Path,
    panel_width: int,
    panel_height: int,
    fps: int,
    total_frames: int,
    force: bool = False,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()
    readers = [imageio.get_reader(str(p), "ffmpeg") for p in panel_paths]
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )
    try:
        for frame_idx in range(total_frames):
            panels: list[Image.Image] = []
            for reader, label in zip(readers, labels):
                frame = reader.get_data(frame_idx)
                panels.append(
                    panel_from_frame(
                        frame,
                        label,
                        panel_width,
                        panel_height,
                        frame_idx,
                        total_frames,
                        fonts,
                    )
                )
            row1 = Image.new("RGB", (panel_width * 3, panel_height + 48), "white")
            row2 = Image.new("RGB", (panel_width * 3, panel_height + 48), "white")
            for idx, panel in enumerate(panels[:3]):
                row1.paste(panel, (idx * panel_width, 0))
            for idx, panel in enumerate(panels[3:]):
                row2.paste(panel, (idx * panel_width, 0))
            grid = Image.new("RGB", (panel_width * 3, (panel_height + 48) * 2), "white")
            grid.paste(row1, (0, 0))
            grid.paste(row2, (0, panel_height + 48))
            writer.append_data(np.asarray(grid))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def main() -> None:
    args = parse_args()
    require_remote_roots()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = load_manifest(manifest_path)
    records = manifest["records"]
    record_index = index_records(records)
    windows = select_windows(records, args.count, args.seed)

    download_jobs: list[tuple[str, Path]] = []
    render_jobs: list[dict[str, Any]] = []

    for ordinal, window_id in enumerate(windows, start=1):
        source_record = record_index[
            (window_id, "noaction_shifted_step003000", "correct")
        ]
        panel_paths: list[Path] = []
        labels: list[str] = []

        source_relpath = source_record["source_relpath"]
        source_uri = f"{DATA_PREFIX}/{source_relpath}"
        source_local = local_path_for_source(output_dir, source_relpath)
        download_jobs.append((source_uri, source_local))
        panel_paths.append(source_local)
        labels.append(PANEL_SPECS[0][1])

        for _, label, model_key, mode in PANEL_SPECS[1:]:
            assert model_key is not None and mode is not None
            rec = record_index[(window_id, model_key, mode)]
            uri = gcs_uri_for_record(rec)
            local = local_path_for_record(output_dir, rec)
            download_jobs.append((uri, local))
            panel_paths.append(local)
            labels.append(label)

        scene_token = source_record["scene_token"]
        window_idx = source_record["window_idx"]
        output_path = (
            output_dir
            / "side_by_side"
            / (
                f"{ordinal:02d}_{window_id}__gt_noaction_"
                "v4correct_v4zero_v4shuffled_v4reversed.mp4"
            )
        )
        render_jobs.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "scene_token": scene_token,
                "window_idx": window_idx,
                "source_relpath": source_relpath,
                "output_path": str(output_path),
                "panels": [
                    {"label": label, "path": str(path)}
                    for label, path in zip(labels, panel_paths)
                ],
            }
        )

    download_inputs(download_jobs, args.download_workers, args.force_download)

    for job in render_jobs:
        render_grid_video(
            [Path(p["path"]) for p in job["panels"]],
            [p["label"] for p in job["panels"]],
            Path(job["output_path"]),
            args.panel_width,
            args.panel_height,
            args.fps,
            args.total_frames,
            args.force_render,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest_side_by_side_random10.json").open("w") as f:
        json.dump(
            {
                "count": len(render_jobs),
                "seed": args.seed,
                "source_manifest": str(manifest_path),
                "panel_order": [spec[1] for spec in PANEL_SPECS],
                "videos": render_jobs,
            },
            f,
            indent=2,
        )
    with (output_dir / "manifest_side_by_side_random10.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ordinal",
                "window_id",
                "scene_token",
                "window_idx",
                "source_relpath",
                "output_path",
            ],
        )
        writer.writeheader()
        for job in render_jobs:
            writer.writerow(
                {
                    "ordinal": job["ordinal"],
                    "window_id": job["window_id"],
                    "scene_token": job["scene_token"],
                    "window_idx": job["window_idx"],
                    "source_relpath": job["source_relpath"],
                    "output_path": job["output_path"],
                }
            )

    print(f"Wrote {len(render_jobs)} side-by-side videos to {output_dir / 'side_by_side'}")


if __name__ == "__main__":
    main()
