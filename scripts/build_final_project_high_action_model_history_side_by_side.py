#!/usr/bin/env python3
"""Build high-action model-history side-by-side videos for final presentation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SELECTED_CSV = Path(
    "data/final_action_alignment_showcase_top10_seed231/selected_showcase_windows.csv"
)
SHOWCASE_DOWNLOADS = Path("data/final_action_alignment_showcase_top10_seed231/downloads")
HISTORY_DOWNLOADS = Path(
    "data/final_project_high_action_model_history_seed231/downloads/generated"
)

PANELS = [
    ("gt", "GT recorded"),
    ("noaction", "No-action baseline"),
    ("adaln", "AdaLN action"),
    ("hfteacher_v2", "HF-teacher v2"),
    ("v4_actionstrong", "V4 action-strong"),
    ("v4_final", "Final V4 correct"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/final_project_high_action_model_history_seed231",
    )
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--panel-height", type=int, default=216)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-frames", type=int, default=121)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_selected() -> list[dict[str, str]]:
    with SELECTED_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_one(pattern: str) -> Path:
    matches = sorted(Path(".").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found={len(matches)}")
    return matches[0]


def paths_for_row(row: dict[str, str]) -> dict[str, Path]:
    scene = row["scene_token"]
    scenario = row["scenario_id"]
    window = row["window_id"]
    return {
        "gt": resolve_one(
            str(SHOWCASE_DOWNLOADS / "gt" / "mp4_windows" / "val" / scenario / f"{window}.mp4")
        ),
        "noaction": resolve_one(
            str(
                SHOWCASE_DOWNLOADS
                / "generated"
                / "noaction"
                / "**"
                / f"scene_{scene}_49ctx_72future_lora_step003000_seed231_24fps_121f.mp4"
            )
        ),
        "adaln": resolve_one(
            str(
                HISTORY_DOWNLOADS
                / "adaln"
                / "**"
                / f"scene_{scene}_49ctx_72future_action_lora_step002500_correct_g1p000_v1p000_seed231_24fps_121f.mp4"
            )
        ),
        "hfteacher_v2": resolve_one(
            str(
                HISTORY_DOWNLOADS
                / "hfteacher_v2"
                / "**"
                / f"scene_{scene}_49ctx_72future_action_lora_step003000_correct_g1p000_v1p000_seed231_24fps_121f.mp4"
            )
        ),
        "v4_actionstrong": resolve_one(
            str(
                HISTORY_DOWNLOADS
                / "v4_actionstrong"
                / "**"
                / f"scene_{scene}_49ctx_72future_action_lora_step007992_correct_g1p000_v1p000_seed231_24fps_121f.mp4"
            )
        ),
        "v4_final": resolve_one(
            str(
                SHOWCASE_DOWNLOADS
                / "generated"
                / "final_action_alignment"
                / "**"
                / f"scene_{scene}_49ctx_72future_action_lora_step018000_correct_g1p000_v1p000_seed231_24fps_121f.mp4"
            )
        ),
    }


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    regular_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if bold_path.exists() and regular_path.exists():
        return ImageFont.truetype(str(bold_path), 18), ImageFont.truetype(str(regular_path), 12)
    return ImageFont.load_default(), ImageFont.load_default()


def panel_from_frame(
    frame: np.ndarray,
    label: str,
    frame_idx: int,
    total_frames: int,
    width: int,
    height: int,
    fonts: tuple[ImageFont.ImageFont, ImageFont.ImageFont],
) -> Image.Image:
    bold, regular = fonts
    image = Image.fromarray(frame).convert("RGB").resize((width, height))
    panel = Image.new("RGB", (width, height + 48), (248, 248, 248))
    draw = ImageDraw.Draw(panel)
    panel.paste(image, (0, 48))
    draw.text((10, 6), label, fill=(10, 10, 10), font=bold)
    phase = "context" if frame_idx < 49 else "future"
    draw.text(
        (10, 29),
        f"frame {frame_idx:03d}/{total_frames - 1:03d}  {phase}",
        fill=(80, 80, 80),
        font=regular,
    )
    return panel


def render_side_by_side(
    row: dict[str, str],
    video_paths: dict[str, Path],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    total_frames: int,
) -> None:
    fonts = load_fonts()
    readers = {key: imageio.get_reader(path) for key, path in video_paths.items()}
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
            for frame_idx in range(total_frames):
                panels = []
                for key, label in PANELS:
                    frame = readers[key].get_data(frame_idx)
                    panels.append(
                        panel_from_frame(frame, label, frame_idx, total_frames, width, height, fonts)
                    )
                canvas = Image.new("RGB", (width * len(panels), height + 48), (255, 255, 255))
                for i, panel in enumerate(panels):
                    canvas.paste(panel, (i * width, 0))
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers.values():
            reader.close()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    side_root = out_root / "side_by_side"
    selected = load_selected()
    manifest: list[dict[str, str]] = []

    for row in selected:
        rank = int(row["rank"])
        window_id = row["window_id"]
        stratum = row["stratum"].replace("+", "_")
        output_path = side_root / (
            f"{rank:02d}_{stratum}_{window_id}_gt_noaction_adaln_"
            "hfteacherv2_v4actionstrong_finalv4correct.mp4"
        )
        paths = paths_for_row(row)
        if output_path.exists() and not args.force:
            pass
        else:
            render_side_by_side(
                row,
                paths,
                output_path,
                width=args.panel_width,
                height=args.panel_height,
                fps=args.fps,
                total_frames=args.total_frames,
            )
        manifest.append(
            {
                "rank": row["rank"],
                "window_id": window_id,
                "scene_token": row["scene_token"],
                "stratum": row["stratum"],
                "side_by_side_path": output_path.as_posix(),
                **{f"{key}_path": path.as_posix() for key, path in paths.items()},
            }
        )

    manifest_path = out_root / "manifest_high_action_model_history_side_by_side.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    preview_src = side_root / Path(manifest[0]["side_by_side_path"]).name
    preview_path = out_root / "preview_rank01_frame072.png"
    reader = imageio.get_reader(preview_src)
    try:
        frame = reader.get_data(72)
        Image.fromarray(frame).save(preview_path)
    finally:
        reader.close()

    print(json.dumps({"videos": len(manifest), "manifest": manifest_path.as_posix(), "preview": preview_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
