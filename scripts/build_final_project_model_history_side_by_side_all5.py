#!/usr/bin/env python3
"""Build final-project model-history side-by-side videos on the all5 clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENES = [
    "0081c4821701",
    "00c73f2a3515",
    "00f755ca8954",
    "01d758be9706",
    "023de465f73a",
]


PANELS = [
    (
        "gt",
        "GT recorded",
        "data/action_checkpoint_sweep_generated/adaln/"
        "adaln_action_full7992_step002500_seed231_all5/"
        "source_minterpolate_24fps_full20s/*{scene}*.mp4",
    ),
    (
        "noaction",
        "No-action baseline",
        "data/noaction_cleanctx_shifted_timestep_ablation_generated/"
        "distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs/"
        "shifted_lognormal_step003000_seed231_all5_cleanctx/"
        "generated_lora_checkpoints/step_003000/scene_{scene}_*/*.mp4",
    ),
    (
        "adaln",
        "AdaLN action",
        "data/action_checkpoint_sweep_generated/adaln/"
        "adaln_action_full7992_step002500_seed231_all5/"
        "generated_action_lora/step_002500/scene_{scene}_*/*.mp4",
    ),
    (
        "hfteacher_v2",
        "HF-teacher v2",
        "data/corrected_global_mlp_hfteacher_v2_generated/"
        "distilled098_framebottleneck_hfteacher_v2_action_lora_24fps_minterpolate_seed231_runs/"
        "hfteacher_v2_step003000_g1p000_seed231_all5/"
        "generated_action_lora/step_003000/scene_{scene}_*/*.mp4",
    ),
    (
        "v4_actionstrong",
        "V4 action-strong",
        "data/final_h100_v4_campaign_generated/"
        "distilled098_full112_lowfreq_motion_v4_action_lora_24fps_minterpolate_seed231_runs/"
        "v4_actionstrong_text_step007992_g1p000_seed231_all5/"
        "generated_action_lora/step_007992/scene_{scene}_*/*.mp4",
    ),
    (
        "v4_final",
        "Final V4 correct",
        "data/b200_v4_three_epoch_continuation_generated/"
        "distilled098_full112_lowfreq_motion_v4_b200_three_epoch_24fps_minterpolate_seed231_runs/"
        "v4_main_text_r64_3epoch_step018000_g1p000_seed231_all5/"
        "generated_action_lora/step_018000/scene_{scene}_*/*.mp4",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/final_project_model_history_side_by_side_all5_seed231",
    )
    parser.add_argument("--panel-width", type=int, default=384)
    parser.add_argument("--panel-height", type=int, default=216)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total-frames", type=int, default=121)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_one(pattern: str, scene: str) -> Path:
    matches = sorted(Path(".").glob(pattern.format(scene=scene)))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for scene={scene}, pattern={pattern}, "
            f"found={len(matches)}"
        )
    return matches[0]


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


def render(
    scene: str,
    panel_paths: list[Path],
    labels: list[str],
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
    readers = [imageio.get_reader(str(path), "ffmpeg") for path in panel_paths]
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
                    frame_idx,
                    total_frames,
                    width,
                    height,
                    fonts,
                )
                for reader, label in zip(readers, labels)
            ]
            grid = Image.new("RGB", (3 * width, 2 * (height + 48)), "white")
            for i, panel in enumerate(panels[:3]):
                grid.paste(panel, (i * width, 0))
            for i, panel in enumerate(panels[3:]):
                grid.paste(panel, (i * width, height + 48))
            writer.append_data(np.asarray(grid))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    jobs = []
    for scene in SCENES:
        panel_paths = [resolve_one(pattern, scene) for _, _, pattern in PANELS]
        labels = [label for _, label, _ in PANELS]
        output_path = (
            output_dir
            / "side_by_side"
            / (
                f"scene_{scene}_gt_noaction_adaln_hfteacherv2_"
                "v4actionstrong_finalv4correct.mp4"
            )
        )
        render(
            scene,
            panel_paths,
            labels,
            output_path,
            args.panel_width,
            args.panel_height,
            args.fps,
            args.total_frames,
            args.force,
        )
        jobs.append(
            {
                "scene": scene,
                "output_path": str(output_path),
                "panels": [
                    {"key": key, "label": label, "path": str(path)}
                    for (key, label, _), path in zip(PANELS, panel_paths)
                ],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest_model_history_side_by_side_all5.json").open("w") as f:
        json.dump({"videos": jobs, "panel_order": [p[1] for p in PANELS]}, f, indent=2)
    with (output_dir / "manifest_model_history_side_by_side_all5.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "output_path"])
        writer.writeheader()
        for job in jobs:
            writer.writerow({"scene": job["scene"], "output_path": job["output_path"]})

    print(f"Wrote {len(jobs)} side-by-side videos to {output_dir / 'side_by_side'}")


if __name__ == "__main__":
    main()
