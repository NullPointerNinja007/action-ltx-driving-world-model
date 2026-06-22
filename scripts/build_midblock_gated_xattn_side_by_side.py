from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
NOACTION_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "noaction_shifted_timestep_longer_seed231_all5"
    / "manifest_noaction_shifted_timestep_longer_seed231_all5.json"
)
FRAME_ACTION_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
    / "manifest_frame_action_2epoch_checkpoint_sweep_seed231_all5.json"
)
MIDBLOCK_MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
    / "manifest_midblock_gated_xattn_checkpoint_sweep_seed231_all5.json"
)
OUT_DIR = ROOT / "data" / "midblock_gated_xattn_side_by_side_seed231_all5"

FPS = 24
TOTAL_FRAMES = 121
PANEL_W = 320
PANEL_H = 320

PANELS = [
    ("ground_truth", "GT"),
    ("shifted_lognormal_step003000", "no-action shifted 3k"),
    ("frame_transformer_step014000", "old frame-xf 14k"),
    ("frame_adaln_step012000", "old AdaLN 12k"),
    ("frame_midblock_gated_xattn_step000100", "midblock gate 100"),
]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def load_records(path: Path) -> dict[tuple[str, str], dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {(record["model_mode"], record["scene_token"]): record for record in manifest["records"]}


def main() -> None:
    noaction = load_records(NOACTION_MANIFEST)
    frame_action = load_records(FRAME_ACTION_MANIFEST)
    midblock = load_records(MIDBLOCK_MANIFEST)
    combined = {**noaction, **frame_action, **midblock}

    scenes = sorted({scene for mode, scene in midblock if mode == "frame_midblock_gated_xattn_step000100"})
    if not scenes:
        raise RuntimeError("No mid-block gated xattn step 100 scenes found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    outputs: list[str] = []

    for scene in scenes:
        source_record = combined[("shifted_lognormal_step003000", scene)]
        inputs = [SOURCE_DIR / source_record["source_filename"]]
        for mode, _label in PANELS[1:]:
            record = combined[(mode, scene)]
            inputs.append(Path(record["local_file"]))

        cmd = [ffmpeg, "-hide_banner", "-y"]
        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(path)
            cmd.extend(["-i", str(path)])

        filter_parts = []
        for idx, (_mode, _label) in enumerate(PANELS):
            filter_parts.append(
                f"[{idx}:v]fps={FPS},trim=start_frame=0:end_frame={TOTAL_FRAMES},"
                f"setpts=PTS-STARTPTS,"
                f"scale={PANEL_W}:{PANEL_H}:force_original_aspect_ratio=increase,"
                f"crop={PANEL_W}:{PANEL_H}[v{idx}]"
            )
        layout = "|".join(f"{idx * PANEL_W}_0" for idx in range(len(PANELS)))
        filter_parts.append(
            "".join(f"[v{idx}]" for idx in range(len(PANELS)))
            + f"xstack=inputs={len(PANELS)}:layout={layout}:fill=black[v]"
        )

        out_path = OUT_DIR / f"{scene}_gt_noaction_oldxf_oldadaln_midblockgate_seed231.mp4"
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[v]",
                "-frames:v",
                str(TOTAL_FRAMES),
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        run(cmd)
        outputs.append(str(out_path.relative_to(ROOT)))

    summary_path = OUT_DIR / "manifest_side_by_side.json"
    summary_path.write_text(
        json.dumps({"outputs": outputs, "panels": PANELS, "created_from": {
            "noaction": str(NOACTION_MANIFEST.relative_to(ROOT)),
            "frame_action": str(FRAME_ACTION_MANIFEST.relative_to(ROOT)),
            "midblock": str(MIDBLOCK_MANIFEST.relative_to(ROOT)),
        }}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUT_DIR), "num_outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
