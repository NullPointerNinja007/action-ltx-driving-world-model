from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "inference_input_clips" / "interpolated_24fps_waymo_full20s"
MANIFEST = (
    ROOT
    / "data"
    / "benchmarks"
    / "noaction_shifted_timestep_longer_seed231_all5"
    / "manifest_noaction_shifted_timestep_longer_seed231_all5.json"
)
OUT_DIR = ROOT / "data" / "noaction_shifted_timestep_longer_side_by_side_seed231"

FPS = 24
TOTAL_FRAMES = 121
PANEL_W = 384
PANEL_H = 384

MODES = [
    ("ground_truth", "GT"),
    ("shifted_lognormal_base", "base"),
    ("uniform_cleanctx_step002000", "uniform 2k"),
    ("shifted_lognormal_step003000", "shifted 3k"),
    ("shifted_lognormal_step005000", "shifted 5k"),
]


def ffmpeg_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_mode_scene: dict[tuple[str, str], dict] = {}
    for record in manifest["records"]:
        by_mode_scene[(record["model_mode"], record["scene_token"])] = record

    scenes = sorted({record["scene_token"] for record in manifest["records"]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    outputs = []

    for scene in scenes:
        base_record = by_mode_scene[("shifted_lognormal_base", scene)]
        source_path = SOURCE_DIR / base_record["source_filename"]
        inputs = [source_path]
        for mode, _label in MODES[1:]:
            record = by_mode_scene[(mode, scene)]
            inputs.append(Path(record["local_file"]))

        cmd = [ffmpeg, "-hide_banner", "-y"]
        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(path)
            cmd.extend(["-i", str(path)])

        filter_parts = []
        for idx, (_mode, label) in enumerate(MODES):
            filter_parts.append(
                f"[{idx}:v]fps={FPS},trim=start_frame=0:end_frame={TOTAL_FRAMES},"
                f"setpts=PTS-STARTPTS,"
                f"scale={PANEL_W}:{PANEL_H}:force_original_aspect_ratio=increase,"
                f"crop={PANEL_W}:{PANEL_H}[v{idx}]"
            )
        layout = "|".join(f"{idx * PANEL_W}_0" for idx in range(len(MODES)))
        filter_parts.append(
            "".join(f"[v{idx}]" for idx in range(len(MODES)))
            + f"xstack=inputs={len(MODES)}:layout={layout}:fill=black[v]"
        )

        out_path = OUT_DIR / f"{scene}_gt_base_uniform2k_shifted3k_shifted5k_seed231.mp4"
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
    summary_path.write_text(json.dumps({"outputs": outputs, "modes": MODES}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT_DIR), "num_outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
