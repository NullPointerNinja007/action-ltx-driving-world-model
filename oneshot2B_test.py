from pathlib import Path
import shutil
import subprocess
from datetime import datetime

import modal

app = modal.App("ltx-video-run")

vol = modal.Volume.from_name("models")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "huggingface_hub",
        "av",
        "imageio",
        "imageio-ffmpeg",
        "imageio[ffmpeg]",
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video",
        "cd /workspace/LTX-Video && python -m pip install -e '.[inference-script]'",
    )
)

REPO = Path("/workspace/LTX-Video")
DATA = Path("/data")

CKPT_2B = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER = "ltxv-spatial-upscaler-0.9.8.safetensors"


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/data": vol},
)
def run_ltx(
    input_video_relpath: str,
    prompt: str,
    mode: str = "condition",  # "condition" or "v2v"
    num_frames: int = 97,
    seed: int = 0,
):
    """
    input_video_relpath example:
      videos/waymo/clip_0001.mp4

    mode="condition":
      uses the input video as conditioning for future/continuation generation.

    mode="v2v":
      uses LTX's input_media_path video-to-video mode.
    """

    input_video = DATA / input_video_relpath
    if not input_video.exists():
        raise FileNotFoundError(f"Missing input video: {input_video}")

    # LTX config expects checkpoint filenames in the repo root.
    for fname in [CKPT_2B, UPSCALER]:
        src = DATA / "ltx" / fname
        dst = REPO / fname

        if not src.exists():
            raise FileNotFoundError(f"Missing checkpoint in volume: {src}")

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        dst.symlink_to(src)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "inference.py",
        "--prompt",
        prompt,
        "--height",
        "512",
        "--width",
        "512",
        "--num_frames",
        str(num_frames),
        "--seed",
        str(seed),
        "--pipeline_config",
        "configs/ltxv-2b-0.9.8-distilled.yaml",
        "--output_path",
        str(out_dir),
    ]

    if mode == "condition":
        cmd += [
            "--conditioning_media_paths",
            str(input_video),
            "--conditioning_start_frames",
            "0",
        ]
    elif mode == "v2v":
        cmd += [
            "--input_media_path",
            str(input_video),
        ]
    else:
        raise ValueError("mode must be either 'condition' or 'v2v'")

    subprocess.run(cmd, cwd=str(REPO), check=True)

    mp4s = sorted(out_dir.rglob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"No mp4 generated in {out_dir}")

    clean_output = DATA / "outputs" / f"{run_name}_{mode}_clean.mp4"
    shutil.copy2(mp4s[0], clean_output)

    vol.commit()

    return str(clean_output)


@app.local_entrypoint()
def main(
    input_video_relpath: str,
    mode: str = "condition",
    num_frames: int = 97,
    seed: int = 0,
):
    prompt = (
        "Forward-facing autonomous driving dashcam video. Continue the road scene "
        "naturally with realistic ego-motion, stable road geometry, consistent lane "
        "markings, vehicles, buildings, sidewalks, traffic lights, and no camera cuts."
    )

    output_path = run_ltx.remote(
        input_video_relpath=input_video_relpath,
        prompt=prompt,
        mode=mode,
        num_frames=num_frames,
        seed=seed,
    )

    print("Saved output in Modal volume:")
    print(output_path)
