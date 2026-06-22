from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "ltx-waymo-training-volume-verify"
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
MODELS_VOLUME_NAME = "models"
CHECKPOINT_VOLUME_NAME = "ltx2b-dev-waymo24fps-visual-lora-r16-checkpoints"
LATENT_PREFIXES = ["latents", "latents_distilled098"]

DATA_ROOT = Path("/data")
MODELS_ROOT = Path("/models")
CKPT_ROOT = Path("/checkpoints")

MODEL_FILES = [
    "ltx/ltxv-2b-0.9.6-dev-04-25.safetensors",
    "ltx/ltxv-spatial-upscaler-0.9.8.safetensors",
]


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10")


def replace_latent_prefix(latent_relpath: str, latent_prefix: str) -> str:
    parts = Path(latent_relpath).parts
    if not parts:
        raise ValueError("Empty latent_relpath")
    return str(Path(latent_prefix, *parts[1:]))


def verify_split(split: str, latent_prefix: str = "") -> dict[str, Any]:
    manifest = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    missing_mp4: list[str] = []
    missing_latent: list[str] = []
    scenarios = set()
    frame_counts = set()
    context_counts = set()
    future_counts = set()
    fps_values = set()
    for row in rows:
        scenarios.add(row["scenario_id"])
        frame_counts.add(int(row["num_frames"]))
        context_counts.add(int(row["context_frames"]))
        future_counts.add(int(row["future_frames"]))
        fps_values.add(int(row["fps"]))
        if not (DATA_ROOT / row["mp4_relpath"]).exists():
            missing_mp4.append(row["mp4_relpath"])
        latent_relpath = replace_latent_prefix(row["latent_relpath"], latent_prefix) if latent_prefix else row["latent_relpath"]
        if not (DATA_ROOT / latent_relpath).exists():
            missing_latent.append(latent_relpath)

    return {
        "rows": len(rows),
        "scenarios": len(scenarios),
        "fps_values": sorted(fps_values),
        "frame_counts": sorted(frame_counts),
        "context_counts": sorted(context_counts),
        "future_counts": sorted(future_counts),
        "missing_mp4_count": len(missing_mp4),
        "missing_latent_count": len(missing_latent),
        "missing_mp4_examples": missing_mp4[:5],
        "missing_latent_examples": missing_latent[:5],
        "latent_prefix": latent_prefix or "manifest_default",
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=20 * 60,
    volumes={
        str(DATA_ROOT): data_volume,
        str(MODELS_ROOT): models_volume,
        str(CKPT_ROOT): checkpoint_volume,
    },
)
def verify() -> dict[str, Any]:
    data_volume.reload()
    models_volume.reload()
    checkpoint_volume.reload()
    model_files = {}
    for relpath in MODEL_FILES:
        path = MODELS_ROOT / relpath
        model_files[relpath] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }

    return {
        "data_volume": DATA_VOLUME_NAME,
        "models_volume": MODELS_VOLUME_NAME,
        "checkpoint_volume": CHECKPOINT_VOLUME_NAME,
        "latent_cache_by_prefix": {
            latent_prefix: {
                "train": verify_split("train", latent_prefix),
                "val": verify_split("val", latent_prefix),
            }
            for latent_prefix in LATENT_PREFIXES
        },
        "model_files": model_files,
        "checkpoint_entries": sorted(path.name for path in CKPT_ROOT.iterdir()) if CKPT_ROOT.exists() else [],
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(verify.remote(), indent=2, sort_keys=True))
