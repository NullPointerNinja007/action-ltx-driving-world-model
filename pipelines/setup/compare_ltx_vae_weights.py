from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "ltx-vae-weight-compare"
MODELS_VOLUME_NAME = "models"
MODELS_ROOT = Path("/models")

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME)

image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy", "safetensors", "torch")


def tensors_equal(left_reader, right_reader, key: str) -> bool:
    import torch

    left = left_reader.get_tensor(key).detach().cpu()
    right = right_reader.get_tensor(key).detach().cpu()
    return bool(torch.equal(left, right))


@app.function(image=image, cpu=4, memory=32768, timeout=60 * 60, volumes={str(MODELS_ROOT): models_volume})
def compare(
    left_name: str = "ltxv-2b-0.9.6-dev-04-25.safetensors",
    right_name: str = "ltxv-2b-0.9.8-distilled.safetensors",
) -> dict:
    from safetensors import safe_open

    left_path = MODELS_ROOT / "ltx" / left_name
    right_path = MODELS_ROOT / "ltx" / right_name
    if not left_path.exists():
        raise FileNotFoundError(left_path)
    if not right_path.exists():
        raise FileNotFoundError(right_path)

    with safe_open(left_path, framework="pt", device="cpu") as left, safe_open(
        right_path, framework="pt", device="cpu"
    ) as right:
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        common = sorted(left_keys & right_keys)
        vae_keys = [
            key
            for key in common
            if key.startswith("vae.")
            or key.startswith("first_stage_model.")
            or "autoencoder" in key.lower()
            or "per_channel_statistics" in key
        ]

        if not vae_keys:
            prefixes: dict[str, int] = {}
            for key in sorted(left_keys)[:2000]:
                prefixes[key.split(".", 1)[0]] = prefixes.get(key.split(".", 1)[0], 0) + 1
            return {
                "left": str(left_path),
                "right": str(right_path),
                "left_key_count": len(left_keys),
                "right_key_count": len(right_keys),
                "common_key_count": len(common),
                "vae_key_count": 0,
                "left_prefix_counts_sample": prefixes,
                "left_key_examples": sorted(left_keys)[:50],
            }

        mismatches = []
        for key in vae_keys:
            if not tensors_equal(left, right, key):
                mismatches.append(key)
                if len(mismatches) >= 20:
                    break

        return {
            "left": str(left_path),
            "right": str(right_path),
            "left_key_count": len(left_keys),
            "right_key_count": len(right_keys),
            "common_key_count": len(common),
            "vae_key_count": len(vae_keys),
            "vae_weights_identical_for_detected_keys": not mismatches,
            "mismatch_examples": mismatches,
            "vae_key_examples": vae_keys[:20],
        }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(compare.remote(), indent=2, sort_keys=True))
