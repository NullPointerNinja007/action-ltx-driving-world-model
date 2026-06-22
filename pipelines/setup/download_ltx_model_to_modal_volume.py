from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "ltx-model-modal-volume-download")
MODELS_VOLUME_NAME = "models"
MODELS_ROOT = Path("/models")

HF_REPO_ID = "Lightricks/LTX-Video"
DEFAULT_MODEL_FILENAME = "ltxv-2b-0.9.6-dev-04-25.safetensors"
DEFAULT_EXTRA_FILES = "ltxv-spatial-upscaler-0.9.8.safetensors"


app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub[hf_xet]")


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3 * 60 * 60,
    volumes={str(MODELS_ROOT): models_volume},
)
def download_files(
    repo_id: str = HF_REPO_ID,
    filenames: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    models_volume.reload()
    filenames = filenames or [DEFAULT_MODEL_FILENAME]
    out_dir = MODELS_ROOT / "ltx"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for filename in filenames:
        target = out_dir / filename
        if target.exists() and not overwrite:
            skipped.append({"filename": filename, "path": str(target), "bytes": target.stat().st_size})
            continue

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        path = Path(cached)
        if path != target and path.exists():
            shutil.copy2(path, target)
        downloaded.append({"filename": filename, "path": str(target), "bytes": target.stat().st_size})

    models_volume.commit()
    return {"repo_id": repo_id, "downloaded": downloaded, "skipped": skipped}


@app.local_entrypoint()
def main(
    model_filename: str = DEFAULT_MODEL_FILENAME,
    extra_files_csv: str = DEFAULT_EXTRA_FILES,
    overwrite: bool = False,
) -> None:
    filenames = [model_filename]
    filenames.extend(item.strip() for item in extra_files_csv.split(",") if item.strip())
    result = download_files.remote(HF_REPO_ID, filenames, overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))
