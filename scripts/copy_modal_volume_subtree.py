from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("COPY_MODAL_VOLUME_APP", "copy-modal-volume-subtree")
SRC_VOLUME_NAME = os.environ.get("COPY_MODAL_SRC_VOLUME_NAME", "")
DST_VOLUME_NAME = os.environ.get("COPY_MODAL_DST_VOLUME_NAME", "")

if not SRC_VOLUME_NAME or not DST_VOLUME_NAME:
    # These are intentionally configured by the caller through env so the same
    # utility can copy between arbitrary Modal volumes.
    SRC_VOLUME_NAME = SRC_VOLUME_NAME or "missing-source-volume"
    DST_VOLUME_NAME = DST_VOLUME_NAME or "missing-destination-volume"

SRC_ROOT = Path("/src")
DST_ROOT = Path("/dst")

app = modal.App(APP_NAME)
src_volume = modal.Volume.from_name(SRC_VOLUME_NAME)
dst_volume = modal.Volume.from_name(DST_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.10")


def path_size_and_count(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    total_size = 0
    total_files = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_size += child.stat().st_size
            total_files += 1
    return total_size, total_files


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes={str(SRC_ROOT): src_volume, str(DST_ROOT): dst_volume},
)
def copy_subtree(src_relpath: str, dst_relpath: str, force: bool = False) -> dict[str, Any]:
    src = SRC_ROOT / src_relpath.strip("/")
    dst = DST_ROOT / dst_relpath.strip("/")
    if not src.exists():
        raise FileNotFoundError(f"Missing source subtree: {src}")
    if dst.exists():
        if not force:
            size, files = path_size_and_count(dst)
            return {
                "status": "skipped_existing",
                "src_relpath": src_relpath,
                "dst_relpath": dst_relpath,
                "bytes": size,
                "files": files,
            }
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    size, files = path_size_and_count(dst)
    dst_volume.commit()
    return {
        "status": "copied",
        "src_volume": SRC_VOLUME_NAME,
        "dst_volume": DST_VOLUME_NAME,
        "src_relpath": src_relpath,
        "dst_relpath": dst_relpath,
        "bytes": size,
        "files": files,
        "copied_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.local_entrypoint()
def main(src_relpath: str, dst_relpath: str, force: bool = False) -> None:
    result = copy_subtree.remote(src_relpath, dst_relpath, force)
    print(json.dumps(result, indent=2, sort_keys=True))
