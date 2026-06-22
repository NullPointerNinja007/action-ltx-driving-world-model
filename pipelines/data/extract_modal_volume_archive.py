from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "modal-volume-archive-extract")
DATA_VOLUME_NAME = os.environ.get("WAYMO24_DATA_VOLUME_NAME", "waymo-e2e-24fps-121f-visual-continuation-data")
DATA_ROOT = Path("/data")


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)


image = modal.Image.debian_slim(python_version="3.10")


def safe_extract_tar(archive_path: Path, dest_root: Path) -> int:
    dest_root_resolved = dest_root.resolve()
    count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (dest_root / member.name).resolve()
            if not str(target).startswith(str(dest_root_resolved)):
                raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
            archive.extract(member, dest_root)
            if member.isfile():
                count += 1
    return count


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def extract_archive(archive_relpath: str, remove_archive: bool = False) -> dict[str, Any]:
    data_volume.reload()
    archive_path = DATA_ROOT / archive_relpath
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive in volume: {archive_path}")
    extracted_files = safe_extract_tar(archive_path, DATA_ROOT)
    if remove_archive:
        archive_path.unlink()
    data_volume.commit()
    return {
        "archive_relpath": archive_relpath,
        "extracted_files": extracted_files,
        "removed_archive": remove_archive,
    }


@app.local_entrypoint()
def main(archive_relpath: str, remove_archive: bool = False) -> None:
    result = extract_archive.remote(archive_relpath, remove_archive=remove_archive)
    print(json.dumps(result, indent=2, sort_keys=True))
