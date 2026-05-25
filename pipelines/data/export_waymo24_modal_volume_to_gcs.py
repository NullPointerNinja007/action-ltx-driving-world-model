from __future__ import annotations

import csv
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "waymo24-modal-volume-gcs-export")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
GCP_SECRET_NAME = os.environ.get("GCP_MODAL_SECRET", "gcp-cs231n-waymo")

DATA_ROOT = Path("/data")
DEFAULT_BUCKET = "maleeka-waymo-interpolated-share-20260524"
DEFAULT_PREFIX = "waymo_24fps_121f_visual_continuation"
DEFAULT_SPLITS = "train,val"

MAX_EXPORT_CONTAINERS = int(os.environ.get("WAYMO24_EXPORT_MAX_CONTAINERS", "64"))


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
gcp_secret = modal.Secret.from_name(GCP_SECRET_NAME)

base_image = modal.Image.debian_slim(python_version="3.10").pip_install("google-cloud-storage")


def gcs_client():
    from google.cloud import storage
    from google.oauth2.credentials import Credentials

    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError(
            "Missing GCP_ACCESS_TOKEN in Modal secret. Refresh the secret with "
            "`gcloud auth print-access-token` before running this export."
        )
    return storage.Client(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "cs231n-496521",
        credentials=Credentials(token=access_token),
    )


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def content_type_for_path(path: str) -> str | None:
    if path.endswith(".jsonl"):
        return "application/x-ndjson"
    guessed, _ = mimetypes.guess_type(path)
    return guessed


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def split_list(splits_csv: str) -> list[str]:
    return [item.strip() for item in splits_csv.split(",") if item.strip()]


@app.function(
    image=base_image,
    cpu=2,
    memory=4096,
    timeout=20 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def list_export_relpaths(splits_csv: str = DEFAULT_SPLITS) -> dict[str, Any]:
    data_volume.reload()
    relpaths: list[str] = []

    for split in split_list(splits_csv):
        manifest_csv = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.csv"
        manifest_jsonl = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.jsonl"
        for manifest in (manifest_csv, manifest_jsonl):
            if not manifest.exists():
                raise FileNotFoundError(f"Missing manifest in Modal volume: {manifest}")
            relpaths.append(str(manifest.relative_to(DATA_ROOT)))

        with manifest_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                relpath = row["mp4_relpath"]
                local_path = DATA_ROOT / relpath
                if not local_path.exists():
                    raise FileNotFoundError(f"Missing MP4 in Modal volume: {local_path}")
                relpaths.append(relpath)

    unique_relpaths = sorted(set(relpaths))
    total_bytes = sum((DATA_ROOT / relpath).stat().st_size for relpath in unique_relpaths)
    return {
        "files": len(unique_relpaths),
        "bytes": total_bytes,
        "relpaths": unique_relpaths,
    }


@app.function(
    image=base_image,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
    max_containers=MAX_EXPORT_CONTAINERS,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[gcp_secret],
)
def upload_relpath_batch(
    relpaths: list[str],
    bucket_name: str,
    prefix: str = DEFAULT_PREFIX,
    overwrite: bool = True,
) -> dict[str, Any]:
    data_volume.reload()
    client = gcs_client()
    bucket = client.bucket(bucket_name)
    clean_prefix = normalize_prefix(prefix)

    uploaded = 0
    skipped = 0
    uploaded_bytes = 0
    for relpath in relpaths:
        local_path = DATA_ROOT / relpath
        if not local_path.exists():
            raise FileNotFoundError(f"Missing file in Modal volume: {local_path}")

        dst_name = f"{clean_prefix}/{relpath}" if clean_prefix else relpath
        blob = bucket.blob(dst_name)
        if not overwrite and blob.exists(client):
            skipped += 1
            continue

        content_type = content_type_for_path(relpath)
        blob.upload_from_filename(str(local_path), content_type=content_type)
        uploaded += 1
        uploaded_bytes += local_path.stat().st_size

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "bytes": uploaded_bytes,
        "first": relpaths[0] if relpaths else None,
        "last": relpaths[-1] if relpaths else None,
    }


@app.local_entrypoint()
def main(
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    splits_csv: str = DEFAULT_SPLITS,
    batch_size: int = 32,
    overwrite: bool = True,
) -> None:
    listing = list_export_relpaths.remote(splits_csv)
    relpaths = listing["relpaths"]
    batches = batched(relpaths, max(1, batch_size))
    print(
        json.dumps(
            {
                "bucket": bucket,
                "prefix": normalize_prefix(prefix),
                "splits": split_list(splits_csv),
                "files": listing["files"],
                "bytes": listing["bytes"],
                "batches": len(batches),
                "max_export_containers": MAX_EXPORT_CONTAINERS,
            },
            indent=2,
            sort_keys=True,
        )
    )

    total_uploaded = 0
    total_skipped = 0
    total_bytes = 0
    for result in upload_relpath_batch.starmap(
        [(batch, bucket, prefix, overwrite) for batch in batches],
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(result, Exception):
            raise result
        total_uploaded += result["uploaded"]
        total_skipped += result["skipped"]
        total_bytes += result["bytes"]
        print(
            json.dumps(
                {
                    "uploaded": total_uploaded,
                    "skipped": total_skipped,
                    "bytes": total_bytes,
                    "latest_first": result["first"],
                    "latest_last": result["last"],
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "done": True,
                "bucket": f"gs://{bucket}",
                "prefix": normalize_prefix(prefix),
                "uploaded": total_uploaded,
                "skipped": total_skipped,
                "bytes": total_bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
