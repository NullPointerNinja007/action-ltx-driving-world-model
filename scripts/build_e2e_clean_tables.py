#!/usr/bin/env python3
"""
Merge rich metadata parquet shards into clean sorted frame-level and scenario-level tables.

Outputs:
  front_frames_<split>_clean.parquet
  front_frames_<split>_clean.csv        optional, disabled by default
  scenarios_<split>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from google.cloud import storage
from tqdm import tqdm


def parse_gcs_uri(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def gcs_join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts if p.strip("/"))


def list_parquet_uris(prefix_uri: str) -> List[str]:
    bucket_name, prefix = parse_gcs_uri(prefix_uri)
    client = storage.Client()
    uris = []
    for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
        if blob.name.endswith(".parquet") and blob.size and blob.size > 0:
            uris.append(f"gs://{bucket_name}/{blob.name}")
    return sorted(uris)


def download_gcs_uri(uri: str, dst: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    dst.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dst))


def upload_file(local_path: Path, uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))


def scenario_summary(g: pd.DataFrame) -> pd.Series:
    frame_ids = sorted(g["frame_id"].astype(int).unique())
    if not frame_ids:
        return pd.Series({
            "num_front_frames": 0,
            "min_frame_id": None,
            "max_frame_id": None,
            "is_contiguous": False,
            "missing_frame_count": None,
        })
    min_f, max_f = frame_ids[0], frame_ids[-1]
    expected = max_f - min_f + 1
    is_contig = expected == len(frame_ids)
    return pd.Series({
        "num_front_frames": len(frame_ids),
        "min_frame_id": min_f,
        "max_frame_id": max_f,
        "is_contiguous": bool(is_contig),
        "missing_frame_count": int(expected - len(frame_ids)),
        "source_tfrecord": g["source_tfrecord"].iloc[0] if "source_tfrecord" in g else "",
        "shard": g["shard"].iloc[0] if "shard" in g else "",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_shards_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--tmp_dir", default="/tmp/waymo_e2e_clean")
    parser.add_argument("--write_csv", action="store_true", help="Also write full clean frame table as CSV. Parquet is strongly preferred.")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    uris = list_parquet_uris(args.metadata_shards_prefix)
    if not uris:
        raise RuntimeError(f"No parquet shards under {args.metadata_shards_prefix}")

    print(f"Found metadata shards: {len(uris)}")
    dfs = []
    for i, uri in enumerate(tqdm(uris, desc="Downloading/reading shards")):
        local = tmp_dir / f"shard_{i:05d}.parquet"
        download_gcs_uri(uri, local)
        df = pd.read_parquet(local)
        if not df.empty:
            dfs.append(df)
        local.unlink(missing_ok=True)

    if not dfs:
        raise RuntimeError("All shards were empty.")

    df = pd.concat(dfs, ignore_index=True)
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["scenario_id", "frame_id"])
    df["frame_id"] = df["frame_id"].astype(int)

    # One frame row per scenario/frame. Keep the first if duplicate occurs.
    before = len(df)
    df = df.drop_duplicates(subset=["split", "scenario_id", "frame_id"], keep="first")
    dropped = before - len(df)

    df = df.sort_values(["split", "scenario_id", "frame_id"]).reset_index(drop=True)

    scenarios = (
        df.groupby(["split", "scenario_id"], sort=True)
        .apply(scenario_summary)
        .reset_index()
    )
    scenarios["fps"] = 10

    local_frames = tmp_dir / f"front_frames_{args.split}_clean.parquet"
    local_scenarios = tmp_dir / f"scenarios_{args.split}.csv"
    df.to_parquet(local_frames, index=False)
    scenarios.to_csv(local_scenarios, index=False)

    frames_uri = gcs_join(args.out_prefix, f"front_frames_{args.split}_clean.parquet")
    scenarios_uri = gcs_join(args.out_prefix, f"scenarios_{args.split}.csv")
    upload_file(local_frames, frames_uri)
    upload_file(local_scenarios, scenarios_uri)

    if args.write_csv:
        local_csv = tmp_dir / f"front_frames_{args.split}_clean.csv"
        df.to_csv(local_csv, index=False)
        upload_file(local_csv, gcs_join(args.out_prefix, f"front_frames_{args.split}_clean.csv"))

    print("=== Clean table complete ===")
    print(f"Rows: {len(df)}")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"Dropped duplicate frame rows: {dropped}")
    print(f"Frames parquet: {frames_uri}")
    print(f"Scenarios CSV: {scenarios_uri}")


if __name__ == "__main__":
    main()
