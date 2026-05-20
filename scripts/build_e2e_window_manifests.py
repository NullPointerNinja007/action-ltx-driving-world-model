#!/usr/bin/env python3
"""
Build clean temporal window manifests from the sorted frame-level metadata.

This does not create video files. It creates rows like:
  clip_id, split, scenario_id, start_frame, end_frame, num_frames, context_frames, future_frames, fps

The dataloader should reconstruct frame paths as:
  frames_front_512/<scenario_id>/<frame_id:06d>.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from google.cloud import storage


def parse_gcs_uri(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def gcs_join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts if p.strip("/"))


def download_gcs_uri(uri: str, dst: Path) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    dst.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dst))


def upload_file(local_path: Path, uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))


def default_context(num_frames: int) -> int:
    # Reasonable defaults for future video prediction.
    if num_frames <= 17:
        return 9
    if num_frames <= 81:
        return 17
    return 33


def default_stride(num_frames: int) -> int:
    if num_frames <= 33:
        return 8
    if num_frames <= 81:
        return 16
    return 32


def make_windows(df: pd.DataFrame, split: str, num_frames: int, stride: int, context_frames: int, fps: int = 10) -> pd.DataFrame:
    rows = []
    clip_idx = 0

    for scenario_id, g in df.groupby("scenario_id", sort=True):
        frame_ids = sorted(g["frame_id"].astype(int).unique())
        if len(frame_ids) < num_frames:
            continue

        available = set(frame_ids)
        min_f, max_f = frame_ids[0], frame_ids[-1]

        for start_frame in range(min_f, max_f - num_frames + 2, stride):
            end_frame = start_frame + num_frames - 1
            if all(f in available for f in range(start_frame, end_frame + 1)):
                rows.append({
                    "clip_id": f"{split}_{num_frames}f_{clip_idx:08d}",
                    "split": split,
                    "scenario_id": scenario_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_frames": num_frames,
                    "context_frames": context_frames,
                    "future_frames": num_frames - context_frames,
                    "fps": fps,
                    "stride_used": stride,
                    "first_image_path": f"frames_front_512/{scenario_id}/{start_frame:06d}.jpg",
                    "last_image_path": f"frames_front_512/{scenario_id}/{end_frame:06d}.jpg",
                })
                clip_idx += 1

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_parquet", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--tmp_dir", default="/tmp/waymo_e2e_windows")
    parser.add_argument("--horizons", nargs="+", type=int, default=[17, 33, 49, 81, 129, 193])
    parser.add_argument("--strides", nargs="+", type=int, default=None, help="Optional same-length list matching --horizons")
    parser.add_argument("--contexts", nargs="+", type=int, default=None, help="Optional same-length list matching --horizons")
    args = parser.parse_args()

    if args.strides is not None and len(args.strides) != len(args.horizons):
        raise ValueError("--strides must have same length as --horizons")
    if args.contexts is not None and len(args.contexts) != len(args.horizons):
        raise ValueError("--contexts must have same length as --horizons")

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_frames = tmp_dir / f"front_frames_{args.split}_clean.parquet"
    download_gcs_uri(args.frames_parquet, local_frames)
    df = pd.read_parquet(local_frames)
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["frame_id", "scenario_id"])
    df["frame_id"] = df["frame_id"].astype(int)
    df = df.sort_values(["scenario_id", "frame_id"])

    for i, horizon in enumerate(args.horizons):
        stride = args.strides[i] if args.strides is not None else default_stride(horizon)
        context = args.contexts[i] if args.contexts is not None else default_context(horizon)
        if context >= horizon:
            raise ValueError(f"context_frames must be < num_frames for horizon {horizon}")

        windows = make_windows(df, args.split, horizon, stride, context)
        local_csv = tmp_dir / f"windows_{args.split}_{horizon}f.csv"
        windows.to_csv(local_csv, index=False)
        out_uri = gcs_join(args.out_prefix, f"windows_{args.split}_{horizon}f.csv")
        upload_file(local_csv, out_uri)
        print(f"{horizon}f: {len(windows)} windows -> {out_uri}")

    print("Done.")


if __name__ == "__main__":
    main()
