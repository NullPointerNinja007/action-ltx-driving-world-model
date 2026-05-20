#!/usr/bin/env python3
"""
Extract rich frame-level metadata from Waymo E2E raw TFRecords into Parquet shards.

This is metadata-only. It assumes 512x512 FRONT frames already exist at:
  gs://.../processed_v2/front_512_<split>/frames_front_512/<scenario_id>/<frame_id>.jpg

Design goal: do not miss E2E metadata again.
- Explicit top-level columns for the fields we will condition on.
- JSON dumps for every populated metadata sub-message we care about.
- Never serializes camera JPEG bytes into metadata.
- Processes one raw shard at a time, safe for ~250GB VM disk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
from google.cloud import storage
from google.protobuf.message import Message
from tqdm import tqdm

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

FRONT_CAMERA_ID = 1

# These are the exact E2E trajectory-state fields in the public proto docs.
# Extra fallback names are also included below so this remains robust to package/version changes.
EGO_STATE_FIELDS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y",
    "accel_x", "accel_y",
    "preference_score",
]

EXTRA_STATE_FALLBACK_FIELDS = [
    "vel_z",
    "velocity_x", "velocity_y", "velocity_z",
    "acceleration_x", "acceleration_y", "acceleration_z",
    "heading", "yaw", "speed", "valid",
    "score", "rating",
]


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def gcs_join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(p.strip("/") for p in parts if p.strip("/"))


def list_raw_blobs(raw_prefix: str, name_regex: Optional[str] = None) -> List[storage.Blob]:
    bucket_name, prefix = parse_gcs_uri(raw_prefix)
    client = storage.Client()
    pat = re.compile(name_regex) if name_regex else None
    blobs: List[storage.Blob] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
        base = Path(blob.name).name
        if blob.name.endswith("/") or base in {".keep", "_SUCCESS"}:
            continue
        if blob.size == 0:
            continue
        if pat and not pat.search(base):
            continue
        blobs.append(blob)
    return sorted(blobs, key=lambda b: b.name)


def download_blob(blob: storage.Blob, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))


def upload_file(local_path: Path, gcs_uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))


def gcs_exists(gcs_uri: str) -> bool:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    return client.bucket(bucket_name).blob(blob_name).exists()


def split_context_name(context_name: str) -> Tuple[str, int]:
    if "-" not in context_name:
        return context_name, -1
    scenario_id, frame_id_str = context_name.rsplit("-", 1)
    try:
        return scenario_id, int(frame_id_str)
    except ValueError:
        return scenario_id, -1


def enum_name(enum_obj: Any, value: Optional[int]) -> str:
    if value is None:
        return ""
    try:
        return enum_obj.Name(value)
    except Exception:
        return str(value)


def json_dumps(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def bytes_summary(b: bytes) -> Dict[str, Any]:
    return {"bytes_omitted": True, "num_bytes": len(b)}


def protobuf_to_dict_safe(msg: Optional[Message], *, omit_bytes: bool = True) -> Dict[str, Any]:
    """Recursively converts a protobuf message to a JSON-safe dict.

    Critical: bytes fields are NOT expanded by default. This avoids accidentally
    writing JPEG payloads / segmentation PNGs into metadata parquet.
    """
    if msg is None:
        return {}

    out: Dict[str, Any] = {}
    for field in msg.DESCRIPTOR.fields:
        name = field.name
        try:
            val = getattr(msg, name)
        except Exception:
            continue

        # Repeated field.
        if field.label == field.LABEL_REPEATED:
            if field.message_type is not None:
                out[name] = [protobuf_to_dict_safe(v, omit_bytes=omit_bytes) for v in val]
            else:
                if field.type == field.TYPE_BYTES and omit_bytes:
                    out[name] = [bytes_summary(v) for v in val]
                else:
                    out[name] = list(val)
            continue

        # Singular message field.
        if field.message_type is not None:
            try:
                present = msg.HasField(name)
            except Exception:
                present = True
            out[name] = protobuf_to_dict_safe(val, omit_bytes=omit_bytes) if present else {}
            continue

        # Singular scalar/bytes field.
        if field.type == field.TYPE_BYTES and omit_bytes:
            out[name] = bytes_summary(val) if val else {"bytes_omitted": True, "num_bytes": 0}
        else:
            try:
                out[name] = val
            except Exception:
                out[name] = str(val)

    return out


def repeated_json(msg: Optional[Message], field_name: str) -> str:
    if msg is None or not hasattr(msg, field_name):
        return "[]"
    try:
        return json_dumps(list(getattr(msg, field_name)))
    except Exception:
        return "[]"


def numeric(obj: Any, field_name: str) -> Optional[float]:
    if obj is None or not hasattr(obj, field_name):
        return None
    try:
        return float(getattr(obj, field_name))
    except Exception:
        return None


def integer(obj: Any, field_name: str) -> Optional[int]:
    if obj is None or not hasattr(obj, field_name):
        return None
    try:
        return int(getattr(obj, field_name))
    except Exception:
        return None


def transform_json(obj: Any) -> str:
    if obj is None:
        return "[]"
    if hasattr(obj, "transform"):
        try:
            return json_dumps(list(obj.transform))
        except Exception:
            return "[]"
    return "[]"


def get_camera_image(e2e: wod_e2ed_pb2.E2EDFrame, camera_id: int) -> Optional[Message]:
    for image in e2e.frame.images:
        if int(image.name) == camera_id:
            return image
    return None


def get_camera_calibration(e2e: wod_e2ed_pb2.E2EDFrame, camera_id: int) -> Optional[Message]:
    for calib in e2e.frame.context.camera_calibrations:
        if int(calib.name) == camera_id:
            return calib
    return None


CAMERA_ID_TO_NAME = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR_RIGHT",
    8: "REAR",
}


def camera_name(camera_id: int) -> str:
    # Avoid the dataset proto module. Some Waymo E2E package builds expose only the E2E proto cleanly.
    return CAMERA_ID_TO_NAME.get(int(camera_id), f"CAMERA_{int(camera_id)}")


def velocity_fields(image: Optional[Message], prefix: str) -> Dict[str, Optional[float]]:
    out = {
        f"{prefix}_velocity_v_x": None,
        f"{prefix}_velocity_v_y": None,
        f"{prefix}_velocity_v_z": None,
        f"{prefix}_velocity_w_x": None,
        f"{prefix}_velocity_w_y": None,
        f"{prefix}_velocity_w_z": None,
        f"{prefix}_yaw_rate_rad_s": None,
        f"{prefix}_speed_mps": None,
    }
    if image is None or not hasattr(image, "velocity"):
        return out
    vel = image.velocity
    vx, vy, vz = numeric(vel, "v_x"), numeric(vel, "v_y"), numeric(vel, "v_z")
    wx, wy, wz = numeric(vel, "w_x"), numeric(vel, "w_y"), numeric(vel, "w_z")
    out.update({
        f"{prefix}_velocity_v_x": vx,
        f"{prefix}_velocity_v_y": vy,
        f"{prefix}_velocity_v_z": vz,
        f"{prefix}_velocity_w_x": wx,
        f"{prefix}_velocity_w_y": wy,
        f"{prefix}_velocity_w_z": wz,
        f"{prefix}_yaw_rate_rad_s": wz,
    })
    if vx is not None and vy is not None and vz is not None:
        out[f"{prefix}_speed_mps"] = math.sqrt(vx * vx + vy * vy + vz * vz)
    elif vx is not None and vy is not None:
        out[f"{prefix}_speed_mps"] = math.sqrt(vx * vx + vy * vy)
    return out


def image_metadata_dict(image: Optional[Message]) -> Dict[str, Any]:
    if image is None:
        return {}
    # This omits image bytes and segmentation payload bytes while keeping metadata.
    return protobuf_to_dict_safe(image, omit_bytes=True)


def calibration_metadata_dict(calib: Optional[Message]) -> Dict[str, Any]:
    if calib is None:
        return {}
    return protobuf_to_dict_safe(calib, omit_bytes=True)


def top_level_state_arrays(prefix: str, states: Optional[Message]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in EGO_STATE_FIELDS + EXTRA_STATE_FALLBACK_FIELDS:
        out[f"{prefix}_{name}"] = repeated_json(states, name)
    return out


def preference_summary(e2e: wod_e2ed_pb2.E2EDFrame) -> Dict[str, Any]:
    prefs = [protobuf_to_dict_safe(traj, omit_bytes=True) for traj in e2e.preference_trajectories]
    scores: List[float] = []
    for item in prefs:
        for key in ["preference_score", "score", "scores", "rating", "ratings"]:
            if key not in item:
                continue
            val = item[key]
            if isinstance(val, list):
                for x in val:
                    try:
                        scores.append(float(x))
                    except Exception:
                        pass
            else:
                try:
                    scores.append(float(val))
                except Exception:
                    pass
    return {
        "num_preference_trajectories": len(prefs),
        "preference_trajectories_json": json_dumps(prefs),
        "preference_scores_json": json_dumps(scores),
        "max_preference_score": max(scores) if scores else None,
    }


def extract_one_record(
    e2e: wod_e2ed_pb2.E2EDFrame,
    *,
    split: str,
    source_gcs_uri: str,
    shard_name: str,
    record_idx: int,
    frames_gcs_prefix: str,
) -> Optional[Dict[str, Any]]:
    context_name = e2e.frame.context.name
    scenario_id, frame_id = split_context_name(context_name)

    front = get_camera_image(e2e, FRONT_CAMERA_ID)
    if front is None:
        return None

    front_calib = get_camera_calibration(e2e, FRONT_CAMERA_ID)
    image_path = f"frames_front_512/{scenario_id}/{frame_id:06d}.jpg"
    gcs_image_uri = gcs_join(frames_gcs_prefix, scenario_id, f"{frame_id:06d}.jpg")

    try:
        intent_value = int(e2e.intent)
    except Exception:
        intent_value = None

    all_images_meta = []
    for image in e2e.frame.images:
        cam_id = int(image.name)
        d = image_metadata_dict(image)
        d["camera_name_text"] = camera_name(cam_id)
        all_images_meta.append(d)

    all_calibs_meta = []
    for calib in e2e.frame.context.camera_calibrations:
        cam_id = int(calib.name)
        d = calibration_metadata_dict(calib)
        d["camera_name_text"] = camera_name(cam_id)
        all_calibs_meta.append(d)

    row: Dict[str, Any] = {
        "split": split,
        "source_tfrecord": source_gcs_uri,
        "shard": shard_name,
        "record_idx": record_idx,
        "context_name": context_name,
        "scenario_id": scenario_id,
        "frame_id": frame_id,
        "timestamp_micros": integer(e2e.frame, "timestamp_micros"),
        "camera_id": FRONT_CAMERA_ID,
        "camera_name": camera_name(FRONT_CAMERA_ID),
        "image_path": image_path,
        "gcs_image_uri": gcs_image_uri,
        "intent_value": intent_value,
        "intent_name": enum_name(wod_e2ed_pb2.EgoIntent.Intent, intent_value),
        "all_camera_images_metadata_json": json_dumps(all_images_meta),
        "all_camera_calibrations_json": json_dumps(all_calibs_meta),
        "frame_context_json": json_dumps(protobuf_to_dict_safe(e2e.frame.context, omit_bytes=True)),
        "front_camera_image_metadata_json": json_dumps(image_metadata_dict(front)),
        "front_calibration_metadata_json": json_dumps(calibration_metadata_dict(front_calib)),
        "front_pose_transform_json": transform_json(front.pose) if hasattr(front, "pose") else "[]",
        "frame_pose_transform_json": transform_json(e2e.frame.pose) if hasattr(e2e.frame, "pose") else "[]",
        "front_pose_timestamp": numeric(front, "pose_timestamp"),
        "front_shutter": numeric(front, "shutter"),
        "front_camera_trigger_time": numeric(front, "camera_trigger_time"),
        "front_camera_readout_done_time": numeric(front, "camera_readout_done_time"),
        "front_intrinsic_json": repeated_json(front_calib, "intrinsic"),
        "front_extrinsic_json": transform_json(front_calib.extrinsic) if front_calib is not None and hasattr(front_calib, "extrinsic") else "[]",
        "front_width": integer(front_calib, "width"),
        "front_height": integer(front_calib, "height"),
        "front_rolling_shutter_direction": integer(front_calib, "rolling_shutter_direction"),
    }

    row.update(velocity_fields(front, "front"))
    # Aliases that are convenient for action conditioning.
    row["camera_velocity_v_x"] = row["front_velocity_v_x"]
    row["camera_velocity_v_y"] = row["front_velocity_v_y"]
    row["camera_velocity_v_z"] = row["front_velocity_v_z"]
    row["camera_velocity_w_x"] = row["front_velocity_w_x"]
    row["camera_velocity_w_y"] = row["front_velocity_w_y"]
    row["camera_velocity_w_z"] = row["front_velocity_w_z"]
    row["approx_yaw_rate_rad_s"] = row["front_yaw_rate_rad_s"]
    row["speed_mps"] = row["front_speed_mps"]

    row.update(top_level_state_arrays("past", e2e.past_states))
    row.update(top_level_state_arrays("future", e2e.future_states))
    row["past_states_json"] = json_dumps(protobuf_to_dict_safe(e2e.past_states, omit_bytes=True))
    row["future_states_json"] = json_dumps(protobuf_to_dict_safe(e2e.future_states, omit_bytes=True))
    row.update(preference_summary(e2e))

    return row


def process_local_tfrecord(
    local_tfrecord: Path,
    *,
    split: str,
    source_gcs_uri: str,
    shard_name: str,
    frames_gcs_prefix: str,
    max_records_per_file: Optional[int],
) -> pd.DataFrame:
    dataset = tf.data.TFRecordDataset(str(local_tfrecord), compression_type="")
    rows: List[Dict[str, Any]] = []
    for record_idx, raw in enumerate(tqdm(dataset, desc=shard_name)):
        if max_records_per_file is not None and record_idx >= max_records_per_file:
            break
        e2e = wod_e2ed_pb2.E2EDFrame()
        e2e.ParseFromString(raw.numpy())
        row = extract_one_record(
            e2e,
            split=split,
            source_gcs_uri=source_gcs_uri,
            shard_name=shard_name,
            record_idx=record_idx,
            frames_gcs_prefix=frames_gcs_prefix,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--frames_gcs_prefix", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--tmp_dir", default="/tmp/waymo_e2e_metadata")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None, help="exclusive")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_records_per_file", type=int, default=None)
    parser.add_argument("--name_regex", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    raw_blobs = list_raw_blobs(args.raw_prefix, name_regex=args.name_regex)
    if not raw_blobs:
        raise RuntimeError(f"No raw blobs found under {args.raw_prefix}")

    end = args.end_index if args.end_index is not None else len(raw_blobs)
    selected = raw_blobs[args.start_index:end]
    if args.max_files is not None:
        selected = selected[: args.max_files]

    print(f"Found raw shards: {len(raw_blobs)}")
    print(f"Selected shards: {len(selected)}")
    print(f"Output prefix: {args.out_prefix}")

    for local_idx, blob in enumerate(selected, start=args.start_index):
        shard_name = Path(blob.name).name
        source_gcs_uri = f"gs://{blob.bucket.name}/{blob.name}"
        out_uri = gcs_join(args.out_prefix, f"{args.split}_batch_{local_idx:05d}_rich.parquet")
        if (not args.overwrite) and gcs_exists(out_uri):
            print(f"SKIP existing: {out_uri}")
            continue

        local_raw = tmp_dir / shard_name
        local_parquet = tmp_dir / f"{args.split}_batch_{local_idx:05d}_rich.parquet"

        print(f"\n=== [{local_idx}] {source_gcs_uri} ===")
        download_blob(blob, local_raw)
        try:
            df = process_local_tfrecord(
                local_raw,
                split=args.split,
                source_gcs_uri=source_gcs_uri,
                shard_name=shard_name,
                frames_gcs_prefix=args.frames_gcs_prefix,
                max_records_per_file=args.max_records_per_file,
            )
            df.to_parquet(local_parquet, index=False)
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Uploading -> {out_uri}")
            upload_file(local_parquet, out_uri)
        finally:
            local_raw.unlink(missing_ok=True)
            local_parquet.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
