#!/usr/bin/env python3
"""
Audit Waymo E2E proto schema and a small sample of actual raw records.
Writes a JSON report to GCS so we can verify populated fields before the full metadata run.

Important: this script only imports the E2E proto module. It does NOT import
the broader dataset proto module, because some Waymo E2E package builds do not expose it under
`waymo_open_dataset.protos`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tensorflow as tf
from google.cloud import storage
from google.protobuf.message import Message
from tqdm import tqdm

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


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
    return CAMERA_ID_TO_NAME.get(int(camera_id), f"CAMERA_{int(camera_id)}")


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def list_raw_blobs(raw_prefix: str) -> List[storage.Blob]:
    bucket_name, prefix = parse_gcs_uri(raw_prefix)
    client = storage.Client()
    blobs = []
    for blob in client.list_blobs(bucket_name, prefix=prefix + "/"):
        base = Path(blob.name).name
        if blob.size and blob.size > 0 and not base.startswith("."):
            blobs.append(blob)
    return sorted(blobs, key=lambda b: b.name)


def upload_text(text: str, gcs_uri: str) -> None:
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_string(text, content_type="application/json")


def download_blob(blob: storage.Blob, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))


def descriptor_fields(msg_cls_or_msg: Any) -> List[Dict[str, Any]]:
    desc = msg_cls_or_msg.DESCRIPTOR
    fields = []
    for f in desc.fields:
        fields.append({
            "name": f.name,
            "number": f.number,
            "label": f.label,
            "type": f.type,
            "message_type": f.message_type.full_name if f.message_type else None,
            "enum_type": f.enum_type.full_name if f.enum_type else None,
        })
    return fields


def field_nonempty(msg: Message, field_name: str) -> bool:
    if not hasattr(msg, field_name):
        return False
    val = getattr(msg, field_name)
    try:
        field_desc = msg.DESCRIPTOR.fields_by_name[field_name]
        if field_desc.label == field_desc.LABEL_REPEATED:
            return len(val) > 0
    except Exception:
        pass
    try:
        return msg.HasField(field_name)
    except Exception:
        if isinstance(val, (int, float, bool)):
            return bool(val)
        return val not in [None, "", []]


def bump(d: Dict[str, int], k: str) -> None:
    d[k] = d.get(k, 0) + 1


def audit_records(local_tfrecord: Path, max_records: int) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    camera_counts: Dict[str, int] = {}
    calib_counts: Dict[str, int] = {}
    state_counts: Dict[str, int] = {}
    dynamic_descriptors: Dict[str, Any] = {}
    preference_count = 0
    total = 0

    ds = tf.data.TFRecordDataset(str(local_tfrecord), compression_type="")
    for i, raw in enumerate(tqdm(ds, desc=local_tfrecord.name)):
        if i >= max_records:
            break
        e2e = wod_e2ed_pb2.E2EDFrame()
        e2e.ParseFromString(raw.numpy())
        total += 1

        # Capture descriptors from actual parsed objects. This avoids requiring the dataset proto module.
        dynamic_descriptors.setdefault("Frame", descriptor_fields(e2e.frame))
        dynamic_descriptors.setdefault("Context", descriptor_fields(e2e.frame.context))
        if len(e2e.frame.images) > 0:
            dynamic_descriptors.setdefault("CameraImage", descriptor_fields(e2e.frame.images[0]))
            if hasattr(e2e.frame.images[0], "velocity"):
                dynamic_descriptors.setdefault("Velocity", descriptor_fields(e2e.frame.images[0].velocity))
        if len(e2e.frame.context.camera_calibrations) > 0:
            dynamic_descriptors.setdefault(
                "CameraCalibration",
                descriptor_fields(e2e.frame.context.camera_calibrations[0]),
            )

        for f in e2e.DESCRIPTOR.fields:
            if field_nonempty(e2e, f.name):
                bump(counts, f"E2EDFrame.{f.name}")
        for f in e2e.frame.DESCRIPTOR.fields:
            if field_nonempty(e2e.frame, f.name):
                bump(counts, f"Frame.{f.name}")
        for f in e2e.frame.context.DESCRIPTOR.fields:
            if field_nonempty(e2e.frame.context, f.name):
                bump(counts, f"Context.{f.name}")

        for img in e2e.frame.images:
            cam_name = camera_name(int(img.name))
            bump(camera_counts, cam_name)
            for f in img.DESCRIPTOR.fields:
                if f.name == "image":
                    # Do not report bytes as metadata other than presence.
                    if len(img.image) > 0:
                        bump(counts, f"CameraImage.{cam_name}.image_bytes_present")
                    continue
                if field_nonempty(img, f.name):
                    bump(counts, f"CameraImage.{cam_name}.{f.name}")
            if hasattr(img, "velocity"):
                for f in img.velocity.DESCRIPTOR.fields:
                    if field_nonempty(img.velocity, f.name):
                        bump(counts, f"Velocity.{cam_name}.{f.name}")

        for calib in e2e.frame.context.camera_calibrations:
            cam_name = camera_name(int(calib.name))
            bump(calib_counts, cam_name)
            for f in calib.DESCRIPTOR.fields:
                if field_nonempty(calib, f.name):
                    bump(counts, f"CameraCalibration.{cam_name}.{f.name}")

        for prefix, states in [("past_states", e2e.past_states), ("future_states", e2e.future_states)]:
            for f in states.DESCRIPTOR.fields:
                if field_nonempty(states, f.name):
                    bump(state_counts, f"{prefix}.{f.name}")

        if len(e2e.preference_trajectories) > 0:
            preference_count += 1

    return {
        "sampled_records": total,
        "dynamic_schema_descriptors": dynamic_descriptors,
        "populated_field_counts": counts,
        "camera_image_counts": camera_counts,
        "camera_calibration_counts": calib_counts,
        "ego_state_populated_counts": state_counts,
        "records_with_preference_trajectories": preference_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_prefix", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--tmp_dir", default="/tmp/waymo_e2e_audit")
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--max_records_per_file", type=int, default=200)
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    blobs = list_raw_blobs(args.raw_prefix)[: args.max_files]
    if not blobs:
        raise RuntimeError(f"No raw files under {args.raw_prefix}")

    sample_reports = []
    for blob in blobs:
        local = tmp_dir / Path(blob.name).name
        download_blob(blob, local)
        try:
            sample_reports.append({
                "source_gcs_uri": f"gs://{blob.bucket.name}/{blob.name}",
                "report": audit_records(local, args.max_records_per_file),
            })
        finally:
            local.unlink(missing_ok=True)

    report = {
        "schema_descriptors": {
            "E2EDFrame": descriptor_fields(wod_e2ed_pb2.E2EDFrame),
            "EgoTrajectoryStates": descriptor_fields(wod_e2ed_pb2.EgoTrajectoryStates),
            "EgoIntent.Intent_values": list(wod_e2ed_pb2.EgoIntent.Intent.keys()),
            "Frame/CameraImage/CameraCalibration/Velocity": (
                "Captured dynamically from parsed records inside each sample report. "
                "No the dataset proto module import is required."
            ),
        },
        "sample_reports": sample_reports,
    }
    upload_text(json.dumps(report, indent=2, sort_keys=True), args.out_json)
    print(f"Wrote audit report: {args.out_json}")


if __name__ == "__main__":
    main()
