import argparse
import csv
import io
import json
from pathlib import Path

import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


FRONT_CAMERA_ID = 1


def split_context_name(context_name: str):
    if "-" not in context_name:
        return context_name, -1

    scenario_id, frame_id_str = context_name.rsplit("-", 1)

    try:
        frame_id = int(frame_id_str)
    except ValueError:
        frame_id = -1

    return scenario_id, frame_id


def repeated_to_json(x):
    return json.dumps(list(x))


def safe_repeated_to_json(msg, field_name: str):
    if hasattr(msg, field_name):
        return json.dumps(list(getattr(msg, field_name)))
    return json.dumps([])


def get_front_image(e2e_frame):
    for image_content in e2e_frame.frame.images:
        if image_content.name == FRONT_CAMERA_ID:
            return image_content
    return None


def print_state_fields_once(e2e):
    print("past_states fields:")
    for f in e2e.past_states.DESCRIPTOR.fields:
        val = getattr(e2e.past_states, f.name)
        try:
            n = len(val)
        except TypeError:
            n = "scalar"
        print(f"  {f.name}: len={n}")

    print("future_states fields:")
    for f in e2e.future_states.DESCRIPTOR.fields:
        val = getattr(e2e.future_states, f.name)
        try:
            n = len(val)
        except TypeError:
            n = "scalar"
        print(f"  {f.name}: len={n}")


def process_tfrecord(tfrecord_path: Path, out_dir: Path, writer, resize: int, max_records=None):
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

    shard_name = tfrecord_path.name
    written = 0
    printed_fields = False

    for record_idx, raw in enumerate(tqdm(dataset, desc=shard_name)):
        if max_records is not None and record_idx >= max_records:
            break

        e2e = wod_e2ed_pb2.E2EDFrame()
        e2e.ParseFromString(raw.numpy())

        if not printed_fields:
            print_state_fields_once(e2e)
            printed_fields = True

        context_name = e2e.frame.context.name
        scenario_id, frame_id = split_context_name(context_name)

        front = get_front_image(e2e)
        if front is None:
            continue

        img = Image.open(io.BytesIO(front.image)).convert("RGB")
        if resize is not None:
            img = img.resize((resize, resize))

        scenario_dir = out_dir / "frames_front_512" / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        image_rel_path = Path("frames_front_512") / scenario_id / f"{frame_id:06d}.jpg"
        image_abs_path = out_dir / image_rel_path

        img.save(image_abs_path, quality=90)

        writer.writerow({
            "source_tfrecord": str(tfrecord_path),
            "shard": shard_name,
            "record_idx": record_idx,
            "context_name": context_name,
            "scenario_id": scenario_id,
            "frame_id": frame_id,
            "camera_id": FRONT_CAMERA_ID,
            "image_path": str(image_rel_path),

            "past_pos_x": safe_repeated_to_json(e2e.past_states, "pos_x"),
            "past_pos_y": safe_repeated_to_json(e2e.past_states, "pos_y"),
            "past_vel_x": safe_repeated_to_json(e2e.past_states, "vel_x"),
            "past_vel_y": safe_repeated_to_json(e2e.past_states, "vel_y"),
            "past_velocity_x": safe_repeated_to_json(e2e.past_states, "velocity_x"),
            "past_velocity_y": safe_repeated_to_json(e2e.past_states, "velocity_y"),
            "past_heading": safe_repeated_to_json(e2e.past_states, "heading"),
            "past_yaw": safe_repeated_to_json(e2e.past_states, "yaw"),

            "future_pos_x": safe_repeated_to_json(e2e.future_states, "pos_x"),
            "future_pos_y": safe_repeated_to_json(e2e.future_states, "pos_y"),
            "future_vel_x": safe_repeated_to_json(e2e.future_states, "vel_x"),
            "future_vel_y": safe_repeated_to_json(e2e.future_states, "vel_y"),
            "future_velocity_x": safe_repeated_to_json(e2e.future_states, "velocity_x"),
            "future_velocity_y": safe_repeated_to_json(e2e.future_states, "velocity_y"),
            "future_heading": safe_repeated_to_json(e2e.future_states, "heading"),
            "future_yaw": safe_repeated_to_json(e2e.future_states, "yaw"),
        })

        written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_records_per_file", type=int, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    tfrecord_files = sorted(raw_dir.glob("*.tfrecord-*"))
    if args.max_files is not None:
        tfrecord_files = tfrecord_files[:args.max_files]

    if not tfrecord_files:
        raise RuntimeError(f"No TFRecord files found in {raw_dir}")

    csv_path = metadata_dir / "front_frames.csv"

    fieldnames = [
        "source_tfrecord",
        "shard",
        "record_idx",
        "context_name",
        "scenario_id",
        "frame_id",
        "camera_id",
        "image_path",

        "past_pos_x",
        "past_pos_y",
        "past_vel_x",
        "past_vel_y",
        "past_velocity_x",
        "past_velocity_y",
        "past_heading",
        "past_yaw",

        "future_pos_x",
        "future_pos_y",
        "future_vel_x",
        "future_vel_y",
        "future_velocity_x",
        "future_velocity_y",
        "future_heading",
        "future_yaw",
    ]

    total = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tfrecord_path in tfrecord_files:
            total += process_tfrecord(
                tfrecord_path=tfrecord_path,
                out_dir=out_dir,
                writer=writer,
                resize=args.resize,
                max_records=args.max_records_per_file,
            )

    print(f"Wrote {total} front-camera frames")
    print(f"Metadata CSV: {csv_path}")


if __name__ == "__main__":
    main()