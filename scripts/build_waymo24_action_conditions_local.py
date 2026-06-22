from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


FPS = 24
SOURCE_FPS = 10
CONTEXT_FRAMES = 49
PAST_STATE_LEN = 16
FUTURE_STATE_LEN = 20

ACTION_FEATURE_ORDER = (
    [f"future_pos_x_{idx:02d}" for idx in range(FUTURE_STATE_LEN)]
    + [f"future_pos_y_{idx:02d}" for idx in range(FUTURE_STATE_LEN)]
    + [f"past_vel_x_{idx:02d}" for idx in range(PAST_STATE_LEN)]
    + [f"past_vel_y_{idx:02d}" for idx in range(PAST_STATE_LEN)]
    + [f"past_accel_x_{idx:02d}" for idx in range(PAST_STATE_LEN)]
    + [f"past_accel_y_{idx:02d}" for idx in range(PAST_STATE_LEN)]
    + [
        "speed_mps",
        "approx_yaw_rate_rad_s",
        "camera_velocity_v_x",
        "camera_velocity_v_y",
        "camera_velocity_v_z",
        "camera_velocity_w_x",
        "camera_velocity_w_y",
        "camera_velocity_w_z",
    ]
)

ACTION_SOURCE_COLUMNS = [
    "scenario_id",
    "frame_id",
    "intent_value",
    "intent_name",
    "speed_mps",
    "approx_yaw_rate_rad_s",
    "camera_velocity_v_x",
    "camera_velocity_v_y",
    "camera_velocity_v_z",
    "camera_velocity_w_x",
    "camera_velocity_w_y",
    "camera_velocity_w_z",
    "past_vel_x",
    "past_vel_y",
    "past_accel_x",
    "past_accel_y",
    "future_pos_x",
    "future_pos_y",
    "future_pos_z",
]


@dataclass(frozen=True)
class ActionConditionRecord:
    split: str
    scenario_id: str
    window_idx: int
    window_id: str
    fps: int
    source_fps: int
    num_frames: int
    context_frames: int
    future_frames: int
    start_frame_24fps: int
    context_end_frame_24fps: int
    action_source_frame_id_10fps: int
    source_min_frame_id_10fps: int
    source_max_frame_id_10fps: int
    intent_value: int
    intent_name: str
    action_dim: int
    action_feature_order_json: str
    action_vector_json: str
    action_relpath: str
    mp4_relpath: str
    latent_relpath: str


def parse_float_array(raw: Any, expected_len: int, field_name: str, key: tuple[str, int]) -> list[float]:
    if raw is None or raw == "" or raw == "[]":
        raise ValueError(f"Missing {field_name} for {key}")
    values = json.loads(raw)
    if len(values) != expected_len:
        raise ValueError(f"{field_name} for {key} has length {len(values)}, expected {expected_len}")
    return [float(value) for value in values]


def finite_float(value: Any, field_name: str, key: tuple[str, int]) -> float:
    if value is None:
        raise ValueError(f"Missing {field_name} for {key}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {field_name}={out} for {key}")
    return out


def context_boundary_source_frame_id(row: dict[str, str]) -> int:
    start_24 = int(row["start_frame_24fps"])
    source_min = int(row["source_min_frame_id_10fps"])
    source_max = int(row["source_max_frame_id_10fps"])
    context_end_24 = start_24 + CONTEXT_FRAMES - 1
    source_offset = round(context_end_24 * SOURCE_FPS / FPS)
    return max(source_min, min(source_max, source_min + source_offset))


def build_action_lookup(parquet_path: Path, needed_keys: set[tuple[str, int]]) -> dict[tuple[str, int], dict[str, Any]]:
    table = pq.read_table(parquet_path, columns=ACTION_SOURCE_COLUMNS)
    columns = {name: table[name].to_pylist() for name in ACTION_SOURCE_COLUMNS}
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for idx in range(table.num_rows):
        key = (columns["scenario_id"][idx], int(columns["frame_id"][idx]))
        if key in needed_keys:
            lookup[key] = {name: columns[name][idx] for name in ACTION_SOURCE_COLUMNS}
    return lookup


def action_vector_from_source(source: dict[str, Any], key: tuple[str, int]) -> list[float]:
    vector = (
        parse_float_array(source["future_pos_x"], FUTURE_STATE_LEN, "future_pos_x", key)
        + parse_float_array(source["future_pos_y"], FUTURE_STATE_LEN, "future_pos_y", key)
        + parse_float_array(source["past_vel_x"], PAST_STATE_LEN, "past_vel_x", key)
        + parse_float_array(source["past_vel_y"], PAST_STATE_LEN, "past_vel_y", key)
        + parse_float_array(source["past_accel_x"], PAST_STATE_LEN, "past_accel_x", key)
        + parse_float_array(source["past_accel_y"], PAST_STATE_LEN, "past_accel_y", key)
        + [
            finite_float(source["speed_mps"], "speed_mps", key),
            finite_float(source["approx_yaw_rate_rad_s"], "approx_yaw_rate_rad_s", key),
            finite_float(source["camera_velocity_v_x"], "camera_velocity_v_x", key),
            finite_float(source["camera_velocity_v_y"], "camera_velocity_v_y", key),
            finite_float(source["camera_velocity_v_z"], "camera_velocity_v_z", key),
            finite_float(source["camera_velocity_w_x"], "camera_velocity_w_x", key),
            finite_float(source["camera_velocity_w_y"], "camera_velocity_w_y", key),
            finite_float(source["camera_velocity_w_z"], "camera_velocity_w_z", key),
        ]
    )
    if len(vector) != len(ACTION_FEATURE_ORDER):
        raise AssertionError(f"Action vector dim {len(vector)} != {len(ACTION_FEATURE_ORDER)}")
    return vector


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_split(split: str, manifest_path: Path, parquet_path: Path, out_root: Path) -> dict[str, Any]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    needed_keys = {
        (row["scenario_id"], context_boundary_source_frame_id(row))
        for row in manifest_rows
    }
    lookup = build_action_lookup(parquet_path, needed_keys)
    missing = sorted(needed_keys - set(lookup.keys()))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} source action rows for {split}; first={missing[:5]}")

    out_rows: list[ActionConditionRecord] = []
    feature_order_json = json.dumps(ACTION_FEATURE_ORDER, separators=(",", ":"))
    for row in manifest_rows:
        source_frame_id = context_boundary_source_frame_id(row)
        key = (row["scenario_id"], source_frame_id)
        source = lookup[key]
        vector = action_vector_from_source(source, key)
        action_relpath = f"action_conditions/{split}/{row['scenario_id']}/{row['window_id']}.json"
        payload = {
            "split": split,
            "scenario_id": row["scenario_id"],
            "window_id": row["window_id"],
            "window_idx": int(row["window_idx"]),
            "context_end_frame_24fps": int(row["start_frame_24fps"]) + CONTEXT_FRAMES - 1,
            "action_source_frame_id_10fps": source_frame_id,
            "intent_value": int(source["intent_value"]),
            "intent_name": source["intent_name"],
            "action_feature_order": list(ACTION_FEATURE_ORDER),
            "action_vector": vector,
            "raw_source": {
                "future_pos_x": json.loads(source["future_pos_x"]),
                "future_pos_y": json.loads(source["future_pos_y"]),
                "future_pos_z": json.loads(source["future_pos_z"]),
                "past_vel_x": json.loads(source["past_vel_x"]),
                "past_vel_y": json.loads(source["past_vel_y"]),
                "past_accel_x": json.loads(source["past_accel_x"]),
                "past_accel_y": json.loads(source["past_accel_y"]),
                "speed_mps": source["speed_mps"],
                "approx_yaw_rate_rad_s": source["approx_yaw_rate_rad_s"],
                "camera_velocity_v_x": source["camera_velocity_v_x"],
                "camera_velocity_v_y": source["camera_velocity_v_y"],
                "camera_velocity_v_z": source["camera_velocity_v_z"],
                "camera_velocity_w_x": source["camera_velocity_w_x"],
                "camera_velocity_w_y": source["camera_velocity_w_y"],
                "camera_velocity_w_z": source["camera_velocity_w_z"],
            },
        }
        write_json(out_root / action_relpath, payload)
        out_rows.append(
            ActionConditionRecord(
                split=split,
                scenario_id=row["scenario_id"],
                window_idx=int(row["window_idx"]),
                window_id=row["window_id"],
                fps=int(row["fps"]),
                source_fps=int(row["source_fps"]),
                num_frames=int(row["num_frames"]),
                context_frames=int(row["context_frames"]),
                future_frames=int(row["future_frames"]),
                start_frame_24fps=int(row["start_frame_24fps"]),
                context_end_frame_24fps=int(row["start_frame_24fps"]) + CONTEXT_FRAMES - 1,
                action_source_frame_id_10fps=source_frame_id,
                source_min_frame_id_10fps=int(row["source_min_frame_id_10fps"]),
                source_max_frame_id_10fps=int(row["source_max_frame_id_10fps"]),
                intent_value=int(source["intent_value"]),
                intent_name=source["intent_name"],
                action_dim=len(vector),
                action_feature_order_json=feature_order_json,
                action_vector_json=json.dumps(vector, separators=(",", ":")),
                action_relpath=action_relpath,
                mp4_relpath=row["mp4_relpath"],
                latent_relpath=row["latent_relpath"],
            )
        )

    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifests_dir / f"{split}_windows_24fps_121f_action_conditions.csv"
    jsonl_path = manifests_dir / f"{split}_windows_24fps_121f_action_conditions.jsonl"
    fieldnames = list(ActionConditionRecord.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for out_row in out_rows:
            writer.writerow(out_row.__dict__)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for out_row in out_rows:
            handle.write(json.dumps(out_row.__dict__, sort_keys=True) + "\n")

    summary = {
        "split": split,
        "windows": len(out_rows),
        "scenarios": len({row.scenario_id for row in out_rows}),
        "action_dim": len(ACTION_FEATURE_ORDER),
        "action_feature_order": list(ACTION_FEATURE_ORDER),
        "manifest_csv": str(csv_path.relative_to(out_root)),
        "manifest_jsonl": str(jsonl_path.relative_to(out_root)),
        "action_conditions_root": f"action_conditions/{split}",
        "intent_counts": {},
    }
    for row in out_rows:
        summary["intent_counts"][row.intent_name] = summary["intent_counts"].get(row.intent_name, 0) + 1
    write_json(manifests_dir / f"{split}_windows_24fps_121f_action_conditions_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    summaries = {}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        summaries[split] = build_split(
            split,
            args.work_root / "manifests" / f"{split}_windows_24fps_121f.csv",
            args.work_root / f"front_frames_{split}_clean.parquet",
            args.out_root,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
