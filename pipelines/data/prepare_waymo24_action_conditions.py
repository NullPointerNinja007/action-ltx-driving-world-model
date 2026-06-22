from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "waymo24-121f-action-condition-prep")
PROCESSED_V2_ROOT = os.environ.get("WAYMO24_PROCESSED_V2_ROOT", "")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
GCP_SECRET_NAME = os.environ.get("GCP_MODAL_SECRET", "gcp-cs231n-waymo")

DATA_ROOT = Path("/data")

FPS = 24
SOURCE_FPS = 10
CONTEXT_FRAMES = 49

PAST_STATE_LEN = 16
FUTURE_STATE_LEN = 20

if not PROCESSED_V2_ROOT:
    raise RuntimeError("Set WAYMO24_PROCESSED_V2_ROOT to the processed Waymo source root before running action prep.")

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


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
gcp_secret = modal.Secret.from_name(GCP_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("google-cloud-storage", "pyarrow")
)


def gcs_join(base: str, *parts: str) -> str:
    return base.rstrip("/") + "/" + "/".join(part.strip("/") for part in parts if part.strip("/"))


def parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def configure_gcp_credentials() -> None:
    raw_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        cred_path = Path("/tmp/gcp_service_account.json")
        cred_path.write_text(raw_json, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)


def gcs_client():
    configure_gcp_credentials()
    from google.cloud import storage
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials

    raw_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if raw_json:
        info = json.loads(raw_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=credentials)

    if os.environ.get("private_key") and os.environ.get("client_email"):
        service_account_keys = [
            "type",
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
            "auth_uri",
            "token_uri",
            "auth_provider_x509_cert_url",
            "client_x509_cert_url",
            "universe_domain",
        ]
        info = {key: os.environ[key] for key in service_account_keys if os.environ.get(key)}
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=credentials)

    access_token = os.environ.get("GCP_ACCESS_TOKEN")
    if access_token:
        return storage.Client(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "cs231n-496521",
            credentials=Credentials(token=access_token),
        )

    return storage.Client()


def download_gcs_file(uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = parse_gs_uri(uri)
    gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))


def frame_metadata_uri(split: str) -> str:
    return gcs_join(PROCESSED_V2_ROOT, f"front_512_{split}", "metadata_clean", f"front_frames_{split}_clean.parquet")


def read_manifest(split: str) -> list[dict[str, str]]:
    path = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing window manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=ACTION_SOURCE_COLUMNS)
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    columns = {name: table[name].to_pylist() for name in ACTION_SOURCE_COLUMNS}
    for idx in range(table.num_rows):
        key = (columns["scenario_id"][idx], int(columns["frame_id"][idx]))
        if key not in needed_keys:
            continue
        lookup[key] = {name: columns[name][idx] for name in ACTION_SOURCE_COLUMNS}
    return lookup


def action_vector_from_source(source: dict[str, Any], key: tuple[str, int]) -> list[float]:
    future_pos_x = parse_float_array(source["future_pos_x"], FUTURE_STATE_LEN, "future_pos_x", key)
    future_pos_y = parse_float_array(source["future_pos_y"], FUTURE_STATE_LEN, "future_pos_y", key)
    past_vel_x = parse_float_array(source["past_vel_x"], PAST_STATE_LEN, "past_vel_x", key)
    past_vel_y = parse_float_array(source["past_vel_y"], PAST_STATE_LEN, "past_vel_y", key)
    past_accel_x = parse_float_array(source["past_accel_x"], PAST_STATE_LEN, "past_accel_x", key)
    past_accel_y = parse_float_array(source["past_accel_y"], PAST_STATE_LEN, "past_accel_y", key)
    scalars = [
        finite_float(source["speed_mps"], "speed_mps", key),
        finite_float(source["approx_yaw_rate_rad_s"], "approx_yaw_rate_rad_s", key),
        finite_float(source["camera_velocity_v_x"], "camera_velocity_v_x", key),
        finite_float(source["camera_velocity_v_y"], "camera_velocity_v_y", key),
        finite_float(source["camera_velocity_v_z"], "camera_velocity_v_z", key),
        finite_float(source["camera_velocity_w_x"], "camera_velocity_w_x", key),
        finite_float(source["camera_velocity_w_y"], "camera_velocity_w_y", key),
        finite_float(source["camera_velocity_w_z"], "camera_velocity_w_z", key),
    ]
    vector = future_pos_x + future_pos_y + past_vel_x + past_vel_y + past_accel_x + past_accel_y + scalars
    if len(vector) != len(ACTION_FEATURE_ORDER):
        raise AssertionError(f"Action vector dim {len(vector)} != feature order dim {len(ACTION_FEATURE_ORDER)}")
    return vector


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_action_conditions_for_split(split: str) -> dict[str, Any]:
    data_volume.reload()
    manifest_rows = read_manifest(split)
    if not manifest_rows:
        raise RuntimeError(f"No rows in {split} window manifest")

    needed_keys = {
        (row["scenario_id"], context_boundary_source_frame_id(row))
        for row in manifest_rows
    }
    with tempfile.TemporaryDirectory(prefix=f"waymo24_action_{split}_") as tmp:
        parquet_path = Path(tmp) / f"front_frames_{split}_clean.parquet"
        download_gcs_file(frame_metadata_uri(split), parquet_path)
        action_lookup = build_action_lookup(parquet_path, needed_keys)

    missing = sorted(needed_keys - set(action_lookup.keys()))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} action rows for {split}; first={missing[:5]}")

    out_rows: list[ActionConditionRecord] = []
    feature_order_json = json.dumps(ACTION_FEATURE_ORDER, separators=(",", ":"))
    for row in manifest_rows:
        source_frame_id = context_boundary_source_frame_id(row)
        key = (row["scenario_id"], source_frame_id)
        source = action_lookup[key]
        vector = action_vector_from_source(source, key)
        action_relpath = f"action_conditions/{split}/{row['scenario_id']}/{row['window_id']}.json"
        payload = {
            "split": split,
            "scenario_id": row["scenario_id"],
            "window_id": row["window_id"],
            "window_idx": int(row["window_idx"]),
            "fps": int(row["fps"]),
            "source_fps": int(row["source_fps"]),
            "num_frames": int(row["num_frames"]),
            "context_frames": int(row["context_frames"]),
            "future_frames": int(row["future_frames"]),
            "start_frame_24fps": int(row["start_frame_24fps"]),
            "context_end_frame_24fps": int(row["start_frame_24fps"]) + CONTEXT_FRAMES - 1,
            "action_source_frame_id_10fps": source_frame_id,
            "source_min_frame_id_10fps": int(row["source_min_frame_id_10fps"]),
            "source_max_frame_id_10fps": int(row["source_max_frame_id_10fps"]),
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
        write_json(DATA_ROOT / action_relpath, payload)
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

    manifests_dir = DATA_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    action_csv = manifests_dir / f"{split}_windows_24fps_121f_action_conditions.csv"
    action_jsonl = manifests_dir / f"{split}_windows_24fps_121f_action_conditions.jsonl"
    fieldnames = list(ActionConditionRecord.__dataclass_fields__.keys())
    with action_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for out_row in out_rows:
            writer.writerow(out_row.__dict__)
    with action_jsonl.open("w", encoding="utf-8") as handle:
        for out_row in out_rows:
            handle.write(json.dumps(out_row.__dict__, sort_keys=True) + "\n")

    summary = {
        "split": split,
        "windows": len(out_rows),
        "scenarios": len({row.scenario_id for row in out_rows}),
        "action_dim": len(ACTION_FEATURE_ORDER),
        "action_feature_order": list(ACTION_FEATURE_ORDER),
        "manifest_csv": str(action_csv.relative_to(DATA_ROOT)),
        "manifest_jsonl": str(action_jsonl.relative_to(DATA_ROOT)),
        "action_conditions_root": f"action_conditions/{split}",
        "intent_counts": {},
    }
    for row in out_rows:
        summary["intent_counts"][row.intent_name] = summary["intent_counts"].get(row.intent_name, 0) + 1
    write_json(manifests_dir / f"{split}_windows_24fps_121f_action_conditions_summary.json", summary)
    data_volume.commit()
    return summary


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[gcp_secret],
)
def build_action_conditions_modal(split: str) -> dict[str, Any]:
    return build_action_conditions_for_split(split)


@app.local_entrypoint()
def main(splits_csv: str = "train,val") -> None:
    splits = [item.strip() for item in splits_csv.split(",") if item.strip()]
    summaries = {}
    for split in splits:
        summaries[split] = build_action_conditions_modal.remote(split)
    print(json.dumps(summaries, indent=2, sort_keys=True))
