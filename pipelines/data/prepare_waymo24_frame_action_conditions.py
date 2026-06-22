from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("LTX_MODAL_APP", "waymo24-121f-frame-action-condition-prep")
PROCESSED_V2_ROOT = os.environ.get("WAYMO24_PROCESSED_V2_ROOT", "")
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
GCP_SECRET_NAME = os.environ.get("GCP_MODAL_SECRET", "gcp-cs231n-waymo")

DATA_ROOT = Path("/data")

FPS = 24
SOURCE_FPS = 10
TOTAL_FRAMES = 121
CONTEXT_FRAMES = 49
FUTURE_FRAMES = 72
PAST_STATE_LEN = 16
FUTURE_STATE_LEN = 20
MODEL_ACTION_DIM = 18

FRAME_ACTION_ROOT = "frame_action_conditions_24fps"
AUDIT_ROOT = "action_frame_audits"
STATS_RELPATH = "manifests/frame_action_24fps_normalization_stats.json"

if not PROCESSED_V2_ROOT:
    raise RuntimeError("Set WAYMO24_PROCESSED_V2_ROOT to the processed Waymo source root before running action prep.")

FULL_ACTION_FEATURE_ORDER = (
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

MODEL_ACTION_FEATURE_ORDER = [
    "speed_mps",
    "approx_yaw_rate_rad_s",
    "camera_velocity_v_x",
    "camera_velocity_v_y",
    "camera_velocity_v_z",
    "camera_velocity_w_x",
    "camera_velocity_w_y",
    "camera_velocity_w_z",
    "current_accel_x",
    "current_accel_y",
    "future_pos_x_0p5s",
    "future_pos_y_0p5s",
    "future_pos_x_1p0s",
    "future_pos_y_1p0s",
    "future_pos_x_1p5s",
    "future_pos_y_1p5s",
    "future_pos_x_2p0s",
    "future_pos_y_2p0s",
]

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
class FrameActionConditionRecord:
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
    end_frame_24fps: int
    source_min_frame_id_10fps: int
    source_max_frame_id_10fps: int
    frame_action_dim: int
    frame_action_shape_json: str
    frame_action_feature_order_json: str
    full_action_dim: int
    full_action_feature_order_json: str
    frame_action_relpath: str
    mp4_relpath: str
    latent_relpath: str
    linear_interpolation_frames: int
    nearest_due_missing_frames: int
    future_pos_time_origin: str


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
gcp_secret = modal.Secret.from_name(GCP_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "google-cloud-storage",
        "imageio",
        "imageio-ffmpeg",
        "numpy",
        "pillow",
        "pyarrow",
    )
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


def read_window_manifest(split: str) -> list[dict[str, str]]:
    path = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing 24 FPS window manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float_array(raw: Any, expected_len: int, field_name: str, key: tuple[str, int]) -> list[float]:
    if raw is None or raw == "" or raw == "[]":
        raise ValueError(f"Missing {field_name} for {key}")
    if isinstance(raw, str):
        values = json.loads(raw)
    else:
        values = list(raw)
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


def full_action_vector_from_source(source: dict[str, Any], key: tuple[str, int]) -> list[float]:
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
    if len(vector) != len(FULL_ACTION_FEATURE_ORDER):
        raise AssertionError(f"Full action vector dim {len(vector)} != {len(FULL_ACTION_FEATURE_ORDER)}")
    return vector


def load_source_lookup(parquet_path: Path, needed_scenarios: set[str]) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, list[int]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=ACTION_SOURCE_COLUMNS)
    columns = {name: table[name].to_pylist() for name in ACTION_SOURCE_COLUMNS}
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    available_frames: dict[str, list[int]] = {}
    for idx in range(table.num_rows):
        scenario_id = columns["scenario_id"][idx]
        if scenario_id not in needed_scenarios:
            continue
        frame_id = int(columns["frame_id"][idx])
        key = (scenario_id, frame_id)
        lookup[key] = {name: columns[name][idx] for name in ACTION_SOURCE_COLUMNS}
        available_frames.setdefault(scenario_id, []).append(frame_id)
    for frame_ids in available_frames.values():
        frame_ids.sort()
    return lookup, available_frames


def nearest_available_frame(frame_ids: list[int], frame_id: int) -> int:
    import bisect

    if not frame_ids:
        raise ValueError("No available frames")
    pos = bisect.bisect_left(frame_ids, frame_id)
    if pos <= 0:
        return frame_ids[0]
    if pos >= len(frame_ids):
        return frame_ids[-1]
    before = frame_ids[pos - 1]
    after = frame_ids[pos]
    return before if abs(frame_id - before) <= abs(after - frame_id) else after


def calibrate_future_pos_origin(lookup: dict[tuple[str, int], dict[str, Any]]) -> str:
    import numpy as np

    values = []
    for key, source in lookup.items():
        future_x = parse_float_array(source["future_pos_x"], FUTURE_STATE_LEN, "future_pos_x", key)
        future_y = parse_float_array(source["future_pos_y"], FUTURE_STATE_LEN, "future_pos_y", key)
        values.append(abs(future_x[0]) + abs(future_y[0]))
    median_first_displacement = float(np.median(np.asarray(values, dtype=np.float32)))
    return "includes_t0" if median_first_displacement < 0.25 else "starts_at_0p1s"


def horizon_index(seconds: float, future_pos_time_origin: str) -> int:
    if future_pos_time_origin == "includes_t0":
        idx = round(seconds * SOURCE_FPS)
    elif future_pos_time_origin == "starts_at_0p1s":
        idx = round(seconds * SOURCE_FPS) - 1
    else:
        raise ValueError(f"Unknown future_pos_time_origin={future_pos_time_origin}")
    return max(0, min(FUTURE_STATE_LEN - 1, int(idx)))


def compact_model_actions(full_actions: Any, future_pos_time_origin: str):
    import numpy as np

    full = np.asarray(full_actions, dtype=np.float32)
    actions = np.empty((full.shape[0], MODEL_ACTION_DIM), dtype=np.float32)
    speed_idx = 104
    yaw_idx = 105
    camera_linear_start = 106
    camera_angular_start = 109
    past_accel_x_last = 87
    past_accel_y_last = 103
    horizon_indices = [
        horizon_index(0.5, future_pos_time_origin),
        horizon_index(1.0, future_pos_time_origin),
        horizon_index(1.5, future_pos_time_origin),
        horizon_index(2.0, future_pos_time_origin),
    ]

    actions[:, 0] = full[:, speed_idx]
    actions[:, 1] = full[:, yaw_idx]
    actions[:, 2:5] = full[:, camera_linear_start : camera_linear_start + 3]
    actions[:, 5:8] = full[:, camera_angular_start : camera_angular_start + 3]
    actions[:, 8] = full[:, past_accel_x_last]
    actions[:, 9] = full[:, past_accel_y_last]
    out_col = 10
    for idx in horizon_indices:
        actions[:, out_col] = full[:, idx]
        actions[:, out_col + 1] = full[:, FUTURE_STATE_LEN + idx]
        out_col += 2
    return actions


def build_frame_actions_for_window(
    row: dict[str, str],
    lookup: dict[tuple[str, int], dict[str, Any]],
    available_frames: dict[str, list[int]],
    future_pos_time_origin: str,
) -> dict[str, Any]:
    import numpy as np

    scenario_id = row["scenario_id"]
    source_min = int(row["source_min_frame_id_10fps"])
    source_max = int(row["source_max_frame_id_10fps"])
    start_24 = int(row["start_frame_24fps"])
    full_actions = np.empty((TOTAL_FRAMES, len(FULL_ACTION_FEATURE_ORDER)), dtype=np.float32)
    source_float = np.empty((TOTAL_FRAMES,), dtype=np.float32)
    source_frame_0 = np.empty((TOTAL_FRAMES,), dtype=np.int32)
    source_frame_1 = np.empty((TOTAL_FRAMES,), dtype=np.int32)
    alpha_values = np.empty((TOTAL_FRAMES,), dtype=np.float32)
    status_codes = np.zeros((TOTAL_FRAMES,), dtype=np.int8)

    frame_ids = available_frames.get(scenario_id, [])
    if not frame_ids:
        raise RuntimeError(f"No source metadata rows loaded for scenario {scenario_id}")

    for j in range(TOTAL_FRAMES):
        abs_frame_24 = start_24 + j
        frame_float = source_min + abs_frame_24 * SOURCE_FPS / FPS
        f0 = max(source_min, min(source_max, math.floor(frame_float)))
        f1 = max(source_min, min(source_max, math.ceil(frame_float)))
        alpha = frame_float - math.floor(frame_float)

        key0 = (scenario_id, f0)
        key1 = (scenario_id, f1)
        status = 0
        if key0 not in lookup:
            f0 = nearest_available_frame(frame_ids, f0)
            key0 = (scenario_id, f0)
            status = 1
        if key1 not in lookup:
            f1 = nearest_available_frame(frame_ids, f1)
            key1 = (scenario_id, f1)
            status = 1
        if f0 == f1:
            alpha = 0.0

        v0 = np.asarray(full_action_vector_from_source(lookup[key0], key0), dtype=np.float32)
        v1 = np.asarray(full_action_vector_from_source(lookup[key1], key1), dtype=np.float32)
        full_actions[j] = (1.0 - alpha) * v0 + alpha * v1
        source_float[j] = frame_float
        source_frame_0[j] = f0
        source_frame_1[j] = f1
        alpha_values[j] = alpha
        status_codes[j] = status

    actions = compact_model_actions(full_actions, future_pos_time_origin)
    if actions.shape != (TOTAL_FRAMES, MODEL_ACTION_DIM):
        raise AssertionError(f"actions shape {actions.shape} != {(TOTAL_FRAMES, MODEL_ACTION_DIM)}")
    if full_actions.shape != (TOTAL_FRAMES, len(FULL_ACTION_FEATURE_ORDER)):
        raise AssertionError(f"full action shape {full_actions.shape} is invalid")
    if not np.isfinite(actions).all() or not np.isfinite(full_actions).all():
        raise ValueError(f"Non-finite actions for {row['window_id']}")

    boundary_float = source_min + (start_24 + CONTEXT_FRAMES - 1) * SOURCE_FPS / FPS
    old_boundary_frame = max(source_min, min(source_max, round(boundary_float)))

    return {
        "actions": actions,
        "actions_full_112": full_actions,
        "frame_timestamps_sec": np.arange(TOTAL_FRAMES, dtype=np.float32) / FPS,
        "source_frame_float": source_float,
        "source_frame_0": source_frame_0,
        "source_frame_1": source_frame_1,
        "alpha": alpha_values,
        "interpolation_status_code": status_codes,
        "linear_interpolation_frames": int(np.sum(status_codes == 0)),
        "nearest_due_missing_frames": int(np.sum(status_codes == 1)),
        "old_context_boundary_source_frame_id_10fps": int(old_boundary_frame),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def compute_and_write_normalization_stats(actions_for_stats: list[Any], future_pos_time_origin: str) -> dict[str, Any]:
    import numpy as np

    stacked = np.concatenate(actions_for_stats, axis=0).astype(np.float32)
    stats = {
        "feature_order": MODEL_ACTION_FEATURE_ORDER,
        "mean": np.mean(stacked, axis=0).astype(float).tolist(),
        "std": np.std(stacked, axis=0).astype(float).tolist(),
        "p01": np.percentile(stacked, 1, axis=0).astype(float).tolist(),
        "p99": np.percentile(stacked, 99, axis=0).astype(float).tolist(),
        "num_train_windows": len(actions_for_stats),
        "num_train_action_frames": int(stacked.shape[0]),
        "action_shape_per_window": [TOTAL_FRAMES, MODEL_ACTION_DIM],
        "future_pos_time_origin": future_pos_time_origin,
        "clip_after_normalization": [-5.0, 5.0],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(DATA_ROOT / STATS_RELPATH, stats)
    return stats


def load_future_pos_time_origin_from_stats() -> str | None:
    stats_path = DATA_ROOT / STATS_RELPATH
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return str(stats.get("future_pos_time_origin") or "")


def build_frame_action_conditions_for_split(split: str, *, force_future_pos_time_origin: str = "") -> dict[str, Any]:
    import numpy as np

    data_volume.reload()
    manifest_rows = read_window_manifest(split)
    if not manifest_rows:
        raise RuntimeError(f"No rows in {split} 24 FPS window manifest")
    needed_scenarios = {row["scenario_id"] for row in manifest_rows}

    with tempfile.TemporaryDirectory(prefix=f"waymo24_frame_action_{split}_") as tmp:
        parquet_path = Path(tmp) / f"front_frames_{split}_clean.parquet"
        download_gcs_file(frame_metadata_uri(split), parquet_path)
        lookup, available_frames = load_source_lookup(parquet_path, needed_scenarios)

    if set(available_frames) != needed_scenarios:
        missing = sorted(needed_scenarios - set(available_frames))
        raise RuntimeError(f"Missing metadata for {len(missing)} {split} scenarios; first={missing[:5]}")

    if force_future_pos_time_origin:
        future_pos_time_origin = force_future_pos_time_origin
    elif split != "train":
        future_pos_time_origin = load_future_pos_time_origin_from_stats() or calibrate_future_pos_origin(lookup)
    else:
        future_pos_time_origin = calibrate_future_pos_origin(lookup)

    manifest_records: list[FrameActionConditionRecord] = []
    actions_for_stats: list[Any] = []
    total_nearest = 0
    total_linear = 0
    native_reconstruction_checks = 0
    native_reconstruction_failures = 0

    for idx, row in enumerate(manifest_rows, start=1):
        if int(row["fps"]) != FPS or int(row["num_frames"]) != TOTAL_FRAMES:
            raise ValueError(f"Unexpected window metadata for {row['window_id']}: fps={row['fps']} frames={row['num_frames']}")
        built = build_frame_actions_for_window(row, lookup, available_frames, future_pos_time_origin)
        action_relpath = f"{FRAME_ACTION_ROOT}/{split}/{row['scenario_id']}/{row['window_id']}.npz"
        out_path = DATA_ROOT / action_relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            actions=built["actions"],
            actions_full_112=built["actions_full_112"],
            frame_timestamps_sec=built["frame_timestamps_sec"],
            source_frame_float=built["source_frame_float"],
            source_frame_0=built["source_frame_0"],
            source_frame_1=built["source_frame_1"],
            alpha=built["alpha"],
            interpolation_status_code=built["interpolation_status_code"],
        )

        exact_indices = np.where(np.abs(built["alpha"]) < 1e-6)[0]
        for frame_idx in exact_indices:
            native_reconstruction_checks += 1
            source_key = (row["scenario_id"], int(built["source_frame_0"][frame_idx]))
            native = np.asarray(full_action_vector_from_source(lookup[source_key], source_key), dtype=np.float32)
            if not np.allclose(native, built["actions_full_112"][frame_idx], atol=1e-5):
                native_reconstruction_failures += 1

        if split == "train":
            actions_for_stats.append(built["actions"])
        total_linear += built["linear_interpolation_frames"]
        total_nearest += built["nearest_due_missing_frames"]
        manifest_records.append(
            FrameActionConditionRecord(
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
                end_frame_24fps=int(row["end_frame_24fps"]),
                source_min_frame_id_10fps=int(row["source_min_frame_id_10fps"]),
                source_max_frame_id_10fps=int(row["source_max_frame_id_10fps"]),
                frame_action_dim=MODEL_ACTION_DIM,
                frame_action_shape_json=json.dumps([TOTAL_FRAMES, MODEL_ACTION_DIM], separators=(",", ":")),
                frame_action_feature_order_json=json.dumps(MODEL_ACTION_FEATURE_ORDER, separators=(",", ":")),
                full_action_dim=len(FULL_ACTION_FEATURE_ORDER),
                full_action_feature_order_json=json.dumps(FULL_ACTION_FEATURE_ORDER, separators=(",", ":")),
                frame_action_relpath=action_relpath,
                mp4_relpath=row["mp4_relpath"],
                latent_relpath=row["latent_relpath"],
                linear_interpolation_frames=built["linear_interpolation_frames"],
                nearest_due_missing_frames=built["nearest_due_missing_frames"],
                future_pos_time_origin=future_pos_time_origin,
            )
        )
        if idx % 500 == 0:
            print(f"[{split}] wrote {idx}/{len(manifest_rows)} frame-action windows")

    if native_reconstruction_failures:
        raise RuntimeError(
            f"{split}: {native_reconstruction_failures}/{native_reconstruction_checks} native reconstruction checks failed"
        )

    fieldnames = list(FrameActionConditionRecord.__dataclass_fields__.keys())
    row_dicts = [asdict(record) for record in manifest_records]
    manifest_csv = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f_frame_action_conditions.csv"
    manifest_jsonl = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f_frame_action_conditions.jsonl"
    write_csv(manifest_csv, row_dicts, fieldnames)
    write_jsonl(manifest_jsonl, row_dicts)

    if split == "train":
        stats = compute_and_write_normalization_stats(actions_for_stats, future_pos_time_origin)
    else:
        stats = {}

    missing_mp4 = [record.mp4_relpath for record in manifest_records if not (DATA_ROOT / record.mp4_relpath).exists()]
    missing_latents = [record.latent_relpath for record in manifest_records if not (DATA_ROOT / record.latent_relpath).exists()]
    missing_actions = [record.frame_action_relpath for record in manifest_records if not (DATA_ROOT / record.frame_action_relpath).exists()]
    if missing_mp4 or missing_latents or missing_actions:
        raise RuntimeError(
            f"{split}: missing files after build: mp4={len(missing_mp4)} latent={len(missing_latents)} action={len(missing_actions)}"
        )

    summary = {
        "split": split,
        "windows": len(manifest_records),
        "scenarios": len({record.scenario_id for record in manifest_records}),
        "fps": FPS,
        "source_fps": SOURCE_FPS,
        "num_frames": TOTAL_FRAMES,
        "context_frames": CONTEXT_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "frame_action_dim": MODEL_ACTION_DIM,
        "frame_action_shape": [TOTAL_FRAMES, MODEL_ACTION_DIM],
        "full_action_shape": [TOTAL_FRAMES, len(FULL_ACTION_FEATURE_ORDER)],
        "frame_action_feature_order": MODEL_ACTION_FEATURE_ORDER,
        "full_action_feature_order": list(FULL_ACTION_FEATURE_ORDER),
        "future_pos_time_origin": future_pos_time_origin,
        "manifest_csv": str(manifest_csv.relative_to(DATA_ROOT)),
        "manifest_jsonl": str(manifest_jsonl.relative_to(DATA_ROOT)),
        "normalization_stats": STATS_RELPATH if split == "train" else "",
        "linear_interpolation_frames": total_linear,
        "nearest_due_missing_frames": total_nearest,
        "native_reconstruction_checks": native_reconstruction_checks,
        "native_reconstruction_failures": native_reconstruction_failures,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_stats_preview": {
            "mean": stats.get("mean", [])[:4],
            "std": stats.get("std", [])[:4],
        },
    }
    write_json(DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f_frame_action_conditions_summary.json", summary)
    data_volume.commit()
    return summary


def evenly_sample_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    import numpy as np

    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int).tolist()
    return [rows[idx] for idx in indices]


def load_frame_action_manifest(split: str) -> list[dict[str, str]]:
    path = DATA_ROOT / "manifests" / f"{split}_windows_24fps_121f_frame_action_conditions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scale_value(value: float, min_value: float, max_value: float, y0: int, y1: int) -> int:
    if abs(max_value - min_value) < 1e-12:
        return (y0 + y1) // 2
    frac = (value - min_value) / (max_value - min_value)
    return int(round(y1 - frac * (y1 - y0)))


def render_overlay(local_mp4: Path, action_npz: Path, out_mp4: Path, title: str) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    data = np.load(action_npz)
    actions = data["actions"]
    source_float = data["source_frame_float"]
    status = data["interpolation_status_code"]
    if actions.shape != (TOTAL_FRAMES, MODEL_ACTION_DIM):
        raise ValueError(f"{action_npz} has action shape {actions.shape}")

    curves = [
        ("speed", actions[:, 0], (244, 116, 59)),
        ("accel_x", actions[:, 8], (46, 134, 171)),
        ("yaw_rate", actions[:, 1], (95, 168, 95)),
        ("future_y_1s", actions[:, 13], (160, 99, 173)),
    ]
    curve_ranges = []
    for _, values, _ in curves:
        lo = float(np.percentile(values, 2))
        hi = float(np.percentile(values, 98))
        if abs(hi - lo) < 1e-6:
            hi = lo + 1.0
        curve_ranges.append((lo, hi))

    font = ImageFont.load_default()
    reader = imageio.get_reader(str(local_mp4))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    try:
        for frame_idx, frame in enumerate(reader):
            if frame_idx >= TOTAL_FRAMES:
                break
            frame_img = Image.fromarray(frame).convert("RGB").resize((512, 512))
            canvas = Image.new("RGB", (512, 736), (14, 17, 20))
            canvas.paste(frame_img, (0, 0))
            draw = ImageDraw.Draw(canvas)

            draw.rectangle((0, 0, 512, 52), fill=(0, 0, 0))
            draw.text((8, 6), title[:78], fill=(255, 255, 255), font=font)
            draw.text(
                (8, 25),
                (
                    f"frame={frame_idx:03d}/120 t={frame_idx / FPS:.3f}s "
                    f"speed={actions[frame_idx, 0]:.2f} accel_x={actions[frame_idx, 8]:.2f} "
                    f"yaw={actions[frame_idx, 1]:.3f} src={source_float[frame_idx]:.2f} "
                    f"interp={'nearest' if int(status[frame_idx]) else 'linear'}"
                ),
                fill=(230, 230, 230),
                font=font,
            )
            draw.line((0, 512, 512, 512), fill=(240, 240, 240), width=2)
            draw.text((8, 518), "context", fill=(210, 210, 210), font=font)
            draw.text((218, 518), "| future starts at frame 49", fill=(255, 204, 102), font=font)

            chart_left, chart_top, chart_right, chart_bottom = 32, 552, 496, 704
            draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(75, 82, 90), width=1)
            x_boundary = chart_left + round((CONTEXT_FRAMES - 1) * (chart_right - chart_left) / (TOTAL_FRAMES - 1))
            x_current = chart_left + round(frame_idx * (chart_right - chart_left) / (TOTAL_FRAMES - 1))
            draw.line((x_boundary, chart_top, x_boundary, chart_bottom), fill=(255, 204, 102), width=1)
            draw.line((x_current, chart_top, x_current, chart_bottom), fill=(255, 255, 255), width=1)

            for curve_idx, ((name, values, color), (lo, hi)) in enumerate(zip(curves, curve_ranges)):
                points = []
                for j, value in enumerate(values):
                    x = chart_left + round(j * (chart_right - chart_left) / (TOTAL_FRAMES - 1))
                    y = scale_value(float(value), lo, hi, chart_top, chart_bottom)
                    points.append((x, y))
                draw.line(points, fill=color, width=2)
                draw.text((chart_left, 710 + curve_idx * 6), "", fill=color, font=font)
                legend_x = 38 + curve_idx * 116
                draw.rectangle((legend_x, 714, legend_x + 10, 724), fill=color)
                draw.text((legend_x + 14, 712), name, fill=(230, 230, 230), font=font)

            writer.append_data(np.asarray(canvas))
    finally:
        reader.close()
        writer.close()


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[gcp_secret],
)
def build_frame_action_conditions_modal(split: str, force_future_pos_time_origin: str = "") -> dict[str, Any]:
    return build_frame_action_conditions_for_split(split, force_future_pos_time_origin=force_future_pos_time_origin)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=20 * 60,
    max_containers=16,
    volumes={str(DATA_ROOT): data_volume},
)
def render_one_overlay_modal(split: str, row: dict[str, str]) -> dict[str, Any]:
    data_volume.reload()
    mp4_path = DATA_ROOT / row["mp4_relpath"]
    action_path = DATA_ROOT / row["frame_action_relpath"]
    if not mp4_path.exists():
        raise FileNotFoundError(mp4_path)
    if not action_path.exists():
        raise FileNotFoundError(action_path)
    out_relpath = f"{AUDIT_ROOT}/{split}/{row['window_id']}_overlay.mp4"
    out_path = DATA_ROOT / out_relpath
    render_overlay(mp4_path, action_path, out_path, f"{split} {row['window_id']}")
    data_volume.commit()
    return {
        "split": split,
        "window_id": row["window_id"],
        "overlay_relpath": out_relpath,
    }


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def select_overlay_rows_modal(split: str, count: int) -> list[dict[str, str]]:
    data_volume.reload()
    return evenly_sample_rows(load_frame_action_manifest(split), count)


def render_overlays_parallel(split: str, count: int) -> dict[str, Any]:
    rows = select_overlay_rows_modal.remote(split, count)
    outputs = []
    if not rows:
        return {"split": split, "requested": count, "rendered": 0, "overlays": []}
    inputs = [(split, row) for row in rows]
    for result in render_one_overlay_modal.starmap(inputs, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            raise result
        outputs.append(result)
        print(f"[{split}] rendered overlay {len(outputs)}/{len(rows)}: {result['window_id']}")
    outputs.sort(key=lambda item: item["window_id"])
    summary = {
        "split": split,
        "requested": count,
        "rendered": len(outputs),
        "overlays": outputs,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        temp_path = Path(handle.name)
    # Upload the local summary through a tiny Modal function to keep all artifacts in the volume.
    write_overlay_summary_modal.remote(split, temp_path.read_text(encoding="utf-8"))
    return summary


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=5 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def write_overlay_summary_modal(split: str, summary_json: str) -> None:
    data_volume.reload()
    path = DATA_ROOT / AUDIT_ROOT / split / "overlay_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_json, encoding="utf-8")
    data_volume.commit()


@app.local_entrypoint()
def main(
    splits_csv: str = "train,val",
    train_overlay_count: int = 50,
    val_overlay_count: int = 20,
    force_future_pos_time_origin: str = "",
    skip_build: bool = False,
    skip_overlays: bool = False,
) -> None:
    splits = [item.strip() for item in splits_csv.split(",") if item.strip()]
    summaries: dict[str, Any] = {}
    if not skip_build:
        for split in splits:
            force_origin = force_future_pos_time_origin
            if split != "train" and not force_origin:
                # Val/test should use the train-calibrated future-position origin written by the train split.
                force_origin = ""
            summaries[f"{split}_frame_actions"] = build_frame_action_conditions_modal.remote(split, force_origin)

    if not skip_overlays:
        if "train" in splits:
            summaries["train_overlays"] = render_overlays_parallel("train", train_overlay_count)
        if "val" in splits:
            summaries["val_overlays"] = render_overlays_parallel("val", val_overlay_count)

    print(json.dumps(summaries, indent=2, sort_keys=True))
