from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "waymo24-full112-action-stats"
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
DATA_ROOT = Path("/data")
DEFAULT_MANIFEST = "manifests/train_windows_24fps_121f_frame_action_conditions.csv"
DEFAULT_OUTPUT = "manifests/frame_action_24fps_full112_normalization_stats.json"


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def compute_full112_stats(
    manifest_relpath: str = DEFAULT_MANIFEST,
    output_relpath: str = DEFAULT_OUTPUT,
    limit: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np

    data_volume.reload()
    output_path = DATA_ROOT / output_relpath
    if output_path.exists() and not force:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "status": "exists",
            "output_relpath": output_relpath,
            "num_rows": payload.get("num_rows"),
            "num_values": payload.get("num_values"),
            "feature_dim": len(payload.get("mean", [])),
        }

    manifest_path = DATA_ROOT / manifest_relpath
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frame-action manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("No rows found for full112 action stats.")

    chunks = []
    feature_order = None
    for row in rows:
        if len(chunks) > 0 and len(chunks) % 250 == 0:
            print(json.dumps({"loaded_rows": len(chunks), "total_rows": len(rows)}), flush=True)
        if feature_order is None and row.get("full_action_feature_order_json"):
            feature_order = json.loads(row["full_action_feature_order_json"])
        action_path = DATA_ROOT / row["frame_action_relpath"]
        if not action_path.exists():
            raise FileNotFoundError(f"Missing frame-action cache: {action_path}")
        with np.load(action_path) as payload:
            if "actions_full_112" not in payload:
                raise KeyError(f"Missing actions_full_112 in {action_path}; keys={list(payload.keys())}")
            actions = payload["actions_full_112"].astype("float32")
        if actions.ndim != 2 or actions.shape[1] != 112:
            raise ValueError(f"Expected actions_full_112 shape [T,112], got {actions.shape} for {action_path}")
        chunks.append(actions)

    stacked = np.concatenate(chunks, axis=0)
    std = stacked.std(axis=0).astype("float64")
    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_relpath": manifest_relpath,
        "output_relpath": output_relpath,
        "action_key": "actions_full_112",
        "stats_mode": "full" if limit <= 0 else f"first_{limit}_rows",
        "num_rows": len(rows),
        "num_values": int(stacked.shape[0]),
        "feature_dim": int(stacked.shape[1]),
        "feature_order": feature_order or [f"actions_full_112_{idx:03d}" for idx in range(stacked.shape[1])],
        "mean": stacked.mean(axis=0).astype("float64").tolist(),
        "std": np.maximum(std, 1e-6).tolist(),
        "p01": np.percentile(stacked, 1, axis=0).astype("float64").tolist(),
        "p99": np.percentile(stacked, 99, axis=0).astype("float64").tolist(),
    }
    save_json(output_path, stats)
    data_volume.commit()
    return {
        "status": "written",
        "output_relpath": output_relpath,
        "num_rows": len(rows),
        "num_values": int(stacked.shape[0]),
        "feature_dim": int(stacked.shape[1]),
    }


@app.local_entrypoint()
def main(limit: int = 0, force: bool = False) -> None:
    print(json.dumps(compute_full112_stats.remote(limit=limit, force=force), indent=2, sort_keys=True))
