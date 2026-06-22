from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "waymo24-full112-action-audit"
DATA_VOLUME_NAME = "waymo-e2e-24fps-121f-visual-continuation-data"
DATA_ROOT = Path("/data")
TRAIN_MANIFEST = "manifests/train_windows_24fps_121f_frame_action_conditions.csv"
VAL_MANIFEST = "manifests/val_windows_24fps_121f_frame_action_conditions.csv"
STATS_RELPATH = "manifests/frame_action_24fps_full112_normalization_stats.json"
AUDIT_RELPATH = "action_frame_audits/v4_full112_action_alignment_audit.json"
TOTAL_FRAMES = 121
ACTION_DIM = 112


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy")


def replace_latent_prefix(latent_relpath: str, latent_prefix: str) -> str:
    parts = Path(latent_relpath).parts
    if not parts:
        raise ValueError("Empty latent_relpath")
    return str(Path(latent_prefix, *parts[1:]))


def load_rows(relpath: str, limit: int = 0) -> list[dict[str, str]]:
    path = DATA_ROOT / relpath
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit > 0 else rows


def audit_rows(rows: list[dict[str, str]], *, split: str, latent_prefix: str) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    checked = 0
    sample_action_norms = []
    for row in rows:
        checked += 1
        window_id = row.get("window_id", "")
        scenario_id = row.get("scenario_id", "")
        if scenario_id and scenario_id[:12] not in window_id:
            errors.append(f"{split}:{window_id}: scenario_id prefix not present in window_id")
        latent_relpath = row.get("latent_relpath", "")
        if latent_relpath:
            latent_path = DATA_ROOT / replace_latent_prefix(latent_relpath, latent_prefix)
            if not latent_path.exists():
                errors.append(f"{split}:{window_id}: missing latent {latent_path}")
        action_relpath = row.get("frame_action_relpath", "")
        if not action_relpath:
            errors.append(f"{split}:{window_id}: empty frame_action_relpath")
            continue
        action_path = DATA_ROOT / action_relpath
        if not action_path.exists():
            errors.append(f"{split}:{window_id}: missing action npz {action_path}")
            continue
        with np.load(action_path) as payload:
            if "actions_full_112" not in payload:
                errors.append(f"{split}:{window_id}: missing actions_full_112 key; keys={list(payload.keys())}")
                continue
            actions = payload["actions_full_112"].astype("float32")
            if "actions" not in payload:
                errors.append(f"{split}:{window_id}: missing compact actions key")
        if actions.shape != (TOTAL_FRAMES, ACTION_DIM):
            errors.append(f"{split}:{window_id}: actions_full_112 shape={actions.shape}, expected={(TOTAL_FRAMES, ACTION_DIM)}")
            continue
        if not np.isfinite(actions).all():
            errors.append(f"{split}:{window_id}: non-finite action values")
        sample_action_norms.append(float(np.linalg.norm(actions, axis=1).mean()))
    return {
        "split": split,
        "checked_rows": checked,
        "num_errors": len(errors),
        "errors": errors[:200],
        "mean_full112_row_norm": float(sum(sample_action_norms) / max(len(sample_action_norms), 1)),
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={str(DATA_ROOT): data_volume},
)
def run_audit(train_limit: int = 512, val_limit: int = 128, latent_prefix: str = "latents") -> dict[str, Any]:
    data_volume.reload()
    stats_path = DATA_ROOT / STATS_RELPATH
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing full112 stats: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in ("mean", "std", "p01", "p99"):
        if len(stats.get(key, [])) != ACTION_DIM:
            raise ValueError(f"Stats key {key} has length {len(stats.get(key, []))}, expected {ACTION_DIM}.")
    if stats.get("action_key") not in {"actions_full_112", None}:
        raise ValueError(f"Unexpected stats action_key: {stats.get('action_key')}")

    train_rows = load_rows(TRAIN_MANIFEST, train_limit)
    val_rows = load_rows(VAL_MANIFEST, val_limit)
    train_report = audit_rows(train_rows, split="train", latent_prefix=latent_prefix)
    val_report = audit_rows(val_rows, split="val", latent_prefix=latent_prefix)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats_relpath": STATS_RELPATH,
        "latent_prefix": latent_prefix,
        "train": train_report,
        "val": val_report,
        "passed": train_report["num_errors"] == 0 and val_report["num_errors"] == 0,
    }
    out_path = DATA_ROOT / AUDIT_RELPATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    data_volume.commit()
    return payload


@app.local_entrypoint()
def main(train_limit: int = 512, val_limit: int = 128, latent_prefix: str = "latents") -> None:
    print(json.dumps(run_audit.remote(train_limit=train_limit, val_limit=val_limit, latent_prefix=latent_prefix), indent=2, sort_keys=True))
