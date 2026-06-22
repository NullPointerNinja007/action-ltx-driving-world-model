from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "phase1-midblock-gate-value-extract"
CHECKPOINT_VOLUME = os.environ.get("LTX_GATE_CHECKPOINT_VOLUME", "ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts")
CHECKPOINT_ROOT = Path("/checkpoints")

app = modal.App(APP_NAME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME)
image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")


def step_to_int(step: str) -> int:
    if step == "step_000000_base_reference":
        return 0
    return int(step.removeprefix("step_"))


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={str(CHECKPOINT_ROOT): checkpoint_volume},
)
def extract_gate_rows(run_name: str, checkpoints: list[str]) -> dict[str, Any]:
    import torch

    checkpoint_volume.reload()
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        injector_path = CHECKPOINT_ROOT / run_name / checkpoint / "action_injector.pt"
        if not injector_path.exists():
            rows.append(
                {
                    "checkpoint_name": checkpoint,
                    "checkpoint_step": step_to_int(checkpoint),
                    "enabled_block_indices": "",
                    "gate_name": "",
                    "gate_value": "",
                    "abs_gate_value": "",
                    "mean_abs_gate": "",
                    "max_abs_gate": "",
                    "missing_action_injector": True,
                }
            )
            continue
        payload = torch.load(injector_path, map_location="cpu")
        metadata = dict(payload.get("metadata") or {})
        state_dict = payload.get("state_dict") or {}
        block_indices = [int(idx) for idx in metadata.get("block_indices", [])]
        gate_values: list[tuple[str, float]] = []
        for key, value in state_dict.items():
            if key.startswith("gates."):
                gate_values.append((key, float(value.reshape(()).item())))
        gate_values.sort(key=lambda item: int(item[0].split(".")[-1]))
        abs_values = [abs(value) for _, value in gate_values]
        mean_abs = sum(abs_values) / len(abs_values) if abs_values else 0.0
        max_abs = max(abs_values) if abs_values else 0.0
        for gate_name, gate_value in gate_values:
            rows.append(
                {
                    "checkpoint_name": checkpoint,
                    "checkpoint_step": step_to_int(checkpoint),
                    "enabled_block_indices": json.dumps(block_indices, separators=(",", ":")),
                    "gate_name": gate_name,
                    "gate_value": gate_value,
                    "abs_gate_value": abs(gate_value),
                    "mean_abs_gate": mean_abs,
                    "max_abs_gate": max_abs,
                    "missing_action_injector": False,
                }
            )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_volume": CHECKPOINT_VOLUME,
        "run_name": run_name,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@app.local_entrypoint()
def main(
    run_name: str,
    checkpoints_json: str,
    output_csv: str,
    output_json: str = "",
) -> None:
    checkpoints = json.loads(checkpoints_json)
    result = extract_gate_rows.remote(run_name, checkpoints)
    write_csv(Path(output_csv), result["rows"])
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(result["rows"]), "output_csv": output_csv, "output_json": output_json}, sort_keys=True))
