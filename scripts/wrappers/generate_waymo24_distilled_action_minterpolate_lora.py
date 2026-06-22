"""Compatibility wrapper for distilled numeric action-conditioned LTX-2B Waymo LoRA inference."""

import os

os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-distilled098-waymo24fps-action-lora-r16-checkpoints",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-action-lora-infer",
)
os.environ.setdefault("LTX_RUNS_ROOT", "distilled098_action_lora_24fps_minterpolate_seed231_runs")

from pipelines.inference.generate_waymo24_action_minterpolate_lora import *  # noqa: F401,F403
