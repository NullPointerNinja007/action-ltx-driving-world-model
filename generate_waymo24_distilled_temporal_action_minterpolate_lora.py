"""Compatibility wrapper for distilled temporal action-conditioned LTX-2B Waymo LoRA inference."""

import os

os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-temporal-action-lora-r16-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-temporal-action-lora-infer",
)
os.environ.setdefault("LTX_RUNS_ROOT", "distilled098_temporal_action_lora_24fps_minterpolate_seed231_runs")

from pipelines.inference.generate_waymo24_action_minterpolate_lora import *  # noqa: F401,F403
