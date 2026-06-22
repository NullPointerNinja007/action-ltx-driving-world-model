"""Compatibility wrapper for LTX-2B Waymo numeric action-conditioned LoRA training."""

import os

os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-waymo24fps-action-lora-r16-checkpoints",
)

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
