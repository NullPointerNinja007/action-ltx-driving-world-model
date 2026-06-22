"""Compatibility wrapper for distilled LTX-2B Waymo tiny-transformer action LoRA training."""

import os

os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault("WAYMO24_ACTION_ENCODER_TYPE", "tiny_transformer")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-tinyxf-action-lora-r16-ckpts",
)

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
