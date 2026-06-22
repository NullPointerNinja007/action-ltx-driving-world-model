"""Compatibility wrapper for distilled LTX-2B Waymo latent caching."""

import os

os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents_distilled098")

from pipelines.data.prepare_waymo24_visual_data import *  # noqa: F401,F403
