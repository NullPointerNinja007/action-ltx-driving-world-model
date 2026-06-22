"""Short no-action visual LoRA ablation with LTX-style shifted log-normal timestep sampling."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts",
)
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
