"""Stricter Phase 2 v2 temporal bottleneck action-conditioning wrapper.

This keeps the corrected shifted-lognormal no-action visual LoRA frozen and
trains only the temporal action bottleneck, residual projectors, and gates.
"""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framebottleneck-hfteacher-v2-action-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault("WAYMO24_ACTION_ENCODER_TYPE", "frame_temporal_bottleneck_hf_teacher")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")
os.environ.setdefault("WAYMO24_FREEZE_TRANSFORMER_LORA", "1")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framebneck-hft-v2-r16-ckpts",
)
os.environ.setdefault(
    "LTX_BASELINE_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts",
)
os.environ.setdefault(
    "WAYMO24_BASELINE_LORA_RUN_NAME",
    "ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000",
)
os.environ.setdefault("WAYMO24_BASELINE_LORA_STEP", "step_003000")

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
