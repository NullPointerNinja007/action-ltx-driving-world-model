"""Middle-block gated frame-action cross-attention training for distilled LTX-2B.

This wrapper intentionally starts from the corrected shifted-lognormal no-action
visual LoRA and freezes that visual LoRA, so action gradients only train the
frame action encoder and middle-block action injector.
"""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framemidxattn-action-lora-r16-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault("WAYMO24_ACTION_ENCODER_TYPE", "frame_midblock_gated_xattn")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")
os.environ.setdefault("WAYMO24_FREEZE_TRANSFORMER_LORA", "1")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts",
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
