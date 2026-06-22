"""Frame-aligned action-token Transformer LoRA training for distilled LTX-2B."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framexf-action-lora-r16-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault("WAYMO24_ACTION_ENCODER_TYPE", "frame_transformer")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framexf-action-lora-r16-ckpts",
)
os.environ.setdefault(
    "LTX_BASELINE_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-ckpts",
)
os.environ.setdefault(
    "WAYMO24_BASELINE_LORA_RUN_NAME",
    "ltx2b_distilled098_waymo24_noaction_visual_lora_r16_seed231_full7992_lr5e6_2epochs_steps15984",
)
os.environ.setdefault("WAYMO24_BASELINE_LORA_STEP", "step_010000")

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
