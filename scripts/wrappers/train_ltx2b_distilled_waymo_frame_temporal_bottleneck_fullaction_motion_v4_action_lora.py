"""V4 full-112 low-frequency motion action-conditioning wrapper.

This pilot starts from the corrected shifted-lognormal no-action visual LoRA at
step_003000 and freezes it. The trainable path is only the 112-dim frame-action
temporal encoder, temporal residual projectors, bounded gates, and the action
motion auxiliary head.
"""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-action-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("WAYMO24_ACTION_CONDITIONING", "1")
os.environ.setdefault("WAYMO24_ACTION_ENCODER_TYPE", "frame_temporal_bottleneck_fullaction_motion_v4")
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")
os.environ.setdefault("WAYMO24_FREEZE_TRANSFORMER_LORA", "1")
os.environ.setdefault("WAYMO24_FRAME_ACTION_FEATURE_KEY", "actions_full_112")
os.environ.setdefault(
    "WAYMO24_FRAME_ACTION_STATS_RELPATH",
    "manifests/frame_action_24fps_full112_normalization_stats.json",
)
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-r16-ckpts",
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
