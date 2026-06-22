"""Generate 24 FPS Waymo futures from Phase 2 v2 temporal-bottleneck action LoRA."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framebottleneck-hfteacher-v2-action-infer")
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framebneck-hft-v2-r16-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framebneck-hft-v2-infer",
)
os.environ.setdefault(
    "LTX_LORA_RUN_NAME",
    "ltx2b_dist098_waymo24_frame_temporal_bottleneck_hfteacher_v2_seed231_from_shifted_noaction_step003000_steps1000",
)
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "distilled098_framebottleneck_hfteacher_v2_action_lora_24fps_minterpolate_seed231_runs",
)
os.environ.setdefault("LTX_IMAGE_COND_NOISE_SCALE", "0.0")

from pipelines.inference.generate_waymo24_action_minterpolate_lora import *  # noqa: F401,F403
