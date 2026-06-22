"""Generate 24 FPS Waymo futures from V3 low-frequency temporal bottleneck action LoRA."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framebneck-lowfreq-v3-action-infer")
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framebneck-lowfreq-v3-r16-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-framebneck-lowfreq-v3-infer",
)
os.environ.setdefault(
    "LTX_LORA_RUN_NAME",
    "ltx2b_dist098_waymo24_frame_temporal_bottleneck_lowfreq_v3b_gate1e4_proj1e3_seed231_from_shifted_noaction_step003000_steps3000",
)
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "distilled098_framebneck_lowfreq_v3_action_lora_24fps_minterpolate_seed231_runs",
)
os.environ.setdefault("LTX_IMAGE_COND_NOISE_SCALE", "0.0")

from pipelines.inference.generate_waymo24_action_minterpolate_lora import *  # noqa: F401,F403
