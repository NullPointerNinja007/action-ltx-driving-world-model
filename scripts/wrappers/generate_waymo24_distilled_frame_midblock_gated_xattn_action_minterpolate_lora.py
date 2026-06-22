"""Inference wrapper for distilled LTX-2B middle-block gated frame-action LoRA."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-framemidxattn-action-lora-infer")
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault("LTX_IMAGE_COND_NOISE_SCALE", "0.0")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-midxattn-r16-shift-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-midxattn-r16-shift-infer",
)
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "distilled098_framemidxattn_action_lora_24fps_minterpolate_seed231_shifted_runs",
)

from pipelines.inference.generate_waymo24_action_minterpolate_lora import *  # noqa: F401,F403
