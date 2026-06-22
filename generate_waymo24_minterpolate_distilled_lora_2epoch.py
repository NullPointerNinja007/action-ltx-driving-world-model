"""Inference wrapper for the 2-epoch distilled no-action Waymo visual LoRA run."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-distilled098-waymo24-noaction-2epoch-lora-infer")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-infer",
)
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "distilled098_noaction_visual_lora_2epoch_24fps_minterpolate_seed231_runs",
)

from pipelines.inference.generate_waymo24_minterpolate_lora import *  # noqa: F401,F403
