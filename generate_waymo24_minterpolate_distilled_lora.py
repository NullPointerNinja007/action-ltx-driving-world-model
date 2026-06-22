"""Compatibility wrapper for distilled LTX-2B Waymo LoRA local minterpolate inference."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-distilled098-waymo24-lora-local-minterpolate-infer")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-distilled098-waymo24fps-visual-lora-r16-checkpoints",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-distilled098-waymo24-lora-local-minterpolate-inference",
)
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_LORA_RUN_NAME",
    "ltx2b_distilled098_waymo24_visual_lora_r16_seed231_subset512_lr2e5_steps500",
)
os.environ.setdefault("LTX_DEFAULT_LORA_STEP", "step_000500")
os.environ.setdefault(
    "LTX_LOCAL_OUTPUT_DIR",
    "data/finetuned_ltx2b_distilled098_lora_24fps_minterpolate_seed231",
)
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "finetuned_ltx2b_distilled098_lora_24fps_minterpolate_seed231_runs",
)

from pipelines.inference.generate_waymo24_minterpolate_lora import *  # noqa: F401,F403
