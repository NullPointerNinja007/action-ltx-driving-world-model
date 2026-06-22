"""Inference wrapper for shifted-timestep no-action distilled Waymo visual LoRA ablation."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts",
)
os.environ.setdefault(
    "LTX_ARTIFACTS_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-infer",
)
os.environ.setdefault("LTX_BASE_CKPT", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_RUNS_ROOT",
    "distilled098_noaction_shifted_timestep_ablation_24fps_minterpolate_seed231_runs",
)

from pipelines.inference.generate_waymo24_minterpolate_lora import *  # noqa: F401,F403
