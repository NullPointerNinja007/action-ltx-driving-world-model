"""2-epoch distilled LTX-2B Waymo no-action visual LoRA training wrapper."""

import os

os.environ.setdefault("LTX_MODAL_APP", "ltx2b-distilled098-waymo24-visual-lora-r16-2epoch-train")
os.environ.setdefault("LTX_CKPT_2B", "ltxv-2b-0.9.8-distilled.safetensors")
os.environ.setdefault(
    "LTX_CHECKPOINT_VOLUME_NAME",
    "ltx2b-dist098-waymo24-noaction-visual-lora-r16-2epoch-ckpts",
)
# The full 7992-window latent cache lives under `latents`; `latents_distilled098`
# is incomplete on the current Modal data volume.
os.environ.setdefault("WAYMO24_LATENT_PREFIX", "latents")

from pipelines.training.train_ltx2b_waymo_visual_lora import *  # noqa: F401,F403
