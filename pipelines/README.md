# Pipelines

Production-style Modal entrypoints for data preparation and training.

- `data/prepare_waymo24_visual_data.py`: stages Waymo front-camera clips from GCS into a Modal volume as 24 FPS, 121-frame MP4 windows. It can also cache VAE latents when requested.
- `data/export_waymo24_modal_volume_to_gcs.py`: exports the staged 24 FPS MP4 windows and manifests from the Modal data volume to GCS for sharing.
- `training/train_ltx2b_waymo_visual_lora.py`: trains the LTX-2B visual LoRA continuation baseline from cached latents.

Root-level wrapper files are kept for backward-compatible commands.
