# Pipelines

Production-style Modal entrypoints for data preparation and training.

- `data/prepare_waymo24_visual_data.py`: stages Waymo front-camera clips from GCS into a Modal volume as 24 FPS, 121-frame MP4 windows. It can also cache VAE latents when requested.
- `data/export_waymo24_modal_volume_to_gcs.py`: exports the staged 24 FPS MP4 windows and manifests from the Modal data volume to GCS for sharing.
- `data/import_waymo24_gcs_to_modal_volume.py`: imports the shared 24 FPS MP4 windows and manifests from GCS into a fresh Modal data volume.
- `setup/download_ltx_model_to_modal_volume.py`: downloads LTX model assets from Hugging Face into the Modal `models` volume.
- `setup/verify_training_volumes.py`: verifies imported MP4 windows, manifests, model files, and latent-cache status.
- `training/train_ltx2b_waymo_visual_lora.py`: trains the LTX-2B visual LoRA continuation baseline from cached latents.

Root-level wrapper files are kept for backward-compatible commands.
