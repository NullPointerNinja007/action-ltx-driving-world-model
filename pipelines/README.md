# Pipelines

Production-style Modal entrypoints for data preparation and training.

- `data/prepare_waymo24_visual_data.py`: stages Waymo front-camera clips from a processed-data root into a Modal volume as 24 FPS, 121-frame MP4 windows. It can also cache VAE latents when requested.
- `data/prepare_waymo24_action_conditions.py`: builds compact 18D action tensors aligned to the same 121-frame windows.
- `data/prepare_waymo24_frame_action_conditions.py`: builds the final 112D per-frame action tensors used by the V4 models.
- `setup/download_ltx_model_to_modal_volume.py`: downloads LTX model assets from Hugging Face into the Modal `models` volume.
- `setup/verify_training_volumes.py`: verifies imported MP4 windows, manifests, model files, and latent-cache status.
- `training/train_ltx2b_waymo_visual_lora.py`: trains the LTX-2B visual LoRA continuation baseline from cached latents.
- `evaluation/benchmark_video_quality.py`: evaluates generated continuation videos against the real future frames with future-only PSNR/SSIM/MSE, sharpness, temporal-delta, and copy-leakage metrics.

Root-level wrapper files are kept for backward-compatible commands.
