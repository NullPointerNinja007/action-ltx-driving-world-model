# Action-Sensitive Driving Video Generation

Final CS231N project repository for action-conditioned front-camera driving video continuation with a pretrained video diffusion transformer.

For the packaged submission overview, see [`README_FINAL.md`](README_FINAL.md). It points to the final report, poster, compact result artifacts, and the scripts used for training and evaluation.

## Layout

- `pipelines/data/`: dataset staging pipelines. The main entrypoint is `prepare_waymo24_visual_data.py`, which builds 24 FPS, 121-frame Waymo MP4 windows on Modal from an environment-provided processed-data root.
- `pipelines/training/`: model fine-tuning pipelines. The main entrypoint is `train_ltx2b_waymo_visual_lora.py`.
- `experiments/inference/`: one-off and baseline Modal inference experiments.
- `experiments/comparisons/`: side-by-side comparison experiments, including RIFE vs FFmpeg minterpolate.
- `scripts/`: metadata extraction, cleaning, auditing, and manifest-building utilities.
- `docs/`: final report, poster, compact result tables, and figures.

Root-level Python files are compatibility wrappers so older `modal run <script>.py` commands still work.

## Final Dataset Setup

The active Waymo visual-continuation dataset is staged on Modal volume:

```text
waymo-e2e-24fps-121f-visual-continuation-data
```

The staged MP4 windows use:

```text
24 FPS
121 total frames
49 context frames
72 future frames
4 adaptive windows per contiguous scenario
```

The final experiments use 7,992 training windows per epoch and 1,904 held-out validation windows. Large videos, latents, checkpoints, and generated outputs are intentionally not tracked in Git.

The Modal data-prep scripts expect the processed Waymo source root through:

```text
WAYMO24_PROCESSED_V2_ROOT
```
