# Action-LTX Driving World Model

Utilities and Modal pipelines for Waymo front-camera video continuation experiments with LTX-Video.

## Layout

- `pipelines/data/`: dataset staging pipelines. The main entrypoint is `prepare_waymo24_visual_data.py`, which builds 24 FPS, 121-frame Waymo MP4 windows on Modal.
- `pipelines/training/`: model fine-tuning pipelines. The main entrypoint is `train_ltx2b_waymo_visual_lora.py`.
- `experiments/inference/`: one-off and baseline Modal inference experiments.
- `experiments/comparisons/`: side-by-side comparison experiments, including RIFE vs FFmpeg minterpolate.
- `scripts/`: metadata extraction, cleaning, auditing, and manifest-building utilities.
- `docs/`: project slides and written milestone material.
- `data/`: local input clips and local comparison outputs. Large/generated data is not the source of truth.

Root-level Python files are compatibility wrappers so older `modal run <script>.py` commands still work.

## Current Dataset Pipeline

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

Manifests are stored in the Modal volume under:

```text
/manifests/train_windows_24fps_121f.csv
/manifests/val_windows_24fps_121f.csv
```
