# Action-Sensitive Driving Video Generation

CS231N final project on controllable driving video generation with a pretrained video diffusion transformer.

Given `49` front-camera context frames at `24 FPS`, the model generates the next `72` frames. The key question is not only whether the future looks plausible, but whether it changes in the right way when the ego-vehicle action sequence changes.

## Core Idea

Visual-only video continuation can produce realistic-looking driving futures, but it is action-blind: changing future ego actions cannot affect the output. We fine-tune a pretrained video diffusion transformer so that it remains visually close to the no-action baseline while becoming sensitive to future ego actions.

The final method separates action control from visual rendering:

- `112D` per-frame ego-action vector aligned to every video frame.
- Temporal action encoder that maps frame actions into latent-time control features.
- Low-frequency target and delta losses that supervise coarse future motion.
- High-frequency teacher preservation from a frozen no-action visual model.
- LoRA adaptation and gated action residuals, with the base transformer, VAE, and text encoder kept frozen.

This matters because naive action conditioning tended to either get ignored or corrupt the pretrained video prior, causing blur, wobble, or motion collapse.

## Main Result

The final validation uses `1,904` held-out Waymo front-camera windows.

The no-action visual baseline has zero counterfactual action sensitivity by construction. The selected V4 action model becomes measurably action-sensitive while staying close to the no-action visual baseline:

| Model / mode | Action sensitivity `S_rgb` | FVD-style ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| No-action visual baseline | `0.000` | `11.702` | `17.153` | `0.7079` |
| V4 correct actions | `2.262` | `11.747` | `17.180` | `0.7082` |

On high-action clips, correct recorded actions are slightly but consistently closer to the real future than zero, shuffled, or reversed-future actions. The effect is strongest on acceleration and turning strata. The result is best interpreted as evidence of action alignment, not as a claim that PSNR/SSIM are perfect controllability metrics.

## Repository Layout

```text
pipelines/
  data/          Data preparation and frame-action tensor construction.
  training/      LoRA/action-conditioning training code.
  inference/     Video generation wrappers for no-action and action models.
  evaluation/    Future-frame quality metrics.

scripts/
  run_*          Modal experiment launchers and checkpoint sweeps.
  compute_*      Video quality, FVD-style, and counterfactual metrics.
  plot_*         Result plotting utilities.
  build_final_*  Final qualitative side-by-side/showcase builders.
  wrappers/      Legacy Modal preset wrappers used by older campaign scripts.

docs/
  final_report/  Final report PDF, LaTeX source, and Overleaf zip.
  poster/        Final poster PDF.
  poster_assets/ Poster-ready figures.
  results/       Compact final metrics, summaries, and plots.
```

## Important Entry Points

Training and inference:

- `pipelines/training/train_ltx2b_waymo_visual_lora.py`
- `pipelines/inference/generate_waymo24_action_minterpolate_lora.py`
- `scripts/run_b200_v4_rank_capacity_campaign.py`
- `scripts/run_b200_v4_three_epoch_continuation.py`

Evaluation:

- `pipelines/evaluation/benchmark_video_quality.py`
- `scripts/compute_final_action_alignment_metrics_modal.py`
- `scripts/run_final_action_alignment_validation.py`
- `scripts/plot_v4_and_all_method_checkpoint_curves.py`

Final presentation artifacts:

- `docs/final_report/cs231n_final_action_ltx_waymo_report.pdf`
- `docs/final_report/action_ltx_final_overleaf_project.zip`
- `docs/poster/cs231n_poster_final.pdf`
- `docs/results/final_action_alignment/final_action_alignment_report.md`

## Dataset Setup

The experiments use the Waymo Open Dataset front camera, processed into 24 FPS video-continuation windows:

```text
121 total frames
49 context frames
72 future frames
7,992 training windows per epoch
1,904 held-out final validation windows
```

Each frame has an aligned `112D` ego-action vector containing future path samples, recent velocity/acceleration history, and current ego/camera motion cues.

The raw/interpolated videos, cached latents, checkpoints, generated videos, and Modal volume exports are intentionally not tracked in Git. To run the full pipelines, restore equivalent external data/model volumes and set:

```text
WAYMO24_PROCESSED_V2_ROOT=<processed Waymo source root>
```

## What Is Tracked

This repository contains:

- training, inference, data-prep, and evaluation code,
- compact result tables and plots,
- final report and poster assets,
- scripts used for checkpoint sweeps, rank/capacity campaigns, and final action-alignment validation.

This repository intentionally excludes:

- trained checkpoints,
- raw/interpolated videos,
- generated MP4 outputs,
- cached latent tensors,
- private storage locations,
- full benchmark dumps too large for Git.

## Quick Checks

Run a syntax check over the public Python code:

```bash
find . -name '*.py' -print0 | xargs -0 python3 -m py_compile
```

Inspect compact final metrics:

```bash
python3 - <<'PY'
import pandas as pd
print(pd.read_csv("docs/results/final_action_alignment/model_mode_summary.csv"))
print(pd.read_csv("docs/results/final_action_alignment/correct_vs_wrong_advantage_summary.csv"))
PY
```

## Project Status

The final story is that action conditioning in pretrained driving video diffusion models is possible, but only when the action pathway is constrained to affect low-frequency temporal evolution while preserving the high-frequency visual prior. The final V4 model does not dominate the no-action baseline on every visual metric; instead, it adds measurable action sensitivity and weak positive correct-action advantage while keeping visual quality near the baseline.
