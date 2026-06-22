# Final Submission Guide

This repository is the cleaned final package for the CS231N project **Action-Sensitive Driving Video Generation**. It contains the code, report, poster, compact metrics, and plotting scripts needed to understand the final experiments.

## What To Read First

1. Final report: `docs/final_report/cs231n_final_action_ltx_waymo_report.pdf`
2. Poster: `docs/poster/cs231n_poster_final.pdf`
3. Final action-alignment summary: `docs/results/final_action_alignment/final_action_alignment_report.md`
4. Three-epoch V4 analysis: `docs/results/b200_three_epoch/three_epoch_deep_analysis.md`

## Final Model Family

The selected model is the V4 full-action, low-frequency action-conditioning model. It uses:

- full `112D` per-frame ego actions,
- a temporal action bottleneck,
- low-frequency target and temporal-delta losses,
- high-frequency teacher preservation from the no-action visual model,
- gated residual action injection,
- LoRA adaptation with frozen base video transformer weights.

The most important scripts are:

- `pipelines/training/train_ltx2b_waymo_visual_lora.py`
- `pipelines/inference/generate_waymo24_action_minterpolate_lora.py`
- `scripts/run_b200_v4_rank_capacity_campaign.py`
- `scripts/run_b200_v4_three_epoch_continuation.py`
- `scripts/run_final_action_alignment_validation.py`
- `scripts/compute_final_action_alignment_metrics_modal.py`

## Final Evaluation

The final validation set contains `1,904` held-out windows. For each window, the selected action model was evaluated with:

- correct recorded actions,
- zero actions,
- shuffled actions from another validation clip,
- reversed future actions.

This supports two separate questions:

- **Action sensitivity:** does changing the action sequence change the generated future?
- **Action alignment:** is the generation from the correct action sequence closer to the recorded future than wrong-action generations?

The no-action baseline has action sensitivity `S_rgb = 0.000` by construction. The selected action model reaches `S_rgb = 2.262` while remaining close to the no-action baseline on FVD-style, PSNR, and SSIM. Correct-action advantage is small globally but positive on high-action clips, especially acceleration and turning.

## Compact Result Files

Final alignment:

- `docs/results/final_action_alignment/model_mode_summary.csv`
- `docs/results/final_action_alignment/per_clip_action_alignment.csv`
- `docs/results/final_action_alignment/correct_vs_wrong_advantage_summary.csv`
- `docs/results/final_action_alignment/stratum_summary.csv`
- `docs/results/final_action_alignment/fvd_summary.csv`

B200 rank/capacity and continuation:

- `docs/results/final_v4_b200/b200_rank_capacity_analysis.md`
- `docs/results/final_v4_b200/model_summary_with_fvd.csv`
- `docs/results/b200_three_epoch/three_epoch_deep_analysis.md`
- `docs/results/b200_three_epoch/model_summary_with_fvd.csv`

All-method checkpoint curves:

- `docs/results/all_methods_checkpoint_curves/all_methods_plot_rows.csv`
- `docs/results/all_methods_checkpoint_curves/v4_plot_rows.csv`
- `docs/results/all_methods_checkpoint_curves/metric_plots/`

## What Is Not In Git

Large or private artifacts are intentionally excluded:

- model checkpoints,
- raw/interpolated video windows,
- generated videos,
- cached latents,
- Modal logs,
- private storage paths,
- full benchmark dumps.

The repository is meant to be a public, reviewable final package rather than a full artifact mirror.
