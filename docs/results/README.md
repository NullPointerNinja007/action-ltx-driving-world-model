# Result Artifacts

This directory contains compact, versioned copies of the final metrics and plots. It does not contain generated videos or checkpoint files.

## Directory Map

- `final_action_alignment/`: full held-out validation subset used for the final action-alignment conclusion. This compares no-action baseline generation with the selected V4 action model under correct, zero, shuffled, and reversed-future actions.
- `final_v4_b200/`: B200 V4 rank-capacity campaign summaries and plots for the first capacity sweep.
- `b200_three_epoch/`: three-epoch continuation and rank sweep summaries for the final V4 candidates.
- `all_methods_checkpoint_curves/`: compact comparison plots over checkpoints across major method families.

## Main Metrics

- `S_rgb`: counterfactual action sensitivity, measured as generated-future RGB MAE between correct-action output and wrong-action outputs.
- `delta_psnr` / `delta_ssim`: correct-vs-wrong action advantage; positive values mean the correct-action generation is closer to the recorded future than wrong-action generations.
- `FVD-style`: feature-distribution distance used as a distributional video-quality diagnostic. Lower is better.
- `PSNR` and `SSIM`: frame-level similarity to the recorded future.
- `sharpness_ratio` and `motion_ratio`: diagnostics for visual detail retention and temporal motion.

## Important Caveat

`S_rgb` measures whether the model changes when the action sequence changes. It does not by itself prove semantic action following. The semantic evidence is the correct-vs-wrong advantage by action stratum, especially high-action, acceleration, and turning clips.

