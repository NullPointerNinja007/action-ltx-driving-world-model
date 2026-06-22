# CS231N Final Submission: Action-Sensitive Driving Video Generation

This branch is the final packaged version of the project for CS231N. It contains the code, report, poster, compact result artifacts, and analysis scripts needed to understand and reproduce the final experiments.

## Project Summary

We study front-camera driving video continuation with a pretrained video diffusion transformer. Given 49 observed frames at 24 FPS, the model generates the next 72 frames. The main question is whether future ego actions can make the generated future controllable without destroying visual quality.

The final method uses per-frame 112D ego-action vectors, a temporal action bottleneck, low-frequency future-token losses, high-frequency teacher preservation, and LoRA adaptation. The final validation campaign compares the selected action-conditioned model against a no-action visual baseline and against counterfactual wrong-action generations.

## What Is Included

- `pipelines/`: data preparation, training, inference, and evaluation modules.
- `scripts/`: Modal launchers, checkpoint sweeps, metric computation, plotting, side-by-side builders, and final validation scripts.
- `docs/final_report/`: final report PDF, LaTeX source, and Overleaf zip.
- `docs/poster/`: final poster PDF.
- `docs/poster_assets/`: poster-ready diagrams and plots.
- `docs/results/`: compact final metrics, plots, and reports copied from the ignored local benchmark tree.
- `docs/figures/`: figures used by the final report.

## What Is Not Included

Large generated artifacts are intentionally excluded from Git:

- trained checkpoint volumes,
- raw/interpolated video windows,
- generated inference videos,
- cached latents,
- full Modal volume exports,
- large per-step loss dumps.

Those assets are external to the repository. The code and compact result files in this branch are enough for review and for recreating the final analyses if equivalent external volumes are restored.

## Key External Data Assumptions

The final dataset uses Waymo Open Dataset front-camera clips processed into:

- 24 FPS interpolated windows,
- 121 frames per window,
- 49 context frames,
- 72 future frames,
- 7,992 training windows per epoch,
- 1,904 held-out final validation windows.

The active data volume name used throughout the Modal scripts is:

```text
waymo-e2e-24fps-121f-visual-continuation-data
```

The final V4 checkpoint volume used for the best action-conditioned models is:

```text
ltx2b-v4-b200-rank-capacity-ckpts
```

## Final Result Artifacts

The main final result files are:

- `docs/results/final_action_alignment/model_mode_summary.csv`
- `docs/results/final_action_alignment/correct_vs_wrong_advantage_summary.csv`
- `docs/results/final_action_alignment/stratum_summary.csv`
- `docs/results/final_action_alignment/final_action_alignment_report.md`
- `docs/results/b200_three_epoch/three_epoch_deep_analysis.md`
- `docs/results/b200_three_epoch/model_summary_with_fvd.csv`
- `docs/results/final_v4_b200/b200_rank_capacity_analysis.md`

The final report and poster are:

- `docs/final_report/cs231n_final_action_ltx_waymo_report.pdf`
- `docs/poster/cs231n_poster_final.pdf`
