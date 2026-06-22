# Action-Conditioning Checkpoint Trajectory Report

## Scope

This report plots metrics over checkpoint step for the action-conditioning methods we have benchmarked so far. Unlike the earlier best-row comparison, these plots preserve the training trajectory so we can see when each method starts improving, blurring, or collapsing.

Metrics include FVD-style future distance, PSNR, SSIM, Laplacian sharpness ratio, motion ratio, temporal delta error, and the newer FFT/high-frequency/action-sensitivity diagnostics where available.

The dashed gray line in trajectory plots is the corrected no-action shifted/log-normal LoRA reference trajectory when that metric is available. It is a reference, not an action-conditioned method.

## Quality Gate Used For Best-Step Table

The best-step table chooses the lowest FVD checkpoint after requiring `sharpness_ratio >= 0.22`, `motion_ratio >= 0.7`, `PSNR >= 15.0`, and `SSIM >= 0.65`. If a line never passes this gate, it falls back to lowest FVD and marks `Quality = no`.

## Best Checkpoint Per Trajectory Line

| Group | Line | Step | Gate | FVD | PSNR | SSIM | Sharp | Motion | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frame_action_2epoch | Frame AdaLN, 2 epochs | 4000 | NA | 77.608 | 19.067 | 0.790 | 0.135 | 0.551 | no |
| frame_action_2epoch | Frame global MLP, 2 epochs | 100 | NA | 67.607 | 17.380 | 0.749 | 0.260 | 0.793 | yes |
| frame_action_2epoch | Frame temporal pool, 2 epochs | 100 | NA | 66.273 | 17.700 | 0.755 | 0.228 | 0.719 | yes |
| frame_action_2epoch | Frame transformer, 2 epochs | 100 | NA | 67.068 | 17.563 | 0.748 | 0.247 | 0.742 | yes |
| frame_action_3000 | Frame AdaLN, 3000 steps | 250 | NA | 68.076 | 18.838 | 0.781 | 0.155 | 0.577 | no |
| frame_action_3000 | Frame global MLP, 3000 steps | 100 | NA | 70.427 | 17.550 | 0.745 | 0.262 | 0.795 | yes |
| frame_action_3000 | Frame temporal pool, 3000 steps | 100 | NA | 76.037 | 17.872 | 0.754 | 0.234 | 0.709 | yes |
| frame_action_3000 | Frame transformer, 3000 steps | 100 | NA | 66.625 | 17.468 | 0.741 | 0.244 | 0.731 | yes |
| legacy_tokens | Global AdaLN action | 100 | NA | 73.871 | 17.677 | 0.737 | 0.128 | 0.558 | no |
| legacy_tokens | Global MLP action tokens | 500 | NA | 61.335 | 17.904 | 0.757 | 0.212 | 0.652 | no |
| legacy_tokens | Temporal per-point action tokens | 100 | NA | 68.524 | 16.713 | 0.722 | 0.279 | 0.678 | no |
| legacy_tokens | Transformer action tokens | 100 | NA | 64.820 | 17.808 | 0.758 | 0.223 | 0.695 | no |
| midblock_gate_sweep | gate=0.0 | 250 | 0.0 | 67.672 | 17.855 | 0.763 | 0.267 | 0.792 | yes |
| midblock_gate_sweep | gate=0.1 | 100 | 0.1 | 65.550 | 18.032 | 0.765 | 0.254 | 0.768 | yes |
| midblock_gate_sweep | gate=0.25 | 100 | 0.25 | 70.367 | 18.223 | 0.766 | 0.231 | 0.731 | yes |
| midblock_gate_sweep | gate=0.5 | 100 | 0.5 | 71.269 | 18.448 | 0.768 | 0.201 | 0.674 | no |
| midblock_gate_sweep | gate=1.0 | 100 | 1.0 | 71.220 | 18.824 | 0.777 | 0.148 | 0.532 | no |
| midblock_raw | Middle-block gated XAttn, raw | 500 | NA | 69.331 | 18.502 | 0.759 | 0.144 | 0.519 | no |
| temporal_bottleneck_v1_gate_sweep | gate=0.0 | 100 | 0.0 | 65.846 | 17.867 | 0.763 | 0.268 | 0.789 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=0.025 | 3000 | 0.025 | 65.672 | 17.932 | 0.764 | 0.262 | 0.781 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=0.05 | 3000 | 0.05 | 65.137 | 17.960 | 0.765 | 0.256 | 0.773 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=0.1 | 3000 | 0.1 | 64.172 | 18.083 | 0.769 | 0.246 | 0.757 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=0.25 | 250 | 0.25 | 64.734 | 18.213 | 0.772 | 0.233 | 0.734 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=0.5 | 100 | 0.5 | 66.459 | 17.970 | 0.767 | 0.246 | 0.760 | yes |
| temporal_bottleneck_v1_gate_sweep | gate=1.0 | 100 | 1.0 | 66.260 | 18.135 | 0.769 | 0.225 | 0.737 | yes |
| temporal_bottleneck_v1_raw | Temporal bottleneck HF teacher v1, raw | 100 | 1.0 | 71.303 | 18.111 | 0.763 | 0.225 | 0.738 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.0 | 100 | 0.0 | 65.856 | 17.863 | 0.763 | 0.268 | 0.788 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.025 | 50 | 0.025 | 64.711 | 17.856 | 0.762 | 0.268 | 0.788 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.05 | 50 | 0.05 | 66.205 | 17.854 | 0.763 | 0.269 | 0.789 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.1 | 500 | 0.1 | 66.236 | 17.921 | 0.765 | 0.261 | 0.780 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.25 | 500 | 0.25 | 63.780 | 18.010 | 0.768 | 0.247 | 0.760 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=0.5 | 500 | 0.5 | 63.988 | 18.156 | 0.765 | 0.228 | 0.714 | yes |
| temporal_bottleneck_v2_gate_sweep | gate=1.0 | 50 | 1.0 | 67.314 | 17.849 | 0.763 | 0.268 | 0.793 | yes |

## Main Checkpoint-Level Findings

Best quality-passing checkpoint by FVD is `gate=0.25` in `temporal_bottleneck_v2_gate_sweep` at step `500` with FVD `63.780`, sharpness `0.247`, and motion `0.760`.

Most action methods show the same pattern: useful-looking metrics happen early, often around step 100 to 500, while longer training tends to reduce sharpness and motion. This is visible in the frame-action 3000-step and 2-epoch plots.

AdaLN-style action injection is consistently the riskiest pathway. It can raise PSNR/SSIM, but the checkpoint trajectories show sharpness and motion dropping below the quality gate. That matches the visually blurry outputs.

Raw middle-block gated XAttn also degrades as training proceeds, but the Phase 1 gate sweep shows that smaller inference gate scales can recover some visual quality. This supports the idea that action-path strength, not just architecture, is causing blur.

Temporal bottleneck HF-teacher trajectories are the most promising among the newer methods. They retain more sharpness, FFT high-frequency energy, and motion at low/intermediate gate scales while getting better FVD than the no-action step-3000 reference.

The action-sensitivity plots are available only for the diagnostic gate-sweep methods. They show temporal bottleneck v2 has the strongest correct-vs-counterfactual response while still preserving visual quality at low gate values.

The main negative result is that simply training longer is not enough. If the action path is too global or too strong, later checkpoints become smoother/static even when PSNR or SSIM look acceptable.

## Lines That Never Passed The Quality Gate

These trajectories need caution: their best FVD checkpoint still fails at least one of sharpness, motion, PSNR, or SSIM.

- `Frame AdaLN, 2 epochs` in `frame_action_2epoch`: best fallback step `4000`, FVD `77.608`, sharpness `0.135`, motion `0.551`.
- `Frame AdaLN, 3000 steps` in `frame_action_3000`: best fallback step `250`, FVD `68.076`, sharpness `0.155`, motion `0.577`.
- `Global AdaLN action` in `legacy_tokens`: best fallback step `100`, FVD `73.871`, sharpness `0.128`, motion `0.558`.
- `Global MLP action tokens` in `legacy_tokens`: best fallback step `500`, FVD `61.335`, sharpness `0.212`, motion `0.652`.
- `Temporal per-point action tokens` in `legacy_tokens`: best fallback step `100`, FVD `68.524`, sharpness `0.279`, motion `0.678`.
- `Transformer action tokens` in `legacy_tokens`: best fallback step `100`, FVD `64.820`, sharpness `0.223`, motion `0.695`.
- `gate=0.5` in `midblock_gate_sweep`: best fallback step `100`, FVD `71.269`, sharpness `0.201`, motion `0.674`.
- `gate=1.0` in `midblock_gate_sweep`: best fallback step `100`, FVD `71.220`, sharpness `0.148`, motion `0.532`.
- `Middle-block gated XAttn, raw` in `midblock_raw`: best fallback step `500`, FVD `69.331`, sharpness `0.144`, motion `0.519`.

## Plot Index

- All families FVD: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_fvd_future_over_checkpoints.png`
- All families FVD zoomed: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_fvd_future_zoomed_under125_over_checkpoints.png`
- All families PSNR: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_future_psnr_over_checkpoints.png`
- All families SSIM: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_future_ssim_over_checkpoints.png`
- All families sharpness: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_sharpness_ratio_over_checkpoints.png`
- All families motion: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_motion_ratio_over_checkpoints.png`
- All families FFT-HF: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/all_families_fft_high_frequency_ratio_over_checkpoints.png`

Each family also has a primary-metric overview plot and, when available, a secondary-metric plot:
- `Legacy action tokens / AdaLN`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/legacy_tokens_primary_metrics_over_checkpoints.png`
- `Legacy action tokens / AdaLN` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/legacy_tokens_secondary_metrics_over_checkpoints.png`
- `Frame-action methods, 3000-step runs`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/frame_action_3000_primary_metrics_over_checkpoints.png`
- `Frame-action methods, 3000-step runs` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/frame_action_3000_secondary_metrics_over_checkpoints.png`
- `Frame-action methods, 2-epoch runs`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/frame_action_2epoch_primary_metrics_over_checkpoints.png`
- `Frame-action methods, 2-epoch runs` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/frame_action_2epoch_secondary_metrics_over_checkpoints.png`
- `Middle-block gated XAttn, learned gate`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/midblock_raw_primary_metrics_over_checkpoints.png`
- `Middle-block gated XAttn, learned gate` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/midblock_raw_secondary_metrics_over_checkpoints.png`
- `Middle-block gated XAttn, inference gate sweep`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/midblock_gate_sweep_primary_metrics_over_checkpoints.png`
- `Middle-block gated XAttn, inference gate sweep` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/midblock_gate_sweep_secondary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v1, learned gate`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v1_raw_primary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v1, learned gate` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v1_raw_secondary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v1, inference gate sweep`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v1_gate_sweep_primary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v1, inference gate sweep` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v1_gate_sweep_secondary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v2, inference gate sweep`: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v2_gate_sweep_primary_metrics_over_checkpoints.png`
- `Temporal bottleneck HF-teacher v2, inference gate sweep` secondary: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/metric_plots/temporal_bottleneck_v2_gate_sweep_secondary_metrics_over_checkpoints.png`

## Data Artifacts

- Full trajectory rows: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/checkpoint_trajectory_rows.csv`
- Best checkpoint per line: `data/benchmarks/action_conditioning_checkpoint_trajectories_seed231_all5/best_checkpoint_per_line.csv`
