# Action-Conditioning Basic Comparison Report

## Scope

This report aggregates the existing action-conditioning benchmark outputs for the 5 local Waymo validation clips. The comparison focuses on the basic metrics we have consistently used: FVD-style future distance, future PSNR, future SSIM, Laplacian sharpness ratio, motion ratio, temporal delta error, copy leakage, and FFT high-frequency retention when available.

The FVD number here is a project-level FVD-style future distance over 5 clips, not a publication-grade full-dataset FVD. It is still useful for relative comparisons because every row uses the same validation clips, frame count, FPS, and benchmark implementation.

## Selection Rule

For every method variation, the script first keeps rows that pass `sharpness_ratio >= 0.22`, `motion_ratio >= 0.7`, `PSNR >= 15.0`, and `SSIM >= 0.65`. It then selects the lowest FVD-style future distance in that usable pool. If a method has no usable row, it falls back to lowest FVD and marks that method as a failed quality-gate selection.

The no-action shifted LoRA `step_003000` is included as a gray reference because that was the corrected visual baseline used for later action experiments. It is not treated as an action-conditioning method.

Action-method selections exclude `step_000000` rows because those are base-reference states before the action pathway has trained.

## Selected Rows

| Method | Step | Gate | FVD | PSNR | SSIM | Sharp | FFT-HF | Motion | Action sens. | Selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No-action shifted LoRA step3000 | 3000 | NA | 71.424 | 17.892 | 0.763 | 0.271 | NA | 0.790 | NA | fixed_reference_step3000 |
| Frame AdaLN, 2 epochs | 4000 | NA | 77.608 | 19.067 | 0.790 | 0.135 | NA | 0.551 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Frame AdaLN, 3000 steps | 250 | NA | 68.076 | 18.838 | 0.781 | 0.155 | NA | 0.577 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Global AdaLN action | 100 | NA | 73.871 | 17.677 | 0.737 | 0.128 | NA | 0.558 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Frame global MLP, 2 epochs | 100 | NA | 67.607 | 17.380 | 0.749 | 0.260 | NA | 0.793 | NA | min_fvd_positive_step_among_quality_gate |
| Frame global MLP, 3000 steps | 100 | NA | 70.427 | 17.550 | 0.745 | 0.262 | NA | 0.795 | NA | min_fvd_positive_step_among_quality_gate |
| Frame temporal pool, 2 epochs | 100 | NA | 66.273 | 17.700 | 0.755 | 0.228 | NA | 0.719 | NA | min_fvd_positive_step_among_quality_gate |
| Frame temporal pool, 3000 steps | 100 | NA | 76.037 | 17.872 | 0.754 | 0.234 | NA | 0.709 | NA | min_fvd_positive_step_among_quality_gate |
| Frame transformer, 2 epochs | 100 | NA | 67.068 | 17.563 | 0.748 | 0.247 | NA | 0.742 | NA | min_fvd_positive_step_among_quality_gate |
| Frame transformer, 3000 steps | 100 | NA | 66.625 | 17.468 | 0.741 | 0.244 | NA | 0.731 | NA | min_fvd_positive_step_among_quality_gate |
| Middle-block gated XAttn, gate sweep | 100 | 0.1 | 65.550 | 18.032 | 0.765 | 0.254 | 0.799 | 0.768 | NA | min_fvd_positive_step_among_quality_gate |
| Middle-block gated XAttn, raw | 500 | NA | 69.331 | 18.502 | 0.759 | 0.144 | 0.546 | 0.519 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Global MLP action tokens | 500 | NA | 61.335 | 17.904 | 0.757 | 0.212 | NA | 0.652 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Temporal bottleneck HF teacher v1, gate sweep | 3000 | 0.1 | 64.172 | 18.083 | 0.769 | 0.246 | 0.778 | 0.757 | NA | min_fvd_positive_step_among_quality_gate |
| Temporal bottleneck HF teacher v1, raw | 100 | 1.0 | 63.472 | 18.079 | 0.770 | 0.224 | 0.747 | 0.737 | NA | min_fvd_positive_step_among_quality_gate |
| Temporal bottleneck HF teacher v2, gate sweep | 500 | 0.25 | 63.780 | 18.010 | 0.768 | 0.247 | 0.777 | 0.760 | NA | min_fvd_positive_step_among_quality_gate |
| Temporal per-point action tokens | 100 | NA | 68.524 | 16.713 | 0.722 | 0.279 | NA | 0.678 | NA | min_fvd_positive_step_fallback_failed_quality_gate |
| Transformer action tokens | 100 | NA | 64.820 | 17.808 | 0.758 | 0.223 | NA | 0.695 | NA | min_fvd_positive_step_fallback_failed_quality_gate |

## Counterfactual Action Sensitivity Rows

These are selected separately from the quality/FVD rows. They choose the strongest available correct-vs-counterfactual response among rows that still pass the same quality gate when possible.

| Method | Step | Gate | Action sens. | Temporal sens. | FVD | Sharp | Motion | Selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Temporal bottleneck HF teacher v2, gate sweep | 50 | 0.025 | 2.729 | 2.796 | 64.711 | 0.268 | 0.788 | max_action_sensitivity_positive_step_among_quality_gate |
| Temporal bottleneck HF teacher v1, gate sweep | 100 | 1.0 | 2.531 | 2.387 | 66.260 | 0.225 | 0.737 | max_action_sensitivity_positive_step_among_quality_gate |
| Middle-block gated XAttn, gate sweep | 100 | 1.0 | 1.455 | 1.586 | 71.220 | 0.148 | 0.532 | max_action_sensitivity_positive_step_fallback_failed_quality_gate |

## Main Findings

The best quality-passing action-conditioned FVD-style row is **Temporal bottleneck HF teacher v1, raw** at step `100` with gate `1.0`: FVD `63.472`, sharpness `0.224`, motion `0.737`.

The lowest FVD-style row overall is **Global MLP action tokens** at FVD `61.335`, but it fails the quality gate with sharpness `0.212` and motion `0.652`. This is why the report does not treat lowest FVD alone as the winner.

Among quality-passing action rows, the strongest high-frequency retention is **Frame global MLP, 3000 steps** with sharpness ratio `0.262` and FVD `70.427`.

The strongest measured counterfactual action sensitivity is **Temporal bottleneck HF teacher v2, gate sweep** with mean future RGB MAE `2.729`. This metric is only available for the diagnostic gate-sweep methods, so older token methods should not be interpreted as action-insensitive solely because the field is missing.

Across the runs, the recurring failure mode is clear: methods that let action features globally affect the visual pathway often improve or maintain PSNR/SSIM while suppressing sharpness and motion. PSNR/SSIM alone are therefore not enough; blur can look numerically closer to the average future while becoming visually unusable.

The middle-block and temporal-bottleneck experiments support the current hypothesis that action should be routed through a constrained temporal path. Full-strength learned gates hurt high-frequency detail, but low-gate inference settings preserve much more of the corrected no-action visual quality.

The temporal bottleneck HF-teacher v2 gate sweep is currently the most promising direction because it preserves no-action-like sharpness/FFT/motion at low gate values while showing stronger counterfactual action sensitivity than the Phase 1 middle-block diagnostics.

However, the best v2 rows are still early/low-gate operating points. That means the model may be only weakly using actions. The next architectural step should add an explicit low-frequency action-following objective instead of relying only on diffusion MSE plus gates.

## Quality-Gate Failures

The following selected rows did not have any checkpoint/configuration passing the full quality gate, so their best row is a fallback by FVD only:

- Frame AdaLN, 2 epochs: selected FVD `77.608`, sharpness `0.135`, motion `0.551`.
- Frame AdaLN, 3000 steps: selected FVD `68.076`, sharpness `0.155`, motion `0.577`.
- Global AdaLN action: selected FVD `73.871`, sharpness `0.128`, motion `0.558`.
- Middle-block gated XAttn, raw: selected FVD `69.331`, sharpness `0.144`, motion `0.519`.
- Global MLP action tokens: selected FVD `61.335`, sharpness `0.212`, motion `0.652`.
- Temporal per-point action tokens: selected FVD `68.524`, sharpness `0.279`, motion `0.678`.
- Transformer action tokens: selected FVD `64.820`, sharpness `0.223`, motion `0.695`.

## Metric Interpretation

FVD-style future distance: lower is better, but this is computed over only 5 clips and should be used as a relative diagnostic.

PSNR and SSIM: higher means generated future is closer to the ground-truth future under pixel/structural similarity. These metrics can reward blurry averages, so they must be read with sharpness and motion.

Laplacian sharpness ratio: generated future sharpness divided by reference future sharpness. Higher is better; collapse here matches the visually blurry outputs we saw.

Motion ratio: generated temporal-change magnitude divided by reference. Higher is better here because most runs are below 1. Low values indicate static/copy-like futures.

FFT high-frequency ratio: generated high-frequency energy divided by reference. Higher is better; this is available only for newer benchmark runs.

Action sensitivity: mean future RGB MAE between correct-action generation and counterfactual-action generation. Higher means actions affect the output more. This is not itself quality; high sensitivity plus blur means the action path is active but harmful.

## Plots

- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/basic_metric_overview_grid.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_future_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_future_bar_zoomed_under120.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/future_psnr_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/future_ssim_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/sharpness_ratio_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/motion_ratio_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_vs_sharpness.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_vs_sharpness_zoomed_under120.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_vs_motion.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fvd_vs_motion_zoomed_under120.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/fft_high_frequency_ratio_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/action_sensitivity_rgb_mae_bar.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/action_sensitivity_vs_sharpness.png`
- `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/metric_plots/normalized_metric_heatmap.png`

## Data Artifacts

- Full candidates: `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/all_action_method_candidate_rows.csv`
- Selected rows: `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/best_action_method_rows.csv`
- Method coverage: `data/benchmarks/action_conditioning_basic_method_comparison_seed231_all5/method_coverage.csv`

