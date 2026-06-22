# Final Action-Alignment Validation Report

Generated at `2026-06-05T22:50:52.038089+00:00`.

This campaign evaluates whether correct actions beat wrong actions on the full validation set, not just whether the model changes under action perturbation.

## Correct-vs-Wrong Advantage

| Stratum | Model | clips | S_rgb | delta PSNR | delta SSIM | delta temporal error | delta LF temporal error |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | v4_r64_selected_step018000 | 1904 | 2.262 | -0.0042 | 0.00015 | -0.0004 | -0.0002 |
| high_action | v4_r64_selected_step018000 | 1055 | 2.303 | 0.0019 | 0.00027 | -0.0005 | -0.0002 |

Positive delta PSNR/SSIM means correct actions are closer to ground truth than wrong actions. Positive delta temporal error means wrong actions have larger temporal error than correct actions.

## Correct-Mode Visual Quality

| Model | clips | PSNR | SSIM | sharpness | motion | FVD-style |
|---|---:|---:|---:|---:|---:|---:|
| noaction_shifted_step003000 | 1904 | 17.153 | 0.708 | 0.243 | 1.147 | 11.702 |
| v4_r64_selected_step018000 | 1904 | 17.180 | 0.708 | 0.233 | 1.136 | 11.747 |

## Interpretation Rules

- If high-action delta metrics are positive for V4, we have semantic action-alignment evidence.
- If S_rgb is high but delta metrics are not positive, the model is action-sensitive but not proven semantically correct.
- If no-action has stronger PSNR/SSIM but V4 has positive high-action advantage, average-case fidelity metrics understate controllability.
