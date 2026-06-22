# B200 V4 Three-Epoch Continuation: Deep Numerical Analysis

This analysis summarizes the final B200 V4 continuation evaluation. All numbers below use the existing 5 local validation clips, 24 FPS, 49 context frames, 72 future frames, seed 231, corrected inference with `image_cond_noise_scale=0.0`, and the same FVD-style backend used in the recent V4 runs.

Important limitation: this is still a 5-clip diagnostic set. The curves are useful for model selection and failure analysis, but full claims should be validated on the larger validation set.

## Metrics

- `S`: counterfactual action sensitivity. This is the average future RGB MAE between the correct-action generation and the zero, shuffled, and reversed-future action generations. Higher means the output depends more on the action tensor. `S` does not prove semantic correctness.
- FVD-style: lower is better, but this is the lightweight R3D18-style diagnostic used throughout the project, not official large-sample FVD.
- PSNR/SSIM: future-frame pixel similarity to ground truth. Higher is better, but these are imperfect for stochastic future generation.
- Sharpness ratio: generated future Laplacian sharpness over reference future sharpness. This should not be treated naively as "higher is always better" because the pretrained generator can be sharper than the Waymo training videos. The relevant failure is not lower sharpness alone, but blur/wobble/collapse.
- Motion ratio: generated temporal motion over reference temporal motion. Values near the reference are preferred; severe collapse or excessive wobble are failures.

## Corrected No-Action Reference

The corrected no-action shifted/log-normal LoRA is the visual baseline. It cannot use actions, so its action sensitivity is exactly zero by construction.

| Model | Step | S | FVD | PSNR | SSIM | Sharpness | Motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-action shifted LoRA | 3000 | 0.000 | 71.424 | 17.892 | 0.763 | 0.271 | 0.790 |

This model is visually strong, but it is not action-conditioned.

## New B200 Continuation Results

### Rank 32, continued to 3 epochs

| Step | S | FVD | PSNR | SSIM | Sharpness | Motion | FFT-HF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7992 | 2.437 | 86.505 | 17.261 | 0.701 | 0.358 | 0.712 | 0.983 |
| 15984 | 2.637 | 87.825 | 17.255 | 0.700 | 0.361 | 0.719 | 0.975 |
| 22000 | 2.569 | 86.264 | 17.270 | 0.701 | 0.360 | 0.713 | 0.973 |
| 23976 | 1.904 | 89.231 | 17.257 | 0.700 | 0.360 | 0.715 | 0.977 |

Best points:

- Best FVD: step 18000, FVD 83.958.
- Best PSNR/SSIM among measured checkpoints: step 22000, PSNR 17.270, SSIM 0.701.
- Best measured action sensitivity: step 15984, `S=2.637`.

Conclusion: rank 32 does not benefit from training to the final 3-epoch checkpoint. The final checkpoint loses action sensitivity and has worse FVD than its intermediate checkpoints. If we use r32 at all, the best candidate is around step 15984-22000, not final.

### Rank 64, continued to 3 epochs

| Step | S | FVD | PSNR | SSIM | Sharpness | Motion | FFT-HF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7992 | 2.735 | 83.906 | 17.262 | 0.701 | 0.357 | 0.709 | 0.987 |
| 15984 | 2.763 | 86.397 | 17.264 | 0.701 | 0.357 | 0.714 | 0.971 |
| 22000 | 2.458 | 85.672 | 17.279 | 0.702 | 0.353 | 0.711 | 0.979 |
| 23976 | 3.157 | 86.805 | 17.269 | 0.700 | 0.359 | 0.711 | 0.973 |

Best points:

- Best FVD: step 10000, FVD 82.769, PSNR 17.278, SSIM 0.703, sharpness 0.360, motion 0.710.
- Best PSNR/SSIM: step 18000, PSNR 17.298, SSIM 0.704.
- Best measured action sensitivity: step 23976, `S=3.157`.

Conclusion: rank 64 is the best new continuation model. It preserves a stable visual-quality band while reaching strong action sensitivity at the final checkpoint. There is a tradeoff: the best FVD point is earlier than the best measured action-sensitivity point.

### Rank 128, partial run

| Step | S | FVD | PSNR | SSIM | Sharpness | Motion | FFT-HF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7992 | 3.281 | 88.569 | 17.282 | 0.701 | 0.356 | 0.712 | 0.973 |
| 12000 | 2.944 | 83.707 | 17.278 | 0.701 | 0.354 | 0.711 | 0.979 |
| 15000 | 2.591 | 85.809 | 17.276 | 0.701 | 0.352 | 0.707 | 0.983 |

Best points:

- Best FVD: step 6000, FVD 83.487.
- Best PSNR: step 7000, PSNR 17.291.
- Best SSIM: step 10000, SSIM 0.703.
- Best measured action sensitivity: step 7992, `S=3.281`.

Conclusion: rank 128 gives strong early action sensitivity, but the sensitivity decreases with more training. It is not clearly better than rank 64, and it is not worth blindly continuing without a different regularization or checkpoint-selection strategy.

## Comparison To Previous B200 Rank-Capacity Run

| Model | Step | S | FVD | PSNR | SSIM | Sharpness | Motion |
|---|---:|---:|---:|---:|---:|---:|---:|
| r16 continued | 15984 | 2.336 | 90.066 | 17.207 | 0.698 | 0.370 | 0.725 |
| r16 continued | 22000 | 2.827 | 89.422 | 17.210 | 0.699 | 0.372 | 0.724 |
| r16 continued | 23976 | 3.450 | 88.465 | 17.214 | 0.699 | 0.369 | 0.723 |
| r32 1 epoch | 7992 | 2.621 | 87.427 | 17.257 | 0.700 | 0.361 | 0.717 |
| r64 1 epoch | 7992 | 3.057 | 85.602 | 17.256 | 0.700 | 0.360 | 0.714 |
| r64 3 epochs | 23976 | 3.157 | 86.805 | 17.269 | 0.700 | 0.359 | 0.711 |

The strongest action sensitivity remains r16 continued at step 23976 (`S=3.450`), but rank 64 gives a better FVD/quality compromise than r16. Rank 64 also improves over r32. Rank 128 does not give a clean additional gain.

## Loss And Optimization Evidence

The loss curve file contains 62,952 unique method/step rows after deduplicating repeated cumulative histories:

- r32: 23,976 unique steps.
- r64: 23,976 unique steps.
- r128: 15,000 unique steps.

Representative LoRA gradient norms are nonzero throughout:

| Model | Step | Loss | Diffusion loss | Action aux loss | LoRA grad norm | Mean abs gate |
|---|---:|---:|---:|---:|---:|---:|
| r32 | 1 | 0.276 | 0.482 | 2.811 | 0.043 | 0.0 |
| r32 | 7992 | 0.243 | 0.795 | 0.609 | 0.033 | 6.48e-7 |
| r32 | 23976 | 0.244 | 0.478 | 2.252 | 0.039 | 8.67e-7 |
| r64 | 1 | 0.233 | 0.482 | 1.949 | 0.042 | 0.0 |
| r64 | 7992 | 0.255 | 0.794 | 0.852 | 0.033 | 5.20e-7 |
| r64 | 23976 | 0.229 | 0.477 | 1.951 | 0.027 | 1.90e-6 |
| r128 | 1 | 0.247 | 0.482 | 2.222 | 0.042 | 0.0 |
| r128 | 7992 | 0.294 | 0.792 | 1.647 | 0.033 | 1.59e-6 |
| r128 | 15000 | 0.270 | 0.739 | 1.398 | 0.042 | 1.05e-6 |

Interpretation:

- The models are actually training; LoRA gradients are nonzero.
- Single-step losses are noisy because timesteps are sampled stochastically, so the raw loss is not expected to be monotonic.
- `loss_hf_teacher` is almost always zero or extremely small in these logs. Across the three-epoch continuation file, only 174 of 413,496 rows had nonzero `loss_hf_teacher`, with max value `3.67e-4`. Therefore, we should not overclaim that the HF-teacher term itself drove the result. The more defensible claim is that V4's corrected initialization, low-frequency motion/control losses, and capacity settings preserved a stable quality band while allowing action dependence to grow.

## Main Patterns

### 1. Action conditioning improved substantially relative to no-action

The no-action baseline has `S=0.000`. The best V4 models reach:

- r16 continued step 23976: `S=3.450`.
- r64 three-epoch step 23976: `S=3.157`.
- r128 step 7992: `S=3.281`.

This is strong evidence that the action tensor affects generated futures. It is not yet proof that the model obeys specific driving semantics like braking or steering.

### 2. Rank helps up to 64, but 128 is not a clear win

At one epoch:

- r32: `S=2.621`, FVD 87.427.
- r64: `S=3.057`, FVD 85.602.
- r128: `S=3.281`, FVD 88.569.

Rank 64 improves both sensitivity and FVD relative to rank 32. Rank 128 increases early sensitivity, but its FVD is worse and its sensitivity decreases with additional training. This suggests the system is capacity-sensitive but not monotonically improved by larger rank.

### 3. Longer training is model-dependent

- r16 improved with longer training: `S=2.336 -> 2.827 -> 3.450`, while FVD slightly improved `90.066 -> 89.422 -> 88.465`.
- r32 worsened by the final checkpoint: `S=2.637` at step 15984 to `S=1.904` at step 23976.
- r64 is non-monotonic but promising: `S=2.735 -> 2.763 -> 2.458 -> 3.157`.
- r128 declines after early training: `S=3.281 -> 2.944 -> 2.591`.

The correct policy is not "train everything longer." The promising longer-training candidates are r16 and r64. r32 and r128 should not be continued without a new reason.

### 4. Visual metrics are stable but below no-action

The B200 V4 action models stay in a tight band:

- PSNR: about 17.21-17.30.
- SSIM: about 0.698-0.704.
- Sharpness ratio: about 0.35-0.37.
- Motion ratio: about 0.71-0.72.

The no-action baseline is still better on PSNR/SSIM/FVD:

- no-action: PSNR 17.892, SSIM 0.763, FVD 71.424.
- best new r64 final: PSNR 17.269, SSIM 0.700, FVD 86.805.

So V4 has not beaten the no-action baseline in pure visual fidelity. The gain is controllability/action dependence, not visual quality.

### 5. FVD and action sensitivity trade off

For r64:

- best FVD: step 10000, FVD 82.769, but counterfactual sensitivity was not measured at that checkpoint.
- best measured sensitivity: step 23976, `S=3.157`, FVD 86.805.

This is the core tradeoff: action dependence increases at some cost to the FVD-style visual distribution metric.

## Current Best Checkpoints

Use these for reporting and qualitative video comparison:

1. Best visual/action compromise among new continuations:
   - `v4_main_text_r64_3epoch`, step 23976.
   - `S=3.157`, FVD 86.805, PSNR 17.269, SSIM 0.700.

2. Best action sensitivity overall:
   - `v4_main_text_r16_continue`, step 23976.
   - `S=3.450`, FVD 88.465, PSNR 17.214, SSIM 0.699.

3. Best FVD among new rank/capacity models:
   - `v4_main_text_r64_3epoch`, step 10000.
   - FVD 82.769, PSNR 17.278, SSIM 0.703.
   - Needs counterfactual sensitivity generation before it can be claimed as best overall.

4. Strong but unstable early high-rank model:
   - `v4_main_text_r128_partial`, step 7992.
   - `S=3.281`, FVD 88.569.
   - Not recommended for further training without changes.

## Recommendation

Do not continue r32 or r128 as-is.

If running one small follow-up:

1. Generate counterfactuals for r64 step 10000 and step 18000. These are the high-quality checkpoints; we need to know whether they already have enough action sensitivity.
2. If r64 step 10000 or 18000 has `S >= 2.8`, use it as the main final checkpoint because it has better FVD/quality than r64 final.
3. If their `S` is low, use r64 step 23976 or r16 step 23976 depending on whether the final story prioritizes quality or action dependence.

For the final report, the clean claim should be:

> Pure no-action fine-tuning gives the strongest visual fidelity but no controllability. Earlier action-conditioning paths either damaged quality or produced weak/unstable action dependence. V4 changes the tradeoff: it reaches clear counterfactual action sensitivity in the `S=3.1-3.45` range while avoiding catastrophic collapse. The best current model does not beat the no-action baseline visually; it improves the controllability axis. The remaining open problem is semantic action correctness, which requires action-alignment metrics beyond counterfactual sensitivity.

