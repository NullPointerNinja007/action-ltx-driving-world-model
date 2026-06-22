# Selected Checkpoint Operating-Point Analysis

This file adds the missing counterfactual action-sensitivity numbers for intermediate checkpoints that looked promising by FVD/PSNR/SSIM. The goal is to answer: did we pass through a better model before the final checkpoint?

All results use the same 5 validation clips, seed 231, 24 FPS, 49 context frames, 72 future frames, corrected `image_cond_noise_scale=0.0`, and the same FVD-style backend used in the B200 V4 campaign.

## Main Table

| Model | Step | S | FVD | PSNR | SSIM | Sharpness | FFT-HF | Motion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-action shifted | 3000 | 0.000 | 71.424 | 17.892 | 0.763 | 0.271 | - | 0.790 |
| r16 continued | 23976 | 3.450 | 88.465 | 17.214 | 0.699 | 0.369 | 0.990 | 0.723 |
| r32 | 10000 | 3.893 | 88.303 | 17.257 | 0.700 | 0.362 | 0.980 | 0.717 |
| r64 | 10000 | 2.454 | **82.769** | 17.278 | 0.703 | 0.360 | 0.983 | 0.710 |
| r64 | 18000 | 3.499 | 86.108 | **17.298** | **0.704** | 0.357 | 0.978 | 0.709 |
| r64 | 23976 | 3.157 | 86.805 | 17.269 | 0.700 | 0.359 | 0.973 | 0.711 |
| r128 | 12000 | 2.944 | 83.707 | 17.278 | 0.701 | 0.354 | 0.979 | 0.711 |

`S` is the average RGB MAE between correct-action generation and zero/shuffled/reversed-future action generations. It measures action dependence, not semantic correctness.

## Answer

Yes, we did pass through a better checkpoint. The best final selection is **not** the final r64 checkpoint. The strongest balanced checkpoint is:

```text
v4_main_text_r64_3epoch step_018000
```

Why:

- It has `S=3.499`, which is higher than r64 final `S=3.157`.
- It has better FVD than r64 final: `86.108` vs `86.805`.
- It has better PSNR than r64 final: `17.298` vs `17.269`.
- It has better SSIM than r64 final: `0.704` vs `0.700`.
- Its sharpness/motion remain in the same stable band: sharpness `0.357`, motion `0.709`.

This means continuing r64 to 23,976 steps did not improve the best operating point. It slightly hurt the balanced tradeoff.

## Model-Specific Findings

### r64 is the best balanced family

r64 has two useful checkpoints:

- `step_010000`: best FVD, `82.769`, but action sensitivity is only `S=2.454`.
- `step_018000`: best overall balance, `S=3.499`, FVD `86.108`, PSNR `17.298`, SSIM `0.704`.

If the final report needs one action-conditioned checkpoint, use `r64 step_018000`.

### r32 has the highest sensitivity but is less trustworthy

r32 `step_010000` has the highest observed action sensitivity:

```text
S = 3.893
FVD = 88.303
PSNR = 17.257
SSIM = 0.700
```

Its per-mode sensitivities are all high:

- correct vs reversed future: `3.998`
- correct vs shuffled: `3.742`
- correct vs zero: `3.938`

This means the output changes strongly under action perturbations, but the FVD/quality tradeoff is worse than r64 `step_018000`. It is a useful "maximum action sensitivity" checkpoint, not the best final model.

### r128 improves FVD but not action sensitivity enough

r128 `step_012000` is a good quality checkpoint:

```text
S = 2.944
FVD = 83.707
PSNR = 17.278
SSIM = 0.701
```

This is a good FVD/action compromise if the threshold is `S >= 2.8`. But r64 `step_018000` has much stronger action sensitivity and slightly better PSNR/SSIM. r128 still does not justify itself as the final model.

### r16 remains the strongest old sensitivity/visual-stability point

r16 `step_023976` has:

```text
S = 3.450
FVD = 88.465
PSNR = 17.214
SSIM = 0.699
sharpness = 0.369
motion = 0.723
```

It preserves the most sharpness/motion among the high-sensitivity candidates, but r64 `step_018000` has better FVD, PSNR, and SSIM with slightly higher `S`. Therefore r16 is now secondary.

## Best Checkpoints By Use Case

| Use case | Best checkpoint | Reason |
|---|---|---|
| Main final action-conditioned model | r64 `step_018000` | Best balanced `S`, FVD, PSNR, SSIM |
| Maximum action sensitivity | r32 `step_010000` | Highest observed `S=3.893` |
| Best FVD among action models | r64 `step_010000` | Lowest action-model FVD `82.769`, but weaker `S=2.454` |
| Best quality with `S >= 2.8` | r128 `step_012000` | FVD `83.707`, `S=2.944` |
| Best old stable high-sensitivity baseline | r16 `step_023976` | `S=3.450`, high sharpness/motion |

## Final Recommendation

Use r64 `step_018000` as the primary action-conditioned model in the report. It is the clearest evidence that intermediate checkpoint selection matters:

```text
r64 step_018000: S=3.499, FVD=86.108, PSNR=17.298, SSIM=0.704
r64 step_023976: S=3.157, FVD=86.805, PSNR=17.269, SSIM=0.700
```

This lets us state a stronger and more precise conclusion:

> The best action-conditioned model was not the final checkpoint. For rank 64, training to three epochs passed through a stronger controllability/fidelity operating point at step 18,000. This checkpoint improves action sensitivity and reconstruction metrics over the final checkpoint while staying in the same sharpness/motion band.

## Files

- Full table: `selected_checkpoint_sensitivity_quality_table.csv`
- Operating-point plot: `selected_checkpoint_operating_points.png`
- Updated counterfactual summary: `counterfactual_sensitivity_summary.csv`

