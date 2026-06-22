# B200 V4 Rank/Capacity Campaign Analysis

This campaign tested the remaining V4 hypothesis: the full-112 low-frequency action objective was promising, but the previous rank-16 action adapter may have been capacity-limited or simply under-trained.

The run used the corrected recipe throughout: LTX-2B distilled, 24 FPS cached 121-frame windows, 49 context frames, 72 future frames, seed 231, shifted/log-normal timestep sampling, upsampled full-112 frame actions, and inference `image_cond_noise_scale=0.0`. No latents were recached.

## Guardrails

- B200 calibration measured `0.4606 sec/step`, so all three runs were launched.
- Rank-32 audit passed: `adapter_rank=32`, `expand_baseline_lora_to_rank=true`, `freeze_transformer_lora=false`.
- Trainable parameters for rank-32 calibration were nonzero: `38.67M` LoRA params, `7.34M` action encoder params, `14.96M` action injector params, `60.97M` total trainable params.
- LoRA gradients were nonzero in calibration: `last_grad_norm_lora=0.0302`. This matters because otherwise the rank sweep would have been meaningless.

## Runs

| Method | Init | Rank | Training target |
|---|---|---:|---:|
| `v4_main_text_r16_continue` | existing V4 main text `step_015984` | 16 | continue to `step_023976` |
| `v4_main_text_r32` | no-action shifted LoRA `step_003000` | 32 | one epoch, `7992` steps |
| `v4_main_text_r64` | no-action shifted LoRA `step_003000` | 64 | one epoch, `7992` steps |

## Latest Checkpoint Summary

| method | step | rank | FVD-style | PSNR | SSIM | sharpness | motion | FFT HF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v4_main_text_r16_continue` | 23976 | 16 | 88.47 | 17.214 | 0.6989 | 0.3692 | 0.7227 | 0.9903 |
| `v4_main_text_r32` | 7992 | 32 | 87.43 | 17.257 | 0.6997 | 0.3612 | 0.7174 | 0.9764 |
| `v4_main_text_r64` | 7992 | 64 | 85.60 | 17.256 | 0.7000 | 0.3595 | 0.7136 | 0.9778 |

## Best Checkpoints By FVD-Style

| method | best step | FVD-style | PSNR | SSIM | sharpness | motion |
|---|---:|---:|---:|---:|---:|---:|
| `v4_main_text_r16_continue` | 20000 | 87.33 | 17.204 | 0.6986 | 0.3732 | 0.7269 |
| `v4_main_text_r32` | 7000 | 83.15 | 17.265 | 0.7013 | 0.3592 | 0.7152 |
| `v4_main_text_r64` | 4000 | 83.27 | 17.269 | 0.7013 | 0.3562 | 0.7113 |

FVD-style is noisy on only five clips, so the strongest claim is not the exact ranking. The robust pattern is that rank-32 and rank-64 produce lower best FVD-style scores than the rank-16 continuation while maintaining similar PSNR/SSIM. The cost is a small reduction in sharpness and motion ratios.

## Counterfactual Sensitivity

Counterfactual sensitivity is measured as future RGB MAE between correct-action generations and zero/shuffled/reversed-future action generations. Higher sensitivity means the model output changes more when the action signal changes.

| method | step | mean RGB MAE across modes | strongest mode |
|---|---:|---:|---|
| `v4_main_text_r16_continue` | 15984 | 2.34 | reversed-future, 2.99 |
| `v4_main_text_r16_continue` | 22000 | 2.83 | reversed-future, 3.81 |
| `v4_main_text_r16_continue` | 23976 | 3.45 | shuffled, 3.51 |
| `v4_main_text_r32` | 1000 | 1.79 | reversed-future, 1.89 |
| `v4_main_text_r32` | 3000 | 1.82 | shuffled, 1.96 |
| `v4_main_text_r32` | 7992 | 2.62 | shuffled, 2.97 |
| `v4_main_text_r64` | 1000 | 1.87 | shuffled, 1.99 |
| `v4_main_text_r64` | 3000 | 2.85 | zero, 3.44 |
| `v4_main_text_r64` | 7992 | 3.06 | shuffled, 3.98 |

This is the most important positive result. Longer training and higher rank both increase counterfactual action sensitivity. Rank-64 reaches strong sensitivity by one epoch, while rank-16 continuation reaches the highest average sensitivity after a third epoch-equivalent continuation.

## Loss Patterns

The raw total loss is stochastic because timesteps are sampled, so it should not be interpreted as a smooth validation curve. The useful patterns are component-level:

- Rank-32 action auxiliary loss decreased from `2.81` at the start to `0.61` at step `7992`.
- Rank-64 action auxiliary loss decreased from `1.95` to `0.85`.
- Low-frequency target loss decreased for high-rank runs from about `0.0071` to about `0.0057`.
- Diffusion loss rose in the final sampled step for rank-32/rank-64, so the high-rank models may be trading denoising alignment for low-frequency/action matching.

The deduplicated loss table is `loss_curves_by_model_dedup.csv`; the original `loss_curves_by_model.csv` contains cumulative checkpoint-history duplicates.

## Conclusions

1. V4 is not flat. The earlier concern that V4 was not learning is not supported by this run. Counterfactual sensitivity increases with both more training and more rank.
2. Rank/capacity matters. Rank-64 gives the clearest one-epoch action sensitivity gain and the best latest FVD-style among the three final checkpoints.
3. Longer rank-16 training also matters. Continuing rank-16 from `15984` to `23976` increases average counterfactual sensitivity from `2.34` to `3.45` while keeping PSNR, SSIM, sharpness, motion, and FFT HF almost flat.
4. There is still a quality/control tradeoff. Rank-32/rank-64 improve FVD-style and action sensitivity but slightly reduce sharpness/motion ratios relative to rank-16 continuation.
5. The most defensible final story is that action conditioning in pretrained video DiTs is not solved by naive token injection; the successful direction is low-frequency action supervision with constrained full-112 temporal conditioning, and both adapter capacity and training duration materially affect controllability.

## Best Current Checkpoint Choices

- Best quality/control compromise for final report: `v4_main_text_r64` at `step_007992`.
- Best FVD-style checkpoint: `v4_main_text_r32` at `step_007000` or `v4_main_text_r64` at `step_004000`; treat this as noisy because it is five-clip FVD-style.
- Best evidence for longer-training benefit: `v4_main_text_r16_continue` from `step_015984` to `step_023976`.

## Caveat

All metrics here are still on the five local validation clips. These results are strong enough to choose the final story and final examples, but the final report should avoid claiming whole-validation superiority unless the separate full/stratified validation run confirms it.
