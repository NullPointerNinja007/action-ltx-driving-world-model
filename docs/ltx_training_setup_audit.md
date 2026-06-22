# LTX-Waymo Training Setup Audit

Date: 2026-06-01

This audit checks the current LTX-Waymo fine-tuning setup against the concern list in the pasted notes and the attached LTX-Video paper. The main question was whether the training runs were invalid because we missed a paper-specific loss or mismatched LTX's rectified-flow / conditioning conventions.

## Bottom Line

The training setup is **not fundamentally wrong because we skipped the LTX VAE losses**. The paper's Reconstruction-GAN, Video-DWT, LPIPS, and pixel reconstruction losses are VAE / decoder training losses. Our project freezes the VAE and trains transformer LoRA plus optional action encoders on cached latents, so those losses are not required for the current setup.

The most important high-risk items are mostly handled correctly:

- Rectified-flow target: **correct**. We train against `noise - clean_latents`, matching the LTX scheduler velocity convention.
- Context timestep conditioning: **correct in training**. Context tokens get timestep `0`; future tokens get sampled timestep `t`.
- Context/future latent boundary: **correct for 49 context frames**. `49 = 1 + 6 * 8`, so the boundary is exactly 7 latent frames under LTX's first-frame-plus-8-frame temporal compression.
- VAE latent normalization: **appears consistent**. Latents are cached with `vae_per_channel_normalize=True` and inference also uses `vae_per_channel_normalize=True`.

Two concerns are real or partially real:

- Inference context noise was mismatched: **real and patched**. Inference had `image_cond_noise_scale=0.15` while training used clean context latents. This is now configurable and defaults to `0.0`.
- Timestep sampling is not LTX-paper-equivalent: **real deviation, not a sign/target bug**. Training currently samples `t` uniformly. The paper says LTX transformer training uses shifted log-normal timestep sampling biased by token count. This should be tested as an ablation before more large runs.

## Sources Checked

- Attached paper: `LTXvideo.pdf`
- Concern list: `pasted-text.txt`
- Current trainer: `pipelines/training/train_ltx2b_waymo_visual_lora.py`
- Current no-action inference wrapper: `pipelines/inference/generate_waymo24_minterpolate_lora.py`
- Current action inference wrapper: `pipelines/inference/generate_waymo24_action_minterpolate_lora.py`
- Local LTX implementation: `<ltx-video>/ltx_video/`

## Detailed Checklist

| Concern | Status | Evidence / conclusion |
|---|---:|---|
| Missing LTX VAE losses make LoRA training invalid | False alarm | The paper's rGAN, Video-DWT, LPIPS, and pixel reconstruction losses are for Video-VAE / denoising decoder training. We freeze the VAE and train in latent space. |
| Transformer target might be wrong | Verified OK | Trainer uses `target_tokens = patchify(noise - clean_latents)`. LTX scheduler says model output is velocity and updates `prev_sample = sample - dt * model_output`. |
| Sign convention might be wrong | Verified OK | With `z_t = (1-t) z_0 + t epsilon`, velocity is `epsilon - z_0`; scheduler step subtracts predicted velocity. Our target has the same sign. |
| Clean context tokens might be passed with noisy global timestep | Verified OK in training | Trainer builds `token_timesteps` per token: future tokens use sampled `t`, context tokens use `0`. |
| Future-only loss might be wrong | Verified OK | For continuation, context is conditioning and future is prediction target. Loss mask is future-only. |
| Context/future mask may ignore temporal compression | Verified OK for current setup | `CONTEXT_FRAMES = 49`, `CONTEXT_LATENT_FRAMES = 49 // 8 + 1 = 7`. Because LTX has first-frame latent plus 8-frame chunks, this is an exact boundary. |
| Inference and training conditioning may mismatch | Real issue patched | Inference used `image_cond_noise_scale=0.15`, while training context was clean. Now `LTX_IMAGE_COND_NOISE_SCALE` defaults to `0.0` in training validation, no-action inference, and action inference. |
| Timestep sampling may not match LTX | Real deviation | Trainer uses uniform `torch.rand`. The paper describes shifted log-normal sampling. This is not a fatal objective bug, but it is a plausible quality issue. |
| Action/frame alignment after VAE compression may be wrong | Mostly OK, still worth testing | Frame-action tensors are `[121,18]` and encoders know context/future segment IDs. However, action tokens are global conditioning tokens, not latent-token-aligned controls. This is by design, not a loader bug. |
| AdaLN overpowering visual prior | Confirmed likely | Metrics already show AdaLN gives high PSNR/SSIM but poor sharpness/motion/FVD-style behavior. This is a modeling issue, not a data bug. |
| LoRA over-adapts and causes blur | Confirmed likely | No-action LoRA improves PSNR/SSIM but reduces sharpness/motion. Earlier checkpoints or gentler adapters are more visually plausible. |
| Need correct baseline | Verified OK | We now treat no-action visual LoRA as the baseline, not zero-shot LTX. |
| Need action causality tests | Still open | Metrics do not prove action use. Need same-context / same-seed counterfactual actions. |

## Code-Level Findings

### Rectified-Flow Target

Current trainer:

```python
noisy_latents = pipeline.scheduler.add_noise(clean_latents.float(), noise.float(), timesteps)
target_tokens, _ = pipeline.patchifier.patchify((noise - clean_latents).to(torch.bfloat16))
```

LTX scheduler implementation:

```python
prev_sample = sample - dt * model_output
noisy_samples = (1 - timesteps) * original_samples + timesteps * noise
```

This confirms the model output is velocity, not clean latent and not raw noise.

### Per-Token Context Timesteps

Current trainer:

```python
future_token_mask = latent_coords[:, 0] >= CONTEXT_LATENT_FRAMES
token_timesteps = torch.where(
    future_token_mask,
    timesteps[:, None].expand_as(future_token_mask).to(torch.float32),
    torch.zeros_like(future_token_mask, dtype=torch.float32),
)
```

This matches the LTX paper's timestep-based conditioning idea: conditioning tokens can have timestep `0` while generated tokens have a nonzero diffusion timestep.

### Patched Inference Context Noise

Before patch:

```python
image_cond_noise_scale=0.15
```

After patch:

```python
IMAGE_COND_NOISE_SCALE = float(os.environ.get("LTX_IMAGE_COND_NOISE_SCALE", "0.0"))
image_cond_noise_scale=IMAGE_COND_NOISE_SCALE
```

Files patched:

- `pipelines/inference/generate_waymo24_minterpolate_lora.py`
- `pipelines/inference/generate_waymo24_action_minterpolate_lora.py`
- `pipelines/training/train_ltx2b_waymo_visual_lora.py`

This means future evaluations now match the clean-context training convention by default. If we intentionally want the old LTX-style conditioning noise, set:

```bash
export LTX_IMAGE_COND_NOISE_SCALE=0.15
```

## What Is Still Not Proved

The audit does **not** prove the model learned action causality. It only verifies that the core training math is not obviously broken.

Still needed:

1. Run a counterfactual action suite with same context and seed but changed future actions.
2. Add an ablation with LTX-style shifted log-normal timestep sampling.
3. Compare inference with `LTX_IMAGE_COND_NOISE_SCALE=0.0` versus `0.15` on the same checkpoints.
4. Prefer model selection using sharpness, motion, temporal error, FVD-style metrics, and video inspection, not PSNR/SSIM alone.

## Practical Recommendation

Before running another expensive full training sweep:

1. Do a 500--1000 step no-action ablation with current clean-context inference default.
2. Do a 500--1000 step no-action ablation with shifted log-normal timestep sampling.
3. Generate the same five clips from both.
4. If shifted sampling improves sharpness/motion without hurting stability, make it the default.
5. For action conditioning, use token-based action encoders first; avoid AdaLN unless it is strongly gated or trained at a much lower action LR.
