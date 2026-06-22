# V4 Full112 Low-Frequency Action Pilot Plan

Date: 2026-06-03

## Summary

Run a fast, controlled V4 pilot before committing to another long action-conditioning campaign. The pilot tests whether richer full-action conditioning plus low-frequency motion supervision can improve action responsiveness without introducing the wobble and temporal instability seen in earlier action methods.

The pilot uses two 1000-step variants in parallel on H100:

- `V4-main`: stronger low-frequency action supervision.
- `V4-conservative`: stronger diffusion anchoring and lower auxiliary pressure.

This plan is based on our prior results and the papers stored in `docs/inspiration_papers/`. The main paper-derived lesson is that successful conditional video systems preserve the video prior, use dense or geometrically meaningful conditions, avoid invasive global conditioning, and supervise temporal/geometric structure explicitly.

## Hard Context For Interpretation

Do not treat raw sharpness drop against zero-shot LTX as the main failure signal. Zero-shot LTX can generate sharper and more polished video than the lower-sharpness 512px Waymo training clips. Fine-tuning toward Waymo can legitimately reduce sharpness.

Meaningful comparisons are:

- action-conditioned model vs corrected no-action shifted LoRA;
- action-conditioned model vs the same checkpoint with action gate scale `0`;
- correct actions vs zero/shuffled/reversed actions;
- temporal consistency, wobble, motion ratio, FVD-style distance, and action sensitivity.

The real failure mode is not simply lower sharpness. The real failure mode is extra wobble, temporal weirdness, low-motion smoothing, weak counterfactual sensitivity, or degradation relative to a matched no-action/gate-zero baseline.

## Pre-Implementation Checklist

Before changing training behavior:

- Confirm the current Modal account has the required data volumes and checkpoint volumes.
- Confirm cached latents are present; do not recache latents.
- Confirm frame-action `.npz` files contain both `actions` `[121,18]` and `actions_full_112` `[121,112]`.
- Confirm the current training loader only consumes compact `actions` unless explicitly extended.
- Confirm corrected no-action shifted LoRA `step_003000` is accessible.
- Confirm inference uses `image_cond_noise_scale=0.0`.
- Confirm all heavy metrics run on Modal, not locally.
- Confirm text conditioning remains enabled.
- Confirm no original 10 FPS action sequence is used unless explicitly requested.

## Action/Data Audit

Run a cheap Modal audit before training. This should not take more than roughly 15-40 minutes if implemented with low-res frame-difference or block-motion proxies rather than heavy optical flow.

Use:

- existing 24 FPS upsampled windows;
- existing MP4 windows;
- existing frame-action `.npz` files;
- `512` train windows;
- `128` val windows;
- the 5 local validation clips.

Compare:

- compact action tensor: `actions [121,18]`;
- full action tensor: `actions_full_112 [121,112]`.

Compute simple visual motion proxies:

- low-res frame delta magnitude;
- horizontal motion proxy;
- temporal delta / acceleration proxy for wobble;
- optional block-matching proxy if cheap enough.

Sweep action-image temporal offsets:

```text
-12, -6, 0, +6, +12 frames
```

Proceed only if offset `0` is top-2 or quantitatively close to top-2. If another offset clearly dominates, fix alignment before training.

Default V4 input is `full112` unless this audit clearly shows compact18 is better.

## V4 Model

Add a new action mode:

```text
frame_temporal_bottleneck_fullaction_motion_v4
```

Architecture:

```text
input actions: [B,121,112]
per-frame MLP: 112 -> 384
temporal transformer: 4 layers, 8 heads, width 384, dropout 0.0
latent-time pooling: use latent_coords
injection blocks: [10,14,18]
bounded gates: 0.25 * tanh(gamma)
```

The action residual is broadcast to spatial tokens at the same latent-time bin. It should affect low-frequency temporal evolution, not directly rewrite every spatial detail.

Trainable:

```text
full-action temporal encoder
temporal residual projectors
bounded gates
low-frequency motion auxiliary head
```

Frozen:

```text
VAE
text encoder
base transformer
corrected no-action shifted LoRA step_003000
```

Keep text conditioning enabled. The no-text ablation did not show that removing text solves the problem.

## Training Pilot

Run both variants in parallel on separate H100 Modal containers.

Shared setup:

```text
base model: ltxv-2b-0.9.8-distilled.safetensors
baseline LoRA volume: ltx2b-dist098-waymo24-noaction-shifted-timestep-ablation-ckpts
baseline run: ltx2b_dist098_waymo24_noaction_shifted_lognormal_r16_seed231_lr5e6_steps6000_resume1000
baseline step: step_003000
seed: 231
fps: 24
context frames: 49
future frames: 72
total frames: 121
batch size: 1
max steps: 1000
checkpoint steps: 0,100,250,500,750,1000
timestep sampling: shifted_lognormal
timestep_lognormal_mean: 0.0
timestep_lognormal_std: 1.0
inference image_cond_noise_scale: 0.0
gpu: H100
```

Run names:

```text
ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_seed231_from_shifted_noaction_step003000_steps1000
ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_conservative_seed231_from_shifted_noaction_step003000_steps1000
```

Checkpoint volume:

```text
ltx2b-dist098-waymo24-full112-lowfreq-motion-v4-r16-ckpts
```

### V4-main Loss Weights

```text
diffusion = 0.25
lowfreq_target = 1.00
lowfreq_delta = 1.00
hf_teacher = 0.20
action_motion_aux = 0.05
residual_norm = 0.002
gate = 0.002
action_lr = 3e-5
injector_lr = 3e-5
gate_lr = 3e-5
```

Purpose: aggressively test whether low-frequency geometric/action supervision improves controllability.

### V4-conservative Loss Weights

```text
diffusion = 0.50
lowfreq_target = 0.50
lowfreq_delta = 0.50
hf_teacher = 0.10
action_motion_aux = 0.05
residual_norm = 0.002
gate = 0.002
action_lr = 3e-5
injector_lr = 3e-5
gate_lr = 3e-5
```

Purpose: preserve the diffusion objective more strongly in case V4-main over-regularizes or creates artificial temporal behavior.

### Required Training Logs

Log:

- raw loss components;
- weighted loss components;
- `mean_abs_gate`;
- `max_abs_gate`;
- `action_residual_norm`;
- per-group gradient norms where feasible.

The gradient norms matter because scalar loss weights alone do not reveal which objective dominates optimization.

## Evaluation

### 5-Clip Checkpoint Sweep

Generate all checkpoints for both variants:

```text
2 variants * 6 checkpoints * 5 clips = 60 videos
```

Metrics:

```text
FVD-style future distance
PSNR
SSIM
sharpness ratio vs GT
FFT high-frequency ratio
motion ratio
low-frequency motion ratio
temporal delta error
temporal acceleration / wobble error
copy leakage
boundary error
```

Use Modal sharding:

```text
pilot quality/FVD containers: 16-32
full validation containers later: 64-100
```

Do not run metrics locally. The pilot workload is small enough that 100 containers would mostly waste startup overhead. Use 16-32 containers for the pilot, and reserve 64-100 containers for full validation.

### Counterfactual Sensitivity

Run counterfactuals only at:

```text
step_000500
step_001000
```

Modes:

```text
correct
zero
shuffled
reversed_future
```

Count:

```text
2 variants * 2 checkpoints * 4 modes * 5 clips = 80 videos
```

Compute:

```text
RGB MAE correct vs counterfactual
temporal delta MAE correct vs counterfactual
low-frequency motion delta correct vs counterfactual
```

### Comparison Targets

Compare V4 against:

- corrected no-action shifted LoRA;
- best V3 low-frequency bottleneck;
- best temporal bottleneck HF-teacher;
- best middle-block gated cross-attention;
- best frame transformer action;
- best AdaLN action.

## Timing And Resource Estimate

Prior V3 logs show:

```text
3000 H100 steps took about 28 minutes
steady-state training was about 0.55 sec/step
```

Full112 V4 may be slightly slower, but the pilot is still small.

Expected wall-clock:

```text
audit: 15-40 min
full112 normalization/stats support: 10-20 min
two 1000-step trainings in parallel: 20-40 min
60-video checkpoint generation: 30-75 min
quality + FVD metrics: 5-20 min
80-video counterfactual suite: 30-90 min
total pilot: roughly 2-4 hours
```

Use H100 for the pilot. B200 is not worth switching to for this stage because the training part is short; the bottleneck is orchestration, generation, and evaluation.

## Decision Rules

Continue only one variant unless both provide clearly useful but different tradeoffs.

If V4-main wins:

```text
Continue V4-main to longer training.
Conclusion: stronger low-frequency action supervision is useful.
```

If V4-conservative wins:

```text
Continue V4-conservative.
Conclusion: V4 architecture helps, but stronger diffusion anchoring is needed.
```

If both improve action sensitivity but one preserves temporal stability better:

```text
Continue the more stable one, not necessarily the highest-sensitivity one.
```

If both show low action sensitivity:

```text
Do not extend.
Likely issue: weak action signal, action/latent mismatch, or insufficient objective coupling.
```

If both cause wobble:

```text
Stop V4.
Move toward explicit dense geometric conditioning or low-res motion-field supervision.
```

## Test Plan Before Full Pilot

Before running the pilot:

- Syntax-check changed training, inference, audit, and runner scripts.
- Run a 10-step smoke test for each variant on `train_limit=8`.
- Verify `full112` tensors load as `[121,112]`.
- Verify full112 normalization stats contain 112 features.
- Verify checkpoint output contains:
  - `action_encoder.pt`;
  - `action_injector.pt`;
  - `trainer_state.pt`;
  - `training_config.json`;
  - `loss_history.json`.
- Verify zero-gate output matches the corrected no-action path.
- Verify inference records:

```text
image_cond_noise_scale = 0.0
action_encoder_type = frame_temporal_bottleneck_fullaction_motion_v4
frame_action_feature_mode = full112
```

## Side Note: Prior Models That May Need Longer Training

These are not the first priority before the V4 pilot, but they should not be dismissed solely because 3000 steps looked weak. For a 2B model with constrained action pathways, 3000 steps can be undertraining.

### 1. `frame_temporal_bottleneck_lowfreq_v3`

Reason:

- This was the best direction before V4.
- Low-gate settings showed smoother tradeoffs and some promising FVD/action-sensitivity behavior.
- The architecture is constrained enough that it may need longer training.

Recommended follow-up:

- Revisit only after the V4 pilot.
- Continue the best V3 low-gate setting or rerun with better gate LR if V4 does not dominate.

### 2. `frame_temporal_bottleneck_hf_teacher`

Reason:

- Conceptually aligned with high-frequency preservation.
- Early versions may have undertrained or used suboptimal gate scaling.

Recommended follow-up:

- Not first priority.
- Revisit only if V4 confirms low-frequency bottlenecks are useful but full112 adds little.

### 3. `frame_transformer`

Reason:

- Earlier frame-transformer conditioning preserved the generated distribution better than AdaLN in some metrics.
- It may be undertrained rather than structurally dead.

Recommended follow-up:

- Longer run only if V4 fails and a simpler token-based fallback is needed.

### 4. Corrected No-Action Shifted LoRA

Reason:

- This remains the visual baseline.
- It already has longer runs, but improving it can strengthen all comparisons if action models continue to fail.

Recommended follow-up:

- Not urgent unless baseline quality needs to be improved for final comparisons.

### Deprioritized Models

Do not spend remaining time extending these unless there is a specific diagnostic reason:

- AdaLN action conditioning: high risk of visual/temporal corruption.
- Global MLP action tokens: too weak and global.
- No-text action variants: prior ablation did not suggest text removal solves the issue.

## Assumptions

- No latent recaching.
- No original 10 FPS actions.
- No T5/text encoder training.
- Text conditioning stays enabled.
- Corrected no-action shifted LoRA remains the main baseline.
- Heavy generation and metrics run on Modal.
- Sharpness is interpreted relative to GT and corrected no-action LoRA, not zero-shot LTX.
