---
title: "Milestone 3 — Preliminary Results"
subtitle: "Action-Conditioned Video World Models for Long-Tail Driving"
date: "May 29, 2026"
course: "Stanford CS231N"
authors: "Andrew Liang and Maleeka Raddygala"
---

# Milestone 3 — Preliminary Results

## Action-Conditioned Video World Models for Long-Tail Driving

Stanford CS231N, May 29, 2026

Andrew Liang and Maleeka Raddygala

---

# Goal

We are building a driving video world model that generates plausible future front-camera video from an observed driving context.

The long-term goal is action-conditioned future generation, but this milestone focuses on the prerequisite visual question:

**Can LTX-Video 2B be adapted to Waymo-style front-camera video without degrading the base video prior?**

---

# What We Completed

- Staged Waymo-style front-camera clips for repeated local and Modal inference.
- Interpolated selected 20 second Waymo clips from 10 FPS to 24 FPS.
- Ran fixed-seed LTX-Video 2B distilled inference on 5 held-out scenarios.
- Trained and evaluated both a failed 512-window smoke-test LoRA and a corrected full-data LoRA.
- Ran a narrow full-data LoRA rank sweep with ranks 8, 16, and 32 using the same LR, data, seed, and checkpoints.
- Evaluated all saved checkpoints for ranks 8, 16, and 32 to select the best checkpoint rather than assuming the final step is best.
- Built a benchmark harness that compares generated future frames against the real future.
- Added quality gates for blur and under-motion so PSNR alone cannot hide visual collapse.

---

# Experimental Setup

Evaluation clips:

- 5 held-out Waymo front-camera scenarios.
- Source clips: 20 seconds, originally 10 FPS, interpolated to 24 FPS.
- Context: 49 frames at 24 FPS, approximately 2 seconds.
- Future: 72 frames at 24 FPS, exactly 3 seconds.
- Total generated sequence: 121 frames.

Models:

- Baseline: LTX-Video 2B 0.9.8 distilled, no LoRA.
- Fine-tuned: LTX-Video 2B 0.9.8 distilled + LoRA rank 16, full-data step 3000.
- Rank sweep: LoRA ranks 8, 16, and 32, each trained to step 3000 with LR 5e-6.

---

# Training Runs

The first LoRA run was a smoke test:

- Training subset: 512 windows, not the full approximately 8,000-window training set.
- LoRA rank: 16.
- Learning rate: 2e-5.
- Checkpoint evaluated: step 500.
- No ego/action conditioning.

That run visibly collapsed sharpness and motion. We then ran the corrected full-data training recipe:

- Training set: 7,992 cached latent windows.
- Validation manifest: 1,904 cached latent windows, with 32 rows loaded during training.
- LoRA rank: 16.
- Learning rate: 5e-6.
- Checkpoint evaluated: step 3000.
- Runtime: approximately 0.5 hours on one H100.

No latent recaching was needed; the complete cache already existed under `latents/`.

We then trained ranks 8 and 32 with the same data, LR, seed, and checkpoint schedule to test whether LoRA capacity was responsible for the remaining blur/quality tradeoff.

Finally, we evaluated every saved checkpoint for ranks 8, 16, and 32: steps 100, 250, 500, 1000, 1500, 2000, 2500, and 3000.

---

# Evaluation Metrics

Metrics are computed on the future region only unless otherwise noted.

Reference-based metrics:

- Future PSNR, MSE, MAE.
- Future global SSIM.

Quality diagnostics:

- Laplacian sharpness ratio: generated future sharpness divided by real future sharpness.
- Motion ratio: generated frame-to-frame change divided by real future frame-to-frame change.
- Boundary continuity at context-to-future transition.
- Context-copy leakage score.

---

# Quantitative Results

| Model | Future PSNR ↑ | Future SSIM ↑ | Future MSE ↓ | Sharpness Ratio ↑ | Motion Ratio ↑ |
|---|---:|---:|---:|---:|---:|
| Base distilled, no LoRA | 17.53 | 0.749 | 2151 | 0.359 | 0.889 |
| Full-data LoRA, step 3000 | 17.83 | 0.761 | 2081 | 0.275 | 0.807 |

The full-data LoRA improves future similarity metrics and mostly recovers motion, but it still loses too much sharpness.

---

# Checkpoint Sweep

Same data, LR 5e-6, seed 231, and 5 held-out scenarios:

| Selection | Rank / Step | Future PSNR ↑ | Future SSIM ↑ | Sharpness Retention | Motion Retention | Gate |
|---|---:|---:|---:|---:|---:|---|
| Best valid | r8 / 1500 | 17.65 | 0.755 | 89.2% | 96.5% | Pass |
| Raw SSIM best | r16 / 3000 | 17.83 | 0.761 | 76.7% | 90.7% | Fail |
| Best r16 pass | r16 / 100 | 17.55 | 0.750 | 99.8% | 99.6% | Pass |
| Best r32 pass | r32 / 100 | 17.54 | 0.750 | 98.5% | 99.7% | Pass |

The best gate-passing checkpoint is rank 8 at step 1500. The raw SSIM/PSNR winner is still rank 16 at step 3000, but it fails the sharpness gate.

---

# Quality Gate

We added a base-vs-LoRA quality gate:

- LoRA sharpness retention vs base must be at least 80%.
- LoRA motion retention vs base must be at least 80%.
- LoRA future SSIM delta vs base must be at least -0.01.

Observed:

- Sharpness retention: 76.7%.
- Motion retention: 90.7%.
- Future SSIM delta: +0.0124.

For rank 16 step 3000: **FAIL, narrowly due to sharpness**.

For rank 8 step 1500: **PASS**, with future SSIM above base and acceptable sharpness/motion retention.

---

# Key Insight

Pixel metrics are not enough for video generation.

The corrected full-data LoRA run is clearly better than the 512-window smoke test, but the quality gate still catches remaining over-smoothing.

The metrics show:

- Future SSIM and PSNR improve over base on average.
- Motion retention now passes the gate.
- Sharpness remains below the pass threshold.
- Base distilled is still sharper, while full-data LoRA is closer to the real future by several reference metrics.
- Checkpoint selection matters: r8 step 1500 passes the gate, while r16 step 3000 is the raw metric winner but too smooth.

---

# Qualitative Analysis

Visual inspection of side-by-side rollouts showed:

- The full-data LoRA is much less broken than the initial 512-window LoRA.
- Base distilled rollouts retain more texture and fine detail.
- Full-data LoRA rollouts are smoother and closer by SSIM, but still somewhat over-smoothed.
- The generated futures still struggle with stable long-horizon driving geometry.

Interpretation: the corrected LoRA recipe is moving in the right direction, but checkpoint selection and/or context length still need work.

---

# What Is Working

- The inference pipeline is reproducible across Modal and local outputs.
- Fixed-seed comparisons are now possible for base and LoRA checkpoints.
- The benchmark harness catches blur and under-motion that PSNR misses.
- The source clips, generated outputs, and evaluation summaries are all traceable through manifests.
- Full-data lower-LR training improves over the failed smoke-test checkpoint.
- The rank sweep gives a controlled capacity signal without changing the data pipeline.
- The checkpoint sweep found a usable gate-passing checkpoint: rank 8 step 1500.

These pieces give us a controlled loop for future training runs.

---

# What Is Not Working

- The raw-best r16 step 3000 checkpoint still misses the sharpness gate.
- Only 5 clips have been benchmarked so far.
- The current 49-frame context may be too short for driving dynamics.
- The training objective and conditioning/loss masking need auditing before scaling up.

---

# Limitations

Current results are preliminary.

- Only 5 scenarios were evaluated in this benchmark run.
- We have not yet run LPIPS, DINO/CLIP feature distance, FVD, or RAFT flow-warp metrics.
- Global SSIM is a lightweight proxy, not a full perceptual video metric.
- The full-data result is still visual continuation only, not action conditioning.
- This milestone evaluates visual continuation, not action conditioning.

---

# Next Steps

Immediate:

- Audit the training objective, timestep target, latent scaling, and future-only loss masking.
- Visually inspect r8 step 1500 against base and r16 step 3000 on the same five scenarios.
- Expand checkpoint validation from 5 clips to 50-200 held-out clips.
- Increase context to 81 frames or 121 frames while keeping a 72-frame future.
- Use the existing full-data latent cache for the current 121-frame setup; recaching is only needed if we change frame counts.

Evaluation:

- Add LPIPS and DINO/CLIP feature distance.
- Add RAFT-style flow consistency or a stronger temporal metric.
- Expand the validation set from 5 clips to 50-200 held-out clips.
- Keep side-by-side human preference checks for every checkpoint.

---

# Current Conclusion

The full-data lower-LR LoRA is a substantial improvement over the failed smoke test, but the final checkpoint is not the best usable checkpoint.

The best gate-passing checkpoint is rank 8 step 1500. It improves future SSIM over base while retaining 89.2% sharpness and 96.5% motion.

This result changes the plan: before action conditioning, we should validate r8 step 1500 on more clips, test longer context, and expand the benchmark metrics.
