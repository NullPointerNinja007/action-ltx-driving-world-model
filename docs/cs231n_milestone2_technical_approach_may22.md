---
title: "Milestone 2 — Technical Approach"
subtitle: "Action-Conditioned Video World Models for Long-Tail Driving"
date: "May 22, 2026"
course: "Stanford CS231N"
authors: "Andrew Liang and Maleeka Raddygala"
---

# Milestone 2 — Technical Approach

## Action-Conditioned Video World Models for Long-Tail Driving

Stanford CS231N, May 22, 2026

Andrew Liang and Maleeka Raddygala

---

# Goal

Build and evaluate a driving video world model that generates plausible future video conditioned on an ego action.

The key question is not only whether the video looks realistic, but whether different actions produce meaningfully different futures.

Target setting: long-tail driving scenarios from WOD-E2E, where rare situations matter more than average-case visual quality.

---

# Core Hypothesis

Action-conditioned video world models fail if the base video prior is not aligned with driving dynamics.

Our current hypothesis after the LTX experiments:

- The base model pretraining may matter more than the action injection mechanism.
- Aggregate video metrics can hide action-conditioning failures and visual collapse.
- A useful evaluation must measure both video realism and action discriminability.

---

# Dataset Plan

Main dataset: WOD-E2E front-camera driving clips.

- 20 second Waymo-style scenarios at approximately 10 FPS.
- Processed train set: 2,037 segments and 4,040 latent clips.
- Processed validation set: 950 latent files.
- Stored and trained through Modal volumes, including `wod-e2e-latents`.
- Ego action is derived from future XY trajectories and kinematic features.

We focus on front-camera video because it directly matches the rollout view used for planning-style evaluation.

---

# Baselines Already Established

Zero-shot LTX-Video 2B on WOD-E2E is our locked baseline.

Metrics from the CS231N repo:

| Metric | Value |
|---|---:|
| FVD | 1391.30 |
| FID | 89.28 |
| AFE | 7.84 |
| Flow-warp | 0.0159 |

This baseline gives a measurable floor for video realism and motion consistency before any driving-specific adaptation.

---

# What We Learned So Far

Several LTX-based action-conditioning variants did not produce reliable driving rollouts.

- LTX + AdaLN-Action + LoRA produced visually broken rollouts.
- Lower-rank LoRA and Vista-style losses partially changed behavior but did not solve collapse.
- Decoupled cross-attention failed in a similar way.
- The failure pattern suggests the bottleneck is upstream of the conditioning mechanism.

Interpretation: an action interface cannot compensate for a weak driving-specific video prior.

---

# Model Backbones

We use LTX-Video as the initial backbone because it supports image/video conditioning and efficient 2B-scale inference.

Current model roles:

- LTX-Video 2B: primary zero-shot and adapter baseline.
- LTX-Video 13B: larger-backbone comparison for longer-context rollouts.
- Cosmos-Predict2.5-2B or Vista: pivot candidates with stronger driving/action pretraining.

The May 22 plan is to compare whether a driving-aligned base model improves action-conditioned future generation under matched adapter recipes.

---

# Action Representation

The action encoder maps future ego motion into a compact conditioning vector.

Input: future XY trajectory from WOD-E2E.

Derived features per timestep:

- x position and y position
- velocity x and velocity y
- speed
- heading
- curvature

Encoder: 4-layer causal Transformer, 256-dimensional output, producing both pooled and sequence action embeddings.

---

# Conditioning Mechanisms

We evaluate action injection mechanisms under matched compute and parameter budgets.

Primary mechanisms:

- AdaLN-Action: widen the diffusion transformer AdaLN modulator with a 256-D action embedding.
- Token prepend: add action tokens to the conditioning stream.
- Decoupled cross-attention or AVID-style adapter: separate action path when the backbone supports it.

AdaLN-Action is initialized so the new action slice is zero at step 0, preserving pretrained behavior before action gradients flow.

---

# Proposed Method

Condition on observed driving context plus an explicit future ego command, then generate only the future continuation.

Training recipe:

- Freeze most pretrained video model weights.
- Train lightweight adapters or LoRA modules.
- Inject action embeddings through AdaLN, token conditioning, or adapter paths.
- Use WOD-E2E video-action pairs for supervised diffusion fine-tuning.

Inference protocol:

- Provide observed context frames.
- Provide a command such as `go_forward` or `brake`.
- Generate a future horizon without regenerating the observed history.

---

# Baseline Comparison Matrix

The central experimental comparison is:

| Model | No action | Action-conditioned |
|---|---|---|
| LTX base | zero-shot / LoRA-only | LTX + action adapter |
| Driving-pretrained base | zero-shot / adapter-only | Cosmos or Vista + action adapter |

Expected pattern:

- Domain adaptation should improve visual metrics such as FVD/FID.
- Action conditioning should improve AFE and action discriminability.
- Driving-pretrained bases should reduce visual collapse and improve long-tail motion.

---

# Temporal Drift and FPS Controls

We added matched-duration FPS sweeps to isolate whether low FPS causes temporal drift.

Current branch includes five Waymo scenes at:

- 10 FPS: 33, 49, 81, and 105 input frames.
- 24 FPS: 81, 121, 193, and 257 input frames.
- 30 FPS: 105, 153, 241, and 321 input frames.

All frame counts satisfy the LTX constraint `8n + 1`, and the generated future horizon is held near 4 seconds for side-by-side comparison.

---

# Training Setup

Compute platform: Modal with H100 or A100 instances, using the smallest GPU sufficient for each run.

Planned training controls:

- Fixed train/validation split from WOD-E2E.
- Fixed frame-count protocol compatible with LTX.
- LoRA/adapters instead of full fine-tuning.
- W&B logging under the CS231N project.
- Reusable evaluation script for every checkpoint.

Loss stack:

- Diffusion denoising loss.
- Optional dynamics and structure losses from Vista-style training.
- Optional action-following auxiliary from visual motion estimates.

---

# Evaluation

We evaluate three properties: realism, temporal consistency, and action relevance.

Core metrics:

- FVD for video distribution quality.
- FID for frame-level visual quality.
- AFE for action-following error using flow or visual-odometry proxies.
- Flow-warp consistency for temporal coherence.

Action-specific metrics:

- Counterfactual divergence between different action prompts.
- Discriminability curves over action perturbation magnitude.
- Per-bucket WOD-E2E metrics for long-tail categories.

---

# Why This Should Work

The method separates three factors that are usually confounded.

- Base video prior: LTX vs driving-pretrained alternatives.
- Domain adaptation: no fine-tune vs LoRA/adapters.
- Action interface: no action vs explicit action conditioning.

If action conditioning works, generated futures should separate under `go_forward` versus `brake` while preserving the same initial scene.

If the base model is the bottleneck, driving-pretrained models should outperform LTX even with similar adapter mechanisms.

---

# Implementation Status

Completed or available in the repos:

- WOD-E2E preprocessing and Modal latent storage.
- Zero-shot LTX-2B baseline and evaluation harness.
- Action encoder implementation.
- AdaLN-Action implementation.
- Modal inference scripts for LTX 2B and 13B.
- Matched FPS and duration sweeps for 10, 24, and 30 FPS clips.
- Local side-by-side comparison videos for FPS analysis.

Main branch used for recent artifacts: `ltx-inference-scripts-for-cs231n`.

---

# Risks and Mitigations

Risk: LTX action-conditioned rollouts remain visually broken.

Mitigation: treat this as evidence for the base-pretraining hypothesis and run the Cosmos/Vista pivot gate.

Risk: aggregate metrics miss qualitative failure.

Mitigation: report per-bucket metrics, action-discriminability curves, and visual failure galleries.

Risk: compute budget is tight.

Mitigation: run tiered smoke tests first, use adapters/LoRA, and avoid unnecessary multi-GPU jobs.

---

# Near-Term Plan

1. Finish the pivot viability gate: license check, checkpoint download, and smoke-test inference for Cosmos or Vista.
2. Run zero-shot driving comparison on the same validation subset as LTX.
3. Train the smallest action-conditioned adapter that passes the smoke test.
4. Evaluate using FVD, FID, AFE, flow-warp, and action-discriminability.
5. Assemble the final result around whether base pretraining or action injection is the dominant factor.

---

# Deliverables for Final Project

Expected final outputs:

- Technical method for action-conditioned video generation on WOD-E2E.
- Baseline table comparing zero-shot, domain-adapted, and action-conditioned models.
- Long-tail and action-discriminability evaluation.
- Temporal/FPS drift analysis from matched 10 FPS, 24 FPS, and 30 FPS clips.
- Reproducible Modal scripts, configs, and evaluation protocol.

The final claim will be calibrated to the evidence: either a working action-conditioned world model, or a clear negative result showing why base-model pretraining dominates.

---

# Source Repos and Evidence

Primary CS231N repo:

- `CS231N-proposal-v2.md`
- `CS231N-plan.md`
- `CS231N-checklist-v2.md`
- `models/action_encoder.py`
- `models/adaln_action.py`
- `evaluate.py`

Current inference repo:

- `oneshot2B_test.py`
- `internet_highfps_oneshot2B_test.py`
- `local_30fps_waymo_oneshot2B_test.py`
- `matched_fps_duration_go_forward_sweep.py`
- Base 10 FPS, 24 FPS, and 30 FPS clips for matched-duration tests.

