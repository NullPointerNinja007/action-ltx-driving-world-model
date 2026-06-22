# Action-Conditioning Failure Analysis Against Recent Video-Diffusion Papers

Date: 2026-06-03

This note compares our Waymo/LTX action-conditioning experiments against four attached papers:

- `Improving Video Diffusion Transformer Training by Multi-Feature Fusion and Alignment from Self-Supervised Vision Encoders`
- `Enhance-A-Video: Better Generated Video for Free`
- `Conditional Video Generation for High-Efficiency Video Compression`
- `FullDiT2: Efficient In-Context Conditioning for Video Diffusion Transformers`

The goal is not to force their claims onto our project. Some of these papers solve adjacent problems, not exactly ego-action-conditioned driving world modeling. The goal is to identify what they did structurally differently, why those choices likely helped, and which of our failure explanations remain plausible.

## Hard Context For Interpreting Our Results

Do not frame raw sharpness drop against zero-shot LTX as the primary failure. Zero-shot LTX can generate sharper and more visually polished video than the lower-sharpness 512px Waymo data. Fine-tuning on Waymo can legitimately move the model toward the dataset's visual statistics.

The meaningful comparisons are:

- action-conditioned model vs corrected no-action LoRA;
- action-conditioned model vs the same checkpoint with action gate scale 0;
- correct actions vs zero/shuffled/reversed actions;
- temporal consistency/wobble/motion metrics, not sharpness alone.

The real problem is not simply "sharpness drops." The real problem is that action pathways often add extra wobble, low-motion smoothing, temporal weirdness, or weak counterfactual action sensitivity relative to matched no-action/gate-zero baselines.

## What The Papers Actually Do

### 1. Align4Gen / Multi-Feature Fusion And Alignment

Main mechanism:

- Adds an auxiliary training loss that aligns intermediate V-DiT patch tokens with frozen self-supervised vision encoder features.
- Uses multiple encoders with complementary frequency behavior:
  - DINOv2-like features for low-frequency semantics;
  - SAM2.1/Hiera-like features for higher-frequency details.
- Aligns diffusion features through a lightweight mapper and cosine/projection loss.
- The final training objective is the base diffusion/flow loss plus a weighted feature-alignment term.
- Alignment is a training-time regularizer only; the external feature mapper is removed at inference.

Relevant design facts:

- They train V-DiT-L for 1M steps with batch size 24 and V-DiT-XL for 200K steps with batch size 16.
- They do not claim that plain diffusion MSE is enough. Their thesis is that representation supervision improves convergence and quality.
- Their ablations show that which feature representation is used matters. Temporally unstable encoders can hurt.
- They explicitly evaluate temporal consistency and note that classical FVD can over-reward per-frame quality when motion is poor.

Why this matters for us:

- Our action training mostly used diffusion MSE, with only limited teacher/high-frequency penalties in later bottleneck runs.
- We did not align LTX internal features to an external temporally stable representation.
- We did not use dense tracking/flow/feature consistency as an auxiliary target.
- We expected a small numeric action module to discover both action semantics and temporal rendering behavior through future diffusion loss alone. Align4Gen is evidence that auxiliary representation supervision can materially improve video diffusion training efficiency and quality.

Adversarial implication:

- It is not enough to say our architecture is wrong. We may simply be under-supervising the feature space that should carry temporal/action information.
- Our V3/low-gate bottleneck may preserve visuals because it barely perturbs the feature space, but it may need a feature/motion alignment term to learn a useful action manifold.

### 2. Enhance-A-Video

Main mechanism:

- Training-free inference-time method.
- Measures non-diagonal temporal attention as cross-frame intensity.
- Scales temporal attention outputs modestly through a residual branch:
  - strengthens cross-frame correlations;
  - avoids directly changing the full attention logits too aggressively;
  - clips the enhancement to avoid instability.
- Applied to multiple DiT video models, including LTX-Video.

Relevant design facts:

- They explicitly show that naive temperature scaling of temporal attention can cause blur, loss of visual detail, and unstable generation.
- Their method modifies the attention output residual, not the entire hidden state distribution.
- Their empirical CFI values are modest; the method is designed to be a small correction, not a strong new condition pathway.
- They treat temporal consistency and spatial detail as coupled: bad cross-frame attention can damage spatial appearance.

Why this matters for us:

- Our bad AdaLN/global-token/midblock results are consistent with their warning: broad or excessive intervention in temporal/attention pathways can produce blur and instability.
- Some of our action gates/residuals directly modify hidden states or normalization conditions. This is much more invasive than modest attention-output scaling.
- Their result suggests that temporal coherence may be improved by carefully rebalancing cross-frame attention, not by injecting large action vectors into every token or every block.

Adversarial implication:

- Our "action conditioning" may be corrupting the same temporal attention structure that already needs careful balance.
- We should inspect whether action conditioning changes temporal attention diagonal vs off-diagonal patterns. If action gates reduce useful cross-frame attention or over-amplify irrelevant cross-frame mixing, wobble is expected.
- It may be useful to run Enhance-A-Video-style inference on our no-action and action checkpoints to test whether some wobble is attention-balance related and fixable without retraining.

### 3. Conditional Video Generation For High-Efficiency Video Compression

Main mechanism:

- Reframes compression as conditional video generation.
- Uses multi-granular conditions:
  - first frame and last frame;
  - text;
  - segmentation;
  - human motion;
  - optical flow.
- Converts dense conditions into visual modalities, encodes them with a 3D causal VAE, and concatenates condition latents with noise in the diffusion backbone.
- Uses condition dropout with dropout ratio 0.3 and role-aware embeddings to prevent over-reliance on one condition and to disambiguate real zeros from dropped conditions.

Relevant design facts:

- They fine-tune a pretrained VAST-10B FL2V model for 1 epoch with Adam, learning rate 2e-5, batch size 8.
- Their training data scale is very large: hundreds of thousands of videos across multiple categories.
- Ablation shows the full condition set is best.
- Removing human motion causes the biggest drop in low-bitrate settings; optical flow and segmentation also matter.

Why this matters for us:

- Their "control" signals are dense and visually grounded. Optical flow and segmentation are close to the actual target video formation process.
- Our ego action vector is sparse and indirect. It does not specify:
  - road layout;
  - traffic light state;
  - nearby agent behavior;
  - route intent;
  - object motion;
  - camera/ego SE(3) trajectory in image space;
  - dense optical flow induced by ego motion.
- They condition a model built for first/last-frame video reconstruction. We condition LTX with context frames but do not give a future endpoint or dense future visual scaffold.
- They train full/foundation-level components at a much larger data/model scale.

Adversarial implication:

- Ego actions alone may be too low-information for the visual future unless represented geometrically.
- If the future is underdetermined, the model can minimize diffusion loss by using context statistics and ignoring actions.
- A successful driving version probably needs to transform ego actions into a stronger visual/geometric condition: ego trajectory, camera motion, projected flow field, homography/SE(3) warp, or low-resolution future-motion map.

### 4. FullDiT2 / In-Context Conditioning For Video DiTs

Main mechanism:

- Starts from in-context conditioning: concatenate condition tokens and noisy latent tokens into one sequence and process them through the native DiT self-attention.
- FullDiT2 is mostly an efficiency improvement over this strong ICC baseline:
  - dynamic token selection chooses important condition tokens;
  - selective context caching avoids recomputing stable reference tokens;
  - block importance index chooses which layers should process reference tokens.

Relevant design facts:

- They initialize from a pretrained FullDiT checkpoint and fine-tune all model parameters.
- Training uses AdamW, learning rate 1e-5, 400,000 iterations, distributed across 32 80GB GPUs.
- Input videos are 77 frames at 672x384, compressed by a 3D VAE into 20 temporal latent frames.
- Conditional tasks use dense reference tokens:
  - 20 noisy video latent frames;
  - 20 reference video/pose/camera trajectory frames;
  - sometimes additional ID image latents.
- They process reference information in Layer 0 plus the four blocks with highest precomputed block importance.
- They explicitly warn about training-inference mismatch in caching and use decoupled attention to avoid it.

Why this matters for us:

- Their success is not from a tiny adapter attached to a frozen model. It relies on a base model trained for in-context conditioning and then full-parameter fine-tuning.
- Our middle-block-only intuition is only partially aligned. FullDiT2 finds that condition-token impact is layer-dependent and data/task-specific. They include Layer 0 because early projection can be critical.
- We selected blocks heuristically, not by a block-importance analysis.
- Our action tokens are not dense reference latents. They are low-dimensional numeric controls, so they may not be naturally compatible with DiT self-attention without stronger tokenization/geometric grounding.

Adversarial implication:

- Our model may fail not because in-context conditioning is bad, but because LTX was not trained for our kind of in-context numeric action control and we only trained small frozen-base modules.
- The best layer for action may not be "middle blocks." It should be measured with a block importance index or gradient/activation sensitivity, not assumed.
- We may need either full/partial DiT fine-tuning or a stronger adapter trained long enough to let the DiT learn action-token semantics.

## What We Did Differently

| Axis | Successful papers | Our current project |
|---|---|---|
| Condition signal | Dense visual/geometric/reference conditions, or no new condition but temporal-attention repair | Sparse numeric ego actions, 18 dims per frame |
| Conditioning representation | VAE latents, visual modalities, full in-context tokens, temporal attention output scaling | MLP tokens, temporal pool/transformer tokens, AdaLN deltas, gated residuals |
| Training scale | 200K-1M steps or 400K full-parameter iterations; large batches/GPU counts | Mostly 3K steps; some 2-epoch 15,984-step runs; LoRA/small modules |
| Trainable parameters | Often full model or at least foundation-level training | Mostly frozen base/VAE/T5 with LoRA/action modules |
| Losses | Diffusion plus feature alignment, perceptual metrics, role/dropout mechanisms, dense condition supervision | Primarily diffusion MSE; later HF teacher/lowfreq terms but limited |
| Temporal mechanism | Explicit cross-frame attention manipulation, flow/motion conditions, temporal consistency evaluation | Action modules rely on future diffusion loss to discover motion semantics |
| Layer selection | Measured or architecture-native: full attention, BI-selected layers, careful residual scaling | Heuristic injection into text tokens/middle blocks/fixed layers |
| Evaluation scale | Thousands of generated clips or task-specific metrics/user studies | Mostly 5 fixed clips for iteration; full validation not run for every method |

## Why They Likely Succeeded And We Did Not

### Hypothesis 1: Our action signal is under-informative.

Likelihood: high.

The papers use conditions that are close to the visual target: optical flow, segmentation, pose, reference video, camera trajectory, keyframes. Our action vector is a compact ego signal. It does not encode all scene dynamics. In driving, a future frame depends on ego motion, other agents, road geometry, traffic controls, route, and stochastic behavior. Ego action alone is not a dense visual scaffold.

Evidence in our runs:

- Correct/zero/shuffled action sensitivity is often weak.
- Gate-disabled or low-gate outputs are frequently visually better.
- Higher gate strength changes outputs but often through smoothing/wobble, not precise control.

Test:

- Compute correlation between action channels and low-frequency optical flow/camera-motion estimates.
- Predict future low-resolution flow or frame delta from context-only vs context+actions. If action adds little predictive value, direct action conditioning will remain weak.
- Audit action timestamp alignment against image-derived ego motion.

### Hypothesis 2: Our injection pathway corrupts rendering.

Likelihood: high for AdaLN/global/raw midblock; moderate for bottlenecks.

Enhance-A-Video shows that naive temporal attention scaling can cause blur and detail loss. Our AdaLN and global token methods are more invasive than their modest attention-output correction. The gate sweeps show that increasing action strength often decreases sharpness, FFT-HF, and motion.

Evidence in our runs:

- Frame AdaLN 3000: step 0 sharpness 0.277, motion 0.804; step 3000 sharpness 0.123, motion 0.527.
- Midblock raw: step 0 sharpness 0.270, motion 0.792; step 3000 sharpness 0.113, motion 0.459.
- V3 gate 1.0 improves some FVD/PSNR but reduces sharpness/motion relative to gate 0.0.

Test:

- Record temporal attention diagonal/off-diagonal statistics under gate 0 vs gate 1.
- Apply Enhance-A-Video-style inference to action checkpoints and see if wobble improves.
- Compare hidden-state norm/residual norm of action path vs base residual/attention norm.

### Hypothesis 3: We are undertraining constrained action pathways.

Likelihood: high for low-gate bottlenecks; low for broad/global methods.

The papers train for far longer and often with full parameters. A 3K-step run is not enough to conclude that a low-gate action path cannot learn. If a method is deliberately constrained to avoid damage, it may need more training before action sensitivity grows.

Evidence in our runs:

- Temporal bottleneck v1 gate 0.025/0.05 improved FVD over checkpoints with small sharpness/motion loss and measurable sensitivity.
- V3/no-text barely changed; this can mean "stable but undertrained," not "failed."
- Broad methods, by contrast, already show collapse, so more training would likely worsen them.

Test:

- Extend only low-gate V3/v1/v2 variants with dense checkpoints.
- Track action sensitivity, motion ratio, low-frequency motion, and temporal delta, not only FVD/PSNR.

### Hypothesis 4: We lack explicit action/motion supervision.

Likelihood: high.

The compression paper shows that motion/flow conditions are critical. Align4Gen shows auxiliary feature alignment accelerates and stabilizes training. Our future diffusion loss does not force actions to explain motion; context-only prediction can be sufficient for minimizing average loss.

Evidence:

- Correct/zero/shuffled actions often produce similar outputs.
- FVD/PSNR improvements can happen through smoothing.
- No-text ablation did not fix action usage.

Test:

- Add low-frequency action/motion objective on blurred/downsampled future deltas.
- Add optical-flow proxy or ego-induced flow target.
- Add feature alignment/teacher loss that preserves temporal consistency.

### Hypothesis 5: Our layer selection is heuristic and may be wrong.

Likelihood: moderate to high.

FullDiT2 explicitly measures block importance and uses Layer 0 plus top BI layers. Our middle-only strategy assumes early layers should stay visual and middle layers should carry control. That is plausible, but not proven. For some DiTs, early projection layers may be where condition tokens need to enter.

Evidence:

- Middle-block gating helped relative to fully global injection, but did not solve action sensitivity.
- V3 bottleneck did better, but still trades control strength for quality.

Test:

- Compute block importance index for our action conditions:
  - run with and without action at each candidate layer;
  - measure output/hidden difference on future tokens;
  - select layers by measured effect rather than depth heuristic.
- Try a small Layer 0 + top-BI adapter, not arbitrary middle blocks.

### Hypothesis 6: We froze too much.

Likelihood: moderate.

FullDiT2 fine-tunes all parameters for 400K iterations. The compression paper fine-tunes a large pretrained video model. We mostly froze the base transformer and trained small adapters/LoRA/action modules. A frozen DiT may not know how to interpret a new numeric action manifold.

Counterpoint:

- Full unfreezing may destroy the useful LTX prior unless the loss is constrained.
- Our early full/action-ish runs blurred quickly, so unfreezing alone is not safe.

Test:

- Unfreeze only selected temporal-attention projections or selected LoRA blocks after a stable low-gate warmup.
- Use stronger regularization against corrected no-action teacher.

### Hypothesis 7: Action/frame alignment or action normalization may be wrong.

Likelihood: plausible and important to rule out.

We upsampled actions to 24 FPS and interpolate frames. If action timestamps are offset or linearly upsampled in a way that does not match the frame interpolation, the model sees contradictory supervision. A small temporal offset can look like wobble or weak action response.

Test:

- Compare action-derived yaw/speed/acceleration against image-derived optical flow over time.
- Sweep action temporal offsets, e.g. -12, -6, 0, +6, +12 frames at 24 FPS, and measure correlation with low-frequency image motion.
- Visual overlay actions/flow on validation windows and inspect unusual windows.

### Hypothesis 8: Our evaluation may be too small/noisy.

Likelihood: moderate.

Five clips are useful for iteration but too small for final claims. FVD-style on five videos is unstable and can over-reward smoothing. However, visual wobble and matched gate comparisons are still meaningful.

Test:

- Run full validation or at least 100-200 diverse windows for the promising methods only.
- Stratify by action magnitude: straight/low-motion, braking, turning, acceleration.
- Report both aggregate and per-action-bin metrics.

### Hypothesis 9: Text conditioning was overwhelming actions.

Likelihood: low based on our ablation.

The no-text/action-only ablation did not solve the issue. Text removal often hurt visual stability or produced smoothing. For V3, text-enabled results were generally better visually.

Conclusion:

- Do not remove T5/text by default.
- Text is likely stabilizing scene semantics rather than blocking action.

## Most Important Differences From The Papers

The most important gap is not one single hyperparameter. It is this combination:

1. The papers use dense, semantically/visually grounded conditioning or explicit temporal-attention repair.
2. We use sparse numeric ego actions and expect the model to infer their visual consequence.
3. The papers often train full models or very large models for hundreds of thousands of iterations.
4. We mostly train small modules for thousands of steps.
5. The papers include auxiliary losses or architectural safeguards for temporal/spatial consistency.
6. We mostly use future diffusion MSE and then diagnose blur after the fact.

This explains why our action path can either be ignored or become harmful:

- If action influence is weak, the model uses context and ignores actions.
- If action influence is strong, the model perturbs the video manifold and creates smoothing/wobble.
- Without an explicit low-frequency action objective, there is no guarantee the action pathway learns the intended visual control.

## What This Means For The Next Experiment

The next serious run should combine the strongest lessons:

### Keep

- Corrected no-action shifted LoRA as visual baseline.
- Text enabled.
- Upsampled 24 FPS frame-action data.
- Low-gate temporal bottleneck/V3-style routing.
- Dense checkpointing.

### Add

1. Low-frequency action/motion loss.
   - Operate on blurred/downsampled future deltas or optical-flow proxies.
   - The action branch should explain coarse temporal evolution, not high-frequency texture.

2. Feature/teacher alignment.
   - Preserve high-frequency visual features against corrected no-action teacher.
   - Consider DINO/SAM-style feature alignment for decoded frames or a cheaper latent proxy.

3. Action-to-geometry conversion.
   - Convert ego action/trajectory into an image-space motion field, homography, or low-resolution flow target.
   - Numeric actions should not enter as abstract MLP tokens only.

4. Block-importance layer selection.
   - Measure which LTX blocks respond usefully to action conditions.
   - Do not assume middle-only is optimal.

5. Condition dropout and role embeddings.
   - Train with correct action, zero action, and dropped action roles.
   - This prevents the model from over-relying on actions and makes counterfactual tests meaningful.

6. Temporal attention diagnostics.
   - Measure diagonal/non-diagonal attention changes with action gate scale.
   - Test Enhance-A-Video-style inference on promising checkpoints.

## Recommended Experiment Order

1. Data/action alignment audit.
   - Cheap and necessary.
   - Verify action sequences align with visual motion after 24 FPS interpolation.

2. Full-val or larger-val evaluation for existing best candidates.
   - Temporal bottleneck v1 gate 0.025/0.05.
   - Temporal bottleneck v2 gate 0.025/0.05/0.25.
   - Low-frequency V3 gate 0.025/0.05/0.1/0.25.

3. Train low-frequency V3 for longer only if alignment audit passes.
   - 2 epochs minimum.
   - Dense checkpoints.
   - Keep text enabled.

4. Implement V3 plus explicit low-frequency action/motion loss.
   - This is the most paper-aligned next architecture.

5. Add feature alignment/teacher loss.
   - Start cheap: teacher/latent high-frequency consistency.
   - If feasible, add DINO/SAM feature loss on decoded subsets.

6. Only after this, consider partial unfreezing/full DiT adaptation.
   - Use low LR.
   - Selected blocks only.
   - Strong teacher regularization.

## Bottom-Line Diagnosis

The papers succeeded because they did not ask a small numeric adapter to learn the whole action-to-video manifold from diffusion MSE alone.

They either:

- repair temporal attention at inference without retraining;
- use dense visual/geometric conditions;
- train a DiT that is already designed for in-context conditioning;
- add representation/feature/motion losses;
- train at much larger scale;
- carefully control where and how conditions affect the DiT.

Our failure is therefore not surprising. The current project has strong evidence that broad action injection corrupts temporal rendering, and constrained bottleneck action injection preserves quality but probably underlearns. The next path is to make actions geometrically/temporally explicit and supervise the action branch with a low-frequency motion objective while protecting the visual prior with teacher/feature alignment.
