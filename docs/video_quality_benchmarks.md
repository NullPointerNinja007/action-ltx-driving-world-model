# Video Quality Benchmarks

The current benchmark harness evaluates generated continuation videos against the real future frames from the same Waymo clip.

Default command:

```bash
./.venv/bin/python benchmark_video_quality.py
```

Default input:

```text
data/distilled098_24fps_49ctx_72future_base_vs_lora_seed231_all5/manifest_24fps_49ctx_72future_base_vs_lora_seed231_all5.json
```

Default output:

```text
data/benchmarks/distilled098_24fps_49ctx_72future_seed231_all5/
```

Metrics are computed on:

```text
context frames: 0-48
future frames: 49-120
```

The main quality metrics use only the generated future region. The context-region metrics are diagnostic checks for whether conditioning was preserved.

Core metrics:

- `future_psnr`, `future_mse`, `future_mae`: low-level similarity to the real future. PSNR can reward blurry averages, so it is not sufficient by itself.
- `future_global_ssim`: frame-averaged global grayscale SSIM against the real future.
- `sharpness_ratio_generated_over_reference`: generated future Laplacian sharpness divided by real future sharpness. Values far below `1.0` indicate blur.
- `motion_ratio_generated_over_reference`: generated frame-to-frame motion magnitude divided by real future motion magnitude. Values far below `1.0` indicate frozen or over-smoothed motion.
- `temporal_delta_error_mae`: frame-to-frame motion error against the real future. Lower is better.
- `copy_leakage_ratio_min_context_over_future`: detects whether generated future frames look closer to some context frame than to the real future.
- `fvd_future`: optional Frechet Video Distance over generated-vs-real future video features. Lower is better.

FVD command:

```bash
./.venv/bin/python benchmark_video_quality.py \
  --manifest path/to/manifest.json \
  --source-dir data/inference_input_clips/interpolated_24fps_waymo_full20s \
  --output-dir data/benchmarks/my_run_with_fvd \
  --fps 24 \
  --context-frames 49 \
  --future-frames 72 \
  --total-frames 121 \
  --compute-fvd
```

By default, `--compute-fvd` uses `torchvision` `r3d_18` Kinetics features. This is useful for internal ranking across our checkpoints, but it is not directly comparable to canonical FVD numbers from papers.

Canonical I3D-style FVD:

```bash
./.venv/bin/python benchmark_video_quality.py \
  --manifest path/to/manifest.json \
  --source-dir data/inference_input_clips/interpolated_24fps_waymo_full20s \
  --output-dir data/benchmarks/my_run_with_i3d_fvd \
  --fps 24 \
  --context-frames 49 \
  --future-frames 72 \
  --total-frames 121 \
  --compute-fvd \
  --fvd-backend torchscript \
  --fvd-torchscript-path path/to/i3d_feature_extractor.pt
```

The TorchScript extractor should accept RGB video tensors shaped `[B, C, T, H, W]` after uniform temporal sampling and spatial resize, and return one feature vector per video. If using an official I3D checkpoint with different preprocessing, wrap it in TorchScript so the wrapper handles the expected normalization/layout.

FVD outputs:

- `model_summary.csv`: includes FVD columns when `--compute-fvd` is enabled.
- `fvd_summary.csv`: FVD-only summary by model/checkpoint.
- `fvd_features.npz`: generated/reference feature arrays for auditing and recomputing distances.

Important: FVD is unstable with very small clip counts. The 5-clip diagnostics are useful for relative checkpoint triage, but final reporting should use a much larger validation subset.

Quality gate:

The report includes a base-vs-LoRA gate. LoRA fails if it improves PSNR but loses too much sharpness or motion. This is intentional because blurry video can score deceptively well on pixel metrics.
