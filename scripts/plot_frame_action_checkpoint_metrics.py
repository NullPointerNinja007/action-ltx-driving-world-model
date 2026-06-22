from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(
    "data/benchmarks/frame_action_checkpoint_sweep_seed231_all5/"
    "frame_action_checkpoint_sweep_summary_with_fvd.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/benchmarks/frame_action_checkpoint_sweep_seed231_all5/metric_plots")
DEFAULT_BASELINE = Path(
    "data/benchmarks/noaction_2epoch_checkpoint_sweep_seed231_all5/"
    "noaction_2epoch_summary_with_fvd.csv"
)

METRICS = [
    ("mean_future_psnr", "Future PSNR", "higher is better"),
    ("mean_future_global_ssim", "Future SSIM", "higher is better"),
    ("fvd_future", "FVD-Style Future Distance", "lower is better"),
    ("mean_future_mse", "Future MSE", "lower is better"),
    ("mean_future_mae", "Future MAE", "lower is better"),
    ("mean_context_psnr", "Context PSNR", "higher is better"),
    ("mean_context_mse", "Context MSE", "lower is better"),
    ("mean_generated_future_laplacian_sharpness", "Generated Future Sharpness", "higher means sharper"),
    ("mean_sharpness_ratio_generated_over_reference", "Sharpness Ratio vs Reference", "near 1 is best"),
    ("mean_motion_ratio_generated_over_reference", "Motion Ratio vs Reference", "near 1 is best"),
    ("mean_temporal_delta_error_mae", "Temporal Delta Error MAE", "lower is better"),
    ("mean_boundary_mae_ratio_generated_over_reference", "Context/Future Boundary MAE Ratio", "near 1 is best"),
    ("mean_copy_leakage_ratio_min_context_over_future", "Copy Leakage Ratio", "higher generally means less context copying"),
]

METHOD_ORDER = [
    "frame_transformer",
    "frame_temporal_pool",
    "frame_global_mlp",
    "frame_adaln",
]

METHOD_LABELS = {
    "frame_transformer": "Frame Transformer",
    "frame_temporal_pool": "Frame Temporal Pool",
    "frame_global_mlp": "Frame Global MLP",
    "frame_adaln": "Frame AdaLN",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def safe_name(metric: str) -> str:
    return metric.replace("mean_", "").replace("_generated_over_reference", "").replace("_", "-")


def load_baseline(path: Path, model_mode: str) -> dict[str, float]:
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("model_mode") == model_mode:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value in {"", None}:
                    continue
                try:
                    parsed[key] = as_float(value)
                except ValueError:
                    continue
            return parsed
    return {}


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    direction: str,
    output_dir: Path,
    baseline: dict[str, float],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_key in METHOD_ORDER:
        method_rows = [row for row in rows if row["method_key"] == method_key]
        method_rows.sort(key=lambda row: int(row["checkpoint_step"]))
        x = [int(row["checkpoint_step"]) for row in method_rows]
        y = [as_float(row.get(metric, "")) for row in method_rows]
        if not x or all(math.isnan(value) for value in y):
            continue
        ax.plot(x, y, marker="o", linewidth=2, label=METHOD_LABELS.get(method_key, method_key))

    baseline_value = baseline.get(metric)
    if baseline_value is not None and not math.isnan(baseline_value):
        ax.axhline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.65,
            label="No-action LoRA step 10000",
        )

    if metric in {
        "mean_sharpness_ratio_generated_over_reference",
        "mean_motion_ratio_generated_over_reference",
        "mean_boundary_mae_ratio_generated_over_reference",
    }:
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="Reference ratio 1.0")

    ax.set_title(f"{title} Over Checkpoints ({direction})")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"{safe_name(metric)}_over_checkpoints.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_index(paths: list[Path], output_dir: Path) -> Path:
    index_path = output_dir / "README.md"
    lines = ["# Frame-Action Checkpoint Metric Plots", ""]
    for path in paths:
        lines.append(f"- [{path.name}]({path.name})")
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-model-mode", default="noaction_visual_lora_step010000")
    args = parser.parse_args()

    rows = read_csv(args.input)
    baseline = load_baseline(args.baseline, args.baseline_model_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_metric(
            rows,
            metric=metric,
            title=title,
            direction=direction,
            output_dir=args.output_dir,
            baseline=baseline,
        )
        for metric, title, direction in METRICS
    ]
    index_path = write_index(paths, args.output_dir)
    print(f"Wrote {len(paths)} metric plots to {args.output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
