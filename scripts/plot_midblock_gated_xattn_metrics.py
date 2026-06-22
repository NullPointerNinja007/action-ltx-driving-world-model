from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_MIDBLOCK = Path(
    "data/benchmarks/midblock_gated_xattn_checkpoint_sweep_seed231_all5/"
    "midblock_gated_xattn_summary_with_fvd.csv"
)
DEFAULT_BASELINE = Path(
    "data/benchmarks/noaction_shifted_timestep_longer_seed231_all5/"
    "noaction_shifted_timestep_longer_summary_with_fvd.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/benchmarks/midblock_gated_xattn_checkpoint_sweep_seed231_all5/metric_plots"
)


@dataclass(frozen=True)
class Metric:
    key: str
    column: str
    title: str
    direction: str
    reference_line: float | None = None


METRICS = [
    Metric("future_psnr", "mean_future_psnr", "Future PSNR", "higher is better"),
    Metric("future_ssim", "mean_future_global_ssim", "Future SSIM", "higher is better"),
    Metric("fvd_future", "fvd_future", "FVD-Style Future Distance", "lower is better"),
    Metric("future_mse", "mean_future_mse", "Future MSE", "lower is better"),
    Metric("future_mae", "mean_future_mae", "Future MAE", "lower is better"),
    Metric(
        "sharpness_ratio",
        "mean_sharpness_ratio_generated_over_reference",
        "Generated / Real Sharpness Ratio",
        "near 1 is best",
        1.0,
    ),
    Metric(
        "fft_high_frequency_ratio",
        "mean_fft_high_frequency_energy_ratio_generated_over_reference",
        "FFT High-Frequency Energy Ratio",
        "near 1 is best",
        1.0,
    ),
    Metric(
        "motion_ratio",
        "mean_motion_ratio_generated_over_reference",
        "Generated / Real Motion Ratio",
        "near 1 is best",
        1.0,
    ),
    Metric(
        "low_frequency_motion_ratio",
        "mean_low_frequency_motion_ratio_generated_over_reference",
        "Low-Frequency Motion Ratio",
        "near 1 is best",
        1.0,
    ),
    Metric(
        "temporal_delta_error",
        "mean_temporal_delta_error_mae",
        "Temporal Delta Error MAE",
        "lower is better",
    ),
    Metric(
        "low_frequency_temporal_delta_error",
        "mean_low_frequency_temporal_delta_error_mae",
        "Low-Frequency Temporal Delta Error MAE",
        "lower is better",
    ),
    Metric(
        "boundary_ratio",
        "mean_boundary_mae_ratio_generated_over_reference",
        "Context/Future Boundary MAE Ratio",
        "near 1 is best",
        1.0,
    ),
    Metric(
        "copy_leakage",
        "mean_copy_leakage_ratio_min_context_over_future",
        "Copy Leakage Ratio",
        "higher generally means less context copying",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    return float(value)


def safe_name(value: str) -> str:
    return value.replace("_", "-").replace("/", "over").lower()


def normalize_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
    method_filter: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if method_filter and row.get("method_key") != method_filter:
            continue
        step_value = row.get("checkpoint_step", row.get("step", ""))
        if step_value == "":
            continue
        out: dict[str, Any] = {
            "method_label": label,
            "checkpoint_step": int(step_value),
            "model_mode": row.get("model_mode", ""),
        }
        for metric in METRICS:
            out[metric.key] = parse_float(row.get(metric.column, ""))
        normalized.append(out)
    return sorted(normalized, key=lambda item: item["checkpoint_step"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = ["method_label", "checkpoint_step", "model_mode"] + [metric.key for metric in METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows_by_method: dict[str, list[dict[str, Any]]],
    metric: Metric,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for label, rows in rows_by_method.items():
        x = [row["checkpoint_step"] for row in rows]
        y = [float(row[metric.key]) for row in rows]
        if not x or all(math.isnan(value) for value in y):
            continue
        ax.plot(x, y, marker="o", linewidth=2.1, label=label)

    if metric.reference_line is not None:
        ax.axhline(metric.reference_line, color="black", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_title(f"{metric.title} Over Checkpoints ({metric.direction})")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(metric.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{safe_name(metric.key)}_over_checkpoints.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_grid(rows_by_method: dict[str, list[dict[str, Any]]], output_dir: Path) -> Path:
    grid_metrics = METRICS[:12]
    fig, axes = plt.subplots(4, 3, figsize=(21, 22))
    for ax, metric in zip(axes.flatten(), grid_metrics):
        for label, rows in rows_by_method.items():
            x = [row["checkpoint_step"] for row in rows]
            y = [float(row[metric.key]) for row in rows]
            if not x or all(math.isnan(value) for value in y):
                continue
            ax.plot(x, y, marker="o", linewidth=2.0, label=label)
        if metric.reference_line is not None:
            ax.axhline(metric.reference_line, color="black", linestyle=":", linewidth=1.0, alpha=0.65)
        ax.set_title(f"{metric.title}\n{metric.direction}")
        ax.set_xlabel("Checkpoint step")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Middle-Block Gated Action Cross-Attention Checkpoint Sweep", y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    path = output_dir / "all_metrics_over_checkpoints.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def metric_score(row: dict[str, Any], metric: Metric) -> float:
    value = float(row[metric.key])
    if "near 1" in metric.direction:
        return -abs(value - 1.0)
    if "lower" in metric.direction:
        return -value
    return value


def best_midblock_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric in METRICS:
        valid = [row for row in rows if not math.isnan(float(row[metric.key]))]
        if not valid:
            continue
        selected = max(valid, key=lambda row: metric_score(row, metric))
        best[metric.key] = {
            "metric": metric.title,
            "direction": metric.direction,
            "checkpoint_step": selected["checkpoint_step"],
            "model_mode": selected["model_mode"],
            "value": float(selected[metric.key]),
        }
    return best


def write_readme(
    output_dir: Path,
    *,
    plot_paths: list[Path],
    normalized_csv: Path,
    best_json: Path,
    midblock_input: Path,
    baseline_input: Path,
) -> Path:
    lines = [
        "# Middle-Block Gated XAttn Metric Plots",
        "",
        f"Midblock source CSV: `{midblock_input}`",
        f"Corrected no-action shifted baseline CSV: `{baseline_input}`",
        f"Normalized CSV: `{normalized_csv}`",
        f"Best-checkpoint JSON: `{best_json}`",
        "",
        "The corrected no-action shifted curve is included where the same metric exists.",
        "FFT high-frequency and low-frequency motion diagnostics are available only for the newer midblock benchmark run.",
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- [{path.name}]({path.name})")
    readme = output_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midblock", type=Path, default=DEFAULT_MIDBLOCK)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    midblock_rows = normalize_rows(
        read_csv(args.midblock),
        label="Middle-block gated action xattn",
        method_filter="frame_midblock_gated_xattn",
    )
    baseline_rows = normalize_rows(
        read_csv(args.baseline),
        label="Corrected no-action shifted LoRA",
        method_filter="shifted_lognormal",
    )
    rows_by_method = {
        "Middle-block gated action xattn": midblock_rows,
        "Corrected no-action shifted LoRA": baseline_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = args.output_dir / "normalized_midblock_vs_noaction_metrics.csv"
    write_csv(normalized_csv, midblock_rows + baseline_rows)
    best = best_midblock_rows(midblock_rows)
    best_json = args.output_dir / "best_midblock_checkpoints_by_metric.json"
    best_json.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_paths = [plot_grid(rows_by_method, args.output_dir)]
    plot_paths.extend(plot_metric(rows_by_method, metric, args.output_dir) for metric in METRICS)
    readme = write_readme(
        args.output_dir,
        plot_paths=plot_paths,
        normalized_csv=normalized_csv,
        best_json=best_json,
        midblock_input=args.midblock,
        baseline_input=args.baseline,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "normalized_csv": str(normalized_csv),
                "best_json": str(best_json),
                "readme": str(readme),
                "plots": [str(path) for path in plot_paths],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
