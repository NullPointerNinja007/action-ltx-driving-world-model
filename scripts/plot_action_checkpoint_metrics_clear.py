from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    direction: str
    reference_line: float | None = None


METRICS = [
    Metric("future_psnr", "Future PSNR", "higher is better"),
    Metric("future_global_ssim", "Future SSIM", "higher is better"),
    Metric("fvd_future", "FVD-Style Future Distance", "lower is better"),
    Metric("future_mse", "Future MSE", "lower is better"),
    Metric("sharpness_ratio", "Generated / Real Sharpness Ratio", "near 1 is best", 1.0),
    Metric("motion_ratio", "Generated / Real Motion Ratio", "near 1 is best", 1.0),
    Metric("temporal_delta_error_mae", "Temporal Delta Error MAE", "lower is better"),
    Metric("copy_leakage_ratio", "Copy Leakage Ratio", "higher generally means less context copying"),
]


FRAME_COLUMN_MAP = {
    "future_psnr": "mean_future_psnr",
    "future_global_ssim": "mean_future_global_ssim",
    "fvd_future": "fvd_future",
    "future_mse": "mean_future_mse",
    "sharpness_ratio": "mean_sharpness_ratio_generated_over_reference",
    "motion_ratio": "mean_motion_ratio_generated_over_reference",
    "temporal_delta_error_mae": "mean_temporal_delta_error_mae",
    "copy_leakage_ratio": "mean_copy_leakage_ratio_min_context_over_future",
}

LEGACY_COLUMN_MAP = {
    "future_psnr": "future_psnr",
    "future_global_ssim": "future_global_ssim",
    "fvd_future": "fvd_future",
    "future_mse": "future_mse",
    "sharpness_ratio": "sharpness_ratio",
    "motion_ratio": "motion_ratio",
    "temporal_delta_error_mae": "temporal_delta_error_mae",
    "copy_leakage_ratio": "copy_leakage_ratio",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    if value in {"", None}:
        return float("nan")
    return float(value)


def safe_name(name: str) -> str:
    return name.replace(" ", "-").replace("/", "over").replace("_", "-").lower()


def normalize_rows(rows: list[dict[str, str]], schema: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if schema == "frame":
        column_map = FRAME_COLUMN_MAP
        for row in rows:
            out: dict[str, Any] = {
                "method_key": row["method_key"],
                "method_label": row["method_label"],
                "checkpoint_step": int(row["checkpoint_step"]),
                "model_mode": row["model_mode"],
            }
            for metric_key, source_key in column_map.items():
                out[metric_key] = as_float(row.get(source_key, ""))
            normalized.append(out)
        return sorted(normalized, key=lambda item: (item["method_key"], item["checkpoint_step"]))

    if schema == "legacy":
        column_map = LEGACY_COLUMN_MAP
        label_to_key = {
            "Global MLP": "global_mlp",
            "Temporal Per-Point": "temporal_per_point",
            "Transformer Action": "tiny_transformer",
            "AdaLN Action": "adaln",
        }
        for row in rows:
            label = row["encoder"]
            out = {
                "method_key": label_to_key.get(label, safe_name(label)),
                "method_label": label,
                "checkpoint_step": int(row["step"]),
                "model_mode": row["model_mode"],
            }
            for metric_key, source_key in column_map.items():
                out[metric_key] = as_float(row.get(source_key, ""))
            normalized.append(out)
        return sorted(normalized, key=lambda item: (item["method_key"], item["checkpoint_step"]))

    raise ValueError(f"Unknown schema: {schema}")


def method_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "frame_transformer",
        "frame_temporal_pool",
        "frame_global_mlp",
        "frame_adaln",
        "tiny_transformer",
        "temporal_per_point",
        "global_mlp",
        "adaln",
    ]
    present = {row["method_key"] for row in rows}
    ordered = [key for key in preferred if key in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def method_label(rows: list[dict[str, Any]], method_key: str) -> str:
    for row in rows:
        if row["method_key"] == method_key:
            return str(row["method_label"])
    return method_key


def plot_metric(rows: list[dict[str, Any]], metric: Metric, output_dir: Path, title_prefix: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for key in method_order(rows):
        selected = [row for row in rows if row["method_key"] == key]
        selected.sort(key=lambda row: row["checkpoint_step"])
        x = [row["checkpoint_step"] for row in selected]
        y = [float(row[metric.key]) for row in selected]
        if not x or all(math.isnan(value) for value in y):
            continue
        ax.plot(x, y, marker="o", linewidth=2, label=method_label(rows, key))

    if metric.reference_line is not None:
        ax.axhline(metric.reference_line, color="black", linestyle=":", linewidth=1.4, alpha=0.7)

    ax.set_title(f"{title_prefix}: {metric.title} ({metric.direction})")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(metric.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{safe_name(metric.key)}_over_checkpoints.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_grid(rows: list[dict[str, Any]], output_dir: Path, title_prefix: str) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    for ax, metric in zip(axes.flatten(), METRICS):
        for key in method_order(rows):
            selected = [row for row in rows if row["method_key"] == key]
            selected.sort(key=lambda row: row["checkpoint_step"])
            x = [row["checkpoint_step"] for row in selected]
            y = [float(row[metric.key]) for row in selected]
            if not x or all(math.isnan(value) for value in y):
                continue
            ax.plot(x, y, marker="o", linewidth=2, label=method_label(rows, key))
        if metric.reference_line is not None:
            ax.axhline(metric.reference_line, color="black", linestyle=":", linewidth=1.1, alpha=0.65)
        ax.set_title(f"{metric.title}\n{metric.direction}")
        ax.set_xlabel("Checkpoint step")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title_prefix, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = output_dir / "all_metrics_over_checkpoints.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def best_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for metric in METRICS:
        valid = [row for row in rows if not math.isnan(float(row[metric.key]))]
        if not valid:
            continue
        lower = "lower" in metric.direction
        if "near 1" in metric.direction:
            selected = min(valid, key=lambda row: abs(float(row[metric.key]) - 1.0))
        elif lower:
            selected = min(valid, key=lambda row: float(row[metric.key]))
        else:
            selected = max(valid, key=lambda row: float(row[metric.key]))
        best[metric.key] = {
            "metric": metric.title,
            "direction": metric.direction,
            "method_key": selected["method_key"],
            "method_label": selected["method_label"],
            "checkpoint_step": selected["checkpoint_step"],
            "model_mode": selected["model_mode"],
            "value": float(selected[metric.key]),
        }
    return best


def write_readme(
    output_dir: Path,
    *,
    title: str,
    source_csv: Path,
    normalized_csv: Path,
    plot_paths: list[Path],
    best: dict[str, Any],
) -> Path:
    lines = [
        f"# {title}",
        "",
        f"Source CSV: `{source_csv}`",
        f"Normalized CSV: `{normalized_csv}`",
        "",
        "These plots use the same five local Waymo validation clips and seed 231.",
        "PSNR/SSIM compare against the actual future; FVD-style distance, sharpness, motion, and temporal error are better for visual-quality trends.",
        "",
        "## Best Checkpoints By Metric",
        "",
    ]
    for key, row in best.items():
        lines.append(
            f"- `{key}`: {row['method_label']} step {row['checkpoint_step']} = {row['value']:.6g} ({row['direction']})"
        )
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        lines.append(f"- [{path.name}]({path.name})")
    readme = output_dir / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", choices=("frame", "legacy"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    rows = normalize_rows(read_csv(args.input), args.schema)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = args.output_dir / "normalized_checkpoint_metrics.csv"
    write_csv(normalized_csv, rows)

    plot_paths = [plot_grid(rows, args.output_dir, args.title)]
    plot_paths.extend(plot_metric(rows, metric, args.output_dir, args.title) for metric in METRICS)

    best = best_rows(rows)
    best_json = args.output_dir / "best_checkpoints_by_metric.json"
    best_json.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = write_readme(
        args.output_dir,
        title=args.title,
        source_csv=args.input,
        normalized_csv=normalized_csv,
        plot_paths=plot_paths,
        best=best,
    )
    print(
        json.dumps(
            {
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
