#!/usr/bin/env python3
"""Plot V4 and cross-method checkpoint metric trajectories.

This script intentionally avoids pandas so it can run in the current project
venv. It reads the benchmark CSVs produced by the Modal evaluation pipeline and
creates:

1. V4-only plots: all six V4 variants over checkpoint.
2. All-method plots: major action-conditioning families tried so far.

For gate-sweep methods, the all-method comparison keeps the normal gate=1.0
operating point to avoid duplicating the same trained checkpoint many times.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "data" / "benchmarks"
OUT_DIR = BENCH_ROOT / "v4_and_all_methods_checkpoint_curves_seed231_all5"
PLOT_DIR = OUT_DIR / "metric_plots"


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    ylabel: str
    direction: str


METRICS = [
    Metric("fvd", "FVD-style future distance", "lower is better", "lower"),
    Metric("psnr", "Future PSNR", "higher is better", "higher"),
    Metric("ssim", "Future SSIM", "higher is better", "higher"),
    Metric("sharpness", "Laplacian sharpness ratio", "higher is better", "higher"),
    Metric("fft_hf", "FFT high-frequency ratio", "higher is better", "higher"),
    Metric("motion", "Temporal motion ratio", "higher is better", "higher"),
    Metric("lowfreq_motion", "Low-frequency motion ratio", "higher is better", "higher"),
    Metric("temporal_delta", "Temporal delta error MAE", "lower is better", "lower"),
]


ALIASES = {
    "psnr": ["mean_future_psnr", "future_psnr"],
    "ssim": ["mean_future_global_ssim", "future_global_ssim"],
    "sharpness": [
        "mean_sharpness_ratio_generated_over_reference",
        "sharpness_ratio",
    ],
    "fft_hf": [
        "mean_fft_high_frequency_energy_ratio_generated_over_reference",
        "fft_high_frequency_ratio",
    ],
    "motion": ["mean_motion_ratio_generated_over_reference", "motion_ratio"],
    "lowfreq_motion": [
        "mean_low_frequency_motion_ratio_generated_over_reference",
        "lowfreq_motion_ratio",
    ],
    "temporal_delta": [
        "mean_temporal_delta_error_mae",
        "temporal_delta_error_mae",
    ],
    "fvd": ["future_fvd_style", "fvd_future", "fvd_future_distance"],
}


V4_LABELS = {
    "v4_main_text": "V4 main + text",
    "v4_conservative_text": "V4 conservative + text",
    "v4_main_notext": "V4 main, no text",
    "v4_conservative_notext": "V4 conservative, no text",
    "v4_actionstrong_text": "V4 action-strong + text",
    "v4_qualitystrict_text": "V4 quality-strict + text",
}


ALL_METHOD_FILES = [
    # Pure visual/reference baselines.
    (
        "noaction_shifted",
        "No-action shifted LoRA",
        BENCH_ROOT
        / "noaction_shifted_timestep_longer_seed231_all5"
        / "noaction_shifted_timestep_longer_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "shifted_lognormal",
    ),
    # Early action-token/AdaLN experiments.
    (
        "legacy_global_mlp",
        "Legacy Global MLP tokens",
        BENCH_ROOT
        / "action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5"
        / "action_checkpoint_sweep_summary_all_encoders_with_adaln_fvd.csv",
        lambda r: r.get("encoder") == "Global MLP",
    ),
    (
        "legacy_temporal_per_point",
        "Legacy temporal per-point tokens",
        BENCH_ROOT
        / "action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5"
        / "action_checkpoint_sweep_summary_all_encoders_with_adaln_fvd.csv",
        lambda r: r.get("encoder") == "Temporal Per-Point",
    ),
    (
        "legacy_transformer",
        "Legacy transformer action tokens",
        BENCH_ROOT
        / "action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5"
        / "action_checkpoint_sweep_summary_all_encoders_with_adaln_fvd.csv",
        lambda r: r.get("encoder") == "Transformer Action",
    ),
    (
        "legacy_adaln",
        "Legacy AdaLN action",
        BENCH_ROOT
        / "action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5"
        / "action_checkpoint_sweep_summary_all_encoders_with_adaln_fvd.csv",
        lambda r: r.get("encoder") == "AdaLN Action",
    ),
    # Frame-action two-epoch reruns.
    (
        "frame_global_mlp_2epoch",
        "Frame Global MLP, 2 epoch",
        BENCH_ROOT
        / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
        / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "frame_global_mlp",
    ),
    (
        "frame_temporal_pool_2epoch",
        "Frame temporal pool, 2 epoch",
        BENCH_ROOT
        / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
        / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "frame_temporal_pool",
    ),
    (
        "frame_transformer_2epoch",
        "Frame transformer, 2 epoch",
        BENCH_ROOT
        / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
        / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "frame_transformer",
    ),
    (
        "frame_adaln_2epoch",
        "Frame AdaLN, 2 epoch",
        BENCH_ROOT
        / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
        / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "frame_adaln",
    ),
    # Later gated / bottleneck methods.
    (
        "midblock_xattn",
        "Mid-block gated XAttn",
        BENCH_ROOT
        / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
        / "midblock_gated_xattn_summary_with_fvd.csv",
        lambda r: r.get("method_key") == "frame_midblock_gated_xattn",
    ),
    (
        "hfteacher_v1",
        "Temporal bottleneck HF-teacher v1",
        BENCH_ROOT
        / "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5"
        / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv",
        lambda r: "counterfactual" not in r.get("model_mode", ""),
    ),
    (
        "hfteacher_v2",
        "Temporal bottleneck HF-teacher v2",
        BENCH_ROOT
        / "corrected_global_mlp_hfteacher_v2_seed231_all5"
        / "model_summary_with_fvd.csv",
        lambda r: r.get("model_mode", "").startswith("hfteacher_v2_step"),
    ),
    (
        "corrected_global_mlp",
        "Corrected Global MLP",
        BENCH_ROOT
        / "corrected_global_mlp_hfteacher_v2_seed231_all5"
        / "model_summary_with_fvd.csv",
        lambda r: r.get("model_mode", "").startswith("corrected_global_mlp_step"),
    ),
    (
        "lowfreq_v3",
        "Low-frequency bottleneck V3, gate=1",
        BENCH_ROOT
        / "frame_temporal_bottleneck_lowfreq_v3_gate_scale_sweep_seed231_all5"
        / "model_summary.csv",
        lambda r: "_g1p000" in r.get("model_mode", ""),
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_step(row: dict[str, str]) -> int:
    for key in ("checkpoint_step", "step"):
        if row.get(key) not in ("", None):
            return int(float(row[key]))
    text = row.get("model_mode", "") or row.get("encoder", "")
    match = re.search(r"step_?0*(\d+)", text)
    return int(match.group(1)) if match else -1


def metric_value(row: dict[str, str], metric_key: str) -> float:
    for alias in ALIASES[metric_key]:
        if alias in row:
            value = to_float(row.get(alias))
            if not math.isnan(value):
                return value
    return math.nan


def add_standard_metrics(record: dict[str, object], row: dict[str, str]) -> None:
    for metric in METRICS:
        record[metric.key] = metric_value(row, metric.key)


def load_v4_rows() -> list[dict[str, object]]:
    paths = [
        BENCH_ROOT / "final_h100_v4_campaign_seed231_all5" / "model_summary_with_fvd.csv",
        BENCH_ROOT
        / "final_h100_v4_campaign_seed231_all5"
        / "second_epoch"
        / "model_summary_with_fvd.csv",
    ]
    out: list[dict[str, object]] = []
    for path in paths:
        for row in read_csv(path):
            method = row.get("method_key", "")
            if not method.startswith("v4_"):
                continue
            record: dict[str, object] = {
                "method_key": method,
                "method_label": V4_LABELS.get(method, method),
                "checkpoint_step": to_step(row),
                "source_file": str(path.relative_to(REPO_ROOT)),
            }
            add_standard_metrics(record, row)
            out.append(record)
    return dedupe_rows(out)


def load_all_method_rows() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for method_key, label, path, keep in ALL_METHOD_FILES:
        for row in read_csv(path):
            if not keep(row):
                continue
            record: dict[str, object] = {
                "method_key": method_key,
                "method_label": label,
                "checkpoint_step": to_step(row),
                "source_file": str(path.relative_to(REPO_ROOT)),
            }
            add_standard_metrics(record, row)
            out.append(record)
    out.extend(load_v4_rows())
    return dedupe_rows(out)


def dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Prefer rows that have FVD, then rows from later files for duplicate method/step.
    keyed: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (str(row["method_key"]), int(row["checkpoint_step"]))
        prev = keyed.get(key)
        if prev is None:
            keyed[key] = row
            continue
        prev_has_fvd = not math.isnan(float(prev.get("fvd", math.nan)))
        row_has_fvd = not math.isnan(float(row.get("fvd", math.nan)))
        if row_has_fvd or not prev_has_fvd:
            keyed[key] = row
    return sorted(keyed.values(), key=lambda r: (str(r["method_key"]), int(r["checkpoint_step"])))


def write_used_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["method_key", "method_label", "checkpoint_step", *[m.key for m in METRICS], "source_file"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def split_by_method(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        step = int(row["checkpoint_step"])
        if step < 0:
            continue
        grouped[str(row["method_key"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: int(r["checkpoint_step"]))
    return grouped


def plot_metric_grid(rows: list[dict[str, object]], out_path: Path, title: str, *, legend_cols: int) -> None:
    grouped = split_by_method(rows)
    fig, axes = plt.subplots(2, 4, figsize=(24, 11), constrained_layout=True)
    axes_flat = list(axes.ravel())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_keys = list(grouped.keys())
    color_for = {key: color_cycle[i % len(color_cycle)] for i, key in enumerate(method_keys)}

    for ax, metric in zip(axes_flat, METRICS):
        for key in method_keys:
            series = grouped[key]
            points = [
                (int(row["checkpoint_step"]), float(row[metric.key]))
                for row in series
                if not math.isnan(float(row.get(metric.key, math.nan)))
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            label = str(series[0]["method_label"])
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=label, color=color_for[key])
        ax.set_title(f"{metric.title} ({metric.ylabel})", fontsize=11)
        ax.set_xlabel("checkpoint step")
        ax.grid(True, alpha=0.28)
        if metric.key in {"fvd", "temporal_delta"}:
            ax.invert_yaxis()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=legend_cols,
        fontsize=8,
        frameon=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_each_metric(rows: list[dict[str, object]], prefix: str, title_prefix: str, *, legend_cols: int) -> None:
    grouped = split_by_method(rows)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_keys = list(grouped.keys())
    color_for = {key: color_cycle[i % len(color_cycle)] for i, key in enumerate(method_keys)}
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))
        for key in method_keys:
            series = grouped[key]
            points = [
                (int(row["checkpoint_step"]), float(row[metric.key]))
                for row in series
                if not math.isnan(float(row.get(metric.key, math.nan)))
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.9, markersize=3.8, label=str(series[0]["method_label"]), color=color_for[key])
        ax.set_title(f"{title_prefix}: {metric.title} ({metric.ylabel})")
        ax.set_xlabel("checkpoint step")
        ax.grid(True, alpha=0.28)
        if metric.key in {"fvd", "temporal_delta"}:
            ax.invert_yaxis()
        ax.legend(loc="best", fontsize=8, ncol=legend_cols)
        fig.savefig(PLOT_DIR / f"{prefix}_{metric.key}_over_checkpoints.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_readme(v4_rows: list[dict[str, object]], all_rows: list[dict[str, object]]) -> None:
    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# V4 And All-Method Checkpoint Curves",
                "",
                "Generated from existing benchmark CSVs. No videos were decoded and no metrics were recomputed.",
                "",
                "## Main Outputs",
                "",
                "- `metric_plots/v4_methods_checkpoint_grid.png`: all six V4 variants over checkpoints.",
                "- `metric_plots/all_methods_checkpoint_grid.png`: major methods tried so far over checkpoints.",
                "- `metric_plots/v4_<metric>_over_checkpoints.png`: per-metric V4 plots.",
                "- `metric_plots/all_methods_<metric>_over_checkpoints.png`: per-metric all-method plots.",
                "- `v4_plot_rows.csv`: exact V4 rows used.",
                "- `all_methods_plot_rows.csv`: exact all-method rows used.",
                "",
                "## Notes",
                "",
                "- FVD-style and temporal-delta plots invert the y-axis so upward visual movement is better.",
                "- For gate-sweep methods, the all-method plot uses the normal `gate=1.0` operating point to avoid duplicating each checkpoint by inference gate.",
                "- All comparisons are based on the fixed five validation clips unless the source benchmark says otherwise.",
                "",
                f"V4 rows plotted: `{len(v4_rows)}`.",
                f"All-method rows plotted: `{len(all_rows)}`.",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    v4_rows = load_v4_rows()
    all_rows = load_all_method_rows()

    write_used_rows(OUT_DIR / "v4_plot_rows.csv", v4_rows)
    write_used_rows(OUT_DIR / "all_methods_plot_rows.csv", all_rows)

    plot_metric_grid(
        v4_rows,
        PLOT_DIR / "v4_methods_checkpoint_grid.png",
        "All V4 variants over checkpoints",
        legend_cols=3,
    )
    plot_metric_grid(
        all_rows,
        PLOT_DIR / "all_methods_checkpoint_grid.png",
        "Major methods tried so far over checkpoints",
        legend_cols=4,
    )
    plot_each_metric(v4_rows, "v4", "All V4 variants", legend_cols=2)
    plot_each_metric(all_rows, "all_methods", "All major methods", legend_cols=2)
    write_readme(v4_rows, all_rows)

    print(f"Wrote plots to {PLOT_DIR}")
    print(f"Wrote V4 rows to {OUT_DIR / 'v4_plot_rows.csv'}")
    print(f"Wrote all-method rows to {OUT_DIR / 'all_methods_plot_rows.csv'}")


if __name__ == "__main__":
    main()
