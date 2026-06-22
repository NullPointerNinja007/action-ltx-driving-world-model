#!/usr/bin/env python3
"""Plot action-conditioning metrics over checkpoints.

The previous comparison script selects one representative row per method. This
script keeps the checkpoint dimension and produces trajectory plots.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_action_conditioning_basic_comparison import (  # noqa: E402
    BENCH_ROOT,
    DISPLAY_LABELS,
    METHOD_FAMILIES,
    METRIC_ALIASES,
    PSNR_USABLE_THRESHOLD,
    SHARPNESS_USABLE_THRESHOLD,
    SSIM_USABLE_THRESHOLD,
    MOTION_USABLE_THRESHOLD,
    attach_counterfactual_sensitivity,
    canonicalize_row,
    color_for_family,
    load_frame_action_methods,
    load_legacy_token_methods,
    load_midblock_methods,
    load_temporal_bottleneck_methods,
    read_csv,
    to_float,
    to_int,
)


OUT_DIR = BENCH_ROOT / "action_conditioning_checkpoint_trajectories_seed231_all5"
PLOT_DIR = OUT_DIR / "metric_plots"
REPORT_PATH = REPO_ROOT / "docs" / "action_conditioning_checkpoint_trajectory_report.md"

PRIMARY_METRICS = [
    ("fvd_future", "FVD-style future distance", "lower"),
    ("future_psnr", "Future PSNR", "higher"),
    ("future_ssim", "Future SSIM", "higher"),
    ("sharpness_ratio", "Sharpness ratio", "higher"),
    ("motion_ratio", "Motion ratio", "higher"),
    ("temporal_delta_error_mae", "Temporal delta error MAE", "lower"),
]

SECONDARY_METRICS = [
    ("fft_high_frequency_ratio", "FFT high-frequency ratio", "higher"),
    ("low_frequency_motion_ratio", "Low-frequency motion ratio", "higher"),
    ("low_frequency_temporal_delta_error_mae", "Low-frequency temporal delta error", "lower"),
    ("boundary_error_ratio", "Boundary error ratio", "lower"),
    ("copy_leakage_ratio", "Copy leakage ratio", "lower"),
    ("action_sensitivity_rgb_mae", "Action sensitivity RGB MAE", "higher"),
]

GROUPS = [
    (
        "legacy_tokens",
        "Legacy action tokens / AdaLN",
        [
            "legacy_global_mlp_tokens",
            "legacy_temporal_per_point_tokens",
            "legacy_tiny_transformer_tokens",
            "legacy_adaln",
        ],
        "method",
    ),
    (
        "frame_action_3000",
        "Frame-action methods, 3000-step runs",
        [
            "frame3000_frame_global_mlp",
            "frame3000_frame_temporal_pool",
            "frame3000_frame_transformer",
            "frame3000_frame_adaln",
        ],
        "method",
    ),
    (
        "frame_action_2epoch",
        "Frame-action methods, 2-epoch runs",
        [
            "frame2epoch_frame_global_mlp",
            "frame2epoch_frame_temporal_pool",
            "frame2epoch_frame_transformer",
            "frame2epoch_frame_adaln",
        ],
        "method",
    ),
    (
        "midblock_raw",
        "Middle-block gated XAttn, learned gate",
        ["midblock_raw"],
        "method",
    ),
    (
        "midblock_gate_sweep",
        "Middle-block gated XAttn, inference gate sweep",
        ["midblock_gate_tuned"],
        "gate",
    ),
    (
        "temporal_bottleneck_v1_raw",
        "Temporal bottleneck HF-teacher v1, learned gate",
        ["temporal_bottleneck_v1_raw"],
        "method",
    ),
    (
        "temporal_bottleneck_v1_gate_sweep",
        "Temporal bottleneck HF-teacher v1, inference gate sweep",
        ["temporal_bottleneck_v1_gate_tuned"],
        "gate",
    ),
    (
        "temporal_bottleneck_v2_gate_sweep",
        "Temporal bottleneck HF-teacher v2, inference gate sweep",
        ["temporal_bottleneck_v2_gate_tuned"],
        "gate",
    ),
]


def finite(value: object) -> bool:
    return not math.isnan(to_float(value))


def metric_value(row: dict[str, object], metric: str) -> float:
    return to_float(row.get(metric))


def line_label(row: dict[str, object], mode: str) -> str:
    if mode == "gate":
        gate = row.get("action_gate_scale") or "1.0"
        return f"gate={gate}"
    return str(row.get("method_label") or row.get("method_key"))


def read_noaction_shifted_reference() -> list[dict[str, object]]:
    path = (
        BENCH_ROOT
        / "noaction_shifted_timestep_longer_seed231_all5"
        / "noaction_shifted_timestep_longer_summary_with_fvd.csv"
    )
    rows = []
    for row in read_csv(path):
        if row.get("method_key") != "shifted_lognormal":
            continue
        rows.append(
            canonicalize_row(
                row,
                method_key="reference_noaction_shifted",
                method_label="No-action shifted LoRA",
                family="reference",
                source_name="noaction_shifted_trajectory_reference",
                source_path=path,
                is_reference=True,
                notes="Corrected no-action shifted/log-normal visual LoRA trajectory.",
            )
        )
    return rows


def keep_primary_diagnostic_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    kept = []
    for row in rows:
        method_key = str(row.get("method_key", ""))
        mode = str(row.get("counterfactual_action_mode", ""))
        diagnostic_group = str(row.get("diagnostic_group", ""))

        if mode not in ("", "correct"):
            continue

        if method_key in {"midblock_gate_tuned", "temporal_bottleneck_v1_gate_tuned", "temporal_bottleneck_v2_gate_tuned"}:
            if diagnostic_group != "gate_scale_sweep":
                continue
        if method_key == "temporal_bottleneck_v1_raw":
            if diagnostic_group and diagnostic_group != "checkpoint_sweep":
                continue
        kept.append(row)
    return kept


def load_all_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference_rows = read_noaction_shifted_reference()
    action_rows: list[dict[str, object]] = []
    action_rows.extend(load_legacy_token_methods())
    action_rows.extend(load_frame_action_methods())
    action_rows.extend(load_midblock_methods())
    action_rows.extend(load_temporal_bottleneck_methods())
    action_rows = keep_primary_diagnostic_rows(action_rows)
    attach_counterfactual_sensitivity(action_rows)
    action_rows = [row for row in action_rows if finite(row.get("fvd_future"))]
    reference_rows = [row for row in reference_rows if finite(row.get("fvd_future"))]
    return action_rows, reference_rows


def group_rows(rows: list[dict[str, object]], method_keys: list[str]) -> list[dict[str, object]]:
    method_set = set(method_keys)
    return [row for row in rows if row.get("method_key") in method_set]


def write_rows_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "plot_group",
        "line_label",
        "method_key",
        "method_label",
        "method_family",
        "checkpoint_step",
        "action_gate_scale",
        "model_mode",
        "future_psnr",
        "future_ssim",
        "fvd_future",
        "sharpness_ratio",
        "fft_high_frequency_ratio",
        "motion_ratio",
        "low_frequency_motion_ratio",
        "temporal_delta_error_mae",
        "low_frequency_temporal_delta_error_mae",
        "boundary_error_ratio",
        "copy_leakage_ratio",
        "action_sensitivity_rgb_mae",
        "action_sensitivity_temporal_delta_mae",
        "source_name",
        "source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    out[field] = "" if math.isnan(value) else f"{value:.10g}"
                else:
                    out[field] = value
            writer.writerow(out)


def build_plot_rows(action_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    plot_rows = []
    for group_key, _, methods, line_mode in GROUPS:
        for row in group_rows(action_rows, methods):
            out = dict(row)
            out["plot_group"] = group_key
            out["line_label"] = line_label(out, line_mode)
            plot_rows.append(out)
    return plot_rows


def series_by_label(rows: list[dict[str, object]], metric: str) -> dict[str, list[tuple[int, float, dict[str, object]]]]:
    grouped: dict[str, list[tuple[int, float, dict[str, object]]]] = defaultdict(list)
    for row in rows:
        step = to_int(row.get("checkpoint_step"))
        value = metric_value(row, metric)
        if step is None or math.isnan(value):
            continue
        grouped[str(row.get("line_label"))].append((step, value, row))
    for label in list(grouped):
        # Keep one row per step/label. Duplicate rows can happen if a diagnostic
        # file includes both checkpoint and counterfactual metadata.
        dedup: dict[int, tuple[int, float, dict[str, object]]] = {}
        for step, value, row in grouped[label]:
            dedup[step] = (step, value, row)
        grouped[label] = sorted(dedup.values(), key=lambda item: item[0])
    return grouped


def reference_series(reference_rows: list[dict[str, object]], metric: str) -> list[tuple[int, float]]:
    points = []
    for row in reference_rows:
        step = to_int(row.get("checkpoint_step"))
        value = metric_value(row, metric)
        if step is not None and not math.isnan(value):
            points.append((step, value))
    return sorted(points)


def plot_group_overview(
    group_key: str,
    group_title: str,
    rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (metric, metric_title, direction) in zip(axes.ravel(), PRIMARY_METRICS):
        grouped = series_by_label(rows, metric)
        for idx, (label, points) in enumerate(grouped.items()):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            row = points[0][2]
            color = color_for_family(str(row.get("method_family")))
            linestyle = "-" if idx < 8 else "--"
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=label, color=color, linestyle=linestyle)

        ref = reference_series(reference_rows, metric)
        if ref:
            rx = [p[0] for p in ref]
            ry = [p[1] for p in ref]
            ax.plot(rx, ry, color="#6b7280", linestyle="--", linewidth=1.4, alpha=0.75, label="no-action ref")

        ax.set_title(f"{metric_title} ({direction})", fontsize=10)
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
        if metric == "fvd_future" and any(metric_value(r, metric) > 140 for r in rows):
            ax.set_yscale("symlog", linthresh=80)
            ax.set_ylabel("symlog scale")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(group_title, fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(PLOT_DIR / f"{group_key}_primary_metrics_over_checkpoints.png", dpi=180)
    plt.close(fig)


def plot_group_secondary(
    group_key: str,
    group_title: str,
    rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
) -> None:
    available = [
        spec for spec in SECONDARY_METRICS if any(finite(row.get(spec[0])) for row in rows)
    ]
    if not available:
        return
    cols = 3
    row_count = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(row_count, cols, figsize=(18, max(4, 4 * row_count)))
    axes_arr = np.array(axes).reshape(-1)
    for ax, (metric, metric_title, direction) in zip(axes_arr, available):
        grouped = series_by_label(rows, metric)
        for idx, (label, points) in enumerate(grouped.items()):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            row = points[0][2]
            color = color_for_family(str(row.get("method_family")))
            linestyle = "-" if idx < 8 else "--"
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=label, color=color, linestyle=linestyle)
        ref = reference_series(reference_rows, metric)
        if ref:
            ax.plot([p[0] for p in ref], [p[1] for p in ref], color="#6b7280", linestyle="--", linewidth=1.4, alpha=0.75, label="no-action ref")
        ax.set_title(f"{metric_title} ({direction})", fontsize=10)
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
    for ax in axes_arr[len(available) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(f"{group_title}: secondary metrics", fontsize=15)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(PLOT_DIR / f"{group_key}_secondary_metrics_over_checkpoints.png", dpi=180)
    plt.close(fig)


def plot_metric_family_grid(
    plot_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    metric: str,
    metric_title: str,
    direction: str,
    *,
    zoom_fvd: bool = False,
) -> None:
    cols = 2
    row_count = math.ceil(len(GROUPS) / cols)
    fig, axes = plt.subplots(row_count, cols, figsize=(18, 4.4 * row_count))
    axes_arr = np.array(axes).reshape(-1)

    for ax, (group_key, group_title, _, _) in zip(axes_arr, GROUPS):
        rows = [row for row in plot_rows if row.get("plot_group") == group_key]
        grouped = series_by_label(rows, metric)
        if not grouped:
            ax.axis("off")
            continue
        for idx, (label, points) in enumerate(grouped.items()):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            row = points[0][2]
            color = color_for_family(str(row.get("method_family")))
            linestyle = "-" if idx < 8 else "--"
            ax.plot(x, y, marker="o", linewidth=1.6, markersize=3.5, label=label, color=color, linestyle=linestyle)
        ref = reference_series(reference_rows, metric)
        if ref:
            ax.plot([p[0] for p in ref], [p[1] for p in ref], color="#6b7280", linestyle="--", linewidth=1.2, alpha=0.70, label="no-action ref")
        if zoom_fvd and metric == "fvd_future":
            ax.set_ylim(50, 125)
        elif metric == "fvd_future" and any(metric_value(r, metric) > 140 for r in rows):
            ax.set_yscale("symlog", linthresh=80)
        ax.set_title(group_title, fontsize=10)
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in axes_arr[len(GROUPS) :]:
        ax.axis("off")
    zoom_text = " zoomed" if zoom_fvd else ""
    fig.suptitle(f"{metric_title} over checkpoints{zoom_text} ({direction})", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    suffix = "_zoomed_under125" if zoom_fvd else ""
    fig.savefig(PLOT_DIR / f"all_families_{metric}{suffix}_over_checkpoints.png", dpi=180)
    plt.close(fig)


def quality_passes(row: dict[str, object]) -> bool:
    return (
        metric_value(row, "sharpness_ratio") >= SHARPNESS_USABLE_THRESHOLD
        and metric_value(row, "motion_ratio") >= MOTION_USABLE_THRESHOLD
        and metric_value(row, "future_psnr") >= PSNR_USABLE_THRESHOLD
        and metric_value(row, "future_ssim") >= SSIM_USABLE_THRESHOLD
    )


def best_quality_step_by_line(plot_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in plot_rows:
        if not finite(row.get("fvd_future")):
            continue
        grouped[(str(row.get("plot_group")), str(row.get("line_label")))].append(row)

    best = []
    for (_, _), rows in grouped.items():
        positive = [r for r in rows if (to_int(r.get("checkpoint_step")) or 0) > 0]
        pool = positive or rows
        quality = [r for r in pool if quality_passes(r)]
        final_pool = quality or pool
        chosen = min(final_pool, key=lambda r: metric_value(r, "fvd_future"))
        out = dict(chosen)
        out["quality_passed"] = "yes" if quality_passes(chosen) else "no"
        out["selection_rule"] = "min_fvd_after_quality_gate" if quality else "min_fvd_fallback"
        best.append(out)
    return sorted(best, key=lambda r: (str(r.get("plot_group")), str(r.get("line_label"))))


def write_best_steps_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "plot_group",
        "line_label",
        "method_label",
        "checkpoint_step",
        "action_gate_scale",
        "fvd_future",
        "future_psnr",
        "future_ssim",
        "sharpness_ratio",
        "fft_high_frequency_ratio",
        "motion_ratio",
        "temporal_delta_error_mae",
        "action_sensitivity_rgb_mae",
        "quality_passed",
        "selection_rule",
    ]
    with (OUT_DIR / "best_checkpoint_per_line.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    out[field] = "" if math.isnan(value) else f"{value:.10g}"
                else:
                    out[field] = value
            writer.writerow(out)


def markdown_float(row: dict[str, object], key: str, digits: int = 3) -> str:
    value = metric_value(row, key)
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = [
        ("Group", "plot_group"),
        ("Line", "line_label"),
        ("Step", "checkpoint_step"),
        ("Gate", "action_gate_scale"),
        ("FVD", "fvd_future"),
        ("PSNR", "future_psnr"),
        ("SSIM", "future_ssim"),
        ("Sharp", "sharpness_ratio"),
        ("Motion", "motion_ratio"),
        ("Quality", "quality_passed"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for _, key in columns:
            if key in {"fvd_future", "future_psnr", "future_ssim", "sharpness_ratio", "motion_ratio"}:
                cells.append(markdown_float(row, key))
            else:
                cells.append(str(row.get(key, "")) or "NA")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    plot_rows: list[dict[str, object]],
    reference_rows: list[dict[str, object]],
    best_rows: list[dict[str, object]],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_rel = PLOT_DIR.relative_to(REPO_ROOT)
    out_rel = OUT_DIR.relative_to(REPO_ROOT)

    quality_best = [r for r in best_rows if r.get("quality_passed") == "yes"]
    best_fvd_quality = sorted(quality_best, key=lambda r: metric_value(r, "fvd_future"))[:5]
    failed = [r for r in best_rows if r.get("quality_passed") != "yes"]

    lines = [
        "# Action-Conditioning Checkpoint Trajectory Report",
        "",
        "## Scope",
        "",
        "This report plots metrics over checkpoint step for the action-conditioning methods we have benchmarked so far. Unlike the earlier best-row comparison, these plots preserve the training trajectory so we can see when each method starts improving, blurring, or collapsing.",
        "",
        "Metrics include FVD-style future distance, PSNR, SSIM, Laplacian sharpness ratio, motion ratio, temporal delta error, and the newer FFT/high-frequency/action-sensitivity diagnostics where available.",
        "",
        "The dashed gray line in trajectory plots is the corrected no-action shifted/log-normal LoRA reference trajectory when that metric is available. It is a reference, not an action-conditioned method.",
        "",
        "## Quality Gate Used For Best-Step Table",
        "",
        f"The best-step table chooses the lowest FVD checkpoint after requiring `sharpness_ratio >= {SHARPNESS_USABLE_THRESHOLD}`, `motion_ratio >= {MOTION_USABLE_THRESHOLD}`, `PSNR >= {PSNR_USABLE_THRESHOLD}`, and `SSIM >= {SSIM_USABLE_THRESHOLD}`. If a line never passes this gate, it falls back to lowest FVD and marks `Quality = no`.",
        "",
        "## Best Checkpoint Per Trajectory Line",
        "",
        markdown_table(best_rows),
        "",
        "## Main Checkpoint-Level Findings",
        "",
    ]

    if best_fvd_quality:
        top = best_fvd_quality[0]
        lines.append(
            f"Best quality-passing checkpoint by FVD is `{top.get('line_label')}` in `{top.get('plot_group')}` at step `{top.get('checkpoint_step')}`"
            f" with FVD `{markdown_float(top, 'fvd_future')}`, sharpness `{markdown_float(top, 'sharpness_ratio')}`, and motion `{markdown_float(top, 'motion_ratio')}`."
        )
        lines.append("")

    lines.extend(
        [
            "Most action methods show the same pattern: useful-looking metrics happen early, often around step 100 to 500, while longer training tends to reduce sharpness and motion. This is visible in the frame-action 3000-step and 2-epoch plots.",
            "",
            "AdaLN-style action injection is consistently the riskiest pathway. It can raise PSNR/SSIM, but the checkpoint trajectories show sharpness and motion dropping below the quality gate. That matches the visually blurry outputs.",
            "",
            "Raw middle-block gated XAttn also degrades as training proceeds, but the Phase 1 gate sweep shows that smaller inference gate scales can recover some visual quality. This supports the idea that action-path strength, not just architecture, is causing blur.",
            "",
            "Temporal bottleneck HF-teacher trajectories are the most promising among the newer methods. They retain more sharpness, FFT high-frequency energy, and motion at low/intermediate gate scales while getting better FVD than the no-action step-3000 reference.",
            "",
            "The action-sensitivity plots are available only for the diagnostic gate-sweep methods. They show temporal bottleneck v2 has the strongest correct-vs-counterfactual response while still preserving visual quality at low gate values.",
            "",
            "The main negative result is that simply training longer is not enough. If the action path is too global or too strong, later checkpoints become smoother/static even when PSNR or SSIM look acceptable.",
            "",
        ]
    )

    if failed:
        lines.extend(
            [
                "## Lines That Never Passed The Quality Gate",
                "",
                "These trajectories need caution: their best FVD checkpoint still fails at least one of sharpness, motion, PSNR, or SSIM.",
                "",
            ]
        )
        for row in failed:
            lines.append(
                f"- `{row.get('line_label')}` in `{row.get('plot_group')}`: best fallback step `{row.get('checkpoint_step')}`, FVD `{markdown_float(row, 'fvd_future')}`, sharpness `{markdown_float(row, 'sharpness_ratio')}`, motion `{markdown_float(row, 'motion_ratio')}`."
            )
        lines.append("")

    lines.extend(
        [
            "## Plot Index",
            "",
            f"- All families FVD: `{plot_rel / 'all_families_fvd_future_over_checkpoints.png'}`",
            f"- All families FVD zoomed: `{plot_rel / 'all_families_fvd_future_zoomed_under125_over_checkpoints.png'}`",
            f"- All families PSNR: `{plot_rel / 'all_families_future_psnr_over_checkpoints.png'}`",
            f"- All families SSIM: `{plot_rel / 'all_families_future_ssim_over_checkpoints.png'}`",
            f"- All families sharpness: `{plot_rel / 'all_families_sharpness_ratio_over_checkpoints.png'}`",
            f"- All families motion: `{plot_rel / 'all_families_motion_ratio_over_checkpoints.png'}`",
            f"- All families FFT-HF: `{plot_rel / 'all_families_fft_high_frequency_ratio_over_checkpoints.png'}`",
            "",
            "Each family also has a primary-metric overview plot and, when available, a secondary-metric plot:",
        ]
    )
    for group_key, group_title, _, _ in GROUPS:
        lines.append(f"- `{group_title}`: `{plot_rel / (group_key + '_primary_metrics_over_checkpoints.png')}`")
        secondary_path = PLOT_DIR / f"{group_key}_secondary_metrics_over_checkpoints.png"
        if secondary_path.exists():
            lines.append(f"- `{group_title}` secondary: `{plot_rel / secondary_path.name}`")
    lines.extend(
        [
            "",
            "## Data Artifacts",
            "",
            f"- Full trajectory rows: `{out_rel / 'checkpoint_trajectory_rows.csv'}`",
            f"- Best checkpoint per line: `{out_rel / 'best_checkpoint_per_line.csv'}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, reference_rows = load_all_rows()
    plot_rows = build_plot_rows(action_rows)
    write_rows_csv(plot_rows, OUT_DIR / "checkpoint_trajectory_rows.csv")

    for group_key, group_title, methods, _ in GROUPS:
        rows = [row for row in plot_rows if row.get("plot_group") == group_key]
        plot_group_overview(group_key, group_title, rows, reference_rows)
        plot_group_secondary(group_key, group_title, rows, reference_rows)

    metric_specs = PRIMARY_METRICS + SECONDARY_METRICS[:3]
    for metric, title, direction in metric_specs:
        if any(finite(row.get(metric)) for row in plot_rows):
            plot_metric_family_grid(plot_rows, reference_rows, metric, title, direction)
            if metric == "fvd_future":
                plot_metric_family_grid(
                    plot_rows,
                    reference_rows,
                    metric,
                    title,
                    direction,
                    zoom_fvd=True,
                )

    best_rows = best_quality_step_by_line(plot_rows)
    write_best_steps_csv(best_rows)
    write_report(plot_rows, reference_rows, best_rows)

    print(f"Wrote {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Trajectory rows: {len(plot_rows)}")
    print(f"Best trajectory lines: {len(best_rows)}")


if __name__ == "__main__":
    main()
