#!/usr/bin/env python3
"""Aggregate action-conditioning benchmarks into readable comparison plots.

This script intentionally avoids pandas because the local environment used for
analysis does not always have it installed.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "data" / "benchmarks"
OUT_DIR = BENCH_ROOT / "action_conditioning_basic_method_comparison_seed231_all5"
PLOT_DIR = OUT_DIR / "metric_plots"
REPORT_PATH = REPO_ROOT / "docs" / "action_conditioning_basic_comparison_report.md"

SHARPNESS_USABLE_THRESHOLD = 0.22
MOTION_USABLE_THRESHOLD = 0.70
PSNR_USABLE_THRESHOLD = 15.0
SSIM_USABLE_THRESHOLD = 0.65


DISPLAY_LABELS = {
    "reference_noaction_shifted_step3000": "No-action shifted LoRA step3000",
    "legacy_global_mlp_tokens": "Global MLP action tokens",
    "legacy_temporal_per_point_tokens": "Temporal per-point action tokens",
    "legacy_tiny_transformer_tokens": "Transformer action tokens",
    "legacy_adaln": "Global AdaLN action",
    "frame3000_frame_global_mlp": "Frame global MLP, 3000 steps",
    "frame3000_frame_temporal_pool": "Frame temporal pool, 3000 steps",
    "frame3000_frame_transformer": "Frame transformer, 3000 steps",
    "frame3000_frame_adaln": "Frame AdaLN, 3000 steps",
    "frame2epoch_frame_global_mlp": "Frame global MLP, 2 epochs",
    "frame2epoch_frame_temporal_pool": "Frame temporal pool, 2 epochs",
    "frame2epoch_frame_transformer": "Frame transformer, 2 epochs",
    "frame2epoch_frame_adaln": "Frame AdaLN, 2 epochs",
    "midblock_raw": "Middle-block gated XAttn, raw",
    "midblock_gate_tuned": "Middle-block gated XAttn, gate sweep",
    "temporal_bottleneck_v1_raw": "Temporal bottleneck HF teacher v1, raw",
    "temporal_bottleneck_v1_gate_tuned": "Temporal bottleneck HF teacher v1, gate sweep",
    "temporal_bottleneck_v2_gate_tuned": "Temporal bottleneck HF teacher v2, gate sweep",
}

METHOD_FAMILIES = {
    "reference_noaction_shifted_step3000": "reference",
    "legacy_global_mlp_tokens": "global/token",
    "legacy_temporal_per_point_tokens": "temporal/token",
    "legacy_tiny_transformer_tokens": "temporal/token",
    "legacy_adaln": "adaln",
    "frame3000_frame_global_mlp": "frame-action",
    "frame3000_frame_temporal_pool": "frame-action",
    "frame3000_frame_transformer": "frame-action",
    "frame3000_frame_adaln": "adaln",
    "frame2epoch_frame_global_mlp": "frame-action",
    "frame2epoch_frame_temporal_pool": "frame-action",
    "frame2epoch_frame_transformer": "frame-action",
    "frame2epoch_frame_adaln": "adaln",
    "midblock_raw": "gated-middle",
    "midblock_gate_tuned": "gated-middle",
    "temporal_bottleneck_v1_raw": "temporal-bottleneck",
    "temporal_bottleneck_v1_gate_tuned": "temporal-bottleneck",
    "temporal_bottleneck_v2_gate_tuned": "temporal-bottleneck",
}

METRIC_ALIASES = {
    "future_psnr": ["mean_future_psnr", "future_psnr"],
    "future_ssim": ["mean_future_global_ssim", "future_global_ssim"],
    "sharpness_ratio": [
        "mean_sharpness_ratio_generated_over_reference",
        "sharpness_ratio",
    ],
    "fft_high_frequency_ratio": [
        "mean_fft_high_frequency_energy_ratio_generated_over_reference",
        "fft_high_frequency_ratio",
    ],
    "motion_ratio": [
        "mean_motion_ratio_generated_over_reference",
        "motion_ratio",
    ],
    "low_frequency_motion_ratio": [
        "mean_low_frequency_motion_ratio_generated_over_reference",
        "low_frequency_motion_ratio",
    ],
    "temporal_delta_error_mae": [
        "mean_temporal_delta_error_mae",
        "temporal_delta_error_mae",
    ],
    "low_frequency_temporal_delta_error_mae": [
        "mean_low_frequency_temporal_delta_error_mae",
        "low_frequency_temporal_delta_error_mae",
    ],
    "boundary_error_ratio": [
        "mean_boundary_mae_ratio_generated_over_reference",
        "boundary_mae_ratio_generated_over_reference",
        "boundary_error_ratio",
    ],
    "copy_leakage_ratio": [
        "mean_copy_leakage_ratio_min_context_over_future",
        "copy_leakage_ratio",
    ],
    "future_mse": ["mean_future_mse", "future_mse"],
    "future_mae": ["mean_future_mae", "future_mae"],
}

CANONICAL_FIELDS = [
    "method_key",
    "method_label",
    "method_family",
    "source_name",
    "source_path",
    "is_reference",
    "checkpoint_step",
    "model_mode",
    "action_gate_scale",
    "action_vector_scale",
    "diagnostic_group",
    "counterfactual_action_mode",
    "selection_source",
    "selected_from_usable_pool",
    "usable_by_quality_gate",
    "num_clips",
    "future_mse",
    "future_mae",
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
    "fvd_backend",
    "notes",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    method_key: str | None = None
    method_label: str | None = None
    family: str | None = None
    prefix_method_key: str | None = None
    source_note: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def to_int(value: object) -> int | None:
    number = to_float(value)
    if math.isnan(number):
        return None
    return int(number)


def first_present(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def metric_value(row: dict[str, str], metric: str) -> float:
    return to_float(first_present(row, METRIC_ALIASES[metric]))


def canonicalize_row(
    row: dict[str, str],
    *,
    method_key: str,
    method_label: str,
    family: str,
    source_name: str,
    source_path: Path,
    is_reference: bool = False,
    notes: str = "",
) -> dict[str, object]:
    sharpness = metric_value(row, "sharpness_ratio")
    motion = metric_value(row, "motion_ratio")
    psnr = metric_value(row, "future_psnr")
    ssim = metric_value(row, "future_ssim")
    usable = (
        not math.isnan(sharpness)
        and not math.isnan(motion)
        and not math.isnan(psnr)
        and not math.isnan(ssim)
        and sharpness >= SHARPNESS_USABLE_THRESHOLD
        and motion >= MOTION_USABLE_THRESHOLD
        and psnr >= PSNR_USABLE_THRESHOLD
        and ssim >= SSIM_USABLE_THRESHOLD
    )
    out: dict[str, object] = {
        "method_key": method_key,
        "method_label": method_label,
        "method_family": family,
        "source_name": source_name,
        "source_path": str(source_path.relative_to(REPO_ROOT)),
        "is_reference": "yes" if is_reference else "no",
        "checkpoint_step": first_present(row, ["checkpoint_step", "step"]),
        "model_mode": first_present(row, ["model_mode", "checkpoint_name"]),
        "action_gate_scale": first_present(row, ["action_gate_scale"]),
        "action_vector_scale": first_present(row, ["action_vector_scale"]),
        "diagnostic_group": first_present(row, ["diagnostic_group"]),
        "counterfactual_action_mode": first_present(row, ["counterfactual_action_mode"]),
        "selection_source": "",
        "selected_from_usable_pool": "",
        "usable_by_quality_gate": "yes" if usable else "no",
        "num_clips": first_present(row, ["num_clips"]),
        "fvd_future": to_float(row.get("fvd_future")),
        "fvd_backend": first_present(row, ["fvd_backend"]),
        "notes": notes,
    }
    for metric in METRIC_ALIASES:
        out[metric] = metric_value(row, metric)
    return out


def load_noaction_reference() -> list[dict[str, object]]:
    path = (
        BENCH_ROOT
        / "noaction_shifted_timestep_longer_seed231_all5"
        / "noaction_shifted_timestep_longer_summary_with_fvd.csv"
    )
    rows = read_csv(path)
    selected = [
        r
        for r in rows
        if r.get("method_key") == "shifted_lognormal" and r.get("checkpoint_step") == "3000"
    ]
    out = []
    for row in selected[:1]:
        out.append(
            canonicalize_row(
                row,
                method_key="reference_noaction_shifted_step3000",
                method_label=DISPLAY_LABELS["reference_noaction_shifted_step3000"],
                family=METHOD_FAMILIES["reference_noaction_shifted_step3000"],
                source_name="noaction_shifted_step3000_reference",
                source_path=path,
                is_reference=True,
                notes="Corrected no-action shifted/log-normal LoRA reference used for interpretation.",
            )
        )
    return out


def load_legacy_token_methods() -> list[dict[str, object]]:
    path = (
        BENCH_ROOT
        / "action_checkpoint_sweep_all_encoders_with_adaln_seed231_all5"
        / "action_checkpoint_sweep_summary_all_encoders_with_adaln_fvd.csv"
    )
    mapping = {
        "global_mlp": "legacy_global_mlp_tokens",
        "Global MLP": "legacy_global_mlp_tokens",
        "temporal_per_point": "legacy_temporal_per_point_tokens",
        "Temporal Per-Point": "legacy_temporal_per_point_tokens",
        "tiny_transformer": "legacy_tiny_transformer_tokens",
        "Transformer Action": "legacy_tiny_transformer_tokens",
        "adaln": "legacy_adaln",
        "AdaLN Action": "legacy_adaln",
    }
    out = []
    for row in read_csv(path):
        encoder = row.get("encoder", "")
        method_key = mapping.get(encoder)
        if not method_key:
            continue
        out.append(
            canonicalize_row(
                row,
                method_key=method_key,
                method_label=DISPLAY_LABELS[method_key],
                family=METHOD_FAMILIES[method_key],
                source_name="legacy_action_checkpoint_sweep",
                source_path=path,
                notes="Original action-conditioning sweep before frame-action upsampling diagnostics.",
            )
        )
    return out


def load_frame_action_methods() -> list[dict[str, object]]:
    specs = [
        SourceSpec(
            name="frame_action_3000_step_sweep",
            path=BENCH_ROOT
            / "frame_action_checkpoint_sweep_seed231_all5"
            / "frame_action_checkpoint_sweep_summary_with_fvd.csv",
            prefix_method_key="frame3000",
            source_note="Frame-action upsampled conditioning, 3000-step checkpoint sweep.",
        ),
        SourceSpec(
            name="frame_action_2epoch_sweep",
            path=BENCH_ROOT
            / "frame_action_2epoch_checkpoint_sweep_seed231_all5"
            / "frame_action_2epoch_checkpoint_sweep_summary_with_fvd.csv",
            prefix_method_key="frame2epoch",
            source_note="Frame-action upsampled conditioning, 2-epoch checkpoint sweep.",
        ),
    ]
    out = []
    for spec in specs:
        for row in read_csv(spec.path):
            raw_key = row.get("method_key", "")
            method_key = f"{spec.prefix_method_key}_{raw_key}"
            if method_key not in DISPLAY_LABELS:
                continue
            out.append(
                canonicalize_row(
                    row,
                    method_key=method_key,
                    method_label=DISPLAY_LABELS[method_key],
                    family=METHOD_FAMILIES[method_key],
                    source_name=spec.name,
                    source_path=spec.path,
                    notes=spec.source_note,
                )
            )
    return out


def load_midblock_methods() -> list[dict[str, object]]:
    raw_path = (
        BENCH_ROOT
        / "midblock_gated_xattn_checkpoint_sweep_seed231_all5"
        / "midblock_gated_xattn_summary_with_fvd.csv"
    )
    out = [
        canonicalize_row(
            row,
            method_key="midblock_raw",
            method_label=DISPLAY_LABELS["midblock_raw"],
            family=METHOD_FAMILIES["midblock_raw"],
            source_name="midblock_checkpoint_sweep",
            source_path=raw_path,
            notes="Middle-block action cross-attention checkpoint sweep at learned gate scale.",
        )
        for row in read_csv(raw_path)
    ]

    gate_path = (
        BENCH_ROOT
        / "phase1_action_diagnostics_midblock_seed231_all5"
        / "phase1_model_summary_with_fvd.csv"
    )
    for row in read_csv(gate_path):
        if row.get("diagnostic_group") != "gate_scale_sweep":
            continue
        if row.get("counterfactual_action_mode", "correct") not in ("", "correct"):
            continue
        out.append(
            canonicalize_row(
                row,
                method_key="midblock_gate_tuned",
                method_label=DISPLAY_LABELS["midblock_gate_tuned"],
                family=METHOD_FAMILIES["midblock_gate_tuned"],
                source_name="phase1_midblock_gate_scale_sweep",
                source_path=gate_path,
                notes="Inference-only gate-scale sweep for middle-block gated cross-attention.",
            )
        )
    return out


def load_temporal_bottleneck_methods() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    raw_path = (
        BENCH_ROOT
        / "frame_temporal_bottleneck_hfteacher_checkpoint_sweep_seed231_all5"
        / "frame_temporal_bottleneck_hfteacher_summary_with_fvd.csv"
    )
    for row in read_csv(raw_path):
        out.append(
            canonicalize_row(
                row,
                method_key="temporal_bottleneck_v1_raw",
                method_label=DISPLAY_LABELS["temporal_bottleneck_v1_raw"],
                family=METHOD_FAMILIES["temporal_bottleneck_v1_raw"],
                source_name="temporal_bottleneck_v1_checkpoint_sweep",
                source_path=raw_path,
                notes="Temporal bottleneck HF-teacher checkpoint sweep at learned gate scale.",
            )
        )

    gate_specs = [
        (
            "temporal_bottleneck_v1_gate_tuned",
            "temporal_bottleneck_v1_gate_scale_sweep",
            BENCH_ROOT
            / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"
            / "gate_scale_summary.csv",
            "Temporal bottleneck HF-teacher v1 inference-only gate-scale sweep.",
        ),
        (
            "temporal_bottleneck_v2_gate_tuned",
            "temporal_bottleneck_v2_gate_scale_sweep",
            BENCH_ROOT
            / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_seed231_all5"
            / "gate_scale_summary.csv",
            "Temporal bottleneck HF-teacher v2 inference-only gate-scale sweep.",
        ),
    ]
    for method_key, source_name, path, note in gate_specs:
        for row in read_csv(path):
            if row.get("counterfactual_action_mode", "correct") not in ("", "correct"):
                continue
            out.append(
                canonicalize_row(
                    row,
                    method_key=method_key,
                    method_label=DISPLAY_LABELS[method_key],
                    family=METHOD_FAMILIES[method_key],
                    source_name=source_name,
                    source_path=path,
                    notes=note,
                )
            )
    return out


def attach_counterfactual_sensitivity(rows: list[dict[str, object]]) -> None:
    summary_specs = [
        (
            "midblock_gate_tuned",
            BENCH_ROOT
            / "phase1_action_diagnostics_midblock_seed231_all5"
            / "counterfactual_sensitivity_summary.csv",
        ),
        (
            "temporal_bottleneck_v1_gate_tuned",
            BENCH_ROOT
            / "frame_temporal_bottleneck_hfteacher_gate_scale_sweep_seed231_all5"
            / "counterfactual_sensitivity_summary.csv",
        ),
        (
            "temporal_bottleneck_v2_gate_tuned",
            BENCH_ROOT
            / "frame_temporal_bottleneck_hfteacher_v2_gate_scale_sweep_seed231_all5"
            / "counterfactual_sensitivity_summary.csv",
        ),
    ]

    summaries: dict[tuple[str, str, str], tuple[float, float]] = {}
    for method_key, path in summary_specs:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(path):
            step = row.get("checkpoint_step", "")
            gate = row.get("action_gate_scale", "1.0") or "1.0"
            grouped[(step, gate)].append(row)
        for (step, gate), group in grouped.items():
            rgb_values = [
                to_float(r.get("mean_future_rgb_mae_correct_vs_mode")) for r in group
            ]
            temporal_values = [
                to_float(r.get("mean_future_temporal_delta_mae_correct_vs_mode"))
                for r in group
            ]
            rgb_values = [v for v in rgb_values if not math.isnan(v)]
            temporal_values = [v for v in temporal_values if not math.isnan(v)]
            if rgb_values:
                summaries[(method_key, step, gate)] = (
                    float(np.mean(rgb_values)),
                    float(np.mean(temporal_values)) if temporal_values else math.nan,
                )

    for row in rows:
        method_key = str(row["method_key"])
        step = str(row.get("checkpoint_step", ""))
        gate = str(row.get("action_gate_scale", "") or "1.0")
        pair = summaries.get((method_key, step, gate))
        if pair:
            row["action_sensitivity_rgb_mae"] = pair[0]
            row["action_sensitivity_temporal_delta_mae"] = pair[1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in CANONICAL_FIELDS:
                value = row.get(field, "")
                if isinstance(value, float):
                    out[field] = "" if math.isnan(value) else f"{value:.10g}"
                else:
                    out[field] = value
            writer.writerow(out)


def finite(row: dict[str, object], key: str) -> bool:
    return not math.isnan(to_float(row.get(key)))


def select_best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not finite(row, "fvd_future"):
            continue
        grouped[str(row["method_key"])].append(row)

    selected = []
    for method_key, candidates in grouped.items():
        if method_key == "reference_noaction_shifted_step3000":
            chosen = candidates[0]
            chosen = dict(chosen)
            chosen["selected_from_usable_pool"] = chosen["usable_by_quality_gate"]
            chosen["selection_source"] = "fixed_reference_step3000"
            selected.append(chosen)
            continue

        positive_step_candidates = [
            r for r in candidates if (to_int(r.get("checkpoint_step")) or 0) > 0
        ]
        selection_candidates = positive_step_candidates or candidates
        usable = [
            r
            for r in selection_candidates
            if r.get("usable_by_quality_gate") == "yes"
        ]
        pool = usable if usable else selection_candidates
        chosen = min(pool, key=lambda r: to_float(r.get("fvd_future")))
        chosen = dict(chosen)
        chosen["selected_from_usable_pool"] = "yes" if usable else "no"
        chosen["selection_source"] = (
            "min_fvd_positive_step_among_quality_gate"
            if usable
            else "min_fvd_positive_step_fallback_failed_quality_gate"
        )
        selected.append(chosen)

    return sorted(
        selected,
        key=lambda r: (
            0 if r["method_key"] == "reference_noaction_shifted_step3000" else 1,
            str(r["method_family"]),
            str(r["method_label"]),
        ),
    )


def select_action_sensitivity_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("is_reference") == "yes":
            continue
        if not finite(row, "action_sensitivity_rgb_mae"):
            continue
        grouped[str(row["method_key"])].append(row)

    selected = []
    for _, candidates in grouped.items():
        positive_step_candidates = [
            r for r in candidates if (to_int(r.get("checkpoint_step")) or 0) > 0
        ]
        selection_candidates = positive_step_candidates or candidates
        usable = [
            r
            for r in selection_candidates
            if r.get("usable_by_quality_gate") == "yes"
        ]
        pool = usable if usable else selection_candidates
        chosen = max(pool, key=lambda r: to_float(r.get("action_sensitivity_rgb_mae")))
        chosen = dict(chosen)
        chosen["selected_from_usable_pool"] = "yes" if usable else "no"
        chosen["selection_source"] = (
            "max_action_sensitivity_positive_step_among_quality_gate"
            if usable
            else "max_action_sensitivity_positive_step_fallback_failed_quality_gate"
        )
        selected.append(chosen)
    return sorted(selected, key=lambda r: to_float(r.get("action_sensitivity_rgb_mae")), reverse=True)


def color_for_family(family: str) -> str:
    return {
        "reference": "#6b7280",
        "global/token": "#9a3412",
        "temporal/token": "#2563eb",
        "frame-action": "#059669",
        "adaln": "#7c2d12",
        "gated-middle": "#9333ea",
        "temporal-bottleneck": "#0f766e",
    }.get(family, "#374151")


def method_labels(rows: list[dict[str, object]]) -> list[str]:
    labels = []
    for row in rows:
        label = str(row["method_label"])
        step = str(row.get("checkpoint_step", ""))
        gate = str(row.get("action_gate_scale", ""))
        suffix_parts = []
        if step:
            suffix_parts.append(f"s{step}")
        if gate:
            suffix_parts.append(f"g{gate}")
        if suffix_parts:
            label = f"{label} ({', '.join(suffix_parts)})"
        labels.append(label)
    return labels


def plot_metric_bars(
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    xlabel: str,
    output: Path,
    *,
    lower_is_better: bool,
) -> None:
    metric_rows = [r for r in rows if finite(r, metric)]
    metric_rows = sorted(
        metric_rows,
        key=lambda r: to_float(r.get(metric)),
        reverse=not lower_is_better,
    )
    labels = method_labels(metric_rows)
    values = [to_float(r.get(metric)) for r in metric_rows]
    colors = [color_for_family(str(r["method_family"])) for r in metric_rows]

    fig_height = max(5.5, 0.44 * len(metric_rows) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y = np.arange(len(metric_rows))
    ax.barh(y, values, color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3g}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_metric_bars_filtered(
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    xlabel: str,
    output: Path,
    *,
    lower_is_better: bool,
    max_value: float | None = None,
) -> None:
    metric_rows = [r for r in rows if finite(r, metric)]
    if max_value is not None:
        metric_rows = [r for r in metric_rows if to_float(r.get(metric)) <= max_value]
    plot_metric_bars(
        metric_rows,
        metric,
        title,
        xlabel,
        output,
        lower_is_better=lower_is_better,
    )


def plot_overview_grid(rows: list[dict[str, object]]) -> None:
    specs = [
        ("fvd_future", "FVD-style future distance", "lower is better", True),
        ("future_psnr", "Future PSNR", "higher is better", False),
        ("future_ssim", "Future SSIM", "higher is better", False),
        ("sharpness_ratio", "Laplacian sharpness ratio", "higher is better", False),
        ("motion_ratio", "Motion ratio", "higher is better", False),
        ("temporal_delta_error_mae", "Temporal delta error MAE", "lower is better", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    labels = method_labels(rows)
    y = np.arange(len(rows))
    colors = [color_for_family(str(r["method_family"])) for r in rows]
    for ax, (metric, title, subtitle, lower_is_better) in zip(axes.ravel(), specs):
        vals = np.array([to_float(r.get(metric)) for r in rows], dtype=float)
        order = np.argsort(vals)
        if not lower_is_better:
            order = order[::-1]
        order = [i for i in order if not math.isnan(vals[i])]
        ax.barh(
            np.arange(len(order)),
            vals[order],
            color=[colors[i] for i in order],
            alpha=0.88,
        )
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        for idx, original_idx in enumerate(order):
            ax.text(vals[original_idx], idx, f" {vals[original_idx]:.3g}", fontsize=6.5, va="center")
    fig.suptitle(
        "Best observed action-conditioning rows under quality-gated FVD selection",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "basic_metric_overview_grid.png", dpi=180)
    plt.close(fig)


def plot_scatter(
    rows: list[dict[str, object]],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output: Path,
) -> None:
    metric_rows = [r for r in rows if finite(r, x_metric) and finite(r, y_metric)]
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in metric_rows:
        x = to_float(row.get(x_metric))
        y = to_float(row.get(y_metric))
        color = color_for_family(str(row["method_family"]))
        marker = "*" if row.get("is_reference") == "yes" else "o"
        ax.scatter(x, y, s=95, color=color, marker=marker, edgecolor="black", linewidth=0.5)
        label = str(row["method_label"]).replace(", ", "\n")
        ax.annotate(label, (x, y), fontsize=7, xytext=(5, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter_filtered(
    rows: list[dict[str, object]],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output: Path,
    *,
    max_y: float | None = None,
) -> None:
    metric_rows = [r for r in rows if finite(r, x_metric) and finite(r, y_metric)]
    if max_y is not None:
        metric_rows = [r for r in metric_rows if to_float(r.get(y_metric)) <= max_y]
    plot_scatter(metric_rows, x_metric, y_metric, title, xlabel, ylabel, output)


def normalized_score(values: list[float], lower_is_better: bool) -> list[float]:
    finite_values = [v for v in values if not math.isnan(v)]
    if not finite_values:
        return [math.nan for _ in values]
    min_v, max_v = min(finite_values), max(finite_values)
    if math.isclose(min_v, max_v):
        return [0.5 if not math.isnan(v) else math.nan for v in values]
    scores = []
    for value in values:
        if math.isnan(value):
            scores.append(math.nan)
        elif lower_is_better:
            scores.append((max_v - value) / (max_v - min_v))
        else:
            scores.append((value - min_v) / (max_v - min_v))
    return scores


def plot_heatmap(rows: list[dict[str, object]]) -> None:
    metrics = [
        ("fvd_future", "FVD", True),
        ("future_psnr", "PSNR", False),
        ("future_ssim", "SSIM", False),
        ("sharpness_ratio", "Sharp", False),
        ("fft_high_frequency_ratio", "FFT-HF", False),
        ("motion_ratio", "Motion", False),
        ("temporal_delta_error_mae", "Delta err", True),
        ("copy_leakage_ratio", "Copy leak", True),
        ("action_sensitivity_rgb_mae", "Action sens.", False),
    ]
    matrix = []
    for metric, _, lower in metrics:
        values = [to_float(r.get(metric)) for r in rows]
        matrix.append(normalized_score(values, lower))
    data = np.array(matrix, dtype=float).T
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e5e7eb")
    fig_height = max(6, 0.42 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(method_labels(rows), fontsize=8)
    ax.set_title("Normalized comparison score, selected rows only\n1 is best within this table; gray means metric unavailable")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not math.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if data[i, j] < 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "normalized_metric_heatmap.png", dpi=180)
    plt.close(fig)


def write_method_coverage(rows: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method_key"])].append(row)
    coverage_fields = [
        "method_key",
        "method_label",
        "method_family",
        "candidate_rows",
        "rows_with_fvd",
        "rows_passing_quality_gate",
        "has_fft",
        "has_counterfactual_sensitivity",
        "source_names",
    ]
    with (OUT_DIR / "method_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=coverage_fields)
        writer.writeheader()
        for method_key, group in sorted(grouped.items()):
            writer.writerow(
                {
                    "method_key": method_key,
                    "method_label": group[0]["method_label"],
                    "method_family": group[0]["method_family"],
                    "candidate_rows": len(group),
                    "rows_with_fvd": sum(finite(r, "fvd_future") for r in group),
                    "rows_passing_quality_gate": sum(r.get("usable_by_quality_gate") == "yes" for r in group),
                    "has_fft": any(finite(r, "fft_high_frequency_ratio") for r in group),
                    "has_counterfactual_sensitivity": any(finite(r, "action_sensitivity_rgb_mae") for r in group),
                    "source_names": ";".join(sorted({str(r["source_name"]) for r in group})),
                }
            )


def markdown_float(row: dict[str, object], key: str, digits: int = 3) -> str:
    value = to_float(row.get(key))
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], keys: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for _, key in keys:
            if key == "method_label":
                cells.append(str(row[key]))
            elif key in {"checkpoint_step", "action_gate_scale", "selection_source"}:
                cells.append(str(row.get(key, "")) or "NA")
            else:
                cells.append(markdown_float(row, key))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    best_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    action_rows = [r for r in best_rows if r.get("is_reference") != "yes"]
    fvd_sorted = sorted(action_rows, key=lambda r: to_float(r.get("fvd_future")))
    quality_passing_action_rows = [
        r for r in action_rows if r.get("selected_from_usable_pool") == "yes"
    ]
    fvd_sorted_quality = sorted(
        quality_passing_action_rows, key=lambda r: to_float(r.get("fvd_future"))
    )
    sharp_sorted = sorted(
        [
            r
            for r in quality_passing_action_rows
            if to_float(r.get("fvd_future")) <= 120.0
        ],
        key=lambda r: to_float(r.get("sharpness_ratio")),
        reverse=True,
    )
    sensitivity_sorted = sorted(
        sensitivity_rows,
        key=lambda r: to_float(r.get("action_sensitivity_rgb_mae")),
        reverse=True,
    )
    failed_gate = [
        r for r in action_rows if r.get("selected_from_usable_pool") == "no"
    ]

    missing_fvd = [
        r
        for r in all_rows
        if not finite(r, "fvd_future") and r.get("is_reference") != "yes"
    ]

    plot_rel = PLOT_DIR.relative_to(REPO_ROOT)
    out_rel = OUT_DIR.relative_to(REPO_ROOT)
    lines = [
        "# Action-Conditioning Basic Comparison Report",
        "",
        "## Scope",
        "",
        "This report aggregates the existing action-conditioning benchmark outputs for the 5 local Waymo validation clips. The comparison focuses on the basic metrics we have consistently used: FVD-style future distance, future PSNR, future SSIM, Laplacian sharpness ratio, motion ratio, temporal delta error, copy leakage, and FFT high-frequency retention when available.",
        "",
        "The FVD number here is a project-level FVD-style future distance over 5 clips, not a publication-grade full-dataset FVD. It is still useful for relative comparisons because every row uses the same validation clips, frame count, FPS, and benchmark implementation.",
        "",
        "## Selection Rule",
        "",
        f"For every method variation, the script first keeps rows that pass `sharpness_ratio >= {SHARPNESS_USABLE_THRESHOLD}`, `motion_ratio >= {MOTION_USABLE_THRESHOLD}`, `PSNR >= {PSNR_USABLE_THRESHOLD}`, and `SSIM >= {SSIM_USABLE_THRESHOLD}`. It then selects the lowest FVD-style future distance in that usable pool. If a method has no usable row, it falls back to lowest FVD and marks that method as a failed quality-gate selection.",
        "",
            "The no-action shifted LoRA `step_003000` is included as a gray reference because that was the corrected visual baseline used for later action experiments. It is not treated as an action-conditioning method.",
            "",
            "Action-method selections exclude `step_000000` rows because those are base-reference states before the action pathway has trained.",
            "",
            "## Selected Rows",
        "",
        markdown_table(
            best_rows,
            [
                ("Method", "method_label"),
                ("Step", "checkpoint_step"),
                ("Gate", "action_gate_scale"),
                ("FVD", "fvd_future"),
                ("PSNR", "future_psnr"),
                ("SSIM", "future_ssim"),
                ("Sharp", "sharpness_ratio"),
                ("FFT-HF", "fft_high_frequency_ratio"),
                ("Motion", "motion_ratio"),
                ("Action sens.", "action_sensitivity_rgb_mae"),
                ("Selection", "selection_source"),
            ],
        ),
        "",
    ]

    if sensitivity_rows:
        lines.extend(
            [
                "## Counterfactual Action Sensitivity Rows",
                "",
                "These are selected separately from the quality/FVD rows. They choose the strongest available correct-vs-counterfactual response among rows that still pass the same quality gate when possible.",
                "",
                markdown_table(
                    sensitivity_rows,
                    [
                        ("Method", "method_label"),
                        ("Step", "checkpoint_step"),
                        ("Gate", "action_gate_scale"),
                        ("Action sens.", "action_sensitivity_rgb_mae"),
                        ("Temporal sens.", "action_sensitivity_temporal_delta_mae"),
                        ("FVD", "fvd_future"),
                        ("Sharp", "sharpness_ratio"),
                        ("Motion", "motion_ratio"),
                        ("Selection", "selection_source"),
                    ],
                ),
                "",
            ]
        )

    lines.extend(["## Main Findings", ""])

    if fvd_sorted_quality:
        top = fvd_sorted_quality[0]
        lines.append(
            f"The best quality-passing action-conditioned FVD-style row is **{top['method_label']}** at step `{top.get('checkpoint_step')}`"
            f" with gate `{top.get('action_gate_scale') or 'NA'}`: FVD `{markdown_float(top, 'fvd_future')}`,"
            f" sharpness `{markdown_float(top, 'sharpness_ratio')}`, motion `{markdown_float(top, 'motion_ratio')}`."
        )
        lines.append("")
    if fvd_sorted and fvd_sorted[0].get("selected_from_usable_pool") != "yes":
        top_raw = fvd_sorted[0]
        lines.append(
            f"The lowest FVD-style row overall is **{top_raw['method_label']}** at FVD `{markdown_float(top_raw, 'fvd_future')}`,"
            f" but it fails the quality gate with sharpness `{markdown_float(top_raw, 'sharpness_ratio')}` and motion `{markdown_float(top_raw, 'motion_ratio')}`."
            " This is why the report does not treat lowest FVD alone as the winner."
        )
        lines.append("")
    if sharp_sorted:
        top_sharp = sharp_sorted[0]
        lines.append(
            f"Among quality-passing action rows, the strongest high-frequency retention is **{top_sharp['method_label']}** with sharpness ratio `{markdown_float(top_sharp, 'sharpness_ratio')}`"
            f" and FVD `{markdown_float(top_sharp, 'fvd_future')}`."
        )
        lines.append("")
    if sensitivity_sorted:
        top_sens = sensitivity_sorted[0]
        lines.append(
            f"The strongest measured counterfactual action sensitivity is **{top_sens['method_label']}** with mean future RGB MAE `{markdown_float(top_sens, 'action_sensitivity_rgb_mae')}`."
            " This metric is only available for the diagnostic gate-sweep methods, so older token methods should not be interpreted as action-insensitive solely because the field is missing."
        )
        lines.append("")

    lines.extend(
        [
            "Across the runs, the recurring failure mode is clear: methods that let action features globally affect the visual pathway often improve or maintain PSNR/SSIM while suppressing sharpness and motion. PSNR/SSIM alone are therefore not enough; blur can look numerically closer to the average future while becoming visually unusable.",
            "",
            "The middle-block and temporal-bottleneck experiments support the current hypothesis that action should be routed through a constrained temporal path. Full-strength learned gates hurt high-frequency detail, but low-gate inference settings preserve much more of the corrected no-action visual quality.",
            "",
            "The temporal bottleneck HF-teacher v2 gate sweep is currently the most promising direction because it preserves no-action-like sharpness/FFT/motion at low gate values while showing stronger counterfactual action sensitivity than the Phase 1 middle-block diagnostics.",
            "",
            "However, the best v2 rows are still early/low-gate operating points. That means the model may be only weakly using actions. The next architectural step should add an explicit low-frequency action-following objective instead of relying only on diffusion MSE plus gates.",
            "",
            "## Quality-Gate Failures",
            "",
        ]
    )
    if failed_gate:
        lines.append(
            "The following selected rows did not have any checkpoint/configuration passing the full quality gate, so their best row is a fallback by FVD only:"
        )
        lines.append("")
        for row in failed_gate:
            lines.append(
                f"- {row['method_label']}: selected FVD `{markdown_float(row, 'fvd_future')}`, sharpness `{markdown_float(row, 'sharpness_ratio')}`, motion `{markdown_float(row, 'motion_ratio')}`."
            )
        lines.append("")
    else:
        lines.append("Every selected method had at least one row passing the basic sharpness/motion quality gate.")
        lines.append("")

    lines.extend(
        [
            "## Metric Interpretation",
            "",
            "FVD-style future distance: lower is better, but this is computed over only 5 clips and should be used as a relative diagnostic.",
            "",
            "PSNR and SSIM: higher means generated future is closer to the ground-truth future under pixel/structural similarity. These metrics can reward blurry averages, so they must be read with sharpness and motion.",
            "",
            "Laplacian sharpness ratio: generated future sharpness divided by reference future sharpness. Higher is better; collapse here matches the visually blurry outputs we saw.",
            "",
            "Motion ratio: generated temporal-change magnitude divided by reference. Higher is better here because most runs are below 1. Low values indicate static/copy-like futures.",
            "",
            "FFT high-frequency ratio: generated high-frequency energy divided by reference. Higher is better; this is available only for newer benchmark runs.",
            "",
            "Action sensitivity: mean future RGB MAE between correct-action generation and counterfactual-action generation. Higher means actions affect the output more. This is not itself quality; high sensitivity plus blur means the action path is active but harmful.",
            "",
            "## Plots",
            "",
            f"- `{plot_rel / 'basic_metric_overview_grid.png'}`",
            f"- `{plot_rel / 'fvd_future_bar.png'}`",
            f"- `{plot_rel / 'fvd_future_bar_zoomed_under120.png'}`",
            f"- `{plot_rel / 'future_psnr_bar.png'}`",
            f"- `{plot_rel / 'future_ssim_bar.png'}`",
            f"- `{plot_rel / 'sharpness_ratio_bar.png'}`",
            f"- `{plot_rel / 'motion_ratio_bar.png'}`",
            f"- `{plot_rel / 'fvd_vs_sharpness.png'}`",
            f"- `{plot_rel / 'fvd_vs_sharpness_zoomed_under120.png'}`",
            f"- `{plot_rel / 'fvd_vs_motion.png'}`",
            f"- `{plot_rel / 'fvd_vs_motion_zoomed_under120.png'}`",
            f"- `{plot_rel / 'fft_high_frequency_ratio_bar.png'}`",
            f"- `{plot_rel / 'action_sensitivity_rgb_mae_bar.png'}`",
            f"- `{plot_rel / 'action_sensitivity_vs_sharpness.png'}`",
            f"- `{plot_rel / 'normalized_metric_heatmap.png'}`",
            "",
            "## Data Artifacts",
            "",
            f"- Full candidates: `{out_rel / 'all_action_method_candidate_rows.csv'}`",
            f"- Selected rows: `{out_rel / 'best_action_method_rows.csv'}`",
            f"- Method coverage: `{out_rel / 'method_coverage.csv'}`",
            "",
        ]
    )
    if missing_fvd:
        lines.extend(
            [
                "## Excluded Non-FVD Rows",
                "",
                f"There are `{len(missing_fvd)}` action rows without FVD-style values. They are kept out of the primary comparison because the requested baseline metric is FVD-style future distance. Some one-off full7992 summaries fall into this category.",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    rows.extend(load_noaction_reference())
    rows.extend(load_legacy_token_methods())
    rows.extend(load_frame_action_methods())
    rows.extend(load_midblock_methods())
    rows.extend(load_temporal_bottleneck_methods())
    attach_counterfactual_sensitivity(rows)

    rows = [r for r in rows if finite(r, "fvd_future")]
    best_rows = select_best_rows(rows)
    sensitivity_rows = select_action_sensitivity_rows(rows)

    write_csv(OUT_DIR / "all_action_method_candidate_rows.csv", rows)
    write_csv(OUT_DIR / "best_action_method_rows.csv", best_rows)
    write_csv(OUT_DIR / "best_action_sensitivity_rows.csv", sensitivity_rows)
    write_method_coverage(rows)

    plot_overview_grid(best_rows)
    plot_metric_bars(
        best_rows,
        "fvd_future",
        "FVD-style future distance, selected row per method",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_future_bar.png",
        lower_is_better=True,
    )
    plot_metric_bars_filtered(
        best_rows,
        "fvd_future",
        "FVD-style future distance, selected row per method\nzoomed to rows with FVD <= 120",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_future_bar_zoomed_under120.png",
        lower_is_better=True,
        max_value=120.0,
    )
    plot_metric_bars(
        best_rows,
        "future_psnr",
        "Future PSNR, selected row per method",
        "PSNR, higher is better",
        PLOT_DIR / "future_psnr_bar.png",
        lower_is_better=False,
    )
    plot_metric_bars(
        best_rows,
        "future_ssim",
        "Future SSIM, selected row per method",
        "SSIM, higher is better",
        PLOT_DIR / "future_ssim_bar.png",
        lower_is_better=False,
    )
    plot_metric_bars(
        best_rows,
        "sharpness_ratio",
        "Laplacian sharpness ratio, selected row per method",
        "generated/reference sharpness, higher is better",
        PLOT_DIR / "sharpness_ratio_bar.png",
        lower_is_better=False,
    )
    plot_metric_bars(
        best_rows,
        "motion_ratio",
        "Motion ratio, selected row per method",
        "generated/reference motion, higher is better for these runs",
        PLOT_DIR / "motion_ratio_bar.png",
        lower_is_better=False,
    )
    plot_metric_bars(
        best_rows,
        "temporal_delta_error_mae",
        "Temporal delta error, selected row per method",
        "MAE, lower is better",
        PLOT_DIR / "temporal_delta_error_mae_bar.png",
        lower_is_better=True,
    )
    plot_metric_bars(
        best_rows,
        "fft_high_frequency_ratio",
        "FFT high-frequency retention, selected row per method",
        "generated/reference high-frequency energy, higher is better",
        PLOT_DIR / "fft_high_frequency_ratio_bar.png",
        lower_is_better=False,
    )
    plot_metric_bars(
        sensitivity_rows,
        "action_sensitivity_rgb_mae",
        "Counterfactual action sensitivity, selected sensitivity row per method",
        "mean future RGB MAE correct vs counterfactual, higher means more action effect",
        PLOT_DIR / "action_sensitivity_rgb_mae_bar.png",
        lower_is_better=False,
    )
    plot_scatter(
        best_rows,
        "sharpness_ratio",
        "fvd_future",
        "FVD-style distance vs sharpness retention",
        "Laplacian sharpness ratio, higher is better",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_vs_sharpness.png",
    )
    plot_scatter_filtered(
        best_rows,
        "sharpness_ratio",
        "fvd_future",
        "FVD-style distance vs sharpness retention\nzoomed to rows with FVD <= 120",
        "Laplacian sharpness ratio, higher is better",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_vs_sharpness_zoomed_under120.png",
        max_y=120.0,
    )
    plot_scatter(
        best_rows,
        "motion_ratio",
        "fvd_future",
        "FVD-style distance vs motion retention",
        "Motion ratio, higher is better for these runs",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_vs_motion.png",
    )
    plot_scatter_filtered(
        best_rows,
        "motion_ratio",
        "fvd_future",
        "FVD-style distance vs motion retention\nzoomed to rows with FVD <= 120",
        "Motion ratio, higher is better for these runs",
        "FVD-style future distance, lower is better",
        PLOT_DIR / "fvd_vs_motion_zoomed_under120.png",
        max_y=120.0,
    )
    plot_scatter(
        sensitivity_rows,
        "sharpness_ratio",
        "action_sensitivity_rgb_mae",
        "Action sensitivity vs sharpness retention",
        "Laplacian sharpness ratio, higher is better",
        "Mean future RGB MAE correct vs counterfactual, higher means more action effect",
        PLOT_DIR / "action_sensitivity_vs_sharpness.png",
    )
    plot_heatmap(best_rows)
    write_report(best_rows, rows, sensitivity_rows)

    print(f"Wrote {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"Selected rows: {len(best_rows)}")
    print(f"Selected sensitivity rows: {len(sensitivity_rows)}")
    print(f"Candidate rows with FVD: {len(rows)}")


if __name__ == "__main__":
    main()
