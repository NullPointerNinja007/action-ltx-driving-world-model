#!/usr/bin/env python3
"""Extract and plot V4 training loss curves from Modal logs.

The V4 checkpoints were evaluated heavily, but their training losses are stored
mostly as JSON lines inside Modal log files. This script converts those logs to
auditable CSVs and plots the core loss/gate trajectories.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "benchmarks" / "final_h100_v4_campaign_seed231_all5" / "training_loss_curves"
PLOT_DIR = OUT_DIR / "plots"


LOG_SPECS = [
    # Pilot logs fill the early 0-1000 region for text main/conservative.
    (
        "v4_main_text",
        "V4 main + text",
        REPO_ROOT
        / "data/modal_logs/frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot_seed231_all5/training/"
        / "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_main_seed231_from_shifted_noaction_step003000_steps1000.log",
    ),
    (
        "v4_conservative_text",
        "V4 conservative + text",
        REPO_ROOT
        / "data/modal_logs/frame_temporal_bottleneck_full112_lowfreq_motion_v4_pilot_seed231_all5/training/"
        / "ltx2b_dist098_waymo24_full112_lowfreq_motion_v4_conservative_seed231_from_shifted_noaction_step003000_steps1000.log",
    ),
    # Final first-epoch campaign.
    (
        "v4_main_text",
        "V4 main + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_main_text_to7992.log",
    ),
    (
        "v4_conservative_text",
        "V4 conservative + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_conservative_text_to7992.log",
    ),
    (
        "v4_main_notext",
        "V4 main, no text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_main_notext_to7992.log",
    ),
    (
        "v4_conservative_notext",
        "V4 conservative, no text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_conservative_notext_to7992.log",
    ),
    (
        "v4_actionstrong_text",
        "V4 action-strong + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_actionstrong_text_to7992.log",
    ),
    (
        "v4_qualitystrict_text",
        "V4 quality-strict + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training/v4_qualitystrict_text_to7992.log",
    ),
    # Second epoch exists only for main/conservative text.
    (
        "v4_main_text",
        "V4 main + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training_second_epoch/v4_main_text_to15984.log",
    ),
    (
        "v4_conservative_text",
        "V4 conservative + text",
        REPO_ROOT / "data/modal_logs/final_h100_v4_campaign_seed231_all5/training_second_epoch/v4_conservative_text_to15984.log",
    ),
]


FIELDS = [
    "method_key",
    "method_label",
    "step",
    "source_log",
    "loss",
    "loss_diffusion",
    "loss_lowfreq_target",
    "loss_lowfreq_delta",
    "loss_hf_teacher",
    "loss_action_motion_aux",
    "loss_residual_norm",
    "loss_gate",
    "weighted_loss_diffusion",
    "weighted_loss_lowfreq_target",
    "weighted_loss_lowfreq_delta",
    "weighted_loss_hf_teacher",
    "weighted_loss_action_motion_aux",
    "mean_abs_gate",
    "max_abs_gate",
    "mean_abs_raw_gate",
    "max_abs_raw_gate",
    "grad_norm_action_encoder",
    "grad_norm_action_gate",
    "grad_norm_action_injector",
    "grad_norm_lora",
    "grad_norm_total",
    "sec_per_step",
    "elapsed_hours",
]


PLOT_METRICS = [
    ("loss", "Total weighted training loss", False),
    ("loss_diffusion", "Unweighted diffusion loss", False),
    ("loss_lowfreq_target", "Low-frequency target loss", False),
    ("loss_lowfreq_delta", "Low-frequency delta loss", False),
    ("loss_action_motion_aux", "Action-motion auxiliary loss", False),
    ("loss_hf_teacher", "High-frequency teacher loss", False),
    ("mean_abs_gate", "Mean absolute action gate", True),
    ("max_abs_gate", "Max absolute action gate", True),
    ("grad_norm_action_encoder", "Action encoder grad norm", True),
    ("grad_norm_action_gate", "Action gate grad norm", True),
]


def parse_json_line(line: str) -> dict[str, object] | None:
    text = line.strip()
    if not text.startswith("{") or '"step"' not in text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "step" not in value:
        return None
    if "loss" not in value and "loss_diffusion" not in value:
        return None
    return value


def to_float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_key, method_label, path in LOG_SPECS:
        if not path.exists():
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for line in path.read_text(errors="ignore").splitlines():
            parsed = parse_json_line(line)
            if parsed is None:
                continue
            row = {"method_key": method_key, "method_label": method_label, "source_log": rel}
            for field in FIELDS:
                if field in row:
                    continue
                row[field] = parsed.get(field, "")
            row["step"] = int(float(parsed["step"]))
            rows.append(row)
    # Dedupe by method/step. Prefer later logs so resumed runs overwrite pilot duplicates.
    keyed: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        keyed[(str(row["method_key"]), int(row["step"]))] = row
    return sorted(keyed.values(), key=lambda r: (str(r["method_key"]), int(r["step"])))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_dynamic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(points: list[tuple[int, float]], window: int = 7) -> list[tuple[int, float]]:
    if len(points) <= 2:
        return points
    smoothed = []
    half = max(1, window // 2)
    for idx, (step, _) in enumerate(points):
        lo = max(0, idx - half)
        hi = min(len(points), idx + half + 1)
        vals = [v for _, v in points[lo:hi] if not math.isnan(v)]
        smoothed.append((step, mean(vals) if vals else math.nan))
    return smoothed


def group_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method_key"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: int(r["step"]))
    return grouped


def plot_metric(rows: list[dict[str, object]], metric: str, title: str, log_y: bool) -> None:
    grouped = group_rows(rows)
    fig, ax = plt.subplots(figsize=(13, 7))
    for method_key, series in grouped.items():
        points = [
            (int(row["step"]), to_float(row.get(metric)))
            for row in series
            if not math.isnan(to_float(row.get(metric)))
        ]
        if not points:
            continue
        points = moving_average(points)
        xs, ys = zip(*points)
        label = str(series[0]["method_label"])
        ax.plot(xs, ys, marker="o", markersize=3.0, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(True, alpha=0.28)
    if log_y:
        ax.set_yscale("symlog", linthresh=1e-6)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(PLOT_DIR / f"{metric}_over_steps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    for ax, (metric, title, log_y) in zip(
        axes.ravel(),
        [
            ("loss", "Total weighted loss", False),
            ("loss_diffusion", "Diffusion loss", False),
            ("loss_action_motion_aux", "Action-motion aux loss", False),
            ("mean_abs_gate", "Mean absolute action gate", True),
            ("grad_norm_action_encoder", "Action encoder grad norm", True),
            ("grad_norm_action_gate", "Gate grad norm", True),
        ],
    ):
        for method_key, series in group_rows(rows).items():
            points = [
                (int(row["step"]), to_float(row.get(metric)))
                for row in series
                if not math.isnan(to_float(row.get(metric)))
            ]
            if not points:
                continue
            points = moving_average(points)
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.5, label=str(series[0]["method_label"]))
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.28)
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-6)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False)
    fig.savefig(PLOT_DIR / "v4_training_loss_summary_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for method_key, series in group_rows(rows).items():
        first = series[0]
        last = series[-1]
        record = {
            "method_key": method_key,
            "method_label": first["method_label"],
            "num_logged_points": len(series),
            "first_step": first["step"],
            "last_step": last["step"],
        }
        for metric, _, _ in PLOT_METRICS:
            vals = [to_float(row.get(metric)) for row in series if not math.isnan(to_float(row.get(metric)))]
            if vals:
                record[f"{metric}_first"] = vals[0]
                record[f"{metric}_last"] = vals[-1]
                record[f"{metric}_min"] = min(vals)
                record[f"{metric}_max"] = max(vals)
        summary.append(record)
    return summary


def write_readme(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    lines = [
        "# V4 Training Loss Curves",
        "",
        "Extracted from Modal training logs. These are training-objective diagnostics, not validation video metrics.",
        "",
        "## Important Interpretation",
        "",
        "- `v4_main_text`, `v4_conservative_text`, and `v4_qualitystrict_text` keep `mean_abs_gate` near `1e-6` to `1e-5`, effectively zero for action injection.",
        "- `v4_actionstrong_text` and the no-text variants grow gates much more, but those are also the variants that degraded sharpness/motion.",
        "- Therefore, the flat validation curves for the best-looking V4 text models are not surprising: the action residual path mostly stays closed.",
        "- The action branch can still show nonzero auxiliary/action losses and gradients, but if the gate stays closed, it has little causal influence on generated video.",
        "",
        "## Files",
        "",
        "- `v4_training_loss_rows.csv`: extracted per-log training rows.",
        "- `v4_training_loss_summary.csv`: first/last/min/max per method.",
        "- `plots/v4_training_loss_summary_grid.png`: main overview.",
        "- `plots/*_over_steps.png`: individual metric plots.",
        "",
        f"Extracted rows: `{len(rows)}`.",
        "",
        "## Method Summary",
        "",
        "| method | points | step range | final loss | final gate | final action aux | final diffusion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {method_label} | {num_logged_points} | {first_step}-{last_step} | {loss_last:.4g} | {mean_abs_gate_last:.4g} | {loss_action_motion_aux_last:.4g} | {loss_diffusion_last:.4g} |".format(
                **{
                    **item,
                    "loss_last": float(item.get("loss_last", math.nan)),
                    "mean_abs_gate_last": float(item.get("mean_abs_gate_last", math.nan)),
                    "loss_action_motion_aux_last": float(item.get("loss_action_motion_aux_last", math.nan)),
                    "loss_diffusion_last": float(item.get("loss_diffusion_last", math.nan)),
                }
            )
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_csv(OUT_DIR / "v4_training_loss_rows.csv", rows)
    summary = summarize(rows)
    write_dynamic_csv(OUT_DIR / "v4_training_loss_summary.csv", summary)
    for metric, title, log_y in PLOT_METRICS:
        plot_metric(rows, metric, title, log_y)
    plot_summary_grid(rows)
    write_readme(rows, summary)
    print(f"Wrote {len(rows)} loss rows to {OUT_DIR / 'v4_training_loss_rows.csv'}")
    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
