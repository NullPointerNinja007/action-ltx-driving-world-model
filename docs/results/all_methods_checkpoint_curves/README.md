# V4 And All-Method Checkpoint Curves

Generated from existing benchmark CSVs. No videos were decoded and no metrics were recomputed.

## Main Outputs

- `metric_plots/v4_methods_checkpoint_grid.png`: all six V4 variants over checkpoints.
- `metric_plots/all_methods_checkpoint_grid.png`: major methods tried so far over checkpoints.
- `metric_plots/v4_<metric>_over_checkpoints.png`: per-metric V4 plots.
- `metric_plots/all_methods_<metric>_over_checkpoints.png`: per-metric all-method plots.
- `v4_plot_rows.csv`: exact V4 rows used.
- `all_methods_plot_rows.csv`: exact all-method rows used.

## Notes

- FVD-style and temporal-delta plots invert the y-axis so upward visual movement is better.
- For gate-sweep methods, the all-method plot uses the normal `gate=1.0` operating point to avoid duplicating each checkpoint by inference gate.
- All comparisons are based on the fixed five validation clips unless the source benchmark says otherwise.

V4 rows plotted: `98`.
All-method rows plotted: `257`.
