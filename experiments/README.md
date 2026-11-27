# Experiments

- `log.csv`: one row per run (run ID, date, author, pipeline, stage, method, parameters,
  seed, time, split, metric). Baseline and image-study rows are appended by
  `src.baselines`.
- `figures/`: saved figures referenced in the report.
- `results/`: development-only per-image baseline scores and resolution/gray-level
  study images and tables. Review the saved variants visually before stating the
  lowest usable resolution or gray-level count.

All tuning uses the development split only.
