# Experiments

- `log.csv`: one row per run (run ID, date, author, pipeline, stage, method, parameters,
  seed, time, split, metric). Baseline and image-study rows are appended by
  `src.baselines`.
- `figures/`: saved figures referenced in the report.
- `results/`: development-only per-image baseline scores, predicted-area audit,
  resolution/gray-level scores, and a compact comparison JPEG. The study uses the
  fixed threshold-127 baseline and identifies the lowest tested settings that retain
  nonzero IoU on its development example: 1/8 resolution and 4 gray levels.

All tuning uses the development split only.
