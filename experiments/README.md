# Experiments

- `log.csv`: one row per run (run ID, date, author, pipeline, stage, method, parameters,
  seed, time, split, metric). Baseline and image-study rows are appended by
  `src.baselines`.
- `figures/`: saved figures referenced in the report.
- `results/`: development-only per-image baseline scores, predicted-area audit,
  resolution/gray-level scores, and a compact comparison JPEG. Resolution variants
  include threshold-127 IoU; gray-level variants include PSNR against the original
  256-level image because threshold-127 IoU is invariant under these quantization
  settings.

All tuning uses the development split only.
