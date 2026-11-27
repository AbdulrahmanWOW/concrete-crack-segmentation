# Hazem's assigned issue checklist

There are five assigned issues in total. Update this list as each issue reaches
its completion criteria.

- [ ] #4 First PR and contribution table: README row is added on
  `feature/hazemoraby-team-table`; commit, push, PR, peer review, and review of a
  teammate's PR remain.
- [x] #3 Dataset setup: download instructions, organizer, licence/citation, and
  pairing audit are in place. The retrieved archive yielded 445 valid pairs;
  13 masks lack usable images, including one image entry that failed its RAR
  checksum. See `data/pairing_report.csv`.
- [x] #5 Shared loader and frozen split: loaders and seeded split generation are
  implemented; the committed manifests contain 311 development and 134 test
  pairs. Working scale is set to 1/4 in `configs/default.yaml`.
- [ ] #6 Metrics and hand-worked checks: metric functions and the requested
  examples are implemented in `src/metrics.py` and `tests/test_metrics.py`;
  test and style checks are still pending.
- [x] #7 Baselines and image study: both baselines ran on the development set;
  resolution and gray-level artifacts and the experiment log are saved under
  `experiments/`.
