# Hazem's assigned issue checklist

There are five assigned issues in total. Update this list as each issue reaches
its completion criteria.

- [ ] #4 First PR and contribution table: README row is included; commit `47cdf56` is pushed on `feature/hazemoraby-team-table`. PR creation and peer review, plus reviewing a teammate's PR, remain.
- [x] #3 Dataset setup: download instructions, organizer, licence/citation, and
  pairing audit are in place. The retrieved archive yielded 445 valid pairs;
  13 masks lack usable images, including one image entry that failed its RAR
  checksum. See `data/pairing_report.csv`.
- [x] #5 Shared loader and frozen split: loaders and seeded split generation are
  implemented; the committed manifests contain 311 development and 134 test
  pairs. Working scale is set to 1/4 in `configs/default.yaml`.
- [x] #6 Metrics and hand-worked checks: implemented with hand-worked examples. All 17 tests pass, and Python files in `src/`, `tests/`, and `data/` meet the 95-character limit.
- [x] #7 Baselines and image study: both baselines ran on the development set;
  resolution and gray-level artifacts and the experiment log are saved under
  `experiments/`.
