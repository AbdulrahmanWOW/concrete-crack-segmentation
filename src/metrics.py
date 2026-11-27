"""Evaluation metrics, written once and used everywhere.

Owner: Data and Baseline Engineer.

- ``mse``, ``psnr`` (PSNR = 10 log10(255^2 / MSE)) for restoration quality.
- ``iou`` (|A n B| / |A u B|) and ``dice`` for crack masks.
- ``count_error`` for the number of crack segments.

Each function is checked in ``tests/test_metrics.py`` against values worked by hand.
"""
