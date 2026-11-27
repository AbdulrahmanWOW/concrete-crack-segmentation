"""Shared image and segmentation metrics."""

from __future__ import annotations

import numpy as np


def _same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to arrays and require equal shapes."""
    first = np.asarray(a)
    second = np.asarray(b)
    if first.shape != second.shape:
        raise ValueError(f"Array shapes differ: {first.shape} != {second.shape}")
    if first.size == 0:
        raise ValueError("Metrics are undefined for empty arrays")
    return first, second


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean squared error using float64 arithmetic."""
    first, second = _same_shape(a, b)
    difference = first.astype(np.float64) - second.astype(np.float64)
    return float(np.mean(difference * difference))


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    """Return peak signal-to-noise ratio in dB; identical inputs yield infinity."""
    if peak <= 0:
        raise ValueError("peak must be positive")
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / error))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return binary intersection over union; two empty masks have IoU 1 by convention."""
    first, second = _same_shape(pred, gt)
    predicted = first.astype(bool)
    actual = second.astype(bool)
    union = np.count_nonzero(predicted | actual)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(predicted & actual) / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return binary Dice/F1; two empty masks have Dice 1 by convention."""
    first, second = _same_shape(pred, gt)
    predicted = first.astype(bool)
    actual = second.astype(bool)
    denominator = np.count_nonzero(predicted) + np.count_nonzero(actual)
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(predicted & actual) / denominator)


def count_error(n_pred: int, n_gt: int) -> int:
    """Return absolute error between predicted and reference object counts."""
    if n_pred < 0 or n_gt < 0:
        raise ValueError("Object counts cannot be negative")
    return abs(int(n_pred) - int(n_gt))
