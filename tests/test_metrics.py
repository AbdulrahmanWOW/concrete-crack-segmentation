"""Hand-worked examples for the shared metrics."""

import numpy as np

from src.metrics import count_error, dice, iou, mse, psnr


def test_mse_hand_worked_array() -> None:
    # Differences are [0, 2, 4, 0]; MSE = (0 + 4 + 16 + 0) / 4 = 5.
    first = np.array([[0, 2], [4, 8]], dtype=np.uint8)
    second = np.array([[0, 4], [8, 8]], dtype=np.uint8)
    assert mse(first, second) == 5.0


def test_psnr_matches_hand_formula() -> None:
    # For MSE=2 and peak=255, PSNR = 10 log10(255^2 / 2).
    first = np.array([0, 2], dtype=np.uint8)
    second = np.array([0, 4], dtype=np.uint8)
    expected = 10.0 * np.log10((255.0**2) / 2.0)
    assert np.isclose(psnr(first, second), expected)


def test_psnr_is_infinite_for_identical_arrays() -> None:
    image = np.array([[0, 255]], dtype=np.uint8)
    assert psnr(image, image) == float("inf")


def test_iou_and_dice_hand_worked_masks() -> None:
    # Intersection has 2 pixels; union has 4. IoU=2/4 and Dice=4/6.
    pred = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    gt = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.uint8)
    assert iou(pred, gt) == 0.5
    assert np.isclose(dice(pred, gt), 2.0 / 3.0)


def test_empty_masks_have_perfect_overlap_by_convention() -> None:
    empty = np.zeros((2, 3), dtype=np.uint8)
    assert iou(empty, empty) == 1.0
    assert dice(empty, empty) == 1.0


def test_count_error_is_absolute() -> None:
    assert count_error(4, 7) == 3
