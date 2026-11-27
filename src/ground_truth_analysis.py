"""Ground-truth overlays and intensity analysis for the concrete crack dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt


Pair = tuple[str, str]


def sample_pairs(
    split_path: str | Path,
    count: int = 20,
    seed: int = 281,
    data_root: str | Path = ".",
) -> list[Pair]:
    """Select a reproducible sample from a split manifest."""
    if count < 1:
        raise ValueError("count must be positive")
    root = Path(data_root)
    pairs: list[Pair] = []
    for line_number, line in enumerate(
        Path(split_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"Expected two tab-separated paths on line {line_number}")
        paths = [Path(field) for field in fields]
        pairs.append(
            tuple(str(path if path.is_absolute() else root / path) for path in paths)
        )
    if len(pairs) < count:
        raise ValueError(f"Need at least {count} development pairs; found {len(pairs)}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(pairs), size=count, replace=False))
    return [pairs[int(index)] for index in indices]


def _scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("Resize scale must be positive")
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def pair_arrays(
    image_path: str | Path,
    mask_path: str | Path,
    scale: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an EXIF-oriented grayscale photo and an aligned binary mask."""
    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source).convert("L")
        original_size = oriented.size
        size = _scaled_size(oriented.size, scale)
        image = np.asarray(oriented.resize(size, Image.Resampling.BOX))
    with Image.open(mask_path) as source:
        mask_image = source.convert("L")
        if mask_image.size != original_size:
            raise ValueError(
                f"Image/mask source dimensions differ for {image_path}: "
                f"{original_size} != {mask_image.size}"
            )
        mask = np.asarray(
            mask_image.resize(size, Image.Resampling.NEAREST)
        ) > 127
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {image_path}: "
            f"{image.shape} != {mask.shape}"
        )
    return image.astype(np.uint8, copy=False), mask


def _thin(binary: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a one-pixel centerline using Zhang-Suen steps."""
    skeleton = np.asarray(binary, dtype=bool).copy()
    changed = True
    while changed:
        changed = False
        padded = np.pad(skeleton, 1)
        neighbors = (
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, 2:],
            padded[2:, 2:],
            padded[2:, 1:-1],
            padded[2:, :-2],
            padded[1:-1, :-2],
            padded[:-2, :-2],
        )
        neighbor_count = sum(item.astype(np.uint8) for item in neighbors)
        transitions = sum(
            ~neighbors[index] & neighbors[(index + 1) % 8]
            for index in range(8)
        )
        for step in (0, 1):
            remove = (
                skeleton
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
            )
            if step == 0:
                remove &= ~(neighbors[0] & neighbors[2] & neighbors[4])
                remove &= ~(neighbors[2] & neighbors[4] & neighbors[6])
            else:
                remove &= ~(neighbors[0] & neighbors[2] & neighbors[6])
                remove &= ~(neighbors[0] & neighbors[4] & neighbors[6])
            if remove.any():
                skeleton[remove] = False
                changed = True
            if step == 0:
                padded = np.pad(skeleton, 1)
                neighbors = (
                    padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                    padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                    padded[1:-1, :-2], padded[:-2, :-2],
                )
                neighbor_count = sum(item.astype(np.uint8) for item in neighbors)
                transitions = sum(
                    ~neighbors[index] & neighbors[(index + 1) % 8]
                    for index in range(8)
                )
    return skeleton


def crack_widths(mask: np.ndarray) -> np.ndarray:
    """Estimate local crack width in working-scale pixels."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if not binary.any():
        return np.empty(0, dtype=np.float32)
    distance = distance_transform_edt(binary)
    return (2.0 * distance[_thin(binary)]).astype(np.float32)


def pair_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    image_id: str,
) -> dict[str, float | int | str]:
    """Return crack fraction and an approximate width range for one pair."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must have identical shapes")
    widths = crack_widths(mask)
    return {
        "image_id": image_id,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "crack_pixel_fraction": float(mask.mean()),
        "crack_width_min_px": float(widths.min()) if widths.size else 0.0,
        "crack_width_median_px": float(np.median(widths)) if widths.size else 0.0,
        "crack_width_max_px": float(widths.max()) if widths.size else 0.0,
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB visualization with reference crack pixels in red."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must have identical shapes")
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    rgb[mask.astype(bool)] = (255, 40, 40)
    return rgb


def intensity_samples(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale intensities for crack and background pixels."""
    if image.shape != mask.shape:
        raise ValueError("image and mask must have identical shapes")
    return image[mask.astype(bool)], image[~mask.astype(bool)]


def global_threshold_iou(image: np.ndarray, mask: np.ndarray, threshold: int) -> float:
    """Score a dark-pixel global threshold against the reference mask."""
    prediction = image < threshold
    truth = mask.astype(bool)
    union = np.logical_or(prediction, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, truth).sum() / union)


def summarize_sample(pairs: Sequence[Pair]) -> list[dict[str, float | int | str]]:
    """Load and summarize all selected image/mask pairs."""
    rows = []
    for image_path, mask_path in pairs:
        image, mask = pair_arrays(image_path, mask_path)
        rows.append(pair_statistics(image, mask, Path(image_path).name))
    return rows
