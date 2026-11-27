"""Image loading, reproducible data splits, and controlled degradation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


Pair = tuple[str, str]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 without silently wrapping its values."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger) and image.min() < 0:
            raise ValueError("Image contains negative intensity values")
        if image.size and image.max() <= 255:
            return image.astype(np.uint8)
        max_value = np.iinfo(image.dtype).max
        if max_value <= 255:
            return image.astype(np.uint8)
        return np.rint(image.astype(np.float64) * (255.0 / max_value)).astype(
            np.uint8
        )
    if np.issubdtype(image.dtype, np.floating):
        values = image.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Image contains NaN or infinite values")
        if values.size and values.min() >= 0 and values.max() <= 1:
            values *= 255.0
        if values.size and (values.min() < 0 or values.max() > 255):
            raise ValueError("Floating-point image values must be in [0, 1] or [0, 255]")
        return np.rint(values).astype(np.uint8)
    raise TypeError(f"Unsupported image dtype: {image.dtype}")


def _read(path: str | Path) -> np.ndarray:
    """Read an image and raise a useful error when the path cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _resize(
    image: np.ndarray,
    size: tuple[int, int] | None,
    method: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Resize to a fixed (width, height) or by a preserved-aspect scale factor."""
    if size is None and scale == 1.0:
        return image
    if size is not None and scale != 1.0:
        raise ValueError("Pass either size or scale, not both")
    if size is None:
        if scale <= 0:
            raise ValueError("Resize scale must be positive")
        height = max(1, round(image.shape[0] * scale))
        width = max(1, round(image.shape[1] * scale))
    else:
        width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Resize width and height must be positive")
    return cv2.resize(image, (width, height), interpolation=method)


def load_image(
    path: str | Path,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Load grayscale uint8 [0, 255], resizing with INTER_AREA."""
    image = _to_uint8(_read(path))
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Expected grayscale, BGR, or BGRA image; got {image.shape}")
    return np.ascontiguousarray(_resize(gray, size, cv2.INTER_AREA, scale))


def load_mask(
    path: str | Path,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Load binary uint8 {0, 1}, resizing with INTER_NEAREST."""
    mask = _to_uint8(_read(path))
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif mask.ndim == 3 and mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    elif mask.ndim != 2:
        raise ValueError(f"Expected grayscale, BGR, or BGRA mask; got {mask.shape}")
    mask = _resize(mask, size, cv2.INTER_NEAREST, scale)
    return np.ascontiguousarray((mask > 127).astype(np.uint8))


def _normalised_stem(path: Path) -> str:
    """Normalize common ground-truth suffixes before image-mask matching."""
    stem = path.stem.casefold()
    for suffix in ("_groundtruth", "_ground_truth", "_mask", "_masks", "_gt", "_alpha"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def discover_pairs(image_dir: str | Path, mask_dir: str | Path) -> list[Pair]:
    """Find image/mask pairs by filename stem and report missing or duplicate files."""
    image_root = Path(image_dir).resolve()
    mask_root = Path(mask_dir).resolve()
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Both image and mask directories must exist")

    def index_files(root: Path) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            key = _normalised_stem(path)
            if key in indexed:
                raise ValueError(f"Duplicate normalized filename {key!r} in {root}")
            indexed[key] = path
        return indexed

    images = index_files(image_root)
    masks = index_files(mask_root)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask files do not match. "
            f"Images without masks: {missing_masks[:10]}; "
            f"masks without images: {missing_images[:10]}"
        )
    if not images:
        raise ValueError("No supported image files were found")
    return [(str(images[key]), str(masks[key])) for key in sorted(images)]


def make_split(
    pairs: Sequence[Pair],
    output_dir: str | Path,
    seed: int = 42,
    test_frac: float = 0.30,
) -> tuple[list[Pair], list[Pair]]:
    """Write a deterministic, frozen development/test split as tab-separated paths."""
    if len(pairs) < 2:
        raise ValueError("At least two image-mask pairs are required to split")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be strictly between 0 and 1")
    ordered = sorted((str(Path(image)), str(Path(mask))) for image, mask in pairs)
    if len({image for image, _ in ordered}) != len(ordered):
        raise ValueError("Image paths must be unique")
    n_test = min(len(ordered) - 1, max(1, round(len(ordered) * test_frac)))
    rng = np.random.default_rng(seed)
    test_indices = set(rng.permutation(len(ordered))[:n_test].tolist())
    dev = [pair for index, pair in enumerate(ordered) if index not in test_indices]
    test = [pair for index, pair in enumerate(ordered) if index in test_indices]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_split(destination / "dev.txt", dev)
    _write_split(destination / "test.txt", test)
    return dev, test


def _write_split(path: Path, pairs: Iterable[Pair]) -> None:
    """Write a split with paths relative to the current working directory when possible."""
    rows = []
    for image, mask in pairs:
        image_path = Path(image)
        mask_path = Path(mask)
        try:
            image_value = image_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            image_value = image_path
        try:
            mask_value = mask_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            mask_value = mask_path
        rows.append(f"{image_value.as_posix()}\t{mask_value.as_posix()}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_split(path: str | Path, data_root: str | Path = ".") -> list[Pair]:
    """Read a split manifest, resolving relative paths against ``data_root``."""
    root = Path(data_root)
    pairs: list[Pair] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"Expected two tab-separated paths on line {line_number}")
        image, mask = (Path(value) for value in fields)
        pairs.append((str(image if image.is_absolute() else root / image),
                      str(mask if mask.is_absolute() else root / mask)))
    if not pairs:
        raise ValueError(f"Split manifest is empty: {path}")
    return pairs


def main() -> None:
    """Create frozen development/test manifests from image and mask folders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-frac", type=float)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    split_config = config["split"]
    images = args.images or Path(data_config["raw_images"])
    masks = args.masks or Path(data_config["raw_masks"])
    split_dir = args.split_dir or Path(data_config["splits_dir"])
    seed = split_config["seed"] if args.seed is None else args.seed
    test_frac = (
        split_config["test_fraction"] if args.test_frac is None else args.test_frac
    )
    pairs = discover_pairs(images, masks)
    dev, test = make_split(pairs, split_dir, seed, test_frac)
    print(f"Wrote {len(dev)} development and {len(test)} test pairs to {split_dir}")


if __name__ == "__main__":
    main()
