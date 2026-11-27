"""Organize manually downloaded Mendeley images and masks into canonical folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _normalised_stem(path: Path) -> str:
    """Remove common suffixes used to identify ground-truth masks."""
    stem = path.stem.casefold()
    suffixes = ("_groundtruth", "_ground_truth", "_mask", "_masks", "_gt", "_alpha")
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _index(directory: Path) -> dict[str, Path]:
    """Index supported image files recursively by normalized filename stem."""
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = _normalised_stem(path)
        if key in result:
            raise ValueError(f"Duplicate normalized filename {key!r} in {directory}")
        result[key] = path
    return result


def organize_dataset(
    images: Path,
    masks: Path,
    output: Path,
    report_path: Path,
) -> tuple[int, list[str], list[str]]:
    """Copy matched pairs and write a report for missing image/mask files."""
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError("Pass the extracted image and mask directories")
    image_files = _index(images)
    mask_files = _index(masks)
    unmatched_images = sorted(set(image_files) - set(mask_files))
    unmatched_masks = sorted(set(mask_files) - set(image_files))
    matched_keys = sorted(set(image_files) & set(mask_files))
    if not matched_keys:
        raise ValueError("No matching image/mask pairs were found")

    image_output = output / "images"
    mask_output = output / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    for key in matched_keys:
        image_path = image_files[key]
        mask_path = mask_files[key]
        shutil.copy2(image_path, image_output / f"{key}{image_path.suffix.lower()}")
        shutil.copy2(mask_path, mask_output / f"{key}{mask_path.suffix.lower()}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "status", "image_file", "mask_file"))
        for key in sorted(set(image_files) | set(mask_files)):
            image_path = image_files.get(key)
            mask_path = mask_files.get(key)
            if image_path and mask_path:
                status = "paired"
            elif image_path:
                status = "missing_mask"
            else:
                status = "missing_image"
            writer.writerow((key, status, image_path or "", mask_path or ""))
    return len(matched_keys), unmatched_images, unmatched_masks


def main() -> None:
    """Parse source folders and organize a downloaded dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Extracted directory containing concrete photos")
    parser.add_argument("--masks", type=Path, required=True,
                        help="Extracted directory containing binary crack masks")
    parser.add_argument("--output", type=Path, default=Path("data/raw"),
                        help="Canonical output directory (default: data/raw)")
    parser.add_argument("--report", type=Path, default=Path("data/pairing_report.csv"),
                        help="Pairing audit CSV path")
    args = parser.parse_args()
    count, missing_masks, missing_images = organize_dataset(
        args.images, args.masks, args.output, args.report
    )
    print(f"Organized {count} matched image-mask pairs under {args.output}")
    print(f"Pairing report: {args.report}")
    if missing_masks:
        print(f"Images without masks: {', '.join(missing_masks)}")
    if missing_images:
        print(f"Masks without images: {', '.join(missing_images)}")


if __name__ == "__main__":
    main()
