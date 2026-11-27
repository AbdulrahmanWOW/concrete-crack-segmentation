"""Development-set baselines and resolution/gray-level studies."""

from __future__ import annotations

import argparse
import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data import discover_pairs, load_image, load_mask, make_split, read_split
from src.metrics import dice, iou


def trivial_baseline(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Threshold raw grayscale pixels; dark pixels are crack candidates."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Baseline input must be grayscale uint8")
    return (gray <= threshold).astype(np.uint8)


def simple_baseline(gray: np.ndarray) -> np.ndarray:
    """Apply inverted Otsu thresholding and one 3x3 morphological opening."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Baseline input must be grayscale uint8")
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (opened > 0).astype(np.uint8)


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    """Return population mean/std and best/worst values."""
    return float(np.mean(values)), float(np.std(values)), max(values), min(values)


def evaluate_baselines(
    pairs: list[tuple[str, str]],
    output_csv: Path,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> list[dict[str, str | float | int]]:
    """Evaluate both baselines, writing per-image and aggregate development metrics."""
    rows: list[dict[str, str | float | int]] = []
    working_size = size if size is not None else f"scale={scale}"
    per_method: dict[str, list[float]] = {"trivial_127": [], "otsu_open_3x3": []}
    for image_path, mask_path in pairs:
        gray = load_image(image_path, size=size, scale=scale)
        truth = load_mask(mask_path, size=(gray.shape[1], gray.shape[0]))
        if gray.shape != truth.shape:
            raise ValueError(f"Image/mask dimensions differ for {image_path}")
        for method, function in (
            ("trivial_127", trivial_baseline),
            ("otsu_open_3x3", simple_baseline),
        ):
            start = time.perf_counter()
            prediction = function(gray)
            elapsed = time.perf_counter() - start
            score = iou(prediction, truth)
            per_method[method].append(score)
            rows.append({
                "image": image_path,
                "method": method,
                "parameters": (
                    "threshold=127; dark-pixel inversion"
                    if method == "trivial_127"
                    else "Otsu inverse threshold; 3x3 rectangular opening"
                ) + f"; working_size={working_size}",
                "iou": score,
                "dice": dice(prediction, truth),
                "processing_time_seconds": elapsed,
                "split": "dev",
            })

    for method, scores in per_method.items():
        mean, std, best, worst = _summary(scores)
        rows.append({
            "image": "__summary__",
            "method": method,
            "iou": mean,
            "dice": float("nan"),
            "processing_time_seconds": float("nan"),
            "split": "dev",
            "iou_std": std,
            "best_iou": best,
            "worst_iou": worst,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def append_experiment_log(
    records: list[dict[str, str | float | int]],
    log_path: Path,
    split_seed: int,
) -> None:
    """Append measured development runs to the shared experiment log."""
    fields = [
        "run_id", "date", "author", "pipeline", "stage", "method",
        "parameters", "random_seed", "processing_time_seconds", "split",
        "metric_name", "metric_value",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        for record in records:
            if record.get("image") == "__summary__":
                continue
            method = str(record.get("method", record.get("setting", "")))
            writer.writerow({
                "run_id": uuid.uuid4().hex[:12],
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "author": "HazemOraby",
                "pipeline": record.get("study", "baseline"),
                "stage": "segmentation",
                "method": method,
                "parameters": record.get("parameters", ""),
                "random_seed": split_seed,
                "processing_time_seconds": record.get(
                    "processing_time_seconds", ""
                ),
                "split": "dev",
                "metric_name": "iou",
                "metric_value": record.get("iou", ""),
            })


def save_image_studies(
    pair: tuple[str, str],
    output_dir: Path,
    factors: tuple[int, ...] = (1, 2, 4, 8),
    gray_levels: tuple[int, ...] = (256, 64, 16, 4),
) -> list[dict[str, str | float | int]]:
    """Save resolution and gray-level variants and their IoU on one dev image."""
    image_path, mask_path = pair
    gray = load_image(image_path)
    truth = load_mask(mask_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "source_gray.png"), gray)
    cv2.imwrite(str(output_dir / "ground_truth.png"), truth * 255)
    records: list[dict[str, str | float | int]] = []

    for factor in factors:
        if factor < 1:
            raise ValueError("Resolution reduction factors must be at least 1")
        start = time.perf_counter()
        height = max(1, gray.shape[0] // factor)
        width = max(1, gray.shape[1] // factor)
        reduced = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(reduced, (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        prediction = simple_baseline(restored)
        records.append({
            "image": image_path,
            "study": "resolution",
            "setting": f"1/{factor}",
            "parameters": f"scale=1/{factor}; interpolation=area+nearest",
            "width": width,
            "height": height,
            "iou": iou(prediction, truth),
            "processing_time_seconds": time.perf_counter() - start,
        })
        cv2.imwrite(str(output_dir / f"resolution_1_{factor}.png"), reduced)
        cv2.imwrite(str(output_dir / f"resolution_1_{factor}_restored.png"), restored)

    for levels in gray_levels:
        if levels < 2 or levels > 256:
            raise ValueError("Gray-level count must be in [2, 256]")
        start = time.perf_counter()
        step = 256.0 / levels
        quantized = np.floor(gray.astype(np.float64) / step) * step
        quantized = np.clip(np.rint(quantized), 0, 255).astype(np.uint8)
        prediction = simple_baseline(quantized)
        records.append({
            "image": image_path,
            "study": "gray_levels",
            "setting": levels,
            "parameters": f"gray_levels={levels}",
            "width": gray.shape[1],
            "height": gray.shape[0],
            "iou": iou(prediction, truth),
            "processing_time_seconds": time.perf_counter() - start,
        })
        cv2.imwrite(str(output_dir / f"gray_levels_{levels}.png"), quantized)

    return records


def run_studies(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    split_seed: int = 42,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
    study_only: bool = False,
    log_path: Path | None = None,
) -> list[dict[str, str | float | int]]:
    """Run the baseline evaluation and resolution/quantization study."""
    if not pairs:
        raise ValueError("Development split contains no pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    study_rows = save_image_studies(pairs[0], output_dir / "image_study")
    if log_path is None:
        log_path = output_dir.parent / "log.csv"
    append_experiment_log(study_rows, log_path, split_seed)
    study_path = output_dir / "resolution_gray_level_study.csv"
    with study_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=study_rows[0].keys())
        writer.writeheader()
        writer.writerows(study_rows)
    if study_only:
        return study_rows

    baseline_rows = evaluate_baselines(
        pairs,
        output_dir / "baseline_per_image.csv",
        size=size,
        scale=scale,
    )
    append_experiment_log(baseline_rows, log_path, split_seed)
    return baseline_rows + study_rows


def main() -> None:
    """Discover dataset pairs, load or create the split, and run dev-only studies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--study-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    split_config = config["split"]
    experiment_config = config["experiments"]
    images = args.images or Path(data_config["raw_images"])
    masks = args.masks or Path(data_config["raw_masks"])
    split_dir = Path(data_config["splits_dir"])
    split_path = args.split or split_dir / "dev.txt"
    output_dir = args.output_dir or Path(experiment_config["results_dir"])
    seed = split_config["seed"] if args.seed is None else args.seed
    test_frac = (
        split_config["test_fraction"] if args.test_frac is None else args.test_frac
    )
    scale = data_config["working_scale"] if args.scale is None else args.scale
    log_path = Path(experiment_config["log_file"])
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if args.width is not None and args.scale is not None:
        parser.error("Use either --width/--height or --scale")
    size = None if args.width is None else (args.width, args.height)
    if args.width is not None:
        scale = 1.0

    if split_path.is_file():
        pairs = read_split(split_path, data_root=Path.cwd())
    else:
        all_pairs = discover_pairs(images, masks)
        pairs, _ = make_split(
            all_pairs, split_dir, seed=seed, test_frac=test_frac
        )
    rows = run_studies(
        pairs,
        output_dir,
        split_seed=seed,
        size=size,
        scale=scale,
        study_only=args.study_only,
        log_path=log_path,
    )
    for row in rows:
        if row.get("image") == "__summary__":
            print(
                f"{row['method']}: mean IoU={row['iou']:.4f}, "
                f"std={row['iou_std']:.4f}, best={row['best_iou']:.4f}, "
                f"worst={row['worst_iou']:.4f}"
            )
    print(f"Development artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
