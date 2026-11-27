# Dataset

## Source and citation

Özgenel, Çağlar Fırat (2019), *Concrete Crack Segmentation Dataset*, Mendeley
Data, Version 1, DOI: [10.17632/jwsn7tfbrp.1](https://doi.org/10.17632/jwsn7tfbrp.1).
The record describes 458 high-resolution concrete photographs and matching
black-and-white crack alpha maps. The listed licence is CC BY 4.0.

## Download and organize

1. Open the [Mendeley dataset record](https://data.mendeley.com/datasets/jwsn7tfbrp/1).
2. Download the dataset files. The current “Download All” archive contains a
   ZIP with `concreteCrackSegmentationDataset.rar` inside it; extract both
   archive layers using 7-Zip or WinRAR. Keep both archives outside Git.
3. The extracted folders are named `rgb/` and `BW/`. Run from the repository
   root:

   ```powershell
   python -m data.download --images "C:\path\to\images" `
     --masks "C:\path\to\masks" --output data/raw
   ```

   The organizer matches files by name (allowing common `_mask`, `_gt`, and
   `_alpha` suffixes), copies valid pairs to `data/raw/images/` and
   `data/raw/masks/`, and writes a row-by-row audit to
   `data/pairing_report.csv`. Unmatched files are reported and excluded.
4. Review `data/pairing_report.csv` before creating the fixed split. The
   organizer keeps only filename-matched pairs; resolve any source naming
   differences before treating unmatched examples as missing data.

The dataset landing page currently lists 458 image/mask examples and CC BY 4.0.
The project should retain this citation and licence with any redistributed
dataset material.

## Archive audit in this checkout

The downloaded archive lists 458 `BW` masks and 446 `rgb` entries. `rgb/593.JPG`
is truncated and fails its RAR checksum. Excluding it leaves 445 readable
image/mask pairs and 13 masks without a usable image. The local archive and
organized pairs are under the ignored `data/downloads/` and `data/raw/`
directories; neither is committed. The 445 pairs were decoded and dimension
checked. The 445 stored JPEG dimensions are: 425 at 4032 x 3024, 7 at 2448 x
3264, and 13 at 3264 x 2448 pixels.

Of the 425 landscape-stored photos, 166 carry EXIF Orientation 6, which rotates
them to 3024 x 4032 when displayed; their masks are stored in displayed
orientation. Visual samples confirm that applying EXIF orientation aligns the
image with its mask. `src.data.load_image` now uses OpenCV flags that honor EXIF
orientation, and the review loader applies the same orientation through Pillow.

## Ground-truth review (Nour)

Run `notebooks/01c_gt_check.ipynb` from the repository root after installing
requirements and organizing the dataset above. It selects 20 image/mask pairs
from the frozen development split (`seed=281`), visualizes the reference masks,
and writes a per-image crack-pixel fraction and approximate width estimate to
`experiments/ground_truth/sample_statistics.csv`. It also writes overlays,
crack/background intensity histograms, and per-image IoU for five fixed global
thresholds. The held-out test split is not used.

**20-pair review (seed 281):** the notebook ran on the frozen development split;
the held-out test split was not read. Overlays were inspected after applying
EXIF orientation, including full-resolution follow-up of flagged pairs. No gross
spatial shifts were visible. Image `104` has a visible secondary crack that is
not labeled. Images `183`, `453`, and `562` have masks substantially wider than
the visible crack along some sections or intersections; `453` and `562` show
the clearest broad-label regions. These are recorded as annotation concerns;
the source masks were not edited. Full-resolution views of the flagged pairs
are in `experiments/ground_truth/flagged_full_resolution_review.png`.

At the 1/8 working scale, crack-pixel fraction ranges from 0.86% to 4.62%
(median 1.16%). The pooled crack/background intensity-histogram overlap is
0.332. Mean IoU for global thresholds 64, 96, 127, 160, and 192 is 0.142,
0.336, 0.483, 0.211, and 0.025 respectively; threshold 127 wins on 16 of 20
images, but mean IoU is still only 0.483. The overlapping intensity classes
and per-image variation explain why no single grayscale cutoff separates all
cracks from textured or shadowed concrete. Keep global thresholding as a
baseline; evaluate adaptive/local thresholds and use morphology conservatively
so it does not erase thin branches or expand wide junction labels.

Approximate centerline widths at 1/8 scale have per-image medians from 2.0 to
4.47 pixels. The largest local estimate is 46.04 pixels and is driven by the
broad label regions noted above, so it should not be interpreted as a typical
physical crack width. The notebook writes overlays, per-image statistics, the
pooled histogram, and threshold IoUs under `experiments/ground_truth/`. Raw data
is not committed and must be downloaded separately to reproduce the review.
