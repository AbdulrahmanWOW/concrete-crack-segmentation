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

The retrieved record describes 458 photos and 458 alpha maps. The downloaded
archive listed 458 `BW` masks and 446 `rgb` entries. `rgb/593.JPG` failed its
RAR checksum during extraction; the remaining extracted data yielded 445
matched pairs and 13 masks without a usable image. Their IDs are recorded in
`data/pairing_report.csv`. Only the 445 matched pairs were included in the
development/test manifests. Recheck the archive before reporting this dataset
as having 458 usable pairs.

The 445 readable photos have three raw sizes: 425 are 4032 x 3024, 13 are
2448 x 3264, and 7 are 3264 x 2448 pixels. The loader's scale option preserves
each photo's aspect ratio.
