# Concrete Crack Segmentation with Classical Image Processing

![CI](https://github.com/AbdulrahmanWOW/concrete-crack-segmentation/actions/workflows/ci.yml/badge.svg)

CSE281 Digital Image Processing — course project, Alamein International University.

A classical (no machine learning) image-processing pipeline that finds cracks in photos of
concrete surfaces, outputs a crack mask, measures the cracked area and decides whether the
surface needs inspection.

## Problem statement

**User:** a building inspection engineer who photographs concrete walls, columns and slabs
during a routine survey and has to decide which surfaces need a closer look.

**Output:** for each photo, a binary crack mask, the cracked area as a percentage of the
image, the number of crack segments, and a decision: *inspect* or *no action*.

**Cost of errors:** a missed crack (false negative) can leave structural damage unreported
and is the more expensive error. A false alarm (false positive, e.g. a stain, joint or
shadow marked as crack) costs an engineer's time to re-check the photo.

**Primary metric:** Intersection over Union (IoU) between the predicted and the
ground-truth crack mask, `IoU = |A ∩ B| / |A ∪ B|`. It penalises both missed crack pixels
and false-alarm pixels, and unlike pixel accuracy it is not inflated by the large
background area. Because cracks are only a few pixels wide, IoU is harsh on one-pixel
misalignment, so Dice/F1 is reported as a secondary metric.

## Input and output

| | Description |
|---|---|
| Input | RGB JPEG photo, 8-bit per channel, 4032 × 3024 px (to be verified) |
| Working format | Grayscale `uint8` 0–255, downscaled to `TBD` (chosen in the resolution study) |
| Ground truth | Binary crack mask, `uint8` 0/1 after binarisation |
| Output | Binary mask `uint8` 0/1, crack area %, segment count, decision |

## Dataset

Çağlar Fırat Özgenel, *Concrete Crack Segmentation Dataset*, Mendeley Data, 2019.
DOI: [10.17632/jwsn7tfbrp.1](https://doi.org/10.17632/jwsn7tfbrp.1). Licence: CC BY 4.0
(to be verified on the Mendeley page). 458 images with pixel-level crack masks, taken in
buildings at Middle East Technical University.

The raw data is not stored in this repository. See [`data/README.md`](data/README.md) to
download it. The development / test split is fixed in `data/splits/` (70 / 30, seed 42);
the test set is used once, for the final evaluation.

## Results

Development-set results until the final evaluation. All numbers are reproducible from
`experiments/log.csv`.

| Pipeline | Mean IoU | Std | Mean Dice | Time / image |
|---|---|---|---|---|
| Trivial baseline (threshold 127) | – | – | – | – |
| Simple baseline (Otsu + 3×3 opening) | – | – | – | – |
| Pipeline A | – | – | – | – |
| Pipeline B | – | – | – | – |
| Pipeline C | – | – | – | – |

## Setup

```bash
git clone https://github.com/AbdulrahmanWOW/concrete-crack-segmentation.git
cd concrete-crack-segmentation
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dataset as described in `data/README.md`.

## How to run

Commands marked *(planned)* are added as the project progresses.

```bash
pytest                                              # unit tests
flake8 src tests app                                # style check (max 95 chars)
python -m src.pipeline --config configs/baselines.yaml   # (planned) baselines on dev set
python -m src.evaluate --config configs/final.yaml --split test   # (planned) final test metrics
python -m src.run_pipeline path/to/image.jpg        # (planned) full pipeline on one image
python app/app.py                                   # (planned) demo
```

## Repository structure

```
├── configs/        one file per pipeline configuration
├── data/           download instructions, fixed split files (no raw data)
├── notebooks/      01_image_analysis.ipynb, 02_results.ipynb (import from src/)
├── src/            data, metrics, enhancement, restoration, frequency, segmentation,
│                   morphology, pipeline, evaluate, run_pipeline
├── tests/          unit tests (run by CI on every pull request)
├── experiments/    experiment log (CSV) and saved figures
├── app/            demo application
└── report/         report PDF and slides
```

## Team

| Member | GitHub | Role | Work package |
|---|---|---|---|
| Abdulrahman Alseady | @AbdulrahmanWOW | Tech Lead | Problem formulation, repository, integration, code review, final pipeline |
| Hazem | | Data and Baseline Engineer | Loading, split, ground truth, degradation, metrics, baselines |
| Fares | | Enhancement Engineer | Histograms, point transforms, equalisation, smoothing, sharpening |
| Fatma | | Restoration and Frequency Engineer | Noise models, restoration filters, DFT and frequency filters |
| Nour | | Segmentation Engineer | Edge, point and line detection, thresholding, Hough transform |
| Rahma | | Morphology and Evaluation Engineer | Morphology, measurement, test evaluation, failure analysis, demo |

Each member adds their GitHub username to this table in their first pull request.

## Workflow

- Every change goes through a feature branch `feature/<name>-<task>` and a pull request.
- Each pull request is reviewed and approved by a different member before merging.
- Every member commits from their own GitHub account.
- CI runs `flake8` and `pytest` on every pull request; it must pass before merging.
- Notebooks: one per person during analysis (`01a_…`, `01b_…`), outputs cleared before
  committing.

## Use of AI tools

AI coding assistants were used as a help, not as an author, as allowed by the course
rules. Every member can explain every line in their work package. Where and how such tools
were used is stated in the report.
