# SudokuScan

COMP 482 Computer Vision project for reading printed Sudoku givens from a photo.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use `python` instead of `python3` and activate with
`.venv\Scripts\Activate.ps1` in PowerShell.

## Tesseract

Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

macOS

```bash
brew install tesseract
```

Windows

Install [Tesseract from UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
and add its installation folder to `PATH`.

## Run

Single image

```bash
python sudoku_scan.py --input data/personal/images/personal_01.jpg --outdir out
```

Batch with affine and homography OCR comparisons

```bash
python sudoku_scan.py --batch data/personal/images --outdir results/personal --compare
```

## Test and evaluate

```bash
pip install pytest
python -m pytest
python evaluate.py --annotations data/personal/annotations --results results/personal --outdir results/personal_metrics
```
