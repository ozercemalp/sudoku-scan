# SudokuScan

### Install Python dependencies

Make sure Python 3.10+ is installed:

```bash
python3 --version
```

On Ubuntu/Debian, install virtual environment support if needed:

```bash
sudo apt-get update
sudo apt-get install python3-venv
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

### Install Tesseract OCR

Tesseract must be installed separately from the Python dependencies.

**Ubuntu/Debian**

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Fedora**

```bash
sudo dnf install tesseract tesseract-langpack-eng
```

**Arch Linux**

```bash
sudo pacman -S tesseract tesseract-data-eng
```

### Verify the installation

Check that Tesseract is installed:

```bash
tesseract --version
```

Verify that English language data is available:

```bash
tesseract --list-langs
```

The output should include `eng`.