# PokeAI Final Run Guide

This package contains the fine-tuned 1,025-species model and the tested localhost API.

## Install

Use Python 3.10+ in a fresh virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA GPU, install the matching PyTorch/TorchVision build from pytorch.org before installing the remaining requirements.

## Start the API

From the project root:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

The API loads `models/species_resnet18_robust_best.pt` and exposes 1,025 species labels.

## Test the API

```bash
python test_api.py
curl http://127.0.0.1:8000/model/info
curl -X POST "http://127.0.0.1:8000/predict?top_k=5" \
  -F "file=@pikachu.png"
```

The model is a classifier, not a general full-screen detector. It can return a confident wrong label, especially for heavily cropped, occluded, low-resolution, or out-of-domain images.

## Reports

See `MODEL_IMPROVEMENT_REPORT.md` and `reports/ROBUSTNESS_EVALUATION.md` for measured accuracy and tricky-case results. Windows desktop overlay, live game-screen validation, and a connected dashboard are not included as completed features.
