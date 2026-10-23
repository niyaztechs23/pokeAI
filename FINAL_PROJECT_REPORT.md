# PokeAI Final Project Report

**Project:** PokeAI Pokémon Recognition System  
**Report status:** Final verified sandbox deliverable  
**Model:** Fine-tuned ResNet18, 1,025 base-species outputs  
**API:** Local FastAPI service  
**Original archive:** `PokeAI_ResNet.zip`

## 1. Executive summary

PokeAI is a Pokémon image-classification project upgraded from the supplied ResNet18 baseline. The original project was preserved and audited. A new species-level model was trained for all 1,025 National Pokédex entries, including a repaired Diglett source folder and explicit mappings from alternate-form folders to base species.

The final selected model is `models/species_resnet18_robust_best.pt`. On a controlled holdout containing one reserved official-artwork image for each of the 1,025 species, it achieved **90.24% top-1**, **94.73% top-3**, and **96.20% top-5** accuracy.

This is a measured classifier, not a guaranteed universal recognizer. It can still fail on arbitrary screenshots, sprites, unusual crops, occlusions, low-resolution images, and out-of-domain inputs. The supplied Pikachu sample was still misclassified, which is recorded rather than hidden.

## 2. Delivery status checklist

- [x] Original project backed up
- [x] Model and dataset audited
- [x] Base-species coverage verified
- [x] Baseline accuracy measured
- [x] Recognition model improved and evaluated
- [ ] Live game-screen detection tested end-to-end
- [ ] Windows desktop application tested
- [ ] API and dashboard connected — the API is implemented and tested; no dashboard is connected
- [x] Installation instructions and project test report completed for the packaged project

## 3. Original project audit

The original archive is preserved separately from the working copy. The original model was a ResNet18 classifier with 1,024 output labels.

The prepared dataset contained 1,024 class folders with 5,120 training images and 1,024 validation images. The original validation result was:

- Top-1: **71.6797%** — 734/1,024
- Top-3: **83.9844%** — 860/1,024
- Top-5: **86.8164%** — 889/1,024

The original prepared training data had five exact duplicate groups and no exact train/validation hash overlap. The duplicate groups were documented in `DATASET_REPORT.md` and `reports/full_model_evaluation.json`.

## 4. Base-species coverage

A verified National Pokédex manifest was created for IDs 1–1025. The raw export originally had one empty Diglett folder. Six validated Diglett assets were obtained through the project’s existing public PokeAPI workflow. No synthetic images were created.

Several raw folders represent alternate forms. They were explicitly mapped to base species, for example:

- `Aegislash-shield` → Aegislash
- `Deoxys-normal` → Deoxys
- `Palafin-zero` → Palafin
- `Nidoran-f` → Nidoran♀
- `Nidoran-m` → Nidoran♂
- `Minior-red-meteor` → Minior

The resulting manifest reports **1,025 ready species**, with each species linked to a real source folder and real source images.

## 5. Model improvement

The final model was produced in two stages:

1. A 1,025-class species candidate was trained from the original learned ResNet18 backbone using five source images per class and reserving `official.png` for validation.
2. The candidate was fine-tuned with stronger robustness augmentation, including random resized crops, horizontal flips, color jitter, grayscale variation, and a lower learning rate.

The robust fine-tuning checkpoint was selected by validation accuracy and saved as:

```text
models/species_resnet18_robust_best.pt
```

The original checkpoints remain available and were not overwritten.

## 6. Final model evaluation

The final holdout contains 1,025 reserved `official.png` images, one per species. This is a controlled artwork-domain evaluation, not an independent real-game benchmark.

| Metric | Result |
|---|---:|
| Top-1 | **90.24%** — 925/1,025 |
| Top-3 | **94.73%** — 971/1,025 |
| Top-5 | **96.20%** — 986/1,025 |

The detailed results are in:

- `reports/SPECIES_ROBUST_EVALUATION.md`
- `reports/species_robust_evaluation.json`
- `models/robust_finetune_history.json`

The older 83.22% species candidate and the original 71.68% baseline used different validation setups, so those numbers indicate progress but are not a strict apples-to-apples benchmark.

## 7. Tricky-case evaluation

The final model was tested against deterministic transformations of all 1,025 held-out images:

| Condition | Top-1 accuracy |
|---|---:|
| Clean | **90.24%** |
| Darkened | **88.39%** |
| Horizontally flipped | **89.95%** |
| Grayscale | **71.51%** |
| Brightened | **77.56%** |
| Center crop | **55.22%** |
| Central occlusion | **49.66%** |
| Extreme 32×32 low resolution | **0.78%** |

The model handles normal artwork, flips, and moderate darkening relatively well. It remains weak when the Pokémon is heavily cropped, blocked, or reduced to extremely low resolution.

Full results are in `reports/ROBUSTNESS_EVALUATION.md` and `reports/robustness_evaluation.json`.

## 8. API implementation and test

The package includes a localhost-only FastAPI service:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Implemented endpoints:

- `GET /health`
- `GET /model/info`
- `POST /predict`

The API accepts multipart image uploads, supports `top_k`, supports an optional explicit threshold, limits uploads to 10 MiB, and reports 1,025 model classes.

The API smoke tests pass through `test_api.py`.

The supplied `pikachu.png` was tested through the final API. It returned HTTP 200, but the model predicted Diglett as top-1 and did not place Pikachu in the top five. This confirms that a model score is not a correctness guarantee and that the current clean holdout does not represent arbitrary image performance.

The recorded result is in `reports/api_sample_robust_test.json`.

## 9. How to use the package

From the extracted project directory:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Run the included verification:

```bash
python test_api.py
```

A complete guide is also provided in `FINAL_RUN_GUIDE.md`.

## 10. Known limitations and next steps

The project does not yet include a completed Windows desktop overlay, validated end-to-end live game-screen detection, object localization, or a connected dashboard. The current service classifies one supplied image at a time.

The highest-impact next step is a larger, licensed, domain-matched dataset containing many screenshots, sprites, poses, backgrounds, scales, occlusions, and non-Pokémon images. The model also needs calibrated unknown rejection and a localization stage before it can safely claim full-screen recognition.

## 11. Key files

- `models/species_resnet18_robust_best.pt` — final selected checkpoint
- `api.py` — FastAPI service
- `inference.py` — reusable model loader and prediction layer
- `test_api.py` — API smoke tests
- `train_species.py` — species-level training pipeline
- `finetune_species.py` — robustness fine-tuning pipeline
- `evaluate_species.py` — holdout evaluation
- `robustness_eval.py` — tricky-case evaluation
- `reports/species_manifest.json` — 1,025-species source mapping
- `reports/ROBUSTNESS_EVALUATION.md` — tricky-case report
- `FINAL_RUN_GUIDE.md` — installation and usage instructions
