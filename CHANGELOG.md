# Changelog

## Audit phase — 2026-10-22

- Preserved the uploaded archive and extracted a separate working copy under `/home/ubuntu/work/PokeAI_ResNet`.
- Inspected the original scripts, labels, checkpoints, raw dataset, prepared dataset, and declared dependencies.
- Added `audit_project.py` and machine-readable audit outputs under `reports/`.
- Added initial architecture, dataset, model-evaluation, test, installation, API, and changelog documentation.
- Did not retrain, delete data, overwrite checkpoints, or claim unmeasured metrics.
- Added a reusable `inference.py` loader that supports both the raw classifier state dictionary and the training checkpoint dictionary.
- Added a localhost-only FastAPI service in `api.py` with smoke tests in `test_api.py`.
- Declared API and live-capture dependencies in `requirements.txt`.
- Added `full_model_eval.py` and `reports/FULL_MODEL_EVALUATION.md` with per-class precision/recall, weak-class errors, confusion pairs, and duplicate details.
- Added `reports/api_sample_test.json` documenting a verified `/predict` request using `pikachu.png`.
- Added a complete 1,025-entry species manifest with explicit alternate-form mappings.
- Added and validated six Diglett source assets from the existing public PokeAPI workflow.
- Added `train_species.py` and `evaluate_species.py` for species-level training and holdout evaluation.
- Trained and measured `models/species_resnet18_best.pt`; the API now loads this 1,025-class candidate by default.
- Documented the candidate’s Diglett error attractor and sample-image limitation in `MODEL_IMPROVEMENT_REPORT.md`.
