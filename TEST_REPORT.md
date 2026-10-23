# Test Report — Initial Audit

## Tests executed

- Source and configuration inspection: passed.
- Label JSON parse and uniqueness check: passed; 1,024 unique labels.
- Train/validation folder mapping check: passed; both splits match the label file exactly.
- Image integrity check with Pillow: passed for 6,144 prepared images; zero unreadable files.
- Exact duplicate hash scan: completed; five duplicate groups within train and zero train/validation cross-split exact matches.
- Python dependency probe: Pillow, NumPy, FastAPI, and Uvicorn available; Torch, TorchVision, OpenCV, MSS, scikit-learn, and PySide6 initially unavailable.
- Model execution: passed; both original checkpoint files load into ResNet18 with 1,024 outputs.
- Desktop/API startup and screen capture: not run; no implementation exists for a desktop app or API in the supplied project.
- Original prediction smoke test: passed; `predict.py --image pikachu.png --top_k 3` returned three scores, but the top prediction was Tepig rather than Pikachu, demonstrating that raw confidence is not correctness.

## Reproduction

Run `python audit_project.py` from the project root. It writes machine-readable evidence under `reports/`. Do not treat this phase-one audit as a model evaluation.

## Known failures and unresolved items

End-to-end FPS, Windows packaging, desktop behavior, API responses, dashboard integration, calibration, and per-domain evaluation remain unverified. The source also imports `mss` in `live_detect.py` without declaring it in `requirements.txt`.
- Local API smoke tests: passed; `/health`, `/model/info`, valid multipart `/predict`, and malformed-image rejection were exercised by `test_api.py`.
- Full batched model evaluation: passed over all 1,024 validation images; detailed results saved under `reports/FULL_MODEL_EVALUATION.md` and `reports/full_model_evaluation.json`.
