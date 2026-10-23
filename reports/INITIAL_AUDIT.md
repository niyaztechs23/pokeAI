# PokeAI ResNet Initial Audit

Generated from the preserved working copy at `/home/ubuntu/work/PokeAI_ResNet`. The uploaded archive was not modified.

## Verified baseline

- Label classes: **1024**, unique: **1024**.
- Training set: **1024 classes / 5120 images**, exactly 5 images per class.
- Validation set: **1024 classes / 1024 images**, exactly 1 image per class.
- Image validation: **0 corrupt/unreadable files** found by Pillow.
- Exact duplicates: **5 duplicate groups within train**, **0 exact hashes across train and validation**.
- Label/folder mapping: train and validation folder names match `class_names.json` exactly.

## Architecture and loading path

The source code defines an ImageNet-initialized **ResNet18** with its final fully connected layer replaced by `Linear(512, 1024)`. Training uses 224×224 random resized crops and ImageNet normalization; prediction uses resize-to-256 plus center crop to 224, while live detection uses a direct 224×224 resize. The three preprocessing paths are therefore not identical. `pokemon_classifier.pt` is expected to be a raw ResNet18 state dictionary. `checkpoint.pth` is a dictionary containing `epoch`, `model`, `optimizer`, and `best_accuracy`; both original files were verified as the same 122-tensor ResNet18 state dictionary with a 1,024-output classifier.

## Important findings and risks

- The dataset has only six images per class total (five train, one validation), so the validation result is high variance and is not a reliable mixed-domain accuracy estimate.
- The class list contains 1,024 entries but the stated raw download range is 1–1,025; the project has not defined whether forms are classes or whether the 1,025 entries are base species. `folders.txt` has 1,067 names and includes aliases/forms that do not exactly match the model labels.
- `live_detect.py` classifies a fixed hard-coded screen rectangle; it is not an object detector and cannot locate arbitrary Pokémon across a full screen. It also has no confidence threshold or unknown result, no UI worker separation, and imports `mss`, which is now declared in the audited requirements.
- `train.py` resumes from `checkpoint.pth` unconditionally when present, ignores `--output` for the checkpoint path, saves `best_accuracy` before updating it, and does not record class ordering or training configuration in the checkpoint.
- The supplied virtual environment is Windows-specific and should not be shipped as a source deliverable.

## Files created in this phase

- `reports/audit_inventory.json` — machine-readable integrity and mapping audit.
- `reports/pokedex_catalogue.json` — National Pokédex 1–1025 catalogue, with explicit lookup status per entry.
- `reports/coverage_report.json` — per-entry representation status using exact project labels.
- `reports/INITIAL_AUDIT.md` — this report.

## Catalogue boundary

The machine-readable catalogue uses National Pokédex IDs 1–1025, cross-checked against the official Pokédex and Pokémon Database. The current label file has 949 exact base-name matches and 76 absent exact names; form labels are tracked separately.

## Blocked or not yet measured

Checkpoint tensor loading and the original validation baseline are measured in `MODEL_EVALUATION.md`: 71.6797% top-1, 83.9844% top-3, 86.8164% top-5, and 25.52 ms/image model-only CPU latency. API behavior is covered by `test_api.py`. Desktop startup, Windows packaging, calibration, mixed-domain accuracy, and end-to-end screen FPS remain unverified because those components are not present in the supplied project.
