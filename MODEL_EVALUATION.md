# Model Evaluation — Initial Audit

The verified original baseline on the 1,024-image validation set is **71.6797% top-1 (734/1,024)**, **83.9844% top-3**, and **86.8164% top-5**. CPU model-only inference averaged **25.52 ms/image** over 1,024 single-image calls on this sandbox; this is not end-to-end screen FPS. The supplied `ai_env` remains Windows-specific, but a compatible Linux Torch/TorchVision installation was used for this measurement.

The source indicates a ResNet18 classifier with 1,024 outputs. Both `checkpoint.pth` and `pokemon_classifier.pt` load successfully into that architecture, and their 122 state-dictionary tensors are byte-for-byte identical. `checkpoint.pth` records epoch 10 and best_accuracy 0.7099609375, while the independently rerun validation result is 0.716796875. The intended baseline protocol is to load the original `pokemon_classifier.pt`, verify its state-dictionary keys and tensor shapes, run the unchanged validation transform over `data/val`, and report top-1/top-k accuracy plus per-class results. The result must be interpreted cautiously because validation has one image per class and the dataset contains only six images per class in total.

A future evaluation should add separate held-out groups for in-game screenshots, sprites, 3D renders, artwork, difficult images, and non-Pokémon images. Softmax values should be described as model scores, not guaranteed probabilities. Unknown rejection requires a validation-calibrated threshold and a negative set.

## Full per-class analysis

The reproducible `full_model_eval.py` run is documented in `reports/FULL_MODEL_EVALUATION.md` and `reports/full_model_evaluation.json`. It identifies 290 classes missed on their single validation image, lists high-confidence errors, low-confidence correct predictions, precision/recall, confusion pairs, repeated error-attractor labels, and exact duplicate groups.
