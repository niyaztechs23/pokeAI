# Tricky-Case Robustness Evaluation

The fine-tuned checkpoint `models/species_resnet18_robust_best.pt` was tested on 1,025 held-out `official.png` images. Each image was transformed deterministically; no transformed image was used for training. This is a controlled robustness probe, not a real-world benchmark.

| Test condition | Top-1 | Top-3 |
|---|---:|---:|
| Clean | 90.24% (925/1,025) | 94.73% (971/1,025) |
| Low resolution: 32×32 then enlarged | 0.78% (8/1,025) | 2.93% (30/1,025) |
| Grayscale | 71.51% | 84.98% |
| Darkened | 88.39% | 94.73% |
| Brightened | 77.56% | 88.00% |
| Horizontally flipped | 89.95% | 95.32% |
| Center crop | 55.22% | 75.80% |
| Central occlusion | 49.66% | 74.93% |

## Interpretation

The fine-tune substantially improves clean held-out accuracy from 83.22% to 90.24%. It preserves performance under horizontal flips and moderate darkening. It is not robust to severe resolution loss, aggressive crops, or large occlusions. Those cases need real training examples and an object-localization pipeline rather than only stronger augmentation.

The supplied `pikachu.png` was also tested through the API. The final model returned Diglett at 17.79% as top-1, followed by Bidoof and Exploud; Pikachu was not in the top five. This confirms that the clean official-artwork score does not imply reliable arbitrary-image recognition.
