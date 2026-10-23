
## Robust fine-tuning pass

A second fine-tuning pass used stronger random crops, resize variation, flips, color jitter, grayscale augmentation, and a lower learning rate. The selected checkpoint is `models/species_resnet18_robust_best.pt`.

Its clean official-artwork holdout improved to **90.24% top-1** (925/1,025), with **94.73% top-3** and **96.20% top-5**. The tricky-case suite is recorded in `reports/ROBUSTNESS_EVALUATION.md`: darkened images reached 88.39% top-1 and flips 89.95%, while severe 32×32 resolution reached 0.78%, center crops 55.22%, and central occlusion 49.66%.

The final API uses this fine-tuned checkpoint by default. The supplied Pikachu sample remains a failure case: the model predicts Diglett top-1 and does not place Pikachu in its top five. This is why the model is described as a measured classifier, not a universal recognizer.
