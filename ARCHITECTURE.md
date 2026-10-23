# PokeAI ResNet Architecture

The current implementation is a PyTorch image-classification pipeline built around an ImageNet-initialized ResNet18. The final layer is replaced with a linear classifier whose output count is derived from the training folder order. The current supplied label file contains 1,024 labels.

The data flow is: `data/train` and `data/val` → `torchvision.datasets.ImageFolder` → transforms → ResNet18 → softmax top-k output. `predict.py` handles one image. `live_detect.py` captures one fixed rectangle with MSS and classifies the entire rectangle; it is not a detector or a general full-screen localization system.

Training preprocessing uses random resized crop, horizontal flip, and color jitter. Validation and single-image prediction use resize-to-256 followed by a center crop to 224. Live detection uses a direct resize to 224×224. These paths should be unified or intentionally versioned before accuracy comparisons.

The archive also contains `checkpoint.pth`, which the source expects to be a training-resume dictionary, and `pokemon_classifier.pt`, which the source expects to be a raw model state dictionary. Both files were subsequently verified as the same 122-tensor ResNet18 state dictionary with a 1,024-output classifier. The archive checkpoint also contains optimizer state and epoch metadata for resume training.
