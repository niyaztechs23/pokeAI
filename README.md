# Pokémon Image Classifier

A transfer-learning image classifier that identifies Pokémon from images,
built with PyTorch + a pretrained ResNet18 backbone.

## 1. Get the dataset

This project expects images organized like:

```
data/
  train/
    pikachu/
      img001.jpg
      img002.jpg
      ...
    charmander/
      ...
    bulbasaur/
      ...
  val/
    pikachu/
      ...
    charmander/
      ...
```

Good public sources for Pokémon image datasets:
- Kaggle: search "Pokemon Images Dataset" or "Pokemon Generation One"
- Kaggle: "7,000 Labeled Pokemon" dataset (has folders per species)

Download one, unzip it, and split it into `data/train/<class>/` and
`data/val/<class>/` folders (roughly 80% train / 20% val per class). A
simple split script is included below if your download is one big folder
per class with no train/val split yet (`split_dataset.py`).

## 2. Install dependencies

```bash
pip install torch torchvision pillow
```

(Use a GPU-enabled torch install if you have a CUDA GPU — check
https://pytorch.org/get-started/locally/ for the right command. On Colab,
a GPU is already available for free — just set Runtime > Change runtime
type > GPU.)

## 3. Train

```bash
python train.py --data_dir data --epochs 10 --batch_size 32
```

This fine-tunes a pretrained ResNet18 on your Pokémon folders. Training
prints loss/accuracy per epoch and saves the best model to
`pokemon_classifier.pt`, along with `class_names.json` (the label list,
needed for inference).

Expect real training here — this genuinely learns visual features
(shape, color, silhouette) and will not be 100% accurate, especially on
similar-looking species or low-quality crops. Treat its output as a
best guess with a confidence score, not a certainty.

## 4. Predict on a new image

```bash
python predict.py --image path/to/some_pokemon.jpg
```

Prints the top-3 predicted species with confidence percentages.

## Files

- `train.py` — training loop, transfer learning setup, saves the model
- `predict.py` — loads the saved model and classifies a single image
- `split_dataset.py` — helper to split one folder-per-class dataset into train/val
- `requirements.txt` — Python dependencies

## Audited working copy

The audited working copy adds `inference.py` for reproducible loading of either original checkpoint format, `api.py` for a localhost-only prediction service, and `test_api.py` for API smoke tests. Run `python audit_project.py` to regenerate the integrity reports under `reports/`.

Measured original baseline on `data/val` (1,024 images): 71.6797% top-1, 83.9844% top-3, and 86.8164% top-5. CPU model-only inference averaged 25.52 ms/image in the audit sandbox. These results are not mixed-domain or end-to-end screen-recognition metrics.

Start the local API with:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

The supplied `live_detect.py` remains a fixed-region classifier, not a general detector. A Windows desktop overlay and connected web dashboard were not present in the supplied project and are not falsely marked complete.

## 1,025-species candidate

The audited working copy now includes a species-level candidate trained against all 1,025 National Pokédex entries. Its reserved official-artwork holdout results are 83.2195% top-1, 94.8293% top-3, and 96.8780% top-5. The API loads this candidate by default. This is not a guarantee for arbitrary images: the candidate still has a strong Diglett error attractor and misclassified the supplied `pikachu.png` sample.

See `MODEL_IMPROVEMENT_REPORT.md` and `reports/SPECIES_CANDIDATE_EVALUATION.md` for the measured details.
