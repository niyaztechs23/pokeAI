# Dataset Report — Initial Audit

The supplied project contains 1,025 raw dataset folders under `dataset/`, with 6,144 PNG files, and a prepared dataset under `data/` with 1,024 class folders. The prepared split has 5,120 training images and 1,024 validation images: exactly five training images and one validation image per class.

Pillow successfully opened every prepared image. The images are PNG files with dimensions of 96×96, 475×475, or 512×512. There are five exact duplicate groups inside the training set and no exact hashes shared between training and validation. The groups are: Armarouge `back.png`/`back_shiny.png`; Ceruledge `back.png`/`back_shiny.png`; Finizen `back.png` duplicated as Palafin-zero `back.png`; Finizen `back_shiny.png` duplicated as Palafin-zero `back_shiny.png`; and Ogerpon `back.png`/`back_shiny.png`. Each group has two files, so there are five redundant files beyond one copy. Near-duplicate leakage has not yet been tested.

`class_names.json` contains 1,024 unique labels and matches both prepared split folder sets exactly. The names mix ordinary species names with selected forms such as `Aegislash-shield`, `Basculgion-male`, and `Zygarde-50`; the project does not yet define a formal species-versus-form policy. `folders.txt` contains 1,067 unique names, including aliases and names not present in the model label file.

The intended catalogue boundary for the first audit is National Pokédex IDs 1–1025. A machine-readable catalogue and per-entry coverage file are in `reports/`. The catalogue is now populated from the fetched Pokémon Database National Pokédex page and cross-checked against the official Pokémon Pokédex page. The project’s 1,024 labels are therefore compared against a 1,025-entry boundary, with forms kept distinct from the base-species catalogue.

This dataset is suitable for wiring and smoke tests, but not for a credible generalization claim. One validation image per class cannot support stable per-class precision/recall or mixed-domain conclusions.
