# Full Model Evaluation Report
## Protocol
The original `pokemon_classifier.pt` was evaluated on all **1024** images in `data/val`, using the original prediction preprocessing (resize 256, center crop 224, ImageNet normalization). Evaluation ran on CPU in batches of 32. Because there is one validation image per class, each class recall is either 0% or 100%; weak-class findings are directional, not statistically stable.
## Overall results
- Top-1: **71.6797%** (734/1024)
- Top-3: **83.9844%** (860/1024)
- Top-5: **86.8164%** (889/1024)
- Batched evaluation elapsed time: **30.57s**; this is not end-to-end screen FPS.
## Weak classes and high-confidence errors
The following classes were misclassified on their only validation image. They are sorted by the model’s incorrect top-1 score, so the first rows are the most confident errors.
- **Mismagius** → `Sneasler` at 0.9879; true-class score 0.0000.
- **Meowscarada** → `Floragato` at 0.9877; true-class score 0.0048.
- **Vanillish** → `Vanilluxe` at 0.9847; true-class score 0.0005.
- **Mr-mime** → `Blacephalon` at 0.9828; true-class score 0.0013.
- **Nidoran-m** → `Nidorino` at 0.9771; true-class score 0.0179.
- **Golurk** → `Golett` at 0.9750; true-class score 0.0013.
- **Mienshao** → `Sneasler` at 0.9749; true-class score 0.0024.
- **Klinklang** → `Klang` at 0.9709; true-class score 0.0283.
- **Poliwag** → `Poliwhirl` at 0.9529; true-class score 0.0447.
- **Poochyena** → `Furfrou` at 0.9476; true-class score 0.0061.
- **Torterra** → `Rillaboom` at 0.9197; true-class score 0.0048.
- **Cascoon** → `Silcoon` at 0.9031; true-class score 0.0017.
- **Fletchinder** → `Fletchling` at 0.8973; true-class score 0.0050.
- **Rolycoly** → `Minior-red-meteor` at 0.8966; true-class score 0.0000.
- **Toxtricity-amped** → `Sneasler` at 0.8839; true-class score 0.0093.
- **Naclstack** → `Garganacl` at 0.8814; true-class score 0.0709.
- **Tsareena** → `Sneasler` at 0.8696; true-class score 0.0026.
- **Cottonee** → `Tornadus-incarnate` at 0.8686; true-class score 0.0031.
- **Pupitar** → `Arctibax` at 0.8668; true-class score 0.0104.
- **Bombirdier** → `Latias` at 0.8635; true-class score 0.0038.
- **Dhelmise** → `Tornadus-incarnate` at 0.8541; true-class score 0.0228.
- **Skwovet** → `Quilladin` at 0.8505; true-class score 0.0072.
- **Drowzee** → `Typhlosion` at 0.8462; true-class score 0.0001.
- **Braviary** → `Ho-oh` at 0.8459; true-class score 0.0393.
- **Cyclizar** → `Espeon` at 0.8433; true-class score 0.0688.
- **Rufflet** → `Furfrou` at 0.8176; true-class score 0.0000.
- **Silcoon** → `Cascoon` at 0.7976; true-class score 0.1791.
- **Bisharp** → `Sneasler` at 0.7915; true-class score 0.0471.
- **Gallade** → `Sneasler` at 0.7891; true-class score 0.0196.
- **Tentacruel** → `Sneasler` at 0.7854; true-class score 0.0521.

## Lowest-confidence correct classes
- **Buneary** correct at true-class score 0.0859.
- **Seel** correct at true-class score 0.0929.
- **Eevee** correct at true-class score 0.1120.
- **Primeape** correct at true-class score 0.1262.
- **Dunsparce** correct at true-class score 0.1607.
- **Chi-yu** correct at true-class score 0.1698.
- **Onix** correct at true-class score 0.1790.
- **Huntail** correct at true-class score 0.1795.
- **Roserade** correct at true-class score 0.1845.
- **Luvdisc** correct at true-class score 0.1855.
- **Giratina-altered** correct at true-class score 0.1975.
- **Sylveon** correct at true-class score 0.2006.
- **Galvantula** correct at true-class score 0.2051.
- **Rhyhorn** correct at true-class score 0.2061.
- **Marshtomp** correct at true-class score 0.2073.
- **Chinchou** correct at true-class score 0.2122.
- **Scrafty** correct at true-class score 0.2208.
- **Butterfree** correct at true-class score 0.2263.
- **Xerneas** correct at true-class score 0.2282.
- **Vullaby** correct at true-class score 0.2329.
- **Oshawott** correct at true-class score 0.2346.
- **Vaporeon** correct at true-class score 0.2355.
- **Jirachi** correct at true-class score 0.2394.
- **Charmeleon** correct at true-class score 0.2412.
- **Roselia** correct at true-class score 0.2505.

## Repeated error-attractor labels
Across the 290 top-1 errors, these predicted labels attracted the largest number of incorrect examples. This pattern is more informative than individual pair counts because each class has only one validation image.
- **Sneasler** received 20 incorrect top-1 predictions.
- **Latias** received 13 incorrect top-1 predictions.
- **Snubbull** received 12 incorrect top-1 predictions.
- **Porygon2** received 10 incorrect top-1 predictions.
- **Ho-oh** received 7 incorrect top-1 predictions.
- **Furfrou** received 6 incorrect top-1 predictions.
- **Treecko** received 5 incorrect top-1 predictions.
- **Passimian** received 5 incorrect top-1 predictions.
- **Gliscor** received 5 incorrect top-1 predictions.
- **Tornadus-incarnate** received 5 incorrect top-1 predictions.
- **Floragato** received 5 incorrect top-1 predictions.
- **Quilladin** received 4 incorrect top-1 predictions.
- **Typhlosion** received 4 incorrect top-1 predictions.
- **Shroomish** received 4 incorrect top-1 predictions.
- **Electabuzz** received 3 incorrect top-1 predictions.
- **Geodude** received 3 incorrect top-1 predictions.
- **Bulbasaur** received 3 incorrect top-1 predictions.
- **Clefairy** received 3 incorrect top-1 predictions.
- **Lombre** received 3 incorrect top-1 predictions.
- **Horsea** received 3 incorrect top-1 predictions.

## Most frequent confusion pairs
- `Aipom` → `Treecko`: 1 validation image(s).
- `Amaura` → `Passimian`: 1 validation image(s).
- `Ambipom` → `Magmar`: 1 validation image(s).
- `Applin` → `Treecko`: 1 validation image(s).
- `Arctovish` → `Frosmoth`: 1 validation image(s).
- `Arctozolt` → `Ho-oh`: 1 validation image(s).
- `Azelf` → `Sneasler`: 1 validation image(s).
- `Azurill` → `Porygon2`: 1 validation image(s).
- `Bagon` → `Electabuzz`: 1 validation image(s).
- `Bayleef` → `Flygon`: 1 validation image(s).
- `Beartic` → `Geodude`: 1 validation image(s).
- `Beheeyem` → `Sneasler`: 1 validation image(s).
- `Bergmite` → `Gliscor`: 1 validation image(s).
- `Bewear` → `Snubbull`: 1 validation image(s).
- `Binacle` → `Chespin`: 1 validation image(s).
- `Bisharp` → `Sneasler`: 1 validation image(s).
- `Bombirdier` → `Latias`: 1 validation image(s).
- `Bounsweet` → `Tornadus-incarnate`: 1 validation image(s).
- `Bramblin` → `Gourgeist-average`: 1 validation image(s).
- `Braviary` → `Ho-oh`: 1 validation image(s).
- `Breloom` → `Klawf`: 1 validation image(s).
- `Brionne` → `Totodile`: 1 validation image(s).
- `Bunnelby` → `Passimian`: 1 validation image(s).
- `Capsakid` → `Bulbasaur`: 1 validation image(s).
- `Carbink` → `Clefairy`: 1 validation image(s).
- `Carkol` → `Snubbull`: 1 validation image(s).
- `Cascoon` → `Silcoon`: 1 validation image(s).
- `Castform` → `Smoochum`: 1 validation image(s).
- `Cetoddle` → `Cetitan`: 1 validation image(s).
- `Chansey` → `Lickilicky`: 1 validation image(s).

## Per-class precision and recall
Full machine-readable per-class results, including support, precision, recall, predicted class, and scores, are in `reports/full_model_evaluation.json`. With one validation image per class, recall is 0 or 1 and precision is affected by collisions where multiple true classes receive the same predicted label.

## Duplicate groups
The prepared training set contains **5 exact duplicate groups** (5 redundant files beyond one copy). Exact duplicate details and SHA-256 hashes are included in the JSON report. No exact train/validation hash overlap was found in the audit.
