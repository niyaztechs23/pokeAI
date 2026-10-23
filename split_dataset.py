"""
Splits a folder-per-class dataset into train/val subsets.

Expects input like:
    raw_data/
      pikachu/
      charmander/

Produces:
    data/train/pikachu/...
    data/val/pikachu/...

Usage:
python split_dataset.py --input_dir raw_data --output_dir data --val_split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing one subfolder per Pokémon"
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Where to write train/ and val/ folders"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of images per class to use for validation"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        raise SystemExit(f"No class folders found in {input_dir}")

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    total_train = 0
    total_val = 0

    for class_dir in class_dirs:

        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_ext
        ]

        if len(images) == 0:
            continue

        random.shuffle(images)

        # ---------- FIX ----------
        if len(images) == 1:
            train_images = images
            val_images = []
        else:
            n_val = max(1, int(len(images) * args.val_split))

            # Never allow train to become empty
            if n_val >= len(images):
                n_val = len(images) - 1

            val_images = images[:n_val]
            train_images = images[n_val:]
        # -------------------------

        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / class_dir.name / img.name)

        for img in val_images:
            shutil.copy2(img, val_dir / class_dir.name / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(f"{class_dir.name}: {len(train_images)} train, {len(val_images)} val")

    print("\nDone!")
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_val}")
    print(f"Classes: {len(class_dirs)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()