from pathlib import Path

dataset = Path("dataset")

missing = []

for i in range(1, 1026):
    found = False

    for folder in dataset.iterdir():
        if folder.is_dir() and folder.name.startswith(f"{i:04d}"):
            found = True
            break

    if not found:
        missing.append(i)

print("Missing Pokémon:", len(missing))
print("IDs:", missing)