from pathlib import Path

dataset = Path("dataset")

have = {p.name.lower() for p in dataset.iterdir() if p.is_dir()}

missing = []

with open("pokemon.csv", encoding="utf-8") as f:
    next(f)
    for line in f:
        pid, name = line.strip().split(",")[:2]
        if name.lower() not in have:
            missing.append((pid, name))

print("Missing:", len(missing))
for pid, name in missing:
    print(pid, name)