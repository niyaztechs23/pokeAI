from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
catalogue = json.loads((ROOT / 'reports/pokedex_catalogue.json').read_text(encoding='utf-8'))['entries']
raw_dirs = {p.name: p for p in (ROOT / 'dataset').iterdir() if p.is_dir()}

def norm(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.lower())

raw_norm = {norm(name): name for name in raw_dirs}
aliases = {
    'Nidoran♀': 'Nidoran-f',
    'Nidoran♂': 'Nidoran-m',
    'Flabébé': 'Flabebe',
    'Minior': 'Minior-red-meteor',
    'Maushold': 'Maushold-family-of-four',
}

entries = []
used = {}
for item in catalogue:
    name = item['name']
    folder = aliases.get(name) or raw_norm.get(norm(name))
    if folder is None:
        candidates = [
            candidate for candidate in raw_dirs
            if norm(candidate).startswith(norm(name))
            and any(raw_dirs[candidate].iterdir())
        ]
        if len(candidates) == 1:
            folder = candidates[0]
    if folder is None:
        entries.append({
            'national_dex': item['national_dex'],
            'class_id': f"national_{item['national_dex']:04d}",
            'name': name,
            'source_folder': None,
            'status': 'missing_source_folder',
            'images': [],
        })
        continue
    files = sorted(path for path in raw_dirs[folder].iterdir() if path.is_file())
    status = 'ready' if len(files) >= 2 else 'insufficient_source_images'
    entries.append({
        'national_dex': item['national_dex'],
        'class_id': f"national_{item['national_dex']:04d}",
        'name': name,
        'source_folder': folder,
        'status': status,
        'images': [str(path.relative_to(ROOT)) for path in files],
    })
    used[folder] = used.get(folder, 0) + 1

summary = {
    'catalogue_species': len(entries),
    'ready_species': sum(item['status'] == 'ready' for item in entries),
    'missing_or_insufficient': sum(item['status'] != 'ready' for item in entries),
    'source_folders_used': len(used),
    'source_folder_reuse': [name for name, count in used.items() if count > 1],
}
out = ROOT / 'reports/species_manifest.json'
out.write_text(json.dumps({
    'scope': 'National Pokédex IDs 1–1025',
    'entries': entries,
    'summary': summary,
}, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2))
print('not ready', [item['name'] for item in entries if item['status'] != 'ready'])
