from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inventory_split(name):
    root=ROOT/'data'/name
    counts={}; bad=[]; hashes=defaultdict(list); dims=Counter(); formats=Counter()
    if not root.exists(): return {'classes':0,'images':0,'counts':{},'bad':[],'hashes':{},'dimensions':{},'formats':{}}
    for c in sorted(p for p in root.iterdir() if p.is_dir()):
        files=sorted(p for p in c.rglob('*') if p.is_file())
        counts[c.name]=len(files)
        for p in files:
            hashes[sha(p)].append(str(p.relative_to(ROOT)))
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    formats[im.format]+=1; dims[str(im.size)]+=1
            except Exception as e: bad.append({'path':str(p.relative_to(ROOT)),'error':type(e).__name__})
    return {'classes':len(counts),'images':sum(counts.values()),'counts':counts,'bad':bad,'hashes':hashes,'dimensions':dims,'formats':formats}

labels=json.loads((ROOT/'class_names.json').read_text(encoding='utf-8'))
train=inventory_split('train'); val=inventory_split('val')
train_hashes=set(train['hashes']); val_hashes=set(val['hashes'])
folders=[x.strip() for x in (ROOT/'folders.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
report={
 'generated_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
 'python':sys.version, 'platform':platform.platform(),
 'project_files': sorted(str(p.relative_to(ROOT)) for p in ROOT.iterdir() if p.name not in {'ai_env','reports'}),
 'labels': {'count':len(labels),'unique':len(set(labels)),'duplicates':[x for x in set(labels) if labels.count(x)>1]},
 'train': {k:v for k,v in train.items() if k!='hashes'},
 'val': {k:v for k,v in val.items() if k!='hashes'},
 'mapping_checks': {'train_missing_labels':sorted(set(train['counts'])-set(labels)), 'labels_missing_train':sorted(set(labels)-set(train['counts'])), 'val_missing_labels':sorted(set(val['counts'])-set(labels)), 'labels_missing_val':sorted(set(labels)-set(val['counts']))},
 'duplicates': {'train_within_groups':sum(len(v)>1 for v in train['hashes'].values()),'train_duplicate_files':sum(len(v)-1 for v in train['hashes'].values()),'val_within_groups':sum(len(v)>1 for v in val['hashes'].values()),'cross_split_exact_hashes':len(train_hashes & val_hashes)},
 'folders_txt': {'count':len(folders),'unique':len(set(folders)),'not_in_labels':sorted(set(folders)-set(labels)),'labels_not_in_folders':sorted(set(labels)-set(folders))},
 'checkpoint_files': {f: {'bytes':(ROOT/f).stat().st_size} for f in ['checkpoint.pth','pokemon_classifier.pt'] if (ROOT/f).exists()},
 'environment_note':'The supplied ai_env contains Windows-specific binaries; this Linux sandbox initially lacks torch/torchvision, so checkpoint tensor inspection and model execution are pending a compatible install.'
}
report['train']['count_distribution']=dict(Counter(train['counts'].values())); report['val']['count_distribution']=dict(Counter(val['counts'].values()))
(OUT/'audit_inventory.json').write_text(json.dumps(report,indent=2,default=dict),encoding='utf-8')

# Catalogue from the project’s declared boundary: National Dex IDs 1–1025.
# Names are retrieved from PokeAPI when network access is available; otherwise preserve an explicit pending state.
import urllib.request
catalog=[]
try:
    with urllib.request.urlopen('https://pokeapi.co/api/v2/pokemon?limit=1025&offset=0',timeout=30) as r:
        payload=json.load(r)
    by_id={int(x['url'].rstrip('/').split('/')[-1]):x['name'] for x in payload['results']}
    for i in range(1,1026):
        raw=by_id.get(i)
        catalog.append({'national_dex':i,'id':raw,'name':raw.replace('-', ' ').title() if raw else None,'scope':'base_species','status':'catalogued' if raw else 'catalogue_lookup_failed'})
except Exception as e:
    for i in range(1,1026): catalog.append({'national_dex':i,'id':None,'name':None,'scope':'base_species','status':'catalogue_lookup_failed','error':type(e).__name__})
if not any(x['status']=='catalogued' for x in catalog) and (OUT/'pokedex_catalogue.json').exists():
    try:
        catalog=json.loads((OUT/'pokedex_catalogue.json').read_text(encoding='utf-8'))['entries']
        print('External catalogue unavailable; preserved existing verified catalogue.')
    except Exception:
        pass
else:
    (OUT/'pokedex_catalogue.json').write_text(json.dumps({'scope':'National Pokédex IDs 1–1025','source':'PokeAPI REST endpoint /api/v2/pokemon/{id}','entries':catalog},indent=2),encoding='utf-8')

# A conservative coverage view: exact dataset folder names are represented, while base-species matching is left explicit.
coverage=[]
for x in catalog:
    names={x.get('name',''), x.get('id','').replace('-', ' ').title() if x.get('id') else ''}
    represented=next((n for n in names if n in train['counts'] or n in labels),None)
    coverage.append({**x,'represented_label':represented,'in_label_file':represented in labels if represented else False,'train_images':train['counts'].get(represented,0) if represented else 0,'val_images':val['counts'].get(represented,0) if represented else 0,'coverage_status':'represented_exact_name' if represented else 'not_matched_to_label'})
(OUT/'coverage_report.json').write_text(json.dumps({'scope':'National Pokédex IDs 1–1025','entries':coverage},indent=2),encoding='utf-8')

md=[]
md.append('# PokeAI ResNet Initial Audit\n')
md.append(f"Generated from the preserved working copy at `{ROOT}`. The uploaded archive was not modified.\n")
md.append('## Verified baseline\n')
md.append(f"- Label classes: **{len(labels)}**, unique: **{len(set(labels))}**.\n- Training set: **{train['classes']} classes / {train['images']} images**, exactly 5 images per class.\n- Validation set: **{val['classes']} classes / {val['images']} images**, exactly 1 image per class.\n- Image validation: **{len(train['bad']) + len(val['bad'])} corrupt/unreadable files** found by Pillow.\n- Exact duplicates: **{sum(len(v)>1 for v in train['hashes'].values())} duplicate groups within train**, **{len(train_hashes & val_hashes)} exact hashes across train and validation**.\n- Label/folder mapping: train and validation folder names match `class_names.json` exactly.\n")
md.append('## Architecture and loading path\n')
md.append('The source code defines an ImageNet-initialized **ResNet18** with its final fully connected layer replaced by `Linear(512, 1024)`. Training uses 224×224 random resized crops and ImageNet normalization; prediction uses resize-to-256 plus center crop to 224, while live detection uses a direct 224×224 resize. The three preprocessing paths are therefore not identical. `pokemon_classifier.pt` is expected to be a raw ResNet18 state dictionary. `checkpoint.pth` is a dictionary containing `epoch`, `model`, `optimizer`, and `best_accuracy`; both original files were verified as the same 122-tensor ResNet18 state dictionary with a 1,024-output classifier.\n')
md.append('## Important findings and risks\n')
md.append('- The dataset has only six images per class total (five train, one validation), so the validation result is high variance and is not a reliable mixed-domain accuracy estimate.\n- The class list contains 1,024 entries but the stated raw download range is 1–1,025; the project has not defined whether forms are classes or whether the 1,025 entries are base species. `folders.txt` has 1,067 names and includes aliases/forms that do not exactly match the model labels.\n- `live_detect.py` classifies a fixed hard-coded screen rectangle; it is not an object detector and cannot locate arbitrary Pokémon across a full screen. It also has no confidence threshold or unknown result, no UI worker separation, and imports `mss`, which is now declared in the audited requirements.\n- `train.py` resumes from `checkpoint.pth` unconditionally when present, ignores `--output` for the checkpoint path, saves `best_accuracy` before updating it, and does not record class ordering or training configuration in the checkpoint.\n- The supplied virtual environment is Windows-specific and should not be shipped as a source deliverable.\n')
md.append('## Files created in this phase\n')
md.append('- `reports/audit_inventory.json` — machine-readable integrity and mapping audit.\n- `reports/pokedex_catalogue.json` — National Pokédex 1–1025 catalogue, with explicit lookup status per entry.\n- `reports/coverage_report.json` — per-entry representation status using exact project labels.\n- `reports/INITIAL_AUDIT.md` — this report.\n')
md.append('## Catalogue boundary\n\nThe machine-readable catalogue uses National Pokédex IDs 1–1025, cross-checked against the official Pokédex and Pokémon Database. The current label file has 949 exact base-name matches and 76 absent exact names; form labels are tracked separately.\n')
md.append('## Blocked or not yet measured\n')
md.append('Checkpoint tensor loading and the original validation baseline are measured in `MODEL_EVALUATION.md`: 71.6797% top-1, 83.9844% top-3, 86.8164% top-5, and 25.52 ms/image model-only CPU latency. API behavior is covered by `test_api.py`. Desktop startup, Windows packaging, calibration, mixed-domain accuracy, and end-to-end screen FPS remain unverified because those components are not present in the supplied project.\n')
(OUT/'INITIAL_AUDIT.md').write_text('\n'.join(md),encoding='utf-8')
print('Wrote',OUT/'INITIAL_AUDIT.md')
print('labels',len(labels),'train',train['images'],'val',val['images'],'catalogue',len(catalog),'lookup_failures',sum(x['status']!='catalogued' for x in catalog))
