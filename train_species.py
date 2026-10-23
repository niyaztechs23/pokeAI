from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

class ManifestDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        path, target = self.records[index]
        with Image.open(ROOT / path) as image:
            image = image.convert('RGB')
        return self.transform(image), target

def normalize_name(value: str) -> str:
    return ''.join(ch for ch in value.lower() if ch.isalnum())

def make_model(class_count: int, old_state: dict, old_labels: list[str], manifest: list[dict]) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    current = model.state_dict()
    for key, value in old_state.items():
        if key != 'fc.weight' and key != 'fc.bias' and key in current and current[key].shape == value.shape:
            current[key] = value
    old_by_name = {normalize_name(name): index for index, name in enumerate(old_labels)}
    seed_aliases = {'Diglett': 'Dugtrio'}
    for target, item in enumerate(manifest):
        candidates = [item['source_folder'], item['name'], seed_aliases.get(item['name'])]
        source_index = next((old_by_name.get(normalize_name(x)) for x in candidates if x), None)
        if source_index is not None:
            current['fc.weight'][target] = old_state['fc.weight'][source_index]
            current['fc.bias'][target] = old_state['fc.bias'][source_index]
    model.load_state_dict(current)
    return model

def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0; correct = 0; total = 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, targets)
            if training:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item(); total += targets.size(0)
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='reports/species_manifest.json')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='models/species_resnet18_best.pt')
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = json.loads((ROOT / args.manifest).read_text(encoding='utf-8'))['entries']
    if any(item['status'] != 'ready' for item in manifest): raise SystemExit('Manifest contains classes without source images')
    labels = [item['name'] for item in manifest]
    records_train=[]; records_val=[]
    for target,item in enumerate(manifest):
        images=item['images']
        val=[p for p in images if p.endswith('/official.png')]
        train=[p for p in images if p not in val]
        if not val or not train: raise SystemExit(f'Invalid split for {item["name"]}')
        records_train += [(p,target) for p in train]; records_val += [(p,target) for p in val]
    normalize=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    train_tf=transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224,padding=8),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),normalize])
    val_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
    train_loader=DataLoader(ManifestDataset(records_train,train_tf),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(ManifestDataset(records_val,val_tf),batch_size=args.batch_size,shuffle=False,num_workers=0)
    old_labels=json.loads((ROOT/'class_names.json').read_text())
    checkpoint=torch.load(ROOT/'checkpoint.pth',map_location='cpu',weights_only=False)
    model=make_model(len(labels),checkpoint['model'],old_labels,manifest).to(device)
    # Preserve learned low/mid-level features while adapting the final residual block and classifier.
    for name,param in model.named_parameters(): param.requires_grad = name.startswith('layer4') or name.startswith('fc')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    output=ROOT/args.output; output.parent.mkdir(parents=True,exist_ok=True)
    best=-1.0; patience=0; history=[]
    print(f'device={device} classes={len(labels)} train={len(records_train)} val={len(records_val)}')
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter()
        train_loss,train_acc=run_epoch(model,train_loader,criterion,optimizer,device,True)
        val_loss,val_acc=run_epoch(model,val_loader,criterion,optimizer,device,False)
        row={'epoch':epoch,'train_loss':train_loss,'train_accuracy':train_acc,'val_loss':val_loss,'val_accuracy':val_acc,'seconds':time.perf_counter()-started}
        history.append(row); print(row)
        if val_acc>best:
            best=val_acc; patience=0
            torch.save({'architecture':'resnet18','class_names':labels,'model':model.state_dict(),'epoch':epoch,'val_accuracy':val_acc,'config':vars(args)},output)
        else: patience+=1
        if patience>=3: break
    (output.parent/'species_training_history.json').write_text(json.dumps({'best_val_accuracy':best,'history':history,'manifest':args.manifest,'device':str(device)},indent=2),encoding='utf-8')
    print(json.dumps({'best_val_accuracy':best,'checkpoint':str(output),'epochs_run':len(history)},indent=2))

if __name__=='__main__': main()
