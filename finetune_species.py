from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
ROOT=Path(__file__).resolve().parent
class DS(Dataset):
 def __init__(self,records,tf): self.records=records; self.tf=tf
 def __len__(self): return len(self.records)
 def __getitem__(self,i):
  p,y=self.records[i]
  with Image.open(ROOT/p) as im: im=im.convert('RGB')
  return self.tf(im),y
def epoch(model,loader,loss_fn,opt,train,device):
 model.train(train); loss_sum=correct=n=0
 with torch.set_grad_enabled(train):
  for x,y in loader:
   x,y=x.to(device),y.to(device)
   if train: opt.zero_grad(set_to_none=True)
   z=model(x); loss=loss_fn(z,y)
   if train: loss.backward(); opt.step()
   loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); n+=len(y)
 return loss_sum/n,correct/n
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--init',default='models/species_resnet18_best.pt'); ap.add_argument('--out',default='models/species_resnet18_robust_best.pt'); ap.add_argument('--epochs',type=int,default=4); args=ap.parse_args()
 torch.manual_seed(42); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 manifest=json.loads((ROOT/'reports/species_manifest.json').read_text())['entries']; labels=[e['name'] for e in manifest]
 tr=[]; va=[]
 for y,e in enumerate(manifest):
  va += [(p,y) for p in e['images'] if p.endswith('/official.png')]
  tr += [(p,y) for p in e['images'] if not p.endswith('/official.png')]
 norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
 train_tf=transforms.Compose([transforms.Resize(280),transforms.RandomResizedCrop(224,scale=(.65,1.0)),transforms.RandomHorizontalFlip(),transforms.RandomApply([transforms.ColorJitter(.25,.25,.25,.1)],p=.7),transforms.RandomGrayscale(p=.08),transforms.ToTensor(),norm])
 val_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
 model=models.resnet18(weights=None); model.fc=nn.Linear(512,len(labels)); ck=torch.load(ROOT/args.init,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.to(device)
 for name,p in model.named_parameters(): p.requires_grad=name.startswith('layer4') or name.startswith('fc')
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=2e-4); loss=nn.CrossEntropyLoss(label_smoothing=.08)
 tl=DataLoader(DS(tr,train_tf),batch_size=64,shuffle=True,num_workers=0); vl=DataLoader(DS(va,val_tf),batch_size=64,shuffle=False,num_workers=0)
 best=-1; hist=[]; out=ROOT/args.out; out.parent.mkdir(exist_ok=True)
 for ep in range(1,args.epochs+1):
  t=time.perf_counter(); a=epoch(model,tl,loss,opt,True,device); b=epoch(model,vl,loss,opt,False,device); row={'epoch':ep,'train_loss':a[0],'train_accuracy':a[1],'val_loss':b[0],'val_accuracy':b[1],'seconds':time.perf_counter()-t}; hist.append(row); print(row)
  if b[1]>best: best=b[1]; torch.save({'architecture':'resnet18','class_names':labels,'model':model.state_dict(),'epoch':ep,'val_accuracy':b[1],'base_checkpoint':args.init},out)
 (out.parent/'robust_finetune_history.json').write_text(json.dumps({'history':hist,'best_val_accuracy':best},indent=2))
 print(json.dumps({'best_val_accuracy':best,'checkpoint':str(out)},indent=2))
if __name__=='__main__': main()
