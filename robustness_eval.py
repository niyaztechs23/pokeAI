from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image,ImageEnhance,ImageOps,ImageDraw
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='models/species_resnet18_best.pt'); p.add_argument('--out',default='reports/robustness_evaluation.json'); a=p.parse_args()
ck=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False); labels=ck['class_names']; m=models.resnet18(weights=None); m.fc=nn.Linear(512,len(labels)); m.load_state_dict(ck['model']); m.eval()
manifest=json.loads((ROOT/'reports/species_manifest.json').read_text())['entries']
base=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def variant(im,name):
 im=im.convert('RGB')
 if name=='clean': return im
 if name=='low_resolution': return im.resize((32,32),Image.Resampling.BILINEAR).resize((512,512),Image.Resampling.NEAREST)
 if name=='grayscale': return ImageOps.grayscale(im).convert('RGB')
 if name=='dark': return ImageEnhance.Brightness(im).enhance(.45)
 if name=='bright': return ImageEnhance.Brightness(im).enhance(1.8)
 if name=='flip': return ImageOps.mirror(im)
 if name=='crop': return im.crop((im.width*.15,im.height*.15,im.width*.85,im.height*.85)).resize(im.size)
 if name=='occluded':
  out=im.copy(); d=ImageDraw.Draw(out); d.rectangle((out.width*.35,out.height*.35,out.width*.65,out.height*.65),fill=(128,128,128)); return out
 raise ValueError(name)
variants=['clean','low_resolution','grayscale','dark','bright','flip','crop','occluded']; result={}
for vn in variants:
 correct=top3=0; n=0
 for idx,e in enumerate(manifest):
  path=ROOT/[x for x in e['images'] if x.endswith('/official.png')][0]
  with Image.open(path) as im: x=base(variant(im,vn)).unsqueeze(0)
  with torch.inference_mode(): pr=torch.softmax(m(x),1)[0]; inds=pr.topk(3).indices.tolist()
  correct += int(inds[0]==idx); top3 += int(idx in inds); n+=1
 result[vn]={'top1_accuracy':correct/n,'top3_accuracy':top3/n,'correct':correct,'top3_correct':top3,'n':n}
Path(ROOT/a.out).write_text(json.dumps({'checkpoint':a.checkpoint,'variants':result},indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
