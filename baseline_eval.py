import json,time,torch
from pathlib import Path
from PIL import Image
from torchvision import models,transforms
import torch.nn as nn
from collections import Counter
labels=json.loads(Path('class_names.json').read_text())
device=torch.device('cpu')
model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,len(labels))
sd=torch.load('pokemon_classifier.pt',map_location=device,weights_only=False); model.load_state_dict(sd); model.eval()
tfm=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
paths=[]; y=[]
for i,c in enumerate(labels):
 p=Path('data/val')/c
 for f in sorted(p.iterdir()):
  paths.append(f); y.append(i)
correct=top3=top5=0; per=Counter(); n=len(paths); total=0.0
with torch.inference_mode():
 for p,target in zip(paths,y):
  x=tfm(Image.open(p).convert('RGB')).unsqueeze(0)
  t=time.perf_counter(); out=model(x); total += time.perf_counter()-t
  top=out.softmax(1).topk(5).indices[0].tolist()
  correct += top[0]==target; top3 += target in top[:3]; top5 += target in top[:5]
  per[labels[target]] += top[0]==target
print(json.dumps({'n':n,'top1':correct/n,'top3':top3/n,'top5':top5/n,'correct':correct,'cpu_model_seconds':total,'cpu_model_ms_mean':total/n*1000,'per_class_correct':sum(per.values()),'torch':torch.__version__},indent=2))
