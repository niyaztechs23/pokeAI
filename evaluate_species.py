from __future__ import annotations
import argparse
import json,time
from collections import Counter,defaultdict
from pathlib import Path
import torch
from PIL import Image
from torchvision import models,transforms
import torch.nn as nn
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',default='models/species_resnet18_best.pt')
args=parser.parse_args()
ckpt=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=False)
labels=ckpt['class_names']; model=models.resnet18(weights=None); model.fc=nn.Linear(512,len(labels)); model.load_state_dict(ckpt['model']); model.eval()
tfm=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
manifest=json.loads((ROOT/'reports/species_manifest.json').read_text())['entries']; items=[]
for i,e in enumerate(manifest): items.append((ROOT/e['images'][-1],i,e['name'])) # official.png is last alphabetically
records=[]; start=time.perf_counter()
with torch.inference_mode():
 for off in range(0,len(items),32):
  chunk=items[off:off+32]; x=torch.stack([tfm(Image.open(p).convert('RGB')) for p,_,_ in chunk]); probs=torch.softmax(model(x),1); vals,inds=probs.topk(5,1)
  for row,((p,target,label),v,ix) in enumerate(zip(chunk,vals,inds)):
   top=[{'label':labels[int(i)],'score':float(s)} for s,i in zip(v,ix)]
   records.append({'true_label':label,'top1_label':top[0]['label'],'top1_score':top[0]['score'],'true_score':float(probs[row,target]),'top1_correct':top[0]['label']==label,'top3_correct':label in [z['label'] for z in top[:3]],'top5_correct':label in [z['label'] for z in top]})
N=len(records); by_pred=Counter(r['top1_label'] for r in records); per=[]
for r in records:
 per.append(r)
weak=sorted([r for r in records if not r['top1_correct']],key=lambda r:r['top1_score'],reverse=True)
low=sorted([r for r in records if r['top1_correct']],key=lambda r:r['true_score'])[:25]
attractors=Counter(r['top1_label'] for r in weak)
report={'checkpoint':args.checkpoint,'epoch':ckpt['epoch'],'validation_group':'official.png from each raw species folder','n':N,'top1_correct':sum(r['top1_correct'] for r in records),'top3_correct':sum(r['top3_correct'] for r in records),'top5_correct':sum(r['top5_correct'] for r in records),'top1_accuracy':sum(r['top1_correct'] for r in records)/N,'top3_accuracy':sum(r['top3_correct'] for r in records)/N,'top5_accuracy':sum(r['top5_correct'] for r in records)/N,'elapsed_seconds':time.perf_counter()-start,'weak_classes':weak,'low_confidence_correct':low,'error_attractors':[{'label':k,'count':v} for k,v in attractors.most_common()]}
(ROOT/'reports/species_candidate_evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
md=['# 1,025-Species Candidate Evaluation\n','The candidate was evaluated on the reserved `official.png` image for each of the 1,025 catalogue entries. Training used the other five source images per class. This is a clean domain holdout for the supplied raw dataset, but it is not an independent real-game benchmark.\n\n',f"- Top-1: **{report['top1_accuracy']:.4%}** ({report['top1_correct']}/{N})\n",f"- Top-3: **{report['top3_accuracy']:.4%}** ({report['top3_correct']}/{N})\n",f"- Top-5: **{report['top5_accuracy']:.4%}** ({report['top5_correct']}/{N})\n",f"- Evaluation elapsed: **{report['elapsed_seconds']:.2f}s** on CPU.\n\n",'## Highest-confidence errors\n']
for r in weak[:40]: md.append(f"- **{r['true_label']}** → `{r['top1_label']}` at {r['top1_score']:.4f}; true score {r['true_score']:.4f}.\n")
md += ['\n## Repeated error attractors\n']
for r in report['error_attractors'][:20]: md.append(f"- `{r['label']}` received {r['count']} incorrect top-1 predictions.\n")
md += ['\n## Lowest-confidence correct classes\n']
for r in low: md.append(f"- **{r['true_label']}** correct at {r['true_score']:.4f}.\n")
(ROOT/'reports/SPECIES_CANDIDATE_EVALUATION.md').write_text(''.join(md),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['epoch','n','top1_accuracy','top3_accuracy','top5_accuracy','elapsed_seconds']},indent=2))
print('weak',len(weak),'attractors',report['error_attractors'][:10])
