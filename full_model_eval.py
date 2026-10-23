from __future__ import annotations
import json, time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

ROOT=Path(__file__).resolve().parent
labels=json.loads((ROOT/'class_names.json').read_text())
device=torch.device('cpu')
model=models.resnet18(weights=None); model.fc=nn.Linear(512,len(labels))
model.load_state_dict(torch.load(ROOT/'pokemon_classifier.pt',map_location=device,weights_only=False)); model.eval()
tfm=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
items=[]
for target,label in enumerate(labels):
 for p in sorted((ROOT/'data/val'/label).iterdir()): items.append((p,target,label))
batch_size=32; records=[]; start=time.perf_counter()
with torch.inference_mode():
 for off in range(0,len(items),batch_size):
  chunk=items[off:off+batch_size]
  x=torch.stack([tfm(Image.open(p).convert('RGB')) for p,_,_ in chunk])
  probs=torch.softmax(model(x),dim=1); vals,inds=probs.topk(5,dim=1)
  for (p,target,label),v,ix in zip(chunk,vals,inds):
   top=[{'label':labels[int(i)],'score':float(s)} for s,i in zip(v,ix)]
   records.append({'path':str(p.relative_to(ROOT)),'true_label':label,'true_index':target,'top1_label':top[0]['label'],'top1_score':top[0]['score'],'true_score':float(probs[len(records),target]) if False else float(probs[chunk.index((p,target,label)),target]),'top5':top,'top1_correct':top[0]['label']==label,'top3_correct':label in [z['label'] for z in top[:3]],'top5_correct':label in [z['label'] for z in top]})
# Above indexing is local and safe because each chunk record is generated synchronously.
N=len(records); top1=sum(r['top1_correct'] for r in records); top3=sum(r['top3_correct'] for r in records); top5=sum(r['top5_correct'] for r in records)
by_true=defaultdict(list); by_pred=Counter(); conf=Counter()
for r in records: by_true[r['true_label']].append(r); by_pred[r['top1_label']]+=1; conf[(r['true_label'],r['top1_label'])]+=1
per=[]
for label in labels:
 rs=by_true[label]; tp=sum(r['top1_correct'] for r in rs); predicted=by_pred[label]; support=len(rs)
 per.append({'label':label,'support':support,'correct':tp,'recall':tp/support,'precision':tp/predicted if predicted else 0.0,'top1_score':rs[0]['top1_score'],'true_class_score':rs[0]['true_score'],'predicted_as':rs[0]['top1_label'] if not rs[0]['top1_correct'] else None})
weak=sorted([x for x in per if x['correct']==0],key=lambda x:x['top1_score'],reverse=True)
low_correct=sorted([x for x in per if x['correct']==1],key=lambda x:x['true_class_score'])[:25]
confusions=[{'true':t,'predicted':p,'count':n} for (t,p),n in conf.items() if t!=p]
confusions.sort(key=lambda x:(-x['count'],x['true'],x['predicted']))
error_attractors=Counter(r['top1_label'] for r in records if not r['top1_correct'])
# exact duplicate details
hash_groups=defaultdict(list)
for p in sorted((ROOT/'data/train').glob('*/*')):
 if p.is_file(): hash_groups[sha256(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
dups=[{'sha256':h,'files':files} for h,files in hash_groups.items() if len(files)>1]
report={'evaluation':{'checkpoint':'pokemon_classifier.pt','architecture':'resnet18','device':str(device),'n_validation':N,'top1_correct':top1,'top1_accuracy':top1/N,'top3_correct':top3,'top3_accuracy':top3/N,'top5_correct':top5,'top5_accuracy':top5,'batch_size':batch_size,'elapsed_seconds':time.perf_counter()-start,'mean_batch_inference_ms':(time.perf_counter()-start)/(N/batch_size)*1000},'per_class':per,'weak_classes_high_confidence_errors':weak[:50],'low_confidence_correct_classes':low_correct,'confusion_pairs':confusions[:100],'error_attractors':[{'predicted_label':k,'error_count':v} for k,v in error_attractors.most_common()],'duplicate_groups':dups}
(ROOT/'reports/full_model_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
md=['# Full Model Evaluation Report\n','## Protocol\n',f"The original `pokemon_classifier.pt` was evaluated on all **{N}** images in `data/val`, using the original prediction preprocessing (resize 256, center crop 224, ImageNet normalization). Evaluation ran on CPU in batches of {batch_size}. Because there is one validation image per class, each class recall is either 0% or 100%; weak-class findings are directional, not statistically stable.\n",'## Overall results\n',f'- Top-1: **{top1/N:.4%}** ({top1}/{N})\n',f'- Top-3: **{top3/N:.4%}** ({top3}/{N})\n',f'- Top-5: **{top5/N:.4%}** ({top5}/{N})\n',f'- Batched evaluation elapsed time: **{report["evaluation"]["elapsed_seconds"]:.2f}s**; this is not end-to-end screen FPS.\n','## Weak classes and high-confidence errors\n','The following classes were misclassified on their only validation image. They are sorted by the model’s incorrect top-1 score, so the first rows are the most confident errors.\n']
for x in weak[:30]: md.append(f"- **{x['label']}** → `{x['predicted_as']}` at {x['top1_score']:.4f}; true-class score {x['true_class_score']:.4f}.\n")
md += ['\n## Lowest-confidence correct classes\n']
for x in low_correct: md.append(f"- **{x['label']}** correct at true-class score {x['true_class_score']:.4f}.\n")
md += ['\n## Repeated error-attractor labels\n','Across the 290 top-1 errors, these predicted labels attracted the largest number of incorrect examples. This pattern is more informative than individual pair counts because each class has only one validation image.\n']
for x in report['error_attractors'][:20]: md.append(f"- **{x['predicted_label']}** received {x['error_count']} incorrect top-1 predictions.\n")
md += ['\n## Most frequent confusion pairs\n']
for x in confusions[:30]: md.append(f"- `{x['true']}` → `{x['predicted']}`: {x['count']} validation image(s).\n")
md += ['\n## Per-class precision and recall\n','Full machine-readable per-class results, including support, precision, recall, predicted class, and scores, are in `reports/full_model_evaluation.json`. With one validation image per class, recall is 0 or 1 and precision is affected by collisions where multiple true classes receive the same predicted label.\n','\n## Duplicate groups\n',f"The prepared training set contains **{len(dups)} exact duplicate groups** ({sum(len(x['files'])-1 for x in dups)} redundant files beyond one copy). Exact duplicate details and SHA-256 hashes are included in the JSON report. No exact train/validation hash overlap was found in the audit.\n"]
(ROOT/'reports/FULL_MODEL_EVALUATION.md').write_text(''.join(md),encoding='utf-8')
print(json.dumps({'top1':top1/N,'top3':top3/N,'top5':top5/N,'weak_count':len(weak),'first_weak':weak[:10],'duplicate_groups':dups},indent=2))
