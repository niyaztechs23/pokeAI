"""Reusable PokeAI inference service layer; preserves the original checkpoint format."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class PokemonClassifier:
    def __init__(self, weights: str|Path='models/species_resnet18_robust_best.pt', labels: str|Path='class_names.json', device: str|None=None):
        self.weights_path=Path(weights); self.labels_path=Path(labels)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        state=torch.load(self.weights_path,map_location=self.device,weights_only=False)
        embedded_labels=state.get('class_names') if isinstance(state,dict) else None
        self.class_names=embedded_labels if embedded_labels else json.loads(self.labels_path.read_text(encoding='utf-8'))
        if not isinstance(self.class_names,list) or not self.class_names: raise ValueError('class names must be a non-empty list')
        self.model=models.resnet18(weights=None); self.model.fc=nn.Linear(self.model.fc.in_features,len(self.class_names))
        if isinstance(state,dict) and 'model' in state: state=state['model']
        self.model.load_state_dict(state); self.model.to(self.device).eval()
        self.transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def info(self)->dict[str,Any]: return {'architecture':'resnet18','num_classes':len(self.class_names),'weights':str(self.weights_path),'device':str(self.device),'preprocessing':'Resize(256), CenterCrop(224), ImageNet normalization'}
    def predict_image(self,image:Image.Image,top_k:int=3,threshold:float|None=None)->dict[str,Any]:
        if top_k<1 or top_k>len(self.class_names): raise ValueError('top_k out of range')
        x=self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.inference_mode(): probs=torch.softmax(self.model(x),dim=1)[0]
        vals,inds=probs.topk(top_k); best=float(vals[0]); unknown=threshold is not None and best < threshold
        return {'label':None if unknown else self.class_names[inds[0].item()],'score':best,'unknown':unknown,'top_k':[{'label':self.class_names[i.item()],'score':float(v)} for v,i in zip(vals,inds)]}
    def predict_bytes(self,data:bytes,**kwargs):
        from io import BytesIO
        try: image=Image.open(BytesIO(data))
        except Exception as e: raise ValueError('uploaded data is not a supported image') from e
        return self.predict_image(image,**kwargs)
