"""Local-only API for the verified PokeAI classifier."""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from inference import PokemonClassifier
app=FastAPI(title='PokeAI Local API',version='0.2.0')
_model=None
def get_model():
 global _model
 if _model is None: _model=PokemonClassifier()
 return _model
@app.get('/health')
def health(): return {'status':'ok','model_loaded':_model is not None}
@app.get('/model/info')
def model_info(): return get_model().info()
@app.post('/predict')
async def predict(file:UploadFile=File(...), top_k:int=Query(3,ge=1,le=20), threshold:float|None=Query(None,ge=0,le=1)):
 data=await file.read()
 if len(data)>10*1024*1024: raise HTTPException(413,'image exceeds 10 MiB limit')
 try: return get_model().predict_bytes(data,top_k=top_k,threshold=threshold)
 except ValueError as e: raise HTTPException(400,str(e))
 except Exception as e: raise HTTPException(500,'model inference failed') from e
