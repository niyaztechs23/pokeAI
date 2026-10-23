from fastapi.testclient import TestClient
from api import app
client=TestClient(app)
r=client.get('/health'); assert r.status_code==200 and r.json()['status']=='ok'
r=client.get('/model/info'); assert r.status_code==200 and r.json()['num_classes']==1025
with open('pikachu.png','rb') as f: r=client.post('/predict',files={'file':('pikachu.png',f,'image/png')},params={'top_k':3})
assert r.status_code==200, r.text
body=r.json(); assert body['label'] and len(body['top_k'])==3 and 0 <= body['score'] <= 1
r=client.post('/predict',files={'file':('bad.txt',b'not an image','text/plain')}); assert r.status_code==400
print('API smoke tests passed')
