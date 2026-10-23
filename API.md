# API Status

The original supplied project had no FastAPI service. The audited working copy now provides a tested localhost-only implementation in `api.py`, backed by `inference.py`.

## Implemented endpoints

The audited working copy now includes `api.py` and `inference.py`. Start it with `uvicorn api:app --host 127.0.0.1 --port 8000`. `GET /health`, `GET /model/info`, and `POST /predict` are implemented and covered by `test_api.py`. The service accepts multipart image uploads, limits uploads to 10 MiB, supports `top_k` and an optional explicitly supplied threshold, and binds to localhost in the documented command.

## Sample image test

The current API loads `models/species_resnet18_robust_best.pt` by default and exposes 1,025 species labels. A `pikachu.png` request returned HTTP 200, but the fine-tuned model returned **Diglett** as its top label; Pikachu was not in the top five. The response is saved in `reports/api_sample_robust_test.json`. Earlier candidate responses are preserved in `reports/api_sample_candidate_test.json` and `reports/api_sample_test.json`. This remains a verified example that the score is not a correctness guarantee.
