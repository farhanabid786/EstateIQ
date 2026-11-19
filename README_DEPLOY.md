# EstateIQ — Run, Test and Deploy

## Local run (PowerShell, Python 3.10.11)
```powershell
cd D:\EstateIQ
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                        # must print Python 3.10.11
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env             # then put your OpenRouter key in .env
python -m pytest -q tests
python -m app.ingestion.build_index     # needs data\processed\documents.jsonl and internet (first model download)
python -m uvicorn app.main:app --port 8000
```
Open http://localhost:8000 (UI), http://localhost:8000/docs (API docs), http://localhost:8000/health.
`/health` returns `"status": "degraded"` until `data/index/` holds a valid, consistent index.

If `frontend/index.html` exists it is served at `/`; otherwise a minimal built-in page (`app/static/fallback/`) is served.

## Endpoints
`GET /health`, `POST /api/chat`, `GET /api/sources`, `GET /api/stats`, `GET /api/projects`, `GET /docs`, `GET /openapi.json`.

## Retrieval evaluation
```powershell
Copy-Item data\eval\questions.template.jsonl data\eval\questions.jsonl
# replace every placeholder with real questions and real document_id values from data\index\chunks.jsonl
python -m scripts.evaluate_retrieval --mode hybrid
python -m scripts.evaluate_retrieval --mode dense
python -m scripts.evaluate_retrieval --mode bm25
```
The script refuses to report when no real labelled questions exist. Results from a handful of questions are not statistically meaningful.

## Deploy on Render Free (Docker)
1. Build the real index locally and keep `data/index/` (four files) in the repository. Commit `frontend/` too (the Dockerfile copies it; remove that line if you have no frontend folder). Never commit `.env`.
2. Push to GitHub. In Render choose New → Blueprint (uses `render.yaml`) or New → Web Service → Docker.
3. Set `OPENROUTER_API_KEY` as a secret environment variable in the Render dashboard. Optionally change `OPENROUTER_MODEL`.
4. Render supplies `PORT`; the container binds `0.0.0.0:$PORT` with one worker. Health check path is `/health`.
5. After deploy, check `/health` shows `index_loaded: true`, then ask one English and one Arabic question through the public URL.

Refresh workflow: scrape locally, run `python -m app.ingestion.build_index`, run tests, commit `data/index/`, Render redeploys. The service never scrapes, indexes or writes durable data at runtime.

## Memory and startup risks (not measured)
- Render Free has 512 MB RAM. Nothing here has been measured on Render, and the Docker image has not been built or run by the author of this code.
- Memory consumers: the Python/FastAPI process, the ONNX embedding model (loaded on the first chat request unless `PRELOAD_EMBEDDING_MODEL=true`), the FAISS index, the BM25 token lists, and onnxruntime working memory.
- The embedding model is baked into the image (`FASTEMBED_CACHE_PATH`) so it is not downloaded at runtime. If that build step fails, fix it rather than letting the service download at request time.
- After the first real build, run the container, send several chat requests and read `docker stats`. If memory is too high: lower `DENSE_TOP_K`/`BM25_TOP_K`, keep one worker, or shrink the corpus.
- The free instance sleeps when idle; the first request after a wake-up loads the model and is slow.

## Privacy
Only the user's question and retrieved public company text go to the OpenRouter free endpoint, whose provider may log requests. Tell users not to enter personal data.
