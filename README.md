# EstateIQ

EstateIQ is a source-grounded real-estate intelligence assistant for DarGlobal and indexed public official sources.

## Local setup

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-backend.txt -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Copy `.env.example` to `.env` for local configuration. Never commit `.env`.

## Deployment

The production image uses `Dockerfile` and `render.yaml`. The repository includes the prebuilt
`data/index` artifacts; do not rebuild them unless the source dataset changes.
