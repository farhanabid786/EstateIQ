FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MAX_CONCURRENT_REQUESTS=1 \
    LLM_MAX_OUTPUT_TOKENS=800

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt

# Bake the embedding model into the image so the service never downloads it at runtime.
COPY app ./app
COPY frontend ./frontend
COPY data/index ./data/index

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
