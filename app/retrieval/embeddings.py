import logging
import time
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import httpx
import numpy as np

from app.config import Settings, get_settings
from app.memory_diagnostics import log_rss

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
JINA_DIMENSIONS = 512
PROGRESS_LOG_EVERY_BATCHES = 10
ModelFactory = Callable[[str], Any]


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider:
    def __init__(
        self,
        model_name: str = JINA_MODEL,
        dimension: int = JINA_DIMENSIONS,
        batch_size: int = 32,
        model_factory: Optional[ModelFactory] = None,
        api_key: str = "",
        api_url: str = JINA_API_URL,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if dimension < 1 or batch_size < 1:
            raise ValueError("dimension and batch_size must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.provider_name = "jina"
        self.api_key = api_key
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._factory = model_factory
        self._model: Any = None
        self._client = http_client
        self._owns_client = http_client is None
        if http_client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds))

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingProvider":
        return cls(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            api_key=settings.jina_api_key,
            api_url=settings.jina_api_url,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or self._factory is not None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None if self._factory is not None else self.is_configured

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def verify_registry(self) -> None:
        if self.model_name != JINA_MODEL:
            raise EmbeddingError(f"{self.model_name} is not the configured Jina model")
        if self.dimension != JINA_DIMENSIONS:
            raise EmbeddingError(f"Jina {JINA_MODEL} requires dimension {JINA_DIMENSIONS}")

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        validated = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")
            validated.append(text)
        return validated

    def _request(self, texts: Sequence[str], task: str) -> np.ndarray:
        if not self.api_key and self._factory is None:
            raise EmbeddingError("JINA_API_KEY is not configured")
        if self._factory is not None:
            if self._model is None:
                self._model = self._factory(self.model_name)
            model = self._model
            method = "passage_embed" if task == "retrieval.passage" else "query_embed"
            return np.asarray(list(getattr(model, method)(texts)), dtype=np.float32)

        payload = {
            "model": self.model_name,
            "input": list(texts),
            "task": task,
            "dimensions": self.dimension,
            "embedding_type": "float",
            "normalized": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
                    continue
                raise EmbeddingError("Jina embedding request timed out") from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError("Jina embedding request failed") from exc

            if response.status_code == 401:
                raise EmbeddingError("Jina rejected the API key")
            if response.status_code == 403:
                raise EmbeddingError("Jina API access is forbidden")
            if response.status_code == 400:
                raise EmbeddingError("Jina rejected the embedding request")
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** attempt))
                    continue
                raise EmbeddingError(
                    "Jina embedding service is rate-limited or unavailable"
                    if response.status_code == 429
                    else "Jina embedding service returned an upstream error"
                )
            if response.status_code != 200:
                raise EmbeddingError(f"Jina returned HTTP {response.status_code}")

            try:
                body = response.json()
            except ValueError as exc:
                raise EmbeddingError("Jina returned malformed JSON") from exc
            if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                raise EmbeddingError("Jina returned an unexpected response schema")

            rows = body["data"]
            if len(rows) != len(texts):
                raise EmbeddingError(
                    f"Jina returned {len(rows)} vectors for {len(texts)} inputs"
                )
            try:
                rows.sort(key=lambda row: int(row["index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise EmbeddingError("Jina returned vectors with invalid indexes") from exc
            if [int(row["index"]) for row in rows] != list(range(len(texts))):
                raise EmbeddingError("Jina returned out-of-order or missing vector indexes")

            vectors = []
            for row in rows:
                vector = row.get("embedding")
                if not isinstance(vector, list):
                    raise EmbeddingError("Jina returned an invalid embedding")
                if len(vector) != self.dimension:
                    raise EmbeddingError(
                        f"Jina returned dimension {len(vector)}, expected {self.dimension}"
                    )
                vectors.append(vector)
            return np.asarray(vectors, dtype=np.float32)

        raise EmbeddingError("Jina embedding request failed")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim != 2:
            raise EmbeddingError("Embedding response must be a two-dimensional array")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding model returned non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise EmbeddingError("Embedding model returned a zero vector")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def _encode(self, texts: list[str], task: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = []
        for number, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start:start + self.batch_size]
            try:
                vectors = self._request(batch, task)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("Embedding inference failed") from exc
            if vectors.ndim != 2 or vectors.shape != (len(batch), self.dimension):
                raise EmbeddingError(
                    f"Expected embeddings of shape ({len(batch)}, {self.dimension}), got {vectors.shape}"
                )
            batches.append(vectors)
            if number % PROGRESS_LOG_EVERY_BATCHES == 0:
                logger.info("Embedded %d/%d texts", min(start + len(batch), len(texts)), len(texts))
        vectors = self._normalize(np.vstack(batches))
        log_rss("after_query_embedding" if task == "retrieval.query" else "after_document_embedding")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(self._validate_texts(texts), "retrieval.passage")

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query must be a non-empty string")
        return self._encode([text], "retrieval.query")[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider.from_settings(get_settings())
