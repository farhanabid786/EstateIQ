import logging
import threading
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

ModelFactory = Callable[[str], Any]
PROGRESS_LOG_EVERY_BATCHES = 10


class EmbeddingError(RuntimeError):
    pass


def _fastembed_factory(model_name: str) -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


class EmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: int = 32,
        model_factory: Optional[ModelFactory] = None,
    ) -> None:
        if dimension < 1 or batch_size < 1:
            raise ValueError("dimension and batch_size must be positive")
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._factory = model_factory or _fastembed_factory
        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingProvider":
        return cls(settings.embedding_model, settings.embedding_dim)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def verify_registry(self) -> None:
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry["model"] == self.model_name:
                if entry["dim"] != self.dimension:
                    raise EmbeddingError(
                        f"{self.model_name} has dimension {entry['dim']}, configured {self.dimension}"
                    )
                return
        raise EmbeddingError(f"{self.model_name} is not supported by the installed FastEmbed")

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                logger.info("Loading embedding model %s", self.model_name)
                try:
                    self._model = self._factory(self.model_name)
                except Exception as exc:  # third-party loader raises varied types
                    logger.exception("Embedding model load failed")
                    raise EmbeddingError(
                        "Could not load the embedding model; the first load needs network access "
                        "or a populated FastEmbed cache"
                    ) from exc
            return self._model

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string")
        validated = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")
            validated.append(text)
        return validated

    def _encode(self, texts: list[str], method: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._get_model()
        encode = getattr(model, method)
        batches = []
        for number, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start:start + self.batch_size]
            try:
                array = np.asarray(list(encode(batch)), dtype=np.float32)
            except Exception as exc:  # inference runtime raises varied types
                logger.exception("Embedding inference failed")
                raise EmbeddingError("Embedding inference failed") from exc
            if array.ndim != 2 or array.shape != (len(batch), self.dimension):
                raise EmbeddingError(
                    f"Expected embeddings of shape ({len(batch)}, {self.dimension}), got {array.shape}"
                )
            batches.append(array)
            if number % PROGRESS_LOG_EVERY_BATCHES == 0:
                logger.info("Embedded %d/%d texts", min(start + len(batch), len(texts)), len(texts))
        return self._normalize(np.vstack(batches))

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Embedding model returned non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise EmbeddingError("Embedding model returned a zero vector")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(self._validate_texts(texts), "passage_embed")

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query must be a non-empty string")
        return self._encode([text], "query_embed")[0]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider.from_settings(get_settings())
