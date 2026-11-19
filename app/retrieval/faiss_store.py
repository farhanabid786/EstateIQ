import logging
import os
from pathlib import Path
from typing import Collection, Mapping, Optional, Sequence

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_TYPE_NAME = "IndexIDMap2(IndexFlatIP)"
MAX_SEARCH_CANDIDATES = 20000
NORM_TOLERANCE = 1e-3
FILTER_FETCH_FACTOR = 4


class FaissStoreError(ValueError):
    pass


class FaissStore:
    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise FaissStoreError("dimension must be positive")
        self.dimension = dimension
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._companies: dict[int, str] = {}

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def vector_ids(self) -> list[int]:
        return sorted(self._companies)

    def company_of(self, vector_id: int) -> Optional[str]:
        return self._companies.get(vector_id)

    def _check_vectors(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise FaissStoreError(
                f"Expected vectors with {self.dimension} columns, got shape {array.shape}"
            )
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise FaissStoreError("Vectors contain non-finite values")
        return array

    def add(self, vector_ids: Sequence[int], vectors: np.ndarray, companies: Sequence[str]) -> None:
        array = self._check_vectors(vectors)
        if not len(vector_ids) == len(companies) == array.shape[0]:
            raise FaissStoreError("vector_ids, companies and vectors must have the same length")
        if array.shape[0] == 0:
            return
        ids = [int(i) for i in vector_ids]
        if min(ids) < 0:
            raise FaissStoreError("vector ids must be non-negative")
        if len(set(ids)) != len(ids):
            raise FaissStoreError("Duplicate vector ids in batch")
        clashes = [i for i in ids if i in self._companies]
        if clashes:
            raise FaissStoreError(f"Vector id {clashes[0]} already exists in the index")
        if any(not isinstance(c, str) or not c for c in companies):
            raise FaissStoreError("Every vector needs a non-empty company")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
            raise FaissStoreError("Vectors must be L2-normalized")
        self._index.add_with_ids(array, np.asarray(ids, dtype=np.int64))
        self._companies.update(zip(ids, companies))

    def search(
        self, query: np.ndarray, top_k: int, companies: Optional[Collection[str]] = None
    ) -> list[tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = self._check_vectors(np.asarray(query).reshape(1, -1) if np.ndim(query) == 1 else query)
        if vector.shape[0] != 1:
            raise FaissStoreError("search expects a single query vector")
        if self.size == 0:
            return []
        allowed = None if companies is None else frozenset(companies)
        if allowed is not None and not allowed:
            return []

        cap = min(self.size, MAX_SEARCH_CANDIDATES)
        fetch = min(top_k if allowed is None else top_k * FILTER_FETCH_FACTOR, cap)
        while True:
            scores, ids = self._index.search(vector, fetch)
            results = [
                (int(i), float(s))
                for i, s in zip(ids[0], scores[0])
                if i != -1 and (allowed is None or self._companies[int(i)] in allowed)
            ]
            if allowed is None or len(results) >= top_k or fetch >= cap:
                return results[:top_k]
            fetch = min(fetch * FILTER_FETCH_FACTOR, cap)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        payload = faiss.serialize_index(self._index)
        with temp_path.open("wb") as handle:
            handle.write(payload.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    @classmethod
    def load(cls, path: Path, companies: Mapping[int, str]) -> "FaissStore":
        try:
            payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            index = faiss.deserialize_index(payload)
        except (OSError, RuntimeError) as exc:
            raise FaissStoreError(f"Cannot read FAISS index {path.name}") from exc
        if not isinstance(index, faiss.IndexIDMap2):
            raise FaissStoreError("FAISS index is not an IndexIDMap2")
        if not isinstance(faiss.downcast_index(index.index), faiss.IndexFlatIP):
            raise FaissStoreError("FAISS index is not backed by IndexFlatIP")
        ids = [int(i) for i in faiss.vector_to_array(index.id_map)]
        if len(set(ids)) != len(ids):
            raise FaissStoreError("FAISS index contains duplicate vector ids")
        unmapped = [i for i in ids if i not in companies]
        if unmapped:
            raise FaissStoreError(f"{len(unmapped)} FAISS vector id(s) have no chunk metadata")
        store = cls(index.d)
        store._index = index
        store._companies = {i: companies[i] for i in ids}
        return store
