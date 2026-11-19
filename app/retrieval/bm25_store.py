from typing import Collection, Iterable, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.models import ChunkRecord
from app.text import tokenize


class Bm25Store:
    def __init__(self, chunks: Iterable[ChunkRecord]) -> None:
        ids: list[int] = []
        companies: list[str] = []
        corpus: list[list[str]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.retrieval_text)
            if tokens:
                ids.append(chunk.vector_id)
                companies.append(chunk.company)
                corpus.append(tokens)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate vector ids in BM25 corpus")
        self._ids = np.asarray(ids, dtype=np.int64)
        self._companies = np.asarray(companies, dtype=str)
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(
        self, query: str, top_k: int, companies: Optional[Collection[str]] = None
    ) -> list[tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self._bm25 is None or (companies is not None and not companies):
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        mask = scores > 0
        if companies is not None:
            mask &= np.isin(self._companies, list(companies))
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return []
        order = np.lexsort((self._ids[candidates], -scores[candidates]))[:top_k]
        selected = candidates[order]
        return [(int(self._ids[i]), float(scores[i])) for i in selected]
