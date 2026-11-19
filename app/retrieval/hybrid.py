import logging
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

import numpy as np

from app.config import Settings
from app.retrieval.index_loader import LoadedIndex

logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "dense", "bm25"]


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class HybridResult:
    vector_id: int
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, vector_id in enumerate(ranking, start=1):
            if vector_id in seen:
                continue
            seen.add(vector_id)
            scores[vector_id] = scores.get(vector_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(
    dense: Sequence[tuple[int, float]],
    lexical: Sequence[tuple[int, float]],
    k: int,
    top_k: int,
) -> list[HybridResult]:
    dense_info: dict[int, tuple[int, float]] = {}
    for rank, (vector_id, score) in enumerate(dense, start=1):
        dense_info.setdefault(vector_id, (rank, score))
    lexical_info: dict[int, tuple[int, float]] = {}
    for rank, (vector_id, score) in enumerate(lexical, start=1):
        lexical_info.setdefault(vector_id, (rank, score))

    scores = reciprocal_rank_fusion(
        [[v for v, _ in dense], [v for v, _ in lexical]], k
    )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    results = []
    for vector_id, score in ordered:
        d_rank, d_score = dense_info.get(vector_id, (None, None))
        b_rank, b_score = lexical_info.get(vector_id, (None, None))
        results.append(HybridResult(vector_id, score, d_rank, d_score, b_rank, b_score))
    return results


def has_sufficient_evidence(
    results: Sequence[HybridResult], min_dense_score: float, min_bm25_score: float
) -> bool:
    return any(
        (r.dense_score is not None and r.dense_score >= min_dense_score)
        or (r.bm25_score is not None and r.bm25_score >= min_bm25_score)
        for r in results
    )


class HybridRetriever:
    def __init__(self, index: LoadedIndex, embedder: QueryEmbedder, settings: Settings) -> None:
        self._index = index
        self._embedder = embedder
        self._settings = settings

    def retrieve(
        self,
        query: str,
        companies: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[HybridResult]:
        if not query or not query.strip():
            return []
        top_k = top_k or self._settings.hybrid_top_k
        allowed = None if companies is None else tuple(companies)
        if allowed is not None and not allowed:
            return []

        dense: list[tuple[int, float]] = []
        lexical: list[tuple[int, float]] = []
        if mode in ("hybrid", "dense") and self._index.faiss.size:
            vector = self._embedder.embed_query(query)
            dense = self._index.faiss.search(vector, self._settings.dense_top_k, companies=allowed)
        if mode in ("hybrid", "bm25") and self._index.bm25.size:
            lexical = self._index.bm25.search(query, self._settings.bm25_top_k, companies=allowed)

        results = fuse(dense, lexical, self._settings.rrf_k, top_k)
        if allowed is not None:
            permitted = set(allowed)
            results = [r for r in results if self._index.faiss.company_of(r.vector_id) in permitted]
        logger.debug("Hybrid retrieval: %d dense, %d lexical, %d fused", len(dense), len(lexical), len(results))
        return results
