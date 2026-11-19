import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from app.api.schemas import SourceRef
from app.config import Settings
from app.database.metadata import MetadataStore
from app.models import ChunkRecord
from app.retrieval.hybrid import HybridResult

logger = logging.getLogger(__name__)

GAP_MARKER = "\n[...]\n"


@dataclass(frozen=True)
class ContextChunk:
    chunk: ChunkRecord
    score: float
    is_adjacent: bool


@dataclass(frozen=True)
class ContextBlock:
    citation_id: str
    source: SourceRef
    page_number: Optional[int]
    chunks: tuple[ContextChunk, ...]
    text: str
    score: float

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk.chunk_id for c in self.chunks)


@dataclass(frozen=True)
class EvidenceContext:
    blocks: tuple[ContextBlock, ...]
    sufficient: bool
    reason: Optional[str] = None

    @property
    def char_count(self) -> int:
        return sum(len(c.chunk.text) for b in self.blocks for c in b.chunks)

    def block_for(self, citation_id: str) -> Optional[ContextBlock]:
        return next((b for b in self.blocks if b.citation_id == citation_id), None)


class ContextBuilder:
    def __init__(self, metadata: MetadataStore, settings: Settings) -> None:
        self._metadata = metadata
        self._settings = settings

    def build(self, results: Sequence[HybridResult]) -> EvidenceContext:
        if not results:
            return EvidenceContext((), False, "no_results")
        selected = list(results[: self._settings.max_context_chunks])
        chunks = self._metadata.get_chunks([r.vector_id for r in selected])

        included: dict[int, ContextChunk] = {}
        remaining = self._settings.max_context_chars
        for result, chunk in zip(selected, chunks):
            if chunk.vector_id in included or remaining <= 0:
                continue
            if len(chunk.text) > remaining:
                if included:
                    continue
                chunk = chunk.model_copy(update={"text": chunk.text[:remaining]})
            included[chunk.vector_id] = ContextChunk(chunk, result.rrf_score, False)
            remaining -= len(chunk.text)

        remaining = self._expand(selected, chunks, included, remaining)
        return self._assemble(selected, included)

    def _expand(
        self,
        selected: Sequence[HybridResult],
        chunks: Sequence[ChunkRecord],
        included: dict[int, ContextChunk],
        remaining: int,
    ) -> int:
        window = self._settings.adjacent_chunk_window
        if window <= 0:
            return remaining
        top = list(zip(selected, chunks))[: self._settings.expansion_top_n]
        for result, chunk in top:
            if chunk.vector_id not in included:
                continue
            for neighbour in self._metadata.adjacent_chunks(chunk.document_id, chunk.position, window):
                if neighbour.vector_id in included or len(neighbour.text) > remaining:
                    continue
                included[neighbour.vector_id] = ContextChunk(neighbour, result.rrf_score, True)
                remaining -= len(neighbour.text)
        return remaining

    def _assemble(
        self, selected: Sequence[HybridResult], included: dict[int, ContextChunk]
    ) -> EvidenceContext:
        by_document: dict[str, list[ContextChunk]] = {}
        for item in included.values():
            by_document.setdefault(item.chunk.document_id, []).append(item)

        rank_of = {r.vector_id: rank for rank, r in enumerate(selected)}
        sources = {s.document_id: s for s in self._metadata.get_sources(list(by_document))}

        def best_rank(items: list[ContextChunk]) -> int:
            return min((rank_of[i.chunk.vector_id] for i in items if not i.is_adjacent), default=len(selected))

        blocks = []
        ordered = sorted(by_document.items(), key=lambda kv: (best_rank(kv[1]), kv[0]))
        for document_id, items in ordered:
            source = sources.get(document_id)
            if source is None:
                logger.warning("Dropping evidence for a document missing from metadata")
                continue
            items.sort(key=lambda i: i.chunk.position)
            blocks.append(
                ContextBlock(
                    citation_id=f"S{len(blocks) + 1}",
                    source=source,
                    page_number=items[0].chunk.page_number,
                    chunks=tuple(items),
                    text=self._join(items),
                    score=max(i.score for i in items),
                )
            )
        if not blocks:
            return EvidenceContext((), False, "missing_source_metadata")
        return EvidenceContext(tuple(blocks), True)

    @staticmethod
    def _join(items: list[ContextChunk]) -> str:
        parts = [items[0].chunk.text]
        for previous, current in zip(items, items[1:]):
            gap = current.chunk.position - previous.chunk.position > 1
            parts.append(GAP_MARKER if gap else "\n")
            parts.append(current.chunk.text)
        return "".join(parts)
