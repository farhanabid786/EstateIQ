import re
import unicodedata
from typing import Sequence

from app.models import ChunkRecord, SourceDocument

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"[.!?؟۔…][\"'”’»)\]]*\s+|\n")
_WHITESPACE = re.compile(r"\s+")


def _last_boundary(pattern: re.Pattern[str], text: str, start: int, limit: int, min_end: int) -> int:
    cut = 0
    for match in pattern.finditer(text, start, limit):
        if match.end() >= min_end:
            cut = match.end()
    return cut


def _choose_end(text: str, start: int, size: int, min_end: int) -> int:
    limit = min(start + size, len(text))
    if limit == len(text):
        return limit
    for pattern in (_PARAGRAPH, _SENTENCE, _WHITESPACE):
        cut = _last_boundary(pattern, text, start, limit, min_end)
        if cut:
            return cut
    while limit > start + 1 and unicodedata.category(text[limit]).startswith("M"):
        limit -= 1
    return limit


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    if overlap == 0:
        return end
    candidate = end - overlap
    if candidate <= start:
        return end
    match = _WHITESPACE.search(text, candidate, end)
    if match:
        candidate = match.end()
    return candidate if start < candidate < end else end


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size < 1:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start, previous_end = 0, 0
    while start < len(text):
        min_end = max(start + max(1, size // 3), previous_end + 1)
        end = _choose_end(text, start, size, min_end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        previous_end = end
        start = _next_start(text, start, end, overlap)
    return chunks


def chunk_document(
    document: SourceDocument, size: int, overlap: int, first_vector_id: int = 0
) -> list[ChunkRecord]:
    pieces = split_text(document.content, size, overlap)
    return [
        ChunkRecord(
            vector_id=first_vector_id + position,
            chunk_id=f"{document.document_id}::{position:04d}",
            document_id=document.document_id,
            position=position,
            company=document.company,
            category=document.category,
            title=document.title,
            source_url=document.source_url,
            language=document.language,
            page_number=document.page_number,
            published_at=document.published_at,
            collected_at=document.collected_at,
            text=piece,
        )
        for position, piece in enumerate(pieces)
    ]


def build_chunks(documents: Sequence[SourceDocument], size: int, overlap: int) -> list[ChunkRecord]:
    ordered = sorted(documents, key=lambda d: (d.company, d.document_id))
    ids = [d.document_id for d in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("document_id values must be unique before chunking")
    chunks: list[ChunkRecord] = []
    for document in ordered:
        chunks.extend(chunk_document(document, size, overlap, first_vector_id=len(chunks)))
    return chunks
