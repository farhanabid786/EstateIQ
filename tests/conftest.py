import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings, get_settings
from app.retrieval.embeddings import EmbeddingProvider
from app.text import tokenize

DIM = 384


class HashingModel:
    """Offline stand-in for FastEmbed: hashed bag-of-words vectors."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.batch_sizes: list[int] = []

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim] += 1.0
        return vector

    def passage_embed(self, texts, **_):
        batch = list(texts)
        self.batch_sizes.append(len(batch))
        for text in batch:
            yield self._vector(text)

    def query_embed(self, query, **_):
        items = [query] if isinstance(query, str) else list(query)
        for text in items:
            yield self._vector(text)


@pytest.fixture
def hashing_model_class():
    return HashingModel


@pytest.fixture
def settings() -> Settings:
    return replace(get_settings(), chunk_size_chars=300, chunk_overlap_chars=50, embedding_dim=DIM)


@pytest.fixture
def provider() -> EmbeddingProvider:
    return EmbeddingProvider("test-model", DIM, batch_size=4, model_factory=lambda name: HashingModel())


@pytest.fixture
def make_record():
    counter = {"n": 0}

    def factory(**overrides) -> dict:
        counter["n"] += 1
        n = counter["n"]
        content = overrides.get("content", f"Synthetic content number {n} about fictional topic {n}.")
        record = {
            "document_id": f"doc_{n}",
            "company": "darglobal",
            "title": f"Synthetic Title {n}",
            "content": content,
            "source_url": f"https://example.test/page-{n}",
            "content_type": "webpage",
            "language": "en",
            "collected_at": "2026-11-18T12:00:00Z",
            "published_at": None,
            "file_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "category": "other",
        }
        record.update(overrides)
        if "file_hash" not in overrides:
            record["file_hash"] = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
        return record

    return factory


@pytest.fixture
def write_jsonl():
    def writer(path: Path, records: list, raw_lines: list = ()) -> Path:
        lines = [json.dumps(r, ensure_ascii=False) for r in records] + list(raw_lines)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return writer


class FakeLLM:
    is_configured = True

    def __init__(self, responses=None, error=None) -> None:
        self.responses = list(responses or ["Answer based on the sources. [S1]"])
        self.error = error
        self.calls: list = []

    async def generate(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_llm_class():
    return FakeLLM


@pytest.fixture
def rag_settings(settings):
    return replace(
        settings, min_dense_score=0.25, min_bm25_score=1.0, hybrid_top_k=8, max_context_chunks=6,
        max_context_chars=6000, adjacent_chunk_window=1, expansion_top_n=2, llm_max_retries=0,
    )


@pytest.fixture
def rag_corpus(make_record):
    zorvia = " ".join(
        f"Zorvia Heights paragraph {i} describes a fictional residential tower near the harbour with sea views."
        for i in range(10)
    )
    return [
        make_record(title="Zorvia Heights", content=zorvia, category="project", page_number=3),
        make_record(title="Platform Overview", company="wasalt", category="service",
                    content="The fictional Wasalt platform lets visitors browse apartments for rent and read market guides."),
        make_record(title="مشروع النخيل الأزرق", language="ar", category="project",
                    content="مشروع النخيل الأزرق مجمع سكني افتراضي يقع قرب الشاطئ في مدينة خيالية."),
        make_record(title="Quarterly Report", category="investor",
                    content="The fictional quarterly report discusses shareholder returns and revenue growth for investors."),
    ]


@pytest.fixture
def loaded_index(tmp_path, rag_corpus, write_jsonl, rag_settings, provider):
    from app.ingestion.build_index import build_index
    from app.retrieval.index_loader import load_index

    source = write_jsonl(tmp_path / "docs.jsonl", rag_corpus)
    out = tmp_path / "index"
    build_index(source, out, rag_settings, provider, built_at="2026-11-18T12:00:00Z")
    return load_index(rag_settings, out)


@pytest.fixture
def anyio_backend():
    return "asyncio"
