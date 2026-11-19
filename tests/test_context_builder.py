from dataclasses import replace

import pytest

from app.retrieval.context_builder import ContextBuilder
from app.retrieval.hybrid import HybridResult, HybridRetriever


def _results_for(loaded_index, title: str, count: int = 1) -> list[HybridResult]:
    ids = [i for i in range(loaded_index.faiss.size)
           if loaded_index.metadata.get_chunks([i])[0].title == title]
    return [HybridResult(i, 1.0 / (rank + 1), dense_rank=rank + 1, dense_score=0.9)
            for rank, i in enumerate(ids[:count])]


def test_builds_blocks_with_metadata(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 2)
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(results)
    assert context.sufficient
    block = context.blocks[0]
    assert block.citation_id == "S1"
    assert block.source.title == "Zorvia Heights"
    assert block.source.url.startswith("https://example.test/")
    assert block.source.company == "darglobal"
    assert block.page_number == 3
    assert block.chunk_ids and block.score > 0


def test_adjacent_expansion_marks_neighbours(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 3)[1:2]
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(results)
    items = context.blocks[0].chunks
    assert any(c.is_adjacent for c in items) and any(not c.is_adjacent for c in items)
    positions = [c.chunk.position for c in items]
    assert positions == sorted(positions)


def test_expansion_can_be_disabled_and_is_bounded(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 3)[1:2]
    none = ContextBuilder(loaded_index.metadata, replace(rag_settings, adjacent_chunk_window=0)).build(results)
    assert all(not c.is_adjacent for c in none.blocks[0].chunks)
    top_zero = ContextBuilder(loaded_index.metadata, replace(rag_settings, expansion_top_n=0)).build(results)
    assert all(not c.is_adjacent for c in top_zero.blocks[0].chunks)


def test_no_duplicate_chunks(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 4)
    results = results + results[:2]
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(results)
    ids = [cid for block in context.blocks for cid in block.chunk_ids]
    assert len(ids) == len(set(ids))


def test_context_budget_is_respected(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 4)
    small = replace(rag_settings, max_context_chars=500)
    context = ContextBuilder(loaded_index.metadata, small).build(results)
    assert context.sufficient and context.char_count <= 500


def test_budget_truncates_a_single_oversized_chunk(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 1)
    tiny = replace(rag_settings, max_context_chars=500, adjacent_chunk_window=0)
    context = ContextBuilder(loaded_index.metadata, replace(tiny, max_context_chars=500)).build(results)
    assert context.char_count <= 500


def test_company_separation_and_ranking_order(loaded_index, rag_settings):
    wasalt = _results_for(loaded_index, "Platform Overview")
    dar = _results_for(loaded_index, "Zorvia Heights")
    ranked = [wasalt[0], dar[0]]
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(ranked)
    assert [b.source.company for b in context.blocks] == ["wasalt", "darglobal"]
    assert [b.citation_id for b in context.blocks] == ["S1", "S2"]
    for block in context.blocks:
        assert {c.chunk.company for c in block.chunks} == {block.source.company}


def test_one_block_per_document(loaded_index, rag_settings):
    results = _results_for(loaded_index, "Zorvia Heights", 4)
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(results)
    assert len(context.blocks) == 1


def test_empty_evidence_is_explicitly_insufficient(loaded_index, rag_settings):
    context = ContextBuilder(loaded_index.metadata, rag_settings).build([])
    assert not context.sufficient and context.reason == "no_results" and context.blocks == ()


def test_missing_source_metadata_is_not_fabricated(loaded_index, rag_settings, monkeypatch):
    monkeypatch.setattr(loaded_index.metadata, "get_sources", lambda ids: [])
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(_results_for(loaded_index, "Zorvia Heights"))
    assert not context.sufficient and context.reason == "missing_source_metadata"


def test_works_with_real_retrieval(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    results = retriever.retrieve("Zorvia Heights harbour tower", ["darglobal"])
    context = ContextBuilder(loaded_index.metadata, rag_settings).build(results)
    assert context.blocks[0].source.title == "Zorvia Heights"
