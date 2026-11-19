import numpy as np
import pytest

from app.retrieval.bm25_store import Bm25Store
from app.retrieval.faiss_store import FaissStore
from app.retrieval.hybrid import (
    HybridResult, HybridRetriever, fuse, has_sufficient_evidence, reciprocal_rank_fusion,
)
from app.retrieval.index_loader import LoadedIndex


def test_rrf_formula_and_ordering():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == 1


def test_rrf_constant_is_configurable():
    assert reciprocal_rank_fusion([[7]], k=10)[7] == pytest.approx(1 / 11)


def test_dense_only_and_bm25_only_results():
    dense_only = fuse([(5, 0.9), (6, 0.8)], [], k=60, top_k=5)
    assert [r.vector_id for r in dense_only] == [5, 6]
    assert dense_only[0].bm25_rank is None and dense_only[0].dense_score == 0.9
    lexical_only = fuse([], [(8, 4.2)], k=60, top_k=5)
    assert lexical_only[0].dense_rank is None and lexical_only[0].bm25_score == 4.2


def test_combined_results_prefer_agreement_and_keep_debug_info():
    results = fuse([(1, 0.9), (2, 0.8)], [(2, 7.0), (3, 5.0)], k=60, top_k=5)
    assert [r.vector_id for r in results] == [2, 1, 3]
    top = results[0]
    assert (top.dense_rank, top.bm25_rank, top.dense_score, top.bm25_score) == (2, 1, 0.8, 7.0)


def test_duplicate_vector_ids_are_merged_once():
    results = fuse([(1, 0.9), (1, 0.5)], [(1, 3.0)], k=60, top_k=5)
    assert [r.vector_id for r in results] == [1]
    assert results[0].dense_rank == 1


def test_missing_results_from_both_systems():
    assert fuse([], [], k=60, top_k=5) == []


def test_equal_scores_are_ordered_deterministically_by_vector_id():
    first = fuse([(9, 0.5)], [(4, 2.0)], k=60, top_k=5)
    assert [r.vector_id for r in first] == [4, 9]
    assert first == fuse([(9, 0.5)], [(4, 2.0)], k=60, top_k=5)


def test_top_k_limits_results():
    assert len(fuse([(i, 1.0) for i in range(10)], [], k=60, top_k=3)) == 3


def test_evidence_gate():
    weak = [HybridResult(1, 0.01, dense_rank=1, dense_score=0.1, bm25_rank=1, bm25_score=0.5)]
    assert not has_sufficient_evidence(weak, min_dense_score=0.3, min_bm25_score=3.0)
    assert has_sufficient_evidence(weak, min_dense_score=0.05, min_bm25_score=3.0)
    assert has_sufficient_evidence(weak, min_dense_score=0.9, min_bm25_score=0.4)
    assert not has_sufficient_evidence([], 0.3, 3.0)


def test_retriever_company_filter_is_applied_to_both_systems(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    for company in ("darglobal", "wasalt"):
        results = retriever.retrieve("fictional apartments harbour tower", [company])
        assert results
        for result in results:
            assert loaded_index.faiss.company_of(result.vector_id) == company
        chunks = loaded_index.metadata.get_chunks([r.vector_id for r in results])
        assert {c.company for c in chunks} == {company}


def test_retriever_searches_both_companies_when_asked(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    results = retriever.retrieve("fictional", ["darglobal", "wasalt"], top_k=20)
    chunks = loaded_index.metadata.get_chunks([r.vector_id for r in results])
    assert {c.company for c in chunks} == {"darglobal", "wasalt"}


def test_retriever_modes(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    dense = retriever.retrieve("Zorvia Heights tower", None, mode="dense")
    lexical = retriever.retrieve("Zorvia Heights tower", None, mode="bm25")
    assert dense and all(r.bm25_rank is None for r in dense)
    assert lexical and all(r.dense_rank is None for r in lexical)


def test_arabic_query_finds_arabic_chunk(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    top = retriever.retrieve("أين يقع مشروع النخيل الأزرق", ["darglobal"])[0]
    assert "النخيل" in loaded_index.metadata.get_chunks([top.vector_id])[0].text


def test_empty_query_and_empty_company_list(loaded_index, rag_settings, provider):
    retriever = HybridRetriever(loaded_index, provider, rag_settings)
    assert retriever.retrieve("   ", None) == []
    assert retriever.retrieve("Zorvia", []) == []


def test_empty_index_is_safe(rag_settings, provider):
    empty = LoadedIndex(manifest={}, faiss=FaissStore(384), bm25=Bm25Store([]), metadata=None)
    assert HybridRetriever(empty, provider, rag_settings).retrieve("anything", None) == []
