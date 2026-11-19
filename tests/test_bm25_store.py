import pytest

from app.models import ChunkRecord
from app.retrieval.bm25_store import Bm25Store


def chunk(vector_id: int, text: str, company: str = "darglobal", title: str = "Title") -> ChunkRecord:
    return ChunkRecord(
        vector_id=vector_id, chunk_id=f"d{vector_id}::0000", document_id=f"d{vector_id}", position=0,
        company=company, category=None, title=title, source_url="https://example.test/x",
        language="en", collected_at="2026-11-18T12:00:00Z", text=text,
    )


@pytest.fixture
def store() -> Bm25Store:
    return Bm25Store([
        chunk(10, "Zorvia Heights is a fictional residential tower near the harbour"),
        chunk(25, "The platform lets users browse fictional apartments for rent", company="wasalt"),
        chunk(37, "Quarterly results discuss fictional revenue and shareholder returns"),
        chunk(42, "مشروع النخيل الأزرق مجمع سكني افتراضي قرب الشاطئ"),
        chunk(58, "خدمات المنصة تشمل البحث عن شقق افتراضية للإيجار", company="wasalt"),
        chunk(61, "Cooking recipes and gardening tips for the weekend"),
    ])


def test_english_retrieval_returns_vector_ids(store):
    results = store.search("Zorvia Heights tower", 3)
    assert results[0][0] == 10 and results[0][1] > 0


def test_arabic_retrieval_with_definite_article(store):
    results = store.search("أين يقع مشروع النخيل", 3)
    assert results[0][0] == 42


def test_arabic_matches_across_prefix_variants(store):
    assert store.search("الخدمات", 2)[0][0] == 58


def test_company_filter(store):
    results = store.search("fictional apartments rent", 5, companies=["wasalt"])
    assert [r[0] for r in results][0] == 25
    assert all(r[0] in (25, 58) for r in results)
    assert store.search("Zorvia", 5, companies=["wasalt"]) == []
    assert store.search("Zorvia", 5, companies=[]) == []


def test_top_k_respected_and_sorted(store):
    results = store.search("fictional", 2)
    assert len(results) <= 2
    assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)


def test_no_matching_terms_or_empty_query(store):
    assert store.search("xylophone", 3) == []
    assert store.search("   ", 3) == []
    assert store.search("!!!", 3) == []


def test_empty_corpus_is_safe():
    empty = Bm25Store([])
    assert empty.size == 0 and empty.search("anything", 3) == []


def test_chunks_without_tokens_are_skipped():
    store = Bm25Store([
        chunk(1, "...", title="!!"), chunk(2, "real words here", title="Heading"),
        chunk(3, "other content", title="Another"), chunk(4, "more content", title="Third"),
        chunk(5, "extra content", title="Fourth"),
    ])
    assert store.size == 4 and store.search("words", 3)[0][0] == 2


def test_title_is_searchable(store):
    titled = Bm25Store([
        chunk(1, "Body without the name", title="Unique Heading"),
        chunk(2, "Another body"), chunk(3, "Third body"), chunk(4, "Fourth body"),
    ])
    assert titled.search("Unique Heading", 2)[0][0] == 1


def test_returned_ids_are_always_valid(store):
    valid = {10, 25, 37, 42, 58, 61}
    for query in ("fictional", "مشروع", "tower revenue", "الشاطئ"):
        assert {i for i, _ in store.search(query, 10)} <= valid


def test_duplicate_vector_ids_rejected():
    with pytest.raises(ValueError):
        Bm25Store([chunk(1, "one text"), chunk(1, "another text")])


def test_invalid_top_k(store):
    with pytest.raises(ValueError):
        store.search("fictional", 0)
