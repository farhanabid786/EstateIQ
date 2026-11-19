import pytest

from app.ingestion.chunking import build_chunks, chunk_document, split_text
from app.models import SourceDocument


def _doc(make_record, **overrides) -> SourceDocument:
    return SourceDocument.model_validate(make_record(**overrides))


def test_short_text_is_single_chunk():
    assert split_text("Short text.", 300, 50) == ["Short text."]


def test_empty_and_whitespace_text_yield_nothing():
    assert split_text("", 300, 50) == []
    assert split_text("  \n\n \t ", 300, 50) == []


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 0, 0)
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 100, -1)


def test_long_text_respects_size_and_loses_no_words():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 5
    assert all(0 < len(c) <= 200 for c in chunks)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_overlap_repeats_tail_of_previous_chunk():
    text = " ".join(f"token{i:03d}" for i in range(120))
    chunks = split_text(text, 200, 60)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split()[0] in previous.split()
    assert split_text(text, 200, 0)[1].split()[0] not in split_text(text, 200, 0)[0].split()


def test_prefers_paragraph_boundaries():
    first = "A" * 20 + " " + "b" * 100 + "."
    second = "C" * 20 + " " + "d" * 100 + "."
    chunks = split_text(first + "\n\n" + second, 150, 0)
    assert chunks == [first, second]


def test_prefers_sentence_boundaries():
    sentence_one = "This is the first sentence of the sample text here."
    sentence_two = "This is the second sentence of the sample text here."
    chunks = split_text(sentence_one + " " + sentence_two, 80, 0)
    assert chunks == [sentence_one, sentence_two]


def test_falls_back_to_character_cut_for_unbroken_text():
    chunks = split_text("x" * 1000, 300, 50)
    assert all(0 < len(c) <= 300 for c in chunks)
    assert len(chunks) >= 4


def test_arabic_text_is_preserved_and_sentence_split():
    sentence = "يقع المشروع الافتراضي في مدينة خيالية قرب الشاطئ الجميل؟"
    text = " ".join([sentence] * 6)
    chunks = split_text(text, 130, 0)
    assert len(chunks) > 1
    assert all(c.endswith("؟") for c in chunks)
    assert " ".join(chunks) == text


def test_arabic_combining_marks_not_split():
    text = "بَ" * 400
    chunks = split_text(text, 101, 0)
    assert all(not c.startswith("َ") for c in chunks)
    assert "".join(chunks) == text


def test_deterministic_output():
    text = " ".join(f"sentence number {i}." for i in range(100))
    assert split_text(text, 150, 30) == split_text(text, 150, 30)


def test_terminates_with_pathological_overlap():
    chunks = split_text("a " * 500, 120, 119)
    assert len(chunks) < 1000 and all(chunks)


def test_chunk_records_carry_metadata(make_record):
    doc = _doc(make_record, content="Sentence one. " * 60, category="project", page_number=3,
               language="en", company="wasalt")
    chunks = chunk_document(doc, 200, 30, first_vector_id=10)
    assert len(chunks) > 1
    assert [c.vector_id for c in chunks] == list(range(10, 10 + len(chunks)))
    assert chunks[0].chunk_id == f"{doc.document_id}::0000"
    for position, chunk in enumerate(chunks):
        assert chunk.position == position
        assert (chunk.document_id, chunk.company, chunk.category) == (doc.document_id, "wasalt", "project")
        assert (chunk.source_url, chunk.title, chunk.page_number) == (doc.source_url, doc.title, 3)
        assert chunk.collected_at == doc.collected_at and chunk.language == "en"
    assert chunks[0].retrieval_text.startswith(doc.title)


def test_vector_and_chunk_ids_unique_across_documents(make_record):
    docs = [_doc(make_record, content="Alpha beta gamma. " * 50) for _ in range(5)]
    chunks = build_chunks(docs, 200, 30)
    assert [c.vector_id for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_build_chunks_independent_of_input_order(make_record):
    docs = [_doc(make_record, content="Gamma delta. " * 40) for _ in range(4)]
    forward = build_chunks(docs, 200, 30)
    backward = build_chunks(list(reversed(docs)), 200, 30)
    assert forward == backward


def test_duplicate_document_ids_rejected(make_record):
    doc = _doc(make_record)
    with pytest.raises(ValueError):
        build_chunks([doc, doc], 200, 30)
