from app.api.schemas import SourceRef
from app.retrieval.context_builder import ContextBlock
from app.services.citations import strip_markers, validate_citations


def block(number: int, company: str = "darglobal") -> ContextBlock:
    source = SourceRef(document_id=f"doc{number}", title=f"Title {number}",
                       url=f"https://example.test/{number}", company=company)
    return ContextBlock(f"S{number}", source, None, (), "text", 0.1)


BLOCKS = [block(1), block(2, "wasalt"), block(3)]


def test_valid_markers_map_to_retrieved_sources():
    cited = validate_citations("Fact one [S1]. Fact two [S2].", BLOCKS)
    assert [s.document_id for s in cited.sources] == ["doc1", "doc2"]
    assert cited.text == "Fact one [S1]. Fact two [S2]." and not cited.used_fallback


def test_markers_are_renumbered_in_order_of_first_use():
    cited = validate_citations("First [S3]. Second [S1]. Again [S3].", BLOCKS)
    assert cited.text == "First [S1]. Second [S2]. Again [S1]."
    assert [s.document_id for s in cited.sources] == ["doc3", "doc1"]


def test_invalid_markers_are_rejected_and_never_become_sources():
    cited = validate_citations("Claim [S9]. Real [S2].", BLOCKS)
    assert cited.invalid_markers == 1
    assert "S9" not in cited.text and "[S1]" in cited.text
    assert [s.document_id for s in cited.sources] == ["doc2"]


def test_grouped_markers_are_supported():
    cited = validate_citations("Both agree [S1, S2].", BLOCKS)
    assert cited.text == "Both agree [S1][S2]."
    assert len(cited.sources) == 2


def test_lowercase_and_spaced_markers():
    cited = validate_citations("Fact [ s2 ].", BLOCKS)
    assert [s.document_id for s in cited.sources] == ["doc2"]


def test_no_valid_citation_falls_back_to_retrieved_sources_only():
    cited = validate_citations("No markers at all.", BLOCKS)
    assert cited.used_fallback
    assert [s.document_id for s in cited.sources] == ["doc1", "doc2", "doc3"]
    only_invalid = validate_citations("Claim [S7].", BLOCKS)
    assert only_invalid.used_fallback and "S7" not in only_invalid.text


def test_urls_come_from_metadata_not_from_the_answer():
    cited = validate_citations("See https://evil.example/fake [S1].", BLOCKS)
    assert [s.url for s in cited.sources] == ["https://example.test/1"]


def test_strip_markers():
    assert strip_markers("Summary text [S1] here [S2][S3].") == "Summary text here."
