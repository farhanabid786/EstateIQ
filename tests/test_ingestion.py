import pytest

from app.ingestion.loader import IngestionError, load_documents


def test_loads_valid_records(tmp_path, make_record, write_jsonl):
    path = write_jsonl(tmp_path / "docs.jsonl", [make_record(), make_record(company="wasalt")])
    documents, report = load_documents(path)
    assert len(documents) == 2
    assert report.valid_records == 2 and report.invalid_records == []
    assert [d.company for d in documents] == ["darglobal", "wasalt"]


def test_loads_directory_with_multiple_files(tmp_path, make_record, write_jsonl):
    write_jsonl(tmp_path / "a.jsonl", [make_record()])
    write_jsonl(tmp_path / "b.jsonl", [make_record(company="wasalt")])
    documents, report = load_documents(tmp_path)
    assert len(documents) == 2 and len(report.files) == 2


def test_preserves_arabic_and_metadata(tmp_path, make_record, write_jsonl):
    record = make_record(content="مشروع سكني افتراضي قرب البحر", language="ar", location="Test City")
    documents, _ = load_documents(write_jsonl(tmp_path / "d.jsonl", [record]))
    assert documents[0].content == "مشروع سكني افتراضي قرب البحر"
    assert documents[0].location == "Test City"
    assert documents[0].source_url == record["source_url"]


def test_malformed_json_fails_in_strict_mode(tmp_path, make_record, write_jsonl):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record()], raw_lines=["{not json"])
    with pytest.raises(IngestionError, match="malformed JSON"):
        load_documents(path)


def test_invalid_schema_reports_line_and_field(tmp_path, make_record, write_jsonl):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(company="other")])
    with pytest.raises(IngestionError, match=r"d\.jsonl:1.*company"):
        load_documents(path)


def test_lenient_mode_skips_invalid_but_reports(tmp_path, make_record, write_jsonl):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(), make_record(file_hash="bad")], raw_lines=["[]"])
    documents, report = load_documents(path, strict=False)
    assert len(documents) == 1
    assert len(report.invalid_records) == 2


def test_duplicate_ids_and_content_are_dropped(tmp_path, make_record, write_jsonl):
    first = make_record()
    same_id = dict(first)
    same_content = make_record(content=first["content"])
    path = write_jsonl(tmp_path / "d.jsonl", [first, same_id, same_content])
    documents, report = load_documents(path)
    assert len(documents) == 1
    assert report.duplicate_ids == [first["document_id"]]
    assert report.duplicate_content == [same_content["document_id"]]


def test_same_content_in_different_companies_is_kept(tmp_path, make_record, write_jsonl):
    first = make_record(content="Shared wording text")
    second = make_record(content="Shared wording text", company="wasalt")
    documents, _ = load_documents(write_jsonl(tmp_path / "d.jsonl", [first, second]))
    assert len(documents) == 2


def test_conflicting_duplicate_id_fails_strict_and_resolves_lenient(tmp_path, make_record, write_jsonl):
    first = make_record(content="Version one")
    second = make_record(content="Version two", document_id=first["document_id"],
                         collected_at="2026-11-19T12:00:00Z")
    path = write_jsonl(tmp_path / "d.jsonl", [first, second])
    with pytest.raises(IngestionError, match="conflicting"):
        load_documents(path)
    documents, report = load_documents(path, strict=False)
    assert documents[0].content == "Version two"
    assert report.conflicting_ids == [first["document_id"]]


def test_empty_file_and_empty_directory(tmp_path):
    (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
    assert load_documents(tmp_path / "empty.jsonl")[0] == []
    other = tmp_path / "nothing"
    other.mkdir()
    assert load_documents(other)[0] == []


def test_missing_path_raises(tmp_path):
    with pytest.raises(IngestionError, match="does not exist"):
        load_documents(tmp_path / "missing")


def test_non_utf8_file_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(IngestionError, match="UTF-8"):
        load_documents(path)


def test_output_order_is_deterministic(tmp_path, make_record, write_jsonl):
    records = [make_record(), make_record(company="wasalt"), make_record()]
    forward, _ = load_documents(write_jsonl(tmp_path / "f.jsonl", records))
    backward, _ = load_documents(write_jsonl(tmp_path / "g.jsonl", list(reversed(records))))
    assert [d.document_id for d in forward] == [d.document_id for d in backward]
