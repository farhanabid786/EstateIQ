import json
import sqlite3

import pytest

from app.database.metadata import MetadataStore
from app.ingestion.build_index import IndexBuildError, build_index, main
from app.ingestion.loader import IngestionError
from app.retrieval.index_loader import IndexLoadError, load_index
from app.retrieval.manifest import (
    ARTIFACT_FILENAMES, CHUNKS_FILENAME, FAISS_FILENAME, MANIFEST_FILENAME, SQLITE_FILENAME,
)

BUILT_AT = "2026-11-18T12:00:00Z"


@pytest.fixture
def corpus(make_record):
    long_en = " ".join(
        f"Zorvia Heights sentence {i} describes a fictional residential tower near the harbour." for i in range(12)
    )
    return [
        make_record(title="Zorvia Heights", content=long_en, category="project"),
        make_record(title="Platform Overview", company="wasalt", category="service",
                    content="The fictional platform helps users browse apartments for rent online."),
        make_record(title="مشروع النخيل الأزرق", language="ar", category="project",
                    content="مشروع النخيل الأزرق مجمع سكني افتراضي يقع قرب الشاطئ في مدينة خيالية."),
    ]


@pytest.fixture
def built(tmp_path, corpus, write_jsonl, settings, provider):
    source = write_jsonl(tmp_path / "docs.jsonl", corpus)
    out = tmp_path / "index"
    manifest = build_index(source, out, settings, provider, built_at=BUILT_AT)
    return source, out, manifest


def test_build_produces_all_artifacts_and_no_staging(built):
    _, out, _ = built
    for name in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME):
        assert (out / name).is_file()
    assert sorted(p.name for p in out.iterdir()) == sorted([*ARTIFACT_FILENAMES, MANIFEST_FILENAME])


def test_manifest_contents(built, settings, corpus):
    _, out, manifest = built
    on_disk = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["document_count"] == 3
    assert manifest["chunk_count"] == manifest["faiss_vector_count"] > 3
    assert manifest["embedding_model"] == settings.embedding_model
    assert manifest["embedding_dim"] == settings.embedding_dim == 384
    assert manifest["faiss_index_type"] == "IndexIDMap2(IndexFlatIP)"
    assert manifest["chunk_size_chars"] == 300 and manifest["chunk_overlap_chars"] == 50
    assert manifest["company_distribution"] == {"darglobal": 2, "wasalt": 1}
    assert manifest["content_type_distribution"] == {"webpage": 3}
    assert set(manifest["files"]) == set(ARTIFACT_FILENAMES)
    assert {"fastembed", "faiss-cpu", "rank-bm25", "numpy"} <= set(manifest["packages"])


def test_sqlite_faiss_and_chunks_are_consistent(built):
    _, out, manifest = built
    store = MetadataStore(out / SQLITE_FILENAME)
    assert store.counts() == (3, manifest["chunk_count"])
    assert store.get_meta("build_id") == manifest["build_id"]
    lines = (out / CHUNKS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest["chunk_count"]
    assert any("مشروع النخيل" in line for line in lines)
    document_ids = list(dict.fromkeys(json.loads(line)["document_id"] for line in lines))
    assert len(store.get_sources(document_ids)) == 3


def test_build_is_reproducible_for_same_input(tmp_path, corpus, write_jsonl, settings, provider):
    source = write_jsonl(tmp_path / "docs.jsonl", corpus)
    first = build_index(source, tmp_path / "a", settings, provider, built_at=BUILT_AT)
    second = build_index(source, tmp_path / "b", settings, provider, built_at=BUILT_AT)
    assert first == second


def test_duplicate_input_records_are_deduplicated(tmp_path, corpus, write_jsonl, settings, provider):
    source = write_jsonl(tmp_path / "docs.jsonl", corpus + [corpus[0]])
    manifest = build_index(source, tmp_path / "out", settings, provider, built_at=BUILT_AT)
    assert manifest["document_count"] == 3


def test_invalid_data_fails_without_writing_index(tmp_path, corpus, write_jsonl, settings, provider):
    source = write_jsonl(tmp_path / "docs.jsonl", corpus, raw_lines=["{broken"])
    out = tmp_path / "out"
    with pytest.raises(IngestionError):
        build_index(source, out, settings, provider)
    assert not (out / MANIFEST_FILENAME).exists()
    assert not list(out.glob("*.index"))


def test_empty_input_fails_clearly(tmp_path, settings, provider):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IndexBuildError, match="No valid documents"):
        build_index(empty, tmp_path / "out", settings, provider)


def test_failed_rebuild_keeps_no_valid_looking_partial_state(
    tmp_path, corpus, write_jsonl, settings, provider, built
):
    _, out, _ = built
    bad = write_jsonl(tmp_path / "bad.jsonl", corpus, raw_lines=["nope"])
    with pytest.raises(IngestionError):
        build_index(bad, out, settings, provider)
    assert load_index(settings, out).build_id


def test_loader_loads_and_searches(built, settings, provider):
    _, out, manifest = built
    index = load_index(settings, out)
    assert index.build_id == manifest["build_id"]
    assert index.faiss.size == index.bm25.size == manifest["chunk_count"]

    query = provider.embed_query("Zorvia Heights harbour tower")
    (top_id, _), *_ = index.faiss.search(query, 3)
    top_chunk = index.metadata.get_chunks([top_id])[0]
    assert top_chunk.title == "Zorvia Heights" and top_chunk.company == "darglobal"

    arabic_hit = index.bm25.search("أين يقع مشروع النخيل الأزرق", 1)[0][0]
    assert "النخيل" in index.metadata.get_chunks([arabic_hit])[0].text

    only_wasalt = index.faiss.search(query, 5, companies=["wasalt"])
    assert {index.metadata.get_chunks([i])[0].company for i, _ in only_wasalt} == {"wasalt"}
    source = index.metadata.get_sources([top_chunk.document_id])[0]
    assert source.url == top_chunk.source_url


def test_loader_requires_manifest(tmp_path, settings):
    with pytest.raises(IndexLoadError, match="manifest is missing"):
        load_index(settings, tmp_path)


def test_loader_detects_tampered_chunks(built, settings):
    _, out, _ = built
    path = out / CHUNKS_FILENAME
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(IndexLoadError, match="inconsistent"):
        load_index(settings, out)


def test_loader_detects_missing_artifact(built, settings):
    _, out, _ = built
    (out / FAISS_FILENAME).unlink()
    with pytest.raises(IndexLoadError, match="missing"):
        load_index(settings, out)


def test_loader_rejects_model_mismatch(built, settings):
    from dataclasses import replace

    _, out, _ = built
    with pytest.raises(IndexLoadError, match="embedding model"):
        load_index(replace(settings, embedding_model="other/model"), out)
    with pytest.raises(IndexLoadError, match="dimension"):
        load_index(replace(settings, embedding_dim=128), out)


def test_loader_rejects_unsupported_schema_and_sqlite_mismatch(built, settings):
    _, out, manifest = built
    manifest_path = out / MANIFEST_FILENAME
    changed = dict(manifest, index_schema_version="99")
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(IndexLoadError, match="Unsupported"):
        load_index(settings, out)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with sqlite3.connect(out / SQLITE_FILENAME) as conn:
        conn.execute("UPDATE meta SET value = 'other' WHERE key = 'build_id'")
    with pytest.raises(IndexLoadError, match="inconsistent"):
        load_index(settings, out)
    with pytest.raises(IndexLoadError, match="different build"):
        load_index(settings, out, verify_checksums=False)


def test_loader_does_not_create_embedding_model(built, settings, monkeypatch):
    import app.retrieval.embeddings as embeddings

    def boom(*args, **kwargs):
        raise AssertionError("embedding model must not load")

    monkeypatch.setattr(embeddings, "_fastembed_factory", boom)
    _, out, _ = built
    assert load_index(settings, out).faiss.size > 0


def test_cli_returns_error_code_for_bad_input(tmp_path):
    code = main(["--input", str(tmp_path / "missing.jsonl"), "--output-dir", str(tmp_path / "out"),
                 "--skip-registry-check"])
    assert code == 1
    assert not (tmp_path / "out" / MANIFEST_FILENAME).exists()


def test_live_index_matches_latest_40_document_dataset(settings):
    loaded = load_index(settings)
    assert loaded.manifest["document_count"] == 40
    assert loaded.manifest["content_type_distribution"] == {"press_release": 1, "webpage": 39}
    assert loaded.manifest["company_distribution"] == {"darglobal": 40}
    assert loaded.faiss.size == loaded.bm25.size == loaded.manifest["chunk_count"] == 182
    assert loaded.metadata.counts() == (40, 182)
    assert loaded.manifest["embedding_dim"] == 384
