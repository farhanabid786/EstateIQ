import argparse
import logging
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from app.config import Settings, configure_logging, get_settings
from app.database.metadata import MetadataError, MetadataStore, build_database
from app.ingestion.chunking import build_chunks
from app.ingestion.loader import IngestionError, load_documents
from app.models import ChunkRecord
from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider
from app.retrieval.faiss_store import INDEX_TYPE_NAME, FaissStore, FaissStoreError
from app.retrieval.manifest import (
    ARTIFACT_FILENAMES, CHUNKS_FILENAME, FAISS_FILENAME, INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME, SQLITE_FILENAME, compute_build_id, file_info, package_versions,
    source_data_hash, write_manifest,
)

logger = logging.getLogger(__name__)

STAGING_DIRNAME = ".staging"


class IndexBuildError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_chunks(path: Path, chunks: Sequence[ChunkRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _verify_staging(staging: Path, chunks: Sequence[ChunkRecord], document_count: int, build_id: str) -> None:
    if _count_lines(staging / CHUNKS_FILENAME) != len(chunks):
        raise IndexBuildError("chunks.jsonl line count does not match chunk count")
    companies = {c.vector_id: c.company for c in chunks}
    faiss_store = FaissStore.load(staging / FAISS_FILENAME, companies)
    if faiss_store.size != len(chunks):
        raise IndexBuildError("FAISS vector count does not match chunk count")
    store = MetadataStore(staging / SQLITE_FILENAME)
    if store.counts() != (document_count, len(chunks)):
        raise IndexBuildError("SQLite counts do not match documents and chunks")
    if store.get_meta("build_id") != build_id:
        raise IndexBuildError("SQLite build_id does not match the manifest")


def _promote(staging: Path, output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.unlink(missing_ok=True)
    for name in ARTIFACT_FILENAMES:
        os.replace(staging / name, output_dir / name)
    write_manifest(staging / MANIFEST_FILENAME, manifest)
    os.replace(staging / MANIFEST_FILENAME, manifest_path)


def build_index(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    settings: Settings,
    provider: EmbeddingProvider,
    built_at: Optional[str] = None,
    strict: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    built_at = built_at or _utc_now()

    documents, report = load_documents(input_path, strict=strict)
    if not documents:
        raise IndexBuildError("No valid documents found; nothing to index")

    chunks = build_chunks(documents, settings.chunk_size_chars, settings.chunk_overlap_chars)
    if not chunks:
        raise IndexBuildError("Chunking produced no chunks")
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))

    vectors = provider.embed_documents([chunk.retrieval_text for chunk in chunks])
    if vectors.shape != (len(chunks), settings.embedding_dim):
        raise IndexBuildError(
            f"Embedding shape {vectors.shape} does not match ({len(chunks)}, {settings.embedding_dim})"
        )
    faiss_store = FaissStore(settings.embedding_dim)
    faiss_store.add([c.vector_id for c in chunks], vectors, [c.company for c in chunks])

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / STAGING_DIRNAME
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        faiss_store.save(staging / FAISS_FILENAME)
        _write_chunks(staging / CHUNKS_FILENAME, chunks)
        chunk_hash = file_info(staging / CHUNKS_FILENAME)["sha256"]
        source_hash = source_data_hash(documents)
        build_id = compute_build_id({
            "schema": INDEX_SCHEMA_VERSION,
            "source": source_hash,
            "chunks": chunk_hash,
            "model": settings.embedding_model,
            "dim": settings.embedding_dim,
            "chunk_size": settings.chunk_size_chars,
            "chunk_overlap": settings.chunk_overlap_chars,
        })
        build_database(
            staging / SQLITE_FILENAME, documents, chunks, built_at,
            {
                "build_id": build_id,
                "embedding_model": settings.embedding_model,
                "embedding_dim": str(settings.embedding_dim),
                "chunk_size_chars": str(settings.chunk_size_chars),
                "chunk_overlap_chars": str(settings.chunk_overlap_chars),
            },
        )
        manifest = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "build_id": build_id,
            "built_at": built_at,
            "embedding_provider": "jina",
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "faiss_index_type": INDEX_TYPE_NAME,
            "vectors_normalized": True,
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "faiss_vector_count": faiss_store.size,
            "company_distribution": dict(sorted(Counter(d.company for d in documents).items())),
            "chunk_company_distribution": dict(sorted(Counter(c.company for c in chunks).items())),
            "content_type_distribution": dict(sorted(Counter(d.content_type for d in documents).items())),
            "source_data_hash": source_hash,
            "chunk_data_hash": chunk_hash,
            "packages": package_versions(),
            "files": {name: file_info(staging / name) for name in ARTIFACT_FILENAMES},
        }
        _verify_staging(staging, chunks, len(documents), build_id)
        _promote(staging, output_dir, manifest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "Index build %s complete: %d documents, %d chunks (%d duplicate document(s) dropped)",
        build_id, len(documents), len(chunks),
        len(report.duplicate_ids) + len(report.duplicate_content),
    )
    return manifest


def _parse_args(argv: Optional[Sequence[str]], settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.build_index",
        description="Build the FAISS, chunk, SQLite and manifest artifacts from scraped JSONL.",
    )
    parser.add_argument("--input", type=Path, default=settings.documents_path,
                        help="JSONL file or directory of .jsonl files")
    parser.add_argument("--output-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--lenient", action="store_true",
                        help="skip invalid records instead of failing the build")
    parser.add_argument("--skip-registry-check", action="store_true",
                        help="do not verify the model against the FastEmbed registry")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args(argv, settings)
    provider = EmbeddingProvider.from_settings(settings)
    try:
        if not args.skip_registry_check:
            provider.verify_registry()
        manifest = build_index(args.input, args.output_dir, settings, provider, strict=not args.lenient)
    except (IngestionError, IndexBuildError, EmbeddingError, FaissStoreError, MetadataError) as exc:
        logger.error("Index build failed: %s", exc)
        return 1
    logger.info("Wrote index to %s (build %s)", args.output_dir, manifest["build_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
