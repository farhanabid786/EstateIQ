import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import ValidationError

from app.config import Settings, get_settings
from app.database.metadata import MetadataError, MetadataStore
from app.models import ChunkRecord
from app.retrieval.bm25_store import Bm25Store
from app.retrieval.faiss_store import FaissStore, FaissStoreError
from app.retrieval.manifest import (
    ARTIFACT_FILENAMES, CHUNKS_FILENAME, FAISS_FILENAME, INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME, SQLITE_FILENAME, ManifestError, read_manifest, sha256_file,
)
from app.memory_diagnostics import log_rss

logger = logging.getLogger(__name__)


class IndexLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedIndex:
    manifest: dict[str, Any]
    faiss: FaissStore
    bm25: Bm25Store
    metadata: MetadataStore

    @property
    def build_id(self) -> str:
        return self.manifest["build_id"]


def _read_chunks(path: Path) -> Iterator[ChunkRecord]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield ChunkRecord.model_validate_json(line)
                except ValidationError as exc:
                    raise IndexLoadError(f"{path.name}:{line_number} is not a valid chunk record") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise IndexLoadError(f"Cannot read {path.name}") from exc


def _check_files(index_dir: Path, manifest: dict[str, Any], verify_checksums: bool) -> None:
    for name in ARTIFACT_FILENAMES:
        path = index_dir / name
        if not path.is_file():
            raise IndexLoadError(f"Index artifact is missing: {name}")
        expected = manifest["files"].get(name)
        if expected is None:
            raise IndexLoadError(f"Manifest has no file entry for {name}")
        if path.stat().st_size != expected["bytes"]:
            raise IndexLoadError(f"{name} size differs from the manifest; the index is inconsistent")
        if verify_checksums and sha256_file(path) != expected["sha256"]:
            raise IndexLoadError(f"{name} checksum differs from the manifest; the index is inconsistent")


def load_index(
    settings: Optional[Settings] = None,
    index_dir: Optional[Path] = None,
    verify_checksums: bool = True,
) -> LoadedIndex:
    settings = settings or get_settings()
    index_dir = Path(index_dir) if index_dir else settings.index_dir

    manifest_path = index_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise IndexLoadError("Index manifest is missing; build the index before starting the API")
    try:
        manifest = read_manifest(manifest_path)
    except ManifestError as exc:
        raise IndexLoadError(str(exc)) from exc

    if manifest["index_schema_version"] != INDEX_SCHEMA_VERSION:
        raise IndexLoadError(
            f"Unsupported index schema version {manifest['index_schema_version']!r}; "
            f"expected {INDEX_SCHEMA_VERSION!r}"
        )
    if manifest["embedding_model"] != settings.embedding_model:
        raise IndexLoadError("Index was built with a different embedding model than configured")
    if manifest.get("embedding_provider") != "jina":
        raise IndexLoadError("Index was not built with the Jina embedding provider")
    if manifest["embedding_dim"] != settings.embedding_dim:
        raise IndexLoadError("Index embedding dimension differs from the configured dimension")
    if manifest["chunk_count"] != manifest["faiss_vector_count"]:
        raise IndexLoadError("Manifest chunk and vector counts disagree")
    _check_files(index_dir, manifest, verify_checksums)

    companies: dict[int, str] = {}

    def tracked_chunks() -> Iterator[ChunkRecord]:
        for chunk in _read_chunks(index_dir / CHUNKS_FILENAME):
            if chunk.vector_id in companies:
                raise IndexLoadError(f"Duplicate vector id {chunk.vector_id} in {CHUNKS_FILENAME}")
            companies[chunk.vector_id] = chunk.company
            yield chunk

    bm25 = Bm25Store(tracked_chunks())
    if len(companies) != manifest["chunk_count"]:
        raise IndexLoadError("chunks.jsonl count differs from the manifest")

    try:
        faiss_store = FaissStore.load(index_dir / FAISS_FILENAME, companies)
    except FaissStoreError as exc:
        raise IndexLoadError(str(exc)) from exc
    if faiss_store.dimension != settings.embedding_dim:
        raise IndexLoadError("FAISS index dimension differs from the configured dimension")
    if faiss_store.size != manifest["faiss_vector_count"]:
        raise IndexLoadError("FAISS vector count differs from the manifest")
    if set(faiss_store.vector_ids) != set(companies):
        raise IndexLoadError("FAISS vector ids do not match chunk ids")

    try:
        metadata = MetadataStore(index_dir / SQLITE_FILENAME)
        counts = metadata.counts()
        build_id = metadata.get_meta("build_id")
    except MetadataError as exc:
        raise IndexLoadError(str(exc)) from exc
    if counts != (manifest["document_count"], manifest["chunk_count"]):
        raise IndexLoadError("SQLite counts differ from the manifest")
    if build_id != manifest["build_id"]:
        raise IndexLoadError("SQLite belongs to a different build than the manifest")

    logger.info(
        "Loaded index %s: %d documents, %d chunks", manifest["build_id"], counts[0], counts[1]
    )
    log_rss("after_index_load")
    return LoadedIndex(manifest=manifest, faiss=faiss_store, bm25=bm25, metadata=metadata)
