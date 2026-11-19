import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from app.models import SourceDocument

INDEX_SCHEMA_VERSION = "1"
FAISS_FILENAME = "faiss.index"
CHUNKS_FILENAME = "chunks.jsonl"
SQLITE_FILENAME = "metadata.sqlite3"
MANIFEST_FILENAME = "index_manifest.json"
ARTIFACT_FILENAMES = (FAISS_FILENAME, CHUNKS_FILENAME, SQLITE_FILENAME)

TRACKED_PACKAGES = ("fastembed", "onnxruntime", "faiss-cpu", "rank-bm25", "numpy", "pydantic")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def source_data_hash(documents: Sequence[SourceDocument]) -> str:
    lines = sorted(f"{d.company}\t{d.document_id}\t{d.file_hash}" for d in documents)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def compute_build_id(parts: dict[str, Any]) -> str:
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


REQUIRED_KEYS = (
    "index_schema_version", "build_id", "built_at", "embedding_model", "embedding_dim",
    "faiss_index_type", "chunk_size_chars", "chunk_overlap_chars", "document_count",
    "chunk_count", "faiss_vector_count", "company_distribution",
    "content_type_distribution", "source_data_hash", "chunk_data_hash", "packages", "files",
)


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ManifestError(f"Cannot read {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path.name} is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"{path.name} must contain a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ManifestError(f"{path.name} is missing keys: {', '.join(missing)}")
    return manifest
