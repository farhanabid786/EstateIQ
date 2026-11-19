import logging
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from app.api.schemas import SourceRef, SourceSummary, StatsResponse
from app.config import SCHEMA_VERSION
from app.models import ChunkRecord, SourceDocument

logger = logging.getLogger(__name__)

_SQL_VARIABLE_LIMIT = 500

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    content_type TEXT NOT NULL,
    language TEXT NOT NULL,
    category TEXT,
    description TEXT,
    location TEXT,
    page_number INTEGER,
    parent_document_id TEXT,
    source_domain TEXT,
    extraction_method TEXT,
    collected_at TEXT NOT NULL,
    published_at TEXT,
    file_hash TEXT NOT NULL,
    content_chars INTEGER NOT NULL
);
CREATE TABLE chunks (
    vector_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    position INTEGER NOT NULL,
    company TEXT NOT NULL,
    category TEXT,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    language TEXT NOT NULL,
    page_number INTEGER,
    published_at TEXT,
    collected_at TEXT NOT NULL,
    text TEXT NOT NULL,
    UNIQUE (document_id, position)
);
CREATE INDEX idx_chunks_company ON chunks(company);
CREATE INDEX idx_documents_company ON documents(company);
"""

_DOCUMENT_COLUMNS = (
    "document_id", "company", "title", "source_url", "content_type", "language",
    "category", "description", "location", "page_number", "parent_document_id",
    "source_domain", "extraction_method", "collected_at", "published_at",
    "file_hash", "content_chars",
)
_CHUNK_COLUMNS = (
    "vector_id", "chunk_id", "document_id", "position", "company", "category",
    "title", "source_url", "language", "page_number", "published_at",
    "collected_at", "text",
)


class MetadataError(RuntimeError):
    pass


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    placeholders = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


def build_database(
    path: Path,
    documents: Sequence[SourceDocument],
    chunks: Sequence[ChunkRecord],
    built_at: str,
    extra_meta: Optional[Mapping[str, str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.unlink(missing_ok=True)

    document_rows = []
    for doc in documents:
        values = doc.model_dump()
        values["content_chars"] = len(doc.content)
        document_rows.append(tuple(values[column] for column in _DOCUMENT_COLUMNS))
    chunk_rows = [
        tuple(getattr(chunk, column) for column in _CHUNK_COLUMNS) for chunk in chunks
    ]
    meta = {"schema_version": SCHEMA_VERSION, "built_at": built_at, **(extra_meta or {})}

    with closing(sqlite3.connect(temp_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        with conn:
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
            conn.executemany(_insert_sql("documents", _DOCUMENT_COLUMNS), document_rows)
            conn.executemany(_insert_sql("chunks", _CHUNK_COLUMNS), chunk_rows)
    temp_path.replace(path)
    logger.info("Wrote metadata database with %d documents, %d chunks", len(document_rows), len(chunk_rows))


class MetadataStore:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise MetadataError("Metadata database is missing")
        self._uri = path.resolve().as_uri() + "?mode=ro"
        version = self.get_meta("schema_version")
        if version != SCHEMA_VERSION:
            raise MetadataError(
                f"Metadata schema version {version!r} does not match expected {SCHEMA_VERSION!r}"
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self._uri, uri=True)
        except sqlite3.Error as exc:
            raise MetadataError("Cannot open metadata database") from exc
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except sqlite3.Error as exc:
            logger.exception("Metadata query failed")
            raise MetadataError("Metadata query failed") from exc
        finally:
            conn.close()

    def get_meta(self, key: str) -> Optional[str]:
        with self._connection() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def counts(self) -> tuple[int, int]:
        with self._connection() as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return documents, chunks

    def get_chunks(self, vector_ids: Sequence[int]) -> list[ChunkRecord]:
        found: dict[int, ChunkRecord] = {}
        with self._connection() as conn:
            for start in range(0, len(vector_ids), _SQL_VARIABLE_LIMIT):
                batch = list(vector_ids[start:start + _SQL_VARIABLE_LIMIT])
                marks = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT {', '.join(_CHUNK_COLUMNS)} FROM chunks WHERE vector_id IN ({marks})",
                    batch,
                ).fetchall()
                found.update({row["vector_id"]: ChunkRecord(**dict(row)) for row in rows})
        missing = [vid for vid in vector_ids if vid not in found]
        if missing:
            raise MetadataError(f"Index references {len(missing)} chunk(s) absent from metadata")
        return [found[vid] for vid in vector_ids]

    def adjacent_chunks(self, document_id: str, position: int, window: int) -> list[ChunkRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_CHUNK_COLUMNS)} FROM chunks "
                "WHERE document_id = ? AND position BETWEEN ? AND ? AND position != ? "
                "ORDER BY position",
                (document_id, position - window, position + window, position),
            ).fetchall()
        return [ChunkRecord(**dict(row)) for row in rows]

    def vector_ids_for_companies(self, companies: Sequence[str]) -> list[int]:
        marks = ", ".join("?" for _ in companies)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT vector_id FROM chunks WHERE company IN ({marks}) ORDER BY vector_id",
                list(companies),
            ).fetchall()
        return [row["vector_id"] for row in rows]

    def get_sources(self, document_ids: Sequence[str]) -> list[SourceRef]:
        unique_ids = list(dict.fromkeys(document_ids))
        if not unique_ids:
            return []
        marks = ", ".join("?" for _ in unique_ids)
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT document_id, title, source_url, company, published_at "
                f"FROM documents WHERE document_id IN ({marks})",
                unique_ids,
            ).fetchall()
        by_id = {
            row["document_id"]: SourceRef(
                document_id=row["document_id"],
                title=row["title"],
                url=row["source_url"],
                company=row["company"],
                published_at=row["published_at"],
            )
            for row in rows
        }
        return [by_id[doc_id] for doc_id in unique_ids if doc_id in by_id]

    def list_sources(
        self, company: Optional[str], offset: int, limit: int
    ) -> tuple[list[SourceSummary], int]:
        where, params = ("WHERE company = ?", [company]) if company else ("", [])
        with self._connection() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]
            rows = conn.execute(
                "SELECT document_id, title, source_url, company, content_type, category, "
                f"language, published_at, collected_at FROM documents {where} "
                "ORDER BY company, title, document_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        items = [
            SourceSummary(
                document_id=row["document_id"],
                title=row["title"],
                url=row["source_url"],
                company=row["company"],
                content_type=row["content_type"],
                category=row["category"],
                language=row["language"],
                published_at=row["published_at"],
                collected_at=row["collected_at"],
            )
            for row in rows
        ]
        return items, total

    def stats(self) -> StatsResponse:
        with self._connection() as conn:
            distribution = {
                row["company"]: row["n"]
                for row in conn.execute(
                    "SELECT company, COUNT(*) AS n FROM documents GROUP BY company"
                )
            }
            content_types = {
                row["content_type"]: row["n"]
                for row in conn.execute(
                    "SELECT content_type, COUNT(*) AS n FROM documents GROUP BY content_type"
                )
            }
        documents, chunks = self.counts()
        return StatsResponse(
            document_count=documents,
            chunk_count=chunks,
            company_distribution=distribution,
            content_type_distribution=content_types,
            index_built_at=self.get_meta("built_at"),
            index_build_id=self.get_meta("build_id"),
            embedding_model=self.get_meta("embedding_model"),
        )

    def project_titles(self) -> list[tuple[str, str]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT title, company FROM documents WHERE category = 'project'"
            ).fetchall()
        return [(row["title"], row["company"]) for row in rows]
