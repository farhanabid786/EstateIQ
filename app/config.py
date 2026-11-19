import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1"


class ConfigError(ValueError):
    pass


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    frontend_dir: Path
    embedding_model: str
    embedding_dim: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    dense_top_k: int
    bm25_top_k: int
    rrf_k: int
    max_context_chunks: int
    max_context_chars: int
    adjacent_chunk_window: int
    min_dense_score: float
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    llm_timeout_seconds: int
    llm_max_retries: int
    llm_max_output_tokens: int
    max_concurrent_requests: int
    log_level: str
    hybrid_top_k: int = 10
    expansion_top_n: int = 2
    min_bm25_score: float = 3.0
    request_timeout_seconds: int = 90
    llm_temperature: float = 0.1
    preload_embedding_model: bool = False
    cors_origins: tuple[str, ...] = ()

    @property
    def documents_path(self) -> Path:
        return self.data_dir / "processed" / "documents.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "index_manifest.json"

    @property
    def sqlite_path(self) -> Path:
        return self.index_dir / "metadata.sqlite3"

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key)


def _build_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR") or PROJECT_ROOT / "data")
    frontend_dir = Path(os.environ.get("FRONTEND_DIR") or PROJECT_ROOT / "frontend")
    settings = Settings(
        data_dir=data_dir,
        frontend_dir=frontend_dir,
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        embedding_dim=_get_int("EMBEDDING_DIM", 384),
        chunk_size_chars=_get_int("CHUNK_SIZE_CHARS", 900, 200),
        chunk_overlap_chars=_get_int("CHUNK_OVERLAP_CHARS", 150, 0),
        dense_top_k=_get_int("DENSE_TOP_K", 20),
        bm25_top_k=_get_int("BM25_TOP_K", 20),
        rrf_k=_get_int("RRF_K", 60),
        max_context_chunks=_get_int("MAX_CONTEXT_CHUNKS", 6),
        max_context_chars=_get_int("MAX_CONTEXT_CHARS", 6000, 500),
        adjacent_chunk_window=_get_int("ADJACENT_CHUNK_WINDOW", 1, 0),
        # Provisional value; calibrate on the labelled validation set before relying on it.
        min_dense_score=_get_float("MIN_DENSE_SCORE", 0.30),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 45),
        llm_max_retries=_get_int("LLM_MAX_RETRIES", 1, 0),
        llm_max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 1800),
        max_concurrent_requests=_get_int("MAX_CONCURRENT_REQUESTS", 2),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        hybrid_top_k=_get_int("HYBRID_TOP_K", 10),
        expansion_top_n=_get_int("EXPANSION_TOP_N", 2, 0),
        # Provisional lexical-evidence gate; calibrate on the labelled set like MIN_DENSE_SCORE.
        min_bm25_score=_get_float("MIN_BM25_SCORE", 3.0),
        request_timeout_seconds=_get_int("REQUEST_TIMEOUT_SECONDS", 90),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
        preload_embedding_model=_get_bool("PRELOAD_EMBEDDING_MODEL", False),
        cors_origins=tuple(
            o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
        ),
    )
    if settings.chunk_overlap_chars >= settings.chunk_size_chars:
        raise ConfigError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_file(PROJECT_ROOT / ".env")
    return _build_settings()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
