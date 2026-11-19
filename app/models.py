import re
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Company = Literal["darglobal", "wasalt"]
ContentType = Literal["webpage", "pdf", "press_release"]
DocumentLanguage = Literal["en", "ar", "unknown"]
Category = Literal[
    "project", "company", "leadership", "investor", "service",
    "faq", "news", "policy", "report", "other",
]
ExtractionMethod = Literal["html", "pdf_text", "ocr"]

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _check_iso(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be an ISO-8601 date or datetime") from exc
    return value


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(min_length=1)
    company: Company
    title: str
    content: str
    source_url: str
    content_type: ContentType
    language: DocumentLanguage
    collected_at: str
    published_at: Optional[str] = None
    file_hash: str
    category: Optional[Category] = None
    description: Optional[str] = None
    location: Optional[str] = None
    page_number: Optional[int] = None
    parent_document_id: Optional[str] = None
    source_domain: Optional[str] = None
    extraction_method: Optional[ExtractionMethod] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("source_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must be an absolute http(s) URL")
        return value

    @field_validator("collected_at")
    @classmethod
    def _collected_iso(cls, value: str) -> str:
        return _check_iso(value)

    @field_validator("published_at")
    @classmethod
    def _published_iso(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _check_iso(value)

    @field_validator("file_hash")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not _SHA256_HEX.match(value):
            raise ValueError("must be a 64-character SHA-256 hex digest")
        return value.lower()


class ChunkRecord(BaseModel):
    vector_id: int = Field(ge=0)
    chunk_id: str
    document_id: str
    position: int = Field(ge=0)
    company: Company
    category: Optional[Category] = None
    title: str
    source_url: str
    language: DocumentLanguage
    page_number: Optional[int] = None
    published_at: Optional[str] = None
    collected_at: str
    text: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.title}\n{self.text}"
