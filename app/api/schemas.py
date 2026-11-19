from typing import Annotated, Literal, Optional

from pydantic import BaseModel, StringConstraints

from app.models import Category, Company, ContentType, DocumentLanguage

MAX_QUESTION_CHARS = 2000

CompanySelector = Literal["darglobal", "wasalt", "both"]
LanguageSelector = Literal["auto", "en", "ar"]
DetectedLanguage = Literal["en", "ar", "mixed", "unknown"]

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
]


class ChatRequest(BaseModel):
    question: Question
    company: CompanySelector = "both"
    language: LanguageSelector = "auto"


class SourceRef(BaseModel):
    document_id: str
    title: str
    url: str
    company: Company
    published_at: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    english_summary: Optional[str] = None
    detected_language: DetectedLanguage
    company_scope: CompanySelector
    insufficient_evidence: bool
    sources: list[SourceRef]
    request_id: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    embedding_model_ready: bool
    llm_configured: bool
    embedding_model: Optional[str] = None
    index_build_id: Optional[str] = None
    detail: Optional[str] = None


class SourceSummary(BaseModel):
    document_id: str
    title: str
    url: str
    company: Company
    content_type: ContentType
    category: Optional[Category] = None
    language: DocumentLanguage
    published_at: Optional[str] = None
    collected_at: str


class SourcesPage(BaseModel):
    items: list[SourceSummary]
    total: int
    offset: int
    limit: int


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    company_distribution: dict[str, int]
    content_type_distribution: dict[str, int] = {}
    index_built_at: Optional[str] = None
    index_build_id: Optional[str] = None
    embedding_model: Optional[str] = None


class ProjectSummary(BaseModel):
    title: str
    company: Company


class ProjectsResponse(BaseModel):
    items: list[ProjectSummary]
    total: int
