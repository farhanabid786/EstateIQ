import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas import (
    ChatRequest, ChatResponse, ErrorResponse, HealthResponse, ProjectsResponse,
    ProjectSummary, SourcesPage, StatsResponse,
)
from app.database.metadata import MetadataError, MetadataStore
from app.models import Company

router = APIRouter()

ERROR_RESPONSES = {
    429: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
    503: {"model": ErrorResponse}, 504: {"model": ErrorResponse},
}
MAX_PAGE_SIZE = 100


def _metadata(request: Request) -> MetadataStore:
    index = request.app.state.index
    if index is None:
        raise HTTPException(status_code=503, detail="The knowledge base is not available.")
    return index.metadata


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    state = request.app.state
    index = state.index
    return HealthResponse(
        status="ok" if index is not None else "degraded",
        index_loaded=index is not None,
        embedding_provider="jina",
        embedding_provider_configured=state.embedder.is_configured,
        embedding_dimensions=state.settings.embedding_dim,
        llm_configured=state.settings.llm_configured,
        embedding_model=state.settings.embedding_model,
        index_build_id=index.build_id if index is not None else None,
        detail=None if index is not None else state.index_error,
    )


@router.post("/api/chat", response_model=ChatResponse, responses=ERROR_RESPONSES, tags=["chat"])
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="The knowledge base is not available.")
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    return await service.answer(payload, request_id)


@router.get("/api/sources", response_model=SourcesPage, tags=["knowledge base"])
def sources(
    request: Request,
    company: Optional[Company] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
) -> SourcesPage:
    try:
        items, total = _metadata(request).list_sources(company, offset, limit)
    except MetadataError as exc:
        raise HTTPException(status_code=503, detail="The knowledge base is unavailable.") from exc
    return SourcesPage(items=items, total=total, offset=offset, limit=limit)


@router.get("/api/stats", response_model=StatsResponse, tags=["knowledge base"])
def stats(request: Request) -> StatsResponse:
    try:
        return _metadata(request).stats()
    except MetadataError as exc:
        raise HTTPException(status_code=503, detail="The knowledge base is unavailable.") from exc


@router.get("/api/projects", response_model=ProjectsResponse, tags=["knowledge base"])
def projects(request: Request, company: Optional[Company] = None) -> ProjectsResponse:
    try:
        rows = _metadata(request).project_titles()
    except MetadataError as exc:
        raise HTTPException(status_code=503, detail="The knowledge base is unavailable.") from exc
    items = sorted(
        (ProjectSummary(title=title, company=owner) for title, owner in rows if company in (None, owner)),
        key=lambda item: (item.company, item.title.casefold()),
    )
    return ProjectsResponse(items=items, total=len(items))
