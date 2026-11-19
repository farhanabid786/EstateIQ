import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import Settings, configure_logging, get_settings
from app.llm.openrouter import OpenRouterClient
from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider, get_embedding_provider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.index_loader import IndexLoadError, LoadedIndex, load_index
from app.services.chat_service import ChatService, ChatServiceError

logger = logging.getLogger(__name__)

FALLBACK_FRONTEND = Path(__file__).resolve().parent / "static" / "fallback"
DESCRIPTION = (
    "Source-grounded question answering about DarGlobal and Wasalt using indexed official "
    "public information. Answers cite only retrieved sources."
)


def _error(status: int, detail: str, request: Request, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "request_id": getattr(request.state, "request_id", None)},
        headers=headers,
    )


def create_app(
    settings: Optional[Settings] = None,
    index_loader: Callable[[Settings], LoadedIndex] = load_index,
    embedder: Optional[EmbeddingProvider] = None,
    llm: Optional[OpenRouterClient] = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    embedder = embedder or get_embedding_provider()
    llm = llm or OpenRouterClient(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            index = index_loader(settings)
        except IndexLoadError as exc:
            logger.error("Index unavailable: %s", exc)
            app.state.index_error = "Index artifacts are missing or inconsistent; see server logs."
        else:
            app.state.index = index
            app.state.service = ChatService(index, HybridRetriever(index, embedder, settings), llm, settings)
            if settings.preload_embedding_model:
                try:
                    embedder.embed_query("warm up")
                except EmbeddingError as exc:
                    logger.error("Embedding model preload failed: %s", exc)
        yield
        await llm.aclose()

    app = FastAPI(title="EstateIQ", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.index = None
    app.state.service = None
    app.state.index_error = None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ChatServiceError)
    async def chat_error_handler(request: Request, exc: ChatServiceError) -> JSONResponse:
        headers = {"Retry-After": "10"} if exc.status_code in (429, 503) else None
        return _error(exc.status_code, exc.detail, request, headers)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), request)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        return _error(422, "Invalid request: check " + ", ".join(fields or ["the request body"]), request)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return _error(500, "Internal server error", request)

    app.include_router(router)

    # The page references assets at /frontend/<filename>. Mount the real asset
    # directory at that URL prefix, then serve index.html from the root URL.
    frontend = settings.frontend_dir
    static_dir = frontend if (frontend / "index.html").is_file() else FALLBACK_FRONTEND
    if not static_dir.is_dir():
        raise RuntimeError(f"Frontend directory does not exist: {static_dir}")

    app.mount("/frontend", StaticFiles(directory=static_dir, check_dir=True), name="frontend-assets")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()
