import asyncio
import logging
import re
import time
from typing import Optional, Protocol, Sequence

from app.api.schemas import ChatRequest, ChatResponse
from app.classification.classifier import QueryClassification, QueryClassifier, build_project_aliases
from app.config import Settings
from app.database.metadata import MetadataError
from app.llm.openrouter import LLMError
from app.llm.prompts import (
    INSUFFICIENT_AR, INSUFFICIENT_EN, build_messages, build_summary_messages,
    dominant_language, parse_model_output,
)
from app.memory_diagnostics import log_rss
from app.retrieval.context_builder import ContextBuilder, EvidenceContext
from app.retrieval.embeddings import EmbeddingError
from app.retrieval.hybrid import HybridRetriever, has_sufficient_evidence
from app.retrieval.index_loader import LoadedIndex
from app.services.citations import strip_markers, validate_citations
from app.text import detect_language

logger = logging.getLogger(__name__)

QUEUE_TIMEOUT_SECONDS = 15.0


class LLMClient(Protocol):
    async def generate(self, messages: Sequence[dict[str, str]]) -> str: ...


class ChatServiceError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ChatService:
    def __init__(
        self,
        index: LoadedIndex,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._context_builder = ContextBuilder(index.metadata, settings)
        self._classifier = QueryClassifier(build_project_aliases(index.metadata.project_titles()))
        self._slots = asyncio.Semaphore(settings.max_concurrent_requests)

    async def answer(self, request: ChatRequest, request_id: str) -> ChatResponse:
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise ChatServiceError(503, "The service is busy. Please retry shortly.") from exc
        try:
            return await asyncio.wait_for(
                self._answer(request, request_id), timeout=self._settings.request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise ChatServiceError(504, "The request took too long. Please try again.") from exc
        finally:
            self._slots.release()

    async def _answer(self, request: ChatRequest, request_id: str) -> ChatResponse:
        started = time.monotonic()
        log_rss("before_full_chat")
        classification = self._classifier.classify(request.question, request.company, request.language)
        if self._explicitly_unsupported(request.question):
            return self._insufficient(request.question, classification, request_id)
        context = await self._retrieve(request.question, classification)
        logger.info(
            "request=%s language=%s scope=%s topic=%s question_chars=%d evidence=%s",
            request_id, classification.language, classification.scope, classification.topic,
            len(request.question), context.reason or "sufficient",
        )
        if not context.sufficient:
            return self._insufficient(request.question, classification, request_id)

        messages = build_messages(request.question, context.blocks, classification.language)
        raw = await self._generate(messages)
        parsed = parse_model_output(raw)
        if parsed.insufficient:
            return self._insufficient(request.question, classification, request_id)

        cited = validate_citations(parsed.answer, context.blocks)
        if cited.used_fallback:
            return self._insufficient(request.question, classification, request_id)
        summary = await self._summary(classification, cited.text, parsed.summary)
        logger.info("request=%s answered in %.2fs with %d source(s)", request_id,
                    time.monotonic() - started, len(cited.sources))
        log_rss("after_full_chat")
        return ChatResponse(
            answer=cited.text,
            english_summary=summary,
            detected_language=classification.language,
            company_scope=classification.response_scope,
            insufficient_evidence=False,
            sources=cited.sources,
            request_id=request_id,
        )

    @staticmethod
    def _explicitly_unsupported(question: str) -> bool:
        return re.search(
            r"\b(confidential|private|secret|password|credentials|credential|"
            r"personal (?:details|data|list)|shareholder identities|ownership identities)\b",
            question,
            flags=re.IGNORECASE,
        ) is not None

    async def _retrieve(self, question: str, classification: QueryClassification) -> EvidenceContext:
        try:
            return await asyncio.to_thread(self._retrieve_sync, question, classification)
        except EmbeddingError as exc:
            logger.error("Embedding unavailable: %s", exc)
            raise ChatServiceError(503, "The search model is unavailable.") from exc
        except MetadataError as exc:
            logger.error("Metadata error during retrieval: %s", exc)
            raise ChatServiceError(503, "The knowledge base is unavailable.") from exc

    def _retrieve_sync(self, question: str, classification: QueryClassification) -> EvidenceContext:
        results = self._retriever.retrieve(question, classification.companies)
        if not has_sufficient_evidence(results, self._settings.min_dense_score, self._settings.min_bm25_score):
            return EvidenceContext((), False, "below_threshold" if results else "no_results")
        return self._context_builder.build(results)

    async def _generate(self, messages: list[dict[str, str]]) -> str:
        try:
            return await self._llm.generate(messages)
        except LLMError as exc:
            logger.error("LLM call failed: %s: %s", type(exc).__name__, exc)
            raise ChatServiceError(exc.http_status, exc.public_message) from exc

    async def _summary(
        self, classification: QueryClassification, answer: str, model_summary: Optional[str]
    ) -> Optional[str]:
        needs_summary = classification.language == "ar" or detect_language(answer) == "ar"
        if not needs_summary:
            return None
        if model_summary and model_summary.strip():
            return strip_markers(model_summary)
        try:
            summary = await self._llm.generate(
                build_summary_messages(strip_markers(answer))
            )
            return strip_markers(summary) or None
        except LLMError as exc:
            # The primary Arabic answer is still useful; summary generation is
            # best-effort and must never turn a successful answer into a failure.
            logger.warning(
                "English summary unavailable; returning the primary answer: %s",
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _insufficient(question: str, classification: QueryClassification, request_id: str) -> ChatResponse:
        arabic = classification.language == "ar" or (
            classification.language == "mixed" and dominant_language(question) == "ar"
        )
        return ChatResponse(
            answer=INSUFFICIENT_AR if arabic else INSUFFICIENT_EN,
            english_summary=INSUFFICIENT_EN if arabic else None,
            detected_language=classification.language,
            company_scope=classification.response_scope,
            insufficient_evidence=True,
            sources=[],
            request_id=request_id,
        )
