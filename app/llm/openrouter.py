import asyncio
import logging
import re
from typing import Any, Optional, Sequence

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
MAX_RETRY_AFTER_SECONDS = 5.0


class LLMError(RuntimeError):
    http_status = 502
    public_message = "The language model service is currently unavailable."

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or self.public_message)


class LLMConfigError(LLMError):
    http_status = 503
    public_message = "The language model service is not configured."


class LLMAuthError(LLMError):
    http_status = 502
    public_message = "The language model service rejected the request."


class LLMRateLimitError(LLMError):
    http_status = 429
    public_message = "The language model service is busy. Please retry shortly."


class LLMTimeoutError(LLMError):
    http_status = 504
    public_message = "The language model service timed out."


class LLMUnavailableError(LLMError):
    http_status = 502
    public_message = "The language model service is currently unavailable."


class LLMResponseError(LLMError):
    http_status = 502
    public_message = "The language model service returned an invalid response."


class LLMTruncatedError(LLMResponseError):
    """The provider stopped generation because the output-token limit was reached."""
    public_message = "The answer was too long for the model to complete. Please ask a more specific question."


MAX_TRUNCATION_RETRIES = 1
MAX_OUTPUT_TOKENS_RETRY = 3000


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", ""))[:200]
        if isinstance(error, str):
            return error[:200]
    return ""


def _classify_status(status: int, message: str) -> LLMError:
    lowered = message.lower()
    if status in (401, 403):
        return LLMAuthError(f"HTTP {status}")
    if status == 402:
        return LLMAuthError("HTTP 402 insufficient credits")
    if status == 429:
        return LLMRateLimitError("HTTP 429")
    if status == 404 or (status == 400 and "model" in lowered):
        return LLMUnavailableError(f"HTTP {status} model unavailable")
    return LLMUnavailableError(f"HTTP {status}")


class OpenRouterClient:
    def __init__(
        self,
        settings: Settings,
        http_client: Optional[httpx.AsyncClient] = None,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._settings = settings
        self._backoff = retry_backoff_seconds
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=float(settings.llm_timeout_seconds), write=10.0, pool=10.0)
        )

    @property
    def is_configured(self) -> bool:
        return self._settings.llm_configured

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.is_configured:
            raise LLMConfigError("OPENROUTER_API_KEY is not set")

        output_tokens = self._settings.llm_max_output_tokens
        truncation_retries = 0
        transient_attempt = 0
        max_transient_attempts = self._settings.llm_max_retries + 1

        while transient_attempt < max_transient_attempts:
            try:
                return await self._request(messages, max_tokens=output_tokens)
            except LLMTruncatedError:
                if truncation_retries >= MAX_TRUNCATION_RETRIES:
                    raise
                next_limit = min(max(output_tokens + 512, output_tokens * 2), MAX_OUTPUT_TOKENS_RETRY)
                if next_limit <= output_tokens:
                    raise
                output_tokens = next_limit
                truncation_retries += 1
                logger.warning(
                    "OpenRouter output was truncated; retrying once with max_tokens=%d",
                    output_tokens,
                )
            except (LLMRateLimitError, LLMTimeoutError, LLMUnavailableError) as exc:
                transient_attempt += 1
                if transient_attempt >= max_transient_attempts:
                    raise
                delay = min(self._backoff * (2 ** (transient_attempt - 1)), MAX_RETRY_AFTER_SECONDS)
                logger.warning("LLM call failed (%s); retrying in %.1fs", type(exc).__name__, delay)
                await asyncio.sleep(delay)
        raise LLMUnavailableError()

    async def _request(self, messages: Sequence[dict[str, str]], max_tokens: Optional[int] = None) -> str:
        url = self._settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self._settings.openrouter_model,
            "messages": list(messages),
            "temperature": self._settings.llm_temperature,
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": "EstateIQ",
        }
        try:
            response = await self._client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"network error: {type(exc).__name__}") from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code != 200:
            raise _classify_status(response.status_code, _error_message(payload))
        if payload is None:
            raise LLMResponseError("response body is not JSON")
        if isinstance(payload, dict) and "error" in payload:
            error = payload["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise _classify_status(code if isinstance(code, int) else 502, _error_message(payload))
        return self._extract_text(payload)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize OpenAI-compatible string or structured text content."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    # OpenAI-compatible providers may use either `text` or
                    # `content` for text blocks. Ignore image/audio payloads.
                    block_type = block.get("type")
                    if block_type in (None, "text", "output_text"):
                        value = block.get("text")
                        if value is None and block_type is None:
                            value = block.get("content")
                        if isinstance(value, str) and value.strip():
                            parts.append(value)
            return "\n".join(parts)
        return ""

    @classmethod
    def _extract_text(cls, payload: Any) -> str:
        try:
            if not isinstance(payload, dict):
                raise TypeError("payload is not an object")
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise TypeError("choices is missing or invalid")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise TypeError("message is missing or invalid")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("unexpected response structure") from exc

        finish_reason = choice.get("finish_reason")
        if finish_reason in ("length", "max_tokens"):
            logger.warning(
                "OpenRouter response reached max_tokens (model=%s); increase LLM_MAX_OUTPUT_TOKENS",
                payload.get("model", "unknown") if isinstance(payload, dict) else "unknown",
            )
            raise LLMTruncatedError("Model output was truncated (finish_reason=length)")

        if finish_reason == "content_filter":
            raise LLMResponseError("model response was blocked by its content filter")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise LLMResponseError("model refused the request")

        content = message.get("content") if isinstance(message, dict) else None
        text = cls._content_to_text(content)
        text = _THINK_BLOCK.sub("", text).strip()
        if not text:
            raise LLMResponseError("model returned empty or unsupported content")

        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        logger.info(
            "OpenRouter response received: model=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s",
            payload.get("model", "unknown") if isinstance(payload, dict) else "unknown",
            finish_reason,
            usage.get("prompt_tokens", "unknown") if isinstance(usage, dict) else "unknown",
            usage.get("completion_tokens", "unknown") if isinstance(usage, dict) else "unknown",
        )
        return text
