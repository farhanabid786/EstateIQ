import json
from dataclasses import replace

import httpx
import pytest

from app.llm.openrouter import LLMTruncatedError, OpenRouterClient

pytestmark = pytest.mark.anyio

MESSAGES = [{"role": "user", "content": "Explain this briefly"}]


def make_client(handler, settings):
    cfg = replace(settings, openrouter_api_key="test-key", llm_max_retries=0, llm_max_output_tokens=1200)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterClient(cfg, http_client=http, retry_backoff_seconds=0)


async def test_retries_truncated_output_once_with_larger_budget(settings):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["max_tokens"])
        if len(calls) == 1:
            return httpx.Response(200, json={"model": "test", "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
        return httpx.Response(200, json={"model": "test", "choices": [{"finish_reason": "stop", "message": {"content": "Complete answer [S1]."}}]})

    client = make_client(handler, settings)
    assert await client.generate(MESSAGES) == "Complete answer [S1]."
    assert calls == [1200, 2400]
    await client._client.aclose()


async def test_still_truncated_after_bounded_retry_returns_typed_error(settings):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})

    client = make_client(handler, settings)
    with pytest.raises(LLMTruncatedError):
        await client.generate(MESSAGES)
    assert calls == [1200, 2400]
    await client._client.aclose()


async def test_structured_content_ignores_non_text_blocks(settings):
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": [
        {"type": "image", "image_url": "data:image/png;base64,abc"},
        {"type": "text", "text": "First part."},
        {"type": "output_text", "text": "Second part."},
    ]}}]}
    client = make_client(lambda request: httpx.Response(200, json=payload), settings)
    assert await client.generate(MESSAGES) == "First part.\nSecond part."
    await client._client.aclose()
