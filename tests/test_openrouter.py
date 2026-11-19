import json
from dataclasses import replace

import httpx
import pytest

pytestmark = pytest.mark.anyio

from app.llm.openrouter import (
    LLMAuthError, LLMConfigError, LLMRateLimitError, LLMResponseError, LLMTimeoutError,
    LLMUnavailableError, OpenRouterClient,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def make_client(handler, settings, retries=0, key="test-key"):
    cfg = replace(settings, openrouter_api_key=key, llm_max_retries=retries)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterClient(cfg, http_client=http, retry_backoff_seconds=0)


def ok(content="Hello [S1]"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_successful_response_and_request_shape(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return ok()

    text = await make_client(handler, settings).generate(MESSAGES)
    assert text == "Hello [S1]"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == settings.openrouter_model
    assert seen["body"]["messages"] == MESSAGES


async def test_think_blocks_are_removed(settings):
    client = make_client(lambda r: ok("<think>reasoning</think>Final answer"), settings)
    assert await client.generate(MESSAGES) == "Final answer"


async def test_missing_api_key_raises_config_error(settings):
    called = []
    client = make_client(lambda r: called.append(1) or ok(), settings, key="")
    with pytest.raises(LLMConfigError):
        await client.generate(MESSAGES)
    assert called == []


@pytest.mark.parametrize("status", [401, 403, 402])
async def test_authentication_and_payment_errors(settings, status):
    client = make_client(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}), settings)
    with pytest.raises(LLMAuthError):
        await client.generate(MESSAGES)


async def test_rate_limit_error(settings):
    client = make_client(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}), settings)
    with pytest.raises(LLMRateLimitError) as info:
        await client.generate(MESSAGES)
    assert info.value.http_status == 429


async def test_timeout(settings):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError):
        await make_client(handler, settings).generate(MESSAGES)


async def test_network_error(settings):
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(LLMUnavailableError):
        await make_client(handler, settings).generate(MESSAGES)


async def test_invalid_json_body(settings):
    client = make_client(lambda r: httpx.Response(200, content=b"<html>not json"), settings)
    with pytest.raises(LLMResponseError):
        await client.generate(MESSAGES)


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {}}]},
                                     {"choices": [{"message": {"content": 5}}]}])
async def test_unexpected_structure(settings, payload):
    client = make_client(lambda r: httpx.Response(200, json=payload), settings)
    with pytest.raises(LLMResponseError):
        await client.generate(MESSAGES)


@pytest.mark.parametrize("content", ["", "   ", "<think>only thoughts</think>"])
async def test_empty_response(settings, content):
    with pytest.raises(LLMResponseError, match="empty"):
        await make_client(lambda r: ok(content), settings).generate(MESSAGES)


async def test_model_unavailable(settings):
    for response in (httpx.Response(404, json={"error": {"message": "No endpoints found"}}),
                     httpx.Response(400, json={"error": {"message": "model is not a valid model ID"}})):
        with pytest.raises(LLMUnavailableError):
            await make_client(lambda r, resp=response: resp, settings).generate(MESSAGES)


async def test_upstream_5xx_and_error_body_with_200(settings):
    with pytest.raises(LLMUnavailableError):
        await make_client(lambda r: httpx.Response(503, json={}), settings).generate(MESSAGES)
    body = {"error": {"code": 429, "message": "rate limited upstream"}}
    with pytest.raises(LLMRateLimitError):
        await make_client(lambda r: httpx.Response(200, json=body), settings).generate(MESSAGES)


async def test_retries_transient_errors_then_succeeds(settings):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={}) if len(attempts) == 1 else ok("Recovered")

    assert await make_client(handler, settings, retries=1).generate(MESSAGES) == "Recovered"
    assert len(attempts) == 2


async def test_does_not_retry_auth_errors(settings):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(401, json={})

    with pytest.raises(LLMAuthError):
        await make_client(handler, settings, retries=3).generate(MESSAGES)
    assert len(attempts) == 1


async def test_error_messages_do_not_contain_the_api_key(settings):
    client = make_client(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}), settings, key="sk-secret-123")
    with pytest.raises(LLMAuthError) as info:
        await client.generate(MESSAGES)
    assert "sk-secret-123" not in str(info.value) and "sk-secret-123" not in info.value.public_message
