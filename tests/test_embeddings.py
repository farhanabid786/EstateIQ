import numpy as np
import httpx
import pytest

from app.retrieval.embeddings import EmbeddingError, EmbeddingProvider, JINA_DIMENSIONS, JINA_MODEL


def test_query_payload_and_response():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "model": JINA_MODEL,
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0] * JINA_DIMENSIONS}],
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS, batch_size=32,
                                 api_key="test-key", http_client=client)
    vector = provider.embed_query("Tell me about AIDA")
    assert vector.shape == (JINA_DIMENSIONS,)
    payload = requests[0].read()
    assert b'"task":"retrieval.query"' in payload
    assert b'"dimensions":512' in payload
    assert b'"embedding_type":"float"' in payload
    assert b'"normalized":true' in payload
    provider.close()


def test_passage_payload_preserves_batch():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "data": [
                {"index": 1, "embedding": [0.0] * (JINA_DIMENSIONS - 1) + [2.0]},
                {"index": 0, "embedding": [1.0] * JINA_DIMENSIONS},
            ]
        })

    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS, batch_size=2,
                                 api_key="test-key",
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    vectors = provider.embed_documents(["first", "second"])
    assert vectors.shape == (2, JINA_DIMENSIONS)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert np.isclose(np.linalg.norm(vectors[1]), 1.0)
    assert b'"task":"retrieval.passage"' in requests[0].read()
    provider.close()


@pytest.mark.parametrize("status,match", [
    (401, "API key"),
    (403, "forbidden"),
    (400, "rejected"),
])
def test_authentication_errors(status, match):
    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="bad-key",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"error": "bad"})
        )),
    )
    with pytest.raises(EmbeddingError, match=match):
        provider.embed_query("question")
    provider.close()


def test_429_retries():
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(429, json={"error": "busy"})

    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key", max_retries=1,
        backoff_seconds=0.001,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EmbeddingError, match="rate-limited"):
        provider.embed_query("question")
    assert len(attempts) == 2
    provider.close()


def test_5xx_retries():
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={"error": "down"})

    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key", max_retries=1,
        backoff_seconds=0.001,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EmbeddingError, match="upstream"):
        provider.embed_query("question")
    assert len(attempts) == 2
    provider.close()


def test_timeout():
    def handler(request):
        raise httpx.TimeoutException("timeout")

    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EmbeddingError, match="timed out"):
        provider.embed_query("question")
    provider.close()


def test_dimension_and_finite_validation():
    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0] * 511}]})
        )),
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        provider.embed_query("question")
    provider.close()


def test_missing_configuration():
    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS)
    with pytest.raises(EmbeddingError, match="JINA_API_KEY"):
        provider.embed_query("question")
    provider.close()


@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_invalid_text_rejected(bad):
    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS, api_key="test-key")
    with pytest.raises(ValueError):
        provider.embed_documents(["ok", bad])
    provider.close()


def test_empty_input_does_not_request_api():
    def handler(request):
        raise AssertionError("empty input must not call Jina")

    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.embed_documents([]).shape == (0, JINA_DIMENSIONS)
    provider.close()


def test_empty_query_rejected():
    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS, api_key="test-key")
    with pytest.raises(ValueError):
        provider.embed_query("  ")
    provider.close()


def test_single_string_rejected():
    provider = EmbeddingProvider(JINA_MODEL, JINA_DIMENSIONS, api_key="test-key")
    with pytest.raises(TypeError):
        provider.embed_documents("not a list")
    provider.close()


def test_registry_rejects_model_and_dimension_mismatch():
    provider = EmbeddingProvider("old/model", JINA_DIMENSIONS, api_key="test-key")
    with pytest.raises(EmbeddingError, match="configured Jina model"):
        provider.verify_registry()
    provider = EmbeddingProvider(JINA_MODEL, 384, api_key="test-key")
    with pytest.raises(EmbeddingError, match="requires dimension"):
        provider.verify_registry()


def test_malformed_response_schema():
    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"unexpected": []})
        )),
    )
    with pytest.raises(EmbeddingError, match="unexpected response"):
        provider.embed_query("question")
    provider.close()


def test_missing_and_duplicate_indexes_rejected():
    provider = EmbeddingProvider(
        JINA_MODEL, JINA_DIMENSIONS, api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "data": [
                    {"index": 0, "embedding": [1.0] * JINA_DIMENSIONS},
                    {"index": 0, "embedding": [1.0] * JINA_DIMENSIONS},
                ]
            })
        )),
    )
    with pytest.raises(EmbeddingError, match="out-of-order"):
        provider.embed_documents(["one", "two"])
    provider.close()
