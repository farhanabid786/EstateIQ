import pytest
from fastapi.testclient import TestClient

from app.llm.openrouter import LLMRateLimitError, LLMUnavailableError
from app.main import create_app
from app.retrieval.index_loader import IndexLoadError


@pytest.fixture
def make_client(loaded_index, rag_settings, provider, fake_llm_class):
    def factory(llm=None, index=True, raise_exc=True):
        def loader(settings):
            if not index:
                raise IndexLoadError("Index manifest is missing at /secret/path")
            return loaded_index

        app = create_app(rag_settings, index_loader=loader, embedder=provider,
                         llm=llm or fake_llm_class(["Zorvia Heights is a fictional tower [S1]."]))
        return TestClient(app, raise_server_exceptions=raise_exc)

    return factory


def chat(client, **payload):
    body = {"question": "Tell me about Zorvia Heights tower", "company": "both", "language": "auto"}
    body.update(payload)
    return client.post("/api/chat", json=body)


def test_health_ok_without_secrets(make_client, rag_settings):
    with make_client() as client:
        data = client.get("/health").json()
    assert data["status"] == "ok" and data["index_loaded"] is True
    assert data["embedding_model"] == rag_settings.embedding_model
    assert data["embedding_provider"] == "jina"
    assert data["embedding_provider_configured"] is True
    assert data["embedding_dimensions"] == rag_settings.embedding_dim
    assert "llm_configured" in data and data["index_build_id"]
    assert "key" not in str(data).lower().replace("llm_configured", "")


def test_chat_returns_contract_fields(make_client):
    with make_client() as client:
        response = chat(client)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"answer", "english_summary", "detected_language", "company_scope",
                         "insufficient_evidence", "sources", "request_id"}
    assert data["sources"][0]["url"].startswith("https://example.test/")
    assert set(data["sources"][0]) == {"document_id", "title", "url", "company", "published_at"}
    assert response.headers["X-Request-ID"] == data["request_id"]


def test_chat_defaults_for_optional_fields(make_client):
    with make_client() as client:
        response = client.post("/api/chat", json={"question": "Tell me about Zorvia Heights tower"})
    assert response.status_code == 200 and response.json()["company_scope"] == "darglobal"


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 2001}, {},
    {"question": "ok", "company": "other"}, {"question": "ok", "language": "fr"},
])
def test_invalid_chat_requests_return_422(make_client, payload):
    with make_client() as client:
        response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"].startswith("Invalid request") and body["request_id"]


def test_unknown_question_is_declined(make_client, fake_llm_class):
    llm = fake_llm_class()
    with make_client(llm) as client:
        data = chat(client, question="xylophone quantum zebra").json()
    assert data["insufficient_evidence"] is True and data["sources"] == [] and llm.calls == []


def test_sources_pagination_and_filter(make_client):
    with make_client() as client:
        page = client.get("/api/sources", params={"limit": 2, "offset": 0}).json()
        assert page["total"] == 4 and len(page["items"]) == 2 and page["limit"] == 2
        wasalt = client.get("/api/sources", params={"company": "wasalt"}).json()
        assert wasalt["total"] == 1 and wasalt["items"][0]["company"] == "wasalt"
        assert client.get("/api/sources", params={"limit": 1000}).status_code == 422
        assert client.get("/api/sources", params={"company": "x"}).status_code == 422


def test_stats(make_client):
    with make_client() as client:
        data = client.get("/api/stats").json()
    assert data["document_count"] == 4 and data["chunk_count"] > 4
    assert data["company_distribution"] == {"darglobal": 3, "wasalt": 1}
    assert data["content_type_distribution"] == {"webpage": 4}
    assert data["index_built_at"] == "2026-11-18T12:00:00Z" and data["index_build_id"]


def test_projects_use_indexed_titles(make_client):
    with make_client() as client:
        data = client.get("/api/projects").json()
        titles = {item["title"] for item in data["items"]}
        assert titles == {"Zorvia Heights", "مشروع النخيل الأزرق"} and data["total"] == 2
        assert client.get("/api/projects", params={"company": "wasalt"}).json()["total"] == 0


def test_missing_index_reports_degraded_and_fails_clearly(make_client):
    with make_client(index=False) as client:
        health = client.get("/health").json()
        assert health["status"] == "degraded" and health["index_loaded"] is False
        assert "/secret/path" not in str(health)
        for response in (chat(client), client.get("/api/stats"), client.get("/api/sources"), client.get("/api/projects")):
            assert response.status_code == 503
            assert "/secret/path" not in response.text


@pytest.mark.parametrize("error, status", [(LLMUnavailableError(), 502), (LLMRateLimitError(), 429)])
def test_llm_failures_map_to_http_errors(make_client, fake_llm_class, error, status):
    with make_client(fake_llm_class(error=error)) as client:
        response = chat(client)
    assert response.status_code == status
    body = response.json()
    assert body["request_id"] and "Traceback" not in response.text


def test_unexpected_errors_are_hidden(make_client, fake_llm_class):
    class Exploding(fake_llm_class):
        async def generate(self, messages):
            raise RuntimeError("secret internal detail /etc/passwd")

    with make_client(Exploding(), raise_exc=False) as client:
        response = client.post("/api/chat", json={"question": "Tell me about Zorvia Heights tower"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "secret" not in response.text


def test_docs_openapi_and_fallback_frontend(make_client):
    with make_client() as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
        assert {"/api/chat", "/health", "/api/sources", "/api/stats", "/api/projects"} <= set(schema["paths"])
        page = client.get("/")
    assert page.status_code == 200 and "EstateIQ" in page.text


def test_existing_frontend_directory_is_preferred(tmp_path, loaded_index, rag_settings, provider, fake_llm_class):
    from dataclasses import replace

    (tmp_path / "index.html").write_text("<html>custom frontend</html>", encoding="utf-8")
    app = create_app(replace(rag_settings, frontend_dir=tmp_path), index_loader=lambda s: loaded_index,
                     embedder=provider, llm=fake_llm_class())
    with TestClient(app) as client:
        assert "custom frontend" in client.get("/").text


def test_cors_only_when_configured(loaded_index, rag_settings, provider, fake_llm_class):
    from dataclasses import replace

    def build(origins):
        app = create_app(replace(rag_settings, cors_origins=origins), index_loader=lambda s: loaded_index,
                         embedder=provider, llm=fake_llm_class())
        return TestClient(app)

    headers = {"Origin": "https://site.example"}
    with build(()) as client:
        assert "access-control-allow-origin" not in client.get("/health", headers=headers).headers
    with build(("https://site.example",)) as client:
        assert client.get("/health", headers=headers).headers["access-control-allow-origin"] == "https://site.example"
