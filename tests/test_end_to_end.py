import json

import httpx
from fastapi.testclient import TestClient

from app.llm.openrouter import OpenRouterClient
from app.main import create_app


def test_full_pipeline_with_mocked_embedding_and_openrouter(loaded_index, rag_settings, provider):
    from dataclasses import replace

    captured = {}

    def openrouter(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "Zorvia Heights is described as a fictional residential tower near the harbour [S1]."}}]})

    settings = replace(rag_settings, openrouter_api_key="test-key")
    llm = OpenRouterClient(settings, http_client=httpx.AsyncClient(transport=httpx.MockTransport(openrouter)))
    app = create_app(settings, index_loader=lambda s: loaded_index, embedder=provider, llm=llm)

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"question": "Where is the Zorvia Heights tower?", "company": "both"})

    assert response.status_code == 200
    data = response.json()
    assert data["company_scope"] == "darglobal"
    assert data["insufficient_evidence"] is False
    assert data["answer"].endswith("[S1].")
    source = data["sources"][0]
    assert source["title"] == "Zorvia Heights" and source["company"] == "darglobal"
    assert source["url"] in {f"https://example.test/page-{n}" for n in range(1, 40)}

    sent = captured["body"]
    assert sent["model"] == settings.openrouter_model
    user_prompt = sent["messages"][1]["content"]
    assert "<<<SOURCE S1 | company: darglobal | title: Zorvia Heights | page: 3>>>" in user_prompt
    assert "Platform Overview" not in user_prompt
    assert "test-key" not in json.dumps(sent)


def test_full_pipeline_declines_unknown_question_without_calling_openrouter(loaded_index, rag_settings, provider):
    from dataclasses import replace

    calls = []
    llm = OpenRouterClient(
        replace(rag_settings, openrouter_api_key="test-key"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(500))),
    )
    app = create_app(rag_settings, index_loader=lambda s: loaded_index, embedder=provider, llm=llm)
    with TestClient(app) as client:
        data = client.post("/api/chat", json={"question": "xylophone quantum zebra"}).json()
    assert data["insufficient_evidence"] is True and data["sources"] == [] and calls == []


def test_full_pipeline_arabic_question_retrieves_and_summarises(loaded_index, rag_settings, provider, fake_llm_class):
    llm = fake_llm_class(["يقع المشروع قرب الشاطئ [S1].\nENGLISH_SUMMARY: The project is near the beach."])
    app = create_app(rag_settings, index_loader=lambda s: loaded_index, embedder=provider, llm=llm)
    with TestClient(app) as client:
        data = client.post("/api/chat", json={"question": "أين يقع مشروع النخيل الأزرق؟", "company": "darglobal"}).json()
    assert data["detected_language"] == "ar" and data["english_summary"] == "The project is near the beach."
    assert data["sources"][0]["title"] == "مشروع النخيل الأزرق"
