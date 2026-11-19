# API Contract

Serve frontend and API from the same FastAPI origin.

## POST /api/chat

Request:
{
  "question": "Tell me about AIDA",
  "company": "darglobal",
  "language": "auto"
}

Fields:
- `question`: required string, trimmed, length-limited (e.g. 2,000 chars)
- `company`: `darglobal`, `wasalt`, or `both`; optional default `both`
- `language`: `auto`, `en`, or `ar`; optional default `auto`

Response:
{
  "answer": "Grounded answer...",
  "english_summary": null,
  "detected_language": "en",
  "company_scope": "darglobal",
  "insufficient_evidence": false,
  "sources": [
    {
      "document_id": "source-id",
      "title": "Official source title",
      "url": "https://official.example/page",
      "company": "darglobal",
      "published_at": null
    }
  ],
  "request_id": "generated-id"
}

For Arabic:
- `answer` is Arabic.
- `english_summary` contains a concise English summary.
For English:
- `answer` is English.
- `english_summary` is null.
For mixed:
- respond naturally in mixed style; set English summary when the answer is mainly Arabic.

The server, not the LLM, attaches source URLs. Only citations retrieved from the local metadata store may be returned.

## Other endpoints

- `GET /health`: status, index loaded, model readiness (no secrets)
- `GET /api/sources`: paginated source summaries
- `GET /api/stats`: document/chunk counts, company distribution, index build date
- `GET /docs`: FastAPI Swagger UI
- `GET /openapi.json`: OpenAPI schema

Do not expose filesystem paths, API keys, stack traces, or internal environment variables.

## Errors
Use consistent JSON errors with a short human-readable `detail`. Handle invalid input, missing index, OpenRouter timeout, 429 rate limit, upstream 5xx, and malformed model response. Do not return a fabricated answer on failure.
