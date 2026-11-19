# Test Plan

## Scraper
- Restricts crawl to approved domains.
- Respects robots.txt and rate limits.
- Handles 404, timeout, redirect, malformed HTML.
- Extracts page title and meaningful text.
- Deduplicates URLs/content.
- Writes valid UTF-8 JSONL matching DATA_SCHEMA.
- Handles PDF text extraction.
- OCR is attempted only when text extraction is insufficient and within configured limits.
- Failed pages are recorded, not silently ignored.

## Retrieval
Create a small hand-labelled test set from the collected corpus. Include:
- DarGlobal company fact question.
- DarGlobal project name/location question.
- Wasalt product/service question.
- Exact brand/project-name question.
- Arabic question about English source content.
- English question about Arabic source content if Arabic source data exists.
- Both-company comparison question.
- Ambiguous company question.
- Unknown question not answered by the corpus.

Measure:
- Recall@5 for relevant source/chunk retrieval.
- Precision@5 where labels allow.
- Correct company filtering.
- Citation URL correctness.
- Insufficient-evidence behaviour.
- Arabic/English retrieval quality.

Do not invent benchmark scores. Record test-set size and actual results.

## LLM/API
- OpenRouter minimal request succeeds.
- Missing API key produces a clear configuration error.
- Timeout and rate-limit errors are handled.
- LLM cannot invent source URLs.
- Prompt-injection text inside scraped content is treated as untrusted.
- Unsupported facts are declined.
- Arabic response includes English summary.
- English response is English.
- Mixed-language handling is acceptable.
- Input length limits work.

## Frontend
- Sends correct API request.
- Displays answer and citations.
- Handles loading and error states.
- Company scope selection works.
- Arabic answer uses RTL direction.
- Mobile layout is usable.
- API errors are visible and do not crash UI.

## Docker/Render
- Docker image builds.
- Container starts with packaged index/database.
- `/health` succeeds.
- `/docs` and `/openapi.json` work.
- `/api/chat` returns grounded answer.
- No secret is baked into image.
- Memory is measured and stays within Render Free limits.
- Cold start and first request are checked.
- Render ephemeral filesystem is not relied on for persistent data.
