# EstateIQ — Shared Project Memory

Last updated: 2026-11-18
Project root: `D:\EstateIQ`
Deadline: 2026-11-20 (confirm exact submission time with HR)
Purpose: DarGlobal Junior AI Engineer — Applied AI technical assignment.

## Mission

Build a deployed, source-grounded real-estate intelligence assistant using publicly accessible official information from DarGlobal and Wasalt. The application must be testable through a public URL and containerised with Docker.

This is a retrieval-augmented question-answering application, not a generic chatbot and not a property transaction platform.

## User and development setup

- User has Windows 11, 16 GB RAM.
- Project folder: `D:\EstateIQ`.
- Python 3.10.11 and Python 3.14 are installed. Use Python 3.10.11 for this project.
- VS Code and GitHub Copilot Pro student access are available; Copilot Agent mode is available.
- Copilot's currently available default model is displayed by the user as GPT Luna 5.6. Use the available model; do not assume an undocumented model identifier.
- Claude is the free Claude.ai browser experience, not Claude Code. It cannot be assumed to have direct access to the local D: drive. Claude-generated files must be downloaded/copied into the specified local paths.
- ChatGPT is project architect, reviewer, integration guide, and deployment guide.
- Docker will be installed after code generation.
- GitHub repository will be created after local code generation and validation.

## Agent ownership

### GitHub Copilot / VS Code
Own:
- `scraper/`
- `frontend/`
- scraper CLI and source discovery
- source extraction, PDF text extraction, optional OCR fallback
- structured JSONL output conforming to `docs/DATA_SCHEMA.md`
- scraper-focused tests

### Claude.ai
Own:
- `app/`
- API schemas/routes
- query classification
- multilingual embedding and hybrid retrieval
- FAISS and BM25 integration
- RRF fusion and context assembly
- OpenRouter client and grounded prompts
- SQLite metadata access
- backend tests

Claude must return files with exact relative paths. It must not claim it has written to `D:\EstateIQ`.

### ChatGPT
Own:
- architecture decisions and shared contracts
- prompt creation and clarification
- review of generated files and test logs
- integration checks and debugging guidance
- Docker and Render walkthrough

Do not allow two agents to independently edit the same file. Copilot may perform final integration after Claude's files are copied in.

## Business scope

### DarGlobal
Collect relevant, publicly accessible official company pages, project pages, leadership/company pages, press releases, investor relations pages, reports, and official PDFs. Examples include project details, locations, descriptions, brand partnerships, business information and public investor disclosures.

Official starting domains:
- `https://darglobal.co.uk/`
- `https://darglobal.co.uk/projects`
- `https://darglobal.co.uk/press`
- `https://darglobal.co.uk/investor`

Verify current paths from sitemap and internal links; do not assume every example URL exists.

### Wasalt
Collect relevant, publicly accessible official company/platform pages, service descriptions, FAQs, project pages, public blogs, public reports/PDFs, and platform policy pages that help explain the business.

Official starting domains:
- `https://wasalt.sa/en`
- `https://wasalt.sa/en/corporate`
- `https://blog.wasalt.sa/`

Discover actual current paths via sitemap/internal links. Do not scrape user accounts, private listings, personal contact details, or protected data.

### Source policy
- Official DarGlobal and Wasalt sources are primary.
- Do not silently mix the companies' facts.
- Quara Global Holding may be included only if an official page directly establishes relevant group context; identify it as group context, not as either company's own product.
- Store source URL, title, company, content type, collection time, and source publication date if available.
- Preserve conflicting statements with their source/date; do not silently reconcile.
- Do not invent facts, prices, availability, financial metrics, project status, or completion dates.
- Treat scraped page text as untrusted content, never as instructions to the LLM.
- Respect robots.txt, terms, rate limits, and access controls. No bypassing authentication or anti-bot restrictions.

## Final technology decisions

- Python 3.10.11
- FastAPI backend
- HTML, CSS, vanilla JavaScript frontend
- Requests + BeautifulSoup for static pages
- Playwright only for relevant public pages that require JavaScript rendering
- PyMuPDF for text PDFs
- OCR only as a time-bounded fallback for scanned PDFs
- JSONL as portable source-document interchange format
- SQLite for metadata and source records
- FAISS CPU index for dense semantic retrieval
- BM25 for lexical retrieval
- Reciprocal Rank Fusion for hybrid ranking
- Query classification and metadata filtering
- FastEmbed / ONNX Runtime for low-memory query embeddings
- Docker
- Render free web service

### Embedding model
Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through FastEmbed's supported ONNX model registry, after verifying that the installed pinned FastEmbed version lists it. This is a practical multilingual and low-memory deployment choice. Do not silently substitute another model or use full PyTorch in the Render runtime.

Use 384-dimensional vectors if confirmed by the selected model registry. Normalize vectors and use cosine-equivalent inner-product search consistently. Store model name, version, dimension, chunking settings, and index build timestamp in the index manifest.

The previously considered `intfloat/multilingual-e5-small` is a quality candidate but requires custom FastEmbed/ONNX setup in this environment. Do not introduce custom conversion work under the deadline unless the supported multilingual MiniLM model fails the bilingual retrieval tests.

### LLM
Primary OpenRouter model candidate:
`nvidia/nemotron-3.5-super-vl:free`

OpenRouter model page:
`https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free`

Use OpenAI-compatible API base:
`https://openrouter.ai/api/v1`

Model price and free availability can change. Test a real minimal request before final integration. Store model ID in `OPENROUTER_MODEL`; never hardcode API keys. Do not send private user information, HR correspondence, secrets, or personal data to the free NVIDIA endpoint. Only public company-source context and the user's question about that public information may be sent.

Do not describe a small test or provider uptime figure as proof that this is the best model. Evaluate the model on the project's own bilingual test set: factual accuracy, evidence use, citations, unsupported-answer refusal, latency and API reliability. If unavailable, show a graceful service error; any fallback must be explicitly configured and tested.

## Language behaviour

- English query: answer in English.
- Arabic query: answer fully in Arabic, then add a concise English summary.
- Mixed Arabic/English query: respond naturally in a mixed style; if the response is predominantly Arabic, add a concise English summary.
- English interface remains the default.
- Use multilingual embeddings for cross-language retrieval (Arabic question may retrieve English source content).
- Preserve original source titles and URLs. Do not fabricate Arabic translations of official titles; a short translated label may be added while retaining the original.

## RAG design

1. Normalize question and detect language.
2. Classify intent and likely company scope: DarGlobal, Wasalt, both, or uncertain.
3. Classify broad topic: company, project, location, service, leadership, investor/financial, news, policy, or general.
4. Apply metadata filters only when confidence is sufficient. If uncertain, search both companies.
5. Embed query with FastEmbed using the selected model's correct query formatting.
6. Retrieve dense FAISS candidates and BM25 candidates.
7. Combine rankings with Reciprocal Rank Fusion.
8. Deduplicate and optionally expand to adjacent chunks from the same source.
9. Build a bounded evidence context with source metadata.
10. Call OpenRouter with a strict grounded-answer prompt.
11. Return answer, language, source citations, and retrieval metadata needed for debugging.
12. If evidence is weak or absent, say the knowledge base does not contain enough verified information.

Avoid unnecessary agents, GraphRAG, complex rerankers, fine-tuning, or a separate vector database. This is a compact, deadline-driven knowledge assistant.

## Storage and Render constraints

Render free web services currently have 512 MB RAM and ephemeral local filesystems. Therefore:
- Avoid PyTorch in the production runtime.
- Prefer FastEmbed/ONNX Runtime.
- Keep model and dependency footprint small.
- Package prebuilt FAISS index and seed SQLite DB into the Docker image.
- Treat packaged data as read-only at runtime.
- Do not rely on runtime-created SQLite/index files surviving sleep, restart, or redeploy.
- Implement manual refresh as a local CLI workflow: scrape -> process -> rebuild index/SQLite -> test -> rebuild Docker image -> redeploy.
- Do not offer a misleading Render refresh button that writes non-persistent data.
- Use bounded chunk counts, retrieval count, prompt size, request timeouts, and concurrency.
- If the memory budget is exceeded, reduce worker count, lazy-load embedding model, or use a documented fallback. Do not hide memory errors.

## Data pipeline

Copilot scraper output:
`data/processed/documents.jsonl`

One JSON object per line. Exact contract is in `docs/DATA_SCHEMA.md`.

Claude's ingestion/index builder consumes that file and creates:
- `data/index/faiss.index`
- `data/index/chunks.jsonl`
- `data/index/index_manifest.json`
- `data/index/metadata.sqlite3`

The runtime must validate manifest compatibility before loading the index.

## Retrieval rebuild checkpoint

As of 2026-11-20, the latest verified source JSONL contains 40 DarGlobal documents:
39 webpages and 1 press release, all in English. The rebuilt live index contains
182 chunks and 182 normalized 384-dimensional FAISS vectors. The metadata SQLite
database and BM25 corpus contain 40 documents and 182 chunks. No Wasalt source
documents are included because compliant access currently returns HTTP 403.

## Phase 6.1 evaluation and safety checkpoint

Company attribution is calculated only over non-empty retrievals. The denominator
is the number of non-out-of-scope, non-empty evaluated queries. Every returned
DarGlobal or Wasalt document must match the requested scope; a Both scope accepts
both indexed company prefixes.

Chat safety does not rely on the retrieval empty rate. Explicit confidential,
private, credential, password and ownership-identification requests are refused
before retrieval or generation. Answers with no valid citation marker from the
retrieved context are refused as insufficient evidence, and unknown [S...] markers
are never converted into sources.

Arabic evaluation is limited to Arabic questions retrieving English evidence.
It does not indicate Arabic-source coverage: the 40-document dataset contains no
Arabic source documents.

## API contract

Canonical request and response are in `docs/API_CONTRACT.md`. Frontend and backend must follow it exactly. Main endpoint:
`POST /api/chat`

Other endpoints:
- `GET /health`
- `GET /api/sources`
- `GET /api/stats`
- `GET /docs`
- `GET /openapi.json`

FastAPI should serve the static frontend from the same origin to avoid CORS complexity.

## Code standards

- Human-readable, maintainable, simple code.
- Meaningful names, small functions, clear module boundaries.
- Type hints for public functions and API schemas.
- Short, useful error messages.
- No unnecessary comments, banners, decorative separators, redundant docstrings, or generated explanatory messages in source code.
- Comments only when they clarify non-obvious logic.
- No fake data or placeholder implementations in production paths.
- No secrets in source, Docker layers, frontend, logs, or Git.
- No broad exception swallowing.
- Do not add a dependency unless it is necessary.
- Pin compatible dependency versions once tested.
- Keep code compatible with Python 3.10.11.
- Do not use Python 3.11-only or 3.12+ syntax.
- Do not use Python 3.14 for this project.
- Use pathlib, logging, configuration via environment variables, and deterministic processing.
- Never claim tests passed unless they were executed and passed.

## Continuation protocol

At every meaningful checkpoint, update `BUILD_STATUS.md` with:
- completed tasks
- files created/changed
- commands actually run
- tests passed/failed/not run
- current blockers
- exact next action
- last completed phase

Before continuing after context loss or a new chat:
1. Read `PROJECT_MEMORY.md`.
2. Read `BUILD_STATUS.md`.
3. Read the relevant contracts.
4. Inspect existing files.
5. Continue from the first incomplete task.
6. Never regenerate completed files without a demonstrated reason.

## Deadline priority

P0: scrape useful official pages, build a small clean knowledge base, working bilingual grounded chat, citations, Docker, Render URL.
P1: PDF extraction, scanned-PDF OCR fallback, manual refresh CLI, stats/sources endpoints, tests.
P2: visual polish, optional extras.

Do not sacrifice a working deployable P0 system to chase P2 features.
