# Project Specification

## Objective
Deliver EstateIQ, a publicly accessible, source-grounded AI assistant answering questions about DarGlobal and Wasalt using their publicly accessible official information.

## Users
Recruitment evaluators and general users exploring the two companies and their public projects/services.

## Required capabilities
1. Crawl relevant public official pages from approved domains.
2. Extract page text, headings, links, publication dates when available, and PDF content.
3. Use OCR only when a PDF is scanned and normal text extraction yields insufficient text.
4. Save normalized source documents to JSONL.
5. Build chunks, multilingual embeddings, FAISS index, BM25 corpus and SQLite metadata.
6. Classify questions by company and topic.
7. Perform hybrid retrieval with RRF.
8. Generate evidence-grounded responses through OpenRouter.
9. Show source citations for factual claims.
10. Support English and Arabic questions.
11. Serve a responsive frontend and API from one FastAPI application.
12. Build and run as a Docker container.
13. Deploy on Render Free.

## Explicit non-goals
- No property purchase, booking, payment, brokerage, or account functions.
- No private user listing scraping.
- No login bypass or anti-bot bypass.
- No generic uncited answers presented as company facts.
- No GraphRAG, multi-agent orchestration, fine-tuning, or complex reranking under this deadline.
- No runtime dependency on persistent Render local writes.

## User experience
- Premium, minimal real-estate intelligence design.
- Navy/dark background, white text, subtle gold accents.
- Responsive desktop/mobile.
- Company scope selector: DarGlobal / Wasalt / Both.
- English/Arabic language selector or automatic language behaviour.
- Suggested example questions.
- Clear loading, error, empty, and unknown-answer states.
- Sources displayed below the answer with title and URL.
- No fabricated project imagery or unsupported facts.

## Language rules
- English query -> English response.
- Arabic query -> Arabic response followed by a concise English summary.
- Mixed query -> natural mixed response; add English summary if response is mainly Arabic.
- UI can remain English; RTL direction should be applied to Arabic answer text.

## P0 acceptance
- Scraper can collect a bounded, useful set of official pages.
- JSONL conforms to DATA_SCHEMA.
- Index build works from JSONL.
- API health check succeeds.
- At least one English and one Arabic question retrieve relevant evidence.
- Chat response includes real source URLs.
- Unknown question is declined safely.
- Frontend can send a question and render answer/citations.
- Docker build and local container test succeed.
- Render public URL works after deployment.

## P1 acceptance
- PDF text extraction.
- OCR fallback with limits.
- Manual local refresh command.
- Sources and stats endpoints.
- Test suite for scraper, retrieval and API.

## Runtime constraints
Render Free has a 512 MB RAM service limit and ephemeral filesystem. Minimize dependencies, use ONNX embeddings, bounded concurrency, short contexts, one worker unless measurements justify otherwise, and package the built index/database into the image.
