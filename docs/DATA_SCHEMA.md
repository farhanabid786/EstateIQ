# Data Schema — Canonical Scraper Output

Copilot must write `data/processed/documents.jsonl`, UTF-8, one JSON object per line.

Required fields:
- `document_id`: stable deterministic string ID
- `company`: `darglobal` or `wasalt`
- `title`: human-readable page/document title
- `content`: extracted plain text
- `source_url`: canonical public source URL
- `content_type`: `webpage`, `pdf`, or `press_release`
- `language`: `en`, `ar`, or `unknown`
- `collected_at`: ISO-8601 UTC timestamp
- `published_at`: ISO-8601 date/time or null
- `file_hash`: SHA-256 of normalized content

Optional fields:
- `category`: project, company, leadership, investor, service, faq, news, policy, report, other
- `description`
- `location`
- `page_number`
- `parent_document_id`
- `source_domain`
- `extraction_method`: html, pdf_text, ocr
- `metadata`: object containing only serializable values

Example:
{
  "document_id": "darglobal_aida_project_page",
  "company": "darglobal",
  "title": "AIDA",
  "content": "Text extracted from the official source...",
  "source_url": "https://official-source.example/project",
  "content_type": "webpage",
  "language": "en",
  "collected_at": "2026-11-18T12:00:00Z",
  "published_at": null,
  "file_hash": "sha256-hex",
  "category": "project",
  "extraction_method": "html",
  "metadata": {}
}

Rules:
- Never use the example as real company data.
- `content` must contain actual extracted text.
- Keep source URL exact and canonical.
- Do not combine unrelated pages into one document.
- Deduplicate by canonical URL and content hash.
- Store a crawl report separately.
- If extraction fails, log the URL and error; do not create a fake empty document.
- Remove scripts, navigation, cookie banners and repetitive boilerplate where possible.
- Respect robots.txt, site terms, rate limits and same-domain boundaries.
- Do not scrape personal data or protected/private content.
- Store PDF page-level text where practical, preserving page number.
