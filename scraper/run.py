import argparse
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from .config import (
    CACHE_PATH,
    DEFAULT_SEEDS,
    MAX_PAGES_PER_DOMAIN,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_BYTES,
    OUTPUT_PATH,
    REPORT_PATH,
)
from .crawl import Crawler
from .extract import utc_now


def validate_document(document: dict) -> list[str]:
    required = (
        "document_id",
        "company",
        "title",
        "content",
        "source_url",
        "content_type",
        "language",
        "collected_at",
        "published_at",
        "file_hash",
    )
    errors = [field for field in required if field not in document]
    if document.get("company") not in ("darglobal", "wasalt"):
        errors.append("company")
    if document.get("content_type") not in ("webpage", "pdf", "press_release"):
        errors.append("content_type")
    if document.get("language") not in ("en", "ar", "unknown"):
        errors.append("language")
    if len(document.get("file_hash", "")) != 64:
        errors.append("file_hash")
    return errors


def write_jsonl(documents: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_documents = []
    for document in documents:
        errors = validate_document(document)
        if errors:
            raise ValueError(f"Invalid document {document.get('document_id')}: {errors}")
        valid_documents.append(document)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in valid_documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def write_report(crawler: Crawler, collection_time: str, report_path: Path) -> dict:
    report = {
        "collection_time": collection_time,
        "success_count": len(crawler.successes),
        "document_count": len(crawler.documents),
        "pdf_count": crawler.pdf_count,
        "failure_count": len(crawler.failures),
        "skipped_count": len(crawler.skipped),
        "total_bytes": crawler.total_bytes,
        "pages_by_domain": crawler.pages_by_domain,
        "pages_by_company": crawler.pages_by_company,
        "documents_by_company": {
            company: sum(1 for doc in crawler.documents if doc.get("company") == company)
            for company in ("darglobal", "wasalt")
        },
        "successes_by_company": {
            company: sum(1 for url in crawler.successes
                         if (urlsplit(url).hostname or "").lower() in (
                             {"darglobal.co.uk", "www.darglobal.co.uk"}
                             if company == "darglobal"
                             else {"wasalt.sa", "www.wasalt.sa", "blog.wasalt.sa"}
                         ))
            for company in ("darglobal", "wasalt")
        },
        "successes": crawler.successes,
        "failures": crawler.failures,
        "skipped": crawler.skipped,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl bounded official EstateIQ sources")
    parser.add_argument("--max-pages-per-domain", type=int, default=MAX_PAGES_PER_DOMAIN)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL_BYTES)
    parser.add_argument("--max-pdf-bytes", type=int, default=MAX_PDF_BYTES)
    parser.add_argument("--max-pdf-pages", type=int, default=MAX_PDF_PAGES)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    crawler = Crawler(
        max_pages_per_domain=args.max_pages_per_domain,
        max_depth=args.max_depth,
        max_total_bytes=args.max_total_bytes,
        max_pdf_bytes=args.max_pdf_bytes,
        max_pdf_pages=args.max_pdf_pages,
        delay_seconds=args.delay_seconds,
        refresh=args.refresh,
        dry_run=args.dry_run,
    )
    collection_time = utc_now()
    documents = crawler.crawl(DEFAULT_SEEDS)
    if not args.dry_run:
        write_jsonl(documents, OUTPUT_PATH)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(crawler.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_report(crawler, collection_time, REPORT_PATH)
    else:
        print(json.dumps({
            "dry_run": True,
            "success_count": len(crawler.successes),
            "document_count": len(documents),
            "pdf_count": crawler.pdf_count,
            "failure_count": len(crawler.failures),
            "skipped_count": len(crawler.skipped),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
