import json
import tempfile
from pathlib import Path

import fitz
import pytest
import requests

from scraper.extract import (
    detect_language,
    extract_html,
    normalized_content_hash,
)
from scraper.pdf import extract_pdf_text
from scraper.run import validate_document, write_jsonl
from scraper.url_utils import canonicalize_url, is_allowed_url


def test_canonicalize_url_and_allowlist():
    url = canonicalize_url(
        "https://DARGLOBAL.CO.UK/projects/aida/?utm_source=test#details",
        "https://darglobal.co.uk/",
    )
    assert url == "https://darglobal.co.uk/projects/aida"
    assert is_allowed_url(url, {"darglobal.co.uk"})
    assert not is_allowed_url("https://example.com/projects/aida", {"darglobal.co.uk"})
    assert canonicalize_url("https://darglobal.co.uk/login") is None


def test_html_extraction_preserves_text_and_language():
    html = """
    <html><head><title>Official Project</title></head><body>
    <nav>Menu</nav><main><h1>Official Project</h1>
    <p>This is public project information with sufficient extracted text.</p>
    <ul><li>London</li></ul></main><footer>Footer</footer>
    </body></html>
    """
    extracted = extract_html(html, "https://darglobal.co.uk/projects/official")
    assert extracted["title"] == "Official Project"
    assert "public project information" in extracted["content"]
    assert "Menu" not in extracted["content"]
    assert detect_language(extracted["content"]) == "en"


def test_arabic_language_detection():
    assert detect_language("معلومات عامة عن شركة وصل") == "ar"


def test_pdf_text_extraction():
    pdf_bytes = b""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.pdf"
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Public official PDF project information")
        document.save(path)
        document.close()
        pdf_bytes = path.read_bytes()
    pages = extract_pdf_text(pdf_bytes, max_pages=40)
    assert len(pages) == 1
    assert "official PDF" in pages[0]["text"]


def test_content_hash_deduplicates_normalized_text():
    first = normalized_content_hash("Public   official\ntext")
    second = normalized_content_hash("Public official text")
    assert first == second


def test_jsonl_validation_and_output():
    document = {
        "document_id": "darglobal_test",
        "company": "darglobal",
        "title": "Test",
        "content": "Public official test content",
        "source_url": "https://darglobal.co.uk/test",
        "content_type": "webpage",
        "language": "en",
        "collected_at": "2026-11-18T12:00:00Z",
        "published_at": None,
        "file_hash": "a" * 64,
    }
    assert validate_document(document) == []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "documents.jsonl"
        write_jsonl([document], path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["document_id"] == "darglobal_test"


def test_malformed_timeout_error_types():
    with pytest.raises(requests.Timeout):
        raise requests.Timeout("timed out")
    with pytest.raises(ValueError):
        json.loads("{malformed")
