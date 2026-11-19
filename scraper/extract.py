import hashlib
import re
from datetime import datetime, timezone
from html import unescape
from typing import Iterable

from bs4 import BeautifulSoup


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if arabic_count == 0 and latin_count == 0:
        return "unknown"
    if arabic_count > latin_count:
        return "ar"
    if latin_count > 0:
        return "en"
    return "unknown"


def normalized_content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_document_id(company: str, source_url: str, suffix: str = "") -> str:
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    base = f"{company}_{digest}"
    return f"{base}_{suffix}" if suffix else base


def published_at_from_soup(soup: BeautifulSoup) -> str | None:
    for attrs in (
        {"property": "article:published_time"},
        {"name": "date"},
        {"name": "pubdate"},
        {"name": "publish-date"},
        {"name": "article:published_time"},
    ):
        element = soup.find("meta", attrs=attrs)
        if element and element.get("content"):
            value = element["content"].strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}", value):
                return value
    for element in soup.find_all(["time"]):
        value = element.get("datetime") or element.get_text(" ", strip=True)
        if value and re.match(r"^\d{4}-\d{2}-\d{2}", value):
            return value.strip()
    return None


def extract_category(url: str) -> str:
    lower = url.lower()
    if "/project" in lower:
        return "project"
    if "/press" in lower or "/news" in lower or "/blog" in lower:
        return "news"
    if "/investor" in lower or "/report" in lower:
        return "report"
    if "/corporate" in lower or "/about" in lower or "/company" in lower:
        return "company"
    if "/leadership" in lower or "/team" in lower:
        return "leadership"
    if "/faq" in lower:
        return "faq"
    if "/service" in lower:
        return "service"
    if "/policy" in lower or "/privacy" in lower or "/terms" in lower:
        return "policy"
    return "other"


def extract_html(response_text: str, canonical_url: str) -> dict:
    soup = BeautifulSoup(response_text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    for selector in (
        "[class*='cookie']",
        "[id*='cookie']",
        "[class*='newsletter']",
        "[class*='breadcrumb']",
    ):
        for element in soup.select(selector):
            element.decompose()

    title_tag = soup.find("title")
    title = unescape(title_tag.get_text(" ", strip=True)) if title_tag else ""
    if not title:
        heading = soup.find(["h1", "h2"])
        title = heading.get_text(" ", strip=True) if heading else canonical_url

    parts: list[str] = []
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = unescape(element.get_text(" ", strip=True))
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in parts:
            if element.name in ("h1", "h2", "h3", "h4"):
                parts.append(f"\n## {text}\n")
            else:
                parts.append(text)
    content = "\n".join(parts)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    return {
        "title": re.sub(r"\s+", " ", title).strip(),
        "content": content,
        "published_at": published_at_from_soup(soup),
    }
