import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from .config import (
    ALLOWED_DOMAINS,
    CACHE_PATH,
    DELAY_SECONDS,
    MAX_DEPTH,
    MAX_PAGES_PER_DOMAIN,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MAX_TOTAL_BYTES,
)
from .extract import (
    detect_language,
    deterministic_document_id,
    extract_category,
    extract_html,
    normalized_content_hash,
    utc_now,
)
from .http import HttpClient
from .pdf import extract_pdf_with_optional_ocr
from .url_utils import canonicalize_url, is_allowed_url


DISALLOWED_HOSTS = {"auction.wasalt.sa"}
COMPANY_HOSTS = {
    "darglobal": {"darglobal.co.uk", "www.darglobal.co.uk"},
    "wasalt": {"wasalt.sa", "www.wasalt.sa", "blog.wasalt.sa"},
}


def company_for_url(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    for company, hosts in COMPANY_HOSTS.items():
        if host in hosts:
            return company
    return None


class Crawler:
    def __init__(
        self,
        max_pages_per_domain: int = MAX_PAGES_PER_DOMAIN,
        max_depth: int = MAX_DEPTH,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_pdf_bytes: int = MAX_PDF_BYTES,
        max_pdf_pages: int = MAX_PDF_PAGES,
        delay_seconds: float = DELAY_SECONDS,
        refresh: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.max_pages_per_domain = max_pages_per_domain
        self.max_depth = max_depth
        self.max_total_bytes = max_total_bytes
        self.max_pdf_bytes = max_pdf_bytes
        self.max_pdf_pages = max_pdf_pages
        self.refresh = refresh
        self.dry_run = dry_run
        self.http = HttpClient(delay_seconds=delay_seconds)
        self.log = logging.getLogger(__name__)
        self.documents: list[dict] = []
        self.successes: list[str] = []
        self.failures: list[dict] = []
        self.skipped: list[dict] = []
        self.pdf_count = 0
        self.total_bytes = 0
        self.pages_by_domain: dict[str, int] = {}
        self.pages_by_company: dict[str, int] = {"darglobal": 0, "wasalt": 0}
        self.visited: set[str] = set()
        # A matching page from another company must never suppress this source.
        self.content_hashes: set[tuple[str, str]] = set()
        self.cache = self.load_cache()

    def load_cache(self) -> dict:
        if not CACHE_PATH.exists():
            return {}
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.log.warning("Could not read crawl cache: %s", exc)
            return {}

    def record_skip(self, url: str, reason: str) -> None:
        self.skipped.append({"url": url, "reason": reason})

    def can_count_page(self, url: str, company: str | None = None) -> bool:
        company = company or company_for_url(url)
        if company not in self.pages_by_company:
            return False
        if self.pages_by_company[company] >= self.max_pages_per_domain:
            return False
        self.pages_by_company[company] += 1
        host = urlsplit(url).hostname or ""
        self.pages_by_domain[host] = self.pages_by_domain.get(host, 0) + 1
        return True

    def discover_sitemap_urls(self, company: str) -> list[str]:
        hosts = sorted(COMPANY_HOSTS.get(company, set()))
        found: list[str] = []
        seen_sitemaps: set[str] = set()
        pending = deque(f"https://{host}/sitemap.xml" for host in hosts)

        # Sitemap indexes are followed only within the approved company hosts.
        while pending and len(seen_sitemaps) < 12:
            sitemap_url = pending.popleft()
            sitemap_url = canonicalize_url(sitemap_url)
            if not sitemap_url or sitemap_url in seen_sitemaps:
                continue
            if not is_allowed_url(sitemap_url, ALLOWED_DOMAINS):
                continue
            seen_sitemaps.add(sitemap_url)
            if not self.http.allowed_by_robots(sitemap_url):
                self.record_skip(
                    sitemap_url,
                    "sitemap disallowed or robots.txt could not be verified"
                )
                continue
            try:
                response = self.http.get(sitemap_url)
                if len(response.content) > 2 * 1024 * 1024:
                    self.record_skip(sitemap_url, "sitemap exceeds size limit")
                    continue
                root = ET.fromstring(response.content)
                root_type = root.tag.rsplit("}", 1)[-1].lower()
                for location in root.iter():
                    if location.tag.rsplit("}", 1)[-1].lower() != "loc" or not location.text:
                        continue
                    candidate = canonicalize_url(location.text.strip())
                    if not candidate:
                        continue
                    candidate_company = company_for_url(candidate)
                    if candidate_company != company:
                        continue
                    if not is_allowed_url(candidate, ALLOWED_DOMAINS):
                        continue
                    if root_type == "sitemapindex" or candidate.lower().endswith(".xml"):
                        if candidate not in seen_sitemaps:
                            pending.append(candidate)
                    else:
                        found.append(candidate)
            except requests.RequestException as exc:
                self.skipped.append({
                    "url": sitemap_url,
                    "reason": f"sitemap unavailable: {exc}",
                })
            except ET.ParseError as exc:
                self.skipped.append({
                    "url": sitemap_url,
                    "reason": f"malformed sitemap XML: {exc}",
                })
        return list(dict.fromkeys(found))

    def crawl(self, seeds: dict[str, list[str]]) -> list[dict]:
        queue: deque[tuple[str, int, str]] = deque()
        for declared_company, urls in seeds.items():
            for seed in urls:
                canonical = canonicalize_url(seed)
                actual_company = company_for_url(canonical or "")
                if canonical and actual_company == declared_company:
                    queue.append((canonical, 0, actual_company))
                elif canonical:
                    self.record_skip(canonical, "seed company does not match approved host")

            for sitemap_url in self.discover_sitemap_urls(declared_company):
                queue.append((sitemap_url, 1, declared_company))

        while queue:
            url, depth, declared_company = queue.popleft()
            canonical = canonicalize_url(url)
            if not canonical or canonical in self.visited:
                continue

            company = company_for_url(canonical)
            if company is None or company != declared_company:
                self.record_skip(canonical, "company/host mismatch or unapproved host")
                self.visited.add(canonical)
                continue

            host = (urlsplit(canonical).hostname or "").lower()
            if host in DISALLOWED_HOSTS:
                self.record_skip(canonical, "auction host excluded by project crawl policy")
                self.visited.add(canonical)
                continue

            if not is_allowed_url(canonical, ALLOWED_DOMAINS):
                self.record_skip(canonical, "domain not allowlisted")
                self.visited.add(canonical)
                continue
            if depth > self.max_depth:
                self.record_skip(canonical, "max crawl depth exceeded")
                self.visited.add(canonical)
                continue
            if split_url_has_query(canonical) and not canonical.lower().endswith(".pdf"):
                self.record_skip(canonical, "query-string URL excluded")
                self.visited.add(canonical)
                continue
            if not self.http.allowed_by_robots(canonical):
                reason = self.http.robots_status(canonical)
                self.record_skip(
                    canonical,
                    "disallowed by robots.txt" if not reason
                    else f"robots.txt could not be verified: {reason}",
                )
                self.visited.add(canonical)
                continue
            if not self.can_count_page(canonical, company):
                self.record_skip(canonical, f"{company} page limit reached")
                self.visited.add(canonical)
                continue

            if canonical.lower().endswith(".pdf"):
                self.process_pdf(canonical, company)
                self.visited.add(canonical)
                continue

            # Cache entries contain only hashes/timestamps, not document bodies.
            # Skipping cached pages here caused the JSONL export to lose records
            # because each run rewrites the output file. Always refetch pages.
            try:
                response = self.http.get(canonical)
                self.visited.add(canonical)

                final_url = canonicalize_url(response.url)
                if not final_url or company_for_url(final_url) != company:
                    self.record_skip(canonical, "redirect left approved company host")
                    continue
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/pdf" in content_type or canonical.lower().endswith(".pdf"):
                    self.process_pdf(canonical, company)
                    continue
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    self.record_skip(canonical, f"unsupported content type: {content_type}")
                    continue
                if len(response.content) > self.max_total_bytes - self.total_bytes:
                    self.record_skip(canonical, "total byte limit reached")
                    break
                self.total_bytes += len(response.content)

                extracted = extract_html(response.text, canonical)
                content = extracted["content"]
                content_hash = normalized_content_hash(content)
                hash_key = (company, content_hash)
                if len(content.strip()) < 40:
                    self.record_skip(canonical, "insufficient extracted text")
                elif hash_key in self.content_hashes:
                    self.record_skip(canonical, "duplicate content within company")
                else:
                    self.content_hashes.add(hash_key)
                    document_id = deterministic_document_id(company, canonical)
                    category = extract_category(canonical)
                    document = {
                        "document_id": document_id,
                        "company": company,
                        "title": extracted["title"],
                        "content": content,
                        "source_url": canonical,
                        "content_type": "press_release" if category == "news" else "webpage",
                        "language": detect_language(content),
                        "collected_at": utc_now(),
                        "published_at": extracted["published_at"],
                        "file_hash": content_hash,
                        "category": category,
                        "source_domain": urlsplit(canonical).hostname,
                        "extraction_method": "html",
                        "metadata": {"final_url": response.url},
                    }
                    self.documents.append(document)
                    self.successes.append(canonical)
                    self.cache[canonical] = {
                        "content_hash": content_hash,
                        "collected_at": document["collected_at"],
                    }

                soup = BeautifulSoup(response.text, "html.parser")
                for anchor in soup.find_all("a", href=True):
                    linked = canonicalize_url(anchor["href"], canonical)
                    if not linked:
                        continue
                    if company_for_url(linked) != company:
                        continue
                    if not is_allowed_url(linked, ALLOWED_DOMAINS):
                        continue
                    if (urlsplit(linked).hostname or "").lower() in DISALLOWED_HOSTS:
                        continue
                    if depth + 1 <= self.max_depth:
                        if split_url_has_query(linked) and not linked.lower().endswith(".pdf"):
                            continue
                        queue.append((linked, depth + 1, company))

            except requests.Timeout as exc:
                self.failures.append({"url": canonical, "error": f"timeout: {exc}"})
                self.visited.add(canonical)
            except requests.RequestException as exc:
                self.failures.append({"url": canonical, "error": str(exc)})
                self.visited.add(canonical)
            except (UnicodeDecodeError, ValueError) as exc:
                self.failures.append({"url": canonical, "error": f"malformed response: {exc}"})
                self.visited.add(canonical)

        return self.documents

    def process_pdf(self, url: str, company: str) -> None:
        if not self.http.allowed_by_robots(url):
            reason = self.http.robots_status(url)
            self.record_skip(
                url,
                "disallowed by robots.txt" if not reason
                else f"robots.txt could not be verified: {reason}",
            )
            return
        try:
            response = self.http.get(url)
            if len(response.content) > self.max_pdf_bytes:
                self.record_skip(url, "PDF exceeds size limit")
                return
            if len(response.content) > self.max_total_bytes - self.total_bytes:
                self.record_skip(url, "total byte limit reached")
                return
            self.total_bytes += len(response.content)
            pages = extract_pdf_with_optional_ocr(response.content, self.max_pdf_pages)
            parent_id = deterministic_document_id(company, url)
            parent_title = urlsplit(url).path.rsplit("/", 1)[-1] or url
            combined = "\n".join(page["text"] for page in pages if page["text"]).strip()
            if len(combined) < 40:
                self.record_skip(url, "PDF text insufficient after bounded extraction")
                return
            combined_hash = normalized_content_hash(combined)
            hash_key = (company, combined_hash)
            if hash_key in self.content_hashes:
                self.record_skip(url, "duplicate PDF content within company")
                return
            self.content_hashes.add(hash_key)
            collected_at = utc_now()
            for page in pages:
                text = page["text"].strip()
                if not text:
                    continue
                document_id = deterministic_document_id(company, url, f"page{page['page_number']}")
                self.documents.append({
                    "document_id": document_id,
                    "company": company,
                    "title": parent_title,
                    "content": text,
                    "source_url": url,
                    "content_type": "pdf",
                    "language": detect_language(text),
                    "collected_at": collected_at,
                    "published_at": None,
                    "file_hash": normalized_content_hash(text),
                    "category": extract_category(url),
                    "page_number": page["page_number"],
                    "parent_document_id": parent_id,
                    "source_domain": urlsplit(url).hostname,
                    "extraction_method": page["method"],
                    "metadata": {"pdf_page_count": len(pages)},
                })
            self.pdf_count += 1
            self.successes.append(url)
            self.cache[url] = {
                "content_hash": combined_hash,
                "collected_at": collected_at,
            }
        except requests.Timeout as exc:
            self.failures.append({"url": url, "error": f"timeout: {exc}"})
        except requests.RequestException as exc:
            self.failures.append({"url": url, "error": str(exc)})
        except (ValueError, RuntimeError) as exc:
            self.failures.append({"url": url, "error": f"PDF extraction failed: {exc}"})


def split_url_has_query(url: str) -> bool:
    return bool(urlsplit(url).query)
