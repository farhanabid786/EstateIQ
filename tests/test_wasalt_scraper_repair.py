from collections import deque

from scraper.crawl import Crawler, company_for_url
from scraper.url_utils import canonicalize_url, is_allowed_url
from scraper.config import ALLOWED_DOMAINS


def test_wasalt_host_classification():
    assert company_for_url("https://wasalt.sa/en") == "wasalt"
    assert company_for_url("https://www.wasalt.sa/en") == "wasalt"
    assert company_for_url("https://blog.wasalt.sa/") == "wasalt"


def test_lookalike_wasalt_domain_rejected():
    url = canonicalize_url("https://wasalt.sa.evil.example/en")
    assert url is not None
    assert not is_allowed_url(url, ALLOWED_DOMAINS)
    assert company_for_url(url) is None


def test_company_page_budgets_are_independent():
    crawler = Crawler(max_pages_per_domain=1)
    assert crawler.can_count_page("https://darglobal.co.uk/")
    assert not crawler.can_count_page("https://darglobal.co.uk/projects")
    assert crawler.can_count_page("https://wasalt.sa/en")


def test_company_specific_hash_deduplication():
    crawler = Crawler()
    digest = "same-content-hash"
    crawler.content_hashes.add(("darglobal", digest))
    assert ("wasalt", digest) not in crawler.content_hashes


def test_wasalt_seed_host_is_classified():
    assert company_for_url("https://blog.wasalt.sa/article") == "wasalt"


def test_arabic_text_is_preserved_by_extractor():
    from scraper.extract import extract_html
    result = extract_html(
        "<html><head><title>Wasalt</title></head><body><main>"
        "<h1>منصة وصلت</h1><p>خدمات عقارية في المملكة العربية السعودية</p>"
        "</main></body></html>",
        "https://wasalt.sa/en",
    )
    assert "منصة وصلت" in result["content"]
    assert "خدمات عقارية" in result["content"]


def test_company_assignment_comes_from_approved_host():
    assert company_for_url("https://wasalt.sa/en/corporate") == "wasalt"
    assert company_for_url("https://darglobal.co.uk/projects") == "darglobal"


def test_auction_host_is_not_company_approved():
    assert company_for_url("https://auction.wasalt.sa/") is None
