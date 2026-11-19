from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "crawl_report.json"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "crawl_manifest.json"

DEFAULT_SEEDS = {
    "darglobal": [
        "https://darglobal.co.uk/",
        "https://darglobal.co.uk/projects",
        "https://darglobal.co.uk/press",
        "https://darglobal.co.uk/investor",
    ],
    "wasalt": [
        "https://wasalt.sa/en",
        "https://wasalt.sa/en/corporate",
        "https://wasalt.sa/en/s/privacy-policy",
        "https://wasalt.sa/en/s/terms-and-conditions",
        "https://blog.wasalt.sa/",
    ],
}

ALLOWED_DOMAINS = {
    "darglobal.co.uk",
    "www.darglobal.co.uk",
    "wasalt.sa",
    "www.wasalt.sa",
    "blog.wasalt.sa",
}

USER_AGENT = "EstateIQBot/1.0 (+public official source research; respectful crawl)"
MAX_PAGES_PER_DOMAIN = 60
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_DEPTH = 2
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
DELAY_SECONDS = 1.0
