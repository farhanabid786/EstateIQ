from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}

EXCLUDED_PATH_PARTS = (
    "/login",
    "/signin",
    "/sign-in",
    "/account",
    "/search",
    "/calendar",
)


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    if not url:
        return None
    absolute = urljoin(base_url, url.strip()) if base_url else url.strip()
    split = urlsplit(absolute)
    scheme = split.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = split.hostname.lower() if split.hostname else ""
    if not host:
        return None
    port = ""
    if split.port and ((scheme == "http" and split.port != 80) or (scheme == "https" and split.port != 443)):
        port = f":{split.port}"
    path = split.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/")
    lower_path = path.lower()
    if any(part in lower_path for part in EXCLUDED_PATH_PARTS):
        return None
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=False)
        if key.lower() not in TRACKING_PARAMETERS
    ]
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, host + port, path, query, ""))


def is_allowed_url(url: str, allowed_domains: set[str]) -> bool:
    split = urlsplit(url)
    return split.hostname.lower() in allowed_domains if split.hostname else False
