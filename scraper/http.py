import logging
import time
import urllib.robotparser
from typing import Optional
from urllib.parse import urlsplit

import requests

from .config import CONNECT_TIMEOUT, READ_TIMEOUT, USER_AGENT


class RobotsUnavailableError(RuntimeError):
    """Raised when robots.txt could not be checked safely."""


class HttpClient:
    def __init__(self, delay_seconds: float = 1.0) -> None:
        self.delay_seconds = max(1.0, float(delay_seconds))
        self.last_request_at: dict[str, float] = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
        })
        self.log = logging.getLogger(__name__)
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.robots_errors: dict[str, str] = {}

    def load_robots(self, base_url: str) -> urllib.robotparser.RobotFileParser:
        split = urlsplit(base_url)
        origin = f"{split.scheme}://{split.netloc}"
        if origin in self.robots:
            return self.robots[origin]

        parser = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        parser.set_url(robots_url)

        try:
            response = self.session.get(
                robots_url,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 404:
                # A genuine 404 means this host publishes no robots file.
                parser.parse([])
            elif response.status_code in (401, 403):
                self.robots_errors[origin] = f"robots.txt returned HTTP {response.status_code}"
                parser.parse(["User-agent: *", "Disallow: /"])
            elif response.status_code >= 500:
                raise RobotsUnavailableError(
                    f"robots.txt returned HTTP {response.status_code}"
                )
            else:
                response.raise_for_status()
                parser.parse(response.text.splitlines())
        except (requests.RequestException, OSError, RobotsUnavailableError) as exc:
            self.robots_errors[origin] = str(exc)
            self.log.warning("Cannot verify robots.txt for %s: %s", origin, exc)
            # Fail closed: never interpret a network failure as permission.
            parser.parse(["User-agent: *", "Disallow: /"])

        self.robots[origin] = parser
        return parser

    def allowed_by_robots(self, url: str) -> bool:
        parser = self.load_robots(url)
        try:
            return bool(parser.can_fetch(USER_AGENT, url))
        except (AttributeError, ValueError, OSError) as exc:
            self.log.warning("Cannot evaluate robots.txt for %s: %s", url, exc)
            return False

    def robots_status(self, url: str) -> Optional[str]:
        origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        return self.robots_errors.get(origin)

    def get(self, url: str) -> requests.Response:
        host = urlsplit(url).hostname or url
        previous = self.last_request_at.get(host)
        if previous is not None:
            remaining = self.delay_seconds - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)

        response = self.session.get(
            url,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        )
        self.last_request_at[host] = time.monotonic()
        response.raise_for_status()
        return response
