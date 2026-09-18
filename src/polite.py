"""Fetching that follows the rules a site sets for automated visitors.

Three things happen here, and they matter independently:

1. robots.txt is checked before any page on a host is requested. If the file
   disallows the path for our user agent, the page is skipped and the reason
   is written to a log rather than silently dropped.
2. Requests to the same host are spaced out. The study calls for one request
   every two to three seconds. Some Nevada district sites publish their own
   Crawl-delay in robots.txt, and this module waits for whichever delay is
   longer.
3. Every page that is fetched is written to disk. A second run reads the saved
   copy instead of hitting the site again, so developing the parser costs the
   sites nothing.

The user agent names the project and gives a contact address, so a site
administrator who sees the traffic can tell what it is and reach a person.
"""

from __future__ import annotations

import hashlib
import random
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import CACHE_DIR, CONFIG

SCRAPE_CONFIG = CONFIG["scraping"]

# The YAML block scalar preserves newlines, so collapse the whitespace into a
# single line before it goes into an HTTP header.
USER_AGENT = " ".join(SCRAPE_CONFIG["user_agent"].split())


@dataclass
class FetchResult:
    """What happened when we asked for one URL.

    Keeping the outcome in a small object rather than returning a bare string
    means a caller can tell the difference between "the page said nothing
    useful" and "we were not allowed to look". That distinction is what the
    scrape log needs to record.
    """

    url: str
    status: str          # one of: ok, cached, disallowed, error
    reason: str = ""     # human readable explanation, used in the log
    html: str = ""
    from_cache: bool = False


def _cache_path(url: str) -> Path:
    """Map a URL to a stable filename inside the cache directory.

    A hash is used because URLs contain characters Windows will not accept in
    a filename, and because some query strings are longer than the filename
    length limit. The host is kept as a readable prefix so that the cache
    folder can be browsed by eye.
    """
    host = urlparse(url).netloc.replace(":", "_")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{host}__{digest}.html"


class PoliteFetcher:
    """A requests session that respects robots.txt, rate limits, and caches."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        # One robots parser per host, fetched the first time that host is
        # touched and then reused for the rest of the run.
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

        # Timestamp of the most recent request to each host, used for spacing.
        self._last_request_at: dict[str, float] = {}

        # Every robots decision, so the pipeline can write an audit log.
        self.robots_decisions: list[dict] = []

    # -- robots.txt ---------------------------------------------------------

    def _robots_for(self, url: str):
        """Return the robots parser for the host of this URL, fetching once.

        If robots.txt is missing or unreadable, None is returned and the caller
        treats the path as allowed. That matches the convention in the robots
        standard, where absence of the file is not a prohibition. A 404 is a
        different thing from a file that says Disallow.
        """
        parsed = urlparse(url)
        host = parsed.netloc
        if host in self._robots:
            return self._robots[host]

        robots_url = f"{parsed.scheme}://{host}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            response = self.session.get(
                robots_url, timeout=SCRAPE_CONFIG["request_timeout_seconds"]
            )
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                parser = None
        except requests.RequestException:
            parser = None

        self._robots[host] = parser
        return parser

    def is_allowed(self, url: str) -> tuple[bool, str]:
        """Check one URL against robots.txt and explain the answer."""
        parser = self._robots_for(url)
        if parser is None:
            return True, "no robots.txt published, treated as allowed"
        if parser.can_fetch(USER_AGENT, url):
            return True, "allowed by robots.txt"
        return False, "disallowed by robots.txt"

    def crawl_delay_for(self, url: str) -> float:
        """Seconds to wait before the next request to this host.

        The configured delay is a floor. If robots.txt asks for longer, the
        longer value wins. Random jitter inside the configured band keeps the
        request pattern from looking like a metronome.
        """
        floor = random.uniform(
            SCRAPE_CONFIG["min_delay_seconds"], SCRAPE_CONFIG["max_delay_seconds"]
        )
        parser = self._robots_for(url)
        if parser is None:
            return floor
        published = parser.crawl_delay(USER_AGENT)
        if published is None:
            return floor
        return max(floor, float(published))

    # -- fetching -----------------------------------------------------------

    def _wait_for_host(self, url: str) -> None:
        """Sleep long enough that this host is not hit too quickly."""
        host = urlparse(url).netloc
        delay = self.crawl_delay_for(url)
        last = self._last_request_at.get(host)
        if last is not None:
            elapsed = time.monotonic() - last
            if elapsed < delay:
                time.sleep(delay - elapsed)
        self._last_request_at[host] = time.monotonic()

    def get(self, url: str, use_cache: bool = True) -> FetchResult:
        """Fetch one page, or explain why it was not fetched."""
        cache_file = _cache_path(url)

        # The cache is checked before robots.txt on purpose. A cached page has
        # already been fetched under the rules that applied at the time, and
        # reading a local file is not a request to the site at all.
        if use_cache and cache_file.exists():
            return FetchResult(
                url=url,
                status="cached",
                reason="served from local cache",
                html=cache_file.read_text(encoding="utf-8", errors="replace"),
                from_cache=True,
            )

        allowed, reason = self.is_allowed(url)
        self.robots_decisions.append({"url": url, "allowed": allowed, "reason": reason})
        if not allowed:
            return FetchResult(url=url, status="disallowed", reason=reason)

        self._wait_for_host(url)
        try:
            response = self.session.get(
                url, timeout=SCRAPE_CONFIG["request_timeout_seconds"]
            )
        except requests.RequestException as error:
            return FetchResult(
                url=url, status="error", reason=f"{type(error).__name__}: {error}"
            )

        if response.status_code != 200:
            return FetchResult(url=url, status="error", reason=f"HTTP {response.status_code}")

        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            return FetchResult(url=url, status="error", reason=f"not HTML: {content_type}")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(response.text, encoding="utf-8")
        return FetchResult(url=url, status="ok", reason="fetched", html=response.text)


def polite_json_get(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    last_call: list[float] | None = None,
):
    """Fetch JSON from an API with the same delay the scraper uses.

    The data APIs are public and meant to be queried, but there is no reason to
    hammer them. last_call is a one element list used as a mutable timestamp
    that the caller owns, which lets a loop share one clock without needing a
    global variable.
    """
    if last_call is not None and last_call[0] > 0:
        delay = random.uniform(
            SCRAPE_CONFIG["min_delay_seconds"], SCRAPE_CONFIG["max_delay_seconds"]
        )
        elapsed = time.monotonic() - last_call[0]
        if elapsed < delay:
            time.sleep(delay - elapsed)

    try:
        response = session.get(
            url, params=params, timeout=SCRAPE_CONFIG["request_timeout_seconds"]
        )
    except requests.RequestException:
        return None
    finally:
        if last_call is not None:
            last_call[0] = time.monotonic()

    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None
