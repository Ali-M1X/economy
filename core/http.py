"""Shared HTTP session with retries, timeouts and a descriptive User-Agent."""

from __future__ import annotations

from functools import lru_cache

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.settings import get_settings


class SourceError(RuntimeError):
    """A data source failed (network, HTTP status, or unparseable payload)."""

    def __init__(self, source: str, message: str):
        super().__init__(f"[{source}] {message}")
        self.source = source


@lru_cache(maxsize=1)
def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=1,  # a refused/blocked host should fail fast; retries are for 429/5xx
        read=1,
        other=0,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers["User-Agent"] = get_settings().user_agent
    return s


def get(source: str, url: str, *, params: dict | None = None, headers: dict | None = None,
        timeout: float | None = None) -> requests.Response:
    try:
        resp = session().get(url, params=params, headers=headers, timeout=timeout or get_settings().http_timeout)
    except requests.RequestException as exc:
        raise SourceError(source, f"request failed: {exc.__class__.__name__}: {exc}") from exc
    if resp.status_code != 200:
        raise SourceError(source, f"HTTP {resp.status_code} for {resp.url}: {resp.text[:200]!r}")
    return resp


def get_json(source: str, url: str, **kw):
    resp = get(source, url, **kw)
    try:
        return resp.json()
    except ValueError as exc:
        raise SourceError(source, f"invalid JSON from {resp.url}: {resp.text[:200]!r}") from exc
