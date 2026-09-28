"""Minimal RSS/Atom reader (stdlib only)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

from core.http import SourceError, get
from core.timeutil import UTC


def parse_feed(xml: bytes | str, source: str) -> list[dict]:
    """Parse RSS 2.0 / Atom. Pass the raw response *bytes*: the XML declaration (and any BOM) must decide
    the encoding. `requests.Response.text` falls back to ISO-8859-1 for text/* responses without a
    charset, which turns a UTF-8 BOM into 'ï»¿' and makes the XML unparseable ("invalid token, line 1")."""
    raw = xml.encode("utf-8") if isinstance(xml, str) else xml
    head = raw.lstrip(b"\xef\xbb\xbf \t\r\n")[:200].lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise SourceError(source, "got an HTML page instead of a feed (blocked or moved?)")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise SourceError(source, f"bad XML: {exc}; starts with {raw[:40]!r}") from exc
    items = []
    atom = "{http://www.w3.org/2005/Atom}"
    for it in root.iter("item"):
        items.append({"title": (it.findtext("title") or "").strip(), "url": (it.findtext("link") or "").strip(),
                      "summary": (it.findtext("description") or "").strip(), "published": it.findtext("pubDate")})
    for it in root.iter(f"{atom}entry"):
        link = it.find(f"{atom}link")
        items.append({"title": (it.findtext(f"{atom}title") or "").strip(),
                      "url": link.get("href") if link is not None else "",
                      "summary": (it.findtext(f"{atom}summary") or "").strip(),
                      "published": it.findtext(f"{atom}updated") or it.findtext(f"{atom}published")})
    for i in items:
        i["source"] = source
        i["published_utc"] = _parse_date(i.pop("published"))
    return items


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).astimezone(UTC)
    except (TypeError, ValueError):
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts.to_pydatetime()


def fetch_feed(url: str, source: str) -> list[dict]:
    return parse_feed(get(source, url).content, source)
