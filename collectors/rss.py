"""Minimal RSS/Atom reader (stdlib only)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

from core.http import SourceError, get
from core.timeutil import UTC


def parse_feed(xml_text: str, source: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError as exc:
        raise SourceError(source, f"bad XML: {exc}") from exc
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
    return parse_feed(get(source, url).text, source)
