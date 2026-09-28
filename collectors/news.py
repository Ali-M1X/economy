"""High-impact news candidates: official agency feeds + GDELT (free global news index).

Classification (relevance / direction / importance) is done later by Claude; this module only collects.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from collectors.fed import fed_feeds
from collectors.rss import fetch_feed
from core.http import SourceError, get_json

AGENCY_FEEDS = {
    "bls": "https://www.bls.gov/feed/bls_latest.rss",
    "bea": "https://apps.bea.gov/rss/rss.xml",
    "treasury": "https://home.treasury.gov/news/press-releases/feed",
}
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = ('("federal reserve" OR "fed chair" OR powell OR fomc OR "treasury yields" OR inflation OR '
               'bitcoin OR "gold price" OR "bitcoin etf" OR tariff OR sanctions) sourcelang:english')


def item_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title}".encode()).hexdigest()[:20]


def gdelt(timespan: str = "1h", max_records: int = 75) -> list[dict]:
    payload = get_json("gdelt", GDELT, params={"query": GDELT_QUERY, "mode": "artlist", "format": "json",
                                              "maxrecords": max_records, "timespan": timespan, "sort": "datedesc"})
    out = []
    for a in payload.get("articles", []):
        ts = pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        out.append({"source": f"gdelt:{a.get('domain')}", "title": a.get("title", ""), "url": a.get("url", ""),
                    "summary": "", "published_utc": None if pd.isna(ts) else ts.to_pydatetime()})
    return out


def collect() -> tuple[list[dict], dict[str, str]]:
    items, errors = fed_feeds()
    for name, url in AGENCY_FEEDS.items():
        try:
            items.extend(fetch_feed(url, name))
        except SourceError as exc:
            errors[name] = str(exc)
    try:
        items.extend(gdelt())
    except SourceError as exc:
        errors["gdelt"] = str(exc)
    for i in items:
        i["id"] = item_id(i["url"], i["title"])
    return list({i["id"]: i for i in items}.values()), errors
