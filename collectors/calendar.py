"""Economic calendar from official free sources.

- FRED `release/dates` (needs FRED_API_KEY) for every tracked release.
- BLS official release calendar (ICS) — keyless, BLS releases only.
- Fed FOMC calendar for rate decisions.
Consensus forecasts are not freely available from an official source; see DATA_GAPS.md.
Stored times are UTC; Tehran time is derived for display.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from functools import lru_cache

import yaml

from collectors import fed, fred
from core.http import SourceError, get
from core.settings import ROOT
from core.timeutil import NEW_YORK, TEHRAN, UTC, et_to_utc, utcnow

BLS_ICS = "https://www.bls.gov/schedule/news_release/bls.ics"


@lru_cache(maxsize=1)
def tracked_releases() -> list[dict]:
    return yaml.safe_load((ROOT / "config" / "releases.yaml").read_text(encoding="utf-8"))["releases"]


def _event(rel: dict, day: date, source: str) -> dict:
    hh, mm = map(int, rel["time_et"].split(":"))
    utc = et_to_utc(datetime(day.year, day.month, day.day, hh, mm))
    return {"event_id": f"{rel['id']}-{day.isoformat()}", "release_id": rel["id"], "name_en": rel["name_en"],
            "name_fa": rel["name_fa"], "scheduled_utc": utc, "scheduled_tehran": utc.astimezone(TEHRAN),
            "importance": rel["importance"], "series_keys": rel["series_keys"], "date_source": source}


def parse_ics(text: str) -> list[dict]:
    """Minimal VEVENT parser: returns SUMMARY and DTSTART (as UTC datetime)."""
    events, cur = [], None
    for line in re.sub(r"\r?\n[ \t]", "", text).splitlines():  # unfold continuation lines
        if line == "BEGIN:VEVENT":
            cur = {}
        elif line == "END:VEVENT" and cur is not None:
            events.append(cur)
            cur = None
        elif cur is not None and ":" in line:
            k, v = line.split(":", 1)
            name, *params = k.split(";")
            if name == "SUMMARY":
                cur["summary"] = v.strip()
            elif name == "DTSTART":
                tzid = next((p.split("=", 1)[1] for p in params if p.startswith("TZID=")), None)
                if v.endswith("Z"):
                    cur["start_utc"] = datetime.strptime(v, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
                elif "T" in v:
                    naive = datetime.strptime(v, "%Y%m%dT%H%M%S")
                    cur["start_utc"] = et_to_utc(naive) if tzid in (None, "US-Eastern", "America/New_York") else naive.replace(tzinfo=UTC)
                else:
                    cur["start_utc"] = datetime.strptime(v, "%Y%m%d").replace(tzinfo=UTC)
    return events


def from_bls_ics(start: date, end: date) -> list[dict]:
    events = parse_ics(get("bls", BLS_ICS).text)
    out = []
    for rel in tracked_releases():
        title = rel.get("bls_title")
        if not title:
            continue
        for ev in events:
            if ev.get("summary", "").lower().startswith(title.lower()) and start <= ev["start_utc"].date() <= end:
                e = _event(rel, ev["start_utc"].astimezone(NEW_YORK).date(), "bls_ics")
                e["scheduled_utc"], e["scheduled_tehran"] = ev["start_utc"], ev["start_utc"].astimezone(TEHRAN)
                out.append(e)
    return out


def from_fred(start: date, end: date) -> list[dict]:
    out = []
    for rel in tracked_releases():
        if not rel.get("fred_series"):
            continue
        info = fred.series_release(rel["fred_series"])
        if not info:
            continue
        for d in fred.release_dates(info["id"], start.isoformat(), end.isoformat()):
            out.append(_event(rel, d, "fred_release_dates"))
    return out


def from_fomc(start: date, end: date) -> list[dict]:
    rel = next(r for r in tracked_releases() if r["id"] == "fomc")
    out = []
    for m in fed.fomc_meetings():
        if start <= m["decision_date"] <= end:
            e = _event(rel, m["decision_date"], "federalreserve.gov")
            e["sep"] = m["sep"]
            out.append(e)
    return out


def upcoming(days: int = 14) -> tuple[list[dict], dict[str, str]]:
    """Merged calendar for the next `days` days. Official sources win; duplicates are dropped by event_id."""
    start = utcnow().date()
    end = start + timedelta(days=days)
    merged: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for name, fn in (("fomc", from_fomc), ("bls_ics", from_bls_ics), ("fred", from_fred)):
        try:
            for e in fn(start, end):
                merged.setdefault(e["event_id"], e)
        except SourceError as exc:
            errors[name] = str(exc)
    return sorted(merged.values(), key=lambda e: e["scheduled_utc"]), errors
