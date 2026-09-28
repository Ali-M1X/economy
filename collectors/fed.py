"""Federal Reserve: press-release / speech feeds and the FOMC meeting calendar (federalreserve.gov)."""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from collectors.rss import fetch_feed
from core.http import SourceError, get
from core.timeutil import et_to_utc

FEEDS = {
    "fed_monetary": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "fed_all": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
    "fed_testimony": "https://www.federalreserve.gov/feeds/testimony.xml",
}
FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
STATEMENT_HOUR_ET = 14  # FOMC statements are released at 2:00 p.m. ET


def fed_feeds() -> tuple[list[dict], dict[str, str]]:
    items, errors = [], {}
    for name, url in FEEDS.items():
        try:
            items.extend(fetch_feed(url, name))
        except SourceError as exc:
            errors[name] = str(exc)
    return items, errors


def _month(token: str) -> int:
    token = token.strip().lower()[:3]
    for name, num in MONTHS.items():
        if name.startswith(token):
            return num
    raise SourceError("fed", f"unknown month {token!r}")


def parse_fomc_calendar(html: str) -> list[dict]:
    """Returns one dict per scheduled meeting: decision date (last day), SEP flag, statement time (UTC)."""
    soup = BeautifulSoup(html, "html.parser")
    meetings = []
    for panel in soup.select("div.panel"):
        heading = panel.find(["h4", "h5", "a"], string=re.compile(r"\d{4} FOMC Meetings"))
        if heading is None:
            continue
        year = int(re.search(r"(\d{4})", heading.get_text()).group(1))
        for row in panel.select("div.fomc-meeting"):
            month_el = row.select_one(".fomc-meeting__month")
            date_el = row.select_one(".fomc-meeting__date")
            if not month_el or not date_el:
                continue
            month_txt, date_txt = month_el.get_text(" ", strip=True), date_el.get_text(" ", strip=True)
            if "notation" in date_txt.lower() or "unscheduled" in date_txt.lower():
                continue
            days = re.findall(r"\d+", date_txt)
            if not days:
                continue
            # "Apr/May" headers: the meeting starts in the first month and ends (decision) in the last
            parts = month_txt.split("/")
            start_month, month = _month(parts[0]), _month(parts[-1])
            last_day = int(days[-1])
            decision = date(year, month, last_day)
            first = date(year, start_month, int(days[0]))
            meetings.append({
                "start_date": first, "decision_date": decision,
                "sep": "*" in date_txt or "Summary of Economic Projections" in row.get_text(),
                "statement_utc": et_to_utc(datetime(decision.year, decision.month, decision.day, STATEMENT_HOUR_ET)),
            })
    meetings.sort(key=lambda m: m["decision_date"])
    return meetings


def fomc_meetings() -> list[dict]:
    meetings = parse_fomc_calendar(get("fed", FOMC_CALENDAR).text)
    if not meetings:
        raise SourceError("fed", "FOMC calendar parsed to zero meetings (page layout changed?)")
    return meetings
