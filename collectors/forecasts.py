"""Free forecasts for upcoming US releases: consensus, nowcasts and prediction-market odds.

Sources (each optional — a failing source is recorded and skipped, it never breaks a run):
- ForexFactory weekly calendar feed (nfs.faireconomy.media): analyst consensus ("forecast") and previous print for USD
  events. Only this week's file is published reliably (next week's is often 404). The feed asks clients to fetch it at
  most a couple of times per few minutes; the pipeline fetches each file once per run.
- Cleveland Fed inflation nowcast (nowcast_month.json): CPI, core CPI, PCE, core PCE m/m % for the target months.
- Atlanta Fed GDPNow via FRED (fredgraph.csv?id=GDPNOW, keyless): real GDP q/q annualised % for the current quarter.
- Kalshi public market data (read-only, no key): "above X" contracts → P(actual > X) for CPI, core CPI, core PCE,
  payrolls, unemployment rate and GDP.
The FOMC decision uses FedWatch from features_state.json (Phase 2), not this module.

Every value carries its source. Mappings from feed titles / market series to our release ids and series keys live in
config/forecast_sources.yaml.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd
import yaml

from core.http import SourceError, get
from core.settings import ROOT
from core.timeutil import NEW_YORK, UTC

FF_URLS = {"thisweek": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
           "nextweek": "https://nfs.faireconomy.media/ff_calendar_nextweek.json"}
CLEVELAND_URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
GDPNOW_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPNOW"
KALSHI_EVENTS = "https://api.elections.kalshi.com/trade-api/v2/events"
KALSHI_PAUSE_S = 0.3  # the public API allows ~10 requests/s; stay far below

# releases whose nowcast target is the previous calendar month of the release date
MONTHLY_NOWCAST_RELEASES = {"cpi", "pce"}


@lru_cache(maxsize=1)
def sources_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "forecast_sources.yaml").read_text(encoding="utf-8"))


# ───────────────────────────── ForexFactory ─────────────────────────────

_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_amount(s) -> float | None:
    """'0.3%' → 0.3, '195K' → 195000, '7.10M' → 7100000, '-25K' → -25000, '26.5' → 26.5, '' → None."""
    if s is None:
        return None
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*([KMBT%]?)\s*", str(s).replace(",", ""))
    if not m:
        return None
    v = float(m.group(1))
    return v * _MULT[m.group(2)] if m.group(2) in _MULT else v


def to_unit(v: float | None, unit: str) -> float | None:
    """A parsed feed value in the series' own measure (config units)."""
    if v is None:
        return None
    if unit == "thousands":
        return v / 1e3
    if unit == "jobs":  # Kalshi payroll strikes are in jobs; our payroll change is in thousands
        return v / 1e3
    return v  # pct, index, persons


def parse_forexfactory(rows: list[dict]) -> list[dict]:
    """USD rows of the feed that map to a tracked series: release_id, key, ET date/time, consensus, previous."""
    cfg = sources_cfg().get("forexfactory", {})
    out = []
    for r in rows or []:
        if r.get("country") != "USD":
            continue
        m = cfg.get((r.get("title") or "").strip())
        if not m:
            continue
        try:
            t = pd.Timestamp(r["date"])
        except (KeyError, ValueError):
            continue
        if t.tzinfo is None:
            continue
        cons = to_unit(parse_amount(r.get("forecast")), m["unit"])
        prev = to_unit(parse_amount(r.get("previous")), m["unit"])
        if cons is None and prev is None:
            continue
        out.append({"release_id": m["release"], "key": m["key"], "title": r["title"],
                    "time_utc": t.tz_convert(UTC), "et_date": t.tz_convert(NEW_YORK).date(),
                    "consensus": cons, "previous": prev})
    return out


def fetch_forexfactory() -> tuple[list[dict], dict]:
    from core.settings import get_settings

    if not get_settings().enable_ff_calendar:  # ENABLE_FF_CALENDAR=0 turns the consensus feed off
        return [], {"thisweek": "disabled (ENABLE_FF_CALENDAR=0)"}
    rows, status = [], {}
    for name, url in FF_URLS.items():
        try:
            data = get("forexfactory", url, timeout=30).json()
            rows += data if isinstance(data, list) else []
            status[name] = "ok"
        except (SourceError, ValueError) as exc:
            status[name] = f"failed: {str(exc)[:160]}"
    return parse_forexfactory(rows), status


# ───────────────────────────── Cleveland Fed nowcast ─────────────────────────────

def parse_cleveland(charts: list[dict]) -> dict[tuple[str, str], dict]:
    """{(key, 'YYYY-MM'): {value, as_of}} — the latest daily nowcast of each target month and series."""
    names = sources_cfg().get("cleveland", {})
    out: dict[tuple[str, str], dict] = {}
    for ch in charts or []:
        head = ch.get("chart", {})
        m = re.fullmatch(r"(\d{4})-(\d{1,2})", str(head.get("subcaption", "")).strip())
        if not m:
            continue
        month = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
        as_of = str(head.get("_comment", "")).strip()[:10] or None
        for ds in ch.get("dataset", []):
            key = names.get(ds.get("seriesname"))
            if not key:
                continue
            vals = [d.get("value") for d in ds.get("data", []) if d.get("value") not in ("", None)]
            if not vals:
                continue
            try:
                v = float(vals[-1])
            except ValueError:
                continue
            prev = out.get((key, month))
            if prev is None or (as_of or "") >= (prev["as_of"] or ""):
                out[(key, month)] = {"value": v, "as_of": as_of}
    return out


def fetch_cleveland() -> tuple[dict, str]:
    try:
        return parse_cleveland(get("cleveland", CLEVELAND_URL, timeout=90).json()), "ok"
    except (SourceError, ValueError) as exc:
        return {}, f"failed: {str(exc)[:160]}"


# ───────────────────────────── GDPNow (FRED graph csv) ─────────────────────────────

def parse_gdpnow(text: str) -> dict | None:
    """Latest GDPNow estimate: {quarter_start: date, value}."""
    rows = [r for r in csv.reader(io.StringIO(text or "")) if r]
    best = None
    for r in rows[1:]:
        try:
            d, v = date.fromisoformat(r[0]), float(r[1])
        except (ValueError, IndexError):
            continue
        best = {"quarter_start": d, "value": v}
    return best


def fetch_gdpnow() -> tuple[dict | None, str]:
    try:
        return parse_gdpnow(get("gdpnow", GDPNOW_URL, timeout=30).text), "ok"
    except SourceError as exc:
        return None, f"failed: {str(exc)[:160]}"


# ───────────────────────────── Kalshi ─────────────────────────────

def _price(m: dict) -> float | None:
    """P(yes): the bid/ask mid when both quotes exist, else the last trade."""
    f = lambda k: parse_amount(m.get(k))  # noqa: E731
    bid, ask, last = f("yes_bid_dollars"), f("yes_ask_dollars"), f("last_price_dollars")
    if bid is not None and ask is not None and 0 < ask and bid <= ask:
        return (bid + ask) / 2
    return last


def parse_kalshi(series: str, payload: dict) -> list[dict]:
    """One row per open event: release_id, key, close ET date, survival points [(strike, P(actual > strike))]."""
    m = sources_cfg().get("kalshi", {}).get(series)
    if not m:
        return []
    out = []
    for ev in (payload or {}).get("events", []):
        pts, close = [], None
        for mk in ev.get("markets", []):
            if mk.get("strike_type") != "greater" or mk.get("floor_strike") is None:
                continue
            if str(mk.get("status", "active")) in ("closed", "settled", "finalized", "determined"):
                continue
            p = _price(mk)
            if p is None:
                continue
            pts.append((to_unit(float(mk["floor_strike"]), m["unit"]), round(p, 4)))
            close = close or mk.get("close_time")
        if len(pts) < 2 or not close:
            continue
        t = pd.Timestamp(close)
        out.append({"release_id": m["release"], "key": m["key"], "ticker": ev.get("event_ticker"),
                    "et_date": t.tz_convert(NEW_YORK).date(), "points": sorted(pts)})
    return out


def fetch_kalshi() -> tuple[list[dict], dict]:
    rows, status = [], {}
    for series in sources_cfg().get("kalshi", {}):
        try:
            payload = get("kalshi", KALSHI_EVENTS, timeout=30,
                          params={"series_ticker": series, "status": "open", "with_nested_markets": "true",
                                  "limit": 5}).json()
            got = parse_kalshi(series, payload)
            rows += got
            status[series] = "ok" if got else "no open market"
        except (SourceError, ValueError) as exc:
            status[series] = f"failed: {str(exc)[:160]}"
        time.sleep(KALSHI_PAUSE_S)
    return rows, status


# ───────────────────────────── matching to the calendar ─────────────────────────────

def _et_date(ts) -> date:
    return pd.Timestamp(ts).tz_convert(NEW_YORK).date()


def match_event(calendar: pd.DataFrame, release_id: str, et_day: date, tolerance_days: int = 1) -> str | None:
    """The calendar event of `release_id` on (or within a day of) the given US/Eastern date."""
    if calendar is None or calendar.empty:
        return None
    c = calendar[calendar["release_id"] == release_id]
    best, gap = None, None
    for r in c.itertuples():
        g = abs((_et_date(r.scheduled_utc) - et_day).days)
        if g <= tolerance_days and (gap is None or g < gap):
            best, gap = r.event_id, g
    return best


def prev_month(d: date) -> str:
    first = d.replace(day=1) - timedelta(days=1)
    return f"{first.year:04d}-{first.month:02d}"


def assemble(calendar: pd.DataFrame, ff: list[dict], cleveland: dict, gdpnow: dict | None, kalshi: list[dict],
             now: pd.Timestamp, days: int = 14) -> pd.DataFrame:
    """One row per upcoming (event, series key) with every forecast found and its source."""
    cols = ["event_id", "release_id", "key", "scheduled_utc", "consensus", "previous", "consensus_source",
            "nowcast", "nowcast_source", "nowcast_as_of", "market_points", "market_source", "market_ticker"]
    if calendar is None or calendar.empty:
        return pd.DataFrame(columns=cols)
    cal = calendar.copy()
    cal["scheduled_utc"] = pd.to_datetime(cal["scheduled_utc"], utc=True)
    up = cal[(cal["scheduled_utc"] > now - pd.Timedelta(hours=6)) & (cal["scheduled_utc"] <= now + pd.Timedelta(days=days))]
    up = up.drop_duplicates("event_id")
    rows: dict[tuple[str, str], dict] = {}

    def row(event_id: str, release_id: str, key: str) -> dict:
        e = up[up["event_id"] == event_id].iloc[0]
        return rows.setdefault((event_id, key), {c: None for c in cols} | {
            "event_id": event_id, "release_id": release_id, "key": key, "scheduled_utc": e["scheduled_utc"].isoformat()})

    for r in ff:
        eid = match_event(up, r["release_id"], r["et_date"])
        if eid:
            x = row(eid, r["release_id"], r["key"])
            x.update(consensus=r["consensus"], previous=r["previous"], consensus_source="forexfactory")
    for e in up.itertuples():
        d = _et_date(e.scheduled_utc)
        if e.release_id in MONTHLY_NOWCAST_RELEASES:
            for (key, month), v in cleveland.items():
                if month == prev_month(d) and _release_of(key) == e.release_id:
                    row(e.event_id, e.release_id, key).update(nowcast=v["value"], nowcast_source="cleveland",
                                                              nowcast_as_of=v["as_of"])
        if e.release_id == "gdp" and gdpnow:
            q_end = (pd.Timestamp(gdpnow["quarter_start"]) + pd.DateOffset(months=3)).date()
            if q_end <= d <= q_end + timedelta(days=40):  # the advance estimate of the quarter GDPNow tracks
                row(e.event_id, "gdp", "gdp").update(nowcast=gdpnow["value"], nowcast_source="gdpnow")
    for k in kalshi:
        eid = match_event(up, k["release_id"], k["et_date"])
        if eid:
            row(eid, k["release_id"], k["key"]).update(market_points=json.dumps(k["points"]), market_source="kalshi",
                                                       market_ticker=k["ticker"])
    return pd.DataFrame(list(rows.values()), columns=cols)


def _release_of(key: str) -> str | None:
    for m in sources_cfg().get("forexfactory", {}).values():
        if m["key"] == key:
            return m["release"]
    return None


def collect(calendar: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Fetch every source once and assemble the forecasts table. Never raises for a source failure."""
    ff, ff_status = fetch_forexfactory()
    cl, cl_status = fetch_cleveland()
    gd, gd_status = fetch_gdpnow()
    ks, ks_status = fetch_kalshi()
    status = {"forexfactory": ff_status, "cleveland": cl_status, "gdpnow": gd_status, "kalshi": ks_status,
              "counts": {"forexfactory_rows": len(ff), "cleveland_series": len(cl), "kalshi_events": len(ks)},
              "fetched_utc": datetime.now(UTC).isoformat()}
    return assemble(calendar, ff, cl, gd, ks, now), status
