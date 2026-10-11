"""The coming releases as decisions: forecast (and where it comes from), odds of a hotter / in-line / cooler print, and
what each usually does to Bitcoin and gold in the 24 hours after.

Forecast, in order of preference: the analyst consensus (ForexFactory), else the system's own expectation (mean of the
last 3 prints / last 4 levels / the previous print, as in models.events). Odds come from models.forecast: prediction
market brackets (Kalshi), else a nowcast (Cleveland Fed, GDPNow) against the forecast, else even odds (no edge).
σ = the historical std of actual − consensus once enough consensus history exists, else the indicator's surprise scale.
The usual 24-hour move after a 1σ hotter (cooler) print is the event-study beta (reports.advisor.beta); "against
theory" flags a beta whose sign contradicts the economic prior. Volatility is the usual 24-hour sd after the release.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models import events as ev_model
from models import forecast as fm
from reports.facts import Facts, indicators_cfg, releases_cfg

SOURCE_FA = {"consensus": "اجماع", "system": "انتظار سیستم (اجماع رایگان در دسترس نیست)",
             "cleveland": "نوکست فد کلیولند", "gdpnow": "نوکست GDPNow فد آتلانتا", "kalshi": "بازار پیش‌بینی Kalshi",
             "fedwatch": "بازار پیش‌بینی (FedWatch، قراردادهای آتی فدرال)"}
TABLE_MIN_IMPORTANCE = 3


@dataclass
class EventOutlook:
    event_id: str
    release_id: str
    name_fa: str
    importance: int
    when: pd.Timestamp
    key: str | None = None
    forecast: float | None = None
    basis: str = "system"          # consensus | system
    previous: float | None = None
    nowcast: float | None = None
    nowcast_source: str | None = None
    market: bool = False
    sigma: float | None = None
    odds: fm.Odds | None = None
    effects: dict = field(default_factory=dict)  # asset → {beta, sd, conflict, conf}
    fedwatch: dict | None = None

    @property
    def sources(self) -> list[str]:
        out = ["consensus" if self.basis == "consensus" else "system"]
        if self.nowcast_source:
            out.append(self.nowcast_source)
        if self.market:
            out.append("kalshi")
        if self.fedwatch:
            out = ["fedwatch"]
        return out


def _release_keys() -> dict[str, dict]:
    return {r["key"]: r for r in indicators_cfg()["releases"]}


def system_expectation(f: Facts, key: str) -> float | None:
    """The naive expectation for the next print, exactly as models.events measures surprises (same vintage logic)."""
    rc = _release_keys().get(key)
    if not rc:
        return None
    obs = f._obs()
    s = obs[obs["series_key"] == key].dropna(subset=["value"]).sort_values("date")
    if len(s) < 5:
        return None
    s = s.drop_duplicates("date", keep="last").set_index(pd.to_datetime(s["date"]).dt.tz_localize(None))["value"]
    s.index = s.index.rename(None)
    freq = "W" if key == "initial_claims" else "M"
    step = pd.DateOffset(months=1) if freq == "M" else pd.Timedelta(days=7)
    t = s.index[-1] + step
    v = ev_model._expected(s, t, rc["measure"], rc["expected"], freq)
    return float(v) if np.isfinite(v) else None


def surprise_scale(f: Facts, key: str) -> float | None:
    s = (f.surprises or {}).get(key) or {}
    fb = s.get("scale")
    if fb is None and s.get("z") and s.get("actual") is not None and s.get("expected") is not None:
        fb = abs(s["actual"] - s["expected"]) / abs(s["z"])
    return fm.surprise_sigma(s.get("consensus_errors") or [], fb if fb and np.isfinite(fb) else None)


def _forecast_rows(f: Facts, event_id: str) -> pd.DataFrame:
    fc = f.forecasts
    if fc is None or fc.empty or "event_id" not in fc:
        return pd.DataFrame()
    return fc[fc["event_id"] == event_id]


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def pick_key(f: Facts, release_id: str, rows: pd.DataFrame) -> str | None:
    """The series that decides the release: market odds first, then a consensus, then the largest historical effect."""
    from reports.advisor import beta

    rk = _release_keys()
    keys = [k for k in releases_cfg().get(release_id, {}).get("series_keys", []) if k in rk]
    keys += [k for k in rows.get("key", pd.Series(dtype=str)).tolist() if k not in keys]
    if not keys:
        return None

    def rank(k):
        r = rows[rows["key"] == k] if not rows.empty else rows
        has_m = bool(not r.empty and isinstance(r.iloc[0].get("market_points"), str))
        has_c = bool(not r.empty and _num(r.iloc[0].get("consensus")) is not None)
        eff = max(abs((beta(f, k, a, "24h") or {}).get("coef") or 0) for a in ("BTC", "Gold"))
        return (has_m, has_c, eff)
    return max(keys, key=rank)


def event_outlook(f: Facts, e) -> EventOutlook:
    from reports.advisor import beta

    rel = releases_cfg().get(e.release_id, {})
    o = EventOutlook(e.event_id, e.release_id, rel.get("name_fa") or getattr(e, "name_fa", e.release_id),
                     int(getattr(e, "importance", rel.get("importance", 1)) or 1), pd.Timestamp(e.scheduled_utc))
    if e.release_id == "fomc":
        fw = f.state.get("fedwatch_next") or {}
        if fw.get("meeting") and str(fw["meeting"])[:10] == f"{o.when.tz_convert('America/New_York'):%Y-%m-%d}":
            o.fedwatch = fw
            hike, cut = fw.get("p_hike") or 0.0, fw.get("p_cut") or 0.0
            o.odds = fm.Odds(hike, max(0.0, 1 - hike - cut), cut, "market")
        return o
    rows = _forecast_rows(f, e.event_id)
    o.key = pick_key(f, e.release_id, rows)
    if not o.key:
        return o
    r = rows[rows["key"] == o.key].iloc[0] if not rows.empty and (rows["key"] == o.key).any() else None
    cons = _num(r.get("consensus")) if r is not None else None
    if cons is not None:
        o.forecast, o.basis = cons, "consensus"
        o.previous = _num(r.get("previous"))
    else:
        o.forecast = system_expectation(f, o.key)
    if r is not None:
        o.nowcast = _num(r.get("nowcast"))
        o.nowcast_source = r.get("nowcast_source") if o.nowcast is not None else None
    pts = None
    if r is not None and isinstance(r.get("market_points"), str):
        try:
            pts = [tuple(p) for p in json.loads(r["market_points"])]
        except (ValueError, TypeError):
            pts = None
    o.sigma = surprise_scale(f, o.key)
    o.odds = fm.event_odds(o.forecast, o.sigma, nowcast=o.nowcast, market=pts)
    o.market = bool(o.odds and o.odds.method == "market")
    for a in ("BTC", "Gold"):
        b = beta(f, o.key, a, "24h")
        if b:
            o.effects[a] = b
    return o


def upcoming(f: Facts, now: pd.Timestamp, days: int = 7, min_importance: int = TABLE_MIN_IMPORTANCE) -> list[EventOutlook]:
    c = f.calendar
    if c is None or c.empty:
        return []
    up = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=days))]
    if "importance" in up:
        up = up[up["importance"] >= min_importance]
    up = up[up["release_id"] != "h41"]  # the weekly balance sheet has no forecast and no surprise measure
    outs = [event_outlook(f, e) for e in up.drop_duplicates("event_id").sort_values("scheduled_utc").itertuples()]
    # a minor release without any forecast says nothing a reader can act on
    return [o for o in outs if o.importance >= 4 or o.forecast is not None or o.fedwatch]


def most_important(outs: list[EventOutlook]) -> EventOutlook | None:
    """The release that matters most this week: importance, then the size of its usual effect."""
    if not outs:
        return None

    def size(o):
        return max((abs(b.get("beta") or 0) for b in o.effects.values()), default=0)
    return max(outs, key=lambda o: (o.importance, o.fedwatch is not None, size(o), -o.when.value))
