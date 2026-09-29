"""Collect every number a report may use from the job outputs (the "facts"). Messages and Claude only ever see this;
nothing in a report is computed elsewhere, and Claude's text is checked against it (reports.writer)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from core.registry import load_registry
from core.settings import ROOT

MEASURE_FA = {"mom_pct": "تغییر ماهانه (٪)", "level": "سطح", "diff": "تغییر", "qoq_ann": "رشد فصلی سالانه‌شده (٪)",
              "mom_change": "تغییر ماهانه", "yoy_pct": "تغییر سالانه (٪)"}


def indicators_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text(encoding="utf-8"))


def releases_cfg() -> dict[str, dict]:
    rel = yaml.safe_load((ROOT / "config" / "releases.yaml").read_text(encoding="utf-8"))["releases"]
    return {r["id"]: r for r in rel}


_NAMES: dict[str, str] | None = None


def ind_name(key: str) -> str:
    """Persian display name of a series or derived indicator."""
    global _NAMES
    if _NAMES is None:
        from dashboard.cards import SECTIONS

        _NAMES = {c.key: c.title_fa for cs in SECTIONS.values() for c in cs}
        _NAMES.update({k: m.name_fa for k, m in load_registry().items() if m.name_fa})
        _NAMES.update({"net_liquidity_weekly": "نقدینگی خالص (هفتگی)", "copper_futures": "مس", "wti_futures": "نفت WTI"})
    return _NAMES.get(key, key)


def channel_fa(cfg: dict) -> dict[str, str]:
    return {k: v["fa"] for k, v in cfg["channels"].items()}


@dataclass
class Facts:
    root: Path
    as_of: str | None = None
    state: dict = field(default_factory=dict)              # features_state.json
    macro: dict = field(default_factory=dict)              # macro_scores.json
    signals: dict = field(default_factory=dict)            # signals.json
    releases: list[dict] = field(default_factory=list)     # recent_releases.json
    coefs: pd.DataFrame = field(default_factory=pd.DataFrame)
    calendar: pd.DataFrame = field(default_factory=pd.DataFrame)
    news: pd.DataFrame = field(default_factory=pd.DataFrame)  # Claude-classified news
    availability: dict = field(default_factory=dict)
    prepos: dict = field(default_factory=dict)             # prepos.json (pre-positioning backtest + live)

    # ── lookups ──
    def asset(self, a: str) -> dict:
        return next((x for x in self.signals.get("assets", []) if x.get("asset") == a), {})

    def coef(self, indicator: str, asset: str, horizon: str, sample: str = "full") -> dict | None:
        c = self.coefs
        if c.empty:
            return None
        r = c[(c["indicator"] == indicator) & (c["asset"] == asset) & (c["horizon"] == horizon) & (c["sample"] == sample)]
        return None if r.empty else r.iloc[0].to_dict()

    def latest(self, key: str) -> tuple[float, str] | None:
        """Latest observation (value, date) of a series from the snapshot."""
        obs = self._obs()
        s = obs[obs["series_key"] == key]
        if s.empty:
            return None
        r = s.sort_values("date").iloc[-1]
        return float(r["value"]), f"{pd.Timestamp(r['date']):%Y-%m-%d}"

    _obs_cache: pd.DataFrame | None = None

    def _obs(self) -> pd.DataFrame:
        if self._obs_cache is None:
            p = self.root / "cache" / "observations.csv.gz"
            self._obs_cache = pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame(columns=["series_key", "date", "value"])
        return self._obs_cache

    def important_news(self, since: pd.Timestamp, min_importance: int = 4) -> pd.DataFrame:
        n = self.news
        if n.empty or "importance" not in n:
            return n
        n = n[(n["importance"] >= min_importance) & (n["published_utc"] >= since)]
        return n.sort_values(["importance", "published_utc"], ascending=False)

    def summary(self) -> dict:
        """Compact JSON-able view handed to Claude (and used to validate its numbers)."""
        cfg = indicators_cfg()
        rel_cfg = {r["key"]: r for r in cfg["releases"]}
        chan = channel_fa(cfg)
        out = {"as_of": self.as_of, "regime": self.state.get("regime"), "fed_stance": self.state.get("fed_stance"),
               "fedwatch_next": self.state.get("fedwatch_next"), "curve_state": self.state.get("curve_state"),
               "channels_fa": chan, "assets": {}, "releases": []}
        for r in self.releases:
            rc = rel_cfg.get(r["indicator"], {})
            out["releases"].append({**r, "name_fa": ind_name(r["indicator"]), "measure": rc.get("measure"),
                                    "hotter_means": rc.get("up_means"), "channels": rc.get("channels", []),
                                    "coef_24h": {a: (self.coef(r["indicator"], a, "24h") or {}).get("coefficient")
                                                 for a in ("BTC", "Gold")}})
        for a in ("BTC", "Gold"):
            m = self.macro.get(a, {})
            x = self.asset(a)
            out["assets"][a] = {
                "price": x.get("price"),
                "macro_score": {b: (m.get(b) or {}).get("score") for b in ("short", "medium", "long")},
                "top_contributors": {b: [{**t, "name_fa": ind_name(t["indicator"])} for t in (m.get(b) or {}).get("top", [])[:4]]
                                     for b in ("short", "medium", "long")},
                "expected_range": x.get("expected_range"),
                "signals": len(x.get("signals", [])), "watchlist": len(x.get("watchlist", [])),
            }
        return out


def load(root: Path) -> Facts:
    rj = lambda n: json.loads((root / n).read_text(encoding="utf-8")) if (root / n).exists() else {}  # noqa: E731
    avail = rj("data_availability.json")
    f = Facts(root=root, as_of=avail.get("generated_at") or rj("features_state.json").get("as_of"),
              state=rj("features_state.json"), macro=rj("macro_scores.json"), signals=rj("signals.json"),
              releases=rj("recent_releases.json") or [], prepos=rj("prepos.json"),
              availability={r["key"]: r for r in avail.get("series", [])})
    if (root / "impact_coefficients.csv").exists():
        f.coefs = pd.read_csv(root / "impact_coefficients.csv")
    cal = root / "cache" / "calendar_events.csv"
    if cal.exists():
        f.calendar = pd.read_csv(cal)
        f.calendar["scheduled_utc"] = pd.to_datetime(f.calendar["scheduled_utc"], utc=True)
    ns = root / "cache" / "news_scored.csv"
    if ns.exists():
        f.news = pd.read_csv(ns)
        if not f.news.empty:
            f.news["published_utc"] = pd.to_datetime(f.news["published_utc"], utc=True, errors="coerce", format="mixed")
    return f
