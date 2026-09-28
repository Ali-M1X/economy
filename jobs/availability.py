"""Phase-1 Data Availability Report.

Runs every series collector and every non-series probe (exchanges, Fed, calendar, news, futures),
validates the results and writes `output/data_availability.{md,json}`.

    python -m jobs.availability            # fetch + validate + report
    python -m jobs.availability --store    # also upsert into DATABASE_URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from collectors import calendar, crypto, fed, fred, futures, news
from collectors.base import SeriesResult
from collectors.series import fetch_series
from core.http import SourceError
from core.registry import SeriesMeta, load_registry
from core.settings import ROOT, get_settings
from core.timeutil import utcnow
from validation.checks import CheckResult, check_cross, overall, staleness_days, validate_series

OUT_DIR = ROOT / "output"


@dataclass
class SeriesReport:
    key: str
    name_en: str
    category: str
    source: str
    source_id: str
    url: str
    units: str
    frequency: str
    release_lag_days: int
    proxy: bool
    proxy_for: str | None
    status: str = "fail"
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    last_value: float | None = None
    age_days: int | None = None
    missing_values: int = 0
    source_url: str | None = None
    source_last_updated: str | None = None
    source_units: str | None = None
    fetch_seconds: float = 0.0
    checks: list[dict] = field(default_factory=list)
    vintages: str = "n/a"
    error: str | None = None


@dataclass
class ProbeReport:
    name: str
    group: str
    status: str
    detail: str
    proxy: bool = False


def _run_series(meta: SeriesMeta) -> tuple[SeriesReport, SeriesResult | None]:
    rep = SeriesReport(key=meta.key, name_en=meta.name_en, category=meta.category, source=meta.source,
                       source_id=meta.source_id, url=meta.url, units=meta.units, frequency=meta.frequency,
                       release_lag_days=meta.release_lag_days, proxy=meta.proxy, proxy_for=meta.proxy_for)
    t0 = time.monotonic()
    try:
        res = fetch_series(meta)
    except Exception as exc:  # noqa: BLE001 — every failure must land in the report
        rep.error = f"{exc.__class__.__name__}: {exc}"[:400]
        rep.fetch_seconds = round(time.monotonic() - t0, 2)
        return rep, None
    rep.fetch_seconds = round(time.monotonic() - t0, 2)
    df = res.data
    today = utcnow().date()
    try:
        checks = validate_series(df, meta, today, res.source_last_updated)
    except Exception as exc:  # noqa: BLE001 — a validation bug must not abort the whole report
        checks = [CheckResult("validation", "fail", f"{exc.__class__.__name__}: {exc}"[:300])]
    rep.checks = [asdict(c) for c in checks]
    rep.status = overall(checks)
    rep.rows = len(df)
    rep.missing_values = res.missing_values
    rep.source_url = res.source_url
    rep.source_units = res.source_units
    rep.source_last_updated = res.source_last_updated.isoformat() if res.source_last_updated else None
    if not df.empty:
        rep.first_date = f"{df['date'].iloc[0]:%Y-%m-%d}"
        rep.last_date = f"{df['date'].iloc[-1]:%Y-%m-%d}"
        rep.last_value = round(float(df["value"].iloc[-1]), 6)
        rep.age_days = staleness_days(df, today, meta.frequency)
    return rep, res


def run_series(registry: dict[str, SeriesMeta], workers: int = 6):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_run_series, registry.values()))
    reports = {r.key: r for r, _ in out}
    results = {r.key: res for r, res in out if res is not None}
    # Cross-source checks
    for key, meta in registry.items():
        cc = meta.cross_check
        if not cc or key not in results:
            continue
        if cc.key not in results:
            res = CheckResult("cross_check", "warn", f"counterpart {cc.key} unavailable")
        else:
            res = check_cross(results[key].data, results[cc.key].data, cc.tolerance, cc.mode)
            res.message = f"vs {cc.key}: {res.message}"
        reports[key].checks.append(asdict(res))
        reports[key].status = overall([CheckResult(**c) for c in reports[key].checks])
    return reports, results


def run_vintages(registry: dict[str, SeriesMeta], reports: dict[str, SeriesReport]) -> dict[str, pd.DataFrame]:
    has_key = bool(get_settings().fred_api_key)
    out: dict[str, pd.DataFrame] = {}
    for key, meta in registry.items():
        if not meta.vintages:
            continue
        if not has_key:
            reports[key].vintages = "skipped (FRED_API_KEY not set)"
            continue
        try:
            v = fred.fetch_vintages(meta)
            out[key] = v
            reports[key].vintages = f"ok: {len(v)} vintage rows, {v['realtime_start'].nunique()} release dates"
        except Exception as exc:  # noqa: BLE001
            reports[key].vintages = f"fail: {exc}"[:300]
    return out


def _probe(name: str, group: str, fn, proxy: bool = False, known_block: bool = False) -> ProbeReport:
    """`known_block`: the host is documented (DATA_GAPS.md) as refusing US runners, so an HTTP 403/451
    is reported as an expected limitation (warn) rather than a new failure."""
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001
        status, detail = "fail", f"{exc.__class__.__name__}: {exc}"[:400]
        if known_block and ("HTTP 403" in detail or "HTTP 451" in detail):
            status, detail = "warn", "blocked from US runners — expected, see DATA_GAPS.md (" + detail[:160] + ")"
    return ProbeReport(name, group, status, detail, proxy)


def _errs(errs: dict[str, str]) -> str:
    """One short reason per failing source, so no source is hidden by truncation."""
    if not errs:
        return ""
    return "; failed: " + "; ".join(f"{k}: {v.split(']', 1)[-1].strip()[:90]}" for k, v in errs.items())


def run_probes(results: dict[str, SeriesResult] | None = None) -> tuple[list[ProbeReport], dict]:
    probes: list[ProbeReport] = []
    stash: dict = {}

    # Candles: which venue serves each asset/interval (first in fallback chain), and every venue individually.
    for asset in ("BTC", "PAXG"):
        for iv in crypto.INTERVALS:
            def f(asset=asset, iv=iv):
                r = crypto.candles(asset, iv, limit=300)
                stash[(asset, iv)] = r
                last = r.data["ts"].iloc[-1]
                note = f"; skipped: {list(r.errors)}" if r.errors else ""
                return "ok" if not r.errors else "warn", (
                    f"served by {r.venue}: {len(r.data)} bars, last {last:%Y-%m-%d %H:%M}Z close {r.data['close'].iloc[-1]:,.2f}{note}")
            probes.append(_probe(f"candles {asset} {iv}", "prices", f, proxy=asset == "PAXG"))
        for venue in crypto.CANDLE_PROVIDERS:
            def g(asset=asset, venue=venue):
                r = crypto.candles(asset, "1h", limit=50, venues=(venue,))
                return "ok", f"{len(r.data)} bars, last close {r.data['close'].iloc[-1]:,.2f}"
            probes.append(_probe(f"venue {venue} {asset} 1h", "prices", g, proxy=asset == "PAXG",
                                 known_block=venue in crypto.GEO_BLOCKED_FROM_US))

    def btc_vs_paxg_sanity():
        btc = stash.get(("BTC", "1d"))
        pax = stash.get(("PAXG", "1d"))
        if not btc or not pax:
            return "warn", "missing inputs"
        return "ok", f"BTC last {btc.data['close'].iloc[-1]:,.0f} USD; PAXG last {pax.data['close'].iloc[-1]:,.2f} USD/oz"
    probes.append(_probe("latest prices", "prices", btc_vs_paxg_sanity))

    for asset in ("BTC", "PAXG"):
        def ob(asset=asset):
            books, errs = crypto.order_books(asset)
            if not books:
                return "fail", f"no venue: {errs}"
            depth = crypto.aggregate_depth(books, bucket=100.0 if asset == "BTC" else 5.0)
            bids = depth[depth.side == "bid"]
            asks = depth[depth.side == "ask"]
            stash[f"book_{asset}"] = (books, depth)
            return ("ok" if len(books) >= 2 else "warn"), (
                f"venues {sorted(books)}; {len(bids)} bid / {len(asks)} ask buckets; "
                f"bid ${bids.usd.sum()/1e6:,.1f}M ask ${asks.usd.sum()/1e6:,.1f}M" + (f"; failed {list(errs)}" if errs else ""))
        probes.append(_probe(f"order books {asset}", "liquidity", ob, proxy=asset == "PAXG"))

    def deriv():
        d, errs = crypto.derivatives("BTC")
        stash["deriv"] = d
        if not d:
            return "fail", str(errs)
        parts = [f"{v}: OI {x['open_interest']:,.0f} BTC (${x['open_interest_usd']/1e9:,.2f}B), funding {x['funding_rate']*100:.4f}%"
                 for v, x in d.items()]
        blocked = [v for v in errs if v in crypto.GEO_BLOCKED_FROM_US and ("403" in errs[v] or "451" in errs[v])]
        other = {v: e for v, e in errs.items() if v not in blocked}
        if blocked:
            parts.append(f"geo-blocked from US (expected): {blocked}")
        if other:
            parts.append(f"failed: {other}")
        return ("warn" if other else "ok"), "; ".join(parts)
    probes.append(_probe("open interest & funding BTC", "derivatives", deriv))

    def oi_hist():
        df = crypto.open_interest_history("BTC", "1h")
        stash["oi_hist"] = df
        last = df.iloc[-1]
        return "ok", (f"okx OI history: {len(df)} hourly points, last {last.ts:%Y-%m-%d %H:%M}Z "
                      f"{last.open_interest:,.0f} BTC (${last.open_interest_usd/1e9:,.2f}B)")
    probes.append(_probe("open interest history BTC", "derivatives", oi_hist))

    def liq_okx():
        df = crypto.recent_liquidations_okx("BTC")
        stash["liq_okx"] = df
        if df.empty:
            return "warn", "endpoint answered but returned no recent liquidations"
        return "ok", f"{len(df)} liquidations, {df['ts'].min():%m-%d %H:%M}→{df['ts'].max():%m-%d %H:%M}Z"
    probes.append(_probe("liquidations OKX (REST)", "derivatives", liq_okx))

    def liq_bybit():
        rows = asyncio.run(crypto.stream_liquidations_bybit("BTC", seconds=45))
        stash["liq_bybit"] = rows
        return ("ok" if rows else "warn"), f"websocket connected; {len(rows)} liquidations in 45s"
    probes.append(_probe("liquidations Bybit (WebSocket 45s)", "derivatives", liq_bybit))

    # Fed
    def feeds():
        items, errs = fed.fed_feeds()
        latest = max((i["published_utc"] for i in items if i["published_utc"]), default=None)
        return ("ok" if not errs else "warn"), f"{len(items)} items, latest {latest}" + _errs(errs)
    probes.append(_probe("Fed RSS (press, speeches, testimony)", "fed", feeds))

    def fomc():
        ms = fed.fomc_meetings()
        nxt = [m for m in ms if m["decision_date"] >= utcnow().date()]
        stash["fomc"] = ms
        nx = ", ".join(f"{m['decision_date']}{'*' if m['sep'] else ''}" for m in nxt[:4])
        return "ok", f"{len(ms)} meetings parsed ({ms[0]['decision_date'].year}–{ms[-1]['decision_date'].year}); next: {nx} (*=SEP)"
    probes.append(_probe("FOMC meeting calendar", "fed", fomc))

    # Calendar
    def bls_ics():
        today = utcnow().date()
        ev = calendar.from_bls_ics(today, today + pd.Timedelta(days=45))
        return ("ok" if ev else "warn"), f"{len(ev)} tracked BLS releases in next 45d: " + ", ".join(
            f"{e['release_id']} {e['scheduled_utc']:%m-%d %H:%M}Z" for e in ev[:6])
    probes.append(_probe("BLS release calendar (ICS)", "calendar", bls_ics, known_block=True))

    def fred_dates():
        if not get_settings().fred_api_key:
            return "warn", "skipped: FRED_API_KEY not set"
        today = utcnow().date()
        ev = calendar.from_fred(today, today + pd.Timedelta(days=30))
        stash["calendar_fred"] = ev
        return "ok", f"{len(ev)} tracked releases in next 30d: " + ", ".join(f"{e['release_id']} {e['scheduled_utc']:%m-%d}" for e in ev[:8])
    probes.append(_probe("FRED release dates", "calendar", fred_dates))

    # News
    def agency_feeds():
        items, errs = news.collect()
        stash["news"] = items
        by_src = pd.Series([i["source"].split(":")[0] for i in items]).value_counts().to_dict() if items else {}
        return ("ok" if not errs else "warn"), f"{len(items)} unique items {by_src}" + _errs(errs)
    probes.append(_probe("news (Fed/BLS/BEA/Treasury RSS + GDELT)", "news", agency_feeds))

    # Rate expectations
    for root, n in (("ZQ", 13), ("SR3", 8)):
        def fc(root=root, n=n):
            df, errs = futures.curve(root, n)
            stash[f"fut_{root}"] = df
            if df.empty:
                return "fail", f"no contracts resolved; e.g. {list(errs.items())[:1]}"
            path = ", ".join(f"{r.contract_month:%b%y} {r.implied_rate:.3f}%" for r in df.itertuples())
            return ("ok" if not errs else "warn"), f"{len(df)}/{n} contracts: {path}" + (f"; missing {list(errs)}" if errs else "")
        probes.append(_probe(f"{root} futures curve (Yahoo)", "expectations", fc))

    results = results or {}

    def zq_vs_effr():
        curve_df = stash.get("fut_ZQ")
        effr = results.get("fed_funds_eff_daily")
        if curve_df is None or curve_df.empty or effr is None:
            return "warn", "missing inputs (ZQ curve or DFF)"
        fomc_dates = [m["decision_date"] for m in stash.get("fomc", [])]
        return futures.check_current_month(curve_df, effr.data, utcnow().date(), fomc_dates)
    probes.append(_probe("ZQ current-month contract vs realized EFFR", "expectations", zq_vs_effr))

    def continuous_mapping():
        # Which specific contract do Yahoo's "continuous" tickers actually track?
        parts = []
        for cont, root in (("ZQ=F", "ZQ"), ("SR3=F", "SR3")):
            curve_df = stash.get(f"fut_{root}")
            price, _ = futures._last_price(cont)
            if curve_df is None or curve_df.empty:
                parts.append(f"{cont} {price:.3f}")
                continue
            near = curve_df.iloc[(curve_df["price"] - price).abs().argsort()[:1]].iloc[0]
            front = curve_df.iloc[0]
            parts.append(f"{cont} {price:.3f} ≈ {near.contract} {near.price:.3f} (front {front.contract} {front.price:.3f})")
        return "ok", "; ".join(parts)
    probes.append(_probe("Yahoo continuous futures vs contracts", "expectations", continuous_mapping))

    def spot_vs_futures():
        out = []
        for name, spot_k, fut_k in (("Brent", "brent_spot_eia", "brent_futures"), ("WTI", "wti_spot_eia", "wti_futures")):
            if spot_k not in results or fut_k not in results:
                out.append(f"{name}: missing inputs")
                continue
            m = results[spot_k].data.merge(results[fut_k].data, on="date", suffixes=("_spot", "_fut")).tail(60)
            prem = (m["value_spot"] / m["value_fut"] - 1) * 100
            last = m.tail(5)
            recent = ", ".join(f"{r.date:%m-%d} {r.value_spot:.2f}/{r.value_fut:.2f}" for r in last.itertuples())
            out.append(f"{name} spot−futures premium over last {len(m)} common days: mean {prem.mean():+.1f}%, "
                       f"min {prem.min():+.1f}%, max {prem.max():+.1f}%; last (spot/fut): {recent}")
        return "ok", " || ".join(out)
    probes.append(_probe("spot vs front futures (Brent, WTI)", "commodities", spot_vs_futures))

    return probes, stash


def _store_probe_data(stash: dict) -> dict[str, int]:
    from db import schema
    from db.store import upsert

    def py(v):  # numpy/pandas scalars → plain Python for the DB driver
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        return v.item() if hasattr(v, "item") else v

    now = utcnow()
    n: dict[str, int] = {}
    for (asset, iv), r in [(k, v) for k, v in stash.items() if isinstance(k, tuple)]:
        rows = [{"asset": asset, "venue": r.venue, "interval": iv, "ts": t.to_pydatetime(), "open": py(o), "high": py(h),
                 "low": py(lo), "close": py(c), "volume": py(v), "is_proxy": asset == "PAXG"}
                for t, o, h, lo, c, v in r.data[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)]
        n["candles"] = n.get("candles", 0) + upsert(schema.candles, rows)
    for asset in ("BTC", "PAXG"):
        if f"book_{asset}" in stash:
            books, depth = stash[f"book_{asset}"]
            n["orderbook_snapshots"] = n.get("orderbook_snapshots", 0) + upsert(schema.orderbook_snapshots, [{
                "asset": asset, "ts": now, "bucket_size": 100.0 if asset == "BTC" else 5.0, "mid_price": None,
                "venues": sorted(books), "depth": json.loads(depth.to_json(orient="records"))}])
    deriv_rows = [{"asset": "BTC", "venue": venue, "ts": now, **{k: py(v) for k, v in d.items()}}
                  for venue, d in stash.get("deriv", {}).items()]
    oi = stash.get("oi_hist")
    if oi is not None and not oi.empty:
        deriv_rows += [{"asset": "BTC", "venue": "okx", "ts": r.ts.to_pydatetime(), "open_interest": py(r.open_interest),
                        "open_interest_usd": py(r.open_interest_usd), "funding_rate": None, "mark_price": None,
                        "next_funding_ts": None} for r in oi.itertuples()]
    if deriv_rows:
        # multi-row INSERT needs one key set: normalise every row to the table's columns
        n["derivatives_snapshots"] = upsert(schema.derivatives_snapshots, [
            {c.name: r.get(c.name) for c in schema.derivatives_snapshots.columns} for r in deriv_rows])
    liq_rows = []
    liq = stash.get("liq_okx")
    if liq is not None and not liq.empty:
        liq_rows += [{"raw_id": r.raw_id, "venue": "okx", "asset": r.asset, "ts": r.ts.to_pydatetime(), "side": r.side,
                      "price": py(r.price), "qty": py(r.qty_contracts), "usd": None} for r in liq.itertuples()]
    liq_rows += [{**r, "ts": r["ts"].to_pydatetime(), "usd": r["price"] * r["qty"]} for r in stash.get("liq_bybit", [])]
    if liq_rows:
        n["liquidations"] = upsert(schema.liquidations, liq_rows)
    for root in ("ZQ", "SR3"):
        df = stash.get(f"fut_{root}")
        if df is not None and not df.empty:
            n["futures_quotes"] = n.get("futures_quotes", 0) + upsert(
                schema.futures_quotes, [{k: py(v) for k, v in r.items()} for r in df.to_dict("records")])
    events = list(stash.get("calendar_fred", []))
    rel_fomc = next(r for r in calendar.tracked_releases() if r["id"] == "fomc")
    for m in stash.get("fomc", []):
        events.append(calendar._event(rel_fomc, m["decision_date"], "federalreserve.gov"))
    if events:
        n["calendar_events"] = upsert(schema.calendar_events, [{
            "event_id": e["event_id"], "release_id": e["release_id"], "name_en": e["name_en"], "name_fa": e["name_fa"],
            "scheduled_utc": e["scheduled_utc"], "importance": e["importance"], "date_source": e["date_source"],
            "status": "scheduled" if e["scheduled_utc"] > now else "released"} for e in events])
    items = stash.get("news", [])
    if items:
        n["news_items"] = upsert(schema.news_items, [{
            "id": i["id"], "source": i["source"], "published_utc": i["published_utc"], "title": i["title"] or "(untitled)",
            "url": i["url"], "summary": i.get("summary") or None} for i in items])
        doc_type = {"fed_monetary": "statement", "fed_all": "press", "fed_speeches": "speech", "fed_testimony": "testimony"}
        fed_docs = [{"id": i["id"], "doc_type": doc_type[i["source"]], "published_utc": i["published_utc"],
                     "title": i["title"] or "(untitled)", "url": i["url"]} for i in items if i["source"] in doc_type]
        if fed_docs:
            n["fed_documents"] = upsert(schema.fed_documents, fed_docs)
    return n


# ─────────────────────────────── rendering ───────────────────────────────

ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌"}


def render_markdown(series: dict[str, SeriesReport], probes: list[ProbeReport], started,
                    db_report: dict | None = None) -> str:
    s = get_settings()
    counts = pd.Series([r.status for r in series.values()]).value_counts().to_dict()
    pcounts = pd.Series([p.status for p in probes]).value_counts().to_dict()
    lines = [
        "# Data Availability Report",
        "",
        f"Generated {started:%Y-%m-%d %H:%M} UTC · runner: {os.environ.get('RUNNER_NAME', 'local')} · "
        f"FRED_API_KEY: {'set' if s.fred_api_key else 'not set (keyless fredgraph.csv, no vintages)'}",
        "",
        f"**Series:** {len(series)} total — ✅ {counts.get('ok', 0)} · ⚠️ {counts.get('warn', 0)} · ❌ {counts.get('fail', 0)}  ",
        f"**Probes:** {len(probes)} total — ✅ {pcounts.get('ok', 0)} · ⚠️ {pcounts.get('warn', 0)} · ❌ {pcounts.get('fail', 0)}",
        "",
    ]
    by_cat: dict[str, list[SeriesReport]] = {}
    for r in series.values():
        by_cat.setdefault(r.category, []).append(r)
    lines += ["## Series", ""]
    for cat, reps in by_cat.items():
        lines += [f"### {cat}", "", "| | key | source · id | freq | rows | first → last | last value | age | notes |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in reps:
            notes = []
            if r.proxy:
                notes.append(f"**PROXY** for {r.proxy_for}")
            if r.error:
                notes.append(f"`{r.error[:160]}`")
            notes += [f"{c['name']}: {c['message']}" for c in r.checks if c["status"] != "ok" or c["name"] == "cross_check"]
            if r.missing_values:
                notes.append(f"{r.missing_values} missing at source (dropped)")
            if r.vintages != "n/a":
                notes.append(f"vintages {r.vintages}")
            val = "" if r.last_value is None else f"{r.last_value:,.4g} {r.units}"
            lines.append(f"| {ICON[r.status]} | `{r.key}` | {r.source} · `{r.source_id}` | {r.frequency} | {r.rows} | "
                         f"{r.first_date or ''} → {r.last_date or ''} | {val} | {'' if r.age_days is None else f'{r.age_days}d'} | "
                         f"{'<br>'.join(notes).replace('|', '/')} |")
        lines.append("")
    lines += ["## Non-series sources (probes)", "", "| | group | probe | detail |", "|---|---|---|---|"]
    for p in probes:
        name = p.name + (" **(PROXY)**" if p.proxy else "")
        lines.append(f"| {ICON[p.status]} | {p.group} | {name} | {p.detail.replace('|', '/')[:500]} |")
    lines.append("")
    lines += ["## Database", ""]
    if db_report is None:
        lines.append("Not stored (run without `--store`).")
    else:
        lines.append(f"{'❌ **Storage failed:** `' + db_report['error'] + '`' if db_report.get('error') else '✅ Stored'} "
                     f"· target: {db_report.get('target')} · migrations applied this run: "
                     f"{db_report.get('migrations_applied') or 'none (up to date)'}")
        lines.append("")
        if db_report.get("readback"):
            lines.append("Read-back of latest values: " + " · ".join(db_report["readback"]))
            lines.append("")
        if db_report.get("counts"):
            lines += ["| table | rows written this run | total rows in DB |", "|---|---|---|"]
            for t, c in db_report["counts"].items():
                lines.append(f"| {t} | {db_report['stored'].get(t, '')} | {c:,} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true", help="upsert fetched data into DATABASE_URL")
    ap.add_argument("--only", nargs="*", help="limit to these series keys")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--cache-dir", help="also write fetched data to this directory (input for jobs.features --from-cache)")
    args = ap.parse_args(argv)

    started = utcnow()
    registry = load_registry()
    if args.only:
        registry = {k: v for k, v in registry.items() if k in set(args.only)}

    series, results = run_series(registry)
    vintages = run_vintages(registry, series)
    probes, stash = ([], {}) if args.no_probes else run_probes(results)
    stash["_vintages"] = vintages

    if args.cache_dir:
        write_cache(Path(args.cache_dir), results, vintages, stash)
    db_report = store_all(registry, series, results, stash, started) if args.store else None

    OUT_DIR.mkdir(exist_ok=True)
    md = render_markdown(series, probes, started, db_report)
    (OUT_DIR / "data_availability.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "data_availability.json").write_text(json.dumps(
        {"generated_at": started.isoformat(), "series": [asdict(r) for r in series.values()],
         "probes": [asdict(p) for p in probes]}, indent=1, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 1 if db_report and db_report.get("error") else 0


def write_cache(d: Path, results: dict[str, SeriesResult], vintages: dict[str, pd.DataFrame], stash: dict) -> None:
    """Plain CSV snapshot of one run (no database needed): what the features job reads with --from-cache."""
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([r.data.assign(series_key=k) for k, r in results.items() if not r.data.empty], ignore_index=True) \
        .to_csv(d / "observations.csv.gz", index=False)
    if vintages:
        pd.concat([v.assign(series_key=k) for k, v in vintages.items()], ignore_index=True) \
            .to_csv(d / "vintages.csv.gz", index=False)
    fut = [stash[k] for k in ("fut_ZQ", "fut_SR3") if stash.get(k) is not None and not stash[k].empty]
    if fut:
        pd.concat(fut, ignore_index=True).to_csv(d / "futures_quotes.csv", index=False)
    rel = next(r for r in calendar.tracked_releases() if r["id"] == "fomc")
    ev = list(stash.get("calendar_fred", [])) + [calendar._event(rel, m["decision_date"], "federalreserve.gov")
                                                 for m in stash.get("fomc", [])]
    pd.DataFrame([{"event_id": e["event_id"], "release_id": e["release_id"], "scheduled_utc": e["scheduled_utc"]}
                  for e in ev], columns=["event_id", "release_id", "scheduled_utc"]).to_csv(d / "calendar_events.csv", index=False)


def store_all(registry, series, results, stash, started) -> dict:
    """Persist everything fetched in this run, then read it back to prove it landed."""
    from db import schema
    from db.store import describe_target, engine, init_db, save_observations, save_series_meta, save_vintages, table_counts

    rep: dict = {"target": None, "stored": {}, "error": None}
    try:
        eng = engine()
        rep["target"] = describe_target(eng)
        rep["migrations_applied"] = init_db(eng)
        n_obs = sum(save_observations(k, r.data) for k, r in results.items())
        rep["stored"]["observations"] = n_obs
        save_series_meta(registry.values(), status={k: {
            "last_fetched_at": started,
            "last_obs_date": pd.Timestamp(r.last_date).date() if r.last_date else None,
            "source_last_updated": pd.Timestamp(r.source_last_updated).to_pydatetime() if r.source_last_updated else None,
            "last_status": r.status,
            "last_message": r.error or "; ".join(c["message"] for c in r.checks if c["status"] != "ok") or None,
        } for k, r in series.items()})
        rep["stored"]["series_meta"] = len(registry)
        n_vint = 0
        for key, v in stash.get("_vintages", {}).items():
            n_vint += save_vintages(key, v)
        rep["stored"]["observation_vintages"] = n_vint
        rep["stored"].update(_store_probe_data(stash))
        rep["counts"] = table_counts(eng)
        # Read-back: the latest stored value must equal what was just fetched.
        checks = []
        for key in ("ust_10y", "cpi", "fed_balance_sheet", "gold_futures", "btc_usd_daily"):
            if key in results and not results[key].data.empty:
                from db.store import load_series
                db = load_series(key, eng)
                want = results[key].data.iloc[-1]
                got = db.iloc[-1] if not db.empty else None
                okv = got is not None and got["date"] == want["date"] and abs(got["value"] - want["value"]) < 1e-9
                checks.append(f"{'✅' if okv else '❌'} {key} {want['date']:%Y-%m-%d}={want['value']:g}"
                              + ("" if okv else f" (db: {None if got is None else got.to_dict()})"))
        rep["readback"] = checks
        if any(c.startswith("❌") for c in checks):
            rep["error"] = "read-back mismatch"
    except Exception as exc:  # noqa: BLE001 — reported in the summary and fails the job
        from db.store import redact
        rep["error"] = redact(f"{exc.__class__.__name__}: {str(exc)[:500]}")
        target = rep.get("target") or ""
        if ".supabase.co" in target and target.split("@ ")[-1].startswith("db."):
            rep["error"] += (" — hint: Supabase's direct host (db.<ref>.supabase.co) is IPv6-only and GitHub runners "
                             "have no IPv6; use the Session pooler connection string (…pooler.supabase.com:5432).")
        print(redact(traceback.format_exc()), file=sys.stderr)
    return rep


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover — never let a traceback print credentials
        from db.store import redact
        print(redact(traceback.format_exc()), file=sys.stderr)
        sys.exit(1)
