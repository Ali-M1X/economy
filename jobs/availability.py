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
        checks = validate_series(df, meta, today)
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
        rep.age_days = staleness_days(df, today)
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


def run_vintages(registry: dict[str, SeriesMeta], reports: dict[str, SeriesReport], store: bool) -> None:
    has_key = bool(get_settings().fred_api_key)
    for key, meta in registry.items():
        if not meta.vintages:
            continue
        if not has_key:
            reports[key].vintages = "skipped (FRED_API_KEY not set)"
            continue
        try:
            v = fred.fetch_vintages(meta)
            reports[key].vintages = f"ok: {len(v)} vintage rows, {v['realtime_start'].nunique()} release dates"
            if store:
                from db.store import save_vintages
                save_vintages(key, v)
        except Exception as exc:  # noqa: BLE001
            reports[key].vintages = f"fail: {exc}"[:300]


def _probe(name: str, group: str, fn, proxy: bool = False) -> ProbeReport:
    try:
        status, detail = fn()
    except Exception as exc:  # noqa: BLE001
        status, detail = "fail", f"{exc.__class__.__name__}: {exc}"[:400]
    return ProbeReport(name, group, status, detail, proxy)


def run_probes(store: bool) -> list[ProbeReport]:
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
            probes.append(_probe(f"venue {venue} {asset} 1h", "prices", g, proxy=asset == "PAXG"))

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
            if store:
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
        return ("ok" if len(d) >= 2 else "warn"), "; ".join(parts) + (f"; failed {list(errs)}: " + "; ".join(errs.values())[:200] if errs else "")
    probes.append(_probe("open interest & funding BTC", "derivatives", deriv))

    def oi_hist():
        df = crypto.open_interest_history("BTC", "1h")
        return "ok", f"bybit OI history: {len(df)} hourly points, last {df['ts'].iloc[-1]:%Y-%m-%d %H:%M}Z"
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
        return ("ok" if not errs else "warn"), f"{len(items)} items, latest {latest}" + (f"; failed {errs}" if errs else "")
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
    probes.append(_probe("BLS release calendar (ICS)", "calendar", bls_ics))

    def fred_dates():
        if not get_settings().fred_api_key:
            return "warn", "skipped: FRED_API_KEY not set"
        today = utcnow().date()
        ev = calendar.from_fred(today, today + pd.Timedelta(days=30))
        return "ok", f"{len(ev)} tracked releases in next 30d: " + ", ".join(f"{e['release_id']} {e['scheduled_utc']:%m-%d}" for e in ev[:8])
    probes.append(_probe("FRED release dates", "calendar", fred_dates))

    # News
    def agency_feeds():
        items, errs = news.collect()
        by_src = pd.Series([i["source"].split(":")[0] for i in items]).value_counts().to_dict() if items else {}
        return ("ok" if not errs else "warn"), f"{len(items)} unique items {by_src}" + (f"; failed: {errs}" if errs else "")
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

    if store:
        _store_probe_data(stash)
    return probes


def _store_probe_data(stash: dict) -> None:
    from db import schema
    from db.store import upsert

    now = utcnow()
    for (asset, iv), r in [(k, v) for k, v in stash.items() if isinstance(k, tuple)]:
        rows = [{"asset": asset, "venue": r.venue, "interval": iv, "ts": t.to_pydatetime(), "open": o, "high": h,
                 "low": lo, "close": c, "volume": v, "is_proxy": asset == "PAXG"}
                for t, o, h, lo, c, v in r.data[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)]
        upsert(schema.candles, rows)
    for asset in ("BTC", "PAXG"):
        if f"book_{asset}" in stash:
            books, depth = stash[f"book_{asset}"]
            upsert(schema.orderbook_snapshots, [{"asset": asset, "ts": now, "bucket_size": 100.0 if asset == "BTC" else 5.0,
                                                 "mid_price": None, "venues": sorted(books),
                                                 "depth": depth.to_dict(orient="records")}])
    for venue, d in stash.get("deriv", {}).items():
        upsert(schema.derivatives_snapshots, [{"asset": "BTC", "venue": venue, "ts": now, **{
            k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}}])
    liq = stash.get("liq_okx")
    if liq is not None and not liq.empty:
        upsert(schema.liquidations, [{"raw_id": r.raw_id, "venue": "okx", "asset": r.asset, "ts": r.ts.to_pydatetime(),
                                      "side": r.side, "price": r.price, "qty": r.qty_contracts, "usd": None}
                                     for r in liq.itertuples()])
    for r in stash.get("liq_bybit", []):
        upsert(schema.liquidations, [{**r, "ts": r["ts"].to_pydatetime(), "usd": r["price"] * r["qty"]}])
    for root in ("ZQ", "SR3"):
        df = stash.get(f"fut_{root}")
        if df is not None and not df.empty:
            upsert(schema.futures_quotes, [{**r, "quote_ts": r["quote_ts"].to_pydatetime()} for r in df.to_dict("records")])


# ─────────────────────────────── rendering ───────────────────────────────

ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌"}


def render_markdown(series: dict[str, SeriesReport], probes: list[ProbeReport], started) -> str:
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true", help="upsert fetched data into DATABASE_URL")
    ap.add_argument("--only", nargs="*", help="limit to these series keys")
    ap.add_argument("--no-probes", action="store_true")
    args = ap.parse_args(argv)

    started = utcnow()
    registry = load_registry()
    if args.only:
        registry = {k: v for k, v in registry.items() if k in set(args.only)}

    if args.store:
        from db.store import init_db
        init_db()

    series, results = run_series(registry)
    run_vintages(registry, series, args.store)
    if args.store:
        from db.store import save_observations, save_series_meta
        for key, res in results.items():
            save_observations(key, res.data)
        save_series_meta(registry.values(), status={k: {
            "last_fetched_at": started,
            "last_obs_date": pd.Timestamp(r.last_date).date() if r.last_date else None,
            "source_last_updated": pd.Timestamp(r.source_last_updated).to_pydatetime() if r.source_last_updated else None,
            "last_status": r.status,
            "last_message": r.error or "; ".join(c["message"] for c in r.checks if c["status"] != "ok") or None,
        } for k, r in series.items()})

    probes = [] if args.no_probes else run_probes(args.store)

    OUT_DIR.mkdir(exist_ok=True)
    md = render_markdown(series, probes, started)
    (OUT_DIR / "data_availability.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "data_availability.json").write_text(json.dumps(
        {"generated_at": started.isoformat(), "series": [asdict(r) for r in series.values()],
         "probes": [asdict(p) for p in probes]}, indent=1, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover
        traceback.print_exc()
        sys.exit(1)
