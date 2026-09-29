"""Pre-positioning (front-run) signals: walk-forward backtest per release × asset × lead window, the nested
out-of-sample window choice with its edge verdict, and the live check of upcoming releases.

    python -m jobs.prepos --from-cache output/cache [--archive state/archive] [--state state] [--refresh-live]

Target releases: CPI, PCE, NFP and FOMC decisions. Assets: BTC and gold (PAXG 15-minute bars).
Writes output/prepos.json (dashboard + Telegram), output/prepos_backtest.csv, output/prepos.md.
`--archive` keeps the Binance futures archive (open interest, funding) between runs; `--refresh-live` fetches
fresh 15-minute bars, OKX open interest / funding, fed funds futures and order books (the hourly job), and appends
futures / order-book snapshots to `--state` so their drift can be shown live.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import load_registry
from core.settings import OUTPUT_DIR
from features import pit
from models import events as ev_mod
from models import prepos as pp

TARGETS = {  # release id → (series with vintages for the release dates, lead surprises: (key, measure, release id))
    "cpi": ("cpi", []),
    "nfp": ("nonfarm_payrolls", [("adp_private_payrolls", "diff", "adp")]),
    "pce": ("pce", [("core_cpi", "mom_pct", "cpi"), ("ppi_final_demand", "mom_pct", "ppi")]),
    "fomc": (None, []),
}
ASSETS = {"BTC": "BTC", "Gold": "PAXG"}
LIVE_HORIZON = pd.Timedelta(hours=max(pp.WINDOWS_H))


# ───────────────────────────── inputs ─────────────────────────────

def release_dates(vintages: pd.DataFrame, release_id: str) -> pd.DataFrame:
    """obs_date, release_utc per first publication (the bulk first vintage is not a release)."""
    v = vintages.sort_values(["realtime_start", "date"])
    first = v.groupby("date")["realtime_start"].min()
    bulk = v["realtime_start"].min()
    rows = [{"obs_date": obs.index.max(), "release_utc": ev_mod.release_time_utc(release_id, R)}
            for R, obs in first.groupby(first) if R != bulk]
    return pd.DataFrame(rows, columns=["obs_date", "release_utc"])


def target_events(rel: str, vintages: dict, cache: Path, now: pd.Timestamp) -> pd.DataFrame:
    key = TARGETS[rel][0]
    if rel == "fomc":
        p = cache / "fomc_meetings.csv"
        if not p.exists():
            return pd.DataFrame(columns=["event_id", "release_utc", "obs_date"])
        m = pd.read_csv(p, parse_dates=["decision_date"])
        df = pd.DataFrame({"release_utc": pd.to_datetime(m["statement_utc"], utc=True), "obs_date": m["decision_date"]})
    else:
        if key not in vintages:
            return pd.DataFrame(columns=["event_id", "release_utc", "obs_date"])
        df = release_dates(vintages[key], rel)
    df["release_utc"] = pd.to_datetime(df["release_utc"], utc=True)
    df["event_id"] = [f"{rel}-{t:%Y-%m-%d}" for t in df["release_utc"]]
    return df.sort_values("release_utc").reset_index(drop=True)


def lead_frame(rel: str, targets: pd.DataFrame, vintages: dict) -> pd.DataFrame:
    """Surprises of earlier releases about the same month, matched to each target event by observation month."""
    rows = []
    for key, measure, rid in TARGETS[rel][1]:
        if key not in vintages:
            continue
        s = ev_mod.surprises(vintages[key], key, measure, "mean3", rid)
        if s.empty:
            continue
        s["month"] = pd.to_datetime(s["obs_date"]).dt.to_period("M")
        by_month = s.drop_duplicates("month", keep="first").set_index("month")
        for e in targets.itertuples():
            o = pd.Timestamp(e.obs_date)
            m = (o.tz_localize(None) if o.tzinfo else o).to_period("M")
            if m in by_month.index:
                r = by_month.loc[m]
                avail = pd.Timestamp(r["release_utc"])
                avail = avail.tz_localize("UTC") if avail.tzinfo is None else avail
                if avail < e.release_utc:
                    rows.append({"event_id": e.event_id, "available_at": avail, "z": float(r["z"]), "source": key})
    return pd.DataFrame(rows, columns=["event_id", "available_at", "z", "source"])


def rate_frame(data: dict) -> pd.DataFrame | None:
    """2-year Treasury yield as known at each time: FRED daily value, usable from 20:00 UTC of its release-lag day."""
    reg = load_registry()
    if "ust_2y" not in data:
        return None
    k = pit.known_series(data["ust_2y"], reg["ust_2y"])
    return pd.DataFrame({"available_at": pd.to_datetime(k["available_at"]).dt.tz_localize("UTC") + pd.Timedelta(hours=20),
                         "value": k["value"].to_numpy()}).dropna()


def load_bars(cache: Path, sym: str) -> pd.DataFrame | None:
    p = cache / f"candles_{sym}_15m.csv.gz"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    return b.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


# ───────────────────────────── live context (not part of the backtested score) ─────────────────────────────

def book_imbalance(depth: pd.DataFrame, band: float = 0.01) -> float | None:
    """(bid − ask) USD depth within ±band of mid, ÷ total — a live-only snapshot (no free order-book history)."""
    if depth is None or depth.empty:
        return None
    bids, asks = depth[depth["side"] == "bid"], depth[depth["side"] == "ask"]
    if bids.empty or asks.empty:
        return None
    mid = (bids["bucket"].max() + asks["bucket"].min()) / 2
    b = bids[bids["bucket"] >= mid * (1 - band)]["usd"].sum()
    a = asks[asks["bucket"] <= mid * (1 + band)]["usd"].sum()
    return round(float((b - a) / (b + a)), 3) if a + b > 0 else None


def append_state(state: Path | None, name: str, row: dict) -> pd.DataFrame:
    if state is None:
        return pd.DataFrame([row])
    state.mkdir(parents=True, exist_ok=True)
    p = state / name
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).tail(2000)
    df.to_csv(p, index=False)
    return df


def drift_since(hist: pd.DataFrame, col: str, since: pd.Timestamp) -> float | None:
    if hist is None or hist.empty or col not in hist:
        return None
    h = hist.assign(ts=pd.to_datetime(hist["ts"], utc=True)).dropna(subset=[col]).sort_values("ts")
    before = h[h["ts"] <= since]
    if before.empty or h.empty:
        return None
    return float(h[col].iloc[-1] - before[col].iloc[-1])


def refresh_live(cache: Path, state: Path | None, now: pd.Timestamp) -> dict:
    """Hourly job: fresh bars, OKX OI/funding, fed funds futures and order books."""
    from collectors import crypto, futures

    ctx, errors = {}, {}
    for sym in ("BTC", "PAXG"):
        try:
            new = crypto.candles_history(sym, "15m", start=str((now - pd.Timedelta(days=6)).date()), max_requests=5)
            old = load_bars(cache, sym)
            allb = pd.concat([old, new], ignore_index=True) if old is not None else new
            allb.drop_duplicates("ts", keep="last").sort_values("ts").to_csv(cache / f"candles_{sym}_15m.csv.gz", index=False)
        except Exception as e:  # noqa: BLE001
            errors[f"bars {sym}"] = str(e)[:200]
    for name, fn in (("oi_history_BTC.csv", lambda: crypto.open_interest_history("BTC", "1h", 100)),
                     ("funding_history_BTC.csv", lambda: crypto.funding_history_okx("BTC"))):
        try:
            fn().to_csv(cache / name, index=False)
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e)[:200]
    try:
        zq, _ = futures.curve("ZQ", 4)
        zq.to_csv(cache / "futures_quotes.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors["ZQ"] = str(e)[:200]
    for sym, bucket in (("BTC", 100.0), ("PAXG", 5.0)):
        try:
            books, _ = crypto.order_books(sym)
            crypto.aggregate_depth(books, bucket).to_csv(cache / f"orderbook_{sym}.csv", index=False)
        except Exception as e:  # noqa: BLE001
            errors[f"book {sym}"] = str(e)[:200]
    ctx["errors"] = errors
    return ctx


def live_context(cache: Path, state: Path | None, now: pd.Timestamp) -> dict:
    """Snapshot + stored history of the live-only inputs (fed funds futures, order-book imbalance)."""
    out = {}
    fq = cache / "futures_quotes.csv"
    if fq.exists():
        q = pd.read_csv(fq)
        q = q[q.get("root", "ZQ") == "ZQ"] if "root" in q else q
        if not q.empty:
            q = q.sort_values("contract_month")
            front, nxt = q.iloc[0], q.iloc[min(1, len(q) - 1)]
            out["zq"] = append_state(state, "zq_history.csv", {"ts": now.isoformat(), "front": front["implied_rate"],
                                                               "next": nxt["implied_rate"], "contract": nxt["contract"]})
    for asset, sym in ASSETS.items():
        p = cache / f"orderbook_{sym}.csv"
        if p.exists():
            imb = book_imbalance(pd.read_csv(p))
            if imb is not None:
                out[f"book_{asset}"] = append_state(state, f"book_history_{sym}.csv", {"ts": now.isoformat(), "imbalance": imb})
    return out


def live_inputs(asset: str, cache: Path, rate, lead, archive_oi, archive_funding) -> pp.Inputs | None:
    bars = load_bars(cache, ASSETS[asset])
    if bars is None:
        return None
    oi, fund = (None, None)
    if asset == "BTC":
        oi, fund = archive_oi, archive_funding
        for name, attr in (("oi_history_BTC.csv", "oi"), ("funding_history_BTC.csv", "funding")):
            p = cache / name
            if p.exists():
                live = pd.read_csv(p)
                live["ts"] = pd.to_datetime(live["ts"], utc=True)
                if attr == "oi" and not live.empty:  # OKX hourly history for the live window
                    oi = live[["ts", "open_interest"]]
                elif attr == "funding" and not live.empty:
                    fund = pd.concat([archive_funding, live], ignore_index=True).drop_duplicates("ts").sort_values("ts") \
                        if archive_funding is not None and not archive_funding.empty else live
    return pp.Inputs(bars=bars, rate=rate, oi=oi, funding=fund, lead=lead)


# ───────────────────────────── main ─────────────────────────────

def run(cache: Path, archive: Path | None, state: Path | None, now: pd.Timestamp, refresh: bool = False,
        releases=tuple(TARGETS), windows=pp.WINDOWS_H) -> dict:
    from jobs.features import load_cache

    data, vintages, _, cal, _ = load_cache(cache)
    notes = []
    metrics, funding = pd.DataFrame(), pd.DataFrame()
    if archive is not None:
        from collectors import binance_archive

        try:
            notes.append(f"Binance archive: {binance_archive.update(archive, today=now)}")
        except Exception as e:  # noqa: BLE001 — keep going with what is cached
            notes.append(f"Binance archive not updated: {str(e)[:200]}")
        metrics, funding = binance_archive.load(archive)
        if not metrics.empty:
            metrics["ts"] = metrics["ts"] + pd.Timedelta(minutes=5)  # a 5-min snapshot is known at the end of its interval
    if refresh:
        notes.append(f"live refresh: {refresh_live(cache, state, now)}")
    rate = rate_frame(data)
    result = {"as_of": now.isoformat(), "threshold": pp.THRESHOLD, "windows_h": list(windows), "notes": notes,
              "backtest": {}, "live": [], "context": {}}
    csv_rows = []
    for rel in releases:
        tev = target_events(rel, vintages, cache, now)
        if tev.empty:
            notes.append(f"{rel}: no event history in the snapshot")
            continue
        tev = tev[tev["release_utc"] + pp.HOLD <= now]
        lead = lead_frame(rel, tev, vintages) if not tev.empty else None
        for asset, sym in ASSETS.items():
            bars = load_bars(cache, sym)
            if bars is None or tev.empty:
                continue
            inp = pp.Inputs(bars=bars, rate=rate, oi=metrics if asset == "BTC" and not metrics.empty else None,
                            funding=funding if asset == "BTC" and not funding.empty else None, lead=lead)
            ev = tev[tev["release_utc"] > bars["ts"].iloc[0] + pp.BASELINE + pd.Timedelta(hours=max(windows))]
            rows = pp.prepare(inp, ev[["event_id", "release_utc"]], windows)
            wf = pp.walk_forward(rows, bars, windows=windows, tag=f"{rel}:{asset}:")
            ok, why = pp.edge(wf["selected"])
            model = pp.live_model(rows, now, windows)
            last_choice = wf["choices"][-1][1] if wf["choices"] else None
            result["backtest"][f"{rel}:{asset}"] = {
                "release": rel, "asset": asset, "events": len(rows),
                "period": [str(rows[0].T.date()), str(rows[-1].T.date())] if rows else None,
                "per_window": {str(W): s for W, s in wf["per_window"].items()}, "selected": wf["selected"],
                "window_choices": pd.Series([c[1] for c in wf["choices"]]).value_counts().to_dict() if wf["choices"] else {},
                "last_choice": last_choice, "edge": ok, "blocked_by": why, "model": model,
                "trades": wf["selected_trades"][-60:],
                "features_available": {k: int(sum(1 for r in rows if not r.paths[windows[0]].empty
                                                  and np.isfinite(r.paths[windows[0]][k]).any())) for k in pp.FEATURES},
            }
            for W, s in wf["per_window"].items():
                csv_rows.append({"release": rel, "asset": asset, "window_h": W, **s})
            csv_rows.append({"release": rel, "asset": asset, "window_h": "selected", **wf["selected"], "edge": ok})
    result["context"] = live_context(cache, state, now)
    result["live"] = live_checks(cache, vintages, rate, metrics, funding, result, now, windows)
    result["context"] = {k: (v.tail(1).to_dict("records")[0] if isinstance(v, pd.DataFrame) and not v.empty else v)
                         for k, v in result["context"].items()}
    result["_csv"] = csv_rows
    return result


def live_checks(cache: Path, vintages: dict, rate, metrics, funding, result: dict, now: pd.Timestamp, windows) -> list[dict]:
    """Upcoming target releases within the longest lead window: score inside the selected window, gated by the edge."""
    out = []
    cal_p = cache / "calendar_events.csv"
    up = []
    if cal_p.exists():
        cal = pd.read_csv(cal_p)
        cal["scheduled_utc"] = pd.to_datetime(cal["scheduled_utc"], utc=True)
        up = cal[(cal["release_id"].isin(TARGETS)) & (cal["scheduled_utc"] > now) &
                 (cal["scheduled_utc"] <= now + LIVE_HORIZON)].drop_duplicates("event_id").to_dict("records")
    ctx = result.get("context", {})
    for e in up:
        rel, T = e["release_id"], pd.Timestamp(e["scheduled_utc"])
        future_ev = pd.DataFrame([{"event_id": e["event_id"], "release_utc": T,
                                   "obs_date": (T - pd.DateOffset(months=1)).normalize()}])
        lead = lead_frame(rel, future_ev, vintages)
        for asset in ASSETS:
            bt = result["backtest"].get(f"{rel}:{asset}")
            row = {"event_id": e["event_id"], "release": rel, "name_fa": e.get("name_fa"), "asset": asset,
                   "release_utc": T.isoformat(), "status": "no_model"}
            out.append(row)
            if not bt or bt["model"]["selected_window"] is None:
                row["reason"] = "not enough history for this release/asset"
                continue
            W = int(bt["model"]["selected_window"])
            m = bt["model"]["windows"].get(W) or bt["model"]["windows"].get(str(W))  # int keys become str in JSON
            opens = T - pd.Timedelta(hours=W)
            row.update(window_h=W, window_opens=opens.isoformat(), weights=m["weights"])
            if now < opens:
                row["status"] = "waiting"
                continue
            inp = live_inputs(asset, cache, rate, lead, metrics if not metrics.empty else None,
                              funding if not funding.empty else None)
            path = pp.feature_path(inp, T, W, e["event_id"], until=now)
            sc = pp.score(path, m["weights"])
            if sc.empty:
                row.update(status="no_data", reason="no recent bars")
                continue
            last = path.iloc[-1]
            den = np.sqrt(max(sum(1 for k, v in m["weights"].items() if v and np.isfinite(last[k])), 1))
            row.update(score=round(float(sc.iloc[-1]), 2), score_time=sc.index[-1].isoformat(),
                       components={k: round(float(v * last[k] / den), 2) for k, v in m["weights"].items() if v and np.isfinite(last[k])},
                       features={k: (None if not np.isfinite(last[k]) else round(float(last[k]), 2)) for k in pp.FEATURES})
            cross = pp.first_crossing(sc, opens + pp.MIN_ELAPSED)
            if cross is None:
                row["status"] = "quiet"
                continue
            t, d, v = cross
            sig = pp.make_signal(inp.bars, sc.index[-1], d, T, m["vol_mult"], e["event_id"])
            row.update(status="signal" if bt["edge"] else "watchlist", direction="long" if d > 0 else "short",
                       triggered_at=t.isoformat(), trigger_score=round(v, 2), blocked_by=[] if bt["edge"] else bt["blocked_by"],
                       backtest=bt["selected"])
            if sig is not None:
                entry = float(inp.bars["close"].iloc[-1])
                row.update(entry=round(entry, 2), stop_loss=round(sig.stop, 2), targets=[round(x, 2) for x in sig.targets],
                           rr=list(pp.TP_R), exit_by=(T + pp.HOLD).isoformat())
            zq = ctx.get("zq")
            if isinstance(zq, pd.DataFrame):
                ch = drift_since(zq, "next", opens)
                row["zq_drift_bp"] = None if ch is None else round(ch * 100, 1)
            bk = ctx.get(f"book_{asset}")
            if isinstance(bk, pd.DataFrame):
                row["book_imbalance"] = float(bk["imbalance"].iloc[-1])
                ch = drift_since(bk, "imbalance", opens)
                row["book_imbalance_change"] = None if ch is None else round(ch, 3)
    return out


def live_only(cache: Path, model_json: Path, state: Path | None, now: pd.Timestamp, refresh: bool) -> dict:
    """Hourly: reuse the backtest/model of the last full run, refresh live inputs, re-check upcoming releases."""
    from jobs.features import load_cache

    res = json.loads(model_json.read_text(encoding="utf-8"))
    res["notes"] = [f"model from {res.get('as_of', '?')[:16]} UTC"]
    if refresh:
        res["notes"].append(f"live refresh: {refresh_live(cache, state, now)}")
    data, vintages, *_ = load_cache(cache)
    res["as_of_live"] = now.isoformat()
    res["context"] = live_context(cache, state, now)
    res["live"] = live_checks(cache, vintages, rate_frame(data), pd.DataFrame(), pd.DataFrame(), res, now, res.get("windows_h"))
    res["context"] = {k: (v.tail(1).to_dict("records")[0] if isinstance(v, pd.DataFrame) and not v.empty else v)
                      for k, v in res["context"].items()}
    return res


def render(res: dict) -> str:
    L = ["# Pre-positioning (front-run) signals — out-of-sample backtest", "",
         f"As of {res['as_of'][:16]} UTC. Signal = first 15-min step inside the lead window with |composite| ≥ "
         f"{res['threshold']}; market entry on the next bars, 1σ stop, targets 1R/1.75R/2.5R, exit by release + 24 h; "
         "fees 5 bp/side + 2 bp slippage. Orientation of every feature fitted on past events only; the lead window of "
         "each event is chosen from past events only (nested). Gate: ≥ 30 trades, average R > 0 and one-sided p < 0.05.", ""]
    L += ["| release | asset | events | selected-window trades | avg R | win | p | edge | why not |", "|---|---|---|---|---|---|---|---|---|"]
    for k, b in res["backtest"].items():
        s = b["selected"]
        L.append(f"| {b['release']} | {b['asset']} | {b['events']} | {s.get('n_trades', 0)} | {s.get('avg_r', '—')} | "
                 f"{s.get('win_rate', '—')} | {s.get('p_r', '—')} | {'✅' if b['edge'] else '—'} | {'; '.join(b['blocked_by'])} |")
    L += ["", "## Per lead window (each evaluated on its own, out of sample)", "",
          "| release | asset | window | trades | avg R | p(R) | Spearman ρ (score vs 24 h move) | p | strong-signal hit rate | p | effect (bp) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, b in res["backtest"].items():
        for W, s in b["per_window"].items():
            L.append(f"| {b['release']} | {b['asset']} | {W}h | {s.get('n_trades', 0)} | {s.get('avg_r', '—')} | {s.get('p_r', '—')} | "
                     f"{s.get('spearman', '—')} | {s.get('spearman_p', '—')} | {s.get('hit_rate', '—')} | {s.get('hit_p', '—')} | "
                     f"{s.get('effect_bp', '—')} |")
    L += ["", "Per-window p-values are not corrected for testing 5 windows (Bonferroni: multiply by 5); the gate uses only "
          "the nested selection, which is out of sample by construction.", "", "## Live", ""]
    for r in res["live"]:
        L.append(f"- {r['event_id']} {r['asset']}: {r['status']}" + (f" (score {r.get('score')}, window {r.get('window_h')}h)" if "score" in r else "")
                 + (f" — {r.get('reason')}" if r.get("reason") else ""))
    if res["notes"]:
        L += ["", "Notes: " + " · ".join(res["notes"])]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", required=True)
    ap.add_argument("--archive")
    ap.add_argument("--state")
    ap.add_argument("--refresh-live", action="store_true")
    ap.add_argument("--live-only", metavar="PREPOS_JSON", help="skip the backtest; reuse this model (hourly job)")
    args = ap.parse_args(argv)
    now = pd.Timestamp.now(tz="UTC")
    state = Path(args.state) if args.state else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.live_only:
        res = live_only(Path(args.from_cache), Path(args.live_only), state, now, args.refresh_live)
    else:
        res = run(Path(args.from_cache), Path(args.archive) if args.archive else None, state, now, args.refresh_live)
        pd.DataFrame(res.pop("_csv")).to_csv(OUTPUT_DIR / "prepos_backtest.csv", index=False)
    (OUTPUT_DIR / "prepos.json").write_text(json.dumps(res, default=str, indent=1), encoding="utf-8")
    md = render(res)
    (OUTPUT_DIR / "prepos.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n\n" + md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover
        from db.store import redact
        print(redact(traceback.format_exc()), file=sys.stderr)
        sys.exit(1)
