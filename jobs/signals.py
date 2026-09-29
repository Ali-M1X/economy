"""Phase 4: technical analysis, liquidity levels, walk-forward backtest and live trade signals.

    python -m jobs.signals --from-cache output/cache      # reads output/macro_scores.json and features_state.json

A signal is emitted only when (1) the setup rules fire on the latest closed 4h bars, (2) the medium-horizon
Macro Score points the same way with |score| ≥ MACRO_THRESHOLD, (3) blended R:R ≥ 1.5 and (4) the same setup
has a demonstrated edge: out-of-sample average R > 0 over ≥ MIN_EDGE_TRADES walk-forward trades for that
asset and side. Otherwise the setup is listed on a watchlist with the reasons it was blocked.
This is analysis, not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from core.settings import ROOT
from core.timeutil import TEHRAN, utcnow
from signals import engine
from technicals.indicators import atr, macd, resample, rsi, trend
from technicals.levels import liquidation_clusters, orderbook_walls, volume_profile
from technicals.structure import structure

OUT_DIR = ROOT / "output"
MACRO_THRESHOLD = 15.0
MIN_EDGE_TRADES = 30  # a live signal also needs that side's out-of-sample average R > 0 over ≥ 30 trades
DISCLAIMER_FA = "این تحلیل صرفاً جنبه آموزشی و اطلاعاتی دارد و توصیه مالی یا پیشنهاد خرید و فروش نیست."
DISCLAIMER_EN = "This is analysis, not financial advice."
ASSETS = {"BTC": {"file": "BTC", "bucket": 250.0, "proxy": None},
          "Gold": {"file": "PAXG", "bucket": 5.0, "proxy": "PAXG (tokenized gold, 1 token = 1 troy oz) — real-time proxy for XAUUSD"}}


def load_bars(cache: Path, name: str) -> pd.DataFrame | None:
    p = cache / f"candles_{name}_15m.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def technical_state(bars: pd.DataFrame) -> dict:
    """Multi-timeframe trend, structure, momentum and volatility at the latest closed bars."""
    out = {}
    for tf in ("1d", "4h", "1h"):
        df = resample(bars, tf)
        tr = trend(df).iloc[-1]
        st = structure(df).iloc[-1]
        m = macd(df["close"]).iloc[-1]
        out[tf] = {"close": float(df["close"].iloc[-1]), "bar": df["ts"].iloc[-1].isoformat(),
                   "trend": {1: "up", -1: "down", 0: "mixed"}.get(int(tr) if pd.notna(tr) else 0),
                   "structure": {1: "bullish (HH/HL)", -1: "bearish (LH/LL)", 0: "range"}[int(st["state"])],
                   "last_swing_high": None if pd.isna(st["sh"]) else float(st["sh"]),
                   "last_swing_low": None if pd.isna(st["sl"]) else float(st["sl"]),
                   "last_event": next((e for e in structure(df)["event"].iloc[::-1] if e), None),
                   "atr": float(atr(df).iloc[-1]), "rsi": float(rsi(df["close"]).iloc[-1]),
                   "macd_hist": float(m["hist"])}
    return out


def context_levels(bars: pd.DataFrame, cache: Path, cfg: dict, price: float) -> dict:
    """Display-only liquidity context (not used by the backtested rules)."""
    last30 = resample(bars[bars["ts"] >= bars["ts"].iloc[-1] - pd.Timedelta(days=30)], "1h")
    vp = volume_profile(last30)
    ctx = {"volume_profile_30d": {k: (round(v, 2) if isinstance(v, float) else [round(x, 2) for x in v])
                                  for k, v in vp.items() if k != "profile"}}
    ob = cache / f"orderbook_{cfg['file']}.csv"
    if ob.exists():
        walls = orderbook_walls(pd.read_csv(ob), price)
        ctx["orderbook_walls"] = [{"side": r.side, "price": float(r.bucket), "usd_m": round(r.usd / 1e6, 2)}
                                  for r in walls.head(8).itertuples()]
    oi = cache / f"oi_history_{cfg['file']}.csv"
    if oi.exists():
        o = pd.read_csv(oi, parse_dates=["ts"])
        o["ts"] = pd.to_datetime(o["ts"], utc=True)
        px = resample(bars, "1h").set_index("ts")["close"]
        lc = liquidation_clusters(o, px, cfg["bucket"] * 4)
        ctx["liquidation_clusters_estimate"] = [{"side": r.side, "price": float(r.bucket), "usd_m": round(r.usd / 1e6, 1)}
                                                for r in lc.head(8).itertuples()]
    return ctx


def expected_range(bars: pd.DataFrame, macro: float | None) -> dict:
    """Probable ranges: 1 day from daily ATR; 4 weeks from realized volatility, centred on a macro drift of
    (score/100) × 0.5σ (the Macro Score shifts the centre, never the width)."""
    d1 = resample(bars, "1d")
    price = float(bars["close"].iloc[-1])  # centre on the latest price, not the last daily close
    a = float(atr(d1).iloc[-1])
    ret = np.log(d1["close"]).diff().dropna().tail(90)
    sig = float(ret.std() * np.sqrt(20))
    drift = (macro or 0) / 100 * 0.5 * sig
    return {"1d": [round(price - a, 2), round(price + a, 2)],
            "4w_1sigma": [round(float(price * np.exp(drift - sig)), 2), round(float(price * np.exp(drift + sig)), 2)],
            "4w_sigma_pct": round(sig * 100, 1), "macro_drift_pct": round(drift * 100, 2)}


def risk_section(asset: str, now: pd.Timestamp, cache: Path, state: dict, macro: dict, direction: int | None) -> list[str]:
    risks = []
    cal = cache / "calendar_events.csv"
    if cal.exists():
        ev = pd.read_csv(cal)
        ev["scheduled_utc"] = pd.to_datetime(ev["scheduled_utc"], utc=True)
        soon = ev[(ev["scheduled_utc"] > now) & (ev["scheduled_utc"] <= now + pd.Timedelta(hours=72))].sort_values("scheduled_utc")
        for e in soon.drop_duplicates("event_id").itertuples():
            name = getattr(e, "name_en", e.release_id)
            risks.append(f"Upcoming: {name} at {e.scheduled_utc.tz_convert(TEHRAN):%a %Y-%m-%d %H:%M} Tehran — volatility spike likely")
    reg = (state or {}).get("regime") or {}
    if reg.get("probabilities"):
        p = max(reg["probabilities"].values())
        if p < 0.6:
            risks.append(f"Regime uncertain: {reg['regime']} at only {p:.0%} — switch risk {1 - p:.0%}")
    top = (macro or {}).get("medium", {}).get("top", [])
    if direction:
        against = [t for t in top if t["contribution"] * direction < 0]
        if against:
            risks.append("Conflicting indicators: " + ", ".join(f"{t['indicator']} ({t['contribution']:+.1f})" for t in against[:3]))
    wd, hr = now.weekday(), now.hour
    if asset == "BTC" and wd >= 5:
        risks.append("Weekend: thinner BTC liquidity, gaps into Monday's CME open are common")
    if asset == "Gold" and (wd == 5 or (wd == 6 and hr < 22) or (wd == 4 and hr >= 21)):
        risks.append("Gold futures market closed: PAXG trades 24/7 but thinly; spot may gap on reopen")
    if 21 <= hr or hr < 1:
        risks.append("Low-liquidity window (US close → Asia open, ~21:00–01:00 UTC)")
    return risks


def analyse_asset(asset: str, cfg: dict, cache: Path, macro_all: dict, state: dict, now: pd.Timestamp) -> dict:
    bars = load_bars(cache, cfg["file"])
    if bars is None:
        return {"asset": asset, "error": "no price history in snapshot"}
    wf = engine.walk_forward(bars, asset)
    macro = macro_all.get(asset, {})
    m_med = (macro.get("medium") or {}).get("score")
    tech = technical_state(bars)
    price = tech["1h"]["close"]
    live = engine.live_setups(bars, wf["latest_params"])
    signals, watch = [], []
    for s in live:
        stats = wf["summary"]["long" if s.direction > 0 else "short"]
        card = engine.signal_card(s, asset, stats, m_med, "days–2 weeks (4h setup)")
        macro_ok = m_med is not None and abs(m_med) >= MACRO_THRESHOLD and np.sign(m_med) == s.direction
        edge_ok = stats.get("n_trades", 0) >= MIN_EDGE_TRADES and (stats.get("avg_r") or -1) > 0
        card["macro_score_medium"] = m_med
        card["blocked_by"] = [x for x, ok in (("macro not aligned", macro_ok),
                                              (f"no demonstrated edge (OOS avg R {stats.get('avg_r')} over "
                                               f"{stats.get('n_trades', 0)} trades)", edge_ok)) if not ok]
        card["reasons_macro"] = [f"{t['indicator']} {t['contribution']:+.1f}" for t in (macro.get("medium") or {}).get("top", [])[:4]]
        card["risks"] = risk_section(asset, now, cache, state, macro, s.direction)
        (signals if macro_ok and edge_ok else watch).append(card)
    return {
        "asset": asset, "proxy": cfg["proxy"], "price": price, "technical": tech,
        "context": context_levels(bars, cache, cfg, price), "expected_range": expected_range(bars, m_med),
        "macro": {b: (macro.get(b) or {}).get("score") for b in ("short", "medium", "long")},
        "backtest": {"period": wf["period"], "costs": wf["costs"], "summary": wf["summary"],
                     "latest_params": asdict(wf["latest_params"]), "folds": len(wf["folds"])},
        "signals": signals, "watchlist": watch,
        "risks": risk_section(asset, now, cache, state, macro, None),
    }


def render(res: list[dict], now: pd.Timestamp) -> str:
    L = ["# Phase 4 — Technicals & trade signals", "", f"As of {now:%Y-%m-%d %H:%M} UTC · {now.tz_convert(TEHRAN):%H:%M} Tehran", "",
         f"> **{DISCLAIMER_EN}** {DISCLAIMER_FA}", "",
         f"Rules: `signals/rules.py` (trend pullback on 4h, daily trend filter). Signals need |medium Macro Score| ≥ "
         f"{MACRO_THRESHOLD:g} in the same direction, blended R:R ≥ 1.5 and a demonstrated out-of-sample edge "
         f"(avg R > 0 over ≥ {MIN_EDGE_TRADES} trades). Win rates are walk-forward out-of-sample with fees & slippage, "
         "for the technical setup (the macro filter is not in the backtest).", ""]
    for r in res:
        if "error" in r:
            L += [f"## {r['asset']}", "", r["error"], ""]
            continue
        L += [f"## {r['asset']} — {r['price']:,.2f}" + (f"  _(proxy: {r['proxy']})_" if r["proxy"] else ""), ""]
        L += ["| TF | trend | structure | last event | RSI | MACD hist | ATR |", "|---|---|---|---|---|---|---|"]
        for tf, t in r["technical"].items():
            L.append(f"| {tf} | {t['trend']} | {t['structure']} | {t['last_event'] or '—'} | {t['rsi']:.0f} | {t['macd_hist']:+.2f} | {t['atr']:,.2f} |")
        er = r["expected_range"]
        L += ["", f"Macro Score short/medium/long: {r['macro']} · expected range 1d {er['1d']}, 4w (1σ) {er['4w_1sigma']} "
                  f"(σ {er['4w_sigma_pct']}%, macro drift {er['macro_drift_pct']:+}%)", ""]
        c = r["context"]
        vp = c.get("volume_profile_30d", {})
        if vp:
            L.append(f"Volume profile (30d): POC {vp.get('poc')}, VAH {vp.get('vah')}, VAL {vp.get('val')}, "
                     f"HVN {vp.get('hvn', [])[:4]}, LVN {vp.get('lvn', [])[:4]}")
        if c.get("orderbook_walls"):
            L.append("Order-book walls (±5%): " + ", ".join(f"{w['side']} {w['price']:,.0f} (${w['usd_m']}M)" for w in c["orderbook_walls"][:6]))
        if c.get("liquidation_clusters_estimate"):
            L.append("Estimated liquidation clusters (heuristic from OKX OI changes × leverage tiers): " + ", ".join(
                f"{w['side']} {w['price']:,.0f} (${w['usd_m']}M)" for w in c["liquidation_clusters_estimate"][:6]))
        b = r["backtest"]
        L += ["", f"### Backtest (walk-forward OOS, {b['period'][0]} → {b['period'][1]}, {b['folds']} folds, "
                  f"fees {b['costs']['fee_bps']}bp/side + slippage {b['costs']['slippage_bps']}bp)", "",
              "| side | trades | win rate | avg R | total R | max DD (R) | PF | TP1 / TP2 / TP3 hit | small sample |",
              "|---|---|---|---|---|---|---|---|---|"]
        for side in ("all", "long", "short"):
            s = b["summary"][side]
            if not s.get("n_trades"):
                L.append(f"| {side} | 0 | | | | | | | |")
                continue
            L.append(f"| {side} | {s['n_trades']} | {s['win_rate']:.0%} | {s['avg_r']:+.2f} | {s['total_r']:+.1f} | {s['max_drawdown_r']} | "
                     f"{s['profit_factor']} | {s['tp1_rate']:.0%} / {s['tp2_rate']:.0%} / {s['tp3_rate']:.0%} | {'yes' if s['small_sample'] else 'no'} |")
        L += ["", f"Current parameters (chosen on the last 24 months): {b['latest_params']}", ""]
        for title, items in (("Signals", r["signals"]), ("Watchlist (setup fired, but blocked)", r["watchlist"])):
            L += [f"### {title}", ""]
            if not items:
                L.append("none")
            for s in items:
                wr = "n/a" if s["win_rate"] is None else f"{s['win_rate']:.0%}"
                L += [f"- **{s['direction'].upper()}** entry {s['entry_zone']} · SL {s['stop_loss']:,} · TPs {s['targets']} "
                      f"({', '.join(s['target_kinds'])}) · R:R {s['rr']} (blended {s['blended_rr']}) · win rate "
                      f"{wr} over {s['n_backtest']} OOS trades"
                      f"{' (small sample)' if s['small_sample'] else ''} · confidence {s['confidence']}",
                      *([f"  - blocked: {'; '.join(s['blocked_by'])}"] if s.get("blocked_by") else []),
                      f"  - invalidation: {s['invalidation']}", f"  - technical: {'; '.join(s['reasons_technical'])}",
                      f"  - macro (medium score {s['macro_score_medium']}): {', '.join(s['reasons_macro'])}",
                      f"  - risks: {'; '.join(s['risks']) or 'none flagged'}"]
            L.append("")
        L += ["### Risks now", ""] + [f"- {x}" for x in r["risks"] or ["none flagged"]] + [""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", required=True)
    args = ap.parse_args(argv)
    cache = Path(args.from_cache)
    now = pd.Timestamp(utcnow())
    read = lambda p: json.loads(p.read_text()) if p.exists() else {}  # noqa: E731
    macro, state = read(OUT_DIR / "macro_scores.json"), read(OUT_DIR / "features_state.json")
    res = [analyse_asset(a, cfg, cache, macro, state, now) for a, cfg in ASSETS.items()]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "signals.json").write_text(json.dumps({"as_of": now.isoformat(), "disclaimer": [DISCLAIMER_EN, DISCLAIMER_FA],
                                                      "assets": res}, default=str, indent=1), encoding="utf-8")
    md = render(res, now)
    (OUT_DIR / "signals.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n\n" + md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover
        traceback.print_exc()
        sys.exit(1)
