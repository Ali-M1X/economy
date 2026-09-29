"""Build a synthetic snapshot directory with the same layout as a real `output/` folder.

For tests and local UI development only — the folder gets a DEMO marker and the dashboard shows a
"demo data" banner. Values are random walks inside each series' registry range, NOT real data.

    python -m dashboard.demo_snapshot /tmp/demo_output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.registry import load_registry


def build(out: Path, seed: int = 3, now: pd.Timestamp | None = None) -> Path:
    rng = np.random.default_rng(seed)
    now = now or pd.Timestamp.now(tz="UTC").floor("h")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (out / "DEMO").write_text("synthetic data — not real\n")
    reg = load_registry()
    rows, avail = [], []
    end = now.tz_localize(None).normalize()
    for k, m in reg.items():
        if m.projection:
            idx = pd.date_range(f"{end.year}-01-01", periods=4, freq="YS")
        else:
            freq = {"D": "B", "W": "W-WED", "M": "MS", "Q": "QS", "A": "YS"}[m.frequency]
            idx = pd.date_range(max(pd.Timestamp(m.start), end - pd.DateOffset(years=30)), end, freq=freq)
        lo, hi = m.valid_min, m.valid_max
        mid, span = lo + 0.35 * (hi - lo), 0.05 * (hi - lo)
        walk = np.clip(mid + np.cumsum(rng.normal(0, span / np.sqrt(len(idx)), len(idx))) * 3, lo + 1e-9, hi - 1e-9)
        if k in ("fed_target_upper", "fed_target_lower", "iorb", "fed_funds_eff_daily", "nber_recession"):
            walk = np.round(walk * 4) / 4 if k != "nber_recession" else (rng.random(len(idx)) < 0.08).astype(float)
        rows.append(pd.DataFrame({"date": idx, "value": walk, "series_key": k}))
        avail.append({"key": k, "status": "ok", "source": m.source, "source_id": m.source_id, "last_date": f"{idx[-1]:%Y-%m-%d}",
                      "frequency": m.frequency, "proxy": m.proxy, "source_url": m.url})
    obs = pd.concat(rows, ignore_index=True)
    # keep the target range coherent: lower = upper − 0.25
    up = obs[obs.series_key == "fed_target_upper"].set_index("date")["value"]
    obs.loc[obs.series_key == "fed_target_lower", "value"] = up.reindex(obs.loc[obs.series_key == "fed_target_lower", "date"]).to_numpy() - 0.25
    obs.to_csv(cache / "observations.csv.gz", index=False)
    ts = pd.date_range(now - pd.Timedelta(days=200), now, freq="15min", tz="UTC")
    for name, p0 in (("BTC", 80000.0), ("PAXG", 4100.0)):
        c = p0 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))
        o = np.r_[c[0], c[:-1]]
        w = np.abs(rng.normal(0, 0.001, len(ts))) * c
        pd.DataFrame({"ts": ts, "open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w, "close": c,
                      "volume": rng.uniform(1, 50, len(ts))}).to_csv(cache / f"candles_{name}_15m.csv.gz", index=False)
    pd.DataFrame([{"contract": f"ZQ{d:%b%y}", "root": "ZQ", "contract_month": d.date(), "price": 100 - (3.9 + 0.08 * i),
                   "implied_rate": 3.9 + 0.08 * i, "quote_ts": now} for i, d in enumerate(pd.date_range(end.replace(day=1), periods=13, freq="MS"))]
                 ).to_csv(cache / "futures_quotes.csv", index=False)
    ev = [("fomc", "FOMC rate decision", "تصمیم نرخ بهره فدرال (FOMC)", 5, now + pd.Timedelta(days=9)),
          ("cpi", "Consumer Price Index", "شاخص قیمت مصرف‌کننده (CPI)", 5, now + pd.Timedelta(days=3)),
          ("claims", "Initial Jobless Claims", "مدعیان بیمه بیکاری", 3, now + pd.Timedelta(days=1))]
    pd.DataFrame([{"event_id": f"{r}-{t:%Y%m%d}", "release_id": r, "name_en": n, "name_fa": nf, "importance": i, "scheduled_utc": t}
                  for r, n, nf, i, t in ev]).to_csv(cache / "calendar_events.csv", index=False)
    pd.DataFrame([{"id": f"n{i}", "source": "fed_all", "published_utc": now - pd.Timedelta(hours=3 * i),
                   "title": f"Demo headline {i}", "url": "https://www.federalreserve.gov/"} for i in range(8)]).to_csv(cache / "news.csv", index=False)
    (out / "data_availability.json").write_text(json.dumps({"generated_at": now.isoformat(), "series": avail, "probes": []}))
    (out / "macro_scores.json").write_text(json.dumps({a: {b: {"score": None if s is None else float(s), "top": [], "note": None}
                                                           for b, s in zip(("short", "medium", "long"), (None, sc, -sc / 2))}
                                                       for a, sc in (("BTC", 18.0), ("Gold", -6.0))}))
    px = {"BTC": 80000.0, "Gold": 4100.0}
    assets = []
    for a in ("BTC", "Gold"):
        p = px[a]
        tf = {tf_: {"close": p, "bar": now.isoformat(), "trend": "up", "structure": "bullish (HH/HL)", "last_event": "BOS_UP",
                    "last_swing_high": p * 1.03, "last_swing_low": p * 0.97, "atr": p * 0.01, "rsi": 55.0, "macd_hist": 1.0}
              for tf_ in ("1d", "4h", "1h")}
        wl = [{"asset": a, "direction": "long", "horizon": "days", "signal_time": now.isoformat(), "entry_zone": [p * 0.998, p],
               "stop_loss": p * 0.98, "targets": [p * 1.02, p * 1.035, p * 1.05], "target_kinds": ["swing high", "swing high", "4R"],
               "rr": [1.0, 1.75, 2.5], "blended_rr": 1.75, "win_rate": 0.4, "avg_r": -0.05, "n_backtest": 120,
               "max_drawdown_r": 12.0, "small_sample": False, "confidence": "low", "blocked_by": ["no demonstrated edge (demo)"],
               "invalidation": "4h close below stop", "reasons_technical": ["demo"], "risks": ["demo risk"],
               "macro_score_medium": 18.0, "reasons_macro": []}] if a == "BTC" else []
        assets.append({"asset": a, "proxy": None if a == "BTC" else "PAXG (demo)", "price": p, "technical": tf,
                       "context": {"volume_profile_30d": {"poc": p * 0.99, "vah": p * 1.01, "val": p * 0.97}, "orderbook_walls": []},
                       "expected_range": {"1d": [p * 0.98, p * 1.02], "4w_1sigma": [p * 0.9, p * 1.1], "4w_sigma_pct": 9.0,
                                          "macro_drift_pct": 0.3},
                       "macro": {"short": None, "medium": 18.0, "long": -9.0},
                       "backtest": {"period": ["2020-01-01", f"{end:%Y-%m-%d}"], "costs": {"fee_bps": 5, "slippage_bps": 2},
                                    "summary": {s: {"n_trades": 100, "win_rate": 0.4, "avg_r": 0.0, "max_drawdown_r": 10.0}
                                                for s in ("all", "long", "short")},
                                    "latest_params": {}, "folds": 5},
                       "signals": [], "watchlist": wl, "risks": ["demo: upcoming CPI"]})
    (out / "signals.json").write_text(json.dumps({"as_of": now.isoformat(), "assets": assets}, default=str))
    ind = ["cpi", "core_cpi", "nonfarm_payrolls", "dxy", "real_yield_10y", "net_liquidity_weekly", "m2"]
    pd.DataFrame([{"indicator": i, "asset": a, "horizon": h, "bucket": b, "sample": "full", "coefficient": float(rng.uniform(-8, 8)),
                   "beta": 0.1, "t_stat": 2.0, "n": 100, "n_eff": 50, "sd_ret": 1.0, "hit_rate": None, "confidence": "medium",
                   "theory": -1, "conflict": False}
                  for i in ind for a in ("BTC", "Gold") for h, b in (("1h", "short"), ("24h", "short"), ("4w", "medium"), ("6m", "long"))]
                 ).to_csv(out / "impact_coefficients.csv", index=False)
    (out / "features_state.json").write_text(json.dumps({
        "as_of": f"{end:%Y-%m-%d}",
        "regime": {"date": f"{end:%Y-%m-%d}", "regime": "Expansion",
                   "probabilities": {"Expansion": 0.66, "Peak": 0.27, "Recession": 0.02, "Recovery": 0.05},
                   "triggers": ["→ Peak if the momentum outlook (growth momentum vs inflation/liquidity pressure) crosses zero (now +0.30)"]},
        "fed_stance": {"score": -62.0, "components": {"a1_policy_direction": 0.9, "a2_market_path": -1.0, "b_balance_sheet": -1.0,
                                                      "c_communications": None, "d_dot_plot": -1.0}},
        "fedwatch_next": {"meeting": f"{end + pd.Timedelta(days=9):%Y-%m-%d}", "p_cut": 0.7, "p_hold": 0.3, "p_hike": 0.0},
        "curve_state": {"spread_10y_2y": "normal", "spread_10y_3m": "normal", "spread_30y_2y": "inverted"}}))
    (out / "recent_releases.json").write_text(json.dumps([{
        "indicator": "cpi", "release_utc": (now - pd.Timedelta(hours=3)).isoformat(), "obs_date": f"{end - pd.DateOffset(months=1):%Y-%m-01}",
        "actual": 0.42, "expected": 0.25, "surprise": 0.17, "z": 1.8, "basis": "vs mean3 (no free consensus)"}]))
    for n in ("signals", "impact_summary", "features_summary", "data_availability"):
        (out / f"{n}.md").write_text(f"# {n} (demo)\n\nsynthetic\n")
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_output")))
