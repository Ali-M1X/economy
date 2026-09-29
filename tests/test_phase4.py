"""Phase 4: indicators, structure, levels, rules (incl. no look-ahead), walk-forward, job end-to-end."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signals import engine
from signals.rules import Params, evaluate, prepare, scan
from technicals import indicators as ind
from technicals.levels import equal_levels, liquidation_clusters, prior_period_levels, volume_profile
from technicals.structure import structure, swings


def synthetic(hours=24 * 900, seed=7, drift=0.0004, start="2023-01-01"):
    """Trending random walk with cycles, 1h OHLCV."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    r = drift * np.sign(np.sin(2 * np.pi * t / (24 * 120))) + rng.normal(0, 0.006, hours)
    c = 30000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 0.003, hours)) * c
    h, l = np.maximum(o, c) + spread, np.minimum(o, c) - spread
    ts = pd.date_range(start, periods=hours, freq="1h", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": rng.uniform(10, 100, hours)})


def test_resample_drops_incomplete_bucket():
    b = synthetic(hours=10)
    h4 = ind.resample(b, "4h")
    assert len(h4) == 2 and h4["high"].iloc[0] == b["high"].iloc[:4].max()


def test_indicator_sanity():
    b = synthetic(hours=500)
    a = ind.atr(b)
    assert (a.dropna() > 0).all()
    r = ind.rsi(pd.Series(np.arange(100.0)))
    assert r.iloc[-1] == pytest.approx(100.0)  # only gains
    assert set(ind.trend(ind.resample(b, "4h")).dropna().unique()) <= {-1, 0, 1}


def test_swings_are_causal_and_structure_labels():
    # zigzag uptrend: higher highs and higher lows
    lows = [100, 105, 110, 115]
    path = []
    for lo in lows:
        path += list(np.linspace(lo, lo + 20, 6)) + list(np.linspace(lo + 20, lo + 5, 5))
    c = np.array(path, dtype=float)
    df = pd.DataFrame({"high": c + 0.1, "low": c - 0.1, "close": c})
    sw = swings(df, k=2)
    assert (sw["known_at"] == sw["idx"] + 2).all()
    st = structure(df, k=2)
    assert st["state"].iloc[-1] == 1
    assert "BOS_UP" in set(st["event"])


def test_levels():
    assert equal_levels(np.array([100.0, 100.2, 105, 110, 110.1]), 0.3) == [pytest.approx(100.1), pytest.approx(110.05)]
    b = synthetic(hours=24 * 3)
    b.loc[b.index[10:20], "volume"] = 10_000  # heavy trading in a narrow window
    vp = volume_profile(b)
    band = (b["low"].iloc[10:20].min(), b["high"].iloc[10:20].max())
    assert band[0] <= vp["poc"] <= band[1] and vp["val"] <= vp["poc"] <= vp["vah"]
    pp = prior_period_levels(b, pd.Series([pd.Timestamp("2023-01-02 12:00", tz="UTC")]))
    day1 = b[b["ts"] < "2023-01-02"]
    assert pp["pdh"].iloc[0] == day1["high"].max() and pp["pdl"].iloc[0] == day1["low"].min()


def test_liquidation_cluster_estimate():
    oi = pd.DataFrame({"ts": pd.to_datetime(["2026-09-28 10:00", "2026-09-28 11:00"], utc=True),
                       "open_interest_usd": [1e9, 1.1e9]})
    px = pd.Series([100.0, 100.0], index=oi["ts"])
    lc = liquidation_clusters(oi, px, bucket=1.0)
    longs = lc[lc["side"] == "long"].set_index("bucket")["usd"]
    assert longs[90.0] == pytest.approx(0.1e9 * 0.4 / 2)  # 10x longs: 100 × (1 − 0.1 + 0.005) = 90.5


def test_rules_have_no_lookahead():
    b = synthetic()
    p = Params()
    full = scan(prepare(b, p), p)
    assert len(full) >= 5
    for s in full[:: max(1, len(full) // 6)]:
        cut = b[b["ts"] < s.time]  # only data known when the 4h bar closed
        pre = prepare(cut, p)
        t = len(pre["h4"]) - 1
        s2 = evaluate(pre, t, s.direction, p)
        assert s2 is not None, f"setup at {s.time} depends on later data"
        assert (s2.stop, *s2.targets) == pytest.approx((s.stop, *s.targets))


def test_setup_geometry():
    b = synthetic()
    p = Params()
    for s in scan(prepare(b, p), p):
        risk = abs(s.close - s.stop)
        assert risk >= p.sl_atr * s.atr - 1e-9  # stop at least sl_atr × ATR away
        assert all((t - s.close) * s.direction > 0 for t in s.targets)
        assert s.rr[0] >= 1.0 - 1e-9 and s.blended_rr >= p.min_rr


def test_walk_forward_reports_oos_only():
    b = synthetic(hours=24 * 1100)
    wf = engine.walk_forward(b, "BTC", train_months=12, test_months=6)
    first_test = pd.Timestamp(wf["folds"][0]["test_start"], tz="UTC")
    if not wf["trades"].empty:
        assert (pd.to_datetime(wf["trades"]["signal_time"], utc=True) >= first_test).all()
    assert set(wf["summary"]) == {"all", "long", "short"}


def test_signals_job_end_to_end(tmp_path, monkeypatch):
    from jobs import signals as job
    b = synthetic(hours=24 * 800, start="2024-06-01")
    b.to_csv(tmp_path / "candles_BTC_15m.csv.gz", index=False)
    b.assign(close=b["close"] / 10, open=b["open"] / 10, high=b["high"] / 10, low=b["low"] / 10) \
        .to_csv(tmp_path / "candles_PAXG_15m.csv.gz", index=False)
    pd.DataFrame({"event_id": ["cpi-x"], "release_id": ["cpi"], "name_en": ["CPI"], "name_fa": ["CPI"], "importance": [5],
                  "scheduled_utc": [pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=20)]}).to_csv(tmp_path / "calendar_events.csv", index=False)
    out = tmp_path / "out"
    out.mkdir()
    (out / "macro_scores.json").write_text(json.dumps({"BTC": {"medium": {"score": 40, "top": []}}, "Gold": {}}))
    monkeypatch.setattr(job, "OUT_DIR", out)
    assert job.main(["--from-cache", str(tmp_path)]) == 0
    res = json.loads((out / "signals.json").read_text())
    btc = res["assets"][0]
    assert btc["asset"] == "BTC" and {"1d", "4h", "1h"} <= set(btc["technical"])
    assert any("CPI" in r for r in btc["risks"])
    md = (out / "signals.md").read_text()
    assert "not financial advice" in md and "توصیه مالی" in md


def test_signal_needs_demonstrated_edge(monkeypatch):
    """A setup that fires with an aligned macro score is still blocked when its OOS avg R ≤ 0."""
    from jobs import signals as job
    from signals.rules import Setup
    s = Setup(pd.Timestamp("2026-09-28", tz="UTC"), 1, 100.0, 98.0, "swing low", 97.0, (102.0, 104.0, 106.0),
              ("swing high", "swing high", "4R"), (1.67, 2.33, 3.0), 2.33, 1.0, 45.0, ["x"])
    bars = synthetic(hours=24 * 60)

    def fake_wf(b, asset):
        return {"summary": {"long": {"n_trades": 200, "win_rate": 0.4, "avg_r": -0.1}, "short": {"n_trades": 0},
                            "all": {"n_trades": 200}},
                "latest_params": Params(), "period": ["a", "b"], "costs": {"fee_bps": 5, "slippage_bps": 2}, "folds": []}
    monkeypatch.setattr(job.engine, "walk_forward", fake_wf)
    monkeypatch.setattr(job.engine, "live_setups", lambda b, p: [s])
    monkeypatch.setattr(job, "load_bars", lambda c, n: bars)
    res = job.analyse_asset("BTC", job.ASSETS["BTC"], Path("."), {"BTC": {"medium": {"score": 50, "top": []}}}, {},
                            pd.Timestamp("2026-09-29 10:00", tz="UTC"))
    assert res["signals"] == [] and len(res["watchlist"]) == 1
    assert any("no demonstrated edge" in b for b in res["watchlist"][0]["blocked_by"])
