"""Pre-positioning signals: point-in-time features, walk-forward isolation, the edge gate, the job and live check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import prepos as pp

UTC = "UTC"


def make_bars(start="2019-01-01", end="2024-06-01", events=(), drift_pre=0.0, drift_post=0.0, seed=1,
              signs=None) -> pd.DataFrame:
    """15-min random walk; before/after each event an optional steady drift (a planted pre-positioning edge),
    with taker flow leaning the same way."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, end, freq="15min", tz=UTC)
    n = len(ts)
    r = rng.normal(0, 0.0015, n)
    lean = np.zeros(n)
    pos = pd.Series(np.arange(n), index=ts)
    signs = signs if signs is not None else rng.choice([-1, 1], len(events))
    for T, d in zip(events, signs):
        i = pos.get(T)
        if i is None:
            continue
        r[i - 96:i] += d * drift_pre
        r[i:i + 96] += d * drift_post
        lean[i - 96:i] = d * (0.08 if drift_pre else 0)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    w = np.abs(rng.normal(0, 0.0008, n)) * c
    vol = rng.uniform(10, 20, n)
    tb = vol * np.clip(0.5 + lean + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame({"ts": ts, "open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                         "close": c, "volume": vol, "taker_buy": tb})


def monthly_events(start="2019-06-12 12:30", end="2024-04-30", days=(12,)):
    out = []
    for m in pd.date_range(pd.Timestamp(start).normalize(), end, freq="MS", tz=UTC):
        for d in days:
            out.append(m + pd.Timedelta(days=d - 1, hours=12, minutes=30))
    return [t for t in out if t >= pd.Timestamp(start, tz=UTC)]


def aux_inputs(bars: pd.DataFrame, events) -> pp.Inputs:
    rng = np.random.default_rng(7)
    days = pd.date_range(bars["ts"].iloc[0].normalize(), bars["ts"].iloc[-1], freq="D")
    rate = pd.DataFrame({"available_at": days + pd.Timedelta(hours=44), "value": 2 + np.cumsum(rng.normal(0, 0.03, len(days)))})
    t5 = pd.date_range(bars["ts"].iloc[0], bars["ts"].iloc[-1], freq="5min")
    oi = pd.DataFrame({"ts": t5 + pd.Timedelta(minutes=5), "open_interest": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.002, len(t5))))})
    t8 = pd.date_range(bars["ts"].iloc[0], bars["ts"].iloc[-1], freq="8h")
    fund = pd.DataFrame({"ts": t8, "funding_rate": rng.normal(1e-4, 5e-5, len(t8))})
    lead = pd.DataFrame({"event_id": [f"e-{t:%Y%m%d}" for t in events],
                         "available_at": [t - pd.Timedelta(hours=50) for t in events],
                         "z": rng.normal(0, 1, len(events))})
    return pp.Inputs(bars=bars, rate=rate, oi=oi, funding=fund, lead=lead)


def ev_frame(events):
    return pd.DataFrame({"event_id": [f"e-{t:%Y%m%d}" for t in events], "release_utc": events})


# ── point in time ─────────────────────────────────────────────────────────

def test_features_only_see_data_available_at_each_step():
    """Re-run the features at historical timestamps with every input truncated to what existed then: identical."""
    events = monthly_events(end="2020-12-31")
    bars = make_bars(end="2021-01-31", events=events, drift_pre=0.0004, drift_post=0.0004)
    inp = aux_inputs(bars, events)
    T = events[-3]
    full = pp.feature_path(inp, T, 72, f"e-{T:%Y%m%d}")
    assert not full.empty and full[["drift", "steady", "flow", "rate", "oi", "funding", "lead"]].notna().any().all()
    for t in (full.index[5], full.index[len(full) // 2], full.index[-1]):
        cut = pp.feature_path(inp.truncated(t), T, 72, f"e-{T:%Y%m%d}")
        pd.testing.assert_frame_equal(cut, full.loc[:t], check_freq=False)
        # a live-style call with `until` gives the same numbers
        pd.testing.assert_frame_equal(pp.feature_path(inp, T, 72, f"e-{T:%Y%m%d}", until=t), full.loc[:t], check_freq=False)


def test_future_data_cannot_change_a_past_feature():
    events = monthly_events(end="2020-12-31")
    bars = make_bars(end="2021-01-31", events=events)
    inp = aux_inputs(bars, events)
    T = events[-3]
    before = pp.feature_path(inp, T, 24, f"e-{T:%Y%m%d}")
    shocked = bars.copy()
    after = shocked["ts"] >= T  # a crash right at the release must not leak into the pre-release features
    shocked.loc[after, ["open", "high", "low", "close"]] *= 0.5
    inp2 = pp.Inputs(bars=shocked, rate=inp.rate, oi=inp.oi, funding=inp.funding, lead=inp.lead)
    pd.testing.assert_frame_equal(pp.feature_path(inp2, T, 24, f"e-{T:%Y%m%d}"), before)


def test_training_only_uses_outcomes_known_before_the_window_opens():
    rows = [pp.EventRow(f"e{i}", pd.Timestamp("2024-01-01", tz=UTC) + pd.Timedelta(days=i),
                        {W: pd.DataFrame({"drift": [0.0]}) for W in (24, 72)}, y=0.01)
            for i in range(20)]
    tr = pp._train(rows, 10, 24)
    s = rows[10].T - pd.Timedelta(hours=24)
    assert tr and all(r.T + pp.HOLD <= s for r in tr)
    assert rows[9] not in tr  # event 9 is 24 h before event 10: its outcome is not known when event 10's window opens
    assert pp._train(rows, 10, 72) == [r for r in rows[:10] if r.T + pp.HOLD <= rows[10].T - pd.Timedelta(hours=72)]


@pytest.fixture(scope="module")
def planted():
    events = monthly_events(days=(5, 20))
    bars = make_bars(events=events, drift_pre=0.0004, drift_post=0.0006, seed=3)
    rows = pp.prepare(pp.Inputs(bars=bars), ev_frame(events), windows=(6, 24))
    return events, bars, rows


def test_walk_forward_predictions_ignore_later_events(planted):
    _, bars, rows = planted
    k = len(rows) - 15
    short = pp.walk_forward(rows[:k], bars, windows=(6, 24))["selected_trades"]
    full = pp.walk_forward(rows, bars, windows=(6, 24))["selected_trades"]
    common = [t["event_id"] for t in short]
    assert common and [t for t in full if t["event_id"] in common] == short


def test_planted_edge_passes_the_gate_and_trades_enter_after_the_signal(planted):
    _, bars, rows = planted
    wf = pp.walk_forward(rows, bars, windows=(6, 24))
    ok, why = pp.edge(wf["selected"])
    assert ok, why
    assert wf["selected"]["avg_r"] > 0 and wf["per_window"][24]["spearman"] > 0.3
    t = wf["selected_trades"][0]
    sig = pp.make_signal(bars, t["signal_time"], t["direction"], pd.Timestamp(t["signal_time"]) + pd.Timedelta(hours=1), 1.0, "x")
    from backtest import engine
    res = engine.simulate(sig, bars, engine.CostModel())
    assert res.entry_time > t["signal_time"]  # fills only on a later bar


def test_noise_has_no_edge():
    events = monthly_events(days=(5, 20))
    bars = make_bars(events=events, seed=11)
    rows = pp.prepare(pp.Inputs(bars=bars), ev_frame(events), windows=(6, 24))
    ok, why = pp.edge(pp.walk_forward(rows, bars, windows=(6, 24))["selected"])
    assert not ok and why


def test_edge_gate_reasons():
    assert pp.edge({"n_trades": 40, "avg_r": 0.3, "p_r": 0.01}) == (True, [])
    ok, why = pp.edge({"n_trades": 12, "avg_r": -0.1, "p_r": 0.7})
    assert not ok and len(why) == 3


def test_score_normalises_by_available_features():
    path = pd.DataFrame({k: [np.nan] for k in pp.FEATURES}, index=[pd.Timestamp("2024-01-01", tz=UTC)])
    path["drift"], path["flow"] = 2.0, 1.0
    w = {k: 0 for k in pp.FEATURES} | {"drift": 1, "flow": -1, "oi": 1}
    assert pp.score(path, w).iloc[0] == pytest.approx((2.0 - 1.0) / np.sqrt(2))  # oi missing → not counted


# ── job: events from vintages, live check ────────────────────────────────

def write_cache(tmp_path, now):
    from jobs.prepos import ASSETS

    cache = tmp_path / "cache"
    cache.mkdir()
    events = monthly_events(start="2019-06-12 12:30", end=str(now.date()))
    bars = make_bars(end=str((now - pd.Timedelta(minutes=15)).floor("15min").tz_convert(None)), events=events, drift_pre=0.0004,
                     drift_post=0.0006, seed=5)
    for sym in ASSETS.values():
        bars.to_csv(cache / f"candles_{sym}_15m.csv.gz", index=False)
    # CPI vintages: one first print per month, published on the event day (08:30 ET)
    months = pd.date_range("2017-01-01", now.tz_convert(None), freq="MS")
    vint = []
    for i, m in enumerate(months):
        rel = m + pd.DateOffset(months=1) + pd.Timedelta(days=11)
        start = months[0] if i < 24 else rel  # first 24 months arrive in the bulk first vintage
        vint.append({"series_key": "cpi", "date": m, "value": 250 + i * 0.5, "realtime_start": start, "realtime_end": None})
    pd.DataFrame(vint).to_csv(cache / "vintages.csv.gz", index=False)
    days = pd.date_range("2018-01-01", now.tz_convert(None), freq="B")
    pd.DataFrame({"series_key": "ust_2y", "date": days, "value": 2 + np.cumsum(np.random.default_rng(1).normal(0, 0.03, len(days)))}) \
        .to_csv(cache / "observations.csv.gz", index=False)
    nxt = now + pd.Timedelta(hours=10)
    pd.DataFrame([{"event_id": "cpi-next", "release_id": "cpi", "name_en": "Consumer Price Index", "name_fa": "CPI",
                   "importance": 5, "scheduled_utc": nxt}]).to_csv(cache / "calendar_events.csv", index=False)
    return cache, nxt


def test_job_backtests_cpi_and_checks_the_upcoming_release(tmp_path):
    from jobs import prepos as job

    now = pd.Timestamp("2024-05-01 02:00", tz=UTC)
    cache, nxt = write_cache(tmp_path, now)
    res = job.run(cache, None, None, now, releases=("cpi",), windows=(6, 24))
    bt = res["backtest"]["cpi:BTC"]
    assert bt["events"] > 30 and bt["period"][0] >= "2019"
    assert set(bt["per_window"]) == {"6", "24"} and bt["model"]["selected_window"] in (6, 24)
    live = [r for r in res["live"] if r["asset"] == "BTC"][0]
    W = bt["model"]["selected_window"]
    if now < nxt - pd.Timedelta(hours=W):
        assert live["status"] == "waiting" and pd.Timestamp(live["window_opens"]) == nxt - pd.Timedelta(hours=W)
    else:
        assert live["status"] in ("quiet", "signal", "watchlist")
    md = job.render(res)
    assert "| cpi | BTC |" in md


def test_release_dates_skip_the_bulk_vintage():
    from jobs.prepos import release_dates

    v = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
                      "realtime_start": pd.to_datetime(["2019-01-01", "2020-03-11", "2020-04-10"]), "value": [1, 2, 3]})
    d = release_dates(v, "cpi")
    assert list(d["obs_date"].dt.month) == [2, 3]
    assert list(d["release_utc"].dt.strftime("%Y-%m-%d %H:%M")) == ["2020-03-11 12:30", "2020-04-10 12:30"]


# ── collectors ────────────────────────────────────────────────────────────

def _zip(text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def test_binance_archive_parses_metrics_and_funding_with_or_without_header(tmp_path, monkeypatch):
    from collectors import binance_archive as ba

    metrics = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
               "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
               "2024-01-01 00:05:00,BTCUSDT,80000.1,3.4e9,1.2,1.1,1.3,0.95\n2024-01-01 00:10:00,BTCUSDT,80010.0,3.41e9,1.2,1.1,1.3,1.05\n")
    m = ba.parse_metrics(ba._csv_from_zip(_zip(metrics)))
    assert list(m.columns) == ["ts", "open_interest", "open_interest_usd", "taker_ls_ratio"] and m["open_interest"].iloc[1] == 80010.0
    funding_noheader = "1704067200000,8,0.0001\n1704096000000,8,0.00012\n"
    f = ba.parse_funding(ba._csv_from_zip(_zip(funding_noheader)))
    assert f["ts"].iloc[0] == pd.Timestamp("2024-01-01", tz=UTC) and f["funding_rate"].iloc[1] == 0.00012

    calls = []

    def fake_fetch(url):
        calls.append(url)
        if "metrics-2024-01-02" in url:
            return _zip(metrics.replace("2024-01-01", "2024-01-02"))
        if "fundingRate-2023-12" in url:
            return _zip(funding_noheader)
        return None  # 404
    monkeypatch.setattr(ba, "_fetch", fake_fetch)
    monkeypatch.setattr(ba, "METRICS_START", pd.Timestamp("2024-01-01"))
    monkeypatch.setattr(ba, "FUNDING_START", pd.Timestamp("2023-12-01"))
    r = ba.update(tmp_path, today=pd.Timestamp("2024-01-03", tz=UTC))
    assert r["metrics_days"] == 1 and r["funding_months"] == 1
    n = len(calls)
    ba.update(tmp_path, today=pd.Timestamp("2024-01-03", tz=UTC))
    assert len(calls) == n + 1  # only the recent missing day is retried; fetched files are never fetched again
    mm, ff = ba.load(tmp_path)
    assert len(mm) == 2 and len(ff) == 2


def test_okx_funding_and_binance_taker_volume_parsing():
    from collectors import crypto

    f = crypto.parse_okx_funding([{"fundingTime": "1704096000000", "fundingRate": "0.0002", "realizedRate": "0.00019"},
                                  {"fundingTime": "1704067200000", "fundingRate": "0.0001", "realizedRate": ""}])
    assert list(f["funding_rate"]) == [0.0001, 0.00019]
    rows = [[1704067200000, "1", "2", "0.5", "1.5", "10", 0, "15", 5, "6", "9", "0"]]
    b = crypto._frame_binance(rows)
    assert b["taker_buy"].iloc[0] == 6.0 and b["volume"].iloc[0] == 10.0
