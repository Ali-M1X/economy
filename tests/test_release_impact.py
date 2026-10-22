"""Real release impact (models.release_impact) on synthetic 15-minute candles with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import release_impact as ri

DRIFT = 0.01      # % per hour, everywhere
JUMP = 0.5        # % right after every release
START = pd.Timestamp("2025-01-01", tz="UTC")


def _scenario(n_events: int = 36):
    """Releases every other Wednesday 12:30 UTC from week 13. Every release jumps +JUMP% (surprise z = +2, theory +1).
    The first half also drift +1% over the 24 h before and give back 1% four hours after."""
    ts = pd.date_range(START, START + pd.Timedelta(weeks=12 + 2 * n_events + 2), freq="15min")
    hours = (ts - START) / pd.Timedelta(hours=1)
    logp = DRIFT * np.asarray(hours) / 100
    rel = [pd.Timestamp("2025-03-26 12:30", tz="UTC") + pd.Timedelta(weeks=2 * i) for i in range(n_events)]
    end = ts + pd.Timedelta(minutes=15)  # the bar's close is the price at its end
    for i, t in enumerate(rel):
        logp = logp + np.where(end > t, JUMP / 100, 0)
        if i < n_events // 2:
            ramp = np.clip((end - (t - pd.Timedelta(hours=24))) / pd.Timedelta(hours=24), 0, 1)
            logp = logp + np.asarray(ramp) * 0.01 * (end <= t) + 0.01 * (end > t)
            logp = logp - np.where(end > t + pd.Timedelta(hours=4), 0.01, 0)
    bars = pd.DataFrame({"ts": ts, "close": 100 * np.exp(logp)})
    events = pd.DataFrame({"release_utc": rel, "z": 2.0})
    return bars, events, {t.date() for t in rel}


@pytest.fixture(scope="module")
def m():
    bars, events, busy = _scenario()
    return ri.measure(ri.Prices.from_bars(bars), events, busy, theory=1)


def test_shock_size_and_ratio_against_normal_days(m):
    s1, s15 = m["shock"]["1h"], m["shock"]["15m"]
    assert s1["n"] == 36 and s1["median_abs"] == pytest.approx(JUMP + DRIFT, abs=2e-3)
    assert s1["normal_median_abs"] == pytest.approx(DRIFT, abs=1e-3)
    assert s1["ratio"] == pytest.approx((JUMP + DRIFT) / DRIFT, rel=0.02)
    assert s15["median_abs"] == pytest.approx(JUMP + DRIFT / 4, abs=2e-3)


def test_direction_follows_the_surprise(m):
    d = m["direction"]
    assert d["n"] == 36 and d["hit_rate"] == 1.0 and d["p_value"] < 1e-4 and d["predictable"] and d["with_theory"]
    assert m["tier"] == "volatile_directional"


def test_fade_counts_moves_that_end_on_the_other_side(m):
    f = m["fade"]
    assert f["n"] == 36 and f["reversal_rate"] == 0.5
    r1 = JUMP + DRIFT
    kept, gave_back = 1 - (JUMP + 24 * DRIFT) / r1, 1 - (JUMP + 24 * DRIFT - 1.0) / r1
    assert f["median_retrace_pct"] == pytest.approx(50 * (kept + gave_back), abs=1)


def test_front_running_drift_and_what_followed(m):
    p = m["pre"]["24"]
    assert p["median_abs"] == pytest.approx((24 * DRIFT + 24 * DRIFT + 1.0) / 2, abs=5e-3)
    assert p["normal_median_abs"] == pytest.approx(24 * DRIFT, abs=1e-3) and p["ratio"] > 3
    pp = m["pre_post"]
    assert pp["same_direction_rate"] == 0.5
    big = pp["large_drift"]  # the drifted releases are exactly the ones that reversed
    assert big["n"] == 12 and big["reversal_rate"] == 1.0
    assert big["post_median_in_drift_direction"] == pytest.approx(JUMP + 24 * DRIFT - 1.0, abs=5e-3)


def test_small_samples_are_hidden_and_no_theory_means_no_direction():
    bars, events, busy = _scenario(8)
    m = ri.measure(ri.Prices.from_bars(bars), events, busy, theory=0)
    assert m["shock"]["1h"] == {"n": 8} and m["direction"]["n"] == 0 and m["tier"] is None


def test_baseline_skips_release_days():
    t = pd.to_datetime(["2025-03-26 12:30"], utc=True).as_unit("ns").astype("int64").to_numpy()
    busy = {pd.Timestamp("2025-03-19").date()}
    days = {pd.Timestamp(x, tz="UTC").date() for x in ri.baseline_times(t, busy, weeks=3)}
    assert days == {pd.Timestamp("2025-03-12").date(), pd.Timestamp("2025-03-05").date()}


def test_prices_do_not_fill_across_gaps():
    bars = pd.DataFrame({"ts": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 03:00"], utc=True), "close": [100.0, 110.0]})
    px = ri.Prices.from_bars(bars)
    at = lambda s: pd.to_datetime([s], utc=True).as_unit("ns").astype("int64").to_numpy()  # noqa: E731
    assert np.isfinite(px.at(at("2025-01-01 00:30"))[0]) and np.isnan(px.at(at("2025-01-01 02:00"))[0])


def _ramp_bars(release: pd.Timestamp, pct: float):
    ts = pd.date_range(release - pd.Timedelta(days=3), release, freq="15min")
    ramp = np.clip((ts + pd.Timedelta(minutes=15) - (release - pd.Timedelta(hours=24))) / pd.Timedelta(hours=24), 0, 1)
    return pd.DataFrame({"ts": ts, "close": 100 * np.exp(np.asarray(ramp) * pct / 100)})


@pytest.mark.parametrize("pct, verdict", [(0.1, "not_priced"), (0.6, "partly_priced"), (2.0, "heavily_priced")])
def test_front_running_verdict(pct, verdict):
    T = pd.Timestamp("2026-10-14 12:30", tz="UTC")
    m = {"pre": {"24": {"median_abs": 0.5}}, "pre_post": {"large_drift": {"n": 20, "reversal_rate": 0.6}}}
    r = ri.frontrun_read(ri.Prices.from_bars(_ramp_bars(T, pct)), m, T, T - pd.Timedelta(hours=6))
    assert r["window_h"] == 24 and r["elapsed_h"] == 18.0
    assert r["move_pct"] == pytest.approx(pct * 0.75, abs=0.02)
    assert r["normal_pct"] == pytest.approx(0.5 * np.sqrt(0.75), abs=0.01)
    assert r["verdict"] == verdict and r["large"]["n"] == 20
    assert ri.frontrun_read(ri.Prices.from_bars(_ramp_bars(T, pct)), m, T, T + pd.Timedelta(hours=1)) is None
