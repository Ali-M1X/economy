"""Track record of the trend call and the 4-week range calibration (models.track_record)."""

import numpy as np
import pandas as pd

from models import track_record as tr


def _bars(drift: float, n_days: int = 900, seed: int = 1) -> pd.DataFrame:
    """15-minute bars of a random walk with a per-day drift (log %)."""
    rng = np.random.default_rng(seed)
    n = n_days * 96
    ret = rng.normal(drift / 96 / 100, 0.0035, n)
    close = 100 * np.exp(np.cumsum(ret))
    ts = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0})


def test_a_trend_that_continues_scores_better_than_chance():
    # alternating 60-day up and down runs with little noise: following the trend is right most of the time
    rng = np.random.default_rng(3)
    days = 900
    per_day = np.where((np.arange(days) // 60) % 2 == 0, 0.8, -0.8)  # % per day
    ret = np.repeat(per_day / 96 / 100, 96) + rng.normal(0, 0.0004, days * 96)
    close = 100 * np.exp(np.cumsum(ret))
    ts = pd.date_range("2023-01-01", periods=days * 96, freq="15min", tz="UTC")
    d = tr.replay(pd.DataFrame({"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}))
    st = tr.hit_stats(d, 7, d["ts"].iloc[0])
    assert st["n"] >= tr.MIN_N and st["hit"] > 0.7 and st["edge"]


def test_random_walk_has_no_edge_and_the_range_holds_two_thirds():
    b = _bars(0.0, seed=7)
    d = tr.replay(b)
    st = tr.hit_stats(d, 7, d["ts"].iloc[0])
    assert st["hit"] is None or not st["edge"]
    rng = tr.range_factor(d)
    assert 0.55 < rng["coverage_before"] < 0.85 and 1.0 <= rng["factor"] <= 1.6
    assert rng["coverage_after"] >= rng["coverage_before"] - 1e-9


def test_too_few_calls_give_no_hit_rate():
    d = tr.replay(_bars(0.0, n_days=120))
    assert tr.hit_stats(d, 7, d["ts"].iloc[-20])["hit"] is None
