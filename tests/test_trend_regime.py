import numpy as np
import pandas as pd

from models import trend_regime as tr


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="D"))


def test_exposure_zero_in_downtrend_and_capped_in_uptrend():
    down = _series(np.linspace(300, 100, 400))
    up = _series(np.linspace(100, 300, 400))
    assert tr.exposure(down, 365).iloc[-1] == 0
    e = tr.exposure(up, 365).iloc[-1]
    assert 0 < e <= 1


def test_state_reports_uptrend_with_sma_numbers():
    up = _series(np.linspace(100, 300, 400))
    s = tr.state(up, "BTC")
    assert s["up"] is True
    assert s["price"] > s["sma200"]
    assert s["dist_pct"] > 0


def test_record_beats_hold_drawdown_on_crash():
    rng = np.random.default_rng(1)
    up = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 600)))
    down = up[-1] * np.exp(np.cumsum(rng.normal(-0.004, 0.01, 400)))
    s = _series(np.concatenate([up, down]))
    rec = tr.record(s, "BTC")
    assert rec["full"]["rule"]["max_drawdown"] > rec["full"]["hold"]["max_drawdown"]
