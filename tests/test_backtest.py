"""Backtest engine tests with hand-checkable bars."""

import pandas as pd
import pytest

from backtest.engine import CostModel, Signal, run, simulate, summarize, walk_forward_folds

T0 = pd.Timestamp("2026-01-01", tz="UTC")
NOCOST = CostModel(fee_bps=0, slippage_bps=0)


def bars(rows):
    ts = pd.date_range(T0, periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame([{"ts": t, "open": o, "high": h, "low": l, "close": c} for t, (o, h, l, c) in zip(ts, rows)])


def test_long_hits_all_targets():
    b = bars([(100, 100, 100, 100), (100, 102, 99.5, 101), (101, 104, 100.5, 103), (103, 107, 102, 106)])
    s = Signal(T0, +1, None, stop=98, targets=(102, 104, 106))
    r = simulate(s, b, NOCOST)
    # entry at bar 1 open 100, risk 2; TP1 +1R, TP2 +2R, TP3 +3R, 1/3 each → 2R
    assert r.filled and r.entry_price == 100 and r.tps_hit == 3 and r.r == pytest.approx(2.0)


def test_stop_first_when_bar_touches_both():
    b = bars([(100, 100, 100, 100), (100, 103, 97, 100)])
    r = simulate(Signal(T0, +1, None, stop=98, targets=(102, 104, 106)), b, NOCOST)
    assert r.stopped and r.tps_hit == 0 and r.r == pytest.approx(-1.0)


def test_breakeven_after_tp1():
    b = bars([(100, 100, 100, 100), (100, 102.5, 99.5, 102), (102, 102, 99, 99.5)])
    r = simulate(Signal(T0, +1, None, stop=98, targets=(102, 104, 106)), b, NOCOST)
    assert r.tps_hit == 1 and r.r == pytest.approx(1 / 3)  # +1R on a third, rest out at break-even


def test_short_with_costs_and_gap_through_stop():
    b = bars([(100, 100, 100, 100), (100, 100.5, 99, 99.5), (103, 104, 102, 103)])
    costs = CostModel(fee_bps=10, slippage_bps=5)
    r = simulate(Signal(T0, -1, None, stop=101, targets=(98, 96, 94)), b, costs)
    entry = 100 * (1 - 5e-4)  # selling receives less
    assert r.entry_price == pytest.approx(entry)
    assert r.stopped and r.r < -2.5  # gapped to 103 open, far beyond the 1R stop


def test_limit_entry_not_filled_is_not_a_trade():
    b = bars([(100, 100, 100, 100), (101, 102, 100.5, 101.5), (102, 103, 101, 102)])
    r = simulate(Signal(T0, +1, 99, stop=97, targets=(101, 103, 105), entry_window=2), b, NOCOST)
    assert not r.filled
    trades = run([Signal(T0, +1, 99, stop=97, targets=(101, 103, 105), entry_window=2)], b, NOCOST)
    assert summarize(trades)["n_trades"] == 0


def test_signal_cannot_use_its_own_bar():
    b = bars([(100, 110, 90, 100), (100, 100.5, 99.5, 100)])
    r = simulate(Signal(T0, +1, None, stop=98, targets=(105,), max_bars=1), b, NOCOST)
    assert r.entry_time == T0 + pd.Timedelta(hours=1) and r.tps_hit == 0 and r.timed_out


def test_summary_and_no_overlap():
    b = bars([(100, 100, 100, 100)] + [(100, 103, 99.5, 102)] * 5)
    sigs = [Signal(T0, +1, None, 98, (102, 104, 106), max_bars=3),
            Signal(T0 + pd.Timedelta(minutes=30), +1, None, 98, (102, 104, 106))]
    t = run(sigs, b, NOCOST)
    s = summarize(t)
    assert s["n_signals"] == 1 and s["n_trades"] == 1  # second signal skipped while the first is open
    assert s["small_sample"] is True


def test_walk_forward_folds_do_not_overlap():
    f = walk_forward_folds(pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01"), 12, 6)
    assert f[0] == ((pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")), (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-07-01")))
    assert all(tr[1] <= te[0] for tr, te in f) and len(f) == 4
