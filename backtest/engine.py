"""Trade-level backtester for signal rules: fees, slippage, multi-TP scale-out, walk-forward folds.

Conventions (conservative by design)
- A signal is known at `time` (a bar *close*); it can only fill from the next bar on.
- Entry: `market` fills at the next bar's open ± slippage; `limit` fills when a later bar trades through the
  entry price within `entry_window` bars (no fill → the signal is recorded as unfilled, not as a trade).
- Exits are checked bar by bar. If one bar touches both the stop and a take-profit, the **stop** is assumed
  to hit first (intra-bar order is unknown).
- Scale-out: size fractions at TP1/TP2/TP3 (default ⅓ each); after TP1 the stop moves to break-even if
  `breakeven_after_tp1`. Anything still open after `max_bars` closes at that bar's close.
- Costs: `fee_bps` per side on notional, `slippage_bps` on entries, stops and time exits (not on limit TPs).
- Results are in R (multiples of the initial risk |entry − stop|), net of costs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Signal:
    time: pd.Timestamp  # when the signal is known (bar close)
    direction: int  # +1 long, −1 short
    entry: float | None  # None = market
    stop: float
    targets: tuple[float, ...]  # TP1..TP3
    entry_window: int = 4  # bars a limit entry may wait
    max_bars: int = 96  # holding limit
    tag: str = ""


@dataclass
class CostModel:
    fee_bps: float = 5.0  # per side (spot taker ~0.1% on many venues; futures ~0.05%)
    slippage_bps: float = 2.0
    scale_out: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    breakeven_after_tp1: bool = True


@dataclass
class TradeResult:
    tag: str
    signal_time: pd.Timestamp
    direction: int
    filled: bool
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    exit_time: pd.Timestamp | None = None
    r: float = 0.0  # net R multiple
    tps_hit: int = 0
    stopped: bool = False
    timed_out: bool = False
    bars_held: int = 0
    exits: list = field(default_factory=list)


def _slip(price: float, direction: int, bps: float, entering: bool) -> float:
    adverse = 1 if (direction > 0) == entering else -1  # buying pays up, selling receives less
    return price * (1 + adverse * bps / 1e4)


def simulate(sig: Signal, bars: pd.DataFrame, costs: CostModel) -> TradeResult:
    """bars: ts (bar open, sorted), open, high, low, close."""
    d = sig.direction
    res = TradeResult(sig.tag, sig.time, d, filled=False)
    start = bars["ts"].searchsorted(sig.time, side="right")  # first bar strictly after the signal bar close
    if start >= len(bars):
        return res
    o, h, l, c, ts = (bars[k].to_numpy() for k in ("open", "high", "low", "close", "ts"))
    # ── entry ──
    i = start
    if sig.entry is None:
        entry = _slip(o[i], d, costs.slippage_bps, True)
    else:
        for i in range(start, min(start + sig.entry_window, len(bars))):
            if l[i] <= sig.entry <= h[i] or (d > 0 and o[i] <= sig.entry) or (d < 0 and o[i] >= sig.entry):
                break
        else:
            return res
        # a gap through the limit fills at the (better) open
        entry = min(o[i], sig.entry) if d > 0 else max(o[i], sig.entry)
    risk = abs(entry - sig.stop)
    if risk <= 0 or (d > 0 and sig.stop >= entry) or (d < 0 and sig.stop <= entry):
        return res
    res.filled, res.entry_time, res.entry_price = True, pd.Timestamp(ts[i]), entry
    remaining, stop, pnl, tp_idx = 1.0, sig.stop, 0.0, 0
    fee = costs.fee_bps / 1e4
    pnl -= fee * entry / risk  # entry fee in R (size normalized so that 1R = risk per unit)

    def close_part(frac, price, slipped):
        nonlocal remaining, pnl
        px = _slip(price, d, costs.slippage_bps, False) if slipped else price
        pnl += frac * (d * (px - entry) / risk - fee * px / risk)
        remaining -= frac
        return px

    last = min(i + sig.max_bars, len(bars) - 1)
    j = i
    for j in range(i, last + 1):
        hit_stop = (l[j] <= stop) if d > 0 else (h[j] >= stop)
        if hit_stop:  # stop first when ambiguous
            fill = min(o[j], stop) if d > 0 else max(o[j], stop)  # gaps fill at the open
            res.exits.append(("stop", pd.Timestamp(ts[j]), close_part(remaining, fill, True)))
            res.stopped = tp_idx == 0 or stop != entry
            break
        while tp_idx < len(sig.targets):
            tp = sig.targets[tp_idx]
            if (d > 0 and h[j] >= tp) or (d < 0 and l[j] <= tp):
                frac = costs.scale_out[tp_idx] if tp_idx < len(sig.targets) - 1 else remaining
                res.exits.append((f"tp{tp_idx + 1}", pd.Timestamp(ts[j]), close_part(min(frac, remaining), tp, False)))
                tp_idx += 1
                if tp_idx == 1 and costs.breakeven_after_tp1:
                    stop = entry
            else:
                break
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        res.exits.append(("time", pd.Timestamp(ts[j]), close_part(remaining, c[j], True)))
        res.timed_out = True
    res.exit_time, res.bars_held, res.tps_hit, res.r = res.exits[-1][1], j - i + 1, tp_idx, float(pnl)
    return res


def run(signals: list[Signal], bars: pd.DataFrame, costs: CostModel | None = None,
        allow_overlap: bool = False) -> pd.DataFrame:
    """Simulate signals in time order. Without overlap, a signal arriving while a trade is open is skipped."""
    costs = costs or CostModel()
    out, busy_until = [], pd.Timestamp.min.tz_localize("UTC")
    for s in sorted(signals, key=lambda x: x.time):
        if not allow_overlap and s.time < busy_until:
            continue
        r = simulate(s, bars, costs)
        if r.filled and r.exit_time is not None:
            busy_until = r.exit_time
        out.append(asdict(r))
    return pd.DataFrame(out)


def summarize(trades: pd.DataFrame) -> dict:
    t = trades[trades["filled"]] if not trades.empty else trades
    n = int(len(t))
    if n == 0:
        return {"n_signals": int(len(trades)), "n_trades": 0}
    r = t["r"].to_numpy()
    equity = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, equity])[1:] - equity).max())
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "n_signals": int(len(trades)), "n_trades": n, "fill_rate": round(n / len(trades), 3),
        "win_rate": round(float((r > 0).mean()), 3), "avg_r": round(float(r.mean()), 3),
        "median_r": round(float(np.median(r)), 3), "total_r": round(float(r.sum()), 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "max_drawdown_r": round(dd, 2),
        "tp1_rate": round(float((t["tps_hit"] >= 1).mean()), 3), "tp2_rate": round(float((t["tps_hit"] >= 2).mean()), 3),
        "tp3_rate": round(float((t["tps_hit"] >= 3).mean()), 3),
        "small_sample": n < 30,
    }


def walk_forward_folds(start: pd.Timestamp, end: pd.Timestamp, train_months: int = 24, test_months: int = 6):
    """Anchored-rolling (train, test) windows: fit on [t−train, t), evaluate on [t, t+test), step = test."""
    folds, t = [], start + pd.DateOffset(months=train_months)
    while t < end:
        folds.append(((t - pd.DateOffset(months=train_months), t), (t, min(t + pd.DateOffset(months=test_months), end))))
        t += pd.DateOffset(months=test_months)
    return folds
