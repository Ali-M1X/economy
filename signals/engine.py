"""Walk-forward evaluation of the setup rules and construction of live signals.

Win rates are out-of-sample only: for each fold, the parameter set with the best total R on the preceding
24 months (≥ 15 trades) is applied, unchanged, to the next 6 months. Fees and slippage are included
(backtest.engine.CostModel). The macro-direction filter used live is NOT part of the backtest (no
look-ahead-free history of the Macro Score exists yet), so the reported win rate is the technical setup's.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pandas as pd

from backtest.engine import CostModel, Signal, run, summarize, walk_forward_folds
from signals.rules import Params, Setup, prepare, scan
from technicals.indicators import resample

GRID = [Params(zone_atr=z, sl_atr=s) for z, s in product((0.5, 1.0), (1.0, 1.5))]
DEFAULT = Params()
COSTS = {"BTC": CostModel(fee_bps=5, slippage_bps=2), "Gold": CostModel(fee_bps=10, slippage_bps=5)}  # PAXG is thinner


def to_signal(s: Setup, params: Params) -> Signal:
    return Signal(time=s.time, direction=s.direction, entry=None, stop=s.stop, targets=s.targets,
                  max_bars=params.max_hold_hours, tag=f"{'long' if s.direction > 0 else 'short'}")


def walk_forward(bars: pd.DataFrame, asset: str, train_months: int = 24, test_months: int = 6) -> dict:
    """Returns OOS trades (all folds), per-fold chosen params and summary stats (overall and per direction)."""
    exec_bars = resample(bars, "1h")  # exits simulated on 1h bars
    setups = {p: scan(prepare(bars, p), p) for p in GRID}
    costs = COSTS[asset]
    start, end = bars["ts"].iloc[0].tz_localize(None), bars["ts"].iloc[-1].tz_localize(None)
    oos, chosen = [], []
    for (tr0, tr1), (te0, te1) in walk_forward_folds(start, end, train_months, test_months):
        tr0, tr1, te0, te1 = (pd.Timestamp(x, tz="UTC") for x in (tr0, tr1, te0, te1))
        best, best_r = DEFAULT, -np.inf
        for p, ss in setups.items():
            sig = [to_signal(s, p) for s in ss if tr0 <= s.time < tr1]
            t = run(sig, exec_bars, costs)
            st = summarize(t)
            if st.get("n_trades", 0) >= 15 and st["total_r"] > best_r:
                best, best_r = p, st["total_r"]
        test = [to_signal(s, best) for s in setups[best] if te0 <= s.time < te1]
        tt = run(test, exec_bars, costs)
        if not tt.empty:
            tt["fold_start"] = te0
            oos.append(tt)
        chosen.append({"test_start": te0.date().isoformat(), "params": asdict(best),
                       "train_total_r": None if best_r == -np.inf else round(best_r, 2)})
    trades = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    summary = {"all": summarize(trades) if not trades.empty else {"n_trades": 0}}
    for tag in ("long", "short"):
        sub = trades[trades["tag"] == tag] if not trades.empty else trades
        summary[tag] = summarize(sub) if not sub.empty else {"n_trades": 0}
    latest = chosen[-1]["params"] if chosen else asdict(DEFAULT)
    return {"trades": trades, "folds": chosen, "summary": summary, "latest_params": Params(**latest),
            "costs": asdict(costs), "period": [start.date().isoformat(), end.date().isoformat()]}


def confidence(stats: dict, macro_abs: float) -> str:
    n, wr, avg_r = stats.get("n_trades", 0), stats.get("win_rate", 0), stats.get("avg_r", -1)
    if n >= 50 and avg_r > 0.2 and macro_abs >= 30:
        return "high"
    if n >= 30 and avg_r > 0 and macro_abs >= 15:
        return "medium"
    return "low"


def live_setups(bars: pd.DataFrame, params: Params, recent_bars: int = 2) -> list[Setup]:
    """Setups that fired on the last `recent_bars` closed 4h bars and are not yet invalidated
    (price has not closed beyond the stop since)."""
    pre = prepare(bars, params)
    n = len(pre["h4"])
    out = []
    for s in scan(pre, params, start=max(60, n - recent_bars), end=n):
        after = bars[bars["ts"] >= s.time]
        broken = ((after["close"] < s.stop).any() if s.direction > 0 else (after["close"] > s.stop).any())
        if not broken:
            out.append(s)
    return out


def signal_card(s: Setup, asset: str, stats: dict, macro_score: float | None, horizon: str) -> dict:
    """Everything the report/Telegram needs for one signal."""
    side = "long" if s.direction > 0 else "short"
    zone = sorted([s.close, s.close - s.direction * 0.25 * s.atr])
    return {
        "asset": asset, "direction": side, "horizon": horizon, "signal_time": s.time.isoformat(),
        "entry_zone": [round(zone[0], 2), round(zone[1], 2)], "stop_loss": round(s.stop, 2),
        "targets": [round(t, 2) for t in s.targets], "target_kinds": list(s.target_kinds),
        "rr": [round(x, 2) for x in s.rr], "blended_rr": round(s.blended_rr, 2),
        "win_rate": stats.get("win_rate"), "avg_r": stats.get("avg_r"), "n_backtest": stats.get("n_trades", 0),
        "max_drawdown_r": stats.get("max_drawdown_r"), "small_sample": stats.get("small_sample", True),
        "confidence": confidence(stats, abs(macro_score or 0)),
        "invalidation": (f"4h close {'below' if s.direction > 0 else 'above'} {s.stop:,.2f}, or the daily trend flips"),
        "reasons_technical": s.reasons, "support": {"price": round(s.support, 2), "kind": s.support_kind},
    }
