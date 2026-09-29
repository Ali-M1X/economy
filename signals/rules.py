"""Trend-pullback setup, evaluated causally on closed 4h bars. Used identically by the backtest and live.

Long setup at 4h bar t (short is the mirror image):
1. Higher-timeframe trend: the last *completed* daily bar is in an uptrend (close > EMA50 and EMA20 > EMA50).
2. 4h market structure is bullish (last swing high HH and last swing low HL; swings known at t only).
3. Pullback into support: the nearest support S below the close — max of recent swing lows, equal lows,
   prior-day low and prior-week low — was touched within the last 3 bars (low ≤ S + zone_atr·ATR) and held
   (close > S).
4. Trigger/confirmation: close > previous bar's high, RSI(14) between 35 and 65 (not stretched), MACD
   histogram rising.
Orders: entry at the next bar's open; stop = min(S − 0.25·ATR, close − sl_atr·ATR) (beyond structure and
≥ sl_atr × ATR); targets = the next resistances above (swing highs, equal highs, prior-day/week highs), each
at least 1R / +0.5R beyond the previous; where fewer than three exist, R multiples (2R, 3R, 4R) fill in.
The setup is emitted only if the blended reward:risk (⅓ at each target) ≥ min_rr.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from technicals.indicators import atr, ema, macd, resample, rsi, trend
from technicals.levels import equal_levels, prior_period_levels
from technicals.structure import structure, swings


@dataclass(frozen=True)
class Params:
    zone_atr: float = 0.5
    sl_atr: float = 1.0
    swing_k: int = 3
    lookback: int = 60
    min_rr: float = 1.5
    max_hold_hours: int = 120


@dataclass
class Setup:
    time: pd.Timestamp  # 4h bar close (when the setup is known)
    direction: int
    close: float
    support: float  # the structure level the trade leans on (S for longs, R for shorts)
    support_kind: str
    stop: float
    targets: tuple[float, float, float]
    target_kinds: tuple[str, str, str]
    rr: tuple[float, float, float]
    blended_rr: float
    atr: float
    rsi: float
    reasons: list[str] = field(default_factory=list)


def prepare(bars_15m_or_1h: pd.DataFrame, params: Params) -> dict:
    """Pre-compute everything the rules need on the 4h and daily frames (all causal)."""
    h4 = resample(bars_15m_or_1h, "4h")
    d1 = resample(bars_15m_or_1h, "1d")
    d1["trend"] = trend(d1)
    d1["end"] = d1["ts"] + pd.Timedelta(days=1)
    h4["end"] = h4["ts"] + pd.Timedelta(hours=4)
    # daily trend known at each 4h close = trend of the last daily bar that ended at or before it
    m = pd.merge_asof(h4[["end"]], d1[["end", "trend"]].dropna(), on="end", direction="backward")
    h4["d_trend"] = m["trend"].to_numpy()
    h4["atr"] = atr(h4)
    h4["rsi"] = rsi(h4["close"])
    h4["macd_hist"] = macd(h4["close"])["hist"]
    st = structure(h4, params.swing_k)
    h4["state"], h4["event"] = st["state"].to_numpy(), st["event"].to_numpy()
    pp = prior_period_levels(bars_15m_or_1h, h4["end"])
    for c in pp:
        h4[c] = pp[c].to_numpy()
    h4 = h4.reset_index(drop=True)
    sw = swings(h4, params.swing_k)
    arrays = {c: h4[c].to_numpy(dtype="float64") for c in
              ("open", "high", "low", "close", "atr", "rsi", "macd_hist", "d_trend", "state", "pdh", "pdl", "pwh", "pwl")}
    # swings per kind, ordered by bar index (known_at = idx + k, so also ordered by when they became known)
    by_kind = {kd: (g["idx"].to_numpy(), g["price"].to_numpy(dtype="float64"))
               for kd, g in sw.sort_values("idx").groupby("kind")}
    return {"h4": h4, "swings": sw, "a": arrays, "sw": by_kind, "k": params.swing_k}


def _recent(pre: dict, t: int, kind: str, lookback: int) -> np.ndarray:
    """Swing prices of `kind` known at bar t (idx ≤ t − k) with idx ≥ t − lookback — a contiguous slice."""
    if kind not in pre["sw"]:
        return np.empty(0)
    idx, price = pre["sw"][kind]
    lo = np.searchsorted(idx, t - lookback, side="left")
    hi = np.searchsorted(idx, t - pre["k"], side="right")
    return price[lo:hi]


def _levels(pre: dict, t: int, side: str, params: Params, atr_t: float) -> list[tuple[float, str]]:
    """Support ('L') or resistance ('H') candidates known at bar t."""
    a = pre["a"]
    sp = _recent(pre, t, side, params.lookback)
    lv = [(float(p), "swing low" if side == "L" else "swing high") for p in sp]
    lv += [(p, "equal lows" if side == "L" else "equal highs") for p in equal_levels(sp, 0.25 * atr_t)]
    if side == "L":
        lv += [(a["pdl"][t], "prior-day low"), (a["pwl"][t], "prior-week low")]
    else:
        lv += [(a["pdh"][t], "prior-day high"), (a["pwh"][t], "prior-week high")]
    return [(float(p), k) for p, k in lv if np.isfinite(p)]


def evaluate(pre: dict, t: int, direction: int, params: Params) -> Setup | None:
    a = pre["a"]
    if t < 60:
        return None
    at = a["atr"][t]
    if not np.isfinite(at) or at <= 0 or a["d_trend"][t] != direction or a["state"][t] != direction:
        return None
    c = a["close"][t]
    recent_low, recent_high = a["low"][t - 2:t + 1].min(), a["high"][t - 2:t + 1].max()
    if direction > 0:
        sup = [(p, k) for p, k in _levels(pre, t, "L", params, at) if p < c]
        if not sup:
            return None
        s, skind = max(sup, key=lambda x: x[0])
        touched = recent_low <= s + params.zone_atr * at
        trigger = c > a["high"][t - 1]
    else:
        sup = [(p, k) for p, k in _levels(pre, t, "H", params, at) if p > c]
        if not sup:
            return None
        s, skind = min(sup, key=lambda x: x[0])
        touched = recent_high >= s - params.zone_atr * at
        trigger = c < a["low"][t - 1]
    r = {"rsi": a["rsi"][t]}
    rsi_ok = 35 <= r["rsi"] <= 65
    macd_ok = (a["macd_hist"][t] - a["macd_hist"][t - 1]) * direction > 0
    if not (touched and trigger and rsi_ok and macd_ok):
        return None
    stop = min(s - 0.25 * at, c - params.sl_atr * at) if direction > 0 else max(s + 0.25 * at, c + params.sl_atr * at)
    risk = abs(c - stop)
    opp = _levels(pre, t, "H" if direction > 0 else "L", params, at)
    beyond = sorted(((p, k) for p, k in opp if (p - c) * direction > 0), key=lambda x: (x[0] - c) * direction)
    targets, kinds, last = [], [], c + direction * 1.0 * risk  # TP1 ≥ 1R
    for p, k in beyond:
        if (p - last) * direction >= 0 and len(targets) < 3:
            targets.append(float(p))
            kinds.append(k)
            last = p + direction * 0.5 * risk
    for mult in (2.0, 3.0, 4.0):
        if len(targets) >= 3:
            break
        p = c + direction * mult * risk
        if not targets or (p - targets[-1]) * direction > 0:
            targets.append(float(p))
            kinds.append(f"{mult:g}R")
    rr = tuple(float(abs(p - c) / risk) for p in targets[:3])
    blended = float(np.mean(rr))
    if blended < params.min_rr:
        return None
    reasons = [f"daily trend {'up' if direction > 0 else 'down'}", f"4h structure {'bullish' if direction > 0 else 'bearish'}",
               f"pullback held {skind} {s:,.2f}", f"RSI {r['rsi']:.0f}", "MACD histogram turning"]
    end = pre["h4"]["end"].iloc[t]
    return Setup(end, direction, float(c), float(s), skind, float(stop), tuple(targets[:3]), tuple(kinds[:3]),
                 rr, blended, float(at), float(r["rsi"]), reasons)


def scan(pre: dict, params: Params, start: int = 60, end: int | None = None) -> list[Setup]:
    out = []
    for t in range(start, end if end is not None else len(pre["a"]["close"])):
        for d in (1, -1):
            s = evaluate(pre, t, d, params)
            if s is not None:
                out.append(s)
    return out
