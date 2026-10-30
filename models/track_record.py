"""Track record: do the calls the daily report makes come true? Replayed on the snapshot's own candles.

The report's directional lean comes from the trend (daily trend + half the 4-hour trend, as in reports.advisor). Message
history is not archived, so the lean is replayed day by day on past candles (no look-ahead: the trend at a day's close,
the return from that close) and scored against the real 1-day and 7-day returns. The same replay measures whether the
4-week ±1σ range really contains 68 % of outcomes and returns the width factor that makes it so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from technicals.indicators import resample, trend

HORIZONS = (1, 7)
WINDOW_DAYS = 730          # how far back the hit rates look
RANGE_DAYS = 28            # the 4-week range
VOL_DAYS = 90
MIN_N = 30
EDGE_P = 0.10              # binomial p-value below which a hit rate counts as better than chance


def replay(bars: pd.DataFrame) -> pd.DataFrame:
    """Daily rows: close, score/stance of the trend call and forward returns (%) over 1, 7 and 28 days."""
    d1, h4 = resample(bars, "1d"), resample(bars, "4h")
    d1["t1"] = trend(d1).values
    h4["t4"] = trend(h4).values
    t4 = h4.groupby(h4["ts"].dt.floor("1D"))["t4"].last()
    d1["t4"] = d1["ts"].dt.floor("1D").map(t4)
    d1 = d1.dropna(subset=["t1", "t4"]).reset_index(drop=True)
    d1["score"] = d1["t1"] + 0.5 * d1["t4"]
    d1["stance"] = np.select([d1["score"] >= 1, d1["score"] <= -1], [1, -1], 0)
    lr = np.log(d1["close"])
    for h in (*HORIZONS, RANGE_DAYS):
        d1[f"f{h}"] = (lr.shift(-h) - lr) * 100
    d1["sig"] = lr.diff().rolling(VOL_DAYS).std() * np.sqrt(RANGE_DAYS) * 100
    return d1


def hit_stats(d: pd.DataFrame, h: int, since: pd.Timestamp) -> dict:
    """Directional hit rate of the trend call over h days (non-overlapping samples), against chance."""
    y = d[d["ts"] >= since].dropna(subset=[f"f{h}"]).iloc[::h]
    y = y[y["stance"] != 0]
    n = int(len(y))
    if n < MIN_N:
        return {"n": n, "hit": None, "p": None, "edge": False}
    k = int((np.sign(y[f"f{h}"]) == y["stance"]).sum())
    # chance level: a call that always says "up" in a rising market is right by luck, so the null is the share of
    # hits a random call with the same mix of up/down calls would score
    up = float((y[f"f{h}"] > 0).mean())
    chance = float((y["stance"] == 1).mean() * up + (y["stance"] == -1).mean() * (1 - up))
    p = float(binomtest(k, n, chance, alternative="greater").pvalue)
    return {"n": n, "hit": round(k / n, 3), "chance": round(chance, 3), "p": round(p, 3), "edge": bool(p < EDGE_P)}


def range_factor(d: pd.DataFrame, years: int = 3) -> dict:
    """Factor that makes the ±1σ 4-week range hold 68 % of outcomes, with the coverage before and after."""
    x = d.dropna(subset=["sig", f"f{RANGE_DAYS}"])
    x = x[x["ts"] >= x["ts"].iloc[-1] - pd.Timedelta(days=365 * years)]
    if len(x) < 200:
        return {"factor": 1.0, "coverage_before": None, "coverage_after": None, "n": int(len(x))}
    ratio = (x[f"f{RANGE_DAYS}"].abs() / x["sig"]).replace([np.inf], np.nan).dropna()
    k = float(np.clip(ratio.quantile(0.68), 1.0, 1.6))
    return {"factor": round(k, 2), "coverage_before": round(float((ratio <= 1).mean()), 3),
            "coverage_after": round(float((ratio <= k).mean()), 3), "n": int(len(ratio))}


def evaluate(bars: pd.DataFrame, now: pd.Timestamp) -> dict:
    d = replay(bars)
    if d.empty:
        return {}
    since = now - pd.Timedelta(days=WINDOW_DAYS)
    since = since.tz_convert("UTC") if since.tzinfo else since.tz_localize("UTC")
    return {"hit": {f"{h}d": hit_stats(d, h, since) for h in HORIZONS}, "range": range_factor(d),
            "days": int(len(d[d["ts"] >= since])), "asof": str(d["ts"].iloc[-1])}
