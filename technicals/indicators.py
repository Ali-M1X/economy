"""Classic indicators on OHLCV frames (ts, open, high, low, close, volume). All causal (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {"1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON"}
DURATION = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1),
            "1w": pd.Timedelta(days=7)}


def resample(bars: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate to a higher timeframe. Bars are labelled by their open time; the last (incomplete) bucket
    is dropped so every returned bar is closed."""
    g = bars.set_index("ts").resample(RULES[tf], label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna(subset=["open"])
    # a bucket is closed once the source has a bar ending at or after the bucket's end
    source_step = bars["ts"].diff().median()
    data_end = bars["ts"].iloc[-1] + source_step
    out = out[out.index + DURATION[tf] <= data_end]
    return out.reset_index()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()  # Wilder


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # no losses in the window → RSI 100 (50 if flat); keep NaN only during the warm-up
    out = out.mask(dn.eq(0) & up.gt(0), 100.0).mask(dn.eq(0) & up.eq(0), 50.0)
    return out


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def trend(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 up (close > slow EMA and fast > slow), −1 down (mirror), 0 mixed."""
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    up = (df["close"] > s) & (f > s)
    dn = (df["close"] < s) & (f < s)
    return pd.Series(np.select([up, dn], [1, -1], 0), index=df.index).where(s.notna())
