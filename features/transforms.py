"""Small, pure series transforms. Input/output frames use the canonical (date, value) layout."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4}


def as_series(df: pd.DataFrame) -> pd.Series:
    return df.set_index("date")["value"].sort_index()


def to_frame(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    return pd.DataFrame({"date": s.index, "value": s.to_numpy(dtype="float64")}).reset_index(drop=True)


def pct_change(s: pd.Series, periods: int) -> pd.Series:
    """Percent change over `periods` observations. Requires a regular series: gaps are NOT bridged
    (a missing month makes the dependent changes NaN rather than silently spanning two months)."""
    return (s / s.shift(periods) - 1.0) * 100.0


def monthly_regular(s: pd.Series) -> pd.Series:
    """Reindex a monthly series onto a complete month-start calendar so missing months become NaN."""
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(idx)


def mom(s: pd.Series) -> pd.Series:
    return pct_change(monthly_regular(s), 1)


def yoy(s: pd.Series) -> pd.Series:
    return pct_change(monthly_regular(s), 12)


def annualized_3m(s: pd.Series) -> pd.Series:
    m = monthly_regular(s)
    return ((m / m.shift(3)) ** 4 - 1.0) * 100.0


def zscore(s: pd.Series, window: int | None = None, min_periods: int = 60) -> pd.Series:
    """Rolling (window) or expanding (window=None) z-score using only past-and-present data."""
    r = s.rolling(window, min_periods=min_periods) if window else s.expanding(min_periods=min_periods)
    return (s - r.mean()) / r.std(ddof=0).replace(0, np.nan)


def trend_state(s: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """+1 when the fast moving average is above the slow one, −1 below."""
    return np.sign(s.rolling(fast).mean() - s.rolling(slow).mean())


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b on common dates only (no filling across non-trading days)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return j["a"] / j["b"]
