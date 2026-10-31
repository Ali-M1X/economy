"""Long-horizon trend filter with volatility-scaled exposure, and its measured record on our own price history.

Rules (published, widely studied, deliberately simple; nothing is fitted):
- trend: long when the close is above its 200-day average, otherwise flat (Faber 2007; Moskowitz-Ooi-Pedersen 2012 for the
  12-month version, shown as a second opinion);
- size: exposure = min(1, 20 % / 60-day annualised volatility) while the trend is up, so a calmer market allows a larger
  share and a wild one a smaller share (volatility targeting; Moreira-Muir 2017, Harvey et al. 2018);
- execution: the decision is taken at a daily close and held over the next day (no look-ahead), 10 bp (BTC) or 5 bp
  (gold) cost on every change of exposure.

The record compares the rule with buying and holding over the whole history and the last five years. It is a risk-control
rule: it makes no claim about the next week's direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_VOL = 0.20
SMA_DAYS = 200
VOL_DAYS = 60
MOM_DAYS = {"BTC": 365, "Gold": 252}
PPY = {"BTC": 365, "Gold": 252}
COST = {"BTC": 0.0010, "Gold": 0.0005}
MIN_DAYS = SMA_DAYS + 60


def exposure(close: pd.Series, ppy: int) -> pd.Series:
    """Target exposure per day, 0..1, known at that day's close."""
    sma = close.rolling(SMA_DAYS).mean()
    vol = close.pct_change().rolling(VOL_DAYS).std() * np.sqrt(ppy)
    up = (close > sma).astype(float).where(sma.notna())
    return (up * (TARGET_VOL / vol).clip(upper=1.0)).where(vol.notna())


def _stats(r: pd.Series, ppy: int) -> dict | None:
    r = r.dropna()
    if len(r) < 250 or not r.std():
        return None
    eq = (1 + r).cumprod()
    return {"cagr": round(float(eq.iloc[-1] ** (ppy / len(r)) - 1) * 100, 1),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(ppy)), 2),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}


def record(close: pd.Series, asset: str) -> dict:
    """Rule vs buy-and-hold: CAGR, Sharpe and worst drawdown over the whole history and the last five years."""
    ppy, cost = PPY[asset], COST[asset]
    r = close.pct_change()
    w = exposure(close, ppy).shift(1)
    strat = w * r - w.diff().abs().fillna(0) * cost
    out = {}
    for name, start in (("full", None), ("5y", close.index[-1] - pd.Timedelta(days=365 * 5))):
        sl = (lambda x: x) if start is None else (lambda x, s=start: x[x.index >= s])
        a, b = _stats(sl(strat), ppy), _stats(sl(r), ppy)
        if a and b:
            out[name] = {"rule": a, "hold": b, "start": str(sl(r).dropna().index[0].date())}
    return out


def state(close: pd.Series, asset: str) -> dict | None:
    """Today's reading: price vs the 200-day average, the 12-month return, volatility and the allowed share."""
    if len(close) < MIN_DAYS:
        return None
    ppy = PPY[asset]
    sma = float(close.rolling(SMA_DAYS).mean().iloc[-1])
    px = float(close.iloc[-1])
    vol = float(close.pct_change().rolling(VOL_DAYS).std().iloc[-1] * np.sqrt(ppy))
    up = px > sma
    above = (close > close.rolling(SMA_DAYS).mean()).iloc[SMA_DAYS:]
    flips = above[above != above.shift()].index
    since = (close.index[-1] - flips[-1]).days if len(flips) > 1 else None  # days in the present state
    mom = close.pct_change(MOM_DAYS[asset]).iloc[-1]
    return {"asof": str(close.index[-1].date()), "price": px, "sma200": sma, "dist_pct": (px / sma - 1) * 100,
            "up": bool(up), "days_in_state": since, "mom12m_pct": None if pd.isna(mom) else float(mom * 100),
            "vol60_pct": vol * 100, "exposure": float(min(1.0, TARGET_VOL / vol)) if up and vol else 0.0}


def evaluate(close: pd.Series, asset: str) -> dict:
    close = close.dropna()
    close = close[~close.index.duplicated()].sort_index()
    st = state(close, asset)
    return {"state": st, "record": record(close, asset) if st else {}}
