"""Rate-expectation futures curves from Yahoo (delayed, free).

- 30-Day Fed Funds futures (CBOT ZQ): price = 100 − average EFFR expected over the contract month.
- 3-Month SOFR futures (CME SR3): price = 100 − compounded SOFR expected over the reference quarter.
Specific contracts use Yahoo's `ROOT<month-code><yy>.<exchange>` symbols; unavailable contracts are
reported, not guessed.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from collectors import yahoo
from core.http import SourceError
from core.timeutil import utcnow

MONTH_CODES = "FGHJKMNQUVXZ"
QUARTERLY = (3, 6, 9, 12)


def contract_symbol(root: str, year: int, month: int, exchange: str) -> str:
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}.{exchange}"


def upcoming_months(n: int, start: date | None = None, quarterly: bool = False) -> list[tuple[int, int]]:
    d = start or utcnow().date()
    y, m = d.year, d.month
    out = []
    while len(out) < n:
        if not quarterly or m in QUARTERLY:
            out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _last_price(symbol: str) -> tuple[float, pd.Timestamp]:
    df, meta, _ = yahoo.chart(symbol, interval="1d", range_="5d")
    df = df.dropna(subset=["close"])
    price = meta.get("regularMarketPrice") or (df["close"].iloc[-1] if not df.empty else None)
    if price is None:
        raise SourceError("yahoo", f"no price for {symbol}")
    ts = pd.Timestamp(meta.get("regularMarketTime", 0), unit="s", tz="UTC")
    return float(price), ts


def curve(root: str = "ZQ", months: int = 13) -> tuple[pd.DataFrame, dict[str, str]]:
    exchange, quarterly = {"ZQ": ("CBT", False), "SR3": ("CME", True)}[root]
    rows, errors = [], {}
    for y, m in upcoming_months(months, quarterly=quarterly):
        sym = contract_symbol(root, y, m, exchange)
        try:
            price, ts = _last_price(sym)
            rows.append({"contract": sym, "root": root, "contract_month": date(y, m, 1), "price": price,
                         "implied_rate": round(100.0 - price, 4), "quote_ts": ts})
        except SourceError as exc:
            errors[sym] = str(exc)
    return pd.DataFrame(rows), errors
