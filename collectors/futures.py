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


def realized_month_average(effr: pd.DataFrame, year: int, month: int, today: date) -> tuple[float, int, int]:
    """CME settles a ZQ contract on the average daily EFFR over *calendar* days of the month (non-business
    days carry the prior business day's rate — FRED's DFF already includes them). Days not yet published
    are filled with the latest known rate, i.e. "no change for the rest of the month".
    Returns (expected_average, days_known, days_in_month)."""
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    days = pd.date_range(start, end, freq="D")
    s = effr.set_index("date")["value"].sort_index()
    known = s[(s.index >= start) & (s.index <= min(end, pd.Timestamp(today)))]
    last = float(s[s.index <= min(end, pd.Timestamp(today))].iloc[-1])
    filled = known.reindex(days).ffill().fillna(last)
    return float(filled.mean()), len(known), len(days)


def check_current_month(curve_df: pd.DataFrame, effr: pd.DataFrame, today: date,
                        fomc_dates: list[date] | None = None, tolerance: float = 0.05) -> tuple[str, str]:
    """Validate the current-month ZQ contract against realized EFFR. With no FOMC decision left in the month,
    the contract can only differ from realized-so-far + unchanged rate by a few bp."""
    row = curve_df[curve_df["contract_month"] == date(today.year, today.month, 1)]
    if row.empty:
        return "warn", "current-month contract not quoted"
    implied = float(row["implied_rate"].iloc[0])
    expected, known, n = realized_month_average(effr, today.year, today.month, today)
    diff = implied - expected
    fomc_left = [d for d in (fomc_dates or []) if d.year == today.year and d.month == today.month and d >= today]
    msg = (f"{row['contract'].iloc[0]} implies {implied:.3f}% vs realized EFFR average {expected:.3f}% "
           f"({known}/{n} days published, rest at latest EFFR); diff {diff * 100:+.1f}bp")
    if fomc_left:
        return "ok", msg + f" (FOMC {fomc_left[0]} still ahead this month — difference may be priced policy)"
    return ("ok" if abs(diff) <= tolerance else "fail"), msg
