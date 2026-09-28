"""Yahoo Finance chart API (free, delayed). Used for indices, futures, ETFs and FX.

Daily bars are stamped with the *exchange-local* trading date (Yahoo returns UTC epoch at session
open; converting with the exchange timezone avoids off-by-one-day errors).
"""

from __future__ import annotations

import pandas as pd

from collectors.base import SeriesResult, normalize
from core.http import SourceError, get_json
from core.registry import SeriesMeta
from core.timeutil import utcnow

HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
CHART = "/v8/finance/chart/{symbol}"


def parse_chart(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Returns OHLCV frame indexed by exchange-local date/time and the chart meta."""
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise SourceError("yahoo", f"chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise SourceError("yahoo", "empty chart result")
    res = results[0]
    meta = res.get("meta", {})
    ts = res.get("timestamp") or []
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    tz = meta.get("exchangeTimezoneName") or "UTC"
    utc = pd.to_datetime(ts, unit="s", utc=True)
    idx = utc.tz_convert(tz).tz_localize(None)
    df = pd.DataFrame({k: quote.get(k, [None] * len(ts)) for k in ("open", "high", "low", "close", "volume")})
    df.insert(0, "time", idx)  # exchange-local wall clock (daily bars: the trading date)
    df.insert(1, "ts_utc", utc)  # exact bar start, for intraday work
    return df, meta


def chart(symbol: str, *, interval: str = "1d", start: str | None = None, range_: str | None = None) -> tuple[pd.DataFrame, dict, str]:
    params = {"interval": interval, "includePrePost": "false", "events": "div,split"}
    if range_:
        params["range"] = range_
    else:
        params["period1"] = int(pd.Timestamp(start or "2000-01-01", tz="UTC").timestamp())
        params["period2"] = int(utcnow().timestamp())
    last_err: Exception | None = None
    for host in HOSTS:
        url = host + CHART.format(symbol=symbol)
        try:
            df, meta = parse_chart(get_json("yahoo", url, params=params))
            return df, meta, url
        except SourceError as exc:
            last_err = exc
    raise last_err  # type: ignore[misc]


def fetch(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    df, cmeta, url = chart(meta.source_id, start=start or meta.start)
    df["date"] = df["time"].dt.normalize()
    # Yahoo sometimes appends an intraday "live" bar for today; keep one row per date (the last).
    out, missing = normalize(df, "date", "close")
    res = SeriesResult(key=meta.key, data=out, source_url=url, missing_values=missing,
                       source_units=cmeta.get("currency"))
    if cmeta.get("regularMarketTime"):
        res.source_last_updated = pd.Timestamp(cmeta["regularMarketTime"], unit="s", tz="UTC").to_pydatetime()
    return res
