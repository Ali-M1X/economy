"""Crypto exchange public endpoints: candles, order books, open interest, funding, liquidations.

Binance.com blocks US IPs (GitHub Actions runners are in the US), so the spot fallback chain starts
with data-api.binance.vision (Binance's public market-data mirror) and then OKX, Bybit, Coinbase.
Gold real-time is proxied by PAXG (1 token = 1 fine troy oz held by Paxos) — labelled as a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from core.http import SourceError, get_json

INTERVALS = ("15m", "1h", "4h", "1d", "1w")
CANDLE_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]

# asset → per-venue symbol
SPOT_SYMBOLS = {
    "BTC": {"binance_vision": "BTCUSDT", "okx": "BTC-USDT", "bybit": "BTCUSDT", "coinbase": "BTC-USD"},
    "PAXG": {"binance_vision": "PAXGUSDT", "okx": "PAXG-USDT", "bybit": "PAXGUSDT", "coinbase": "PAXG-USD"},
}
PERP_SYMBOLS = {"BTC": {"bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP", "binance_futures": "BTCUSDT"}}
# Venues that refuse US IPs (GitHub Actions runners). Kept for non-US VPS deployments.
GEO_BLOCKED_FROM_US = {"bybit", "binance_futures"}


def _frame(rows: list[list], ts_unit: str = "ms") -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows], columns=CANDLE_COLUMNS)
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit=ts_unit, utc=True)
    for c in CANDLE_COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


# ─────────────────────────────── candles ───────────────────────────────

def _binance_vision(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rows = get_json("binance_vision", "https://data-api.binance.vision/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)})
    return _frame(rows)


def _okx(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    bar = {"15m": "15m", "1h": "1H", "4h": "4Hutc", "1d": "1Dutc", "1w": "1Wutc"}[interval]
    payload = get_json("okx", "https://www.okx.com/api/v5/market/candles",
                       params={"instId": symbol, "bar": bar, "limit": min(limit, 300)})
    if payload.get("code") != "0":
        raise SourceError("okx", f"{payload.get('code')}: {payload.get('msg')}")
    # OKX rows: ts, o, h, l, c, vol(base), ... ; newest first
    return _frame(payload["data"])


def _bybit(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    iv = {"15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W"}[interval]
    payload = get_json("bybit", "https://api.bybit.com/v5/market/kline",
                       params={"category": "spot", "symbol": symbol, "interval": iv, "limit": min(limit, 1000)})
    if payload.get("retCode") != 0:
        raise SourceError("bybit", f"{payload.get('retCode')}: {payload.get('retMsg')}")
    return _frame(payload["result"]["list"])


def _coinbase(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    gran = {"15m": 900, "1h": 3600, "1d": 86400}.get(interval)
    if gran is None:  # Coinbase has no 4h / 1w granularity; resample from 1h / 1d
        base, rule = ("1h", "4h") if interval == "4h" else ("1d", "W-MON")
        df = _coinbase(symbol, base, limit * (4 if interval == "4h" else 7))
        return resample(df, rule)
    rows = get_json("coinbase", f"https://api.exchange.coinbase.com/products/{symbol}/candles",
                    params={"granularity": gran})
    # Coinbase rows: time, low, high, open, close, volume (seconds)
    rows = [[r[0], r[3], r[2], r[1], r[4], r[5]] for r in rows]
    return _frame(rows, ts_unit="s").tail(limit).reset_index(drop=True)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("ts").resample(rule, label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()}).dropna(subset=["open"])
    return out.reset_index()


def candles_history(asset: str, interval: str = "1h", start: str = "2017-08-17",
                    end: pd.Timestamp | None = None, max_requests: int = 400) -> pd.DataFrame:
    """Full candle history from Binance's public market-data mirror (1000 bars per request, paginated
    forward). Used for event studies and backtests. Gaps (exchange outages) are left as gaps — callers
    see them via `hourly_gaps()`; nothing is interpolated."""
    ms = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    stop = int((end or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    frames = []
    for _ in range(max_requests):
        rows = get_json("binance_vision", "https://data-api.binance.vision/api/v3/klines",
                        params={"symbol": SPOT_SYMBOLS[asset]["binance_vision"], "interval": interval,
                                "startTime": cursor, "limit": 1000})
        if not rows:
            break
        frames.append(_frame(rows))
        cursor = int(rows[-1][0]) + ms
        if cursor > stop or len(rows) < 1000:
            break
    if not frames:
        raise SourceError("binance_vision", f"no {asset} {interval} history from {start}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    # the last bar may still be forming; keep only closed bars
    return df[df["ts"] + pd.Timedelta(milliseconds=ms) <= pd.Timestamp.now(tz="UTC")].reset_index(drop=True)


def hourly_gaps(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:  # freq: "1h", "15min", ...
    """Missing bars (start, end, bars) in a candle frame."""
    full = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq, tz="UTC")
    missing = full.difference(pd.DatetimeIndex(df["ts"]))
    if missing.empty:
        return pd.DataFrame(columns=["start", "end", "bars"])
    s = pd.Series(missing)
    grp = (s.diff() != pd.Timedelta(freq)).cumsum()
    return pd.DataFrame([{"start": g.iloc[0], "end": g.iloc[-1], "bars": len(g)} for _, g in s.groupby(grp)])


CANDLE_PROVIDERS: dict[str, Callable[[str, str, int], pd.DataFrame]] = {
    "binance_vision": _binance_vision, "okx": _okx, "bybit": _bybit, "coinbase": _coinbase,
}


@dataclass
class VenueResult:
    venue: str
    data: pd.DataFrame | dict
    errors: dict[str, str]


def candles(asset: str, interval: str, limit: int = 500, venues: tuple[str, ...] | None = None) -> VenueResult:
    """Fetch candles from the first venue that answers. Errors of skipped venues are returned."""
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {INTERVALS}")
    errors: dict[str, str] = {}
    for venue in venues or tuple(CANDLE_PROVIDERS):
        symbol = SPOT_SYMBOLS[asset].get(venue)
        if not symbol:
            continue
        try:
            df = CANDLE_PROVIDERS[venue](symbol, interval, limit)
            if df.empty:
                raise SourceError(venue, "no candles")
            return VenueResult(venue, df, errors)
        except SourceError as exc:
            errors[venue] = str(exc)
    raise SourceError("crypto", f"all venues failed for {asset} {interval}: {errors}")


# ─────────────────────────────── order books ───────────────────────────────

def _book_binance_vision(symbol: str) -> dict:
    p = get_json("binance_vision", "https://data-api.binance.vision/api/v3/depth", params={"symbol": symbol, "limit": 5000})
    return {"bids": p["bids"], "asks": p["asks"]}


def _book_okx(symbol: str) -> dict:
    p = get_json("okx", "https://www.okx.com/api/v5/market/books", params={"instId": symbol, "sz": 400})
    if p.get("code") != "0":
        raise SourceError("okx", p.get("msg", "error"))
    d = p["data"][0]
    return {"bids": [r[:2] for r in d["bids"]], "asks": [r[:2] for r in d["asks"]]}


def _book_bybit(symbol: str) -> dict:
    p = get_json("bybit", "https://api.bybit.com/v5/market/orderbook", params={"category": "spot", "symbol": symbol, "limit": 200})
    if p.get("retCode") != 0:
        raise SourceError("bybit", p.get("retMsg", "error"))
    return {"bids": p["result"]["b"], "asks": p["result"]["a"]}


def _book_coinbase(symbol: str) -> dict:
    p = get_json("coinbase", f"https://api.exchange.coinbase.com/products/{symbol}/book", params={"level": 2})
    return {"bids": [r[:2] for r in p["bids"]], "asks": [r[:2] for r in p["asks"]]}


BOOK_PROVIDERS = {"binance_vision": _book_binance_vision, "okx": _book_okx, "bybit": _book_bybit, "coinbase": _book_coinbase}


def order_books(asset: str) -> tuple[dict[str, dict], dict[str, str]]:
    """Order books from every reachable venue (for cross-venue aggregation)."""
    books, errors = {}, {}
    for venue, fn in BOOK_PROVIDERS.items():
        symbol = SPOT_SYMBOLS[asset].get(venue)
        if not symbol:
            continue
        try:
            books[venue] = fn(symbol)
        except (SourceError, KeyError, IndexError) as exc:
            errors[venue] = str(exc)
    return books, errors


def aggregate_depth(books: dict[str, dict], bucket: float) -> pd.DataFrame:
    """Sum resting size (base units and USD notional) per price bucket across venues."""
    rows = []
    for venue, book in books.items():
        for side in ("bids", "asks"):
            for price, qty in book[side]:
                p, q = float(price), float(qty)
                rows.append({"side": side[:-1], "bucket": (p // bucket) * bucket, "qty": q, "usd": p * q, "venue": venue})
    if not rows:
        return pd.DataFrame(columns=["side", "bucket", "qty", "usd", "venues"])
    df = pd.DataFrame(rows)
    return (df.groupby(["side", "bucket"])
              .agg(qty=("qty", "sum"), usd=("usd", "sum"), venues=("venue", "nunique"))
              .reset_index())


# ──────────────────────── derivatives (OI, funding) ────────────────────────

def _deriv_bybit(symbol: str) -> dict:
    p = get_json("bybit", "https://api.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": symbol})
    if p.get("retCode") != 0:
        raise SourceError("bybit", p.get("retMsg", "error"))
    t = p["result"]["list"][0]
    return {"open_interest": float(t["openInterest"]), "open_interest_usd": float(t["openInterestValue"]),
            "funding_rate": float(t["fundingRate"]), "mark_price": float(t["markPrice"]),
            "next_funding_ts": pd.Timestamp(int(t["nextFundingTime"]), unit="ms", tz="UTC")}


def _deriv_okx(symbol: str) -> dict:
    oi = get_json("okx", "https://www.okx.com/api/v5/public/open-interest", params={"instType": "SWAP", "instId": symbol})
    fr = get_json("okx", "https://www.okx.com/api/v5/public/funding-rate", params={"instId": symbol})
    mk = get_json("okx", "https://www.okx.com/api/v5/public/mark-price", params={"instType": "SWAP", "instId": symbol})
    if "0" != oi.get("code") or "0" != fr.get("code") or "0" != mk.get("code"):
        raise SourceError("okx", f"{oi.get('msg')} / {fr.get('msg')} / {mk.get('msg')}")
    o, f, m = oi["data"][0], fr["data"][0], mk["data"][0]
    mark = float(m["markPx"])
    return {"open_interest": float(o["oiCcy"]), "open_interest_usd": float(o.get("oiUsd") or float(o["oiCcy"]) * mark),
            "funding_rate": float(f["fundingRate"]), "mark_price": mark,
            "next_funding_ts": pd.Timestamp(int(f["nextFundingTime"] or f["fundingTime"]), unit="ms", tz="UTC")}


def _deriv_binance(symbol: str) -> dict:
    # fapi.binance.com answers HTTP 451 to US IPs; kept for non-US VPS deployments.
    oi = get_json("binance_futures", "https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": symbol})
    pi = get_json("binance_futures", "https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": symbol})
    mark = float(pi["markPrice"])
    return {"open_interest": float(oi["openInterest"]), "open_interest_usd": float(oi["openInterest"]) * mark,
            "funding_rate": float(pi["lastFundingRate"]), "mark_price": mark,
            "next_funding_ts": pd.Timestamp(int(pi["nextFundingTime"]), unit="ms", tz="UTC")}


DERIV_PROVIDERS = {"okx": _deriv_okx, "bybit": _deriv_bybit, "binance_futures": _deriv_binance}


def derivatives(asset: str = "BTC") -> tuple[dict[str, dict], dict[str, str]]:
    out, errors = {}, {}
    for venue, fn in DERIV_PROVIDERS.items():
        try:
            out[venue] = fn(PERP_SYMBOLS[asset][venue])
        except (SourceError, KeyError, IndexError, ValueError) as exc:
            errors[venue] = str(exc)
    return out, errors


def open_interest_history(asset: str = "BTC", interval: str = "1h", limit: int = 100) -> pd.DataFrame:
    """OKX perpetual open-interest history — input for liquidation-cluster estimation.

    Bybit's equivalent endpoint is geo-blocked from US IPs (GitHub runners), so OKX is primary.
    Returns ts, open_interest (base units, e.g. BTC), open_interest_usd.
    """
    period = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}[interval]
    p = get_json("okx", "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history",
                 params={"instId": PERP_SYMBOLS[asset]["okx"], "period": period, "limit": min(limit, 100)})
    if p.get("code") != "0" or not p.get("data"):
        raise SourceError("okx", f"open-interest-history: {p.get('code')} {p.get('msg')}")
    return parse_okx_oi_history(p["data"])


def parse_okx_oi_history(rows: list[list]) -> pd.DataFrame:
    # rows (newest first): [ts_ms, oi_contracts, oi_ccy, oi_usd]
    df = pd.DataFrame([r[:4] for r in rows], columns=["ts", "oi_contracts", "open_interest", "open_interest_usd"])
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    for c in ("open_interest", "open_interest_usd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ts", "open_interest", "open_interest_usd"]].sort_values("ts").reset_index(drop=True)


# ─────────────────────────────── liquidations ───────────────────────────────

def recent_liquidations_okx(asset: str = "BTC") -> pd.DataFrame:
    """Realized liquidations from OKX's public REST endpoint (recent window only — store over time)."""
    p = get_json("okx", "https://www.okx.com/api/v5/public/liquidation-orders",
                 params={"instType": "SWAP", "uly": f"{asset}-USDT", "state": "filled", "limit": 100})
    if p.get("code") != "0":
        raise SourceError("okx", p.get("msg", "error"))
    rows = []
    for block in p.get("data", []):
        for d in block.get("details", []):
            rows.append({"venue": "okx", "asset": asset, "ts": pd.Timestamp(int(d["ts"]), unit="ms", tz="UTC"),
                         # posSide=long + side=sell ⇒ a long was liquidated
                         "side": d.get("posSide") or ("long" if d.get("side") == "sell" else "short"),
                         "price": float(d["bkPx"]), "qty_contracts": float(d["sz"]),
                         "raw_id": f"okx-{block.get('instId')}-{d['ts']}-{d['bkPx']}-{d['sz']}"})
    return pd.DataFrame(rows)


async def stream_liquidations_bybit(asset: str = "BTC", seconds: int = 60) -> list[dict]:
    """Capture Bybit's public `allLiquidation` WebSocket stream for `seconds` (long-running on a VPS)."""
    import asyncio
    import json

    import websockets

    out: list[dict] = []
    async with websockets.connect("wss://stream.bybit.com/v5/public/linear", open_timeout=15) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{PERP_SYMBOLS[asset]['bybit']}"]}))
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while (remaining := end - loop.time()) > 0:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except asyncio.TimeoutError:
                break
            for d in msg.get("data", []) if isinstance(msg.get("data"), list) else []:
                out.append({"venue": "bybit", "asset": asset, "ts": pd.Timestamp(int(d["T"]), unit="ms", tz="UTC"),
                            # Bybit: S=Buy means a short position was liquidated
                            "side": "short" if d["S"] == "Buy" else "long",
                            "price": float(d["p"]), "qty": float(d["v"]),
                            "raw_id": f"bybit-{d['s']}-{d['T']}-{d['p']}-{d['v']}"})
    return out
