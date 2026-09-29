"""Binance public data archive (data.binance.vision) — free historical futures positioning data.

- Daily 5-minute metrics for the USDⓈ-M perpetual (from 2021-12): open interest (`sum_open_interest`,
  `sum_open_interest_value`) and the taker buy/sell volume ratio.
- Monthly funding-rate files (8-hourly funding, from 2019-09).

These are static files, not the (US-blocked) futures API. They are the only free source with enough history to
backtest open-interest / funding shifts before releases. Files are fetched incrementally into a persistent folder
(`archive_dir`, kept between runs by the workflow cache): a file is downloaded once, and days that do not exist yet
(404) are remembered so they are not requested again until they could exist. `max_files` bounds one run; the first
runs fill the history over several days.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from core.http import SourceError, get

BASE = "https://data.binance.vision/data/futures/um"
METRICS_START = pd.Timestamp("2021-12-01")
FUNDING_START = pd.Timestamp("2019-09-01")


def _csv_from_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("utf-8")
    first = text.split("\n", 1)[0]
    has_header = any(c.isalpha() for c in first.replace("e-", "").replace("E-", ""))
    return pd.read_csv(io.StringIO(text), header=0 if has_header else None)


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """→ ts (UTC, 5-min), open_interest (base), open_interest_usd, taker_ls_ratio."""
    if "create_time" not in df.columns:  # headerless legacy files
        df = df.set_axis(["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                          "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                          "count_long_short_ratio", "sum_taker_long_short_vol_ratio"][:df.shape[1]], axis=1)
    return pd.DataFrame({
        "ts": pd.to_datetime(df["create_time"], utc=True),
        "open_interest": pd.to_numeric(df["sum_open_interest"], errors="coerce"),
        "open_interest_usd": pd.to_numeric(df["sum_open_interest_value"], errors="coerce"),
        "taker_ls_ratio": pd.to_numeric(df.get("sum_taker_long_short_vol_ratio"), errors="coerce"),
    }).dropna(subset=["open_interest"])


def parse_funding(df: pd.DataFrame) -> pd.DataFrame:
    """→ ts (UTC, funding time), funding_rate."""
    if "calc_time" not in df.columns:
        df = df.set_axis(["calc_time", "funding_interval_hours", "last_funding_rate"][:df.shape[1]], axis=1)
    return pd.DataFrame({"ts": pd.to_datetime(pd.to_numeric(df["calc_time"]), unit="ms", utc=True),
                         "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce")}).dropna()


def _fetch(url: str) -> bytes | None:
    try:
        return get("binance_archive", url, timeout=60).content
    except SourceError as e:
        if "HTTP 404" in str(e):
            return None
        raise


def update(archive_dir: Path, symbol: str = "BTCUSDT", today: pd.Timestamp | None = None, max_files: int = 400) -> dict:
    """Fetch missing daily metrics and monthly funding files; returns counts. Safe to call every run."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    today = (today or pd.Timestamp.now(tz="UTC")).tz_localize(None).normalize()
    idx_p = archive_dir / f"{symbol}_index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {"metrics": [], "funding": [], "missing": []}
    have, missing = set(idx["metrics"]) | set(idx["funding"]), set(idx["missing"])
    todo = []
    # daily files appear the next day; newest first so a partial run still covers the recent past
    for d in pd.date_range(METRICS_START, today - pd.Timedelta(days=1), freq="D")[::-1]:
        k = f"m:{d:%Y-%m-%d}"
        if k not in have and not (k in missing and d < today - pd.Timedelta(days=7)):
            todo.append(("metrics", k, f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip"))
    for m in pd.date_range(FUNDING_START, today - pd.DateOffset(months=1), freq="MS")[::-1]:
        k = f"f:{m:%Y-%m}"
        if k not in have and not (k in missing and m < today - pd.DateOffset(months=2)):
            todo.append(("funding", k, f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m:%Y-%m}.zip"))
    new = {"metrics": [], "funding": []}
    fetched = 0
    for kind, key, url in todo[:max_files]:
        raw = _fetch(url)
        fetched += 1
        if raw is None:
            missing.add(key)
            continue
        df = (parse_metrics if kind == "metrics" else parse_funding)(_csv_from_zip(raw))
        new[kind].append(df)
        idx[kind].append(key)
        missing.discard(key)
    for kind in ("metrics", "funding"):
        if new[kind]:
            p = archive_dir / f"{symbol}_{kind}.csv.gz"
            old = pd.read_csv(p, parse_dates=["ts"]) if p.exists() else pd.DataFrame()
            allf = pd.concat([old, *new[kind]], ignore_index=True)
            allf["ts"] = pd.to_datetime(allf["ts"], utc=True)
            allf.drop_duplicates("ts").sort_values("ts").to_csv(p, index=False)
    idx["missing"] = sorted(missing)
    idx_p.write_text(json.dumps(idx))
    return {"fetched": fetched, "remaining": max(len(todo) - max_files, 0),
            "metrics_days": len(idx["metrics"]), "funding_months": len(idx["funding"])}


def load(archive_dir: Path, symbol: str = "BTCUSDT") -> tuple[pd.DataFrame, pd.DataFrame]:
    out = []
    for kind in ("metrics", "funding"):
        p = archive_dir / f"{symbol}_{kind}.csv.gz"
        df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        out.append(df.sort_values("ts").reset_index(drop=True))
    return out[0], out[1]
