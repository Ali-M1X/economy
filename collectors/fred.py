"""FRED / ALFRED collector.

With FRED_API_KEY: uses the official API (observations, series metadata, ALFRED vintages, release dates).
Without it: falls back to the public keyless `fredgraph.csv` download (latest vintage only, no metadata).
"""

from __future__ import annotations

import io
from datetime import date, datetime

import pandas as pd

from collectors.base import SeriesResult, normalize
from core.http import SourceError, get, get_json
from core.registry import SeriesMeta
from core.settings import get_settings
from core.timeutil import UTC

API = "https://api.stlouisfed.org/fred"
GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SOURCE = "fred"


def _api(path: str, **params) -> dict:
    key = get_settings().fred_api_key
    if not key:
        raise SourceError(SOURCE, "FRED_API_KEY not set")
    return get_json(SOURCE, f"{API}/{path}", params={**params, "api_key": key, "file_type": "json"})


def _parse_last_updated(s: str | None) -> datetime | None:
    # FRED format: "2026-09-26 07:52:03-05"
    if not s:
        return None
    try:
        return datetime.strptime(s + "00", "%Y-%m-%d %H:%M:%S%z").astimezone(UTC)
    except ValueError:
        return None


def parse_api_observations(payload: dict) -> tuple[pd.DataFrame, int]:
    obs = payload.get("observations")
    if obs is None:
        raise SourceError(SOURCE, f"no 'observations' in payload: {str(payload)[:200]}")
    df = pd.DataFrame(obs, columns=["date", "value"]) if obs else pd.DataFrame(columns=["date", "value"])
    return normalize(df, "date", "value")


def parse_graph_csv(text: str, series_id: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    if date_col not in df.columns or series_id not in df.columns:
        raise SourceError(SOURCE, f"unexpected fredgraph columns {list(df.columns)} for {series_id}")
    return normalize(df, date_col, series_id)


def fetch(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    start = start or meta.start
    if get_settings().fred_api_key:
        payload = _api("series/observations", series_id=meta.source_id, observation_start=start)
        df, missing = parse_api_observations(payload)
        info = _api("series", series_id=meta.source_id).get("seriess", [{}])[0]
        return SeriesResult(
            key=meta.key, data=df, source_url=f"{API}/series/observations?series_id={meta.source_id}",
            missing_values=missing,
            source_last_updated=_parse_last_updated(info.get("last_updated")),
            source_units=info.get("units"), source_frequency=info.get("frequency_short"),
        )
    resp = get(SOURCE, GRAPH_CSV, params={"id": meta.source_id, "cosd": start})
    df, missing = parse_graph_csv(resp.text, meta.source_id)
    return SeriesResult(key=meta.key, data=df, source_url=resp.url, missing_values=missing,
                        notes=["keyless fredgraph.csv (no metadata / vintages)"])


def fetch_vintages(meta: SeriesMeta, start: str | None = None) -> pd.DataFrame:
    """ALFRED: every value as it was known over time (for look-ahead-free backtests).

    Returns columns: date, value, realtime_start, realtime_end (datetime64). realtime_start is the
    first day the value was public; use `value_as_of()` to reconstruct any historical vintage.
    """
    payload = _api("series/observations", series_id=meta.source_id, observation_start=start or meta.start,
                   realtime_start="1776-07-04", realtime_end="9999-12-31", output_type=1)
    obs = payload.get("observations", [])
    df = pd.DataFrame(obs, columns=["date", "value", "realtime_start", "realtime_end"])
    df["value"] = pd.to_numeric(df["value"].replace({".": None}), errors="coerce")
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    # "9999-12-31" means "still current" and overflows datetime64[ns]; map it to NaT.
    df["realtime_end"] = pd.to_datetime(df["realtime_end"].where(df["realtime_end"] != "9999-12-31"))
    return df.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def value_as_of(vintages: pd.DataFrame, as_of: date | datetime | str) -> pd.DataFrame:
    """Reconstruct the series exactly as it was published on `as_of` (no look-ahead)."""
    t = pd.Timestamp(as_of)
    known = vintages[(vintages["realtime_start"] <= t) & (vintages["realtime_end"].isna() | (vintages["realtime_end"] >= t))]
    return known[["date", "value"]].drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def series_release(series_id: str) -> dict | None:
    rel = _api("series/release", series_id=series_id).get("releases", [])
    return rel[0] if rel else None


def release_dates(release_id: int, start: str, end: str) -> list[date]:
    payload = _api("release/dates", release_id=release_id, realtime_start=start, realtime_end=end,
                   include_release_dates_with_no_data="true", sort_order="asc", limit=1000)
    return [date.fromisoformat(d["date"]) for d in payload.get("release_dates", [])]
