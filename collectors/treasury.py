"""US Treasury daily par yield curve (home.treasury.gov) and Daily Treasury Statement (fiscaldata)."""

from __future__ import annotations

import io

import pandas as pd

from collectors.base import SeriesResult, normalize
from core.http import SourceError, get, get_json
from core.registry import SeriesMeta
from core.timeutil import utcnow

CURVE_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
             "daily-treasury-rates.csv/{year}/all")
DTS_TGA = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"


def parse_curve_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if "Date" not in df.columns:
        raise SourceError("treasury", f"unexpected columns {list(df.columns)[:6]}")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    return df


def fetch_curve(start_year: int, end_year: int | None = None) -> tuple[pd.DataFrame, str]:
    end_year = end_year or utcnow().year
    frames, url = [], ""
    for year in range(start_year, end_year + 1):
        resp = get("treasury", CURVE_CSV.format(year=year),
                   params={"type": "daily_treasury_yield_curve", "field_tdr_date_value": year, "_format": "csv"})
        url = resp.url
        if resp.text.strip():
            frames.append(parse_curve_csv(resp.text))
    if not frames:
        raise SourceError("treasury", "empty yield-curve download")
    return pd.concat(frames, ignore_index=True), url


def fetch(meta: SeriesMeta, start: str | None = None, _cache: dict = {}) -> SeriesResult:  # noqa: B006
    start_year = pd.Timestamp(start or meta.start).year
    if start_year not in _cache:  # one download serves all tenors in a run
        _cache[start_year] = fetch_curve(start_year)
    curve, url = _cache[start_year]
    if meta.source_id not in curve.columns:
        raise SourceError("treasury", f"tenor column {meta.source_id!r} missing; have {list(curve.columns)}")
    df, missing = normalize(curve, "Date", meta.source_id)
    return SeriesResult(key=meta.key, data=df, source_url=url, missing_values=missing)


def parse_dts_tga(rows: list[dict]) -> pd.DataFrame:
    """Pick the TGA closing balance per day. Since 2022 the DTS reports it in a row named
    'Treasury General Account (TGA) Closing Balance' (value in open_today_bal); earlier rows name the
    account 'Federal Reserve Account' with close_today_bal."""
    out = []
    for r in rows:
        at = (r.get("account_type") or "").lower()
        if "closing balance" in at and "tga" in at:
            val = r.get("open_today_bal")
            if val in (None, "", "null"):
                val = r.get("close_today_bal")
        elif at == "federal reserve account":
            val = r.get("close_today_bal")
        else:
            continue
        out.append({"date": r["record_date"], "value": val})
    return pd.DataFrame(out, columns=["date", "value"])


def fetch_tga(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    rows, page, url = [], 1, DTS_TGA
    while True:
        payload = get_json("fiscaldata", DTS_TGA, params={
            "filter": f"record_date:gte:{start or meta.start}", "sort": "record_date",
            "page[size]": 5000, "page[number]": page,
            "fields": "record_date,account_type,close_today_bal,open_today_bal"})
        rows.extend(payload.get("data", []))
        if page >= int(payload.get("meta", {}).get("total-pages", 1)):
            break
        page += 1
    df, missing = normalize(parse_dts_tga(rows), "date", "value")
    return SeriesResult(key=meta.key, data=df, source_url=url, missing_values=missing)
