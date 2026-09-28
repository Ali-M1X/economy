"""BLS Public Data API v2 (keyless works with lower limits; BLS_API_KEY raises them)."""

from __future__ import annotations

import pandas as pd

from collectors.base import SeriesResult, normalize
from core.http import SourceError, session
from core.registry import SeriesMeta
from core.settings import get_settings
from core.timeutil import utcnow

API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def parse(payload: dict, series_id: str) -> pd.DataFrame:
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise SourceError("bls", f"{payload.get('status')}: {payload.get('message')}")
    series = [s for s in payload["Results"]["series"] if s["seriesID"] == series_id]
    if not series:
        raise SourceError("bls", f"{series_id} missing from response")
    rows = [
        {"date": f"{d['year']}-{d['period'][1:]}-01", "value": d["value"]}
        for d in series[0]["data"] if d["period"].startswith("M") and d["period"] != "M13"
    ]
    return pd.DataFrame(rows, columns=["date", "value"])


def fetch(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    start_year = pd.Timestamp(start or meta.start).year
    end_year = utcnow().year
    # Keyless requests are limited to 10 years per call.
    start_year = max(start_year, end_year - (19 if get_settings().bls_api_key else 9))
    body = {"seriesid": [meta.source_id], "startyear": str(start_year), "endyear": str(end_year)}
    if get_settings().bls_api_key:
        body["registrationkey"] = get_settings().bls_api_key
    try:
        resp = session().post(API, json=body, timeout=get_settings().http_timeout)
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise SourceError("bls", f"{exc.__class__.__name__}: {exc}") from exc
    df, missing = normalize(parse(payload, meta.source_id), "date", "value")
    return SeriesResult(key=meta.key, data=df, source_url=API + meta.source_id, missing_values=missing)
