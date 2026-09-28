"""New York Fed Markets Data API (reference rates) — used to cross-check FRED."""

from __future__ import annotations

import pandas as pd

from collectors.base import SeriesResult, normalize
from core.http import SourceError, get_json
from core.registry import SeriesMeta
from core.timeutil import utcnow

API = "https://markets.newyorkfed.org/api/rates/{path}/search.json"


def parse(payload: dict) -> pd.DataFrame:
    rates = payload.get("refRates")
    if rates is None:
        raise SourceError("nyfed", f"no refRates in payload: {str(payload)[:200]}")
    return pd.DataFrame([{"date": r["effectiveDate"], "value": r.get("percentRate")} for r in rates],
                        columns=["date", "value"])


def fetch(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    # The API is meant for recent windows; two years is plenty for cross-checking.
    start = start or (utcnow() - pd.Timedelta(days=730)).date().isoformat()
    url = API.format(path=meta.source_id)
    payload = get_json("nyfed", url, params={"startDate": start, "endDate": utcnow().date().isoformat()})
    df, missing = normalize(parse(payload), "date", "value")
    return SeriesResult(key=meta.key, data=df, source_url=url, missing_values=missing)
