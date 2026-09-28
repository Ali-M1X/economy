"""Common result type for series collectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from core.timeutil import utcnow

OBS_COLUMNS = ["date", "value"]


@dataclass
class SeriesResult:
    key: str
    data: pd.DataFrame  # columns: date (datetime64, naive, period date), value (float64)
    source_url: str  # the exact endpoint that produced the data
    fetched_at: datetime = field(default_factory=utcnow)
    missing_values: int = 0  # observations the source reported as missing ('.', '', null) — dropped, never filled
    source_last_updated: datetime | None = None
    source_units: str | None = None
    source_frequency: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def last_date(self):
        return None if self.data.empty else self.data["date"].iloc[-1]

    @property
    def last_value(self):
        return None if self.data.empty else float(self.data["value"].iloc[-1])


def normalize(df: pd.DataFrame, date_col: str, value_col: str) -> tuple[pd.DataFrame, int]:
    """Coerce to the canonical (date, value) frame. Returns the frame and the count of missing values dropped."""
    dates = pd.to_datetime(df[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out = pd.DataFrame({
        "date": dates,
        "value": pd.to_numeric(df[value_col].replace({".": None, "": None, "ND": None}), errors="coerce"),
    })
    missing = int(out["value"].isna().sum())
    out = out.dropna(subset=["value"])
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    out["value"] = out["value"].astype("float64")
    out["date"] = out["date"].astype("datetime64[ns]")
    return out, missing
