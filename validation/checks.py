"""Data-quality checks. Every check returns a CheckResult; nothing is ever fixed or filled silently."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from core.registry import SeriesMeta

# Calendar days between observations beyond which a gap is suspicious, per frequency.
MAX_GAP_DAYS = {"D": 10, "W": 15, "M": 40, "Q": 100}


@dataclass
class CheckResult:
    name: str
    status: str  # ok | warn | fail
    message: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def check_schema(df: pd.DataFrame) -> CheckResult:
    if list(df.columns[:2]) != ["date", "value"]:
        return CheckResult("schema", "fail", f"columns {list(df.columns)}")
    if df.empty:
        return CheckResult("schema", "fail", "no observations")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]) or not pd.api.types.is_float_dtype(df["value"]):
        return CheckResult("schema", "fail", f"dtypes {df.dtypes.to_dict()}")
    if df["date"].duplicated().any():
        return CheckResult("schema", "fail", "duplicate dates")
    if not df["date"].is_monotonic_increasing:
        return CheckResult("schema", "fail", "dates not sorted")
    if not np.isfinite(df["value"]).all():
        return CheckResult("schema", "fail", "non-finite values")
    return CheckResult("schema", "ok", f"{len(df)} rows")


def check_range(df: pd.DataFrame, meta: SeriesMeta) -> CheckResult:
    bad = df[(df["value"] < meta.valid_min) | (df["value"] > meta.valid_max)]
    if bad.empty:
        return CheckResult("range", "ok", f"all in [{meta.valid_min}, {meta.valid_max}]")
    sample = ", ".join(f"{d:%Y-%m-%d}={v:g}" for d, v in bad.head(3).itertuples(index=False))
    return CheckResult("range", "fail", f"{len(bad)} values outside [{meta.valid_min}, {meta.valid_max}]: {sample}")


def staleness_days(df: pd.DataFrame, today: date) -> int | None:
    return None if df.empty else (pd.Timestamp(today) - df["date"].iloc[-1]).days


def check_staleness(df: pd.DataFrame, meta: SeriesMeta, today: date) -> CheckResult:
    age = staleness_days(df, today)
    if age is None:
        return CheckResult("staleness", "fail", "no data")
    # Monthly/quarterly observations are dated at period start, so allow one period of dating offset.
    if age > meta.max_stale_days:
        return CheckResult("staleness", "fail", f"last obs {age}d old > {meta.max_stale_days}d")
    return CheckResult("staleness", "ok", f"last obs {age}d old")


def check_weekend_dates(df: pd.DataFrame, meta: SeriesMeta) -> CheckResult:
    """Exchange/business-day daily series must not land on weekends (catches timezone off-by-one bugs)."""
    if meta.frequency != "D" or meta.category == "crypto":
        return CheckResult("weekend_dates", "ok", "n/a")
    recent = df[df["date"] >= df["date"].iloc[-1] - pd.Timedelta(days=730)]
    share = float((recent["date"].dt.dayofweek >= 5).mean()) if len(recent) else 0.0
    if share > 0.02:
        return CheckResult("weekend_dates", "fail", f"{share:.0%} of last-2y dates are weekends (date shift?)")
    return CheckResult("weekend_dates", "ok", f"{share:.1%} weekend dates")


def check_gaps(df: pd.DataFrame, meta: SeriesMeta, lookback_days: int = 730) -> CheckResult:
    limit = MAX_GAP_DAYS.get(meta.frequency)
    if limit is None or len(df) < 2:
        return CheckResult("gaps", "ok", "n/a")
    recent = df[df["date"] >= df["date"].iloc[-1] - pd.Timedelta(days=lookback_days)]
    gaps = recent["date"].diff().dt.days
    worst = gaps.max()
    if pd.notna(worst) and worst > limit:
        at = recent["date"].iloc[int(gaps.values.argmax())]
        return CheckResult("gaps", "warn", f"largest gap {int(worst)}d ending {at:%Y-%m-%d} (> {limit}d)")
    return CheckResult("gaps", "ok", f"largest gap {0 if pd.isna(worst) else int(worst)}d")


def check_cross(a: pd.DataFrame, b: pd.DataFrame, tolerance: float, mode: str, window: int = 60,
                min_overlap: int = 5) -> CheckResult:
    """Compare two sources on their common dates (most recent `window`). Pass if ≥ 90% agree within tolerance."""
    m = a.merge(b, on="date", suffixes=("_a", "_b")).tail(window)
    if len(m) < min_overlap:
        return CheckResult("cross_check", "warn", f"only {len(m)} overlapping dates")
    diff = (m["value_a"] - m["value_b"]).abs()
    if mode == "rel":
        diff = diff / m["value_b"].abs().replace(0, np.nan)
    within = float((diff <= tolerance + 1e-12).mean())
    msg = f"{within:.0%} of {len(m)} common dates within {tolerance:g} ({mode}); max diff {diff.max():.4g}"
    return CheckResult("cross_check", "ok" if within >= 0.9 else "fail", msg)


def validate_series(df: pd.DataFrame, meta: SeriesMeta, today: date) -> list[CheckResult]:
    schema = check_schema(df)
    if not schema.ok:
        return [schema]
    return [schema, check_range(df, meta), check_staleness(df, meta, today), check_weekend_dates(df, meta),
            check_gaps(df, meta)]


def overall(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results}
    return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"
