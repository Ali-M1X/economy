"""Point-in-time data access — the guard against look-ahead bias.

Every observation gets an `available_at` date: the first day its value was public.
- Series with ALFRED vintages: the *first print* of each observation, available at its first
  `realtime_start` (later revisions are ignored, exactly as a trader at that time would have seen).
- Other series: period end + the registry's release lag (conservative; FRED daily market series
  have lag 1, weekly H.4.1 data lag 1, etc.).
`asof_panel()` then samples every series on a date grid using only values available by each date.
"""

from __future__ import annotations

import pandas as pd

from core.registry import SeriesMeta
from validation.checks import period_end


def first_print(vintages: pd.DataFrame) -> pd.DataFrame:
    """date, value, available_at — each observation as first published."""
    v = vintages.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="first")
    return pd.DataFrame({"date": v["date"].to_numpy(), "value": v["value"].to_numpy(),
                         "available_at": v["realtime_start"].to_numpy()}).reset_index(drop=True)


def with_release_lag(df: pd.DataFrame, meta: SeriesMeta) -> pd.DataFrame:
    ends = df["date"].map(lambda d: period_end(d, meta.frequency))
    lag = pd.to_timedelta(max(meta.release_lag_days, 0), unit="D")
    return pd.DataFrame({"date": df["date"].to_numpy(), "value": df["value"].to_numpy(),
                         "available_at": (ends + lag).to_numpy()})


def known_series(df: pd.DataFrame, meta: SeriesMeta, vintages: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observation frame with an `available_at` column. Vintages win when present.

    Observations older than the first vintage (ALFRED starts at different dates per series) fall back
    to the release-lag rule so long histories are still usable."""
    lagged = with_release_lag(df, meta)
    if vintages is None or vintages.empty:
        return lagged
    fp = first_print(vintages)
    older = lagged[lagged["date"] < fp["date"].min()]
    return pd.concat([older, fp], ignore_index=True).sort_values("date").reset_index(drop=True)


def asof_panel(known: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Value of each series as known at each date in `dates` (latest observation whose available_at ≤ date).

    Returns a DataFrame indexed by `dates`; a second frame of observation dates can be derived by the
    caller if needed. Revisions/first prints are handled by `known_series`."""
    out = {}
    grid = pd.DataFrame({"t": dates})
    for key, k in known.items():
        k = k.sort_values("available_at")
        # For each availability time keep the newest *observation* known so far (a late revision of an
        # old period must not replace a newer period's value).
        k = k.assign(obs=k["date"]).sort_values(["available_at", "obs"])
        k["latest_obs"] = k["obs"].cummax()
        k = k[k["obs"] == k["latest_obs"]]
        m = pd.merge_asof(grid, k[["available_at", "value"]].rename(columns={"available_at": "t"}), on="t")
        out[key] = m["value"].to_numpy()
    return pd.DataFrame(out, index=dates)
