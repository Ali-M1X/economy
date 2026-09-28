"""Economic release events and their surprises, reconstructed point-in-time from ALFRED vintages.

For each release date R of an indicator (a date on which a *new* observation first appeared):
- `actual`   = the new observation's measure (e.g. CPI MoM %) computed from the series exactly as it was
               published on R (the previous month is taken from the same vintage, so revisions released
               together with the new print are what the market saw);
- `expected` = a naive expectation from the same vintage (no free consensus exists; see DATA_GAPS.md):
               mean3 = mean of the previous 3 measures, mean4 = previous 4 levels, prev = previous level;
- `surprise` = actual − expected, and `z` = surprise / std of all *earlier* surprises (≥ 12 of them).
Dates with more than one new observation (bulk/benchmark loads) keep only the newest observation.
The release time of day comes from config/releases.yaml (US/Eastern, DST-aware).
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from collectors.calendar import tracked_releases
from core.timeutil import et_to_utc


def _measure(s: pd.Series, t: pd.Timestamp, how: str, freq: str) -> float:
    """The measure for observation t from series s (indexed by observation date)."""
    step = pd.DateOffset(months=1) if freq == "M" else pd.Timedelta(days=7)
    prev = t - step
    if how == "level":
        return float(s.get(t, np.nan))
    if prev not in s.index or t not in s.index:
        return np.nan
    if how == "mom_pct":
        return float((s[t] / s[prev] - 1) * 100)
    if how == "diff":
        return float(s[t] - s[prev])
    raise ValueError(how)


def _expected(s: pd.Series, t: pd.Timestamp, how: str, expected: str, freq: str) -> float:
    step = pd.DateOffset(months=1) if freq == "M" else pd.Timedelta(days=7)
    if expected == "prev":
        p = t - step
        return float(s[p]) if p in s.index else np.nan
    n = {"mean3": 3, "mean4": 4}[expected]
    vals = [_measure(s, t - step * k, how, freq) for k in range(1, n + 1)]
    return float(np.mean(vals)) if not any(np.isnan(vals)) else np.nan


def release_time_utc(release_id: str, day: pd.Timestamp) -> pd.Timestamp:
    rel = next(r for r in tracked_releases() if r["id"] == release_id)
    hh, mm = map(int, rel["time_et"].split(":"))
    return pd.Timestamp(et_to_utc(datetime.combine(day.date(), time(hh, mm))))


def surprises(vintages: pd.DataFrame, key: str, measure: str, expected: str, release_id: str,
              freq: str = "M", min_history: int = 12) -> pd.DataFrame:
    """One row per release: release_utc, obs_date, actual, expected, surprise, z."""
    v = vintages.sort_values(["realtime_start", "date"])
    first_seen = v.groupby("date")["realtime_start"].min()
    bulk_date = v["realtime_start"].min()  # the first vintage loads the whole back history at once
    rows = []
    for R, obs in first_seen.groupby(first_seen):
        if R == bulk_date:
            continue
        t = obs.index.max()
        known = v[(v["realtime_start"] <= R) & (v["realtime_end"].isna() | (v["realtime_end"] >= R))]
        s = known.drop_duplicates("date", keep="last").set_index("date")["value"].sort_index()
        a, e = _measure(s, t, measure, freq), _expected(s, t, measure, expected, freq)
        rows.append({"indicator": key, "release_date": R, "obs_date": t, "actual": a, "expected": e,
                     "n_new_obs": len(obs)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["surprise"] = df["actual"] - df["expected"]
    past_std = df["surprise"].expanding(min_periods=min_history).std().shift(1)
    df["z"] = df["surprise"] / past_std
    df["release_utc"] = [release_time_utc(release_id, d) for d in df["release_date"]]
    df["surprise_basis"] = f"vs {expected} (no free consensus)"
    return df.dropna(subset=["z"]).reset_index(drop=True)


def event_returns(events: pd.DataFrame, bars: pd.DataFrame, horizons_h=(1, 4, 24), bar_minutes: int = 15) -> pd.DataFrame:
    """Log returns (%) from the last trade *before* each release to `h` hours after.

    bars: ts (bar open, UTC), open, close. pre = close of the bar ending at or before the release;
    post(h) = close of the bar ending at or before release + h. Events without bars on both sides are
    dropped (never filled)."""
    step = pd.Timedelta(minutes=bar_minutes)
    close = bars.set_index("ts")["close"].sort_index()
    ends = close.index + step  # bar end times
    by_end = pd.Series(close.to_numpy(), index=ends)
    out = events.copy()
    t0 = pd.to_datetime(out["release_utc"], utc=True)
    pre_t = t0.dt.floor(f"{bar_minutes}min")
    pre = by_end.reindex(pre_t).to_numpy()
    for h in horizons_h:
        post_t = (t0 + pd.Timedelta(hours=h)).dt.floor(f"{bar_minutes}min")
        post = by_end.reindex(post_t).to_numpy()
        out[f"ret_{h}h"] = np.log(post / pre) * 100
    return out.dropna(subset=[f"ret_{h}h" for h in horizons_h], how="all")
