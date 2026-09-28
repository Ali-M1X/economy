"""Yield-curve inversion episodes and re-steepening.

An inversion episode is a run of the spread below zero. Short dips are filtered:
- runs shorter than `min_days` (calendar days) are ignored;
- inversions separated by less than `merge_gap_days` above zero are merged into one episode.
For each episode: start, end (last inverted day), depth (min spread) and date of the minimum, duration,
and **re-steepening**: the first date after the episode's end when the spread closes back above
`resteepen_level` (default +0 bp, i.e. the curve un-inverts). Historically the recession signal fires
around re-steepening, not at the start of the inversion, so both dates are reported.
An episode still in progress has end=None and ongoing=True.
"""

from __future__ import annotations

import pandas as pd

# Well-known 10Y−2Y / 10Y−3M inversion episodes, shown as chart annotations (labels only; dates come
# from the data itself).
NOTABLE = {
    1998: "LTCM / Asia crisis (brief)",
    2000: "Dot-com peak → 2001 recession",
    2006: "Housing peak → 2007-09 GFC",
    2019: "Trade war → 2020 recession (COVID)",
    2022: "Inflation shock → longest inversion on record",
}


def inversion_episodes(spread: pd.Series, min_days: int = 5, merge_gap_days: int = 30,
                       resteepen_level: float = 0.0) -> pd.DataFrame:
    s = spread.dropna().sort_index()
    below = s < 0
    run_id = (below != below.shift()).cumsum()
    runs = [(g.index[0], g.index[-1]) for _, g in s[below].groupby(run_id[below])]
    merged: list[list[pd.Timestamp]] = []
    for start, end in runs:
        if merged and (start - merged[-1][1]).days <= merge_gap_days:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    last_date = s.index[-1]
    rows = []
    for start, end in merged:
        if (end - start).days + 1 < min_days:
            continue
        seg = s[start:end]
        ongoing = end == last_date
        after = s[s.index > end]
        rs = after[after > resteepen_level]
        rows.append({
            "start": start, "end": None if ongoing else end, "ongoing": ongoing,
            "duration_days": (end - start).days + 1,
            "depth": float(seg.min()), "depth_date": seg.idxmin(),
            "resteepen_date": None if ongoing or rs.empty else rs.index[0],
            "label": next((v for y, v in NOTABLE.items() if abs(start.year - y) <= 1), None),
        })
    return pd.DataFrame(rows, columns=["start", "end", "ongoing", "duration_days", "depth", "depth_date",
                                       "resteepen_date", "label"])


def curve_state(spread: pd.Series, episodes: pd.DataFrame, lookback_days: int = 365) -> str:
    """inverted | re-steepening (un-inverted within `lookback_days` after an episode) | normal."""
    if spread.dropna().iloc[-1] < 0:
        return "inverted"
    today = spread.dropna().index[-1]
    recent = episodes.dropna(subset=["resteepen_date"])
    if not recent.empty and (today - recent["resteepen_date"].max()).days <= lookback_days:
        return "re-steepening"
    return "normal"
