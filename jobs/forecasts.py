"""Collect free forecasts for the coming releases and keep the pre-release consensus history.

    python -m jobs.forecasts --cache output/cache --state state

Writes
- <cache>/forecasts.csv           upcoming (event, series) rows: consensus, nowcast, market odds, each with its source
- <cache>/../forecast_sources.json which source worked on this run (shown in the Actions log)
- <state>/forecast_history.csv    the last forecast captured *before* each release (frozen once the release passes),
                                   copied to <cache>/forecast_history.csv for the impact job and the reports.
Surprises are measured against the stored consensus when one exists (models.events.apply_consensus).
Never fails the pipeline: a source error is recorded and the run continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

from collectors import forecasts as fc
from core.timeutil import utcnow

HISTORY_COLS = ["event_id", "release_id", "key", "release_utc", "consensus", "previous", "consensus_source",
                "nowcast", "nowcast_source", "market_points", "market_source", "captured_utc"]


def load_calendar(cache: Path) -> pd.DataFrame:
    p = cache / "calendar_events.csv"
    if not p.exists():
        return pd.DataFrame()
    c = pd.read_csv(p)
    c["scheduled_utc"] = pd.to_datetime(c["scheduled_utc"], utc=True)
    return c


def update_history(hist: pd.DataFrame, fresh: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Replace each not-yet-released row with the newest capture; released rows stay as they were last seen."""
    if fresh.empty:
        return hist if not hist.empty else pd.DataFrame(columns=HISTORY_COLS)
    f = fresh[pd.to_datetime(fresh["scheduled_utc"], utc=True) > now].copy()
    f = f.rename(columns={"scheduled_utc": "release_utc"})
    f["captured_utc"] = now.isoformat()
    f = f[[c for c in HISTORY_COLS if c in f.columns]]
    if hist.empty:
        return f.reindex(columns=HISTORY_COLS)
    keep = hist.set_index(["event_id", "key"])
    new = f.set_index(["event_id", "key"])
    keep = keep[~keep.index.isin(new.index)]
    out = pd.concat([keep, new]).reset_index()
    return out.reindex(columns=HISTORY_COLS).sort_values(["release_utc", "event_id", "key"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="output/cache")
    ap.add_argument("--state", default="state")
    args = ap.parse_args(argv)
    cache, state = Path(args.cache), Path(args.state)
    now = pd.Timestamp(utcnow())
    table, status = fc.collect(load_calendar(cache), now)
    cache.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache / "forecasts.csv", index=False)
    (cache.parent / "forecast_sources.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")
    state.mkdir(parents=True, exist_ok=True)
    hp = state / "forecast_history.csv"
    hist = pd.read_csv(hp) if hp.exists() else pd.DataFrame(columns=HISTORY_COLS)
    hist = update_history(hist, table, now)
    hist.to_csv(hp, index=False)
    hist.to_csv(cache / "forecast_history.csv", index=False)
    print(json.dumps(status, indent=1, default=str))
    print(f"forecasts: {len(table)} upcoming rows; history: {len(hist)} rows")
    if not table.empty:
        print(table.drop(columns=["market_points"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover — forecasts are optional; never fail the pipeline
        traceback.print_exc()
        sys.exit(0)
