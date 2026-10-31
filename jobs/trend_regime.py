"""Write output/trend_regime.json: the long-horizon trend reading and its measured record (models.trend_regime).

    python -m jobs.trend_regime --from-cache output/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.settings import OUTPUT_DIR
from models import trend_regime as tr

SERIES = {"BTC": "btc_usd_daily", "Gold": "gold_futures"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", default=str(OUTPUT_DIR / "cache"))
    args = ap.parse_args(argv)
    cache = Path(args.from_cache)
    obs = pd.read_csv(cache / "observations.csv.gz", usecols=["series_key", "date", "value"], parse_dates=["date"])
    out = {"asof": pd.Timestamp.now(tz="UTC").isoformat(), "assets": {}}
    for asset, key in SERIES.items():
        s = obs[obs["series_key"] == key].dropna(subset=["value"]).drop_duplicates("date").set_index("date")["value"].sort_index()
        s = s[s > 0].astype(float)
        if not s.empty:
            out["assets"][asset] = tr.evaluate(s, asset)
    (cache.parent / "trend_regime.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({a: v.get("state") for a, v in out["assets"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
