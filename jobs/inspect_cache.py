"""Print raw values from a data snapshot, e.g. to check an edge case seen in a report.

    python -m jobs.inspect_cache snapshot/cache spread_10y_2y 2019-08-01 2019-09-15
"""

from __future__ import annotations

import sys

import pandas as pd


def main(argv: list[str]) -> int:
    cache, key, start, end = argv
    obs = pd.read_csv(f"{cache}/observations.csv.gz", parse_dates=["date"])
    s = obs[(obs["series_key"] == key) & obs["date"].between(start, end)]
    print(f"{key} {start}→{end}: {len(s)} rows, min {s['value'].min()} on "
          f"{s.loc[s['value'].idxmin(), 'date']:%Y-%m-%d}" if len(s) else f"{key}: no rows in range")
    print(s[["date", "value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
