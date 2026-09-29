"""15-minute job: scan news (Fed, BLS/BEA/Treasury, GDELT), classify new items with Claude, send alerts for
importance ≥ 4 and ask the workflow for a full report when an item is market-moving (importance 5).

    python -m jobs.intraday --state state/

Light by design (≈1 minute): no series collection. Live prices are fetched by the dashboard itself; order-book and
price snapshots are only worth collecting once database storage is enabled (they have no history otherwise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from collectors import news
from core.settings import OUTPUT_DIR
from jobs import classify, notify
from llm import claude


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--root", default=str(OUTPUT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    state, root = (Path(args.state) if args.state else None), Path(args.root)
    if not claude.available():
        print("ANTHROPIC_API_KEY not set — news cannot be classified, so no news alerts are sent")
        return 0
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    items, errors = news.collect()
    for name, err in errors.items():
        print(f"::warning::news source {name}: {err[:200]}")
    pd.DataFrame([{k: i.get(k) for k in ("id", "source", "published_utc", "title", "url", "summary")} for i in items]) \
        .to_csv(cache / "news.csv", index=False)
    now = pd.Timestamp.now(tz="UTC")
    try:
        print(f"classification: {classify.run(cache, state, now)}")
    except claude.LLMError as e:
        print(f"::warning::classification failed: {e}")
        return 0
    return notify.run("news", root, state, now, None, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
