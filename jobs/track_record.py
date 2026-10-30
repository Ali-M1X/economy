"""Write output/track_record.json: the replayed hit rate of the report's trend call, the calibration of the 4-week range
and what the trade-signal gate has produced.

    python -m jobs.track_record --from-cache output/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.settings import OUTPUT_DIR
from models import track_record as tr

ASSETS = {"BTC": "BTC", "Gold": "PAXG"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", default=str(OUTPUT_DIR / "cache"))
    args = ap.parse_args(argv)
    cache = Path(args.from_cache)
    now = pd.Timestamp.now(tz="UTC")
    sig_path = cache.parent / "signals.json"
    sigs = {a["asset"]: a for a in json.loads(sig_path.read_text()).get("assets", [])} if sig_path.exists() else {}
    out = {"asof": now.isoformat(), "assets": {}}
    for asset, sym in ASSETS.items():
        p = cache / f"candles_{sym}_15m.csv.gz"
        if not p.exists():
            continue
        bars = pd.read_csv(p, usecols=["ts", "open", "high", "low", "close", "volume"])
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        res = tr.evaluate(bars, now)
        a = sigs.get(asset, {})
        allt = (((a.get("backtest") or {}).get("summary") or {}).get("all")) or {}
        res["signals"] = {"live": len(a.get("signals") or []), "watchlist": len(a.get("watchlist") or []),
                          "backtest_trades": allt.get("n_trades"), "backtest_avg_r": allt.get("avg_r")}
        out["assets"][asset] = res
    (cache.parent / "track_record.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({a: {"hit": v["hit"], "range": v["range"]} for a, v in out["assets"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
