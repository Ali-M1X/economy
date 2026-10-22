"""Real release impact on Bitcoin and gold (models.release_impact), from the snapshot's 15-minute candles.

    python -m jobs.release_impact --from-cache output/cache

Writes output/release_impact.json (read by the reports) and output/release_impact.md (readable summary).
Events: every first print of the release indicators in config/indicators.yaml (point-in-time surprises from the
vintages, against the stored consensus where one exists) and the FOMC statements (fomc_meetings.csv; no surprise
measure, so shock and front-running only). Bitcoin uses BTCUSDT 15-minute bars (from 2017), gold uses PAXG (from
2020-09), the 24/7 gold token, as a proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

from core.settings import OUTPUT_DIR
from core.timeutil import utcnow
from jobs.impact import indicator_config, load_consensus_history
from models import events as ev_mod
from models import release_impact as ri

ASSETS = {"BTC": "BTC", "Gold": "PAXG"}
FOMC_KEY = "fomc"


def load_bars(cache: Path, sym: str) -> pd.DataFrame | None:
    p = cache / f"candles_{sym}_15m.csv.gz"
    if not p.exists():
        return None
    b = pd.read_csv(p, usecols=["ts", "close"])
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    return b


def release_events(cache: Path, vintages: dict) -> dict[str, dict]:
    """{key: {release_id, theory, events (release_utc, z)}} for every release indicator plus the FOMC."""
    cfg = indicator_config()
    cons = load_consensus_history(cache)
    out = {}
    for rel in cfg["releases"]:
        v = vintages.get(rel["key"])
        if v is None or v.empty:
            continue
        freq = "W" if rel["key"] == "initial_claims" else "M"
        ev = ev_mod.surprises(v, rel["key"], rel["measure"], rel["expected"], rel["release"], freq=freq)
        ev = ev_mod.apply_consensus(ev, cons)
        if ev.empty:
            continue
        out[rel["key"]] = {"release_id": rel["release"], "theory": rel.get("theory", {}),
                           "events": ev[["release_utc", "z"]].reset_index(drop=True)}
    p = cache / "fomc_meetings.csv"
    if p.exists():
        m = pd.read_csv(p)
        out[FOMC_KEY] = {"release_id": "fomc", "theory": {},
                         "events": pd.DataFrame({"release_utc": pd.to_datetime(m["statement_utc"], utc=True), "z": float("nan")})}
    return out


def compute(cache: Path, vintages: dict, now: pd.Timestamp) -> dict:
    evs = release_events(cache, vintages)
    busy = {d for e in evs.values() for d in pd.to_datetime(e["events"]["release_utc"], utc=True).dt.date}
    res = {"as_of": now.isoformat(), "min_n": ri.MIN_N, "big_shock": ri.BIG_SHOCK, "direction_p": ri.DIRECTION_P,
           "horizons": list(ri.HORIZONS), "pre_windows_h": list(ri.PRE_WINDOWS), "releases": {}}
    for asset, sym in ASSETS.items():
        bars = load_bars(cache, sym)
        if bars is None or bars.empty:
            continue
        px = ri.Prices.from_bars(bars)
        first = bars["ts"].min() + pd.Timedelta(weeks=ri.BASE_WEEKS)
        for key, e in evs.items():
            ev = e["events"]
            ev = ev[(pd.to_datetime(ev["release_utc"], utc=True) >= first) &
                    (pd.to_datetime(ev["release_utc"], utc=True) + pd.Timedelta(hours=24) <= now)]
            if ev.empty:
                continue
            m = ri.measure(px, ev, busy, int(e["theory"].get(asset, 0)))
            m["period"] = [f"{pd.Timestamp(ev['release_utc'].min()):%Y-%m-%d}", f"{pd.Timestamp(ev['release_utc'].max()):%Y-%m-%d}"]
            res["releases"].setdefault(key, {"release_id": e["release_id"], "assets": {}})["assets"][asset] = m
    return res


def render(res: dict) -> str:
    L = ["# Release impact — real moves of Bitcoin and gold around each release", "", f"As of {res['as_of']}", "",
         f"Shock ratio = median |move| after the release ÷ median |move| at the same weekday and time on days without a "
         f"release. Direction = 1-hour hit rate of the textbook sign on |z| ≥ {ri.BIG_Z} surprises. Fade = share of "
         f"first-hour moves larger than a normal hour that end the day on the other side. Statistics with n < {ri.MIN_N} "
         "are hidden.", ""]
    for asset in ASSETS:
        rows = [(k, r["assets"][asset]) for k, r in res["releases"].items() if asset in r["assets"]]
        rows.sort(key=lambda kv: -((kv[1]["shock"].get("1h") or {}).get("ratio") or 0))
        L += [f"## {asset}", "", "| release | n | 15m med% | 1h med% | 1h ratio | 24h ratio | dir hit (n, p) | fade (n) | "
              "pre-24h ratio | large drift → post (n) | tier |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, m in rows:
            s15, s1, s24 = (m["shock"].get(h) or {} for h in ("15m", "1h", "24h"))
            d, f, p = m.get("direction") or {}, m.get("fade") or {}, (m.get("pre") or {}).get("24") or {}
            ld = ((m.get("pre_post") or {}).get("large_drift") or {})
            dir_ = f"{d['hit_rate']:.0%} ({d['n']}, p={d['p_value']})" if d.get("hit_rate") is not None else f"— ({d.get('n', 0)})"
            fad = f"{f['reversal_rate']:.0%} ({f['n']})" if f.get("reversal_rate") is not None else f"— ({f.get('n', 0)})"
            lds = (f"{ld['post_median_in_drift_direction']:+.2f}% , rev {ld['reversal_rate']:.0%} ({ld['n']})"
                   if ld.get("reversal_rate") is not None else f"— ({ld.get('n', 0)})")
            L.append(f"| {k} | {m['n_events']} | {s15.get('median_abs', '—')} | {s1.get('median_abs', '—')} | "
                     f"{s1.get('ratio', '—')} | {s24.get('ratio', '—')} | {dir_} | {fad} | {p.get('ratio', '—')} | {lds} | "
                     f"{m.get('tier') or '—'} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", required=True)
    args = ap.parse_args(argv)
    from jobs.features import load_cache

    cache = Path(args.from_cache)
    _, vintages, *_ = load_cache(cache)
    res = compute(cache, vintages, pd.Timestamp(utcnow()))
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "release_impact.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = render(res)
    (OUTPUT_DIR / "release_impact.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n\n" + md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover
        traceback.print_exc()
        sys.exit(1)

