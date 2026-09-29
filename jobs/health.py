"""Data health gate: decide whether a report may be produced from this snapshot.

Blocking problems (the report is replaced by a Telegram health warning):
- a core series failed validation or is stale (prices, policy rate, yields, dollar, inflation, balance sheet …);
- the 15-minute BTC / PAXG bars are older than MAX_BAR_AGE (signals and ranges would be wrong);
- an analysis output is missing (the Phase 2–4 job failed);
- more than MAX_FAIL_SHARE of all series failed (a wider outage).
Other failures are listed as non-blocking notes in the job log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.registry import load_registry

CORE = ["btc_usd_daily", "gold_futures", "fed_target_upper", "fed_funds_eff_daily", "ust_2y", "ust_10y", "real_yield_10y",
        "dxy", "cpi", "core_cpi", "nonfarm_payrolls", "unemployment_rate", "fed_balance_sheet", "tga_weekly", "reverse_repo",
        "vix", "hy_oas"]
OUTPUTS = ["features_state.json", "macro_scores.json", "signals.json", "impact_coefficients.csv"]
MAX_BAR_AGE = pd.Timedelta(hours=3)
MAX_FAIL_SHARE = 0.25
REPEAT_AFTER = pd.Timedelta(hours=12)


@dataclass
class Health:
    ok: bool
    problems: list[dict] = field(default_factory=list)   # {name, problem, blocking}
    notes: list[dict] = field(default_factory=list)

    @property
    def blocking(self) -> list[dict]:
        return [p for p in self.problems if p["blocking"]]

    def fingerprint(self) -> str:
        return hashlib.sha1("|".join(sorted(p["name"] + p["problem"] for p in self.blocking)).encode()).hexdigest()[:12]


def evaluate(root: Path, now: pd.Timestamp) -> Health:
    reg = load_registry()
    problems, notes = [], []
    p = root / "data_availability.json"
    if not p.exists():
        return Health(False, [{"name": "گزارش دسترس‌پذیری", "problem": "اجرای جمع‌آوری داده انجام نشد", "blocking": True}])
    rep = json.loads(p.read_text(encoding="utf-8"))
    series = {r["key"]: r for r in rep.get("series", [])}
    for k in CORE:
        r = series.get(k)
        name = reg[k].name_fa if k in reg else k
        if r is None:
            problems.append({"name": name, "problem": "در گزارش نیست", "blocking": True})
        elif r["status"] == "fail":
            why = r.get("error") or "; ".join(c["message"] for c in r.get("checks", []) if c.get("status") == "fail")
            problems.append({"name": name, "problem": f"خطا یا کهنه ({(why or 'بدون جزئیات')[:160]})", "blocking": True})
    failed = [k for k, r in series.items() if r["status"] == "fail"]
    notes += [{"name": k, "problem": "fail"} for k in failed if k not in CORE]
    if series and len(failed) / len(series) > MAX_FAIL_SHARE:
        problems.append({"name": "منابع داده", "problem": f"{len(failed)} از {len(series)} سری با خطا", "blocking": True})
    for sym in ("BTC", "PAXG"):
        f = root / "cache" / f"candles_{sym}_15m.csv.gz"
        if not f.exists():
            problems.append({"name": f"قیمت ۱۵ دقیقه‌ای {sym}", "problem": "موجود نیست", "blocking": True})
            continue
        last = pd.to_datetime(pd.read_csv(f, usecols=["ts"])["ts"], utc=True).max()
        if now - last > MAX_BAR_AGE:
            problems.append({"name": f"قیمت ۱۵ دقیقه‌ای {sym}", "problem": f"آخرین کندل {last:%Y-%m-%d %H:%M} UTC", "blocking": True})
    for o in OUTPUTS:
        if not (root / o).exists():
            problems.append({"name": o, "problem": "خروجی تحلیل ساخته نشد", "blocking": True})
    return Health(not any(x["blocking"] for x in problems), problems, notes)


def should_alert(h: Health, state: Path | None, now: pd.Timestamp) -> bool:
    """Send a health warning once per distinct problem set, and again only after REPEAT_AFTER."""
    if state is None:
        return True
    f = state / "health_last.json"
    last = json.loads(f.read_text()) if f.exists() else {}
    if last.get("fingerprint") == h.fingerprint() and now - pd.Timestamp(last["at"]) < REPEAT_AFTER:
        return False
    state.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"fingerprint": h.fingerprint(), "at": now.isoformat()}))
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    args = ap.parse_args(argv)
    h = evaluate(Path(args.root), pd.Timestamp.now(tz="UTC"))
    for p in h.problems:
        print(f"{'BLOCKING' if p['blocking'] else 'note'}: {p['name']}: {p['problem']}")
    print(f"health: {'ok' if h.ok else 'NOT OK'} ({len(h.blocking)} blocking, {len(h.notes)} other failed series)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
