"""Decide what a scheduled run should do, and wait for a release's new value.

    python -m jobs.scheduler plan --schedule "<cron that fired>" [--mode MODE] [--state DIR]
    python -m jobs.scheduler wait --events cpi-2026-10-15,…            # until FRED shows the new values (≤ 2 h)
    python -m jobs.scheduler mark --events cpi-2026-10-15,… --state DIR # remember that these were reported

One workflow (.github/workflows/schedule.yml) carries all crons; `plan` maps the cron that fired to a mode:

  intraday  every 15 min      news scan → Claude classification → alerts; market-moving news triggers a report
  headsup   daily             full refresh + Telegram heads-up for releases scheduled tomorrow (Tehran date)
  release   after each release time (both EDT and EST offsets) → report if a tracked release happened in the last
            RELEASE_WINDOW and was not reported yet
  weekly    Saturday          full refresh, coefficients and backtests recomputed, weekly summary

GitHub cron is UTC-only and may start late, so release crons exist for both US daylight-saving offsets and the
plan checks the real calendar; the state file prevents a second report for the same release.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

INTRADAY = "*/15 * * * *"
HEADSUP = "40 14 * * *"          # 18:10 Tehran — daily report + heads-up for tomorrow's releases
# GitHub Actions often skips scheduled runs under load: later slots retry the daily run until it has been sent
DAILY_RETRY = ["40 15 * * *", "40 16 * * *", "40 17 * * *"]
WEEKLY = "20 3 * * 6"            # Saturday 06:50 Tehran, after Friday's data
PREPOS = "50 * * * *"            # hourly: pre-positioning check while a target release is ≤ 72 h away
PREPOS_RELEASES = ("cpi", "pce", "nfp", "fomc")
PREPOS_HORIZON = pd.Timedelta(hours=72)
# ET release times 08:30, 10:00, 13:00, 14:00, 16:30 → UTC under EDT (−4) and EST (−5), +5 min
RELEASE = ["35 12,13 * * 1-5", "5 14,15,17,18,19 * * 1-5", "35 20,21 * * 1-5"]
RELEASE_WINDOW = pd.Timedelta(minutes=100)
WAIT_MAX = pd.Timedelta(hours=2)
WAIT_EVERY = 600  # seconds
UPDATE_SLACK = pd.Timedelta(minutes=30)  # FRED's last_updated may carry the release minute or a bit earlier


@dataclass
class Plan:
    mode: str                                  # intraday | headsup | release | weekly | report | none
    events: list[dict] = field(default_factory=list)
    reason: str = ""

    @property
    def trigger(self) -> str:
        if self.events:
            return "انتشار " + "، ".join(e["name_fa"] for e in self.events)
        return {"weekly": "به‌روزرسانی هفتگی", "headsup": "به‌روزرسانی روزانه"}.get(self.mode, "")


def mode_for(schedule: str | None) -> str | None:
    if not schedule:
        return None
    if schedule == INTRADAY:
        return "intraday"
    if schedule == HEADSUP or schedule in DAILY_RETRY:
        return "headsup"
    if schedule == WEEKLY:
        return "weekly"
    if schedule == PREPOS:
        return "prepos"
    if schedule in RELEASE:
        return "release"
    return None


def reported(state: Path | None) -> set[str]:
    p = state / "reported_events.json" if state else None
    return set(json.loads(p.read_text()).get("ids", [])) if p and p.exists() else set()


def tehran_date(now: pd.Timestamp) -> str:
    return str(now.tz_convert("Asia/Tehran").date())


def daily_sent(state: Path | None, now: pd.Timestamp) -> bool:
    """Whether today's (Tehran) daily report already went out."""
    p = state / "daily_sent.json" if state else None
    return bool(p and p.exists() and json.loads(p.read_text()).get("date") == tehran_date(now))


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def due_releases(events: list[dict], now: pd.Timestamp, done: set[str]) -> list[dict]:
    out = []
    for e in events:
        t = pd.Timestamp(e["scheduled_utc"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        if now - RELEASE_WINDOW <= t <= now and e["event_id"] not in done:
            out.append({**e, "scheduled_utc": t.isoformat()})
    return out


def plan(schedule: str | None, now: pd.Timestamp, state: Path | None, mode: str | None = None,
         calendar_fn=None) -> Plan:
    mode = mode or mode_for(schedule)
    if mode is None:
        return Plan("none", reason=f"unknown schedule {schedule!r}")
    if mode == "prepos":
        if calendar_fn is None:
            from collectors.calendar import upcoming

            calendar_fn = lambda: upcoming(days=4)[0]  # noqa: E731
        soon = [e for e in calendar_fn() if e["release_id"] in PREPOS_RELEASES
                and now < _utc(e["scheduled_utc"]) <= now + PREPOS_HORIZON]
        if not soon and schedule:  # a manual run always proceeds
            return Plan("none", reason="no CPI/PCE/NFP/FOMC release in the next 72 h")
        return Plan("prepos", [{**e, "scheduled_utc": _utc(e["scheduled_utc"]).isoformat()} for e in soon],
                    reason="upcoming: " + ", ".join(e["event_id"] for e in soon))
    if mode == "headsup" and schedule and daily_sent(state, now):
        return Plan("none", reason="today's daily report was already sent")
    if mode != "release":
        return Plan(mode, reason=f"schedule {schedule!r}" if schedule else "manual")
    if calendar_fn is None:
        from collectors.calendar import upcoming

        calendar_fn = lambda: upcoming(days=1)[0]  # noqa: E731 — today's and tomorrow's events
    due = due_releases(calendar_fn(), now, reported(state))
    if not due:
        return Plan("none", reason="no unreported tracked release in the last 100 minutes")
    return Plan("release", due, reason="released: " + ", ".join(e["event_id"] for e in due))


# ───────────────────────────── wait for the new value ─────────────────────────────

def updated_since(key: str, since: pd.Timestamp) -> bool | None:
    """True if the source shows an update after `since`; None if the source can't tell (then we don't wait)."""
    from collectors.series import fetch_series
    from core.registry import load_registry

    meta = load_registry().get(key)
    if meta is None:
        return None
    res = fetch_series(meta, start=(since - pd.Timedelta(days=800)).date().isoformat())
    if res.source_last_updated is None:
        return None
    lu = pd.Timestamp(res.source_last_updated)
    lu = lu.tz_localize("UTC") if lu.tzinfo is None else lu.tz_convert("UTC")
    return lu >= since


def wait(events: list[dict], now_fn=lambda: pd.Timestamp.now(tz="UTC"), sleep=time.sleep, check=updated_since) -> dict:
    """Poll until every waitable series of the events shows an update after its release time, or WAIT_MAX passes."""
    from collectors.calendar import tracked_releases

    rel = {r["id"]: r for r in tracked_releases()}
    pending = {}
    for e in events:
        since = pd.Timestamp(e["scheduled_utc"]) - UPDATE_SLACK
        for k in rel.get(e["release_id"], {}).get("series_keys", []):
            pending[k] = since
    start = now_fn()
    while True:
        for k, since in list(pending.items()):
            try:
                ok = check(k, since)
            except Exception as exc:  # noqa: BLE001 — a source error is retried on the next round
                print(f"{k}: check failed ({type(exc).__name__}); retrying")
                continue
            if ok is None or ok:
                print(f"{k}: {'new value visible' if ok else 'source has no update timestamp — not waiting'}")
                pending.pop(k)
        if not pending:
            return {"complete": True, "waited_s": int((now_fn() - start).total_seconds())}
        if now_fn() - start >= WAIT_MAX:
            print(f"::warning::still no new value after {WAIT_MAX} for {sorted(pending)} — reporting with what is available")
            return {"complete": False, "missing": sorted(pending), "waited_s": int((now_fn() - start).total_seconds())}
        print(f"waiting for {sorted(pending)} …")
        sleep(WAIT_EVERY)


def event_from_id(event_id: str) -> dict:
    """Rebuild a calendar event from its id "<release_id>-<YYYY-MM-DD>" (release time from config/releases.yaml)."""
    from collectors.calendar import _event, tracked_releases

    rid, y, m, d = event_id.rsplit("-", 3)
    rel = {r["id"]: r for r in tracked_releases()}
    return _event(rel[rid], pd.Timestamp(f"{y}-{m}-{d}").date(), "plan")


def mark(state: Path, ids: list[str]) -> None:
    done = reported(state) | set(ids)
    state.mkdir(parents=True, exist_ok=True)
    (state / "reported_events.json").write_text(json.dumps({"ids": sorted(done)[-300:]}))


def _out(**kv) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={' '.join(str(v).split())}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "wait", "mark"])
    ap.add_argument("--schedule")
    ap.add_argument("--mode")
    ap.add_argument("--state")
    ap.add_argument("--events", default="", help="comma-separated event ids (wait / mark)")
    args = ap.parse_args(argv)
    state = Path(args.state) if args.state else None
    now = pd.Timestamp.now(tz="UTC")
    if args.cmd == "plan":
        p = plan(args.schedule, now, state, args.mode or None)
        print(f"plan: {p.mode} — {p.reason}")
        _out(mode=p.mode, events=",".join(e["event_id"] for e in p.events), trigger=p.trigger)
        return 0
    ids = [x for x in args.events.split(",") if x]
    if args.cmd == "mark":
        if state and ids:
            mark(state, ids)
        return 0
    print(json.dumps(wait([event_from_id(i) for i in ids]), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
