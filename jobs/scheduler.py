"""Decide what a scheduled run should do, and wait for a release's new value.

    python -m jobs.scheduler plan --schedule "<cron that fired>" [--mode MODE] [--state DIR]
    python -m jobs.scheduler wait --events cpi-2026-10-15,…            # until FRED shows the new values (≤ 2 h)
    python -m jobs.scheduler mark --events cpi-2026-10-15,… --state DIR # remember that these were reported

One workflow (.github/workflows/schedule.yml) carries all crons. GitHub starts scheduled runs late (often by
an hour or more) or skips them, so every scheduled run first checks what is due, whichever cron fired:

  release   a tracked release (importance ≥ 4) happened in the last RELEASE_WINDOW and was not reported yet
            → wait for the new value (≤ 2 h) → full pipeline → report with what changed since the last report
  daily     today's 09:00 Tehran report has not gone out: a run that starts up to DAILY_LEAD early waits, refreshes
            the data and sends at 09:00 exactly (a late run sends at once)
  watch     an important release is ≤ WATCH_AHEAD away: wait until PRE_ALERT_LEAD before it → pre-release alert
            (expected value, what a surprise would mean, front-running read) → wait for the release → release run
  otherwise the cron's own mode: intraday (news, every 15 min), weekly (Saturday), prepos (hourly)

Claims in the state cache stop a later run from doing the same work twice.
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
# extra early-morning crons give the 09:00 Tehran report several chances to start in time (any run checks it)
DAILY = "40 0,1,2,3,4 * * *"     # 04:10–08:10 Tehran
DAILY_HOUR_TEHRAN = 9
DAILY_LEAD = pd.Timedelta(hours=5)          # a run may start this long before 09:00 and wait (job limit: 6 h)
DAILY_PREP = pd.Timedelta(minutes=25)       # the data refresh starts this long before the send minute
CLAIM_TTL = pd.Timedelta(hours=6)           # a claim older than this (its run died) no longer blocks the work
WATCH_IMPORTANCE = 4                        # pre- and post-release messages for releases of this importance or more
WATCH_AHEAD = pd.Timedelta(hours=4, minutes=30)
PRE_ALERT_LEAD = pd.Timedelta(minutes=60)
WEEKLY = "20 3 * * 6"            # Saturday 06:50 Tehran, after Friday's data
PREPOS = "50 * * * *"            # hourly: pre-positioning check while a target release is ≤ 72 h away
PREPOS_RELEASES = ("cpi", "pce", "nfp", "fomc")
PREPOS_HORIZON = pd.Timedelta(hours=72)
# ET release times 08:30, 10:00, 13:00, 14:00, 16:30 → UTC under EDT (−4) and EST (−5), +5 min
RELEASE = ["35 12,13 * * 1-5", "5 14,15,17,18,19 * * 1-5", "35 20,21 * * 1-5"]
RELEASE_WINDOW = pd.Timedelta(hours=6)  # a late run still reports a release this long after it
WAIT_MAX = pd.Timedelta(hours=2)
WAIT_EVERY = 600  # seconds
UPDATE_SLACK = pd.Timedelta(minutes=30)  # FRED's last_updated may carry the release minute or a bit earlier


@dataclass
class Plan:
    mode: str                                  # intraday | daily | watch | release | weekly | prepos | report | none
    events: list[dict] = field(default_factory=list)
    reason: str = ""
    wait_until: str = ""                       # daily / watch: sleep until this UTC time before acting

    @property
    def trigger(self) -> str:
        if self.events:
            return "انتشار " + "، ".join(e["name_fa"] for e in self.events)
        return {"weekly": "به‌روزرسانی هفتگی", "daily": "گزارش روزانه", "headsup": "گزارش روزانه"}.get(self.mode, "")


def mode_for(schedule: str | None) -> str | None:
    if not schedule:
        return None
    if schedule == INTRADAY:
        return "intraday"
    if schedule == DAILY:
        return "daily"
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


def daily_target(now: pd.Timestamp) -> pd.Timestamp:
    """09:00 Tehran on today's Tehran date, in UTC."""
    t = now.tz_convert("Asia/Tehran").normalize() + pd.Timedelta(hours=DAILY_HOUR_TEHRAN)
    return t.tz_convert("UTC")


def _json(state: Path | None, name: str) -> dict:
    p = state / name if state else None
    return json.loads(p.read_text()) if p and p.exists() else {}


def daily_claimed(state: Path | None, now: pd.Timestamp) -> bool:
    c = _json(state, "daily_claim.json")
    return c.get("date") == tehran_date(now) and now - pd.Timestamp(c["at"]) < CLAIM_TTL


def watched(state: Path | None, now: pd.Timestamp) -> set[str]:
    return {k for k, at in _json(state, "watched.json").items() if now - pd.Timestamp(at) < CLAIM_TTL}


def claim(state: Path, p: "Plan", now: pd.Timestamp) -> None:
    """Record that this run took the work, so later (late-starting) runs skip it."""
    state.mkdir(parents=True, exist_ok=True)
    if p.mode == "daily":
        (state / "daily_claim.json").write_text(json.dumps({"date": tehran_date(now), "at": now.isoformat()}))
    if p.mode == "watch":
        w = {k: at for k, at in _json(state, "watched.json").items() if now - pd.Timestamp(at) < pd.Timedelta(days=3)}
        w.update({e["event_id"]: now.isoformat() for e in p.events})
        (state / "watched.json").write_text(json.dumps(w))
    if p.mode in ("watch", "release") and p.events:
        mark(state, [e["event_id"] for e in p.events])


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def important(e: dict) -> bool:
    return int(e.get("importance") or 0) >= WATCH_IMPORTANCE


def watch_group(events: list[dict], now: pd.Timestamp, done: set[str]) -> list[dict]:
    """The earliest group of important releases ≤ WATCH_AHEAD away that no run is watching yet."""
    soon = sorted((e for e in events if important(e) and e["event_id"] not in done
                   and now < _utc(e["scheduled_utc"]) <= now + WATCH_AHEAD), key=lambda e: _utc(e["scheduled_utc"]))
    if not soon:
        return []
    t0 = _utc(soon[0]["scheduled_utc"])
    return [{**e, "scheduled_utc": _utc(e["scheduled_utc"]).isoformat()} for e in soon if _utc(e["scheduled_utc"]) == t0]


def due_releases(events: list[dict], now: pd.Timestamp, done: set[str]) -> list[dict]:
    out = []
    for e in events:
        if not important(e):
            continue
        t = pd.Timestamp(e["scheduled_utc"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        if now - RELEASE_WINDOW <= t <= now and e["event_id"] not in done:
            out.append({**e, "scheduled_utc": t.isoformat()})
    return out


def plan(schedule: str | None, now: pd.Timestamp, state: Path | None, mode: str | None = None,
         calendar_fn=None) -> Plan:
    if calendar_fn is None:
        from collectors.calendar import upcoming

        calendar_fn = lambda: upcoming(days=4)[0]  # noqa: E731
    if mode:  # manual run
        if mode == "release":
            due = due_releases(calendar_fn(), now, set())
            return Plan("release", due, reason="manual: " + (", ".join(e["event_id"] for e in due) or "no recent release"))
        return Plan(mode, reason="manual")
    base = mode_for(schedule)
    if base is None:
        return Plan("none", reason=f"unknown schedule {schedule!r}")
    try:
        events = calendar_fn()
    except Exception as exc:  # noqa: BLE001 — the daily report must not depend on the calendar sources
        print(f"::warning::calendar unavailable ({type(exc).__name__}: {exc}) — checking the daily report only")
        events = []
    done = reported(state)
    due = due_releases(events, now, done)
    if due:
        return Plan("release", due, reason="released: " + ", ".join(e["event_id"] for e in due))
    if base == "weekly":  # Saturday's summary keeps its slot; a later run picks up the daily report
        return Plan("weekly", reason=f"schedule {schedule!r}")
    target = daily_target(now)
    if now >= target - DAILY_LEAD and not daily_sent(state, now) and not daily_claimed(state, now):
        return Plan("daily", reason=f"daily report for {tehran_date(now)}",
                    wait_until=max(now, target - DAILY_PREP).isoformat())
    group = watch_group(events, now, done | watched(state, now))
    if group:
        t0 = _utc(group[0]["scheduled_utc"])
        return Plan("watch", group, reason="upcoming: " + ", ".join(e["event_id"] for e in group),
                    wait_until=max(now, t0 - PRE_ALERT_LEAD).isoformat())
    if base == "prepos":
        soon = [e for e in events if e["release_id"] in PREPOS_RELEASES
                and now < _utc(e["scheduled_utc"]) <= now + PREPOS_HORIZON]
        if not soon:
            return Plan("none", reason="no CPI/PCE/NFP/FOMC release in the next 72 h")
        return Plan("prepos", [{**e, "scheduled_utc": _utc(e["scheduled_utc"]).isoformat()} for e in soon],
                    reason="upcoming: " + ", ".join(e["event_id"] for e in soon))
    if base in ("release", "daily"):
        return Plan("none", reason="nothing due")
    return Plan(base, reason=f"schedule {schedule!r}")


def sleep_until(when: str, now_fn=lambda: pd.Timestamp.now(tz="UTC"), sleep=time.sleep) -> None:
    t = _utc(when)
    while (left := (t - now_fn()).total_seconds()) > 0:
        print(f"waiting until {t.isoformat()} ({int(left // 60)} min)", flush=True)
        sleep(min(left, 600))


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
    ap.add_argument("cmd", choices=["plan", "wait", "mark", "sleep"])
    ap.add_argument("--until")
    ap.add_argument("--schedule")
    ap.add_argument("--mode")
    ap.add_argument("--state")
    ap.add_argument("--events", default="", help="comma-separated event ids (wait / mark)")
    args = ap.parse_args(argv)
    state = Path(args.state) if args.state else None
    now = pd.Timestamp.now(tz="UTC")
    if args.cmd == "plan":
        p = plan(args.schedule, now, state, args.mode or None)
        print(f"plan: {p.mode} — {p.reason}" + (f" (waits until {p.wait_until})" if p.wait_until else ""))
        if state and args.schedule:
            claim(state, p, now)
        _out(mode=p.mode, events=",".join(e["event_id"] for e in p.events), trigger=p.trigger, wait_until=p.wait_until,
             release_at=p.events[0]["scheduled_utc"] if p.events else "")
        return 0
    if args.cmd == "sleep":
        sleep_until(args.until)
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
