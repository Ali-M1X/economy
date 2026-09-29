"""Long-running scheduler for a VPS / docker-compose: the same crons and modes as .github/workflows/schedule.yml.

    python -m jobs.runner --data /data         # blocks; logs to stdout

Cron strings are imported from jobs.scheduler (single source for both GitHub Actions and the VPS). Full pipeline
runs go to /data/runs/<ts>/ with /data/current → latest good run (read by the dashboard); run state lives in
/data/state. Pipeline runs are serialised (one at a time); the 15-minute news scan runs independently.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

from jobs import pipeline, scheduler

log = logging.getLogger("runner")
_pipeline_lock = threading.Lock()


def crons() -> list[str]:
    return [scheduler.INTRADAY, scheduler.HEADSUP, scheduler.WEEKLY, *scheduler.RELEASE]


_DOW = ["sun", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def trigger(cron: str):
    """CronTrigger with standard cron semantics. APScheduler 3 numbers weekdays Monday=0, cron numbers Sunday=0, so
    `CronTrigger.from_crontab("20 3 * * 6")` would fire on Sunday instead of Saturday — weekdays are passed as names."""
    from apscheduler.triggers.cron import CronTrigger

    minute, hour, day, month, dow = cron.split()
    if dow != "*":
        dow = ",".join("-".join(_DOW[int(x)] for x in part.split("-")) for part in dow.split(","))
    return CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=dow, timezone="UTC")


def fire(cron: str, data: Path, run_pipeline=pipeline.run, run_intraday=None) -> str:
    """What one cron firing does (returns the mode, for logging and tests)."""
    state = data / "state"
    (data / "heartbeat").touch()  # docker healthcheck: the runner fires at least every 15 minutes
    p = scheduler.plan(cron, pd.Timestamp.now(tz="UTC"), state)
    log.info("cron %r → %s (%s)", cron, p.mode, p.reason)
    if p.mode == "intraday":
        (run_intraday or _intraday)(data, state)
    elif p.mode in ("release", "headsup", "weekly"):
        ids = ",".join(e["event_id"] for e in p.events)
        if ids:
            scheduler.mark(state, [e["event_id"] for e in p.events])  # claim before the (long) run
        with _pipeline_lock:
            run_pipeline(p.mode, ids, p.trigger, data / "runs", state)
    return p.mode


def _intraday(data: Path, state: Path) -> None:
    current = data / "current"
    root = current if current.exists() else data / "news"
    out = subprocess.run([sys.executable, "-m", "jobs.intraday", "--state", str(state), "--root", str(root)],
                         capture_output=True, text=True, env={**os.environ, "GITHUB_OUTPUT": str(data / ".intraday_out")})
    sys.stdout.write(out.stdout)
    flags = (data / ".intraday_out").read_text() if (data / ".intraday_out").exists() else ""
    (data / ".intraday_out").unlink(missing_ok=True)
    if "trigger_report=true" in flags:  # market-moving news → full report, as on GitHub
        text = next((line.split("=", 1)[1] for line in flags.splitlines() if line.startswith("trigger_text=")), "")
        with _pipeline_lock:
            pipeline.run("report", "", text, data / "runs", state)


def main(argv: list[str] | None = None) -> int:
    from apscheduler.schedulers.blocking import BlockingScheduler

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--run-now", choices=["intraday", "headsup", "weekly", "report"], help="run one mode immediately, then schedule")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = Path(args.data)
    (data / "runs").mkdir(parents=True, exist_ok=True)
    if args.run_now == "intraday":
        _intraday(data, data / "state")
    elif args.run_now:
        pipeline.run(args.run_now, "", "", data / "runs", data / "state")
    sched = BlockingScheduler(timezone="UTC", job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 900})
    for c in crons():
        sched.add_job(fire, trigger(c), args=[c, data], id=c, name=c)
    log.info("scheduled %d crons (UTC): %s", len(crons()), ", ".join(crons()))
    sched.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
