"""The full pipeline as one command — the VPS equivalent of the data-availability workflow.

    python -m jobs.pipeline --mode report|release|weekly|headsup|none [--events ids] [--trigger text]
                            [--runs-dir /data/runs --state /data/state]

With --runs-dir every run writes into its own folder (runs/<UTC timestamp>/) and, when the analysis succeeded, the
`current` symlink next to it is swapped atomically, so the dashboard (MACRO_PULSE_SNAPSHOT_DIR=/data/current) never
reads a half-written snapshot. The last KEEP_RUNS runs are kept. Steps run as subprocesses in the same order and with
the same failure rules as the workflow: a failed step is logged and later analysis steps still run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from core.settings import OUTPUT_DIR

KEEP_RUNS = 3


def steps(mode: str, events: str, trigger: str, out: Path, state: Path) -> list[tuple[str, list[str], bool]]:
    """(name, argv, required) — a failed required step marks the run as failed (no `current` swap)."""
    py = [sys.executable, "-m"]
    cache = str(out / "cache")
    store = ["--store"] if os.environ.get("DATABASE_URL") and os.environ.get("ENABLE_DB_STORAGE") == "true" else []
    s: list[tuple[str, list[str], bool]] = []
    if mode == "release" and events:
        s.append(("wait for new values", py + ["jobs.scheduler", "wait", "--events", events], False))
    s.append(("collect + validate", py + ["jobs.availability", "--cache-dir", cache, *store], True))
    if mode != "none":
        s.append(("classify", py + ["jobs.classify", "--cache", cache, "--state", str(state)], False))
    s += [("features", py + ["jobs.features", "--from-cache", cache], True),
          ("impact", py + ["jobs.impact", "--from-cache", cache], True),
          ("signals", py + ["jobs.signals", "--from-cache", cache], True),
          ("pre-positioning", py + ["jobs.prepos", "--from-cache", cache, "--archive", str(state.parent / "archive"),
                                    "--state", str(state)], False),
          ("health", py + ["jobs.health", "--root", str(out)], False)]
    notify = {"release": ["report", "--trigger", trigger], "report": ["report", "--trigger", trigger],
              "weekly": ["weekly"], "headsup": ["headsup"]}.get(mode)
    if notify:
        s.append(("telegram", py + ["jobs.notify", *notify, "--root", str(out), "--state", str(state)], False))
    if mode == "headsup":
        s.insert(len(s) - 1, ("daily report", py + ["jobs.notify", "daily", "--trigger", trigger, "--root", str(out),
                                                     "--state", str(state)], False))
        s.append(("health warning", py + ["jobs.notify", "health", "--root", str(out), "--state", str(state)], False))
    if mode == "release" and events:
        s.append(("mark reported", py + ["jobs.scheduler", "mark", "--events", events, "--state", str(state)], False))
    return s


def swap_current(runs: Path, run_dir: Path) -> None:
    link, tmp = runs.parent / "current", runs.parent / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(run_dir.resolve(), target_is_directory=True)
    os.replace(tmp, link)  # atomic on POSIX
    for old in sorted(p for p in runs.iterdir() if p.is_dir())[:-KEEP_RUNS]:
        shutil.rmtree(old, ignore_errors=True)


def run(mode: str, events: str = "", trigger: str = "", runs: Path | None = None, state: Path | None = None,
        runner=subprocess.run) -> int:
    out = runs / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) if runs else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    state = state or out.parent / "state"
    env = {**os.environ, "MACRO_PULSE_OUTPUT_DIR": str(out)}
    failed = []
    for name, argv, required in steps(mode, events, trigger, out, state):
        print(f"── {name}", flush=True)
        rc = runner(argv, env=env).returncode
        if rc != 0:
            print(f"step '{name}' failed (exit {rc})", flush=True)
            if required:
                failed.append(name)
    if runs and not failed:
        swap_current(runs, out)
    print(f"pipeline {mode}: {'ok' if not failed else 'failed steps: ' + ', '.join(failed)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="none", choices=["none", "report", "release", "weekly", "headsup"])
    ap.add_argument("--events", default="")
    ap.add_argument("--trigger", default="")
    ap.add_argument("--runs-dir")
    ap.add_argument("--state")
    a = ap.parse_args(argv)
    return run(a.mode, a.events, a.trigger, Path(a.runs_dir) if a.runs_dir else None, Path(a.state) if a.state else None)


if __name__ == "__main__":
    sys.exit(main())
