"""Phase 7: VPS pipeline runner and scheduler (docker-compose). No subprocesses or network: steps are faked."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from jobs import pipeline, runner, scheduler
from tests.test_phase6 import ROOT_DIR, workflow_crons


def names(steps):
    return [s[0] for s in steps]


def test_pipeline_steps_per_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    none = pipeline.steps("none", "", "", tmp_path, tmp_path / "state")
    assert names(none) == ["collect + validate", "features", "impact", "signals", "health"]
    rel = pipeline.steps("release", "cpi-2026-10-15", "انتشار CPI", tmp_path, tmp_path / "state")
    assert names(rel)[0] == "wait for new values" and names(rel)[-1] == "mark reported"
    assert "classify" in names(rel) and "telegram" in names(rel)
    tg = dict((n, a) for n, a, _ in rel)["telegram"]
    assert tg[tg.index("--trigger") + 1] == "انتشار CPI"
    assert "health warning" in names(pipeline.steps("headsup", "", "", tmp_path, tmp_path / "state"))
    collect = dict((n, a) for n, a, _ in none)["collect + validate"]
    assert "--store" not in collect  # storage stays off unless explicitly enabled
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    assert "--store" not in dict((n, a) for n, a, _ in pipeline.steps("none", "", "", tmp_path, tmp_path))["collect + validate"]
    monkeypatch.setenv("ENABLE_DB_STORAGE", "true")
    assert "--store" in dict((n, a) for n, a, _ in pipeline.steps("none", "", "", tmp_path, tmp_path))["collect + validate"]


def test_pipeline_swaps_current_atomically_and_prunes(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    seen_env = []

    def ok(argv, env):
        seen_env.append(env["MACRO_PULSE_OUTPUT_DIR"])
        return SimpleNamespace(returncode=0)
    stamps = iter(f"2026092{i}T000000Z" for i in range(5))
    monkeypatch.setattr(pipeline.time, "strftime", lambda fmt, t=None: next(stamps))
    for _ in range(4):
        assert pipeline.run("none", runs=runs, runner=ok) == 0
    current = tmp_path / "current"
    assert current.is_symlink() and current.resolve().name == "20260923T000000Z"
    assert sorted(p.name for p in runs.iterdir()) == ["20260921T000000Z", "20260922T000000Z", "20260923T000000Z"]
    assert set(seen_env) == {str(runs / f"2026092{i}T000000Z") for i in range(4)}

    def failing(argv, env):
        return SimpleNamespace(returncode=1 if "jobs.impact" in argv else 0)
    assert pipeline.run("none", runs=runs, runner=failing) == 1
    assert current.resolve().name == "20260923T000000Z"  # a failed run never replaces the last good one


def test_runner_crons_match_github_and_use_cron_weekdays():
    assert runner.crons() == [scheduler.INTRADAY, scheduler.HEADSUP, scheduler.WEEKLY, *scheduler.RELEASE]
    assert set(runner.crons()) == set(workflow_crons())
    t0 = dt.datetime(2026, 9, 29, 12, 0, tzinfo=dt.timezone.utc)  # a Tuesday
    assert runner.trigger(scheduler.WEEKLY).get_next_fire_time(None, t0).strftime("%a %H:%M") == "Sat 03:20"
    t, days, n = runner.trigger(scheduler.RELEASE[0]), set(), t0
    for _ in range(20):
        n = t.get_next_fire_time(None, n + dt.timedelta(minutes=1))
        days.add(n.strftime("%a"))
    assert days == {"Mon", "Tue", "Wed", "Thu", "Fri"}


def test_runner_fire_dispatches_and_claims_releases(tmp_path, monkeypatch):
    calls = []
    ev = [{"event_id": "cpi-2026-09-29", "name_fa": "CPI"}]
    monkeypatch.setattr(scheduler, "plan", lambda cron, now, state: scheduler.Plan("release", ev, "test"))
    mode = runner.fire(scheduler.RELEASE[0], tmp_path, run_pipeline=lambda *a: calls.append(a))
    assert mode == "release" and calls[0][:3] == ("release", "cpi-2026-09-29", "انتشار CPI")
    assert "cpi-2026-09-29" in scheduler.reported(tmp_path / "state")  # claimed before the run
    assert (tmp_path / "heartbeat").exists()
    monkeypatch.setattr(scheduler, "plan", lambda cron, now, state: scheduler.Plan("intraday"))
    hit = []
    assert runner.fire(scheduler.INTRADAY, tmp_path, run_intraday=lambda d, s: hit.append(d)) == "intraday" and hit


def test_output_dir_is_configurable(tmp_path):
    out = subprocess.run([sys.executable, "-c", "from jobs import impact; print(impact.OUT_DIR)"],
                         env={**os.environ, "MACRO_PULSE_OUTPUT_DIR": str(tmp_path)},
                         capture_output=True, text=True, cwd=ROOT_DIR)
    assert out.stdout.strip() == str(tmp_path), out.stderr[-500:]


def test_compose_services_and_env_example():
    compose = yaml.safe_load((ROOT_DIR / "docker-compose.yml").read_text())
    s = compose["services"]
    assert s["scheduler"]["command"][:3] == ["python", "-m", "jobs.runner"]
    assert s["dashboard"]["environment"]["MACRO_PULSE_SNAPSHOT_DIR"] == "/data/current"
    assert s["dashboard"]["ports"] == ["127.0.0.1:8501:8501"]  # never exposed publicly without a proxy
    env = (ROOT_DIR / ".env.example").read_text()
    for k in ("FRED_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ANTHROPIC_API_KEY", "CLAUDE_MODEL", "DATABASE_URL",
              "ENABLE_DB_STORAGE"):
        assert f"\n{k}=" in "\n" + env
    assert "ENABLE_DB_STORAGE=false" in env


@pytest.mark.parametrize("cron", [scheduler.INTRADAY, scheduler.HEADSUP, scheduler.WEEKLY, *scheduler.RELEASE])
def test_every_cron_maps_to_a_mode(cron):
    assert scheduler.mode_for(cron) in {"intraday", "headsup", "weekly", "release"}
