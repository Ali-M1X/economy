"""Experimental advisory report (reports.advisor)."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from dashboard import demo_snapshot
from jobs import advisor as advisor_job
from reports import advisor
from reports import facts as facts_mod

NOW = pd.Timestamp("2026-09-29 12:00", tz="UTC")


@pytest.fixture(scope="module")
def f(tmp_path_factory):
    return facts_mod.load(demo_snapshot.build(tmp_path_factory.mktemp("adv") / "output", now=NOW))


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch, tmp_path):
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ANTHROPIC_API_KEY", "DASHBOARD_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("notify.telegram.OUTBOX", tmp_path / "outbox")


def _plain(s: str) -> str:
    return re.sub(r"<[^>]+>", "", re.sub("[‎‏⁦-⁩]", "", s))


def test_three_messages_with_story_data_and_advice(f):
    story, btc, gold = advisor.build(f, NOW)
    s = _plain(story)
    assert "داستان در چند جمله" in s and "برآیند کلان" in s
    assert all(len(m) < 4096 for m in (story, btc, gold))
    for msg, name in ((btc, "بیت‌کوین"), (gold, "طلا")):
        t = _plain(msg)
        assert f"— {name}" in t and "خلاصه برای تصمیم" in t and "کجای بازاریم؟" in t and "چه کنم؟" in t
        assert f"اگر {name} داری" in t and "اگر می‌خواهی بخری" in t
    assert "توصیه‌ی مالی" in _plain(gold)  # the experimental note closes the last message


def test_scenarios_sum_to_one_and_follow_the_drift():
    sc = advisor.scenarios(100, 110, 90, 0.0, 9.0)
    assert abs(sc["up"] + sc["base"] + sc["down"] - 1) < 1e-9
    assert abs(sc["higher"] - 0.5) < 1e-9 and abs(sc["up"] - sc["down"]) < 0.03
    assert advisor.scenarios(100, 110, 90, 3.0, 9.0)["higher"] > 0.6
    assert advisor.scenarios(100, None, 90, 0, 9) is None


def test_levels_skip_noise_next_to_the_price():
    x = {"price": 100.0, "technical": {"1d": {"atr": 4.0, "last_swing_high": 101.0, "last_swing_low": 92.0},
                                       "4h": {"last_swing_high": 106.0, "last_swing_low": 99.5}}}
    assert advisor.levels(x, {"hi": 110.0, "lo": 85.0}) == (106.0, 92.0)


def test_effects_are_beta_times_move_and_flag_odd_links(f):
    row = f.coefs[(f.coefs["horizon"] == "24h") & (f.coefs["sample"] == "full") & (f.coefs["asset"] == "Gold")
                  & (f.coefs["coefficient"].abs() >= 1)].iloc[0]
    b = advisor.beta(f, row["indicator"], "Gold", "24h")
    assert b["beta"] == pytest.approx(row["beta"])
    assert advisor.pct(b["beta"] * 2).replace("⁦", "").replace("⁩", "") in \
        _plain(advisor.effect_pair(f, row["indicator"], 2.0, "24h"))
    assert advisor.pct(-0.01).replace("⁦", "").replace("⁩", "") == "0.0٪"


def test_story_reads_inflation_and_jobs():
    t = {"inflation": 1.0, "labor": -0.8}
    assert "دست فدرال رزرو را می‌بندد" in advisor.story_sentence(t)
    assert "فرود نرم" in advisor.story_sentence({"inflation": -1.0, "labor": 1.0})
    assert "نزدیک روند" in advisor.story_sentence({"inflation": 0.1, "labor": None})


def test_tech_story_warns_when_the_daily_low_broke():
    s, score = advisor.tech_story("بیت‌کوین", {"1d": {"trend": "up", "last_event": "CHOCH_DN", "rsi": 47},
                                               "4h": {"trend": "down", "last_event": "CHOCH_DN"}})
    assert "اصلاح داخل روند صعودی" in s and "هشدار" in s and score == 0.0


def test_job_dry_run_writes_three_messages(f, tmp_path):
    assert advisor_job.main(["--root", str(f.root), "--dry-run"]) == 0
    assert len(list((tmp_path / "outbox").glob("*advisor-*.html"))) == 3
