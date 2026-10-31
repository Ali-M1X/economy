"""The decision-first daily report (reports.daily), the pre-release and front-running alerts, the 7-day outlook."""

from __future__ import annotations

import json
import re
import shutil

import pandas as pd
import pytest

from dashboard import demo_snapshot
from reports import advisor, daily, messages, outlook
from reports import facts as facts_mod

NOW = pd.Timestamp("2026-09-29 12:00", tz="UTC")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return demo_snapshot.build(tmp_path_factory.mktemp("daily") / "output", now=NOW)


def _impact(pre_median: float = 0.5) -> dict:
    m = {"n_events": 40, "shock": {"1h": {"n": 40, "median_abs": 0.56, "p75_abs": 0.9, "normal_median_abs": 0.28,
                                          "ratio": 2.0}},
         "direction": {"n": 30, "hit_rate": 0.5, "p_value": 1.0, "predictable": False, "with_theory": False},
         "fade": {"n": 30, "reversal_rate": 0.45, "median_retrace_pct": 60.0},
         "pre": {"24": {"n": 40, "median_abs": pre_median}, "48": {"n": 40, "median_abs": pre_median}},
         "pre_post": {"n": 40, "same_direction_rate": 0.4,
                      "large_drift": {"n": 14, "reversal_rate": 0.64, "post_median_in_drift_direction": -0.3}},
         "tier": "volatile_undirected"}
    g = {**m, "shock": {"1h": {**m["shock"]["1h"], "median_abs": 0.28, "ratio": 1.8}}}
    return {"releases": {k: {"release_id": k, "assets": {"BTC": m, "Gold": g}} for k in ("cpi", "core_cpi", "fomc")}}


def _regime():
    rec = {"full": {"rule": {"cagr": 25.3, "sharpe": 1.36, "max_drawdown": -24.8},
                    "hold": {"cagr": 52.6, "sharpe": 0.97, "max_drawdown": -83.4}, "start": "2014-09-18"},
           "5y": {"rule": {"cagr": 10.7, "sharpe": 0.73, "max_drawdown": -14.3},
                  "hold": {"cagr": 4.7, "sharpe": 0.34, "max_drawdown": -76.6}, "start": "2021-11-01"}}
    gold = {"full": {"rule": {"cagr": 7.8, "sharpe": 0.61, "max_drawdown": -39.3},
                     "hold": {"cagr": 10.7, "sharpe": 0.66, "max_drawdown": -44.4}, "start": "2000-08-31"}}
    up = {"price": 82000.0, "sma200": 74000.0, "dist_pct": 10.8, "up": True, "days_in_state": 73, "mom12m_pct": 5.0,
          "vol60_pct": 36.0, "exposure": 0.55}
    dn = {"price": 3900.0, "sma200": 4500.0, "dist_pct": -13.3, "up": False, "days_in_state": 56, "mom12m_pct": -2.5,
          "vol60_pct": 20.0, "exposure": 0.0}
    return {"assets": {"BTC": {"state": up, "record": rec}, "Gold": {"state": dn, "record": gold}}}


@pytest.fixture
def root_impact(root, tmp_path):
    r = tmp_path / "output"
    shutil.copytree(root, r)
    (r / "release_impact.json").write_text(json.dumps(_impact()), encoding="utf-8")
    (r / "trend_regime.json").write_text(json.dumps(_regime()), encoding="utf-8")
    return r


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch, tmp_path):
    for k in ("LLM_API_KEY", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("notify.telegram.OUTBOX", tmp_path / "outbox")


def _plain(s: str) -> str:
    return re.sub(r"<[^>]+>", "", re.sub("[‎‏⁦-⁩⁧]", "", s))


def _assert_rtl(msg: str):
    bad = [ln for ln in daily.text_lines(msg) if not daily.starts_rtl(ln)]
    assert not bad, bad


# ── the daily report ─────────────────────────────────────────────────────

def test_one_short_message_decision_first(root_impact):
    msgs = daily.build(facts_mod.load(root_impact), NOW)
    assert len(msgs) == 1 and len(msgs[0]) <= daily.TARGET_CHARS
    t = _plain(msgs[0])
    heads = ["گزارش روزانه", "تصمیم امروز", "پیش رو", "₿ بیت‌کوین", "حمایت", "🥇 طلا", "توصیه مالی"]
    pos = [t.find(h) for h in heads]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), dict(zip(heads, pos))
    assert "مبنای تصمیم" not in t
    lines = [x for x in t.splitlines() if x.strip()]
    assert lines[1].startswith("🎯 تصمیم امروز") and lines[2].startswith("• بیت‌کوین:")
    assert "\n\n• طلا:" in t and "\n\n• رویداد کلیدی:" in t  # items are spaced apart
    btc = t[t.index("• بیت‌کوین:"):t.index("• طلا:")]
    gold = t[t.index("• طلا:"):t.index("• رویداد کلیدی:")]
    assert "خروج با بسته شدن روزانه زیر" in btc and "سهم مجاز صفر" in gold
    assert t.count("توصیه مالی") == 1 and t.count("گزارش روزانه") == 1
    assert "<pre>" not in msgs[0]  # no tables in the daily report


def test_every_line_starts_right_to_left(root_impact):
    f = facts_mod.load(root_impact)
    for m in daily.build(f, NOW) + daily.build(f, NOW, changes=[]):
        _assert_rtl(m)
    assert not daily.starts_rtl("JOLTS = فرصت‌های شغلی") and not daily.starts_rtl("(۲/۲)")
    assert daily.starts_rtl("• ⁦3.5⁩ برابر")  # an isolated number is skipped; the first strong letter is Persian
    assert not daily.starts_rtl("⁦3.5⁩ x برابر")
    assert daily.starts_rtl("₿ <b>بیت‌کوین</b> ⁦82,900⁩") and daily.starts_rtl("‏۳۰ مهر")


def test_measured_impact_replaces_strength_words_and_each_fact_appears_once(root_impact):
    f = facts_mod.load(root_impact)
    t = _plain(daily.build(f, NOW)[0])
    assert "ضعیف" not in t and "متوسط" not in t
    key_line = t[t.index("• رویداد کلیدی"):].split("\n\n")[0]
    assert "0.56" in key_line and "2.0 و 1.8 برابر یک ساعت عادی" in key_line and "جهت قابل پیش‌بینی نیست" in key_line
    assert t.count("0.56") == 1  # the key event's shock is not repeated in the 📅 section
    assert t.count("ثابت ماندن") <= 1  # the FOMC odds appear once at most


def test_published_forecasts_are_shown_and_the_system_guess_is_not(root_impact):
    pd.DataFrame([{"event_id": "cpi-20261002", "release_id": "cpi", "key": "cpi", "scheduled_utc": "2026-10-02T12:00:00+00:00",
                   "consensus": 0.3, "previous": 0.4, "consensus_source": "forexfactory", "nowcast": 0.45,
                   "nowcast_source": "cleveland", "nowcast_as_of": "2026-09-28", "market_points": None,
                   "market_source": None, "market_ticker": None}]).to_csv(root_impact / "cache" / "forecasts.csv", index=False)
    f = facts_mod.load(root_impact)
    t = _plain(daily.build(f, NOW)[0])
    assert "پیش‌بینی تحلیلگران 0.30" in t and "نوکست فد کلیولند 0.45" in t
    assert "انتظار سیستم" not in t


def test_market_line_reads_the_kalshi_median():
    from models import forecast as fm

    pts = [(4.0, 0.9), (4.1, 0.6), (4.2, 0.28), (4.3, 0.05)]
    assert fm.market_median(pts) == (4.1, 4.2)
    assert fm.market_median([(0, 0.4), (1, 0.2)]) is None


def test_fomc_keeps_the_market_priced_odds_only(root):
    f = facts_mod.load(root)
    outs = {o.release_id: o for o in outlook.upcoming(f, NOW, days=10)}  # the demo FOMC is 9 days out
    assert outs["fomc"].fedwatch and outs["fomc"].odds.cool == pytest.approx(0.7)
    t = _plain(daily.build(f, NOW)[0])
    assert "انتظار سیستم" not in t and "سیستم برتری ندارد" not in t


def test_plain_numbers_annualised_in_code():
    s = daily.meaning("cpi", 0.40, 0.25)
    ann = ((1 + 0.40 / 100) ** 12 - 1) * 100
    assert f"{ann:.1f}" in _plain(s) and "هدف ۲٪" in s and "بالاتر از قبلی" in s
    assert "پایین‌تر از قبلی" in daily.meaning("nonfarm_payrolls", 20, 120)
    assert "برابر ماه قبل" in daily.meaning("unemployment_rate", 4.2, 4.2)
    assert "کاهش نرخ" not in s and "انتظار" not in s  # no forecast-based reading
    assert "شغل ساخته شد" in daily.print_sentence("nonfarm_payrolls", 29.0, "2026-09-01")


def test_missing_levels_fall_back_to_the_20_day_range_or_atr():
    mv = {"hi": 110.0, "lo": 90.0, "atr_pct": 3.0}
    assert advisor.fallback_levels(100.0, None, None, {}, mv) == (110.0, 90.0, "سقف ۲۰ روزه", "کف ۲۰ روزه")
    r, s, rh, sh = advisor.fallback_levels(100.0, None, None, {}, {"hi": 99.0, "lo": 101.0, "atr_pct": 3.0})
    assert (round(r, 6), round(s, 6)) == (103.0, 97.0) and "نوسان روزانه" in rh and "نوسان روزانه" in sh
    assert advisor.fallback_levels(100.0, 105.0, 95.0, {}, mv) == (105.0, 95.0, None, None)


def test_no_dash_in_place_of_a_missing_number(root_impact):
    t = _plain(daily.build(facts_mod.load(root_impact), NOW)[0])
    assert not re.search(r"(حمایت|مقاومت|زیر|بالای|به|تا) —", t)


def test_split_marker_is_persian_words(root):
    parts = daily.split(["عنوان\nخط دوم", "ا" * 3000, "ب" * 3000], limit=4000)
    assert len(parts) == 2 and parts[0].startswith("عنوان — بخش ۱ از ۲\nخط دوم") and parts[1].startswith("بخش ۲ از ۲\n")
    assert daily.split(["عنوان", "کوتاه"]) == ["عنوان\n\nکوتاه"]  # fits: one message, no marker
    assert all(daily.starts_rtl(p.split("\n")[0]) for p in parts)


def test_post_release_report_starts_with_the_release(root_impact):
    msgs = daily.build(facts_mod.load(root_impact), NOW, changes=[], trigger="انتشار CPI")
    t = _plain(msgs[0])
    assert t.startswith("🔔 گزارش بعد از انتشار") and t.find("تصمیم امروز") > 0
    if "چه منتشر شد" in t:
        assert t.find("چه منتشر شد") < t.find("تصمیم امروز")


# ── pre-release and front-running alerts ─────────────────────────────────

def _cpi_event(at: pd.Timestamp) -> dict:
    return {"event_id": f"cpi-{at:%Y-%m-%d}", "release_id": "cpi", "name_fa": "CPI", "importance": 5, "scheduled_utc": at}


def test_prerelease_alert_is_short_with_impact_front_running_and_advice(root, tmp_path):
    r = tmp_path / "out"
    shutil.copytree(root, r)
    (r / "release_impact.json").write_text(json.dumps(_impact(pre_median=0.0005)), encoding="utf-8")
    f = facts_mod.load(r)
    msg = messages.prerelease(f, NOW - pd.Timedelta(minutes=1), [_cpi_event(NOW + pd.Timedelta(minutes=59))])
    lines = [x for x in msg.split("\n") if x.strip()]
    assert len(lines) <= 8
    t = _plain(msg)
    assert "تکان معمول" in t and "پیشخور: به‌شدت پیشخور شده" in t and "سابقه:" in t and "چه کنم:" in t
    assert "64٪" in t or "۶۴٪" in t  # the history behind the verdict, with its rate
    _assert_rtl(msg)


def test_front_running_alert_once_per_release(root, tmp_path):
    from jobs import notify
    from notify import telegram

    r = tmp_path / "out"
    shutil.copytree(root, r)
    (r / "release_impact.json").write_text(json.dumps(_impact(pre_median=0.0005)), encoding="utf-8")
    f = facts_mod.load(r)
    T = NOW + pd.Timedelta(hours=10)
    f.calendar = pd.DataFrame([{**_cpi_event(T), "name_en": "CPI"}])
    state = tmp_path / "state"
    assert notify.frontrun_alerts(f, NOW, state, True, r) == 1
    assert notify.frontrun_alerts(f, NOW, state, True, r) == 0  # never repeated
    sent = [p for p in telegram.OUTBOX.iterdir() if p.name.endswith("frontrun.html")]
    assert len(sent) == 1
    t = sent[0].read_text(encoding="utf-8")
    assert "هشدار پیشخور" in t and "نه سیگنال معامله" in t
    _assert_rtl(t)
    f.calendar = pd.DataFrame([{**_cpi_event(NOW + pd.Timedelta(hours=30)), "name_en": "CPI"}])  # too early
    assert notify.frontrun_alerts(f, NOW, tmp_path / "s2", True, r) == 0


# ── the free writer may never add numbers ────────────────────────────────

def test_rewrite_is_kept_only_without_new_numbers(monkeypatch):
    from llm import free_writer

    text = "تورم ⁦+0.4⁩٪ بالا رفت و دلار ⁦3.1⁩٪ قوی‌تر شد."
    assert daily.polish(text) == text  # no key → the template
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: ("تورم 0.4٪ بالا رفته و دلار 3.1٪ قوی‌تر شده است.", "m"))
    assert "قوی‌تر شده است" in daily.polish(text)
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: ("تورم 0.4٪ بالا رفت و احتمال کاهش نرخ 65٪ است.", "m"))
    assert daily.polish(text) == text  # an invented number → the draft is kept
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: ("CPI 0.4٪ و دلار 3.1٪.", "m"))
    assert daily.polish(text) == text  # a rewrite starting with a Latin word would break right-to-left


def test_a_rewrite_with_an_invented_number_never_reaches_the_report(root, monkeypatch):
    from llm import free_writer

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: (u + " احتمال رسیدن به 12345 دلار زیاد است.", "m"))
    msgs = daily.build(facts_mod.load(root), NOW)
    assert "12345" not in "".join(msgs) and "تصمیم امروز" in _plain(msgs[0])


def test_decision_comes_from_the_trend_filter_with_its_numbers(root_impact):
    f = facts_mod.load(root_impact)
    t = _plain("\n".join(daily.build(f, NOW)))
    btc = t[t.index("• بیت‌کوین:"):].split("\n\n")[0]
    gold = t[t.index("• طلا:"):].split("\n\n")[0]
    assert "بالای میانگین ۲۰۰ روزه" in btc and "سهم مجاز تا ۵۵٪" in btc and "زیر 74,000" in btc
    assert "زیر میانگین ۲۰۰ روزه" in gold and "سهم مجاز صفر" in gold and "بالای 4,500" in gold
    assert "مبنای تصمیم" not in t
    f.regime = {}
    t2 = _plain("\n".join(daily.build(f, NOW)))
    assert "تصمیمی پیشنهاد نمی‌شود" in t2


def test_numbers_section_compares_with_the_consensus_only(root_impact):
    f = facts_mod.load(root_impact)
    f.surprises = {"core_cpi": {**next(iter(f.surprises.values())), "release_utc": NOW.isoformat()}}
    sec = _plain(daily.facts_section(f, NOW, []))
    assert "قبلی" in sec and "انتظار" not in sec
    assert ("پیش‌بینی" in sec) == (f.surprises["core_cpi"].get("basis") == "vs consensus")
