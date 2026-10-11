"""The unified daily report (reports.daily) and the 7-day outlook (reports.outlook)."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from dashboard import demo_snapshot
from reports import daily, outlook
from reports import facts as facts_mod
from reports.asset_report import TABLE_WIDTH, visible

NOW = pd.Timestamp("2026-09-29 12:00", tz="UTC")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return demo_snapshot.build(tmp_path_factory.mktemp("daily") / "output", now=NOW)


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch):
    for k in ("LLM_API_KEY", "GITHUB_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def _plain(s: str) -> str:
    return re.sub(r"<[^>]+>", "", re.sub("[‎‏⁦-⁩⁧]", "", s))


def _pre_lines(msgs):
    for m in msgs:
        for block in re.findall(r"<pre>(.*?)</pre>", m, flags=re.S):
            yield from block.split("\n")


def test_one_report_with_every_section_in_order(root):
    msgs = daily.build(facts_mod.load(root), NOW)
    assert 1 <= len(msgs) <= 3 and all(len(m) <= 4096 for m in msgs)
    t = _plain("\n".join(msgs))
    heads = ["گزارش روزانه", "خلاصه‌ی امروز", "داستان بازار", "۷ روز آینده", "بیت‌کوین —", "برداشت من", "چه کنم؟",
             "طلا —", "توصیه مالی"]
    pos = [t.find(h) for h in heads]
    assert all(p >= 0 for p in pos) and pos == sorted(pos), dict(zip(heads, pos))
    assert t.count("توصیه مالی") == 1 and t.count("گزارش روزانه") == 1  # one disclaimer, no repeated header
    assert "دارنده:" in t and "خریدار:" in t and "حمایت نزدیک" in t and "۴ هفته‌ی آینده" in t
    assert "راهنمای اصطلاحات" not in t.split("\n")[0]  # glossary moved to the weekly
    summary = [x for x in t.split("خلاصه‌ی امروز")[1].split("📖")[0].splitlines() if x.strip()]
    assert 1 <= len(summary) <= 3, summary


def test_split_marks_parts_without_repeating_the_title(root):
    msgs = daily.build(facts_mod.load(root), NOW)
    if len(msgs) > 1:
        n = len(msgs)
        assert _plain(msgs[0]).splitlines()[0].endswith(f"(۱/{daily.fa_digits(str(n))})")
        assert all(_plain(m).startswith(f"({daily.fa_digits(str(i))}/") for i, m in enumerate(msgs[1:], 2))
    parts = daily.split(["عنوان\nخط دوم", "ا" * 3000, "ب" * 3000], limit=4000)
    assert len(parts) == 2 and parts[0].startswith("عنوان (۱/۲)\nخط دوم") and parts[1].startswith("(۲/۲)\n")
    assert daily.split(["عنوان", "کوتاه"]) == ["عنوان\n\nکوتاه"]  # fits: one message, no marker


def test_tables_fit_a_phone(root):
    lines = list(_pre_lines(daily.build(facts_mod.load(root), NOW)))
    assert lines and max(len(visible(x)) for x in lines) <= TABLE_WIDTH
    assert all(x.startswith("‎") for x in lines)  # LRM: columns stay left-to-right


def test_fomc_uses_fedwatch_and_releases_get_even_odds_without_a_forecast_edge(root):
    f = facts_mod.load(root)
    outs = {o.release_id: o for o in outlook.upcoming(f, NOW, days=10)}  # the demo FOMC is 9 days out
    assert outs["fomc"].fedwatch and outs["fomc"].odds.cool == pytest.approx(0.7)
    cpi = outs["cpi"]
    assert cpi.basis == "system" and (cpi.odds is None or cpi.odds.method == "even")
    t = _plain("\n".join(daily.build(f, NOW)))
    assert "انتظار سیستم" in t and "اجماع رایگان در دسترس نیست" in t


def test_consensus_nowcast_and_market_from_the_forecasts_file(root, tmp_path):
    import shutil

    r = tmp_path / "output"
    shutil.copytree(root, r)
    pd.DataFrame([{"event_id": "cpi-20261002", "release_id": "cpi", "key": "cpi", "scheduled_utc": "2026-10-02T12:00:00+00:00",
                   "consensus": 0.3, "previous": 0.4, "consensus_source": "forexfactory", "nowcast": 0.45,
                   "nowcast_source": "cleveland", "nowcast_as_of": "2026-09-28", "market_points": None,
                   "market_source": None, "market_ticker": None}]).to_csv(r / "cache" / "forecasts.csv", index=False)
    f = facts_mod.load(r)
    o = next(o for o in outlook.upcoming(f, NOW) if o.release_id == "cpi")
    assert o.key == "cpi" and o.basis == "consensus" and o.forecast == 0.3 and o.odds.method == "nowcast"
    assert o.odds.hot > o.odds.cool  # the nowcast is above the consensus
    t = _plain("\n".join(daily.build(f, NOW)))
    assert "اجماع" in t and "نوکست فد کلیولند" in t and "0.3%*" not in t
    # the pre-release alert names the consensus too
    from types import SimpleNamespace

    from reports import messages
    lines = messages.forecast_lines(f, SimpleNamespace(event_id="cpi-20261002", release_id="cpi", importance=5,
                                                       scheduled_utc=pd.Timestamp("2026-10-02 12:00", tz="UTC")))
    assert "اجماع" in _plain(lines[0]) and "0.30" in _plain(lines[0])


def test_post_release_report_puts_what_changed_on_top(root):
    msgs = daily.build(facts_mod.load(root), NOW, changes=["🔄 <b>چه تغییر کرد</b>", "• قیمت"], trigger="انتشار CPI")
    t = _plain(msgs[0])
    assert t.startswith("🔔 گزارش بعد از انتشار") and t.find("چه تغییر کرد") < t.find("خلاصه‌ی امروز")


def test_rewrite_is_kept_only_without_new_numbers(monkeypatch):
    from llm import free_writer

    text = "تورم ⁦+0.4⁩٪ بالا رفت و دلار ⁦3.1⁩٪ قوی‌تر شد."
    assert daily.polish(text) == text  # no key → the template
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: ("تورم 0.4٪ بالا رفته و دلار 3.1٪ قوی‌تر شده است.", "m"))
    assert "قوی‌تر شده است" in daily.polish(text)
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: ("تورم 0.4٪ بالا رفت و احتمال کاهش نرخ 65٪ است.", "m"))
    assert daily.polish(text) == text  # an invented number → the draft is kept


def test_a_rewrite_with_an_invented_number_never_reaches_the_report(root, monkeypatch):
    from llm import free_writer

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(free_writer, "chat", lambda s, u, **k: (u + " احتمال رسیدن به 12345 دلار زیاد است.", "m"))
    msgs = daily.build(facts_mod.load(root), NOW)
    assert "12345" not in "".join(msgs) and "داستان بازار" in _plain(msgs[0])
