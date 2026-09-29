"""Phase 6: Telegram, Claude reports and classification, health gate, scheduling. Fully offline: Claude and Telegram
are mocked, time is injected."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.fa import label
from dashboard import demo_snapshot
from jobs import health, notify, scheduler
from llm import classify as llm_classify
from llm import claude
from notify import telegram
from reports import facts as facts_mod
from reports import asset_report, messages, writer

NOW = pd.Timestamp("2026-09-29 12:00", tz="UTC")
LRI, PDI = "⁦", "⁩"
ROOT_DIR = Path(__file__).resolve().parent.parent


def plain(s: str) -> str:
    return s.replace(LRI, "").replace(PDI, "")


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    return demo_snapshot.build(tmp_path_factory.mktemp("demo") / "output", now=NOW)


@pytest.fixture()
def root(demo_root, tmp_path):
    return Path(shutil.copytree(demo_root, tmp_path / "output"))


@pytest.fixture(autouse=True)
def no_secrets(monkeypatch, tmp_path):
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ANTHROPIC_API_KEY", "DASHBOARD_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(telegram, "OUTBOX", tmp_path / "outbox")


# ── Telegram ─────────────────────────────────────────────────────────────

def test_split_message_respects_limit_and_blocks():
    text = "\n\n".join(f"<b>بخش {i}</b>\n" + "متن " * 300 for i in range(12))
    parts = telegram.split_message(text, limit=4096)
    assert len(parts) > 1 and all(len(p) <= 4096 for p in parts)
    assert all(p.count("<b>") == p.count("</b>") for p in parts)  # no tag pair cut in half
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")


def test_send_without_secrets_writes_outbox():
    r = telegram.send("<b>سلام</b>", kind="test")
    assert not r.sent and Path(r.where).read_text(encoding="utf-8") == "<b>سلام</b>"


class FakeResp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}
        self.ok = 200 <= status < 300
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, timeout, json):
        self.calls.append((url, json))
        return self.responses.pop(0)


def test_send_posts_html_and_retries_flood_control(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)
    s = FakeSession([FakeResp(429, {"parameters": {"retry_after": 1}}), FakeResp(200, {"ok": True})])
    r = telegram.send("<b>x</b>", session=s)
    assert r.sent and len(s.calls) == 2
    assert s.calls[-1][1] == {"chat_id": "42", "text": "<b>x</b>", "parse_mode": "HTML", "disable_web_page_preview": True}


def test_send_error_never_contains_the_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    s = FakeSession([FakeResp(400, {"description": "Bad Request: can't parse entities"})])
    with pytest.raises(telegram.TelegramError) as e:
        telegram.send("<b>x", session=s)
    assert "SECRET" not in str(e.value) and "parse entities" in str(e.value)


GOOD = "1234567890:AAH" + "x" * 32


@pytest.mark.parametrize("raw", [GOOD, f" {GOOD}\n", f'"{GOOD}"', f"bot{GOOD}", f"  bot{GOOD} \r\n"])
def test_token_paste_mistakes_are_cleaned_in_the_url(monkeypatch, raw):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", raw)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", " -1001234 \n")
    s = FakeSession([FakeResp(200, {"ok": True})])
    telegram.send("x", session=s)
    assert s.calls[0][0] == f"https://api.telegram.org/bot{GOOD}/sendMessage"
    assert s.calls[0][1]["chat_id"] == "-1001234"


def test_404_explains_the_token_without_revealing_it(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    with pytest.raises(telegram.TelegramError) as e:
        telegram.send("x", session=FakeSession([FakeResp(404, {"ok": False, "description": "Not Found"})]))
    msg = str(e.value)
    assert "SECRET" not in msg and "bot token" in msg and "10 chars" in msg and "--check" in msg


def test_check_reports_token_then_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", GOOD)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    ok_me = FakeResp(200, {"ok": True, "result": {"username": "mp_bot", "id": 1234567890}})
    ok_chat = FakeResp(200, {"ok": True, "result": {"type": "private", "first_name": "Ali"}})
    lines = telegram.check(FakeSession([ok_me, ok_chat]))
    assert lines[1] == "token OK: bot @mp_bot" and lines[2].startswith("chat OK: private")
    assert not any(GOOD in x or "Ali" in x or "1234567890" in x for x in lines)
    with pytest.raises(telegram.TelegramError, match="getMe → 404"):
        telegram.check(FakeSession([FakeResp(404, {"ok": False, "description": "Not Found"})]))
    with pytest.raises(telegram.TelegramError, match="getChat → 400"):
        telegram.check(FakeSession([ok_me, FakeResp(400, {"ok": False, "description": "Bad Request: chat not found"})]))


# ── Claude: classification and explanation (mocked) ──────────────────────

def test_classify_news_drops_unknown_ids_and_maps_fed_stance(monkeypatch):
    def fake_json_call(system, user, schema, **kw):
        if "Rate these documents" in user:
            return {"items": [{"id": "f1", "stance": "hawkish", "rationale_fa": "…"}]}
        return {"items": [{"id": "n1", "btc": "bearish", "gold": "bullish", "importance": 5, "channels": ["policy"],
                           "title_fa": "فدرال نرخ را افزایش داد"},
                          {"id": "invented", "btc": "neutral", "gold": "neutral", "importance": 1, "channels": [], "title_fa": "x"}]}
    monkeypatch.setattr(llm_classify, "json_call", fake_json_call)
    news = llm_classify.classify_news([{"id": "n1", "title": "Fed hikes"}])
    assert [n["id"] for n in news] == ["n1"]
    fed = llm_classify.classify_fed_docs([{"id": "f1", "title": "FOMC statement", "text": "…"}])
    assert fed[0]["hawkish_score"] == 0.5


def test_classify_job_uses_state_and_feeds_the_stance(root, monkeypatch, tmp_path):
    from jobs import classify as job
    from jobs.features import load_cache

    pd.DataFrame([
        {"id": "a", "source": "gdelt:reuters.com", "published_utc": NOW - pd.Timedelta(hours=1), "title": "Fed signals hike", "url": "u1", "summary": ""},
        {"id": "b", "source": "fed_monetary", "published_utc": NOW - pd.Timedelta(days=2), "title": "FOMC statement", "url": "u2", "summary": "s"},
    ]).to_csv(root / "cache" / "news.csv", index=False)
    calls = {"news": 0, "fed": 0}

    def fake_news(items):
        calls["news"] += len(items)
        return [{"id": i["id"], "btc": "bearish", "gold": "bearish", "importance": 4, "channels": ["policy"], "title_fa": "ف"} for i in items]

    def fake_fed(docs):
        calls["fed"] += len(docs)
        return [{"id": d["id"], "stance": "very_hawkish", "rationale_fa": "…", "hawkish_score": 1.0} for d in docs]
    monkeypatch.setattr(job, "classify_news", fake_news)
    monkeypatch.setattr(job, "classify_fed_docs", fake_fed)
    monkeypatch.setattr(job, "fetch_fed_text", lambda url: "full statement text")
    state = tmp_path / "state"
    # the Fed statement is also news (both classifiers); the GDELT item is news only
    assert job.run(root / "cache", state, NOW) == {"news_new": 2, "news_total": 2, "fed_new": 1, "fed_total": 1}
    assert job.run(root / "cache", state, NOW)["news_new"] == 0  # already classified → not sent again
    assert calls == {"news": 2, "fed": 1}
    docs = load_cache(root / "cache")[4]
    assert list(docs["hawkish_score"]) == [1.0]


def test_writer_accepts_only_numbers_from_the_facts(root, monkeypatch):
    f = facts_mod.load(root)
    s = f.summary()
    good = "CPI بالاتر از انتظار آمد (0.42 در برابر 0.25) و امتیاز کلان BTC در ۴ هفته +18 است."
    bad = "CPI بالاتر آمد و BTC احتمالاً 7.5 درصد می‌ریزد."
    assert writer.unknown_numbers(good, s) == []
    assert writer.unknown_numbers(bad, s) == [7.5]
    template = asset_report.chain_sentences(f, "BTC", NOW)
    monkeypatch.setattr(claude, "available", lambda: True)
    monkeypatch.setattr(claude, "text_call", lambda *a, **k: "1. " + good + "\n2. <b>دوم</b> جمله.")
    lines, src = writer.explain_asset(f, "BTC", template)
    assert src == "claude" and lines[0] == good and lines[1] == "&lt;b&gt;دوم&lt;/b&gt; جمله."  # numbering stripped, escaped
    monkeypatch.setattr(claude, "text_call", lambda *a, **k: bad)
    assert writer.explain_asset(f, "BTC", template) == (template, "template")

    def boom(*a, **k):
        raise claude.LLMError("down")
    monkeypatch.setattr(claude, "text_call", boom)
    assert writer.explain_asset(f, "BTC", template)[1] == "template"


# ── messages ─────────────────────────────────────────────────────────────

def test_report_is_split_into_overview_one_message_per_asset_tables_and_glossary(root):
    f = facts_mod.load(root)
    msgs = dict(asset_report.report_messages(f, NOW, None, "انتشار CPI"))
    assert list(msgs) == ["overview", "report_btc", "report_gold", "table_btc", "table_gold", "glossary"]
    btc, gold, ov = plain(msgs["report_btc"]), plain(msgs["report_gold"]), plain(msgs["overview"])
    # each asset message stands alone: its own price, scores, chain, table, signals and conclusion
    for txt, name in ((btc, "بیت‌کوین"), (gold, "طلا")):
        for part in ("💵 قیمت", "امتیاز کلان", "زنجیره‌ی علّی", "<pre>", "سیگنال معاملاتی", f"جمع‌بندی {name}",
                     messages.DISCLAIMER):
            assert part in txt, (name, part)
    assert "PAXG" not in btc and "طلا" not in btc.split("جمع‌بندی")[0].replace("طلا / XAU", "")
    # scores carry their scale and a reading
    assert "+18</b> از ±۱۰۰ → صعودی متوسط" in btc and "−۱۰۰ (کاملاً نزولی) تا +۱۰۰" in btc
    # the shared context lives in the overview only
    assert "موضع فدرال" in ov and "−۱۰۰ (کاملاً انبساطی)" in ov and "موضع فدرال" not in btc
    assert "شگفتی +1.8σ (متوسط)" in ov and "نسبت به روند (میانگین ۳ انتشار قبلی)" in ov
    # glossary defines every term the messages use
    for term in ("امتیاز کلان", "ضریب اثر", "σ", "R (واحد ریسک)", "خارج از نمونه", "hit rate", "ATR", "GC=F و PAXG"):
        assert term in msgs["glossary"]


def test_gold_message_separates_comex_gold_from_paxg(root):
    f = facts_mod.load(root)
    gold = plain(asset_report.asset_message(f, "Gold", NOW))
    comex = next(ln for ln in gold.splitlines() if ln.startswith("• طلای واقعی — COMEX GC=F"))
    paxg = next(ln for ln in gold.splitlines() if ln.startswith("• PAXG — توکن طلا"))
    assert "4,100.00" in paxg and "پایین‌تر از GC=F" in paxg or "بالاتر از GC=F" in paxg
    assert comex != paxg and "مبنای محاسبات" in gold and "(بر پایه‌ی PAXG)" in gold


def test_messages_fit_telegram_and_never_split_a_table(root):
    f = facts_mod.load(root)
    for kind, text in asset_report.report_messages(f, NOW):
        for part in telegram.split_message(text):
            assert len(part) <= telegram.LIMIT
            assert part.count("<pre>") == part.count("</pre>"), kind
            assert part.count("<b>") == part.count("</b>"), kind


def test_indicator_table_rows_sorted_by_impact_with_surprise_basis(root):
    f = facts_mod.load(root)
    rows = asset_report.indicator_rows(f, "BTC")
    coefs = [abs(r.coef) for r in rows if r.coef is not None]
    assert coefs == sorted(coefs, reverse=True) and len(rows) >= 25
    cpi = next(r for r in rows if r.key == "cpi")
    assert cpi.surprise == "+1.8σT" and cpi.coef_tag == "d"  # latest first-print surprise vs trend
    ry = next(r for r in rows if r.key == "real_yield_10y")
    assert ry.surprise == "-1.2σM" and ry.coef_tag == "w"      # not a release: 4-week move in σ
    assert ry.effect == int(np.sign(ry.coef * -0.6))
    table = asset_report.table_message(f, "BTC")
    assert table.count("<pre>") == -(-len(rows) // asset_report.CHUNK_ROWS)
    assert "بازده واقعی ۱۰ ساله" in table and "DFII10" in table


def test_no_signal_is_explained_in_plain_language(root):
    f = facts_mod.load(root)
    gold = plain(asset_report.asset_message(f, "Gold", NOW))
    assert "ستاپ فعالی وجود ندارد" not in gold
    assert "سیگنالی صادر نشد. به زبان ساده:" in gold
    assert "شرط کلان: امتیاز کلان ۴ هفته باید دست‌کم ±۱۵" in gold and "الان -6 است" in gold
    assert "برتری اثبات‌شده برای خرید: 100 معامله" in gold


def test_conclusion_levels_follow_where_the_price_is():
    x = {"price": 100.0, "technical": {"1d": {"last_swing_high": 110.0, "last_swing_low": 90.0},
                                       "4h": {"last_swing_high": 105.0, "last_swing_low": 95.0}}}
    up = plain(" ".join(asset_report.invalidation(x, 20.0, "{:,.0f}", "")))
    assert "زیر 90 (آخرین کف سوئینگ روزانه) بسته شود" in up and "شکست 95" in up
    down = plain(" ".join(asset_report.invalidation(x, -20.0, "{:,.0f}", "")))
    assert "بالای 110" in down and "شکست 105" in down
    broken = plain(" ".join(asset_report.invalidation({**x, "price": 85.0}, 20.0, "{:,.0f}", "")))
    assert "هم‌اکنون زیر آخرین کف سوئینگ روزانه" in broken and "تأیید نمی‌شود" in broken
    neutral = plain(" ".join(asset_report.invalidation({**x, "price": 85.0}, 2.0, "{:,.0f}", "، قیمت PAXG")))
    assert "ساختار قیمت نزولی است" in neutral and "قیمت PAXG" in neutral


def test_score_readings_have_bands():
    assert asset_report.score_reading(3).startswith("خنثی")
    assert asset_report.score_reading(-10).startswith("تمایل ضعیف نزولی")
    assert asset_report.score_reading(20).startswith("صعودی متوسط")
    assert asset_report.score_reading(-70).startswith("نزولی بسیار قوی")
    assert asset_report.score_direction(8) == "تمایل ضعیف صعودی" and asset_report.score_confidence(40) == "زیاد"


def test_chain_sentences_state_cause_and_effect(root):
    f = facts_mod.load(root)
    s = [plain(x) for x in asset_report.chain_sentences(f, "BTC", NOW)]
    assert any(x.startswith("شاخص قیمت مصرف‌کننده (CPI) (تغییر ماهانه (٪)) بالاتر از انتظار آمد: 0.42 در برابر 0.25") for x in s)
    ry = next(x for x in s if x.startswith("بازده واقعی ۱۰ ساله"))
    assert "پایین آمده است" in ry and "امتیاز ۴ هفته را 8.0 واحد بالا می‌برد" in ry and "←" not in ry


def test_headsup_lists_tomorrows_releases_only(root):
    f = facts_mod.load(root)
    msg = plain(messages.headsup(f, NOW) or "")
    assert "مدعیان بیمه بیکاری" in msg          # scheduled NOW + 1 day in the demo
    assert "شاخص قیمت مصرف‌کننده" not in msg   # NOW + 3 days
    assert "منبع رایگان ندارد" in msg


def test_news_alert_escapes_titles():
    item = {"importance": 5, "title_fa": "<script>alert(1)</script>", "source": "gdelt:reuters.com",
            "published_utc": NOW, "btc": "bearish", "gold": "bullish", "channels": "policy|dollar", "url": "https://x.y/?a=1&b=2"}
    msg = messages.news_alert(item)
    assert "<script>" not in msg and "&lt;script&gt;" in msg and "a=1&amp;b=2" in msg
    assert label("gdelt:reuters.com") == "GDELT · reuters.com"


# ── health gate ──────────────────────────────────────────────────────────

def test_health_ok_on_fresh_snapshot(root):
    assert health.evaluate(root, NOW).ok


def test_health_blocks_on_failed_core_series_and_stale_prices(root):
    p = root / "data_availability.json"
    rep = json.loads(p.read_text())
    for r in rep["series"]:
        if r["key"] == "ust_2y":
            r["status"], r["error"] = "fail", "HTTP 500"
    p.write_text(json.dumps(rep))
    h = health.evaluate(root, NOW + pd.Timedelta(hours=6))  # candles end at NOW → stale 6 h later
    names = " ".join(x["name"] for x in h.blocking)
    assert not h.ok and "بازده" in names and "BTC" in names and "PAXG" in names


def test_health_warning_is_not_repeated(root, tmp_path):
    (root / "signals.json").unlink()
    h = health.evaluate(root, NOW)
    state = tmp_path / "state"
    assert health.should_alert(h, state, NOW)
    assert not health.should_alert(h, state, NOW + pd.Timedelta(hours=1))
    assert health.should_alert(h, state, NOW + pd.Timedelta(hours=13))


# ── notify job end to end (dry run) ──────────────────────────────────────

def test_notify_report_weekly_headsup_dry_run(root, tmp_path):
    state = tmp_path / "state"
    report = ["glossary", "overview", "report_btc", "report_gold", "table_btc", "table_gold"]
    for mode in ("report", "headsup"):
        assert notify.run(mode, root, state, NOW, "انتشار CPI", dry_run=True) == 0
    sent = sorted(p.name.split("-", 2)[-1].removesuffix(".html") for p in telegram.OUTBOX.iterdir())
    assert sent == sorted(["headsup", *report])
    assert sorted(p.stem for p in (root / "messages").glob("*.html")) == sorted(["headsup", *report])
    notify.run("headsup", root, state, NOW, None, dry_run=True)  # same day again → not re-sent
    assert len(list(telegram.OUTBOX.iterdir())) == 7
    assert notify.run("weekly", root, state, NOW, None, dry_run=True) == 0
    assert "خلاصه‌ی هفتگی" in (root / "messages" / "overview.html").read_text(encoding="utf-8")


def test_notify_sends_health_warning_instead_of_report(root, tmp_path):
    (root / "macro_scores.json").unlink()
    assert notify.run("report", root, tmp_path / "state", NOW, None, dry_run=True) == 0
    sent = [p.name for p in telegram.OUTBOX.iterdir()]
    assert len(sent) == 1 and sent[0].endswith("health.html")


def test_notify_news_alerts_once_and_requests_report(root, tmp_path, monkeypatch):
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    pd.DataFrame([{"id": "n1", "source": "fed_all", "published_utc": NOW - pd.Timedelta(hours=1), "title": "Fed emergency cut",
                   "url": "u", "btc": "bullish", "gold": "bullish", "importance": 5, "channels": "policy", "title_fa": "کاهش اضطراری نرخ"},
                  {"id": "n2", "source": "bls", "published_utc": NOW - pd.Timedelta(hours=1), "title": "minor", "url": "u2",
                   "btc": "neutral", "gold": "neutral", "importance": 2, "channels": "", "title_fa": "جزئی"}]
                 ).to_csv(root / "cache" / "news_scored.csv", index=False)
    state = tmp_path / "state"
    notify.run("news", root, state, NOW, None, dry_run=True)
    notify.run("news", root, state, NOW, None, dry_run=True)
    assert len(list(telegram.OUTBOX.iterdir())) == 1
    assert "trigger_report=true" in out.read_text()


# ── scheduling ───────────────────────────────────────────────────────────

def workflow_crons() -> list[str]:
    wf = yaml.safe_load((ROOT_DIR / ".github" / "workflows" / "schedule.yml").read_text())
    on = wf.get("on", wf.get(True))
    return [c["cron"] for c in on["schedule"]]


def test_workflow_crons_match_the_scheduler():
    crons = workflow_crons()
    assert set(crons) == {scheduler.INTRADAY, scheduler.HEADSUP, scheduler.WEEKLY, scheduler.PREPOS, *scheduler.RELEASE}
    assert {scheduler.mode_for(c) for c in crons} == {"intraday", "headsup", "weekly", "prepos", "release"}


def _cron_times(cron: str) -> set[tuple[int, int]]:
    m, h = cron.split()[:2]
    return {(int(hh), int(mm)) for hh in h.split(",") for mm in m.split(",")}


def test_release_crons_cover_every_release_time_in_both_dst_offsets():
    fired = set().union(*(_cron_times(c) for c in scheduler.RELEASE))
    rel = yaml.safe_load((ROOT_DIR / "config" / "releases.yaml").read_text())["releases"]
    for r in rel:
        hh, mm = map(int, r["time_et"].split(":"))
        for offset in (4, 5):  # EDT, EST
            t = (hh + offset) * 60 + mm
            ok = [f for f in fired if 0 <= f[0] * 60 + f[1] - t <= 30]
            assert ok, f"{r['id']} at {r['time_et']} ET (UTC−{offset}) has no release cron within 30 min"


def test_plan_release_window_and_dedupe(tmp_path):
    ev = [{"event_id": "cpi-2026-09-29", "release_id": "cpi", "name_fa": "CPI", "scheduled_utc": NOW - pd.Timedelta(minutes=5)},
          {"event_id": "jolts-2026-09-29", "release_id": "jolts", "name_fa": "JOLTS", "scheduled_utc": NOW + pd.Timedelta(minutes=90)},
          {"event_id": "ppi-2026-09-29", "release_id": "ppi", "name_fa": "PPI", "scheduled_utc": NOW - pd.Timedelta(hours=3)}]
    p = scheduler.plan(scheduler.RELEASE[0], NOW, tmp_path, calendar_fn=lambda: ev)
    assert p.mode == "release" and [e["event_id"] for e in p.events] == ["cpi-2026-09-29"] and "CPI" in p.trigger
    scheduler.mark(tmp_path, ["cpi-2026-09-29"])
    assert scheduler.plan(scheduler.RELEASE[0], NOW, tmp_path, calendar_fn=lambda: ev).mode == "none"
    assert scheduler.plan(scheduler.INTRADAY, NOW, tmp_path).mode == "intraday"
    assert scheduler.plan("", NOW, tmp_path, mode="weekly").mode == "weekly"
    assert scheduler.plan("1 2 3 4 5", NOW, tmp_path).mode == "none"


def test_wait_polls_until_new_value_or_timeout():
    clock = {"t": NOW}

    def now_fn():
        return clock["t"]

    def sleep(s):
        clock["t"] += pd.Timedelta(seconds=s)
    seen = {"n": 0}

    def check(key, since):
        seen["n"] += 1
        return clock["t"] >= NOW + pd.Timedelta(minutes=25) if key == "cpi" else None
    ev = [scheduler.event_from_id("cpi-2026-09-29")]
    res = scheduler.wait(ev, now_fn=now_fn, sleep=sleep, check=check)
    assert res["complete"] and res["waited_s"] == 1800  # three 10-minute rounds
    res = scheduler.wait(ev, now_fn=now_fn, sleep=sleep, check=lambda k, s: False)
    assert not res["complete"] and res["missing"] == ["core_cpi", "cpi"]


# ── Claude client against a local stub of the Messages API (real SDK, no network) ──

def test_claude_client_request_and_parsing(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append(body)
            out = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    def msg(text, stop="end_turn"):
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5", "stop_reason": stop,
                "stop_sequence": None, "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}
    try:
        replies.append(msg('{"items": []}'))
        assert claude.json_call("sys", "user", {"type": "object"}) == {"items": []}
        body = seen[-1]
        assert body["model"] == "claude-sonnet-5" and body["system"] == "sys"
        assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
        replies.append(msg("متن"))
        assert claude.text_call("s", "u") == "متن" and seen[-1]["output_config"] == {"effort": "medium"}
        replies.append(msg("", stop="refusal"))
        with pytest.raises(claude.LLMError, match="declined"):
            claude.text_call("s", "u")
        replies.append(msg("cut", stop="max_tokens"))
        with pytest.raises(claude.LLMError, match="truncated"):
            claude.text_call("s", "u")
    finally:
        srv.shutdown()


# ── pre-positioning: plan, alert, notify ─────────────────────────────────

def test_prepos_plan_only_runs_when_a_target_release_is_near(tmp_path):
    cpi = [{"event_id": "cpi-x", "release_id": "cpi", "scheduled_utc": NOW + pd.Timedelta(hours=30)}]
    far = [{"event_id": "cpi-y", "release_id": "cpi", "scheduled_utc": NOW + pd.Timedelta(hours=80)},
           {"event_id": "claims-z", "release_id": "claims", "scheduled_utc": NOW + pd.Timedelta(hours=5)}]
    p = scheduler.plan(scheduler.PREPOS, NOW, tmp_path, calendar_fn=lambda: cpi)
    assert p.mode == "prepos" and [e["event_id"] for e in p.events] == ["cpi-x"]
    assert scheduler.plan(scheduler.PREPOS, NOW, tmp_path, calendar_fn=lambda: far).mode == "none"
    assert scheduler.plan("", NOW, tmp_path, mode="prepos", calendar_fn=lambda: far).mode == "prepos"  # manual run


def test_prepos_alert_and_notify_send_only_gated_signals_once(root, tmp_path):
    state = tmp_path / "state"
    assert notify.run("prepos", root, state, NOW, None, dry_run=True) == 0
    assert notify.run("prepos", root, state, NOW, None, dry_run=True) == 0  # same release/asset → not repeated
    sent = [p for p in telegram.OUTBOX.iterdir()]
    assert len(sent) == 1  # demo: BTC passed the gate (signal), gold is blocked (watchlist) → dashboard only
    txt = plain(sent[0].read_text(encoding="utf-8"))
    assert "پیش‌موقعیت‌گیری" in txt and "بیت‌کوین" in txt and "R:R 1.75" in txt and "حد ضرر" in txt
    assert "رانش قیمت" in txt and "زمینه‌ی زنده" in txt and messages.DISCLAIMER in txt


def test_headsup_mentions_prepositioning_status(root):
    f = facts_mod.load(root)
    f.calendar = pd.DataFrame([{"event_id": f.prepos["live"][0]["event_id"], "release_id": "cpi", "name_en": "CPI",
                                "name_fa": "CPI", "importance": 5, "scheduled_utc": NOW + pd.Timedelta(days=1)}])
    f.prepos["live"] = [dict(r, event_id=f.calendar["event_id"].iloc[0]) for r in f.prepos["live"]]
    txt = plain(messages.headsup(f, NOW) or "")
    assert "پیش‌موقعیت‌گیری" in txt and "سیگنال فعال" in txt and "مسدود" in txt


def test_impact_job_records_the_latest_surprise_per_indicator():
    from jobs import impact as impact_job

    evs = pd.DataFrame({"indicator": ["cpi", "cpi", "nonfarm_payrolls"],
                        "release_utc": pd.to_datetime(["2026-08-12 12:30", "2026-09-11 12:30", "2026-09-04 12:30"], utc=True),
                        "obs_date": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-08-01"]),
                        "actual": [0.2, 0.4, 142.0], "expected": [0.25, 0.3, 160.0], "z": [-0.5, 1.2, -0.4],
                        "surprise_basis": ["vs mean3 (no free consensus)"] * 3})
    out = impact_job.latest_surprises(evs)
    assert out["cpi"]["z"] == 1.2 and out["cpi"]["obs_date"] == "2026-08-01" and out["nonfarm_payrolls"]["actual"] == 142.0
    assert impact_job.latest_surprises(pd.DataFrame()) == {}
