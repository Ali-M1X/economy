"""Phase 6: Telegram, Claude reports and classification, health gate, scheduling. Fully offline: Claude and Telegram
are mocked, time is injected."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

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





def test_messages_fit_telegram_and_never_split_a_table(root):
    f = facts_mod.load(root)
    for kind, text in asset_report.report_messages(f, NOW):
        for part in telegram.split_message(text):
            assert len(part) <= telegram.LIMIT
            assert part.count("<pre>") == part.count("</pre>"), kind
            assert part.count("<b>") == part.count("</b>"), kind










def test_score_readings_have_bands():
    assert asset_report.score_reading(3).startswith("خنثی")
    assert asset_report.score_reading(-10).startswith("تمایل ضعیف نزولی")
    assert asset_report.score_reading(20).startswith("صعودی متوسط")
    assert asset_report.score_reading(-70).startswith("نزولی بسیار قوی")
    assert asset_report.score_direction(8) == "تمایل ضعیف صعودی" and asset_report.score_confidence(40) == "زیاد"




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


def _daily_done(state, now=NOW):
    state.mkdir(parents=True, exist_ok=True)
    (state / "daily_sent.json").write_text(json.dumps({"date": scheduler.tehran_date(now)}))


def test_daily_report_at_0900_tehran_once_per_day(root, tmp_path):
    state = tmp_path / "state"
    report = ["glossary", "overview", "report_btc", "report_gold", "table_btc", "table_gold"]
    day = pd.Timestamp("2026-10-03 00:00", tz="UTC")  # 03:30 Tehran
    assert scheduler.daily_target(day) == pd.Timestamp("2026-10-03 05:30", tz="UTC")  # 09:00 Tehran
    none = lambda: []  # noqa: E731
    # before 04:00 Tehran nothing is due; from then on any cron that fires takes it and waits until ~08:35
    assert scheduler.plan(scheduler.INTRADAY, day, state, calendar_fn=none).mode == "intraday"
    p = scheduler.plan(scheduler.DAILY, day + pd.Timedelta(hours=1), state, calendar_fn=none)
    assert p.mode == "daily" and p.wait_until == "2026-10-03T05:05:00+00:00" and p.trigger == "گزارش روزانه"
    scheduler.claim(state, p, day + pd.Timedelta(hours=1))
    assert scheduler.plan(scheduler.INTRADAY, day + pd.Timedelta(hours=2), state, calendar_fn=none).mode == "intraday"
    # a run that starts late sends at once; a dead claim stops blocking after CLAIM_TTL
    late = day + pd.Timedelta(hours=8)
    p = scheduler.plan(scheduler.PREPOS, late, state, calendar_fn=none)
    assert p.mode == "daily" and p.wait_until == late.isoformat()
    _daily_done(state, late)
    assert scheduler.plan(scheduler.DAILY, late, state, calendar_fn=none).mode == "none"
    sent_state = tmp_path / "s2"  # the report itself (fixture data is from NOW)
    assert notify.run("daily", root, sent_state, NOW, "گزارش روزانه", dry_run=True) == 0
    sent = sorted(p.name.split("-", 2)[-1].removesuffix(".html") for p in telegram.OUTBOX.iterdir())
    assert sent == sorted(report)
    notify.run("daily", root, sent_state, NOW, None, dry_run=True)  # same day → not re-sent
    assert len(list(telegram.OUTBOX.iterdir())) == len(report)
    assert scheduler.plan(scheduler.DAILY, day + pd.Timedelta(days=1, hours=1), state, calendar_fn=none).mode == "daily"


def test_daily_report_waits_for_0900(monkeypatch):
    slept = []
    notify.wait_for_daily_slot(pd.Timestamp("2026-10-03 05:10", tz="UTC"), sleep=slept.append)
    notify.wait_for_daily_slot(pd.Timestamp("2026-10-03 06:10", tz="UTC"), sleep=slept.append)  # late: no wait
    assert slept == [20 * 60]


def test_watch_plans_prealert_then_release_and_claims(tmp_path):
    _daily_done(tmp_path)
    ev = [{"event_id": "nfp-2026-09-29", "release_id": "nfp", "name_fa": "NFP", "importance": 5,
           "scheduled_utc": NOW + pd.Timedelta(hours=3)},
          {"event_id": "claims-2026-09-29", "release_id": "claims", "name_fa": "Claims", "importance": 3,
           "scheduled_utc": NOW + pd.Timedelta(hours=1)}]
    p = scheduler.plan(scheduler.INTRADAY, NOW, tmp_path, calendar_fn=lambda: ev)
    assert p.mode == "watch" and [e["event_id"] for e in p.events] == ["nfp-2026-09-29"]
    assert p.wait_until == (NOW + pd.Timedelta(hours=2)).isoformat()  # 1 h before the release
    scheduler.claim(tmp_path, p, NOW)
    # a later run neither watches it again nor reports it again (the watch run reports it itself)
    assert scheduler.plan(scheduler.INTRADAY, NOW + pd.Timedelta(minutes=30), tmp_path, calendar_fn=lambda: ev).mode == "intraday"
    assert scheduler.plan(scheduler.RELEASE[0], NOW + pd.Timedelta(hours=3, minutes=10), tmp_path,
                          calendar_fn=lambda: ev).mode == "none"
    far = [{**ev[0], "event_id": "nfp-x", "scheduled_utc": NOW + pd.Timedelta(hours=6)}]
    assert scheduler.plan(scheduler.INTRADAY, NOW, tmp_path, calendar_fn=lambda: far).mode == "intraday"


def test_calendar_failure_still_plans_the_daily_report(tmp_path):
    def broken():
        raise RuntimeError("FRED down")
    assert scheduler.plan(scheduler.INTRADAY, pd.Timestamp("2026-10-03 02:00", tz="UTC"), tmp_path,
                          calendar_fn=broken).mode == "daily"


def test_prerelease_alert_and_changes_block(root, tmp_path):
    f = facts_mod.load(root)
    ev = [{**scheduler.event_from_id("cpi-2026-10-15")}]
    t = pd.Timestamp(ev[0]["scheduled_utc"])
    msg = plain(messages.prerelease(f, t - pd.Timedelta(minutes=60), [{**ev[0], "scheduled_utc": t}]) or "")
    assert "هشدار پیش از انتشار" in msg and "پیشخور" in msg and "۶۰ دقیقه" in msg and "CPI" in msg
    assert "بالاتر از انتظار" in msg and "بعد از انتشار" in msg
    prev = {"asof": "2026-09-29T05:30:00+00:00",
            "assets": {"BTC": {"price": 80000.0, "scores": {"short": 0.0, "medium": -20.0}},
                       "Gold": {"price": 4000.0, "scores": {"short": 1.0, "medium": 2.0}}}}
    cur = {"asof": "x", "assets": {"BTC": {"price": 84000.0, "scores": {"short": 1.0, "medium": 6.0}},
                                   "Gold": {"price": 4000.0, "scores": {"short": 2.0, "medium": 3.0}}}}
    txt = plain("\n".join(asset_report.changes_block(prev, cur)))
    assert "چه تغییر کرد" in txt and "+5.0" in txt and "از نزولی" in txt and "امتیاز کلان تغییر مهمی نکرد" in txt
    assert "در دسترس نیست" in plain("\n".join(asset_report.changes_block({}, cur)))


def test_release_report_carries_changes_and_saves_scores(root, tmp_path):
    state = tmp_path / "state"
    notify.run("daily", root, state, NOW, None, dry_run=True)
    assert (state / "last_scores.json").exists()
    notify.run("report", root, state, NOW, "انتشار CPI", dry_run=True, changes=True)
    assert "چه تغییر کرد" in (root / "messages" / "overview.html").read_text(encoding="utf-8")
    notify.run("prealert", root, state, NOW, None, dry_run=True, events=["cpi-2026-10-15"])
    assert (root / "messages" / "prealert.html").exists()


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
    assert set(crons) == {scheduler.INTRADAY, scheduler.DAILY, scheduler.WEEKLY,
                          scheduler.PREPOS, *scheduler.RELEASE}
    assert {scheduler.mode_for(c) for c in crons} == {"intraday", "daily", "weekly", "prepos", "release"}


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
    _daily_done(tmp_path)
    ev = [{"event_id": "cpi-2026-09-29", "release_id": "cpi", "name_fa": "CPI", "importance": 5, "scheduled_utc": NOW - pd.Timedelta(minutes=5)},
          {"event_id": "claims-2026-09-29", "release_id": "claims", "name_fa": "Claims", "importance": 3, "scheduled_utc": NOW - pd.Timedelta(minutes=5)},
          {"event_id": "ppi-2026-09-29", "release_id": "ppi", "name_fa": "PPI", "importance": 4, "scheduled_utc": NOW - pd.Timedelta(hours=7)}]
    p = scheduler.plan(scheduler.RELEASE[0], NOW, tmp_path, calendar_fn=lambda: ev)
    assert p.mode == "release" and [e["event_id"] for e in p.events] == ["cpi-2026-09-29"] and "CPI" in p.trigger
    scheduler.mark(tmp_path, ["cpi-2026-09-29"])
    assert scheduler.plan(scheduler.RELEASE[0], NOW, tmp_path, calendar_fn=lambda: ev).mode == "none"
    assert scheduler.plan(scheduler.INTRADAY, NOW, tmp_path, calendar_fn=lambda: []).mode == "intraday"
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
    _daily_done(tmp_path)
    cpi = [{"event_id": "cpi-x", "release_id": "cpi", "scheduled_utc": NOW + pd.Timedelta(hours=30)}]
    far = [{"event_id": "cpi-y", "release_id": "cpi", "scheduled_utc": NOW + pd.Timedelta(hours=80)},
           {"event_id": "claims-z", "release_id": "claims", "importance": 3, "scheduled_utc": NOW + pd.Timedelta(hours=5)}]
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


# ── decision summary (top of each asset message) ─────────────────────────

def _summary(msg: str) -> str:
    return msg.split("━━━━━━━━━━━━━━", 1)[0]


def bare(s: str) -> str:
    """Text as a reader sees it: no direction isolates, no HTML tags."""
    return re.sub(r"<[^>]+>", "", plain(s))



















def _plain_text(s: str) -> str:
    """Text as a reader sees it: no direction marks/isolates, no HTML tags."""
    return re.sub(r"<[^>]+>", "", re.sub("[‎⁦-⁩]", "", s))


def _report(root, trigger="انتشار CPI"):
    return dict(asset_report.report_messages(facts_mod.load(root), NOW, None, trigger))


# ── Persian (Jalali) dates ───────────────────────────────────────────────

@pytest.mark.parametrize("greg,jal", [((2026, 10, 2), (1405, 7, 10)), ((2026, 3, 21), (1405, 1, 1)),
                                      ((2026, 3, 20), (1404, 12, 29)), ((2025, 3, 21), (1404, 1, 1)),
                                      ((2024, 3, 20), (1403, 1, 1)), ((2025, 3, 20), (1403, 12, 30)),
                                      ((2026, 9, 29), (1405, 7, 7)), ((2027, 1, 1), (1405, 10, 11))])
def test_jalali_conversion_known_dates(greg, jal):
    from core.jalali import to_jalali
    assert to_jalali(*greg) == jal


def test_jalali_formats_in_tehran_time():
    from core.jalali import jdate, jdate_short, jdatetime_fa
    assert jdatetime_fa("2026-10-01T21:32:00+00:00") == "جمعه ۱۰ مهر ۱۴۰۵، ساعت ۰۱:۰۲"  # UTC → Tehran crosses midnight
    assert messages.tehran(pd.Timestamp("2026-10-01 21:32", tz="UTC")) == "جمعه ۱۰ مهر ۱۴۰۵، ساعت ۰۱:۰۲"
    assert jdate("2026-10-28") == "چهارشنبه ۶ آبان ۱۴۰۵"  # a bare calendar date is not shifted
    assert jdate_short("2026-10-14T12:30:00+00:00", now="2026-10-02") == "۲۲ مهر"
    assert jdate_short("2027-03-25T12:30:00+00:00", now="2026-10-02") == "۵ فروردین ۱۴۰۶"  # year shown when it differs


def test_no_gregorian_dates_left_in_any_message(root):
    f = facts_mod.load(root)
    texts = [t for _, t in asset_report.report_messages(f, NOW, None, "انتشار CPI")]
    texts += [messages.headsup(f, NOW) or "", messages.news_alert({"importance": 5, "title_fa": "x", "source": "fed_all",
                                                                     "published_utc": NOW, "btc": "bullish", "gold": "bearish"})]
    for t in texts:
        assert not re.search(r"20\d\d-\d\d-\d\d", t), t[:200]


# ── overview ─────────────────────────────────────────────────────────────

def test_overview_has_outlook_release_table_and_no_footer_lines(root):
    ov = _report(root)["overview"]
    txt = _plain_text(ov)
    assert "اگر سطح رشد" not in txt and "اگر چشم‌انداز شتاب" not in txt and "← بهبود" not in txt  # transition lines gone
    assert "پیام‌های بعدی" not in txt and messages.DISCLAIMER not in txt
    assert "فاز اقتصادی: رونق — احتمال 66٪" in txt and "چشم‌انداز: ادامه‌ی رونق" in txt
    assert "موضع فدرال" in txt and "جلسه‌ی بعدی FOMC (" in txt
    table = re.findall(r"<pre>(.*?)</pre>", ov, re.S)[0]
    assert "Actual" in table and "Expect" in table and "CPI" in table and "0.42%" in table and "0.25%" in table and "▲" in table
    after = txt.split("چه منتشر شد")[1]
    assert "• شاخص قیمت مصرف‌کننده (CPI): بالاتر از انتظار (روند) — تورم داغ‌تر (شگفتی متوسط)" in after
    assert "تقویم ۷ روز آینده" in txt


def test_outlook_sentence_comes_from_the_regime_outputs():
    reg = {"regime": "Expansion", "probabilities": {"Expansion": 0.56, "Peak": 0.25, "Recession": 0.06, "Recovery": 0.13},
           "axes": {"G": 0.5, "g": -0.2, "p": 0.1, "m": -0.1}}
    s = asset_report.outlook_sentence(reg)
    assert s == "رشد بالای میانگین تاریخی است و رو به کندی؛ فشار تورم و نقدینگی رو به افزایش؛ " \
                "چشم‌انداز: ادامه‌ی رونق با ریسک نزدیک شدن به اوج"
    assert not re.search(r"\d", s)  # no new numbers
    old = asset_report.outlook_sentence({"regime": "Expansion", "probabilities": {"Expansion": 0.9, "Peak": 0.05}})
    assert old == "چشم‌انداز: ادامه‌ی رونق"  # snapshot without axes, no close second phase


# ── decision summary ─────────────────────────────────────────────────────

def _summary_of(msg: str) -> str:
    return msg.split("━━━━━━━━━━━━━━", 1)[0]


def test_decision_summary_release_table_for_this_asset(root):
    f = facts_mod.load(root)
    f.coefs.loc[(f.coefs["indicator"] == "cpi") & (f.coefs["horizon"] == "24h"), "coefficient"] = -4.0  # hot CPI → down
    top = asset_report.asset_message(f, "BTC", NOW)
    summary = _summary_of(top)
    table = re.findall(r"<pre>(.*?)</pre>", summary, re.S)[0]
    assert "CPI" in table and "0.42%" in table and "▲" in table
    txt = _plain_text(summary)
    assert "• شاخص قیمت مصرف‌کننده (CPI): بالاتر از انتظار (روند) — تورم داغ‌تر (شگفتی متوسط) → برای بیت‌کوین نزولی" in txt
    assert "• اثر بر بیت‌کوین: نزولی — اثر تاریخی متوسط" in txt and "طلا" not in txt
    # anti-hallucination: every number in the summary is one the system computed
    assert writer.unknown_numbers(_plain_text(summary), writer.asset_facts(f, "BTC")) == []
    # a release with a negligible historical effect gets no table, just one line
    f.coefs.loc[(f.coefs["indicator"] == "cpi") & (f.coefs["horizon"] == "24h"), "coefficient"] = 0.3
    quiet = _plain_text("\n".join(asset_report.decision_summary(f, "Gold", NOW)))
    assert "منتشر شد؛ اثر تاریخی آن بر طلا ناچیز است" in quiet and "Actual" not in quiet


def test_decision_summary_position_slot_respects_the_edge_gate(root):
    f = facts_mod.load(root)
    btc = _plain_text("\n".join(asset_report.decision_summary(f, "BTC", NOW)))
    assert "پیشنهاد موقعیت (پیش از انتشار): خرید" in btc  # demo: the CPI pre-release setup passed its gate
    f.prepos["live"] = [dict(r, status="watchlist") for r in f.prepos["live"]]
    btc = _plain_text("\n".join(asset_report.decision_summary(f, "BTC", NOW)))
    assert "پیشنهاد موقعیت: فعلاً هیچ — ستاپ تکنیکال هست، اما این قاعده برتری آماری اثبات‌شده" in btc
    assert "ورود ≈" not in btc and "حد ضرر ≈" not in btc


def test_decision_summary_signal_case_uses_only_computed_numbers(root):
    p = root / "signals.json"
    sig = json.loads(p.read_text())
    btc = next(a for a in sig["assets"] if a["asset"] == "BTC")
    card = {**btc["watchlist"][0], "blocked_by": [], "confidence": "medium"}
    btc["signals"], btc["watchlist"] = [card], []
    p.write_text(json.dumps(sig))
    f = facts_mod.load(root)
    f.prepos["live"] = []
    top = _plain_text("\n".join(asset_report.decision_summary(f, "BTC", NOW)))
    assert "ورود ≈ 79,800 تا 80,000" in top and "هدف ۱ ≈ 81,600" in top and "حد ضرر ≈ 78,400" in top
    allowed = {"signal": card, "facts": writer.asset_facts(f, "BTC"),
               "rounded": [round(v, -2) for v in (*card["entry_zone"], card["targets"][0], card["stop_loss"])]}
    assert writer.unknown_numbers(top, allowed) == []
    body = _plain_text(asset_report.asset_message(f, "BTC", NOW).split("━━━━━━━━━━━━━━", 1)[1])
    assert "🎯 سیگنال فعال" in body and "حد ضرر:" in body  # a live signal keeps its full card


def test_decision_summary_falls_back_to_news(root):
    f = facts_mod.load(root)
    f.releases = []
    f.news = pd.DataFrame([{"id": "n1", "title": "Fed emergency cut", "title_fa": "کاهش اضطراری نرخ", "importance": 5,
                            "published_utc": NOW - pd.Timedelta(hours=2), "btc": "bullish", "gold": "bearish", "channels": "policy"}])
    assert "اثر بر بیت‌کوین: صعودی" in _plain_text("\n".join(asset_report.decision_summary(f, "BTC", NOW)))
    assert "اثر بر طلا: نزولی" in _plain_text("\n".join(asset_report.decision_summary(f, "Gold", NOW)))


# ── asset body and indicator table ───────────────────────────────────────

def test_indicator_table_sorted_by_next_update_then_daily_then_unknown(root):
    from core.jalali import jdate_short
    f = facts_mod.load(root)
    trs = asset_report.table_rows(f, asset_report.indicator_rows(f, "BTC"), NOW)
    kinds = [0 if t.when is not None else 1 if t.daily else 2 for t in trs]
    assert kinds == sorted(kinds) and 0 in kinds and 1 in kinds
    dated = [t.when for t in trs if t.when is not None]
    assert dated == sorted(dated)
    claims = next(t for t in trs if t.row.key == "initial_claims")  # demo calendar: claims tomorrow
    assert claims.when == f.calendar.loc[f.calendar["release_id"] == "claims", "scheduled_utc"].min()
    table = asset_report.table_message(f, "BTC", NOW)
    blocks = re.findall(r"<pre>(.*?)</pre>", table, re.S)
    lines = [asset_report.visible(ln) for b in blocks for ln in b.split("\n")]
    assert max(map(len, lines)) <= asset_report.TABLE_WIDTH
    assert all(set(b.split("\n")[1].lstrip("‎")) == {"-"} for b in blocks)  # divider under each header
    assert jdate_short(claims.when, NOW) in lines[2] and "روزانه" in "".join(lines) and "⁧" in blocks[0]
    assert "• CLAIM: مدعیان اولیه بیمه بیکاری" in _plain_text(table)


def test_asset_body_keeps_only_decision_relevant_lines(root):
    f = facts_mod.load(root)
    for asset in ("BTC", "Gold"):
        msg = asset_report.asset_message(f, asset, NOW)
        body = _plain_text(msg.split("━━━━━━━━━━━━━━", 1)[1])
        drivers = body.split("عوامل اصلی")[1].split("شاخص‌های مؤثر")[0]
        numbered = re.findall(r"^\d\. ", drivers, re.M)
        assert 1 <= len(numbered) <= 3 and not re.search(r"[0-9]", re.sub(r"^\d\. ", "", drivers, flags=re.M))
        assert body.count("🎯 سیگنال: فعلاً هیچ") == 1 or "🎯 سیگنال معاملاتی" in body
        concl = body.split("جمع‌بندی")[1].split("معنی اصطلاحات")[0]
        assert len([ln for ln in concl.strip().splitlines() if ln.strip()]) <= 3
        assert messages.DISCLAIMER not in msg
        assert len(telegram.split_message(msg)) == 1
    gold = _plain_text(asset_report.asset_message(f, "Gold", NOW))
    assert "طلا (COMEX GC=F)" in gold and "PAXG (بلادرنگ، مبنای سطوح و سیگنال)" in gold


def test_invalidation_is_one_line_relative_to_the_price():
    x = {"price": 100.0, "technical": {"1d": {"last_swing_high": 110.0, "last_swing_low": 90.0}}}
    assert _plain_text(asset_report.invalidation(x, 20.0, "{:,.0f}", "")[0]) == "• ابطال: بسته‌شدن روزانه زیر 90 (کف روزانه)."
    assert "بالای 110" in _plain_text(asset_report.invalidation(x, -20.0, "{:,.0f}", "")[0])
    assert "تمایل صعودی تأیید نمی‌شود" in _plain_text(asset_report.invalidation({**x, "price": 85.0}, 20.0, "{:,.0f}", "")[0])
    assert "ساختار نزولی" in _plain_text(asset_report.invalidation({**x, "price": 85.0}, 2.0, "{:,.0f}", " (PAXG)")[0])


# ── glossary and disclaimer ──────────────────────────────────────────────

def test_disclaimer_once_at_the_end_and_short_glossary(root):
    rep = _report(root)
    assert list(rep) == ["overview", "report_btc", "report_gold", "table_btc", "table_gold", "glossary"]
    assert sum(t.count(messages.DISCLAIMER) for t in rep.values()) == 1 and rep["glossary"].endswith(messages.DISCLAIMER)
    lines = [ln for ln in rep["glossary"].split("\n") if ln.startswith("• ")]
    assert len(lines) == len(asset_report.GLOSSARY) and all(len(_plain_text(ln)) < 140 for ln in lines)
    joined = _plain_text("\n".join(t for k, t in rep.items() if k != "glossary"))
    for term in ("امتیاز کلان", "انتظار", "شگفتی", "برتری اثبات‌شده", "دامنه‌ی محتمل ۴ هفته", "PAXG", "پیش‌موقعیت‌گیری",
                 "فاز اقتصادی", "موضع فدرال"):
        assert term in joined, term  # every glossary term still appears in the messages
