"""Compose and send the Persian Telegram messages.

    python -m jobs.notify report  --root output [--trigger "CPI release"]   # overview + BTC + gold + tables + glossary
    python -m jobs.notify weekly  --root output                            # the same, as the weekly summary
    python -m jobs.notify daily   --root output                            # the 09:00 Tehran report, once per day
    python -m jobs.notify report  --root output --changes                  # after a release: + what changed
    python -m jobs.notify prealert --root snap --events nfp-2026-10-02     # shortly before an important release
    python -m jobs.notify headsup --root output                            # releases scheduled for tomorrow (Tehran)
    python -m jobs.notify news    --root output                            # alerts for new importance ≥ 4 news
    python -m jobs.notify health  --root output                            # warning only, if the data is unhealthy

`--state DIR` keeps what was already sent (news ids, last health warning, heads-up dates) between scheduled runs;
the workflows restore it from the Actions cache. Without Telegram secrets every message goes to output/outbox/.
Reports are gated by jobs.health: with bad data a health warning is sent instead of the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from jobs import health as health_mod
from jobs import scheduler
from notify import telegram
from reports import facts as facts_mod
from reports import asset_report, messages, writer

MAX_NEWS_ALERTS = 3
NEWS_WINDOW = pd.Timedelta(hours=6)


def _state_json(state: Path | None, name: str) -> dict:
    p = state / name if state else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else {}


def _save_state(state: Path | None, name: str, obj: dict) -> None:
    if state:
        state.mkdir(parents=True, exist_ok=True)
        (state / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _out(key: str, value: str) -> None:
    """Expose a value to later workflow steps."""
    if os.environ.get("GITHUB_OUTPUT"):
        # values can carry external text (news titles): a newline would let it inject further outputs
        value = " ".join(str(value).split())
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")


def _archive(text: str, kind: str, root: Path) -> None:
    """Keep the latest message of each kind in the snapshot (shown on the dashboard's reports tab)."""
    d = root / "messages"
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes")
    (d / f"{kind}.html").write_text(f"<!-- {stamp} -->\n{text}", encoding="utf-8")


def _send(text: str, kind: str, dry_run: bool, root: Path | None = None) -> None:
    if root is not None:
        _archive(text, kind, root)
    r = telegram.send(text, kind=kind, dry_run=dry_run)
    print(f"{kind}: {'sent to Telegram' if r.sent else 'dry run → ' + r.where} ({r.parts} part(s))")


def gate(root: Path, state: Path | None, now: pd.Timestamp, context: str, dry_run: bool) -> bool:
    h = health_mod.evaluate(root, now)
    if h.ok:
        return True
    print(f"::warning::data health check failed ({len(h.blocking)} blocking problems) — sending a warning instead of the {context}")
    if health_mod.should_alert(h, state, now):
        _send(messages.health_warning(h.blocking, context), "health", dry_run, root)
    else:
        print("same health problems already reported recently — not repeating the warning")
    return False


def snapshot_scores(f, now: pd.Timestamp) -> dict:
    """What the next report compares against: price and macro scores per asset."""
    out = {"asof": now.isoformat(), "assets": {}}
    for a in ("BTC", "Gold"):
        m = f.macro.get(a) or {}
        out["assets"][a] = {"price": (f.asset(a) or {}).get("price"),
                            "scores": {b: (m.get(b) or {}).get("score") for b, _ in asset_report.HORIZONS}}
    return out


def _send_report(f, now: pd.Timestamp, trigger: str | None, names: dict, dry_run: bool, root: Path,
                 weekly: bool, state: Path | None = None, changes: bool = False) -> None:
    # one overview (shared context), one self-contained report per asset, the full indicator tables, a glossary
    chains = {}
    for a in ("BTC", "Gold"):
        chains[a], source = writer.explain_asset(f, a, asset_report.chain_sentences(f, a, now))
        print(f"causal chain {a}: {source}")
    current = snapshot_scores(f, now)
    block = asset_report.changes_block(_state_json(state, "last_scores.json"), current) if changes else []
    for kind, text in asset_report.report_messages(f, now, chains, trigger, names, weekly=weekly):
        if kind == "overview" and block:
            head, _, rest = text.partition("\n\n")
            text = head + "\n\n" + "\n".join(block) + "\n\n" + rest
        _send(text, kind, dry_run, root)
    _save_state(state, "last_scores.json", current)


def wait_for_daily_slot(now: pd.Timestamp, sleep=time.sleep) -> None:
    """The data is refreshed shortly before 09:00 Tehran; the report itself goes out at 09:00."""
    left = (scheduler.daily_target(now) - now).total_seconds()
    if 0 < left <= 45 * 60:
        print(f"daily report: waiting {int(left)} s for 09:00 Tehran", flush=True)
        sleep(left)


def run(mode: str, root: Path, state: Path | None, now: pd.Timestamp, trigger: str | None, dry_run: bool,
        events: list[str] | None = None, changes: bool = False) -> int:
    f = facts_mod.load(root)
    names = (dict(zip(f.calendar["name_en"], f.calendar["name_fa"]))
             if not f.calendar.empty and {"name_en", "name_fa"} <= set(f.calendar.columns) else {})
    if mode == "health":
        gate(root, state, now, "بررسی دوره‌ای", dry_run)
        return 0
    if mode == "daily":
        today = scheduler.tehran_date(now)
        if scheduler.daily_sent(state, now):
            print(f"daily report for {today} already sent")
            return 0
        if not gate(root, state, now, "گزارش روزانه", dry_run):
            return 0  # not marked as sent: a later run tries again
        if not dry_run:
            wait_for_daily_slot(now)
            now = pd.Timestamp.now(tz="UTC")
        _send_report(f, now, trigger or "گزارش روزانه", names, dry_run, root, weekly=False, state=state)
        _save_state(state, "daily_sent.json", {"date": today})
        return 0
    if mode in ("report", "weekly"):
        context = "گزارش هفتگی" if mode == "weekly" else "گزارش اثر"
        if not gate(root, state, now, context, dry_run):
            return 0
        _send_report(f, now, trigger, names, dry_run, root, weekly=mode == "weekly", state=state, changes=changes)
        return 0
    if mode == "prealert":
        evs = [scheduler.event_from_id(i) for i in events or []]
        msg = messages.prerelease(f, now, [{**e, "scheduled_utc": pd.Timestamp(e["scheduled_utc"])} for e in evs])
        if msg is None:
            print("prealert: no events given")
            return 0
        _send(msg, "prealert", dry_run, root)
        return 0
    if mode == "headsup":
        tomorrow = str((now.tz_convert("Asia/Tehran") + pd.Timedelta(days=1)).date())
        sent = _state_json(state, "headsup_sent.json")
        if sent.get("date") == tomorrow:
            print(f"heads-up for {tomorrow} already sent")
            return 0
        msg = messages.headsup(f, now)
        if msg is None:
            print(f"no tracked release on {tomorrow} (Tehran)")
            return 0
        _send(msg, "headsup", dry_run, root)
        _save_state(state, "headsup_sent.json", {"date": tomorrow})
        return 0
    if mode == "news":
        seen = set(_state_json(state, "news_alerted.json").get("ids", []))
        imp = f.important_news(now - NEWS_WINDOW)
        new = imp[~imp["id"].isin(seen)] if not imp.empty else imp
        for item in new.head(MAX_NEWS_ALERTS).to_dict("records"):
            _send(messages.news_alert(item), "news", dry_run)
            seen.add(item["id"])
        _save_state(state, "news_alerted.json", {"ids": sorted(seen)[-500:]})
        top = int(new["importance"].max()) if not new.empty else 0
        _out("trigger_report", "true" if top >= 5 else "false")  # market-moving news → full report run
        if top >= 5:
            _out("trigger_text", "خبر مهم: " + str(new.iloc[0].get("title_fa") or new.iloc[0]["title"])[:200])
        print(f"news: {len(new)} new important item(s), max importance {top}")
        return 0
    if mode == "prepos":
        # pre-positioning alerts: only rows whose release/asset passed the out-of-sample edge gate ("signal"),
        # once per release and asset; blocked setups stay on the dashboard (same rule as Phase 4)
        sent = set(_state_json(state, "prepos_sent.json").get("keys", []))
        rows = (f.prepos or {}).get("live", [])
        n = 0
        for r in rows:
            key = f"{r['event_id']}:{r['asset']}"
            if r.get("status") != "signal" or key in sent:
                continue
            bt = (f.prepos.get("backtest") or {}).get(f"{r['release']}:{r['asset']}")
            _send(messages.prepos_alert(r, bt), "prepos", dry_run, root)
            sent.add(key)
            n += 1
        _save_state(state, "prepos_sent.json", {"keys": sorted(sent)[-300:]})
        blocked = sum(1 for r in rows if r.get("status") == "watchlist")
        print(f"prepos: {n} alert(s) sent; {blocked} setup(s) blocked by the edge gate (dashboard only)")
        return 0
    raise SystemExit(f"unknown mode {mode}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report", "daily", "weekly", "headsup", "news", "health", "prepos", "prealert"])
    ap.add_argument("--events", default="", help="comma-separated event ids (prealert)")
    ap.add_argument("--changes", action="store_true", help="report: add what changed since the last report")
    ap.add_argument("--root", default="output")
    ap.add_argument("--state")
    ap.add_argument("--trigger", help="why this report runs (shown in the message)")
    ap.add_argument("--dry-run", action="store_true", help="write to output/outbox/ even if Telegram secrets are set")
    args = ap.parse_args(argv)
    try:
        return run(args.mode, Path(args.root), Path(args.state) if args.state else None, pd.Timestamp.now(tz="UTC"),
                   args.trigger, args.dry_run, [x for x in args.events.split(",") if x], args.changes)
    except telegram.TelegramError as e:
        print(f"::error::{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
