"""Compose and send the Persian Telegram messages.

    python -m jobs.notify report  --root output [--trigger "CPI release"]   # impact report + numeric signals
    python -m jobs.notify weekly  --root output                            # weekly summary
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
from pathlib import Path

import pandas as pd

from jobs import health as health_mod
from notify import telegram
from reports import facts as facts_mod
from reports import messages, writer

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


def run(mode: str, root: Path, state: Path | None, now: pd.Timestamp, trigger: str | None, dry_run: bool) -> int:
    f = facts_mod.load(root)
    names = (dict(zip(f.calendar["name_en"], f.calendar["name_fa"]))
             if not f.calendar.empty and {"name_en", "name_fa"} <= set(f.calendar.columns) else {})
    if mode == "health":
        gate(root, state, now, "بررسی دوره‌ای", dry_run)
        return 0
    if mode in ("report", "weekly"):
        context = "گزارش هفتگی" if mode == "weekly" else "گزارش اثر"
        if not gate(root, state, now, context, dry_run):
            return 0
        if mode == "weekly":
            _send(messages.weekly_summary(f, now), "weekly", dry_run, root)
            return 0
        text, source = writer.explain(f)
        print(f"causal explanation: {source}")
        _send(messages.impact_report(f, now, text, trigger), "impact", dry_run, root)
        _send(messages.signals_message(f, names), "signals", dry_run, root)
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
    raise SystemExit(f"unknown mode {mode}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report", "weekly", "headsup", "news", "health"])
    ap.add_argument("--root", default="output")
    ap.add_argument("--state")
    ap.add_argument("--trigger", help="why this report runs (shown in the message)")
    ap.add_argument("--dry-run", action="store_true", help="write to output/outbox/ even if Telegram secrets are set")
    args = ap.parse_args(argv)
    try:
        return run(args.mode, Path(args.root), Path(args.state) if args.state else None, pd.Timestamp.now(tz="UTC"),
                   args.trigger, args.dry_run)
    except telegram.TelegramError as e:
        print(f"::error::{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
