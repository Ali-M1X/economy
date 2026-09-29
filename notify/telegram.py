"""Telegram Bot API sender (HTML parse mode).

Without TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID the message is written to `output/outbox/` instead (dry run), so every
job can run and be reviewed before the bot exists. The token is part of the API URL: it is never logged, and errors
are re-raised without the URL.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass

import requests

from core.settings import OUTPUT_DIR, get_settings

API = "https://api.telegram.org/bot{token}/sendMessage"
BASE = "https://api.telegram.org/bot{token}/{method}"
LIMIT = 4096
OUTBOX = OUTPUT_DIR / "outbox"
TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")


class TelegramError(RuntimeError):
    pass


def clean_token(raw: str | None) -> str | None:
    """Undo common paste mistakes: surrounding whitespace/newlines, quotes, and a leading "bot" prefix."""
    if not raw:
        return raw
    t = raw.strip().strip("'\"").strip()
    return t[3:] if t.lower().startswith("bot") and TOKEN_RE.match(t[3:]) else t


def describe_token(token: str | None) -> str:
    """Shape of the token without revealing it (safe to log)."""
    if not token:
        return "empty"
    bits = [f"{len(token)} chars", "format ok" if TOKEN_RE.match(token) else "format NOT <digits>:<35-char secret>"]
    if token != token.strip():
        bits.append("has surrounding whitespace")
    return ", ".join(bits)


def _clean_chat(raw: str | None) -> str | None:
    return raw.strip().strip("'\"").strip() if raw else raw


@dataclass
class SendResult:
    sent: bool          # False = dry run (written to the outbox)
    parts: int
    where: str          # "telegram" or the outbox file path


def esc(x) -> str:
    """Escape dynamic text for HTML parse mode (only <, >, & matter)."""
    return html.escape(str(x), quote=False)


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    """Split on blank lines, then lines, never inside an HTML tag pair; each part ≤ limit characters."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for block in re.split(r"(\n\n)", text):
        if len(cur) + len(block) <= limit:
            cur += block
            continue
        if cur.strip():
            parts.append(cur.strip("\n"))
        cur = ""
        while len(block) > limit:  # a single oversized block: cut at the last newline before the limit
            cut = block.rfind("\n", 0, limit)
            cut = cut if cut > 0 else limit
            parts.append(block[:cut])
            block = block[cut:].lstrip("\n")
        cur = block
    if cur.strip():
        parts.append(cur.strip("\n"))
    return parts


def send(text: str, *, kind: str = "message", dry_run: bool | None = None, session: requests.Session | None = None,
         chat_id: str | None = None) -> SendResult:
    s = get_settings()
    token, chat = clean_token(s.telegram_bot_token), _clean_chat(chat_id or s.telegram_chat_id)
    parts = split_message(text)
    if dry_run or not (token and chat):
        OUTBOX.mkdir(parents=True, exist_ok=True)
        path = OUTBOX / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}.html"
        path.write_text("\n\n-----\n\n".join(parts), encoding="utf-8")
        return SendResult(False, len(parts), str(path))
    http = session or requests.Session()
    for i, part in enumerate(parts):
        for attempt in range(3):
            try:
                r = http.post(API.format(token=token), timeout=30, json={
                    "chat_id": chat, "text": part, "parse_mode": "HTML", "disable_web_page_preview": True})
            except requests.RequestException as e:  # network error: the message may contain the URL → drop it
                if attempt == 2:
                    raise TelegramError(f"network error sending part {i + 1}/{len(parts)}: {type(e).__name__}") from None
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429:  # flood control
                time.sleep(float(r.json().get("parameters", {}).get("retry_after", 3)))
                continue
            if r.ok:
                break
            desc = r.json().get("description", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            hint = ""
            if r.status_code in (401, 404):
                hint = (f" — Telegram rejected the bot token ({describe_token(token)}); "
                        "run `python -m notify.telegram --check`")
            elif r.status_code == 400 and "chat not found" in desc.lower():
                hint = " — the token works but TELEGRAM_CHAT_ID is wrong, or the bot was never messaged / added to the chat"
            raise TelegramError(f"Telegram API {r.status_code} on part {i + 1}/{len(parts)}: {desc}{hint}")
        else:
            raise TelegramError(f"Telegram rate limit persisted on part {i + 1}/{len(parts)}")
    return SendResult(True, len(parts), "telegram")


def check(session: requests.Session | None = None, send_test: bool = False) -> list[str]:
    """Verify TELEGRAM_BOT_TOKEN (getMe) and TELEGRAM_CHAT_ID (getChat) without printing either. Returns report lines;
    raises TelegramError on the first failure."""
    s = get_settings()
    raw, token, chat = s.telegram_bot_token, clean_token(s.telegram_bot_token), _clean_chat(s.telegram_chat_id)
    http = session or requests.Session()
    out = [f"TELEGRAM_BOT_TOKEN: {describe_token(raw)}" + (" (cleaned before use)" if raw and raw != token else "")]
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set (GitHub: Settings → Secrets and variables → Actions)")
    if not chat:
        raise TelegramError("TELEGRAM_CHAT_ID is not set")

    def call(method: str, **params):
        try:
            r = http.post(BASE.format(token=token, method=method), json=params, timeout=30)
        except requests.RequestException as e:
            raise TelegramError(f"{method}: network error {type(e).__name__}") from None
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return r.status_code, body

    code, body = call("getMe")
    if code != 200 or not body.get("ok"):
        raise TelegramError(f"getMe → {code} {body.get('description', '')}: the bot token is invalid or revoked. "
                            "In @BotFather: /mybots → your bot → API Token, copy it again (no 'bot' prefix, no spaces).")
    me = body["result"]
    out.append(f"token OK: bot @{me.get('username')}")
    code, body = call("getChat", chat_id=chat)
    if code != 200 or not body.get("ok"):
        raise TelegramError(f"getChat → {code} {body.get('description', '')}: TELEGRAM_CHAT_ID is wrong or the bot "
                            f"has no access. Send /start to @{me.get('username')} (or add it to the group/channel as "
                            "admin), then read the id from getUpdates. Group ids start with -100.")
    ctype = body["result"].get("type")
    out.append(f"chat OK: {ctype} chat (name not printed — Action logs of a public repo are public)")
    if send_test:
        send("✅ Macro Pulse: اتصال تلگرام برقرار است.", kind="check", session=http)
        out.append("test message sent")
    return out


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Verify the Telegram bot token and chat id (neither is printed).")
    ap.add_argument("--check", action="store_true", required=True)
    ap.add_argument("--send-test", action="store_true", help="also send a short test message")
    a = ap.parse_args()
    try:
        for line in check(send_test=a.send_test):
            print(line)
    except TelegramError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
