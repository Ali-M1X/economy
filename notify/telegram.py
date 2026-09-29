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

from core.settings import ROOT, get_settings

API = "https://api.telegram.org/bot{token}/sendMessage"
LIMIT = 4096
OUTBOX = ROOT / "output" / "outbox"


class TelegramError(RuntimeError):
    pass


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
    token, chat = s.telegram_bot_token, chat_id or s.telegram_chat_id
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
            raise TelegramError(f"Telegram API {r.status_code} on part {i + 1}/{len(parts)}: {desc}")
        else:
            raise TelegramError(f"Telegram rate limit persisted on part {i + 1}/{len(parts)}")
    return SendResult(True, len(parts), "telegram")
