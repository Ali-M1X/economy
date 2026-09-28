"""Runtime settings, read only from environment variables (never hard-coded)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env when present (VPS / dev); CI uses real env vars
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value not in ("", None) else default


@dataclass(frozen=True)
class Settings:
    fred_api_key: str | None = field(default_factory=lambda: _env("FRED_API_KEY"))
    bls_api_key: str | None = field(default_factory=lambda: _env("BLS_API_KEY"))
    # Postgres connection string. On Supabase: Project Settings → Database → Connection string (URI).
    # Falls back to a local SQLite file so collectors can run without a database.
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL") or f"sqlite:///{ROOT / 'data' / 'macro_pulse.sqlite'}"
    )
    supabase_url: str | None = field(default_factory=lambda: _env("SUPABASE_URL"))
    supabase_key: str | None = field(default_factory=lambda: _env("SUPABASE_KEY"))
    telegram_bot_token: str | None = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str | None = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-sonnet-5"))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "30")))
    user_agent: str = field(
        default_factory=lambda: _env(
            "HTTP_USER_AGENT", "MacroPulse/0.1 (+https://github.com/ali-m1x/economy; research, non-commercial)"
        )
    )
    # Optional sources whose terms are unclear; opt-in only (see DATA_GAPS.md).
    enable_ff_calendar: bool = field(default_factory=lambda: _env("ENABLE_FF_CALENDAR", "0") == "1")


def get_settings() -> Settings:
    return Settings()
