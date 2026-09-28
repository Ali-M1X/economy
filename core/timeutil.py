"""Time helpers. All storage is UTC; Tehran time is only for display."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
TEHRAN = ZoneInfo("Asia/Tehran")
NEW_YORK = ZoneInfo("America/New_York")


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_tehran(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return dt.astimezone(TEHRAN)


def et_to_utc(naive_et: datetime) -> datetime:
    """Interpret a naive US/Eastern wall-clock time (how BLS/BEA/Fed publish schedules) as UTC."""
    return naive_et.replace(tzinfo=NEW_YORK).astimezone(UTC)
