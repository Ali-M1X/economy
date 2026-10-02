"""Persian (Jalali / Shamsi) dates in Tehran time — the one helper every Telegram message uses.

`to_jalali` is the standard arithmetic Gregorian → Jalali conversion (the 33-year-cycle algorithm used by jalaali-js
and jdatetime); tests pin it against known dates (1 Farvardin, Mehr, leap years).
"""

from __future__ import annotations

import pandas as pd

from core.timeutil import TEHRAN

MONTHS = ("فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند")
WEEKDAYS = ("دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه")  # Python weekday(): Monday = 0
_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_DAYS_BEFORE = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Gregorian (year, month, day) → Jalali (year, month, day)."""
    gy2 = gy + 1 if gm > 2 else gy
    days = (355666 + 365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400 + gd + _DAYS_BEFORE[gm - 1])
    jy = -1595 + 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        return jy, 1 + days // 31, 1 + days % 31
    return jy, 7 + (days - 186) // 30, 1 + (days - 186) % 30


def fa_digits(s) -> str:
    return str(s).translate(_FA_DIGITS)


def tehran_ts(ts) -> pd.Timestamp:
    """Any timestamp/date/ISO string → Tehran time (naive values are taken as UTC; plain dates stay that date)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        if t == t.normalize() and isinstance(ts, str) and len(ts) <= 10:  # a bare calendar date: no time-zone shift
            return t
        t = t.tz_localize("UTC")
    return t.tz_convert(TEHRAN)


def jdate(ts, weekday: bool = True, year: bool = True) -> str:
    """'جمعه ۱۰ مهر ۱۴۰۵' (Tehran date)."""
    t = tehran_ts(ts)
    y, m, d = to_jalali(t.year, t.month, t.day)
    out = f"{fa_digits(d)} {MONTHS[m - 1]}" + (f" {fa_digits(y)}" if year else "")
    return f"{WEEKDAYS[t.weekday()]} {out}" if weekday else out


def jdatetime_fa(ts, weekday: bool = True) -> str:
    """'جمعه ۱۰ مهر ۱۴۰۵، ساعت ۰۱:۰۲' (Tehran time)."""
    t = tehran_ts(ts)
    return f"{jdate(t, weekday=weekday)}، ساعت {fa_digits(f'{t:%H:%M}')}"


def jdate_short(ts, now=None) -> str:
    """'۱۰ مهر' for tables; '۱۰ مهر ۱۴۰۵' when the Jalali year differs from `now`'s."""
    t = tehran_ts(ts)
    y = to_jalali(t.year, t.month, t.day)[0]
    ny = to_jalali(*(lambda n: (n.year, n.month, n.day))(tehran_ts(now if now is not None else pd.Timestamp.now(tz="UTC"))))[0]
    return jdate(t, weekday=False, year=y != ny)

