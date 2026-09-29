"""Persian rendering of the English strings the analytics layer emits (signal reasons, risks, blocks, states).

The engines keep English output (JSON/markdown for developers); everything a user reads goes through `fa()`.
Rules are ordered regex → template pairs; an unmatched string is returned unchanged, so a new English message
shows up untranslated rather than disappearing.
"""

from __future__ import annotations

import re

WORDS = {
    "up": "صعودی", "down": "نزولی", "mixed": "نامشخص", "range": "رنج",
    "bullish (HH/HL)": "صعودی (سقف و کف بالاتر)", "bearish (LH/LL)": "نزولی (سقف و کف پایین‌تر)",
    "BOS_UP": "شکست ساختار رو به بالا", "BOS_DOWN": "شکست ساختار رو به پایین",
    "CHOCH_UP": "تغییر ماهیت رو به بالا", "CHOCH_DOWN": "تغییر ماهیت رو به پایین",
    "bullish": "صعودی", "bearish": "نزولی", "long": "خرید", "short": "فروش",
    "high": "بالا", "medium": "متوسط", "low": "پایین",
    "above": "بالای", "below": "زیر",
    "swing low": "کف سوئینگ", "swing high": "سقف سوئینگ", "POC": "پرحجم‌ترین قیمت", "VAL": "کف ناحیه ارزش",
    "VAH": "سقف ناحیه ارزش", "equal lows": "کف‌های برابر", "equal highs": "سقف‌های برابر",
    "prior-day low": "کف روز قبل", "prior-day high": "سقف روز قبل", "prior-week low": "کف هفته قبل",
    "prior-week high": "سقف هفته قبل", "EMA50": "میانگین ۵۰", "EMA200": "میانگین ۲۰۰",
    "inverted": "وارونه", "re-steepening": "در حال بازگشت شیب", "normal": "عادی",
    "Dot-com peak → 2001 recession": "اوج دات‌کام ← رکود ۲۰۰۱",
    "Housing peak → 2007-09 GFC": "اوج مسکن ← بحران مالی ۲۰۰۷–۰۹",
    "Trade war → 2020 recession (COVID)": "جنگ تجاری ← رکود ۲۰۲۰ (کووید)",
    "Inflation shock → longest inversion on record": "شوک تورمی ← طولانی‌ترین وارونگی ثبت‌شده",
    "Expansion": "رونق", "Peak": "اوج", "Recession": "رکود", "Recovery": "بهبود",
}

LABELS = {  # identifiers → Persian labels for tables
    "a1_policy_direction": "جهت آخرین تغییر نرخ", "a2_market_path": "مسیر نرخ در آتی‌ها", "b_balance_sheet": "روند ترازنامه",
    "c_communications": "لحن بیانیه‌ها و سخنرانی‌ها", "d_dot_plot": "Dot Plot در برابر نرخ فعلی",
    "last_change_date": "تاریخ آخرین تغییر", "last_change_bp": "آخرین تغییر (bp)", "contract": "قرارداد",
    "implied_rate": "نرخ ضمنی", "delta_pp": "فاصله (واحد درصد)", "walcl_13w_ann_pct": "رشد سالانه ترازنامه در ۱۳ هفته (٪)",
    "n_docs": "تعداد اسناد طبقه‌بندی‌شده", "sep_year": "سال پیش‌بینی", "sep_median": "میانه Dot Plot", "target_mid": "میانه محدوده هدف",
    "fed_monetary": "فدرال — سیاست پولی", "fed_all": "فدرال — اطلاعیه‌ها", "fed_speeches": "فدرال — سخنرانی‌ها",
    "fed_testimony": "فدرال — شهادت کنگره", "bls": "BLS", "bea": "BEA", "treasury": "خزانه‌داری", "gdelt": "GDELT",
    "D": "روزانه", "W": "هفتگی", "M": "ماهانه", "Q": "فصلی", "A": "سالانه",
}

RULES: list[tuple[str, str]] = [
    (r"^→ (\w+) if the growth level crosses its historical average \(now ([+-][\d.]+)σ\)$",
     "← {1} اگر سطح رشد از میانگین تاریخی‌اش عبور کند (اکنون {2}σ)"),
    (r"^→ (\w+) if the momentum outlook .* crosses zero \(now ([+-][\d.]+)\)$",
     "← {1} اگر چشم‌انداز شتاب (شتاب رشد در برابر فشار تورم/نقدینگی) از صفر عبور کند (اکنون {2})"),
    (r"^daily trend (up|down)$", "روند روزانه {1}"),
    (r"^4h structure (bullish|bearish)$", "ساختار ۴ساعته {1}"),
    (r"^pullback held (.+?) ([\d,.]+)$", "اصلاح روی {1} {2} نگه داشت"),
    (r"^RSI (\d+)$", "RSI برابر {1}"),
    (r"^MACD histogram turning$", "چرخش هیستوگرام MACD"),
    (r"^4h close (below|above) ([\d,.]+), or the daily trend flips$", "بسته شدن کندل ۴ساعته {1} {2} یا برگشت روند روزانه"),
    (r"^macro not aligned$", "امتیاز کلان هم‌جهت نیست (کمتر از آستانه ۱۵ یا خلاف جهت)"),
    (r"^no demonstrated edge \(OOS avg R (\S+) over (\d+) trades\)$",
     "برتری آماری اثبات‌نشده (میانگین R خارج از نمونه {1} در {2} معامله)"),
    (r"^Upcoming: (.+) at (.+) Tehran — volatility spike likely$", "رویداد پیش‌رو: {1} در {2} به وقت تهران — احتمال جهش نوسان"),
    (r"^Regime uncertain: (\w+) at only (\d+)% — switch risk (\d+)%$",
     "فاز اقتصادی نامطمئن: {1} فقط با احتمال {2}٪ — ریسک تغییر فاز {3}٪"),
    (r"^Conflicting indicators: (.+)$", "شاخص‌های مخالف: {1}"),
    (r"^too few out-of-sample trades \((\d+) < (\d+)\)$", "معاملات خارج از نمونه کم است ({1} از حداقل {2})"),
    (r"^average R not positive \((\S+)\)$", "میانگین R مثبت نیست ({1})"),
    (r"^not significant \(p = (\S+)\)$", "از نظر آماری معنادار نیست (p = {1})"),
    (r"^not enough history for this release/asset$", "تاریخچه‌ی کافی برای این انتشار و دارایی نیست"),
    (r"^Weekend: .*$", "آخر هفته: نقدشوندگی بیت‌کوین کمتر است؛ گپ تا بازگشایی CME دوشنبه رایج است"),
    (r"^Gold futures market closed: .*$", "بازار آتی طلا بسته است: PAXG ۲۴/۷ معامله می‌شود اما کم‌عمق؛ ممکن است هنگام بازگشایی گپ بخورد"),
    (r"^Low-liquidity window .*$", "بازه کم‌نقدشوندگی (بسته شدن آمریکا تا بازگشایی آسیا، حدود ۰۰:۳۰ تا ۰۴:۳۰ تهران)"),
]
_COMPILED = [(re.compile(p), t) for p, t in RULES]
WEEKDAYS = {"Mon": "دوشنبه", "Tue": "سه‌شنبه", "Wed": "چهارشنبه", "Thu": "پنجشنبه", "Fri": "جمعه", "Sat": "شنبه",
            "Sun": "یکشنبه"}
_WD = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b")


def ltr(x) -> str:
    """Isolate a number/date/code inside Persian text so its sign and digit order survive the bidi algorithm."""
    return f"\u2066{x}\u2069"


def label(k: str) -> str:
    if isinstance(k, str) and k.startswith("gdelt:"):
        return "GDELT · " + k.split(":", 1)[1]
    return LABELS.get(k, k)


def word(s: str | None) -> str:
    return "—" if s is None else WORDS.get(s, s)


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?")
_NUMERIC = re.compile(r"[+\-−]?[\d.,:%σ\- ]*\d[\d.,:%σ]*")


def _piece(g: str, lookup: dict[str, str]) -> str:
    g = lookup.get(g, g)
    if _NUMERIC.fullmatch(g):
        return ltr(g)
    g = _WD.sub(lambda w: WEEKDAYS[w.group(1)], g)
    return _DATE.sub(lambda d: ltr(d.group(0)), g)


def fa(s: str, names: dict[str, str] | None = None) -> str:
    """Translate one message. `names` maps extra English names (e.g. calendar release names) to Persian."""
    lookup = {**WORDS, **(names or {})}
    if s in lookup:
        return lookup[s]
    for rx, tpl in _COMPILED:
        m = rx.match(s)
        if m:
            return tpl.format(None, *(_piece(g, lookup) for g in m.groups()))
    return s
