"""The daily report (09:00 Tehran), decision first; also the report after a release, with the release on top.

    ☀️ title
    🎯 decision: one line per asset (bias — action; invalidation level) and the key event with its real impact
    📊 final numbers: the latest print of each release out in the last 7 days or due in the next 7, with forecast,
       previous and what it means in plain words (annualised where it helps)
    📅 ahead: per important release — forecast and odds, the usual 1-hour shock (× a normal hour), whether the direction
       can be predicted, and the live front-running read inside 48 h
    ₿ / 🥇 price, two sentences, levels and 4-week odds
    ⚠️ one disclaimer line

One Telegram message on a normal day (≤ TARGET_CHARS); split at section boundaries ("بخش ۲ از ۲") only when needed.
Every line starts with a Persian letter (after an optional emoji or bullet) so Telegram aligns it right to left.
Every number is computed by code (reports.advisor, reports.outlook, models.release_impact). If LLM_API_KEY is set, a
free model may rephrase the two sentences per asset; a rewrite that adds any number is discarded.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from core.jalali import fa_digits, jdate, jdate_short, tehran_ts
from notify.telegram import LIMIT, esc
from reports import advisor as adv
from reports import asset_report as ar
from reports import outlook as ol
from reports.facts import Facts, indicators_cfg
from reports.messages import ASSET_FA, DISCLAIMER, num

TARGET_CHARS = 3300
MAX_CHARS = LIMIT - 96
ICON = {"BTC": "₿", "Gold": "🥇"}
FACT_LINES = 3
EVENT_LINES = 2
RECENT_DAYS = 7
INDEX_KEYS = ("philly_fed_activity", "empire_state_activity")
# Persian name first, ticker in parentheses (a line must start with a Persian letter)
NAME_FA = {"cpi": "تورم مصرف‌کننده (CPI)", "core_cpi": "تورم هسته (Core CPI)", "ppi_final_demand": "تورم تولیدکننده (PPI)",
           "pce": "تورم مخارج مصرفی (PCE)", "core_pce": "تورم هسته‌ی مخارج مصرفی (Core PCE)",
           "nonfarm_payrolls": "اشتغال غیرکشاورزی (NFP)", "unemployment_rate": "نرخ بیکاری",
           "avg_hourly_earnings": "دستمزد ساعتی", "jolts_openings": "فرصت‌های شغلی خالی (JOLTS)",
           "initial_claims": "مدعیان بیمه‌ی بیکاری", "adp_private_payrolls": "اشتغال خصوصی (ADP)",
           "philly_fed_activity": "نظرسنجی تولیدی فیلادلفیا", "empire_state_activity": "نظرسنجی تولیدی نیویورک",
           "fomc": "تصمیم نرخ فدرال (FOMC)", "gdp": "رشد اقتصادی (GDP)"}
REL_KEY = {"fomc": "fomc", "gdp": "gdp", "philly": "philly_fed_activity", "empire": "empire_state_activity"}
GREG_FA = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]
SOURCE_SHORT = {"consensus": "اجماع", "system": "انتظار سیستم", "cleveland": "نوکست کلیولند", "gdpnow": "GDPNow",
                "kalshi": "بازار Kalshi", "fedwatch": "FedWatch"}
VERDICT_FA = {"not_priced": "بازار هنوز قیمت‌گذاری نکرده",
              "partly_priced": "بخشی از خبر قیمت شده",
              "heavily_priced": "به‌شدت پیشخور شده؛ ریسک واکنش معکوس بعد از انتشار بالاست"}
HOT_FA = {"unemployment_rate": ("بالاتر (بازار کار ضعیف‌تر)", "پایین‌تر"), "initial_claims": ("بالاتر (ضعف اشتغال)", "پایین‌تر")}
INFLATION_MOM = ("cpi", "core_cpi", "pce", "core_pce", "ppi_final_demand")
PCE_TARGET = 2.0


# ───────────────────────────── small formatters ─────────────────────────────

def name_of(key: str | None, release_id: str | None = None, fallback: str | None = None) -> str:
    return NAME_FA.get(key or "") or NAME_FA.get(REL_KEY.get(release_id or "", "")) or esc(fallback or key or release_id or "")


def fval(key: str | None, v: float | None) -> str:
    """A value in the series' own measure, for prose."""
    if v is None or key is None:
        return "—"
    if key in INDEX_KEYS:
        return num(v, "{:.1f}")
    if key == "adp_private_payrolls":
        return num(v, "{:,.0f}") + " هزار شغل"
    return adv.fmt_value(key, v)


def pnum(v: float, d: int = 1) -> str:
    """A positive magnitude: '۰٫۵۶' style through the same LTR-isolated formatter as every number."""
    return num(abs(v), "{:." + str(d) + "f}")


def when_fa(ts) -> str:
    """'چهارشنبه ۲۲ مهر ساعت ۱۶:۰۰' (Tehran)."""
    return f"{jdate(ts, year=False)} ساعت {fa_digits(f'{tehran_ts(ts):%H:%M}')}"


def hot_words(key: str | None) -> tuple[str, str]:
    if key in HOT_FA:
        return HOT_FA[key]
    if key in adv.INFLATION:
        return "داغ‌تر", "سردتر"
    if key in adv.LABOR:
        return "قوی‌تر", "ضعیف‌تر"
    return "بالاتر", "پایین‌تر"


# ───────────────────────────── real impact (models.release_impact) ─────────────────────────────

def impact_of(f: Facts, key: str | None, asset: str) -> dict | None:
    rel = ((f.impact or {}).get("releases") or {}).get(key or "")
    return ((rel or {}).get("assets") or {}).get(asset)


def impact_key(o: ol.EventOutlook) -> str | None:
    return "fomc" if o.release_id == "fomc" else o.key


def shock_clause(f: Facts, key: str | None, short: bool = False) -> str | None:
    """'تکان معمول: بیت‌کوین ±0.56٪ / طلا ±0.28٪ در ۱ ساعت (۲٫۰ و ۱٫۸ برابر عادی)'."""
    parts, ratios = [], []
    for a in ("BTC", "Gold"):
        s = ((impact_of(f, key, a) or {}).get("shock") or {}).get("1h") or {}
        if s.get("median_abs") is None:
            continue
        parts.append(f"{ASSET_FA[a]} ±{pnum(s['median_abs'], 2)}٪")
        if s.get("ratio") and s["ratio"] >= 1.8:  # only a clearly bigger-than-normal hour is worth saying
            ratios.append(pnum(s["ratio"], 1))
    if not parts:
        return None
    head = "معمولاً " if short else "تکان معمول: "
    return head + " / ".join(parts) + " در ساعت اول" + (f" ({' و '.join(ratios)} برابر یک ساعت عادی)" if ratios else "")


def direction_clause(f: Facts, key: str | None) -> str | None:
    """Whether the 1-hour direction follows the surprise, measured on big surprises."""
    best = None
    for a in ("BTC", "Gold"):
        d = (impact_of(f, key, a) or {}).get("direction") or {}
        if d.get("predictable"):
            if best is None or d["hit_rate"] > best[1]["hit_rate"]:
                best = (a, d)
    if best:
        a, d = best
        side = "هم‌جهت با منطق اقتصادی" if d.get("with_theory") else "خلاف منطق اقتصادی"
        return (f"جهت برای {ASSET_FA[a]} قابل پیش‌بینی است: در {adv.prob(d['hit_rate'])} از {fa_digits(d['n'])} "
                f"شگفتی بزرگ، {side}")
    if any(((impact_of(f, key, a) or {}).get("direction") or {}).get("hit_rate") is not None for a in ("BTC", "Gold")):
        return "جهت قابل پیش‌بینی نیست"
    return None


def fade_clause(f: Facts, key: str | None) -> str | None:
    rates = [(a, (impact_of(f, key, a) or {}).get("fade") or {}) for a in ("BTC", "Gold")]
    rates = [(a, x) for a, x in rates if x.get("reversal_rate") is not None]
    if not rates:
        return None
    a, x = max(rates, key=lambda t: t[1]["reversal_rate"])
    return f"حرکت ساعت اول {ASSET_FA[a]} در {adv.prob(x['reversal_rate'])} موارد تا ۲۴ ساعت برعکس شده"


def high_fade(f: Facts, key: str | None, rate: float = 0.4) -> bool:
    """Whether the first-hour move has often (≥ rate) reversed by 24 h for either asset."""
    return any((((impact_of(f, key, a) or {}).get("fade") or {}).get("reversal_rate") or 0) >= rate for a in ("BTC", "Gold"))


# ───────────────────────────── front-running (live) ─────────────────────────────

def frontrun(f: Facts, key: str | None, release: pd.Timestamp, now: pd.Timestamp) -> dict[str, dict]:
    """{asset: read} from models.release_impact.frontrun_read on the snapshot's 15-minute candles."""
    from models import release_impact as ri

    out = {}
    if release - now > pd.Timedelta(hours=48) or release <= now:
        return out
    for a, sym in (("BTC", "BTC"), ("Gold", "PAXG")):
        m = impact_of(f, key, a)
        p = f.root / "cache" / f"candles_{sym}_15m.csv.gz"
        if not m or not p.exists():
            continue
        bars = pd.read_csv(p, usecols=["ts", "close"])
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        bars = bars[bars["ts"] >= release - pd.Timedelta(hours=60)]
        if bars.empty:
            continue
        r = ri.frontrun_read(ri.Prices.from_bars(bars), m, release, now)
        if r:
            out[a] = r
    return out


def frontrun_verdict(reads: dict[str, dict]) -> str | None:
    """The strongest read in one clause: 'به‌شدت پیشخور شده… (بیت‌کوین +1.4٪ در ۱۸ ساعت، ۲٫۳ برابر معمول)'."""
    if not reads:
        return None
    a, r = max(reads.items(), key=lambda kv: kv[1]["ratio"])
    return (f"{VERDICT_FA[r['verdict']]} ({ASSET_FA[a]} {num(r['move_pct'], '{:+.2f}')}٪ در "
            f"{fa_digits(round(r['elapsed_h']))} ساعت، {pnum(r['ratio'], 1)} برابر معمول)")


def frontrun_history(reads: dict[str, dict]) -> str | None:
    """What followed a large pre-release drift in the past, with n."""
    if not reads:
        return None
    a, r = max(reads.items(), key=lambda kv: kv[1]["ratio"])
    ld = r.get("large") or {}
    if ld.get("reversal_rate") is None:
        return None
    rev = ld["reversal_rate"]
    what = "برگشته" if rev >= 0.5 else "ادامه داده"
    rate = rev if rev >= 0.5 else 1 - rev
    return (f"سابقه: بعد از پیشخور بزرگ، {ASSET_FA[a]} در {adv.prob(rate)} از {fa_digits(ld['n'])} مورد تا ۲۴ ساعت "
            f"{what} (میانه {num(ld['post_median_in_drift_direction'], '{:+.2f}')}٪ در جهت پیشخور)")


# ───────────────────────────── forecasts and odds ─────────────────────────────

def odds_phrase(o: ol.EventOutlook) -> str:
    hot, cool = hot_words(o.key)
    od = o.odds
    if od is None:
        return "احتمال قابل محاسبه نیست"
    if o.fedwatch:
        fw = o.fedwatch
        bits = [f"{w} {adv.prob(fw[k])}" for k, w in (("p_hold", "ثابت ماندن"), ("p_cut", "کاهش"), ("p_hike", "افزایش"))
                if fw.get(k)]
        return "بازار قیمت کرده: " + "، ".join(bits)
    if od.method == "even":
        return f"شانس {hot} و {cool} آمدن برابر است ({adv.prob(od.hot)} در برابر {adv.prob(od.cool)})؛ سیستم برتری ندارد"
    return f"احتمال {hot} از پیش‌بینی {adv.prob(od.hot)}، {cool} {adv.prob(od.cool)}"


def source_phrase(o: ol.EventOutlook) -> str:
    names = [ol.SOURCE_FA[s] for s in o.sources]
    if o.nowcast is not None and o.nowcast_source:
        names = [n if n != ol.SOURCE_FA[o.nowcast_source] else f"{n} {fval(o.key, o.nowcast)}" for n in names]
    return " + ".join(names)


def source_short(o: ol.EventOutlook) -> str:
    return "، ".join(SOURCE_SHORT[s] for s in o.sources)


# ───────────────────────────── 📊 final numbers in plain words ─────────────────────────────

def _measures(f: Facts, key: str) -> pd.Series:
    rc = {r["key"]: r for r in indicators_cfg()["releases"]}.get(key)
    if not rc:
        return pd.Series(dtype=float)
    return ar._measure(ar._series(f, key), rc["measure"]).dropna()


def meaning(key: str, actual: float, prev: float | None, forecast: float | None = None) -> str:
    """What the print says: against the published forecast, against the previous print, and (inflation) the
    annualised pace against the Fed's 2 % goal. Every number is computed here."""
    parts = []
    if forecast is not None:
        d = actual - forecast
        parts.append("برابر پیش‌بینی" if abs(d) < 1e-9 else "بیشتر از پیش‌بینی" if d > 0 else "کمتر از پیش‌بینی")
    if prev is not None:
        d = actual - prev
        parts.append("برابر ماه قبل" if abs(d) < 1e-9 else "بالاتر از قبلی" if d > 0 else "پایین‌تر از قبلی")
    if key in INFLATION_MOM and key != "ppi_final_demand":
        ann = ((1 + actual / 100) ** 12 - 1) * 100
        s = f"اگر هر ماه همین‌قدر ادامه یابد، تورم سالانه حدود {pnum(ann, 1)}٪ می‌شود"
        if ann > 0:
            k = ann / PCE_TARGET
            s += (f"، یعنی {pnum(k, 1)} برابر هدف ۲٪ فدرال" if k >= 1.2 else "، یعنی نزدیک هدف ۲٪ فدرال" if k >= 0.8
                  else "، یعنی زیر هدف ۲٪ فدرال")
        parts.append(s)
    cmp_ = [x for x in parts if not x.startswith("اگر")]
    ann_ = [x for x in parts if x.startswith("اگر")]
    return ".\n   ↳ ".join(x for x in ("، ".join(cmp_), *ann_) if x) or "بدون مقایسه با قبلی"


def print_sentence(key: str, actual: float, obs_month: str | None) -> str:
    month = f" {GREG_FA[pd.Timestamp(obs_month).month - 1]}" if obs_month else ""
    if key in INFLATION_MOM:
        what = "قیمت‌ها" if key != "ppi_final_demand" else "قیمت تولیدکننده"
        return f"{what} در ماه{month} {pnum(actual, 2)}٪ {'بالا رفت' if actual >= 0 else 'پایین آمد'}"
    if key == "nonfarm_payrolls":
        return f"در ماه{month} {pnum(actual, 0)} هزار شغل {'ساخته شد' if actual >= 0 else 'از دست رفت'}"
    if key == "adp_private_payrolls":
        return f"بخش خصوصی {pnum(actual, 0)} هزار شغل {'ساخت' if actual >= 0 else 'از دست داد'}"
    if key == "unemployment_rate":
        return f"نرخ بیکاری{month} {pnum(actual, 1)}٪ شد"
    if key == "avg_hourly_earnings":
        return f"دستمزد ساعتی در ماه{month} {pnum(actual, 2)}٪ رشد کرد"
    if key == "jolts_openings":
        return f"فرصت‌های شغلی خالی{month} {pnum(actual / 1000, 2)} میلیون بود"
    if key == "initial_claims":
        return f"درخواست بیمه‌ی بیکاری هفتگی {pnum(actual / 1000, 0)} هزار بود"
    return f"عدد منتشرشده {fval(key, actual)} بود"


def fact_line(f: Facts, key: str) -> str | None:
    """Actual, the published forecast (analyst consensus, when one was stored before the release) and the previous print."""
    s = (f.surprises or {}).get(key)
    if not s or s.get("actual") is None:
        return None
    a = s["actual"]
    fc = s.get("expected") if s.get("basis") == "vs consensus" else None
    ms = _measures(f, key)
    prev = float(ms.iloc[-2]) if len(ms) >= 2 else None
    nums = [f"منتشرشده {fval(key, a)}"]
    if fc is not None:
        nums.append(f"پیش‌بینی {fval(key, fc)}")
    if prev is not None:
        nums.append(f"قبلی {fval(key, prev)}")
    return (f"• <b>{name_of(key)}</b> ({jdate_short(s['release_utc'])})\n"
            f"   ↳ {print_sentence(key, a, s.get('obs_date'))}.\n"
            f"   ↳ {' · '.join(nums)}\n"
            f"   ↳ {meaning(key, a, prev, fc)}.")


def facts_section(f: Facts, now: pd.Timestamp, outs: list[ol.EventOutlook]) -> str | None:
    recent = sorted(((k, s) for k, s in (f.surprises or {}).items()
                     if now - pd.Timestamp(s["release_utc"]) <= pd.Timedelta(days=RECENT_DAYS)),
                    key=lambda kv: kv[1]["release_utc"], reverse=True)
    lines = [x for x in (fact_line(f, k) for k, _ in recent[:FACT_LINES]) if x]
    return "\n\n".join(["📊 <b>اعداد منتشرشده‌ی ۷ روز اخیر</b>", *lines]) if lines else None


# ───────────────────────────── 📅 ahead ─────────────────────────────

def market_line(f: Facts, key: str, pts: list) -> str | None:
    """What the Kalshi market prices for the print: where its median sits, and the odds of a print above the last one."""
    from models import forecast as fm

    med = fm.market_median(pts)
    if not med:
        return None
    lo, hi = med
    fine = hi - lo <= 0.1 + 1e-9
    a_, b_ = fval(key, lo), fval(key, hi)
    sa, sb = a_.split(" ", 1), b_.split(" ", 1)
    lo_s = sa[0] if len(sa) == 2 and len(sb) == 2 and sa[1] == sb[1] else a_
    where = f"حدود {b_}" if fine else f"بین {lo_s} و {b_}"
    out = f"بازار Kalshi برای {name_of(key)}: {where}"
    ms = _measures(f, key)
    if fine and len(ms):
        prev = round(float(ms.iloc[-1]), 1)
        at = [p for s_, p in pts if abs(s_ - prev) < 1e-6]
        if at:
            out += f"؛ احتمال بالاتر از قبلی ({fval(key, prev)}) {adv.prob(at[0])}"
    return out


def forecast_phrase(f: Facts, o: ol.EventOutlook) -> list[str]:
    """Published forecasts only: analyst consensus, the Fed nowcasts, market-priced numbers (Kalshi, FedWatch)."""
    if o.fedwatch:
        return [odds_phrase(o)]
    bits = []
    if o.forecast is not None and o.basis == "consensus":
        bits.append(f"پیش‌بینی تحلیلگران {fval(o.key, o.forecast)}")
    if o.nowcast is not None and o.nowcast_source:
        bits.append(f"{ol.SOURCE_FA[o.nowcast_source]} {fval(o.key, o.nowcast)}")
    if o.previous is not None:
        bits.append(f"قبلی {fval(o.key, o.previous)}")
    out = [" · ".join(bits)] if bits else []
    for k in sorted(o.markets, key=lambda k: k != o.key):
        m = market_line(f, k, o.markets[k])
        if m:
            out.append(m)
    return out or ["پیش‌بینی منتشرشده‌ای هنوز در دسترس نیست"]


def event_line(f: Facts, o: ol.EventOutlook, now: pd.Timestamp, skip_impact: bool = False) -> str:
    """When the release is due, its published forecasts, its measured impact and the live front-running read."""
    k = impact_key(o)
    lines = [f"• <b>{name_of(o.key, o.release_id, o.name_fa)}</b> {when_fa(o.when)}",
             *[f"   ↳ {x}" for x in forecast_phrase(f, o)]]
    for c in (() if skip_impact else (shock_clause(f, k, short=True), direction_clause(f, k))):
        if c:
            lines.append(f"   ↳ {c}")
    fr = frontrun_verdict(frontrun(f, k, o.when, now))
    if fr:
        lines.append(f"   ↳ پیشخور: {fr}")
    return "\n".join(lines)


def ahead_section(f: Facts, now: pd.Timestamp, outs: list[ol.EventOutlook], key: ol.EventOutlook | None = None) -> str:
    imp = [o for o in outs if o.importance >= 4][:EVENT_LINES]
    if not imp:
        return "📅 <b>پیش رو</b>\nانتشار مهمی در ۷ روز آینده نیست."
    return "\n\n".join(["📅 <b>پیش رو</b>", *[event_line(f, o, now, skip_impact=o is key) for o in imp]])


def key_event(f: Facts, outs: list[ol.EventOutlook]) -> ol.EventOutlook | None:
    """The coming release that matters most: importance, then the measured 1-hour shock on Bitcoin."""
    if not outs:
        return None

    def shock(o):
        return (((impact_of(f, impact_key(o), "BTC") or {}).get("shock") or {}).get("1h") or {}).get("ratio") or 0
    return max(outs, key=lambda o: (o.importance, shock(o), -o.when.value))


# ───────────────────────────── 🎯 decision and assets ─────────────────────────────

def regime_of(f: Facts, asset: str) -> dict | None:
    r = (((f.regime or {}).get("assets") or {}).get(asset) or {})
    return r if r.get("state") else None


def decision_line(v: dict, asset: str, reg: dict | None = None) -> str:
    """The action from the long-horizon trend filter and volatility-scaled size (models.trend_regime); the invalidation is
    the 200-day average. Without it (data missing) no decision is offered."""
    if not reg:
        return f"• <b>{ASSET_FA[asset]}</b>: داده‌ی روند بلندمدت در دسترس نیست؛ تصمیمی پیشنهاد نمی‌شود."
    st = reg["state"]
    sma = adv.px(st["sma200"], asset)
    days = f"، {fa_digits(st['days_in_state'])} روز است" if st.get("days_in_state") else ""
    ctx = []
    if st.get("mom12m_pct") is not None:
        ctx.append(f"بازده ۱۲ ماه گذشته {adv.pct(st['mom12m_pct'])}")
    if st.get("vol60_pct") is not None:
        ctx.append(f"نوسان سالانه {pnum(st['vol60_pct'], 0)}٪")
    ctx_s = f"\n   ↳ {'، '.join(ctx)}." if ctx else ""
    if st["up"]:
        return (f"• <b>{ASSET_FA[asset]}</b>: روند بلندمدت صعودی است.\n"
                f"   ↳ قیمت {pnum(st['dist_pct'], 1)}٪ بالای میانگین ۲۰۰ روزه{days}.\n"
                f"   ↳ سهم مجاز تا {adv.prob(st['exposure'])} از پولی که برای این دارایی کنار گذاشته‌ای.\n"
                f"   ↳ خروج با بسته شدن روزانه زیر {sma}.{ctx_s}")
    return (f"• <b>{ASSET_FA[asset]}</b>: روند بلندمدت نزولی است.\n"
            f"   ↳ قیمت {pnum(abs(st['dist_pct']), 1)}٪ زیر میانگین ۲۰۰ روزه{days}.\n"
            f"   ↳ فعلاً خرید نکن؛ سهم مجاز صفر.\n"
            f"   ↳ ورود فقط با بسته شدن روزانه بالای {sma}.{ctx_s}")


def asset_sentences(v: dict) -> str:
    """Two sentences: the trend as it bears on the decision, then whether macro weighs in."""
    t = (v["x"].get("technical") or {})
    d1, h4 = adv._dir((t.get("1d") or {}).get("trend")), adv._dir((t.get("4h") or {}).get("trend"))
    ev1 = (t.get("1d") or {}).get("last_event")
    if d1 > 0 and h4 < 0:
        a = "روند روزانه صعودی است ولی نمودار ۴ ساعته در اصلاح است"
    elif d1 < 0 and h4 > 0:
        a = "روند روزانه نزولی است و جهش ۴ ساعته فعلاً فقط یک برگشت کوتاه است"
    elif d1 > 0:
        a = "روند روزانه و ۴ ساعته هر دو صعودی‌اند"
    elif d1 < 0:
        a = "روند روزانه و ۴ ساعته هر دو نزولی‌اند"
    else:
        a = "روند روشنی ندارد و در محدوده نوسان می‌کند"
    if d1 > 0 and ev1 in ("CHOCH_DN", "BOS_DN"):
        a += "، و کف روزانه شکسته شده که اولین ترک روند است"
    elif d1 < 0 and ev1 in ("CHOCH_UP", "BOS_UP"):
        a += "، ولی سقف روزانه شکسته شده که اولین نشانه‌ی برگشت است"
    sg, drift = v["sg"], v["drift"]
    if sg and abs(drift) >= 0.3 * sg:
        b = (f"کلان {'به نفع' if drift > 0 else 'علیه'} {v['name']} است ({num(drift, '{:+.1f}')}٪ در ۴ هفته در برابر "
             f"نوسان عادی ±{pnum(sg, 1)}٪)")
    else:
        b = ""  # macro without weight is left out: it changes no decision
    return f"کوتاه‌مدت: {a}.\n{b}." if b else f"کوتاه‌مدت: {a}."


def levels_line(v: dict, asset: str) -> str:
    s = adv.px(v["s"], asset) + (f" ({v['s_how']})" if v.get("s_how") else "")
    r = adv.px(v["r"], asset) + (f" ({v['r_how']})" if v.get("r_how") else "")
    out = f"حمایت {s} · مقاومت {r}"
    sc = adv.scenarios(v["price"], v["r"], v["s"], 0.0, v["sg"]) if v.get("sg") and v.get("price") else None  # no macro drift
    if sc:
        out += f"\nشانس تا ۴ هفته (از نوسان ۹۰ روزه): بالای مقاومت {adv.prob(sc['up'])}، زیر حمایت {adv.prob(sc['down'])}"
    return out


def asset_section(v: dict, asset: str, read: str) -> str:
    mv = v["mv"] or {}
    head = f"{ICON[asset]} <b>{v['name']}</b> {adv.px(v['price'], asset)} دلار"
    if mv.get("chg24") is not None:
        head += f" (۲۴ ساعت {adv.pct(mv['chg24'])} / ۷ روز {adv.pct(mv.get('chg7d'))})"
    body = [ln for ln in (read + "\n" + levels_line(v, asset)).split("\n") if ln.strip()]
    return "\n".join([head, *[f"   ↳ {ln}" for ln in body]])


# ───────────────────────────── post-release ─────────────────────────────

def realised_move(f: Facts, key: str | None, release: pd.Timestamp, now: pd.Timestamp) -> str | None:
    """'بعد از انتشار: بیت‌کوین −1.8٪ در ۱ ساعت، ۲٫۶ برابر معمول' (move since the release, up to 1 hour after)."""
    from models import release_impact as ri

    parts = []
    end = min(now, release + pd.Timedelta(hours=1))
    hours = (end - release) / pd.Timedelta(hours=1)
    if hours < 0.25:
        return None
    for a, sym in (("BTC", "BTC"), ("Gold", "PAXG")):
        p = f.root / "cache" / f"candles_{sym}_15m.csv.gz"
        if not p.exists():
            continue
        b = pd.read_csv(p, usecols=["ts", "close"])
        b["ts"] = pd.to_datetime(b["ts"], utc=True)
        b = b[(b["ts"] >= release - pd.Timedelta(hours=2)) & (b["ts"] <= end)]
        if b.empty:
            continue
        mv = float(ri.Prices.from_bars(b).ret(np.array([release.as_unit("ns").value]), hours)[0])
        if not np.isfinite(mv):
            continue
        usual = (((impact_of(f, key, a) or {}).get("shock") or {}).get("1h") or {}).get("median_abs")
        parts.append(f"{ASSET_FA[a]} {num(mv, '{:+.2f}')}٪" + (f" ({pnum(abs(mv) / usual, 1)} برابر تکان معمول)" if usual else ""))
    if not parts:
        return None
    span = "در ساعت اول" if hours >= 1 else f"در {fa_digits(f'{hours * 60:.0f}')} دقیقه"
    return f"بعد از انتشار {span}: " + "، ".join(parts)


def released_section(f: Facts, now: pd.Timestamp) -> str | None:
    rel = [r for r in f.releases if now - pd.Timestamp(r["release_utc"]) <= pd.Timedelta(hours=12)]
    if not rel:
        return None
    L = ["🔔 <b>چه منتشر شد</b>"]
    for key in dict.fromkeys(r["indicator"] for r in rel):
        line = fact_line(f, key)
        if line:
            L.append(line)
    t = pd.Timestamp(rel[0]["release_utc"])
    key = rel[0]["indicator"]
    mv = realised_move(f, key, t, now)
    if mv:
        L.append(mv)
    fd = fade_clause(f, key)
    if fd:
        L.append(f"در گذشته {fd}.")
    return "\n".join(L)


# ───────────────────────────── polish, split, checks ─────────────────────────────

PARA_SYSTEM = """You are a senior macro and markets analyst. Rewrite the Persian (Farsi) TEXT you receive so it reads like
an experienced analyst talking to one client: at most two short connected sentences. Use ONLY numbers that appear in
the text, with the same values; never add, derive or round a number. Plain Persian text, no Markdown or HTML, no new
advice, start with a Persian word."""


def polish(text: str) -> str:
    """Rephrase with the free model when LLM_API_KEY is set and the rewrite keeps to the text's numbers."""
    if not os.environ.get("LLM_API_KEY"):
        return text
    from llm import free_writer
    from llm.claude import LLMError
    from reports.writer import unknown_numbers

    draft = adv._ISOLATES.sub("", adv.plain(text))
    try:
        out, _ = free_writer.chat(PARA_SYSTEM, draft, max_tokens=600)
    except LLMError as e:
        print(f"::warning::daily rewrite unavailable ({str(e)[:200]})")
        return text
    out = re.sub(r"\s+", " ", re.sub(r"<[^>]+>|```\w*", "", out)).strip()
    if (not out or unknown_numbers(out, {"draft": draft}) or len(out) > 2 * len(draft) + 100
            or len(out) < len(draft) // 3 or not starts_rtl(out)):
        print("::warning::daily rewrite rejected (new numbers, bad length or direction); using the template")
        return text
    return re.sub(r"[-+−]?\d[\d,]*(?:\.\d+)?", lambda m: "⁦" + m.group(0) + "⁩", esc(out))


_ISO = re.compile("[⁦⁧⁨][^⁩]*⁩")


def starts_rtl(line: str) -> bool:
    """The first strongly directional character (outside tags and isolates) is right-to-left, so Telegram aligns the
    line right. A line without any strong character also passes when it starts with an RLM."""
    import unicodedata

    s = _ISO.sub("", re.sub(r"<[^>]+>", "", line))
    for ch in s:
        if ch == "‏":
            return True
        b = unicodedata.bidirectional(ch)
        if b in ("R", "AL"):
            return True
        if b == "L":
            return False
    return not s.strip()


def text_lines(msg: str) -> list[str]:
    """Message lines outside <pre> blocks."""
    return [ln for ln in re.sub(r"<pre>.*?</pre>", "", msg, flags=re.S).split("\n") if ln.strip()]


def split(sections: list[str], limit: int = MAX_CHARS) -> list[str]:
    """Pack sections into as few messages as possible; later parts open with "بخش ۲ از ۲" (no repeated header)."""
    msgs, cur = [], ""
    for s in sections:
        if not s:
            continue
        cand = f"{cur}\n\n{s}" if cur else s
        if len(cand) <= limit:
            cur = cand
        else:
            if cur:
                msgs.append(cur)
            cur = s
    if cur:
        msgs.append(cur)
    if len(msgs) > 1:
        n = fa_digits(str(len(msgs)))
        title, nl, body = msgs[0].partition("\n")
        msgs[0] = f"{title} — بخش ۱ از {n}{nl}{body}"
        msgs[1:] = [f"بخش {fa_digits(str(i))} از {n}\n{m}" for i, m in enumerate(msgs[1:], 2)]
    return msgs


# ───────────────────────────── build ─────────────────────────────

def build(f: Facts, now: pd.Timestamp, changes: list[str] | None = None, trigger: str | None = None,
          rewrite: bool = True) -> list[str]:
    """The report as one Telegram message on a normal day (more only when a release day needs it)."""
    outs = ol.upcoming(f, now)
    views = {a: adv.asset_view(f, a, now) for a in ("BTC", "Gold") if f.asset(a)}
    post = changes is not None
    title = (f"🔔 <b>گزارش بعد از انتشار</b> — {jdate(now)}" if post else f"☀️ <b>گزارش روزانه</b> — {jdate(now)}")
    sections = [title]
    if post:
        rs = released_section(f, now)
        if rs:
            sections.append(rs)
    dec = ["🎯 <b>تصمیم امروز</b>", *[decision_line(v, a, regime_of(f, a)) for a, v in views.items()]]
    ke = key_event(f, outs)
    if ke:
        clause = shock_clause(f, impact_key(ke), short=True)
        dirc = direction_clause(f, impact_key(ke)) or fade_clause(f, impact_key(ke))
        bits = "، ".join(x for x in (clause, dirc) if x)
        dec.append(f"• <b>رویداد کلیدی</b>: {name_of(ke.key, ke.release_id, ke.name_fa)} {when_fa(ke.when)}"
                   + (f"\n   ↳ {bits}." if bits else ""))
    sections.append("\n\n".join(dec))
    fs = facts_section(f, now, outs)
    ahead = ahead_section(f, now, outs, ke)
    assets = [asset_section(v, a, polish(asset_sentences(v)) if rewrite else asset_sentences(v)) for a, v in views.items()]
    tail = DISCLAIMER
    body = [fs, ahead, *assets, tail]
    # keep a normal day to one short message: drop the oldest fact items first, then the second event
    msg = "\n\n".join(x for x in sections + body if x)
    while not post and len(msg) > TARGET_CHARS and fs and fs.count("\n\n") > 1:
        fs = fs.rsplit("\n\n", 1)[0]
        body[0] = fs
        msg = "\n\n".join(x for x in sections + body if x)
    while not post and len(msg) > TARGET_CHARS and ahead.count("\n\n") > 1:
        ahead = ahead.rsplit("\n\n", 1)[0]
        body[1] = ahead
        msg = "\n\n".join(x for x in sections + body if x)
    return split(sections + [x for x in body if x])
