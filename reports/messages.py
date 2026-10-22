"""Persian Telegram messages (HTML parse mode), built only from `reports.facts.Facts`.

Every function returns ready-to-send HTML. Dynamic text is escaped; numbers and dates are wrapped in Unicode
isolates (core.fa.ltr) so signs and digit order survive right-to-left rendering in Telegram.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.fa import fa, label, ltr, word
from core.jalali import fa_digits, jdate, jdatetime_fa
from core.timeutil import TEHRAN
from notify.telegram import esc
from reports.facts import MEASURE_FA, Facts, channel_fa, indicators_cfg, ind_name, releases_cfg

DISCLAIMER = "⚠️ این پیام صرفاً تحلیلی و آموزشی است و توصیه مالی یا پیشنهاد خرید و فروش نیست."
ASSET_FA = {"BTC": "بیت‌کوین", "Gold": "طلا"}
HORIZON_FA = {"short": "کوتاه‌مدت (۲۴ ساعت)", "medium": "میان‌مدت (۴ هفته)", "long": "بلندمدت (۶ ماه)"}


# ───────────────────────────── helpers ─────────────────────────────

def num(v, fmt: str = "{:,.2f}", sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return ltr((fmt.replace("{:", "{:+", 1) if sign else fmt).format(v))


def score(v) -> str:
    return "—" if v is None else ltr(f"{v:+.0f}")


def tehran(ts) -> str:
    """Every date-time in every message: Persian calendar, Tehran time ('جمعه ۱۰ مهر ۱۴۰۵، ساعت ۰۱:۰۲')."""
    return jdatetime_fa(ts)


def direction_fa(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or abs(v) < 0.5:
        return "اثر معنادار تاریخی ندارد"
    return "صعودی 🟢" if v > 0 else "نزولی 🔴"


def proxy_fa(p: str) -> str:
    return "PAXG — توکن طلا (هر توکن = یک اونس)، نماینده بلادرنگ XAUUSD" if str(p).startswith("PAXG") else str(p)


def dashboard_link() -> str:
    url = os.environ.get("DASHBOARD_URL")
    return f'\n\n🔗 <a href="{esc(url)}">داشبورد کامل</a>' if url else ""


def footer() -> str:
    return dashboard_link() + "\n\n" + DISCLAIMER


def chain(channels: list[str]) -> str:
    ch = channel_fa(indicators_cfg())
    return " ← ".join(esc(ch.get(c, c)) for c in channels)


def prev_measure(f: Facts, key: str, measure: str) -> tuple[float, str] | None:
    """The last published value in the release's own measure (e.g. CPI m/m %), from the snapshot."""
    obs = f._obs()
    s = obs[obs["series_key"] == key].sort_values("date").set_index("date")["value"]
    if len(s) < 2:
        return None
    if measure == "mom_pct":
        v = (s.iloc[-1] / s.iloc[-2] - 1) * 100
    elif measure == "qoq_ann":
        v = ((s.iloc[-1] / s.iloc[-2]) ** 4 - 1) * 100
    elif measure in ("diff", "mom_change"):
        v = s.iloc[-1] - s.iloc[-2]
    else:
        v = s.iloc[-1]
    return float(v), s.index[-1]


# ───────────────────────────── 1. heads-up (one day before) ─────────────────────────────

def events_on(f: Facts, day_tehran) -> pd.DataFrame:
    c = f.calendar
    if c.empty:
        return c
    d = c.assign(local=c["scheduled_utc"].dt.tz_convert(TEHRAN))
    return d[d["local"].dt.date == day_tehran].drop_duplicates("event_id").sort_values("scheduled_utc")


def forecast_lines(f: Facts, e) -> list[str]:
    """The forecast and its source: the consensus when one was collected, else the system's own expectation; plus the
    odds of a hotter / cooler print (reports.outlook)."""
    from reports import daily, outlook

    try:
        o = outlook.event_outlook(f, e)
    except Exception as exc:  # noqa: BLE001 — a forecast problem must never block the alert
        print(f"::warning::forecast for {getattr(e, 'event_id', '?')} unavailable: {exc}")
        o = None
    if o is None or (o.forecast is None and not o.fedwatch):
        return ["📊 پیش‌بینی: اجماع رایگان در دسترس نیست و انتظار سیستم هم قابل محاسبه نبود"]
    if o.fedwatch:
        return [f"📊 {daily.odds_phrase(o)} (منبع: {outlook.SOURCE_FA['fedwatch']})"]
    name = esc(ind_name(o.key))
    head = (f"📊 اجماع {name}: {daily.fval(o.key, o.forecast)}" if o.basis == "consensus" else
            f"📊 انتظار سیستم برای {name}: {daily.fval(o.key, o.forecast)} (اجماع رایگان در دسترس نیست؛ میانگین دوره‌های قبل)")
    return [head, f"🎲 {daily.odds_phrase(o)}", f"   منبع: {daily.source_phrase(o)}"]


def _event_block(f: Facts, e, pre: list[dict]) -> list[str]:
    """Name, time, previous value, effect channel, what a surprise would mean per asset, front-running read."""
    rel = releases_cfg()
    rcfg = {r["key"]: r for r in indicators_cfg()["releases"]}
    r = rel.get(e.release_id, {})
    L = ["", f"<b>{esc(r.get('name_fa', e.release_id))}</b> {'★' * int(r.get('importance', 1))}",
         f"⏰ زمان: {tehran(e.scheduled_utc)} (به وقت تهران)", *forecast_lines(f, e)]
    for key in r.get("series_keys", [])[:2]:
        cfg = rcfg.get(key)
        measure = cfg["measure"] if cfg else "level"
        pv = prev_measure(f, key, measure)
        if pv:
            L.append(f"↩️ قبلی {esc(ind_name(key))}: {num(pv[0])} ({esc(MEASURE_FA.get(measure, measure))})")
        if not cfg:
            continue
        L.append(f"🔗 کانال اثر: {chain(cfg.get('channels', []))}")
        for a in ("BTC", "Gold"):
            c = f.coef(key, a, "24h") or {}
            k = c.get("coefficient")
            theory = cfg.get("theory", {}).get(a, 0)
            hot = direction_fa(k) if k is not None else ("صعودی 🟢" if theory > 0 else "نزولی 🔴" if theory < 0 else "نامشخص")
            cold = direction_fa(-k) if k is not None else ("نزولی 🔴" if theory > 0 else "صعودی 🟢" if theory < 0 else "نامشخص")
            basis = (f"ضریب اثر ۲۴ ساعته {num(k, '{:.1f}', sign=True)} از ±۱۰" if k is not None
                     else "بر اساس نظریه؛ ضریب تاریخی ندارد")
            L.append(f"   • {ASSET_FA[a]}: بالاتر از انتظار ← {hot}؛ پایین‌تر ← {cold} ({basis})")
    line = prepos_line(pre, e.event_id)
    if line:
        L.append(line)
    return L


def headsup(f: Facts, now: pd.Timestamp) -> str | None:
    tomorrow = (now.tz_convert(TEHRAN) + pd.Timedelta(days=1)).date()
    ev = events_on(f, tomorrow)
    if ev.empty:
        return None
    pre = f.prepos.get("live", []) if getattr(f, "prepos", None) else []
    L = [f"📅 <b>هشدار یک روز قبل — انتشارهای فردا ({jdate(str(tomorrow))})</b>"]
    for e in ev.itertuples():
        L += _event_block(f, e, pre)
    L.append("\n«اجماع» = میانگین پیش‌بینی تحلیلگران (ForexFactory)؛ اگر نباشد «انتظار سیستم» = میانگین دوره‌های قبل. "
             "ضریب‌ها همبستگی تاریخی‌اند، نه پیش‌بینی.")
    return "\n".join(L) + footer()


def _main_outlook(f: Facts, events: list[dict], t: pd.Timestamp):
    """The release that decides the alert (highest importance, then the larger measured shock) and its outlook."""
    from types import SimpleNamespace

    from reports import daily, outlook

    outs = []
    for e in events:
        try:
            outs.append(outlook.event_outlook(f, SimpleNamespace(**{**e, "scheduled_utc": t})))
        except Exception as exc:  # noqa: BLE001 — a forecast problem must never block the alert
            print(f"::warning::outlook for {e.get('event_id')} unavailable: {exc}")
    return daily.key_event(f, outs) if outs else None


def context_line(f: Facts) -> str | None:
    """Fed funds futures drift and order-book imbalance from the hourly pre-positioning check (prepos.json)."""
    ctx = (getattr(f, "prepos", None) or {}).get("context") or {}
    bits = []
    if ctx.get("zq_drift_24h_bp") is not None:
        bits.append(f"نرخ ضمنی قراردادهای آتی فدرال {num(ctx['zq_drift_24h_bp'], '{:+.1f}')} واحد پایه در ۲۴ ساعت")
    for a in ("BTC", "Gold"):
        imb = (ctx.get(f"book_{a}") or {}).get("imbalance")
        if imb is not None:
            bits.append(f"عمق سفارش {ASSET_FA[a]} {num(imb, '{:+.2f}')}")
    return ("بازار: " + "، ".join(bits) + " (عمق مثبت = خریدار بیشتر)") if bits else None


def what_to_do(f: Facts, o, reads: dict) -> str:
    from reports import daily

    k = daily.impact_key(o)
    heavy = any(r["verdict"] == "heavily_priced" for r in reads.values())
    tiers = {(daily.impact_of(f, k, a) or {}).get("tier") for a in ("BTC", "Gold")}
    if heavy:
        return "چه کنم: حجم را کم کن و دنبال حرکت پیش از انتشار نرو؛ بخش بزرگی از خبر قیمت شده."
    if daily.high_fade(f, k):
        return "چه کنم: حجم را کم کن؛ در ۱۵ دقیقه‌ی اول وارد نشو، حرکت اول اغلب برمی‌گردد."
    if "volatile_undirected" in tiers:
        return "چه کنم: حجم را کم کن؛ در ۱۵ دقیقه‌ی اول وارد نشو، تکان بزرگ است ولی جهتش قابل پیش‌بینی نیست."
    if "volatile_directional" in tiers:
        hot, _ = daily.hot_words(o.key)
        return f"چه کنم: جهت این انتشار معمولاً از عدد پیروی می‌کند؛ اگر {hot} آمد، با حجم کوچک و حد ضرر نزدیک."
    return "چه کنم: تکان این انتشار معمولاً کوچک است؛ نیازی به تغییر موقعیت نیست."


def prerelease(f: Facts, now: pd.Timestamp, events: list[dict]) -> str | None:
    """Alert shortly before an important release, at most 8 lines: forecast and odds, the measured shock and direction,
    the live front-running read with its history, and what to do. Risk information, not a trade signal."""
    if not events:
        return None
    from reports import daily

    t = pd.Timestamp(events[0]["scheduled_utc"])
    mins = max(0, int(round((t - now).total_seconds() / 60)))
    o = _main_outlook(f, events, t)
    names = "، ".join(daily.name_of(None, e["release_id"], e.get("name_fa")) for e in events)
    L = [f"⏰ <b>هشدار پیش از انتشار — {names}</b>", f"حدود {fa_digits(str(mins))} دقیقه‌ی دیگر، {daily.when_fa(t)}."]
    reads = {}
    if o is not None:
        k = daily.impact_key(o)
        if o.fedwatch:
            L.append(f"{daily.odds_phrase(o)} (FedWatch).")
        elif o.forecast is not None:
            lab = "پیش‌بینی" if o.basis == "consensus" else "انتظار سیستم"
            L.append(f"{lab} {daily.name_of(o.key, o.release_id)}: {daily.fval(o.key, o.forecast)}؛ {daily.odds_phrase(o)} "
                     f"({daily.source_short(o)}).")
        imp = "، ".join(x for x in (daily.shock_clause(f, k), daily.direction_clause(f, k)) if x)
        if imp:
            L.append(imp + ".")
        reads = daily.frontrun(f, k, t, now)
        fr = daily.frontrun_verdict(reads)
        if fr:
            hist = daily.frontrun_history(reads)
            L.append(f"پیشخور: {fr}." + (f" {hist}." if hist else ""))
        L.append(what_to_do(f, o, reads))
    ctx = context_line(f)
    if ctx and len(L) < 7:
        L.append(ctx)
    L.append(DISCLAIMER)
    return "\n".join(L[:8])


def frontrun_alert(f: Facts, now: pd.Timestamp, event: dict, reads: dict) -> str:
    """Sent once when the pre-release drift passes 2× its usual size 6–24 h before an importance-5 release."""
    from reports import daily

    t = pd.Timestamp(event["scheduled_utc"])
    L = [f"⚡ <b>هشدار پیشخور — {daily.name_of(None, event['release_id'], event.get('name_fa'))}</b>",
         f"انتشار: {daily.when_fa(t)}."]
    for a, r in reads.items():
        L.append(f"{ASSET_FA[a]}: {num(r['move_pct'], '{:+.2f}')}٪ از {fa_digits(r['window_h'])} ساعت پیش از انتشار تاکنون، "
                 f"{daily.pnum(r['ratio'], 1)} برابر پیشخور معمول این انتشار.")
    hist = daily.frontrun_history(reads)
    if hist:
        L.append(hist + ".")
    ctx = context_line(f)
    if ctx:
        L.append(ctx)
    L.append("این اطلاعات ریسک است، نه سیگنال معامله؛ هیچ قاعده‌ی پیش‌موقعیت‌گیری برتری اثبات‌شده ندارد. "
             "حجم را کم کن و دنبال حرکت نرو.")
    L.append(DISCLAIMER)
    return "\n".join(L)


# ───────────────────────────── signal card (used by the per-asset reports) ─────────────────────────────

def signal_card(s: dict, asset: str, blocked: bool) -> list[str]:
    fmt = (lambda v: ltr(f"{v:,.0f}")) if asset == "BTC" else (lambda v: ltr(f"{v:,.2f}"))  # noqa: E731
    head = "👀 ستاپ در فهرست نظارت (سیگنال صادر نشد)" if blocked else "🎯 <b>سیگنال فعال</b>"
    tps = "، ".join(f"هدف {i}: {fmt(t)} (R:R {ltr(f'{r:.2f}')})" for i, (t, r) in enumerate(zip(s["targets"], s["rr"]), 1))
    wr = "—" if s.get("win_rate") is None else ltr(f"{s['win_rate'] * 100:.0f}٪")
    L = [head, f"جهت: <b>{word(s['direction'])}</b> · افق: چند روز تا ۲ هفته",
         f"ورود: {fmt(min(s['entry_zone']))} تا {fmt(max(s['entry_zone']))} · حد ضرر: {fmt(s['stop_loss'])}", tps,
         f"نرخ برد بک‌تست: {wr} در {ltr(s['n_backtest'])} معامله · میانگین R {num(s.get('avg_r'), '{:.2f}', sign=True)} · "
         f"اطمینان: {word(s['confidence'])}" + (" (نمونه کوچک)" if s.get("small_sample") else ""),
         f"ابطال: {esc(fa(s['invalidation']))}",
         "دلایل: " + "؛ ".join(esc(fa(x)) for x in s.get("reasons_technical", []))]
    if blocked and s.get("blocked_by"):
        L.append("دلیل مسدود شدن: " + "؛ ".join(esc(fa(x)) for x in s["blocked_by"]))
    return L


# ───────────────────────────── health warning / news alert ─────────────────────────────

def health_warning(problems: list[dict], context: str) -> str:
    L = [f"🚨 <b>هشدار سلامت داده</b> — {esc(context)}",
         "گزارش این نوبت ارسال نشد تا تحلیل بر پایه داده ناقص یا قدیمی ساخته نشود.", ""]
    for p in problems[:20]:
        L.append(f"• {esc(p['name'])}: {esc(p['problem'])}")
    if len(problems) > 20:
        L.append(f"… و {ltr(len(problems) - 20)} مورد دیگر")
    return "\n".join(L) + dashboard_link()


def news_alert(item: dict) -> str:
    eff = {"bullish": "صعودی 🟢", "bearish": "نزولی 🔴", "neutral": "خنثی", "unrelated": "بی‌ارتباط"}
    ch = channel_fa(indicators_cfg())
    chans = [c for c in str(item.get("channels") or "").split("|") if c]
    L = [f"📰 <b>خبر مهم</b> ({'★' * int(item['importance'])})", esc(item.get("title_fa") or item.get("title")),
         f"منبع: {esc(label(item.get('source', '')))} · {tehran(item['published_utc'])}",
         f"بیت‌کوین: {eff.get(item.get('btc'), '—')} · طلا: {eff.get(item.get('gold'), '—')}"]
    if chans:
        L.append("کانال: " + "، ".join(esc(ch.get(c, c)) for c in chans))
    if item.get("url"):
        L.append(f'<a href="{esc(item["url"])}">متن خبر</a>')
    L.append("\nطبقه‌بندی خودکار با Claude از روی تیتر؛ پیش از هر تصمیمی متن کامل را بخوانید.")
    return "\n".join(L) + footer()


# ───────────────────────────── pre-positioning (front-run) alert ─────────────────────────────

FEATURE_FA = {"drift": "رانش قیمت نسبت به نوسان عادی همان ساعت‌ها", "steady": "یک‌طرفه و کم‌نوسان بودن حرکت (آماره t)",
              "flow": "عدم‌توازن سفارش‌های تهاجمی (taker)", "rate": "تغییر بازده ۲ ساله (انتظار مسیر نرخ)",
              "oi": "تغییر قراردادهای باز هم‌جهت با قیمت", "funding": "تغییر نرخ تأمین مالی (funding)",
              "lead": "شگفتی داده‌ی پیشرو (ADP / CPI / PPI)"}
PREPOS_STATUS_FA = {"signal": "سیگنال فعال", "watchlist": "مسدود (برتری اثبات‌نشده)", "quiet": "بدون نشانه",
                    "waiting": "پنجره هنوز باز نشده", "no_model": "تاریخچه کافی نیست", "no_data": "داده‌ی اخیر نیست"}


def prepos_alert(row: dict, bt: dict | None = None) -> str:
    """A pre-positioning signal that passed the edge gate — sent before the release."""
    a = row["asset"]
    fmt = "{:,.0f}" if a == "BTC" else "{:,.2f}"
    d = row["direction"]
    L = [f"⏱ <b>سیگنال پیش‌موقعیت‌گیری (پیش از انتشار)</b> — {esc(row.get('name_fa') or row['release'])} · {ASSET_FA[a]}",
         f"انتشار: {tehran(row['release_utc'])} (تهران) · پنجره‌ی پیش‌رو: {ltr(row['window_h'])} ساعت",
         f"بازار پیش از انتشار موقعیت گرفته است: جهت <b>{word(d)}</b> · امتیاز {num(row['trigger_score'], '{:.2f}', sign=True)} "
         f"(آستانه ±{ltr('1.5')}) از {tehran(row['triggered_at'])}", "", "<b>آنچه سیگنال را ساخت</b>"]
    for k, v in sorted((row.get("components") or {}).items(), key=lambda kv: -abs(kv[1])):
        L.append(f"• {FEATURE_FA.get(k, k)}: {num(v, '{:.2f}', sign=True)}")
    if row.get("entry") is not None:
        tps = "، ".join(f"هدف {i}: {num(t, fmt)} (R:R {ltr(f'{r:.2f}')})" for i, (t, r) in enumerate(zip(row["targets"], row["rr"]), 1))
        L += ["", f"ورود: {num(row['entry'], fmt)} (بازار) · حد ضرر: {num(row['stop_loss'], fmt)}", tps,
              f"خروج حداکثر: {tehran(row['exit_by'])} (۲۴ ساعت پس از انتشار)"]
    s = row.get("backtest") or (bt or {}).get("selected") or {}
    if s:
        wr = "—" if s.get("win_rate") is None else ltr(f"{s['win_rate'] * 100:.0f}٪")
        L.append(f"بک‌تست خارج از نمونه (پنجره انتخاب‌شده از گذشته): {ltr(s.get('n_trades', 0))} معامله، نرخ برد {wr}، "
                 f"میانگین R {num(s.get('avg_r'), '{:.2f}', sign=True)}، p = {num(s.get('p_r'), '{:.3f}')}")
    ctx = []
    if row.get("zq_drift_bp") is not None:
        ctx.append(f"تغییر نرخ ضمنی آتی وجوه فدرال از باز شدن پنجره: {num(row['zq_drift_bp'], '{:.1f}', sign=True)} واحد پایه")
    if row.get("book_imbalance") is not None:
        ctx.append(f"عدم‌توازن دفتر سفارش (±۱٪): {num(row['book_imbalance'], '{:.2f}', sign=True)}"
                   + (f"، تغییر {num(row['book_imbalance_change'], '{:.2f}', sign=True)}" if row.get("book_imbalance_change") is not None else ""))
    if ctx:
        L += ["", "زمینه‌ی زنده (در امتیاز بک‌تست‌شده نیست؛ تاریخچه‌ی رایگان ندارد):", *[f"• {c}" for c in ctx]]
    L.append("\nاین سیگنال پیش از انتشار داده است؛ عدد واقعی می‌تواند خلاف انتظار بازار باشد.")
    return "\n".join(L) + footer()


def prepos_line(rows: list[dict], event_id: str) -> str | None:
    """One line per asset for the heads-up message."""
    rs = [r for r in rows if r["event_id"] == event_id]
    if not rs:
        return None
    parts = []
    for r in rs:
        st = PREPOS_STATUS_FA.get(r["status"], r["status"])
        if r["status"] in ("signal", "watchlist"):
            st += f" ({word(r['direction'])}، امتیاز {num(r.get('trigger_score'), '{:.2f}', sign=True)})"
        elif r["status"] == "waiting" and r.get("window_opens"):
            st += f" — باز می‌شود {tehran(r['window_opens'])}"
        parts.append(f"{ASSET_FA[r['asset']]}: {st}")
    return "⏱ پیش‌موقعیت‌گیری: " + " · ".join(parts)
