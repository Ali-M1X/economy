"""Persian Telegram messages (HTML parse mode), built only from `reports.facts.Facts`.

Every function returns ready-to-send HTML. Dynamic text is escaped; numbers and dates are wrapped in Unicode
isolates (core.fa.ltr) so signs and digit order survive right-to-left rendering in Telegram.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from core.fa import fa, label, ltr, word
from core.timeutil import TEHRAN
from notify.telegram import esc
from reports.facts import MEASURE_FA, Facts, channel_fa, indicators_cfg, ind_name, releases_cfg

DISCLAIMER = "⚠️ این پیام صرفاً تحلیلی و آموزشی است و توصیه مالی یا پیشنهاد خرید و فروش نیست."
ASSET_FA = {"BTC": "بیت‌کوین", "Gold": "طلا"}
HORIZON_FA = {"short": "کوتاه‌مدت (۲۴ ساعت)", "medium": "میان‌مدت (۴ هفته)", "long": "بلندمدت (۶ ماه)"}
WEEKDAY_FA = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]


# ───────────────────────────── helpers ─────────────────────────────

def num(v, fmt: str = "{:,.2f}", sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return ltr((fmt.replace("{:", "{:+", 1) if sign else fmt).format(v))


def score(v) -> str:
    return "—" if v is None else ltr(f"{v:+.0f}")


def tehran(ts) -> str:
    t = pd.Timestamp(ts)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(TEHRAN)
    return f"{WEEKDAY_FA[t.weekday()]} {ltr(f'{t:%Y-%m-%d %H:%M}')}"


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
    return float(v), f"{s.index[-1]:%Y-%m-%d}"


# ───────────────────────────── 1. heads-up (one day before) ─────────────────────────────

def events_on(f: Facts, day_tehran) -> pd.DataFrame:
    c = f.calendar
    if c.empty:
        return c
    d = c.assign(local=c["scheduled_utc"].dt.tz_convert(TEHRAN))
    return d[d["local"].dt.date == day_tehran].drop_duplicates("event_id").sort_values("scheduled_utc")


def headsup(f: Facts, now: pd.Timestamp) -> str | None:
    tomorrow = (now.tz_convert(TEHRAN) + pd.Timedelta(days=1)).date()
    ev = events_on(f, tomorrow)
    if ev.empty:
        return None
    rel = releases_cfg()
    rcfg = {r["key"]: r for r in indicators_cfg()["releases"]}
    L = [f"📅 <b>هشدار یک روز قبل — انتشارهای فردا ({ltr(str(tomorrow))})</b>"]
    for e in ev.itertuples():
        r = rel.get(e.release_id, {})
        L += ["", f"<b>{esc(r.get('name_fa', e.release_id))}</b> {'★' * int(r.get('importance', 1))}",
              f"⏰ زمان: {tehran(e.scheduled_utc)} (تهران)",
              "📊 پیش‌بینی (اجماع): منبع رایگان ندارد — به‌جای آن «انتظار» سیستم = میانگین ۳ دوره قبل"]
        for key in r.get("series_keys", [])[:2]:
            cfg = rcfg.get(key)
            measure = cfg["measure"] if cfg else "level"
            pv = prev_measure(f, key, measure)
            if pv:
                L.append(f"↩️ قبلی {esc(ind_name(key))}: {num(pv[0])} ({esc(MEASURE_FA.get(measure, measure))}، {ltr(pv[1])})")
            if not cfg:
                continue
            L.append(f"🔗 کانال اثر: {chain(cfg.get('channels', []))}")
            for a in ("BTC", "Gold"):
                c = f.coef(key, a, "24h") or {}
                k = c.get("coefficient")
                theory = cfg.get("theory", {}).get(a, 0)
                hot = direction_fa(k) if k is not None else ("صعودی 🟢" if theory > 0 else "نزولی 🔴" if theory < 0 else "نامشخص")
                cold = direction_fa(-k) if k is not None else ("نزولی 🔴" if theory > 0 else "صعودی 🟢" if theory < 0 else "نامشخص")
                basis = f"ضریب ۲۴ساعته {num(k, '{:.1f}', sign=True)}" if k is not None else "بر اساس نظریه؛ ضریب تاریخی ندارد"
                L.append(f"   • {ASSET_FA[a]}: بالاتر از انتظار ← {hot}؛ پایین‌تر ← {cold} ({basis})")
    L.append("\n«انتظار» = میانگین ۳ دوره قبل (اجماع تحلیلگران رایگان در دسترس نیست). ضریب‌ها همبستگی تاریخی‌اند، نه پیش‌بینی.")
    return "\n".join(L) + footer()


# ───────────────────────────── 2. impact report ─────────────────────────────

def releases_block(f: Facts, since: pd.Timestamp) -> list[str]:
    rcfg = {r["key"]: r for r in indicators_cfg()["releases"]}
    L = []
    for r in f.releases:
        if pd.Timestamp(r["release_utc"]) < since:
            continue
        cfg = rcfg.get(r["indicator"], {})
        hot = "بالاتر از انتظار 🔼" if r["z"] > 0 else "پایین‌تر از انتظار 🔽"
        L += [f"• <b>{esc(ind_name(r['indicator']))}</b> ({ltr(r['obs_date'])}): {num(r['actual'])} در برابر انتظار "
              f"{num(r['expected'])} — {hot}، شدت {num(r['z'], '{:.1f}', sign=True)}σ",
              f"   کانال: {chain(cfg.get('channels', []))}"]
        for a in ("BTC", "Gold"):
            k = (f.coef(r["indicator"], a, "24h") or {}).get("coefficient")
            eff = None if k is None else k * np.sign(r["z"])
            L.append(f"   {ASSET_FA[a]}: {direction_fa(eff)}" + (f" (ضریب ۲۴ساعته {num(k, '{:.1f}', sign=True)})" if k is not None else ""))
    return L


def scores_block(f: Facts) -> list[str]:
    L = ["<b>امتیاز کلان (−۱۰۰ تا +۱۰۰)</b>"]
    for a in ("BTC", "Gold"):
        m = f.macro.get(a, {})
        L.append(f"• {ASSET_FA[a]}: " + " · ".join(f"{HORIZON_FA[b].split(' ')[0]} {score((m.get(b) or {}).get('score'))}"
                                                  for b in ("short", "medium", "long")))
        top = (m.get("medium") or {}).get("top", [])[:3]
        if top:
            L.append("   مهم‌ترین عوامل ۴ هفته: " + "، ".join(
                f"{esc(ind_name(t['indicator']))} {num(t['contribution'], '{:.1f}', sign=True)}" for t in top))
    return L


def context_block(f: Facts) -> list[str]:
    s = f.state
    L = []
    reg = s.get("regime") or {}
    if reg:
        p = reg.get("probabilities", {}).get(reg.get("regime"), 0)
        L.append(f"🧭 فاز اقتصادی: <b>{esc(word(reg.get('regime')))}</b> (احتمال {ltr(f'{p * 100:.0f}٪')})")
    st = (s.get("fed_stance") or {}).get("score")
    if st is not None:
        L.append(f"🏦 موضع فدرال: {score(st)} (−۱۰۰ انبساطی … +۱۰۰ انقباضی)")
    fw = s.get("fedwatch_next") or {}
    if fw.get("meeting"):
        pct = {k: ltr(f"{(fw.get(k) or 0) * 100:.0f}٪") for k in ("p_cut", "p_hold", "p_hike")}
        L.append(f"📈 جلسه بعدی FOMC {ltr(fw['meeting'])}: کاهش {pct['p_cut']} · ثابت {pct['p_hold']} · افزایش {pct['p_hike']}")
    return L


def impact_report(f: Facts, now: pd.Timestamp, explanation: str | None = None, trigger: str | None = None) -> str:
    L = [f"🧾 <b>گزارش اثر کلان بر بیت‌کوین و طلا</b> — {tehran(now)}"]
    if trigger:
        L.append(f"علت گزارش: {esc(trigger)}")
    rel = releases_block(f, now - pd.Timedelta(hours=24))
    if rel:
        L += ["", "<b>چه تغییر کرد</b>", *rel]
    news = f.important_news(now - pd.Timedelta(hours=24))
    if not news.empty:
        L += ["", "<b>خبرهای مهم ۲۴ ساعت اخیر</b>"]
        L += [f"• {esc(r.title_fa)} ({'★' * int(r.importance)})" for r in news.head(5).itertuples()]
    if explanation:
        L += ["", "<b>زنجیره علّی</b>", esc(explanation)]
    L += ["", *scores_block(f), "", *context_block(f)]
    return "\n".join(L) + footer()


# ───────────────────────────── 3. signals ─────────────────────────────

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


def signals_message(f: Facts, names: dict[str, str] | None = None) -> str:
    L = ["📊 <b>سیگنال‌های عددی</b>"]
    for a in ("BTC", "Gold"):
        x = f.asset(a)
        if not x:
            continue
        fmt = "{:,.0f}" if a == "BTC" else "{:,.2f}"
        er = x.get("expected_range") or {}
        L += ["", f"<b>{ASSET_FA[a]}</b>" + (f" ({esc(proxy_fa(x['proxy']))})" if x.get("proxy") else "") + f" — قیمت {num(x.get('price'), fmt)}",
              f"امتیاز کلان ۴ هفته: {score((x.get('macro') or {}).get('medium'))}"]
        if er.get("1d"):
            L.append(f"دامنه محتمل: یک روز {num(min(er['1d']), fmt)} تا {num(max(er['1d']), fmt)} · چهار هفته (±۱σ) "
                     f"{num(min(er['4w_1sigma']), fmt)} تا {num(max(er['4w_1sigma']), fmt)}")
        for s in x.get("signals", []):
            L += ["", *signal_card(s, a, blocked=False)]
        for s in x.get("watchlist", []):
            L += ["", *signal_card(s, a, blocked=True)]
        if not x.get("signals") and not x.get("watchlist"):
            L.append("ستاپ فعالی وجود ندارد.")
        bt = (x.get("backtest") or {}).get("summary", {})
        if bt:
            L.append("بک‌تست خارج از نمونه: " + " · ".join(
                f"{word(sd)} {ltr(bt[sd].get('n_trades', 0))} معامله، میانگین R {num(bt[sd].get('avg_r'), '{:.2f}', sign=True)}"
                for sd in ("long", "short") if sd in bt))
        if x.get("risks"):
            L.append("⚠️ ریسک‌ها: " + "؛ ".join(esc(fa(r, names)) for r in x["risks"]))
    return "\n".join(L) + footer()


# ───────────────────────────── 4. weekly summary ─────────────────────────────

def weekly_summary(f: Facts, now: pd.Timestamp) -> str:
    L = [f"🗓 <b>خلاصه هفتگی</b> — {tehran(now)}", "", *scores_block(f), "", *context_block(f)]
    rel = releases_block(f, now - pd.Timedelta(days=7))
    if rel:
        L += ["", "<b>انتشارهای این هفته</b>", *rel]
    if not f.coefs.empty:
        c = f.coefs[(f.coefs["bucket"] == "medium") & (f.coefs["horizon"] == "4w") & (f.coefs["sample"] == "full")]
        L += ["", "<b>قوی‌ترین ضرایب اثر ۴ هفته (محاسبه مجدد این هفته)</b>"]
        for a in ("BTC", "Gold"):
            top = c[c["asset"] == a].sort_values("coefficient", key=abs, ascending=False).head(4)
            L.append(f"• {ASSET_FA[a]}: " + "، ".join(f"{esc(ind_name(r.indicator))} {num(r.coefficient, '{:.1f}', sign=True)}"
                                                    for r in top.itertuples()))
    for a in ("BTC", "Gold"):
        bt = (f.asset(a).get("backtest") or {}).get("summary", {}).get("all")
        if bt and bt.get("n_trades"):
            wr = ltr(f"{(bt.get('win_rate') or 0) * 100:.0f}٪")
            L.append(f"بک‌تست {ASSET_FA[a]}: {ltr(bt['n_trades'])} معامله، نرخ برد {wr}، "
                     f"میانگین R {num(bt.get('avg_r'), '{:.2f}', sign=True)}")
        elif bt is not None:
            L.append(f"بک‌تست {ASSET_FA[a]}: بدون معامله در دوره آزمون")
    nxt = f.calendar[(f.calendar["scheduled_utc"] > now) & (f.calendar["scheduled_utc"] <= now + pd.Timedelta(days=7))] \
        if not f.calendar.empty else pd.DataFrame()
    if not nxt.empty:
        rel_c = releases_cfg()
        L += ["", "<b>تقویم هفته آینده</b>"]
        L += [f"• {tehran(e.scheduled_utc)} — {esc(rel_c.get(e.release_id, {}).get('name_fa', e.release_id))}"
              for e in nxt.drop_duplicates("event_id").sort_values("scheduled_utc").itertuples()]
    return "\n".join(L) + footer()


# ───────────────────────────── 5. health warning / 6. news alert ─────────────────────────────

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
