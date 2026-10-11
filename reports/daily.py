"""The one daily report (09:00 Tehran), also used after a release with "what changed" on top.

Order: title → 💡 today in ≤3 lines → 📖 the market story (one paragraph) → 📅 the next 7 days (narrow table + one line
per important release: odds, usual BTC/gold move for a hotter or cooler print, normal post-release volatility, the
forecast's source) → one block per asset (price, my read, levels, 4-week scenarios, what to do, ≤5 key indicators) →
one disclaimer line. It is one Telegram message when it fits; otherwise it is split at section boundaries with a
"(۱/۲)" marker and no repeated headers. Full indicator tables and the glossary are in the weekly report.

Every number is computed by code (reports.advisor, reports.outlook, reports.asset_report). If LLM_API_KEY is set, a
free model may rephrase the story and "my read" paragraphs; a rewrite that adds any number is discarded.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from core.jalali import fa_digits, jdate, jdate_short
from notify.telegram import LIMIT, esc
from reports import advisor as adv
from reports import asset_report as ar
from reports import outlook as ol
from reports.facts import Facts
from reports.messages import ASSET_FA, DISCLAIMER, num, tehran

MAX_CHARS = LIMIT - 96  # head-room for the split marker and Telegram's own entity parsing
ICON = {"BTC": "₿", "Gold": "🥇"}
KEY_ROWS = 5
EVENT_LINES = 3
TABLE_ROWS = 7
REL_SHORT = {"cpi": "CPI", "ppi": "PPI", "pce": "PCE", "nfp": "NFP", "jolts": "JOLTS", "claims": "Claims", "adp": "ADP",
             "gdp": "GDP", "fomc": "FOMC", "philly": "Philly", "empire": "Empire", "m2": "M2"}
HOT_FA = {"unemployment_rate": ("بالاتر (بازار کار ضعیف‌تر)", "پایین‌تر"), "initial_claims": ("بالاتر (ضعف اشتغال)", "پایین‌تر")}
FULL_NOTE = "جدول کامل شاخص‌ها و راهنمای اصطلاحات: گزارش هفتگی (شنبه)."


# ───────────────────────────── small formatters ─────────────────────────────

INDEX_KEYS = ("philly_fed_activity", "empire_state_activity")


def fval(key: str | None, v: float | None) -> str:
    """A forecast in the series' own measure, for prose (Persian digits)."""
    if v is None or key is None:
        return "—"
    if key in INDEX_KEYS:
        return num(v, "{:.1f}")
    if key == "adp_private_payrolls":
        return num(v, "{:,.0f}") + " هزار شغل"
    return adv.fmt_value(key, v)


def tval(key: str | None, v: float | None) -> str:
    """The same value for the narrow ASCII table cell."""
    if v is None or key is None:
        return "—"
    if key in ("nonfarm_payrolls", "adp_private_payrolls", "jolts_openings"):
        return ar.nz(f"{v:,.0f}k")
    if key == "initial_claims":
        return f"{v / 1000:,.0f}k"
    if key in INDEX_KEYS:
        return ar.nz(f"{v:.1f}")
    return ar.nz(re.sub(r"(\.\d)0%$", r"\1%", f"{v:.2f}%"))


def p100(p: float) -> str:
    return f"{round(100 * p):.0f}"


def hot_words(key: str | None) -> tuple[str, str]:
    if key in HOT_FA:
        return HOT_FA[key]
    if key in adv.INFLATION:
        return "داغ‌تر", "سردتر"
    if key in adv.LABOR:
        return "قوی‌تر", "ضعیف‌تر"
    return "بالاتر", "پایین‌تر"


# ───────────────────────────── 📅 the next 7 days ─────────────────────────────

def outlook_table(outs: list[ol.EventOutlook], now: pd.Timestamp) -> str:
    """Event · forecast (* = system expectation) · Hot/Cool % · day — ≤ 32 characters per line."""
    head = f"{'Event':<6} {'Fcst':>7} {'H/C':>5} Day"
    body = []
    for o in outs[:TABLE_ROWS]:
        if o.fedwatch:
            fc = "Hold" if (o.odds.inline >= max(o.odds.hot, o.odds.cool)) else "Cut" if o.odds.cool > o.odds.hot else "Hike"
        else:
            fc = tval(o.key, o.forecast) + ("" if o.basis == "consensus" or o.forecast is None else "*")
        hc = f"{p100(o.odds.hot)}/{p100(o.odds.cool)}" if o.odds else "—"
        day = jdate_short(o.when, now)
        body.append(f"{(ar.SHORT.get(o.key) or REL_SHORT.get(o.release_id, o.release_id))[:6]:<6} {fc[:7]:>7} {hc:>5} {ar._rtl(day)}")
    return ar._table(head, body)[0]


def _moves(o: ol.EventOutlook) -> dict[str, bool]:
    """Whether a 1σ surprise usually moves each asset by a noticeable amount."""
    return {a: bool(b and abs(b.get("coef") or 0) >= 0.3 and abs(b["beta"]) >= adv.MIN_EFFECT)
            for a, b in ((a, o.effects.get(a)) for a in ("BTC", "Gold"))}


def _effect(o: ol.EventOutlook, sign: int) -> str:
    parts, mv = [], _moves(o)
    for a in ("BTC", "Gold"):
        b = o.effects.get(a)
        e = sign * b["beta"] if b else None
        if not mv[a]:
            parts.append(f"{ASSET_FA[a]} ناچیز")
        else:
            parts.append(f"{ASSET_FA[a]} {adv.pct(e)}" + (" (خلاف منطق، کم‌اعتبار)" if b["conflict"] else ""))
    return "، ".join(parts)


def odds_phrase(o: ol.EventOutlook) -> str:
    hot, cool = hot_words(o.key)
    od = o.odds
    if od is None:
        return "برای این انتشار احتمال قابل محاسبه نیست."
    if o.fedwatch:
        fw = o.fedwatch
        bits = [f"{w} {adv.prob(fw[k])}" for k, w in (("p_hold", "ثابت ماندن"), ("p_cut", "کاهش"), ("p_hike", "افزایش"))
                if fw.get(k)]
        return "بازار این احتمال‌ها را قیمت کرده: " + "، ".join(bits) + "."
    three = f"{hot} {adv.prob(od.hot)}، مطابق {adv.prob(od.inline)}، {cool} {adv.prob(od.cool)}"
    if od.method == "even":
        return f"شانس {hot} یا {cool} آمدن تقریباً برابر است ({three})؛ سیستم برتری‌ای ندارد."
    if not od.edge:
        return f"{three}؛ کفه به هیچ طرف سنگین نیست."
    return f"{three}؛ کفه به سمت «{hot if od.hot > od.cool else cool}» است."


def source_phrase(o: ol.EventOutlook) -> str:
    names = [ol.SOURCE_FA[s] for s in o.sources]
    if o.nowcast is not None and o.nowcast_source:
        names = [n if n != ol.SOURCE_FA[o.nowcast_source] else f"{n} {fval(o.key, o.nowcast)}" for n in names]
    return " + ".join(names)


def event_line(o: ol.EventOutlook) -> list[str]:
    name = adv.SHORT.get(o.key) if o.key in adv.SHORT else esc(o.name_fa)
    if o.fedwatch:
        head = f"• <b>{esc(o.name_fa)}</b> ({tehran(o.when)}): {odds_phrase(o)}"
        return [head, f"   ↳ منبع: {ol.SOURCE_FA['fedwatch']}"]
    fc = (f"{'اجماع' if o.basis == 'consensus' else 'انتظار سیستم'} {fval(o.key, o.forecast)}"
          if o.forecast is not None else "پیش‌بینی ندارد")
    L = [f"• <b>{name}</b> ({tehran(o.when)}): {fc}. {odds_phrase(o)}"]
    if o.key and o.effects:
        hot, cool = hot_words(o.key)
        if not any(_moves(o).values()):
            L.append("   ↳ واکنش معمول ۲۴ ساعته‌ی بیت‌کوین و طلا به شگفتی این عدد در گذشته ناچیز بوده")
        else:
            L.append(f"   ↳ اگر {hot} بیاید: {_effect(o, 1)} · اگر {cool}: {_effect(o, -1)} (۲۴ ساعت، میانگین گذشته)")
        sds = [f"{ASSET_FA[a]} ±{num(b['sd'], '{:.1f}')}٪" for a, b in o.effects.items() if b.get("sd")]
        if sds:
            L.append(f"   ↳ نوسان عادی ۲۴ ساعت بعد: {'، '.join(sds)}")
    L.append(f"   ↳ منبع پیش‌بینی: {source_phrase(o)}")
    return L


def week_section(f: Facts, now: pd.Timestamp, outs: list[ol.EventOutlook]) -> str:
    L = ["📅 <b>۷ روز آینده</b>"]
    if not outs:
        return "\n".join(L + ["انتشار مهمی در ۷ روز آینده ثبت نشده است."])
    L.append(outlook_table(outs, now))
    legend = "H/C = احتمال بالاتر/پایین‌تر از پیش‌بینی (٪)"
    if any(o.basis != "consensus" and o.forecast is not None and not o.fedwatch for o in outs[:TABLE_ROWS]):
        legend += "؛ * = انتظار سیستم (اجماع رایگان در دسترس نیست)"
    L.append(legend)
    imp = sorted([o for o in outs if o.importance >= 4], key=lambda o: o.when)[:EVENT_LINES]
    for o in imp:
        L += event_line(o)
    return "\n".join(L)


# ───────────────────────────── 💡 summary and 📖 story ─────────────────────────────

def event_brief(o: ol.EventOutlook | None) -> str | None:
    if o is None:
        return None
    name = adv.SHORT.get(o.key) if o.key in adv.SHORT else esc(o.name_fa)
    out = f"🎯 مهم‌ترین رویداد هفته: {name} ({jdate(o.when, year=False)})"
    if o.fedwatch:
        return out + f" — {odds_phrase(o)}"
    if o.odds and o.odds.method != "even":
        hot, _ = hot_words(o.key)
        return out + f" — احتمال {hot} آمدن {adv.prob(o.odds.hot)}"
    return out + " — شانس دو طرف تقریباً برابر"


def story(f: Facts, now: pd.Timestamp) -> str:
    t = adv.theme_reads(f, now)
    parts = [adv.story_sentence(t), adv.fed_sentence(f), adv.rates_sentence(f, now), adv.asset_takeaways(f, t)]
    return " ".join(p for p in parts if p)


# ───────────────────────────── per asset ─────────────────────────────

def key_table(f: Facts, asset: str, now: pd.Timestamp) -> tuple[str, str]:
    rows = [r for r in ar.indicator_rows(f, asset) if r.coef is not None and abs(r.coef) >= 1][:KEY_ROWS]
    if not rows:
        return "", ""
    head = f"{'Name':<5} {'Value':>7} Impact"
    body = []
    for r in rows:
        imp = ar.impact_cell(r)
        body.append(f"{r.ticker[:5]:<5} {ar._compact(r.value) + ar.change_arrow(r.change):>7} {ar._rtl(imp)}")
    names = "، ".join(f"{esc(r.ticker)} = {esc(r.name_fa)}" for r in rows)
    return ar._table(head, body)[0], names


def price_line(v: dict, asset: str) -> str:
    mv = v["mv"]
    out = f"{ICON[asset]} <b>{v['name']}</b> — {adv.px(v['price'], asset)} دلار"
    if asset == "Gold":
        out += " (PAXG)"
    if mv and mv.get("chg24") is not None:
        out += f" · ۲۴ ساعت {adv.pct(mv['chg24'])} · ۷ روز {adv.pct(mv.get('chg7d'))}"
    return out


def asset_section(f: Facts, asset: str, now: pd.Timestamp, v: dict, read: str) -> str:
    L = [price_line(v, asset), f"<b>برداشت من:</b> {read}"]
    if v["r"] or v["s"]:
        L.append(f"حمایت نزدیک {adv.px(v['s'], asset)} · مقاومت نزدیک {adv.px(v['r'], asset)}")
    sc = v["sc"]
    if sc:
        L.append(f"۴ هفته‌ی آینده: 🟢 بالای {adv.px(v['r'], asset)} {adv.prob(sc['up'])} · ⚪ بین دو سطح "
                 f"{adv.prob(sc['base'])} · 🔴 زیر {adv.px(v['s'], asset)} {adv.prob(sc['down'])}")
    L += ["<b>چه کنم؟</b>", f"👤 دارنده: {v['hold']}", f"🛒 خریدار: {v['buy']}"]
    if v["risk"]:
        L.append(f"⚖️ ریسک رویداد: {v['risk']}")
    tbl, names = key_table(f, asset, now)
    if tbl:
        L += ["شاخص‌های کلیدی (▲▼ = جهت اثر فعلی بر قیمت، ؟ = خلاف منطق اقتصادی):", tbl, names]
    return "\n".join(L)


# ───────────────────────────── post-release ─────────────────────────────

def released_section(f: Facts, now: pd.Timestamp, block: list[str]) -> str:
    rel = [r for r in f.releases if pd.Timestamp(r["release_utc"]) >= now - pd.Timedelta(days=1)]
    L = list(block) if block else ["🔄 <b>چه تغییر کرد</b>"]
    if rel:
        L += ["📊 چه منتشر شد (" + ("اجماع" if all(r.get("basis") == "vs consensus" for r in rel) else "انتظار") + "):",
              ar.release_table(rel), *[ar.release_line(r) for r in rel]]
    return "\n".join(L)


# ───────────────────────────── polish and split ─────────────────────────────

PARA_SYSTEM = """You are a senior macro and markets analyst. Rewrite the Persian (Farsi) PARAGRAPH you receive so it
reads like an experienced analyst talking to one client: fluent, connected sentences, cause → effect, ending in a clear
takeaway. Use ONLY numbers that appear in the paragraph, with the same values; never add, derive or round a number.
Plain text, one paragraph, Persian, no Markdown or HTML, no new advice. Keep it about the same length."""


def polish(text: str) -> str:
    """Rephrase a paragraph with the free model when LLM_API_KEY is set and the rewrite keeps to its numbers."""
    if not os.environ.get("LLM_API_KEY"):
        return text
    from llm import free_writer
    from llm.claude import LLMError
    from reports.writer import unknown_numbers

    draft = adv._ISOLATES.sub("", adv.plain(text))
    try:
        out, _ = free_writer.chat(PARA_SYSTEM, draft, max_tokens=1200)
    except LLMError as e:
        print(f"::warning::daily rewrite unavailable ({str(e)[:200]})")
        return text
    out = re.sub(r"\s+", " ", re.sub(r"<[^>]+>|```\w*", "", out)).strip()
    if not out or unknown_numbers(out, {"draft": draft}) or len(out) > 2 * len(draft) + 200 or len(out) < len(draft) // 3:
        print("::warning::daily rewrite rejected (new numbers or bad length); using the template")
        return text
    return esc(out) if "<" not in text else out


def split(sections: list[str], limit: int = MAX_CHARS) -> list[str]:
    """Pack sections into as few messages as possible; mark parts "(۱/۲)" when more than one is needed."""
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
        msgs[0] = f"{title} ({fa_digits('1')}/{n}){nl}{body}"
        msgs[1:] = [f"({fa_digits(str(i))}/{n})\n{m}" for i, m in enumerate(msgs[1:], 2)]
    return msgs


# ───────────────────────────── build ─────────────────────────────

def build(f: Facts, now: pd.Timestamp, changes: list[str] | None = None, trigger: str | None = None,
          rewrite: bool = True) -> list[str]:
    """The unified report as 1 (or, if needed, 2–3) Telegram messages."""
    outs = ol.upcoming(f, now)
    views = {a: adv.asset_view(f, a, now) for a in ("BTC", "Gold") if f.asset(a)}
    post = changes is not None
    title = (f"🔔 <b>گزارش بعد از انتشار</b> — {jdate(now)}" if post else f"☀️ <b>گزارش روزانه</b> — {jdate(now)}")
    head = [title, f"داده‌ها تا {tehran(f.as_of or now)}"]
    if trigger and post:
        head.append(f"علت: {esc(trigger)}")
    sections = ["\n".join(head)]
    if post:
        sections.append(released_section(f, now, changes))
    summ = ["💡 <b>خلاصه‌ی امروز</b>"]
    summ += [f"{ICON[a]} {ASSET_FA[a]}: {v['tldr']}" for a, v in views.items()]
    brief = event_brief(ol.most_important(outs))
    if brief:
        summ.append(brief)
    sections.append("\n".join(summ[:4]))
    st = story(f, now)
    if st:
        sections.append("📖 <b>داستان بازار</b>\n" + (polish(st) if rewrite else st))
    sections.append(week_section(f, now, outs))
    for a, v in views.items():
        sections.append(asset_section(f, a, now, v, polish(v["read"]) if rewrite else v["read"]))
    sections.append(FULL_NOTE + "\n" + DISCLAIMER)
    return split(sections)
