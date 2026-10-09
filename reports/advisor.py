"""Experimental advisory report: the same numbers as the regular reports, told as a story a reader can decide on.

    python -m jobs.advisor --root output [--dry-run]

Three Telegram messages:
1. the macro story: what the latest data say (inflation, jobs, the Fed, rates and dollar, liquidity, the cycle), what
   each number usually does to Bitcoin and gold in %, and the releases ahead with what a surprise would do;
2. and 3. one advisory message per asset: where the price is in its trend, how much the macro picture weighs against
   the asset's normal volatility, three 4-week scenarios with probabilities, and what to do for a holder, a buyer and
   for risk.

Every figure is computed here from the job outputs: % effects are the historical regression betas (asset return per 1σ
of the indicator, reports.facts / jobs.impact) times the current move in σ; scenario probabilities come from the
asset's 4-week volatility and the macro drift the signals job already uses (log-normal). Nothing is a trade signal.
"""

from __future__ import annotations

import re
from math import erf, log, sqrt

import numpy as np
import pandas as pd

from core.jalali import fa_digits, jdate, jdate_short
from jobs import moves
from notify.telegram import esc
from reports.asset_report import against_theory
from reports.facts import Facts, ind_name, indicators_cfg, releases_cfg
from reports.messages import ASSET_FA, num, tehran

TITLE_TAG = "🧪 آزمایشی"
NOTE = ("🧪 این نسخه‌ی آزمایشی گزارش است و کنار گزارش‌های معمول می‌آید. اعداد اثر، میانگین واکنش تاریخی‌اند و "
        "احتمال‌ها از نوسان معمول هر دارایی حساب شده‌اند؛ پیش‌بینی قطعی نیستند و توصیه‌ی مالی هم نیستند.")

INFLATION = ["cpi", "core_cpi", "ppi_final_demand", "pce", "core_pce", "avg_hourly_earnings"]
LABOR = {"nonfarm_payrolls": 1, "unemployment_rate": -1, "jolts_openings": 1, "initial_claims": -1}  # +1 = stronger
SHORT = {"cpi": "CPI", "core_cpi": "CPI هسته", "ppi_final_demand": "PPI", "pce": "PCE", "core_pce": "PCE هسته",
         "avg_hourly_earnings": "دستمزد ساعتی", "nonfarm_payrolls": "اشتغال غیرکشاورزی (NFP)",
         "unemployment_rate": "نرخ بیکاری", "jolts_openings": "فرصت‌های شغلی (JOLTS)",
         "initial_claims": "مدعیان بیمه‌ی بیکاری"}
REGIME_FA = {"Expansion": "رونق", "Peak": "اوج", "Recession": "رکود", "Recovery": "بهبود"}
RECENT_DAYS = 45
MIN_EFFECT = 0.15  # % — smaller usual effects read as "negligible"


# ───────────────────────────── numbers ─────────────────────────────

def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def pct(v: float | None, d: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return num(0.0, "{:." + str(d) + "f}") + "٪" if abs(v) < 0.5 * 10 ** -d else num(v, "{:." + str(d) + "f}", sign=True) + "٪"


def prob(p: float) -> str:
    return fa_digits(f"{round(100 * p):.0f}") + "٪"


def px(v: float | None, asset: str) -> str:
    if v is None:
        return "—"
    return num(round(v, -2) if asset == "BTC" else round(v), "{:,.0f}")


def beta(f: Facts, key: str, asset: str, horizon: str) -> dict | None:
    """% return of the asset per 1σ of the indicator. BTC averages the full sample and the last 3 years, like the
    macro score does, because its macro sensitivity changed over time."""
    rows = [r for r in (f.coef(key, asset, horizon, s) for s in (("full", "recent") if asset == "BTC" else ("full",)))
            if r and r.get("beta") is not None and np.isfinite(r["beta"])]
    if not rows:
        return None
    full = rows[0]
    coef = float(np.mean([r.get("coefficient") or 0 for r in rows]))
    section = "release" if horizon in ("1h", "4h", "24h") else "state"
    return {"beta": float(np.mean([r["beta"] for r in rows])), "coef": coef,
            "conf": full.get("confidence"), "hit": full.get("hit_rate"), "sd": full.get("sd_ret"),
            "conflict": bool(full.get("conflict")) or against_theory(key, asset, coef, section)}


def reliability(b: dict | None) -> str:
    if not b:
        return "بدون داده‌ی تاریخی"
    if b["conflict"]:
        return "خلاف منطق اقتصادی، کم‌اعتبار"
    return {"high": "اعتبار خوب", "medium": "اعتبار متوسط"}.get(b["conf"], "اعتبار کم")


def effect_pair(f: Facts, key: str, z: float, horizon: str) -> str:
    """'بیت‌کوین −0.8٪ · طلا −0.2٪ (اعتبار متوسط)' — the usual move for a z-sized surprise or move."""
    parts = []
    for a in ("BTC", "Gold"):
        b = beta(f, key, a, horizon)
        e = b["beta"] * z if b else 0.0
        if not b or abs(b["coef"] or 0) < 0.3 or abs(e) < MIN_EFFECT:
            parts.append(f"{ASSET_FA[a]} ناچیز")
        else:
            parts.append(f"{ASSET_FA[a]} {pct(e)} ({reliability(b)})")
    return " · ".join(parts)


def has_effect(f: Facts, key: str, z: float, horizon: str) -> bool:
    return any(b and abs(b["coef"] or 0) >= 0.3 and abs(b["beta"] * z) >= MIN_EFFECT
               for b in (beta(f, key, a, horizon) for a in ("BTC", "Gold")))


def fmt_value(key: str, v: float) -> str:
    if key == "unemployment_rate":
        return num(v, "{:.1f}") + "٪"
    if key == "nonfarm_payrolls":
        return num(v, "{:,.0f}") + " هزار شغل"
    if key == "jolts_openings":
        return num(v / 1000, "{:.2f}") + " میلیون"
    if key == "initial_claims":
        return num(v / 1000, "{:,.0f}") + " هزار"
    return num(v, "{:.2f}") + "٪"


def change_since(f: Facts, key: str, now: pd.Timestamp, days: int = 28) -> tuple[float, float] | None:
    """(latest value, change over `days`) from the snapshot observations."""
    obs = f._obs()
    s = obs[obs["series_key"] == key].dropna(subset=["value"]).sort_values("date")
    if s.empty:
        return None
    d = pd.to_datetime(s["date"])
    d = d.dt.tz_localize("UTC") if d.dt.tz is None else d
    last = float(s["value"].iloc[-1])
    before = s[d <= d.iloc[-1] - pd.Timedelta(days=days)]
    return last, (last - float(before["value"].iloc[-1])) if not before.empty else None


# ───────────────────────────── the macro story ─────────────────────────────

def recent_surprises(f: Facts, now: pd.Timestamp, keys) -> list[tuple[str, dict]]:
    out = []
    for k in keys:
        s = (f.surprises or {}).get(k)
        if s and s.get("z") is not None and now - pd.Timestamp(s["release_utc"]) <= pd.Timedelta(days=RECENT_DAYS):
            out.append((k, s))
    return out


def theme_reads(f: Facts, now: pd.Timestamp) -> dict:
    """Average surprise of the inflation prints (+ = hotter) and of the jobs prints (+ = stronger)."""
    inf = recent_surprises(f, now, INFLATION)
    lab = recent_surprises(f, now, LABOR)
    return {"inflation": float(np.mean([s["z"] for _, s in inf])) if inf else None,
            "labor": float(np.mean([LABOR[k] * s["z"] for k, s in lab])) if lab else None, "inf": inf, "lab": lab}


def inflation_word(z: float | None) -> str:
    return "نامشخص" if z is None else "داغ‌تر از روند" if z >= 0.4 else "سردتر از روند" if z <= -0.4 else "نزدیک روند"


def labor_word(z: float | None) -> str:
    return "نامشخص" if z is None else "قوی‌تر از روند" if z >= 0.4 else "رو به ضعف" if z <= -0.4 else "متعادل"


def story_sentence(t: dict) -> str:
    """The one-line economic reading of inflation × jobs, and what it usually means for the Fed and the two assets."""
    i, lab = t["inflation"], t["labor"]
    hot, cool = i is not None and i >= 0.4, i is not None and i <= -0.4
    strong, weak = lab is not None and lab >= 0.4, lab is not None and lab <= -0.4
    if hot and weak:
        return ("تورم داغ‌تر از انتظار آمده و هم‌زمان بازار کار رو به ضعف است. این ترکیب دست فدرال رزرو را می‌بندد: "
                "با تورم بالا نمی‌تواند سریع نرخ را پایین بیاورد، با ضعف اشتغال هم نمی‌تواند سخت‌گیرتر شود. "
                "نتیجه معمولاً بازاری بی‌جهت و حساس به هر عدد تازه‌ی تورم است.")
    if hot and not weak:
        return ("تورم داغ‌تر از انتظار است و بازار کار هم ضعیف نیست؛ یعنی فدرال رزرو دلیلی برای کاهش نرخ ندارد و "
                "احتمال سخت‌گیری بیشتر بالا می‌رود. این فضا معمولاً برای بیت‌کوین و طلا منفی است.")
    if cool and weak:
        return ("تورم آرام شده و بازار کار هم سست است؛ راه کاهش نرخ بهره باز می‌شود. این معمولاً به نفع طلا و "
                "بیت‌کوین است، مگر این‌که ترس از رکود غالب شود.")
    if cool and strong:
        return ("تورم آرام و بازار کار قوی است، یعنی همان «فرود نرم» که بازارها دوست دارند؛ معمولاً برای "
                "دارایی‌های پرریسک مثل بیت‌کوین مثبت است.")
    if cool:
        return "تورم آرام‌تر از روند است و فشار برای سخت‌گیری فدرال رزرو کم شده؛ زمینه کمی به نفع طلا و بیت‌کوین است."
    if weak:
        return "بازار کار نشانه‌ی ضعف داده؛ اگر ادامه پیدا کند، انتظار کاهش نرخ بیشتر می‌شود."
    if strong:
        return "بازار کار قوی‌تر از انتظار است؛ این احتمال کاهش نرخ را کم می‌کند."
    return "داده‌های اخیر تورم و اشتغال نزدیک روند بوده‌اند و پیام روشنی برای فدرال رزرو ندارند."


def fed_sentence(f: Facts) -> str | None:
    s, fw = f.state.get("fed_stance") or {}, f.state.get("fedwatch_next") or {}
    sc = s.get("score")
    if sc is None and not fw:
        return None
    tone = ("موضع فدرال رزرو متمایل به انقباضی است" if sc is not None and sc >= 25 else
            "موضع فدرال رزرو متمایل به انبساطی است" if sc is not None and sc <= -25 else "موضع فدرال رزرو میانه است")
    out = tone + (f" (امتیاز {num(sc, '{:+.0f}')} از ±۱۰۰)" if sc is not None else "")
    if fw.get("meeting"):
        bits = [f"{w} {prob(fw[k])}" for k, w in (("p_hold", "ثابت ماندن"), ("p_hike", "افزایش"), ("p_cut", "کاهش"))
                if fw.get(k)]
        out += f". برای جلسه‌ی {jdate(fw['meeting'], weekday=False, year=False)} بازار این احتمال‌ها را قیمت کرده: " + "، ".join(bits)
        if (fw.get("p_hike") or 0) >= 0.1 and not fw.get("p_cut"):
            out += "؛ یعنی حتی احتمال افزایش نرخ جدی گرفته می‌شود و کاهش نرخ فعلاً روی میز نیست"
    return out + "."


def regime_sentence(f: Facts) -> str | None:
    r = f.state.get("regime") or {}
    if not r.get("regime"):
        return None
    p = (r.get("probabilities") or {})
    nxt = sorted(((v, k) for k, v in p.items() if k != r["regime"]), reverse=True)
    out = f"اقتصاد آمریکا در فاز «{REGIME_FA.get(r['regime'], r['regime'])}» است (احتمال {prob(p.get(r['regime'], 0))})"
    if nxt:
        out += f" و فاز محتمل بعدی «{REGIME_FA.get(nxt[0][1], nxt[0][1])}» است ({prob(nxt[0][0])})"
    return out + "."


def rates_sentence(f: Facts, now: pd.Timestamp) -> str | None:
    """Higher yields and a stronger dollar raise the cost of holding gold and Bitcoin; say so when it happened."""
    y, d = change_since(f, "ust_10y", now), change_since(f, "dxy", now)
    up_y = y and y[1] is not None and y[1] >= 0.15
    up_d = d and d[1] is not None and d[0] - d[1] and d[1] / (d[0] - d[1]) >= 0.01
    dn_y = y and y[1] is not None and y[1] <= -0.15
    dn_d = d and d[1] is not None and d[0] - d[1] and d[1] / (d[0] - d[1]) <= -0.01
    if up_y and up_d:
        return (f"در ۴ هفته‌ی اخیر هم بازده اوراق ({num(y[1], '{:+.2f}')} واحد درصد) و هم دلار "
                f"({num(d[1] / (d[0] - d[1]) * 100, '{:+.1f}')}٪) بالا رفته‌اند. طبق منطق اقتصادی این هزینه‌ی نگه‌داشتن "
                "طلا و بیت‌کوین را بالا می‌برد و باد مخالف است؛ هرچند در داده‌های اخیر اثرش ضعیف دیده شده.")
    if dn_y and dn_d:
        return "در ۴ هفته‌ی اخیر بازده اوراق و دلار پایین آمده‌اند؛ این معمولاً باد موافق طلا و بیت‌کوین است."
    if up_y:
        return f"بازده اوراق ۱۰ ساله در ۴ هفته {num(y[1], '{:+.2f}')} واحد درصد بالا رفته؛ فشاری بالقوه روی طلا و بیت‌کوین."
    if up_d:
        return "دلار در ۴ هفته‌ی اخیر تقویت شده؛ معمولاً فشاری روی طلا."
    return None


def decision_note(t: dict, evs: list[dict]) -> str | None:
    """The single thing to watch, tied to the story."""
    if not evs:
        return None
    e = evs[0]
    hot = sum(1 for _, s in t["inf"] if s["z"] >= 0.3)
    if e["key"] in INFLATION and hot >= 2:
        return (f"🎯 نکته‌ی تصمیم: حساس‌ترین لحظه‌ی پیش رو {esc(e['name'])} در {tehran(e['when'])} است. بعد از "
                f"{fa_digits(str(hot))} عدد داغ تورمی اخیر، یک عدد داغ دیگر احتمال سخت‌گیری فدرال رزرو را بالا می‌برد؛ "
                "یک عدد سرد، برعکس، بزرگ‌ترین خبر خوب ممکن برای بیت‌کوین و طلا در این دو هفته است.")
    return f"🎯 نکته‌ی تصمیم: رویداد مهم بعدی {esc(e['name'])} در {tehran(e['when'])} است؛ تا آن موقع بازار محتاط می‌ماند."


MARKET = [("ust_10y", "بازده اوراق ۱۰ ساله", "pp"), ("real_yield_10y", "بازده واقعی ۱۰ ساله", "pp"),
          ("dxy", "شاخص دلار", "pct"), ("net_liquidity_weekly", "نقدینگی خالص فدرال رزرو", "pct"),
          ("m2", "حجم پول (M2)", "pct"), ("vix", "شاخص ترس (VIX)", "pts")]


def market_lines(f: Facts, now: pd.Timestamp) -> list[str]:
    """4-week moves of rates, dollar and liquidity with the usual 4-week effect on each asset."""
    st = {a: ((f.macro.get(a) or {}).get("medium") or {}).get("state") or {} for a in ("BTC", "Gold")}
    L = []
    for key, name, unit in MARKET:
        s = st["BTC"].get(key, st["Gold"].get(key))
        if s is None:
            continue
        vc = change_since(f, key, now)
        what = ""
        if vc and vc[1] is not None:
            last, ch = vc
            if unit == "pp":
                what = f"{num(last, '{:.2f}')}٪ ({num(ch, '{:+.2f}')} واحد درصد در ۴ هفته)"
            elif unit == "pct" and last and last - ch:
                what = f"{num(ch / (last - ch) * 100, '{:+.1f}')}٪ در ۴ هفته"
            elif unit == "pts":
                what = f"{num(last, '{:.1f}')} ({num(ch, '{:+.1f}')} واحد در ۴ هفته)"
        move = "بالا رفته" if s > 0.1 else "پایین آمده" if s < -0.1 else "تقریباً ثابت مانده"
        line = f"• {name}: {what + '، ' if what else ''}{move}"
        if abs(s) > 0.1 and has_effect(f, key, 2 * s, "4w"):
            line += f"\n   ↳ اثر معمول ۴ هفته‌ای: {effect_pair(f, key, 2 * s, '4w')}"
        L.append(line)
    return L


def release_lines(f: Facts, items: list[tuple[str, dict]], word) -> list[str]:
    L = []
    for k, s in items:
        L.append(f"• {SHORT.get(k, esc(ind_name(k)))} ({jdate_short(s['release_utc'])}): {fmt_value(k, s['actual'])}، "
                 f"انتظار {fmt_value(k, s['expected'])} → {word(k, s['z'])}")
        if has_effect(f, k, s["z"], "24h"):
            L.append(f"   ↳ واکنش معمول ۲۴ ساعته به چنین عددی: {effect_pair(f, k, s['z'], '24h')}")
    return L


def _inf_word(k, z):
    return "داغ‌تر از انتظار" if z >= 0.3 else "سردتر از انتظار" if z <= -0.3 else "نزدیک انتظار"


def _lab_word(k, z):
    s = LABOR[k] * z
    return "قوی‌تر از انتظار" if s >= 0.3 else "ضعیف‌تر از انتظار" if s <= -0.3 else "نزدیک انتظار"


def upcoming(f: Facts, now: pd.Timestamp, days: int = 14) -> list[dict]:
    c = f.calendar
    if c is None or c.empty:
        return []
    up = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=days))]
    up = up[up["importance"] >= 4] if "importance" in up else up
    rel, out = releases_cfg(), []
    rkeys = {r["key"] for r in indicators_cfg()["releases"]}
    for e in up.drop_duplicates("event_id").sort_values("scheduled_utc").itertuples():
        r = rel.get(e.release_id, {})
        keys = sorted((k for k in r.get("series_keys", []) if k in rkeys),
                      key=lambda k: -max(abs((beta(f, k, a, "24h") or {}).get("coef") or 0) for a in ("BTC", "Gold")))
        out.append({"name": r.get("name_fa") or getattr(e, "name_fa", e.release_id), "when": e.scheduled_utc,
                    "key": keys[0] if keys else None, "release_id": e.release_id})
    return out


def sigma_units(f: Facts, key: str) -> float | None:
    """1σ of the surprise in the indicator's own units, from its last surprise."""
    s = (f.surprises or {}).get(key)
    if not s or not s.get("z") or s.get("actual") is None or s.get("expected") is None:
        return None
    v = abs(s["actual"] - s["expected"]) / abs(s["z"])
    return v if np.isfinite(v) and v > 0 else None


def event_lines(f: Facts, ev: dict) -> list[str]:
    L = [f"• <b>{esc(ev['name'])}</b> — {tehran(ev['when'])}"]
    k = ev["key"]
    if not k:
        return L
    sg = sigma_units(f, k)
    size = f" (حدود {fmt_value(k, sg).replace('٪', ' واحد درصد') if k not in ('nonfarm_payrolls', 'initial_claims', 'jolts_openings') else fmt_value(k, sg)})" if sg else ""
    hot = "داغ‌تر" if k in INFLATION else "قوی‌تر" if LABOR.get(k, 1) > 0 else "بالاتر"
    L.append(f"   ↳ اگر {SHORT.get(k, esc(ind_name(k)))} یک واحد شگفتی{size} {hot} از انتظار بیاید: "
             f"{effect_pair(f, k, 1.0, '24h')}؛ اگر پایین‌تر بیاید، تقریباً برعکس.")
    sds = [f"{ASSET_FA[a]} ±{num(b['sd'], '{:.1f}')}٪" for a in ("BTC", "Gold")
           if (b := beta(f, k, a, "24h")) and b.get("sd")]
    if sds:
        L.append(f"   ↳ نوسان معمول ۲۴ ساعت بعد از این انتشار: {'، '.join(sds)}")
    return L


def macro_story(f: Facts, now: pd.Timestamp) -> str:
    t = theme_reads(f, now)
    L = [f"📖 <b>تحلیل روایی ({TITLE_TAG}) — تصویر کلان</b>", f"{jdate(now)} · داده‌ها تا {tehran(f.as_of or now)}", "",
         "<b>داستان در چند جمله</b>", story_sentence(t)]
    evs = upcoming(f, now)
    for s in (fed_sentence(f), rates_sentence(f, now), regime_sentence(f), asset_takeaways(f, t), decision_note(t, evs)):
        if s:
            L.append(s)
    if t["inf"]:
        L += ["", f"🔥 <b>تورم: {inflation_word(t['inflation'])}</b>", *release_lines(f, t["inf"], _inf_word)]
    if t["lab"]:
        L += ["", f"👷 <b>بازار کار: {labor_word(t['labor'])}</b>", *release_lines(f, t["lab"], _lab_word)]
    ml = market_lines(f, now)
    if ml:
        L += ["", "🏦 <b>نرخ‌ها، دلار و نقدینگی (۴ هفته‌ی اخیر)</b>", *ml]
    if evs:
        L += ["", "📅 <b>پیش رو (۱۴ روز آینده)</b>"]
        for e in evs[:4]:
            L += event_lines(f, e)
    L += ["", "راهنما: «اثر معمول» یعنی بیت‌کوین یا طلا در گذشته پس از عددی به همین اندازه به‌طور میانگین چقدر جابه‌جا "
          "شده‌اند. «یک واحد شگفتی» یعنی فاصله‌ی معمول عدد واقعی از انتظار."]
    return "\n".join(L)


def asset_takeaways(f: Facts, t: dict) -> str:
    """One sentence per asset: the net macro tilt in words."""
    bits = []
    for a in ("BTC", "Gold"):
        x = f.asset(a)
        er = x.get("expected_range") or {}
        d, sg = er.get("macro_drift_pct"), er.get("4w_sigma_pct")
        if d is None or not sg:
            continue
        w = "کمی مثبت" if d > 0.05 else "کمی منفی" if d < -0.05 else "خنثی"
        weight = "" if w == "خنثی" else " ولی کم‌وزن" if abs(d) < 0.3 * sg else " و پروزن"
        bits.append(f"برای {ASSET_FA[a]} {w}{weight} ({pct(d)} در ۴ هفته در برابر نوسان عادی ±{num(sg, '{:.1f}')}٪)")
    return ("برآیند کلان: " + "؛ ".join(bits) + ".") if bits else ""


# ───────────────────────────── per-asset advice ─────────────────────────────

TREND_FA = {"up": "صعودی", "down": "نزولی", "mixed": "بی‌جهت"}
EVENT_SHORT = {"BOS_UP": "شکست سقف قبلی", "BOS_DN": "شکست کف قبلی", "CHOCH_UP": "برگشت به صعود",
               "CHOCH_DN": "برگشت به نزول"}


def _dir(t: str | None) -> int:
    return 1 if t == "up" else -1 if t == "down" else 0


def levels(x: dict, mv: dict | None) -> tuple[float | None, float | None]:
    """Nearest resistance above and support below the price, from swings, the 20-day range and the watchlist."""
    p = x.get("price")
    if p is None:
        return None, None
    t = x.get("technical") or {}
    cands = []
    for tf in ("1d", "4h"):
        d = t.get(tf) or {}
        cands += [d.get("last_swing_high"), d.get("last_swing_low")]
    if mv:
        cands += [mv.get("hi"), mv.get("lo")]
    cands += [(w.get("support") or {}).get("price") for w in x.get("watchlist") or []]
    cands = [c for c in cands if c]
    atr = (t.get("1d") or {}).get("atr") or 0
    gap = max(0.003 * p, 0.5 * atr)  # a level closer than half a normal day is noise, not a decision point
    above = [c for c in cands if c > p + gap]
    below = [c for c in cands if c < p - gap]
    return (min(above) if above else None), (max(below) if below else None)


def scenarios(p: float, r: float | None, s: float | None, drift_pct: float, sigma_pct: float) -> dict | None:
    """End-of-4-weeks probabilities above the resistance, below the support and in between (log-normal)."""
    if not r or not s or not sigma_pct:
        return None
    sg, d = sigma_pct / 100, drift_pct / 100
    up = 1 - _phi((log(r / p) - d) / sg)
    dn = _phi((log(s / p) - d) / sg)
    return {"up": up, "down": dn, "base": max(0.0, 1 - up - dn), "higher": _phi(d / sg)}


def tech_story(name: str, t: dict) -> tuple[str, float]:
    """The trend story (daily = main trend, 4-hour = the current swing) and a score: + bullish, − bearish."""
    d1, h4 = t.get("1d") or {}, t.get("4h") or {}
    a, b = _dir(d1.get("trend")), _dir(h4.get("trend"))
    ev1, ev4 = d1.get("last_event"), h4.get("last_event")
    e4 = f" ({EVENT_SHORT[ev4]})" if ev4 in EVENT_SHORT else ""
    if a > 0 and b < 0:
        s = (f"روند اصلی (روزانه) {name} صعودی است، اما در نمودار ۴ ساعته اصلاح شروع شده{e4}. یعنی الان در یک "
             "«اصلاح داخل روند صعودی» هستیم؛ تا وقتی کف‌های اصلی حفظ شوند، این افت بیشتر فرصت است تا خطر.")
    elif a < 0 and b > 0:
        s = (f"روند اصلی (روزانه) {name} نزولی است و جهش اخیر در نمودار ۴ ساعته{e4} فعلاً یک «برگشت کوتاه داخل روند "
             "نزولی» حساب می‌شود، نه شروع صعود؛ مگر این‌که قیمت بالای مقاومت تثبیت شود.")
    elif a < 0 and ev4 == "CHOCH_UP":
        s = (f"روند اصلی (روزانه) {name} نزولی است، ولی نمودار ۴ ساعته اولین نشانه‌ی برگشت را داده (برگشت به صعود). "
             "این هنوز فقط یک نشانه است؛ تا قیمت بالای مقاومت تثبیت نشود، روند اصلی نزولی می‌ماند.")
    elif a > 0 and ev4 == "CHOCH_DN":
        s = (f"روند اصلی (روزانه) {name} صعودی است، ولی نمودار ۴ ساعته اولین نشانه‌ی ضعف را داده (برگشت به نزول). "
             "اگر کف نزدیک حفظ نشود، اصلاح عمیق‌تر می‌شود.")
    elif a > 0:
        s = f"روند روزانه و ۴ ساعته‌ی {name} هم‌سو و صعودی‌اند؛ خریداران کنترل بازار را دارند."
    elif a < 0:
        s = f"روند روزانه و ۴ ساعته‌ی {name} هم‌سو و نزولی‌اند؛ فروشندگان کنترل بازار را دارند."
    else:
        s = f"{name} روند روزانه‌ی روشنی ندارد و در یک محدوده نوسان می‌کند."
    score = a + 0.5 * b
    if a > 0 and ev1 in ("CHOCH_DN", "BOS_DN"):
        s += (f" هشدار: در نمودار روزانه هم آخرین کف شکسته شده ({EVENT_SHORT[ev1]})؛ اگر کف بعدی هم از دست برود، "
              "روند اصلی عوض می‌شود.")
        score -= 0.5
    elif a < 0 and ev1 in ("CHOCH_UP", "BOS_UP"):
        s += f" نکته: در نمودار روزانه آخرین سقف شکسته شده ({EVENT_SHORT[ev1]})؛ اولین نشانه‌ی جدی برگشت."
        score += 0.5
    rsi = d1.get("rsi")
    if rsi is not None and rsi <= 35:
        s += (f" RSI روزانه {num(rsi, '{:.0f}')} است و به اشباع فروش نزدیک شده؛ فروش بیشتر جای کمی دارد و احتمال "
              "یک برگشت کوتاه بالا می‌رود.")
    elif rsi is not None and rsi >= 65:
        s += f" RSI روزانه {num(rsi, '{:.0f}')} است و به اشباع خرید نزدیک شده؛ خرید در این نقطه پرریسک‌تر است."
    return s, score


def macro_forces(f: Facts, asset: str) -> list[tuple[str, float, bool]]:
    """(indicator, usual 4-week % effect now, against theory) for the indicators that matter most now."""
    med = (f.macro.get(asset) or {}).get("medium") or {}
    out = []
    for t in med.get("top") or []:
        b = beta(f, t["indicator"], asset, "4w")
        if not b or abs(t.get("contribution") or 0) < 1:
            continue
        out.append((t["indicator"], b["beta"] * 2 * t["state"],
                    against_theory(t["indicator"], asset, t.get("coef"), "state")))
    return out


def asset_advice(f: Facts, asset: str, now: pd.Timestamp) -> str:
    name, x = ASSET_FA[asset], f.asset(asset)
    p = x.get("price")
    er = x.get("expected_range") or {}
    sg, drift = er.get("4w_sigma_pct"), er.get("macro_drift_pct") or 0.0
    mv = moves.snapshot_stats(f.root, asset, now)
    r, s = levels(x, mv)
    L = [f"🧭 <b>مشاوره‌ی {TITLE_TAG} — {name}</b>",
         f"قیمت: {px(p, asset)} دلار{' (PAXG، نماینده‌ی طلا)' if asset == 'Gold' else ''}", ""]

    # 1. where we are
    L.append("<b>کجای بازاریم؟</b>")
    first = ""
    if mv and mv.get("chg7d") is not None:
        first = (f"{name} در ۷ روز گذشته {pct(mv['chg7d'])} و در ۲۴ ساعت اخیر {pct(mv['chg24'])} تغییر کرده؛ "
                 f"{moves.size_word(mv['chg24'], mv['atr_pct'])} برای یک روز. ")
    ts, tscore = tech_story(name, x.get("technical") or {})
    L.append(first + ts)
    if r or s:
        L.append(f"مقاومت نزدیک: {px(r, asset)} · حمایت نزدیک: {px(s, asset)}")

    # 2. macro weight
    L += ["", "<b>کلان چقدر وزن دارد؟</b>"]
    forces = macro_forces(f, asset)
    if sg:
        tilt = "کمی مثبت" if drift > 0.05 else "کمی منفی" if drift < -0.05 else "تقریباً خنثی"
        heavy = abs(drift) >= 0.3 * sg
        L.append(f"برآیند کلان برای {name} {tilt} است: حدود {pct(drift)} در ۴ هفته، در حالی که نوسان عادی {name} در همین مدت "
                 f"±{num(sg, '{:.1f}')}٪ است. " +
                 ("یعنی کلان الان عامل مهمی است و نباید خلافش موقعیت گرفت." if heavy else
                  "یعنی کلان فعلاً تعیین‌کننده نیست و قیمت بیشتر از تکنیکال و رویدادهای پیش رو اثر می‌گیرد."))
    sound = [(k, e) for k, e, c in forces if not c]
    odd = [(k, e) for k, e, c in forces if c]
    ups, dns = [t for t in sound if t[1] > 0][:3], [t for t in sound if t[1] < 0][:3]

    def flist(fs):
        return "، ".join(f"{esc(ind_name(k))} ({pct(e)})" for k, e in fs)
    if ups:
        L.append(f"🟢 به نفع افزایش: {flist(ups)}")
    if dns:
        L.append(f"🔴 به نفع کاهش: {flist(dns)}")
    if not ups and not dns:
        L.append(f"هیچ شاخص کلانی با رابطه‌ی منطقی الان فشار محسوسی روی {name} نمی‌آورد.")
    if odd:
        same = sum(e for _, e in odd) * drift > 0
        L.append(f"⚠️ کنار گذاشته شد: {flist(odd[:3])}. این رابطه‌ها در داده‌های اخیر دیده می‌شوند ولی خلاف منطق "
                 "اقتصادی‌اند (احتمالاً هم‌زمانی تصادفی)" +
                 ("؛ بخشی از همان برآیند کلان از همین‌ها می‌آید، پس آن هم کم‌اعتبارتر از ظاهرش است." if same else "."))

    # 3. scenarios
    sc = scenarios(p, r, s, drift, sg) if p else None
    if sc:
        L += ["", "<b>سناریوهای ۴ هفته‌ی آینده</b>",
              f"🟢 بالای {px(r, asset)}: احتمال {prob(sc['up'])}",
              f"⚪ بین {px(s, asset)} و {px(r, asset)}: احتمال {prob(sc['base'])}",
              f"🔴 زیر {px(s, asset)}: احتمال {prob(sc['down'])}",
              f"احتمال این‌که ۴ هفته‌ی دیگر قیمت بالاتر از الان باشد: {prob(sc['higher'])} — "
              + ("عملاً یک سکه‌ی شیر یا خط؛ برتری روشنی وجود ندارد." if abs(sc["higher"] - 0.5) < 0.05 else
                 "کفه کمی به سمت صعود است." if sc["higher"] > 0.5 else "کفه کمی به سمت نزول است.")]

    # 4. advice
    evs = upcoming(f, now, days=7)
    mdir = 1 if drift > 0.05 else -1 if drift < -0.05 else 0
    bias = tscore + (mdir if sg and abs(drift) >= 0.3 * sg else 0.5 * mdir)
    stance = 1 if bias >= 1 else -1 if bias <= -1 else 0
    L += ["", "<b>چه کنم؟</b>"]
    if stance > 0:
        hold = f"نگه داشتن منطقی است. حد ضرر ذهنی را زیر {px(s, asset)} بگذار؛ بسته شدن روزانه زیر آن یعنی داستان عوض شده."
        buy = (f"دنبال قیمت ندو. بهترین ورود، برگشت به نزدیکی {px(s, asset)} و واکنش مثبت در آن‌جاست؛"
               f" یا تثبیت روزانه بالای {px(r, asset)} برای تأیید ادامه‌ی صعود.")
    elif stance < 0:
        hold = (f"روند علیه توست. بخشی را سبک کن یا حد ضرر را نزدیک‌تر بیاور؛ تا وقتی قیمت بالای {px(r, asset)} "
                "تثبیت نشده، برگشت روند تأیید نمی‌شود.")
        buy = (f"فعلاً عجله نکن. یا صبر کن قیمت بالای {px(r, asset)} تثبیت شود، یا اگر به {px(s, asset)} رسید و آن‌جا "
               "نشانه‌ی برگشت داد، با حجم کم وارد شو.")
    else:
        hold = f"عجله‌ای برای فروش نیست ولی به موقعیتت اضافه هم نکن. اگر روزانه زیر {px(s, asset)} بسته شد، ریسک را کم کن."
        buy = (f"صبر کن تا بازار تکلیفش را روشن کند: تثبیت روزانه بالای {px(r, asset)} (تأیید صعود) یا رسیدن به "
               f"{px(s, asset)} و برگشت از آن. وسط این محدوده، نسبت سود به ضرر خوب نیست.")
    L.append(f"👤 اگر {name} داری: {hold}")
    L.append(f"🛒 اگر می‌خواهی بخری: {buy}")
    wl = (x.get("watchlist") or [None])[0]
    if wl and wl.get("entry_zone"):
        why = "، و کلان تأییدش نکرده" if "macro not aligned" in (wl.get("blocked_by") or []) else ""
        L.append(f"🔎 سیستم یک ستاپ {'خرید' if wl['direction'] == 'long' else 'فروش'} روی نمودار ۴ ساعته دیده "
                 f"(ورود ≈ {px(min(wl['entry_zone']), asset)}، حد ضرر ≈ {px(wl['stop_loss'], asset)}، هدف ۱ ≈ "
                 f"{px(wl['targets'][0], asset)})؛ اما در آزمون گذشته فقط {prob(wl.get('win_rate') or 0)} برد داشته{why}. "
                 "فقط برای اطلاع، نه پیشنهاد.")
    if evs:
        e = evs[0]
        b = beta(f, e["key"], asset, "24h") if e["key"] else None
        vol = f"؛ در ۲۴ ساعت بعد از آن، نوسان ±{num(b['sd'], '{:.1f}')}٪ برای {name} عادی است" if b and b.get("sd") else ""
        L.append(f"⚖️ ریسک: تا {esc(e['name'])} ({tehran(e['when'])}) حجم موقعیت را کوچک‌تر نگه دار{vol}.")
    ev_name = f"{esc(evs[0]['name'])} ({jdate(evs[0]['when'], year=False)})" if evs else None
    if stance > 0:
        tldr = f"زمینه مثبت است؛ نگه دار و برای خرید، منتظر اصلاح تا نزدیکی {px(s, asset)} بمان."
    elif stance < 0:
        tldr = f"زمینه منفی است؛ ریسک را کم کن و تا تثبیت بالای {px(r, asset)} عجله‌ای برای خرید نکن."
    else:
        tldr = (f"بازار بلاتکلیف است؛ تا شکست یکی از دو سطح {px(s, asset)} یا {px(r, asset)}"
                f"{' یا انتشار ' + ev_name if ev_name else ''} صبر کن.")
    L.insert(3, f"💡 <b>خلاصه برای تصمیم:</b> {tldr}")
    L.insert(4, "")
    conf = "کم" if stance == 0 or not sg or abs(drift) < 0.3 * sg else "متوسط"
    L.append(f"اطمینان این جمع‌بندی: {conf} — سیستم هنوز هیچ قاعده‌ی معاملاتی با برتری اثبات‌شده پیدا نکرده است.")
    return "\n".join(L)


def build(f: Facts, now: pd.Timestamp) -> list[str]:
    msgs = [macro_story(f, now), asset_advice(f, "BTC", now), asset_advice(f, "Gold", now)]
    msgs[-1] += "\n\n" + NOTE
    return msgs


def plain(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)

