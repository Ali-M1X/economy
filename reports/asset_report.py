"""Per-asset Persian Telegram reports: market overview, one self-contained message per asset (BTC, gold), the full
indicator-impact table and a glossary.

Every number comes from `reports.facts.Facts` (the job outputs) or from simple arithmetic on the snapshot's own series
(latest value, change vs the previous print); nothing is estimated here. Scores always carry their scale and a
plain-Persian reading. Tables are ASCII inside <pre> blocks so the columns stay aligned in Telegram; the Persian
indicator name sits on its own line above each data line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.fa import fa, ltr, word
from core.registry import load_registry
from notify.telegram import esc
from reports.facts import MEASURE_FA, Facts, channel_fa, ind_name, indicators_cfg, releases_cfg
from reports.messages import ASSET_FA, DISCLAIMER, PREPOS_STATUS_FA, dashboard_link, num, signal_card, tehran

SCORE_GATE = 15  # |medium Macro Score| a trade signal needs (jobs.signals.MACRO_THRESHOLD)
EDGE_TRADES = 30  # jobs.signals.MIN_EDGE_TRADES
TOP_ROWS = 8
CHUNK_ROWS = 14
HORIZONS = (("short", "۲۴ ساعت"), ("medium", "۴ هفته"), ("long", "۶ ماه"))
CONF_FA = {"high": "زیاد", "medium": "متوسط", "low": "کم"}
CONF_CODE = {"high": "H", "medium": "M", "low": "L"}
ICON = {"BTC": "₿", "Gold": "🥇"}
TICKER = {"philly_fed_activity": "PHILLY", "avg_hourly_earnings": "AHE", "net_liquidity_weekly": "NETLIQ",
          "dxy": "DXY", "fed_funds_eff_daily": "DFF"}
BASIS_CODE = {"mean3": "T", "mean4": "T", "prev": "P"}
LRM = "\u200e"
GLOSSARY_NOTE = "معنی اصطلاحات و ستون‌ها: پیام «راهنمای اصطلاحات» در پایان همین گزارش."


# ───────────────────────────── plain-language readings ─────────────────────────────

def _dir(v: float, up: str = "صعودی", down: str = "نزولی") -> str:
    return up if v > 0 else down


def score_reading(s: float | None) -> str:
    """One-line meaning of a Macro Score (−100 … +100)."""
    if s is None:
        return "داده‌ای نیست"
    a, d = abs(s), _dir(s)
    move = _dir(s, "افزایش", "کاهش")
    if a < 5:
        return "خنثی: شواهد کلان تقریباً در تعادل‌اند و جهتی نشان نمی‌دهند"
    if a < SCORE_GATE:
        return f"تمایل ضعیف {d}: کمی به نفع {move} قیمت، اما زیر آستانه‌ی ±۱۵ که سیستم برای تأیید سیگنال لازم دارد"
    if a < 35:
        return f"{d} متوسط: شواهد کلان به‌طور محسوس به نفع {move} قیمت است"
    if a < 60:
        return f"{d} قوی: بیشتر شاخص‌های مؤثر هم‌زمان به نفع {move} قیمت‌اند"
    return f"{d} بسیار قوی: تقریباً همه‌ی شواهد کلان در یک جهت‌اند"


def score_direction(s: float | None) -> str:
    if s is None:
        return "نامشخص"
    if abs(s) < 5:
        return "خنثی"
    return f"تمایل ضعیف {_dir(s)}" if abs(s) < SCORE_GATE else _dir(s)


def score_confidence(s: float | None) -> str:
    if s is None:
        return "—"
    return "کم" if abs(s) < SCORE_GATE else "متوسط" if abs(s) < 35 else "زیاد"


def coef_reading(k: float | None) -> str:
    """How strong a signed 0–10 impact coefficient is."""
    if k is None or not np.isfinite(k):
        return "ضریب ندارد"
    a = abs(k)
    return "تقریباً بی‌اثر" if a < 1 else "ضعیف" if a < 3 else "متوسط" if a < 6 else "قوی"


def z_reading(z: float) -> str:
    a = abs(z)
    return "تقریباً مطابق انتظار" if a < 0.5 else "کوچک" if a < 1 else "متوسط" if a < 2 else "بزرگ (نادر)"


def stance_reading(s: float) -> str:
    a = abs(s)
    side = "انقباضی (به نفع نرخ بالاتر)" if s > 0 else "انبساطی (به نفع نرخ پایین‌تر)"
    return "خنثی" if a < 15 else f"کمی {side}" if a < 35 else f"{side}" if a < 60 else f"به‌شدت {side}"


def basis_code(basis: str | None) -> str:
    b = str(basis or "")
    return next((c for k, c in BASIS_CODE.items() if k in b), "T")


def basis_fa(basis: str | None) -> str:
    b = str(basis or "")
    if "mean3" in b:
        return "نسبت به روند (میانگین ۳ انتشار قبلی)"
    if "mean4" in b:
        return "نسبت به روند (میانگین ۴ هفته‌ی قبل)"
    if "prev" in b:
        return "نسبت به مقدار قبلی"
    return "نسبت به روند"


# ───────────────────────────── indicator rows ─────────────────────────────

@dataclass
class Row:
    key: str
    name_fa: str
    ticker: str
    value: str
    change: str
    surprise: str      # "+1.2σT" (release surprise) / "+0.9σM" (4-week move) / "—"
    coef: float | None
    coef_tag: str      # "w" = 4 weeks, "d" = 24 h after a release
    effect: int        # current implied effect on the asset: +1 / −1 / 0
    conf: str          # high|medium|low|""
    x: float | None    # the move used for the effect (state or surprise z)


def _series(f: Facts, key: str) -> pd.Series:
    obs = f._obs()
    s = obs[obs["series_key"] == key].sort_values("date").set_index("date")["value"].astype(float)
    return s[~s.index.duplicated(keep="last")]


def _net_liquidity(f: Facts) -> pd.Series:
    b, t, r = _series(f, "fed_balance_sheet"), _series(f, "tga_weekly"), _series(f, "reverse_repo")
    if b.empty or t.empty or r.empty:
        return pd.Series(dtype=float)
    idx = b.index[b.index >= max(t.index.min(), r.index.min())]
    return (b.reindex(idx) / 1000 - t.reindex(idx, method="ffill") / 1000 - r.reindex(idx, method="ffill")).dropna()


def _measure(s: pd.Series, measure: str) -> pd.Series:
    if measure == "mom_pct":
        return s.pct_change() * 100
    if measure == "qoq_ann":
        return ((s / s.shift(1)) ** 4 - 1) * 100
    if measure in ("diff", "mom_change"):
        return s.diff()
    return s


def _fmt_level(v: float, units: str) -> str:
    if units == "%":
        return f"{v:.2f}%"
    if units == "pp":
        return f"{v:+.2f}pp"
    if units == "thousands":
        return f"{v / 1000:.2f}M"
    if units == "persons":
        return f"{v / 1000:.0f}k"
    if units == "USD mn":
        return f"${v / 1e6:.2f}T"
    if units == "USD bn":
        return f"${v / 1e3:.2f}T"
    if units.startswith("USD/"):
        return f"${v:,.2f}"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def _fmt_change(d: float, units: str, how: str) -> str:
    if how == "logdiff":
        return f"{d:+.1f}%"
    if units in ("%", "pp"):
        return f"{d:+.2f}pp"
    if units == "thousands":
        return f"{d:+,.0f}k"
    if units == "persons":
        return f"{d / 1000:+.0f}k"
    return f"{d:+,.0f}" if abs(d) >= 1000 else f"{d:+.2f}"


def _value_change(f: Facts, key: str, rel: dict | None, st: dict | None, meta) -> tuple[str, str]:
    s = _net_liquidity(f) if key == "net_liquidity_weekly" else _series(f, key)
    if len(s) < 2:
        return "—", "—"
    units = meta.units if meta else ("USD bn" if key == "net_liquidity_weekly" else "")
    if rel is not None:  # a release indicator: its own measure (e.g. CPI m/m %), change vs the previous print
        m = rel["measure"]
        ms = _measure(s, m).dropna()
        if len(ms) < 2:
            return "—", "—"
        v, p = float(ms.iloc[-1]), float(ms.iloc[-2])
        if m == "mom_pct":
            return f"{v:+.2f}%/m", f"{v - p:+.2f}pp"
        if m == "diff":
            return f"{v:+,.0f}k", f"{v - p:+,.0f}k"
        return _fmt_level(v, units), _fmt_change(v - p, units, "diff")
    how = (st or {}).get("change", "diff")
    freq = meta.frequency if meta else "W"
    last, t = float(s.iloc[-1]), s.index[-1]
    if freq == "D":  # daily: change over the same 4 weeks the Macro Score uses
        prev = s[s.index <= t - pd.Timedelta(days=28)]
        if prev.empty:
            return _fmt_level(last, units), "—"
        p = float(prev.iloc[-1])
    else:
        p = float(s.iloc[-2])
    d = (np.log(last / p) * 100) if how == "logdiff" and last > 0 and p > 0 else last - p
    return _fmt_level(last, units), _fmt_change(d, units, how)


def _coef(f: Facts, key: str, asset: str, horizon: str) -> tuple[float | None, str]:
    """Signed 0–10 coefficient; for BTC the average of the full sample and the last 3 years (as in the Macro Score)."""
    full = f.coef(key, asset, horizon, "full")
    if not full:
        return None, ""
    vals = [full["coefficient"]]
    if asset == "BTC":
        rec = f.coef(key, asset, horizon, "recent")
        if rec and rec.get("coefficient") is not None and np.isfinite(rec["coefficient"]):
            vals.append(rec["coefficient"])
    return float(np.mean(vals)), str(full.get("confidence") or "low")


def _state(f: Facts, asset: str, key: str) -> float | None:
    med = (f.macro.get(asset) or {}).get("medium") or {}
    if key in (med.get("state") or {}):
        return float(med["state"][key])
    for t in med.get("top", []):
        if t["indicator"] == key:
            return float(t["state"])
    return None


def _surprise(f: Facts, key: str) -> dict | None:
    if key in (f.surprises or {}):
        return f.surprises[key]
    return next((r for r in f.releases if r["indicator"] == key), None)


def indicator_rows(f: Facts, asset: str) -> list[Row]:
    cfg = indicators_cfg()
    rels = {r["key"]: r for r in cfg["releases"]}
    states = {s["key"]: s for s in cfg["state"]}
    reg = load_registry()
    rows = []
    for key in dict.fromkeys([*rels, *states]):
        meta = reg.get(key)
        in_state = key in states
        k, conf = _coef(f, key, asset, "4w" if in_state else "24h")
        sp = _surprise(f, key) if key in rels else None
        st = _state(f, asset, key) if in_state else None
        if sp is not None:
            surprise = f"{sp['z']:+.1f}σ{basis_code(sp.get('basis'))}"
        elif st is not None:
            surprise = (f"{'≥' if abs(st) >= 1 else ''}{2 * st:+.1f}σM")
        else:
            surprise = "—"
        x = st if in_state else (sp["z"] if sp else None)
        eff = 0
        if k is not None and x is not None and abs(k) >= 0.5 and abs(x) >= 0.05:
            eff = int(np.sign(k * x))
        value, change = _value_change(f, key, rels.get(key), states.get(key), meta)
        ticker = TICKER.get(key) or (meta.source_id if meta else key).lstrip("^")
        rows.append(Row(key, ind_name(key), ticker, value, change, surprise, k, "w" if in_state else "d", eff,
                        conf, x))
    return sorted(rows, key=lambda r: -abs(r.coef) if r.coef is not None else 1)


TABLE_WIDTH = 32  # characters per line; fits Telegram's monospace font on a phone without wrapping
TICKER_W = 8


def _pre(lines: list[str]) -> str:
    # a left-to-right mark opens every line, so clients that pick each line's direction from its first strong
    # character keep the columns left-to-right even under a right-to-left message
    return "<pre>" + esc("\n".join(LRM + ln for ln in lines)) + "</pre>"


def _table(head: str, body: list[str]) -> list[str]:
    """Header, a dashed divider, then the rows — split into stacked blocks of CHUNK_ROWS so a Telegram message never
    cuts through a block (the splitter only cuts between blocks)."""
    rule = "-" * len(head)
    return [_pre([head, rule, *body[c:c + CHUNK_ROWS]]) for c in range(0, len(body), CHUNK_ROWS)]


_SIGNED_ZERO = re.compile(r"[+-](?=0(?:\.0+)?(?![\d.,]))")


def nz(cell: str) -> str:
    """Drop the sign of a value that rounds to zero ("-0.0M" → "0.0M")."""
    return _SIGNED_ZERO.sub("", cell)


def effect_table(rows: list[Row]) -> list[str]:
    """Table 1 — what decides: coefficient, current move, confidence, implied direction now."""
    head = f"{'Ticker':<{TICKER_W}}{'Coef':>7}{'Move':>7}{'C':>2}{'D':>2}"
    body = [f"{r.ticker[:TICKER_W]:<{TICKER_W}}{nz('—' if r.coef is None else f'{r.coef:+.1f}{r.coef_tag}'):>7}"
            f"{nz(r.surprise.replace('σ', '')):>7}{CONF_CODE.get(r.conf, '-'):>2}{ {1: '▲', -1: '▼', 0: '·'}[r.effect]:>2}"
            for r in rows]
    return _table(head, body)


def value_table(rows: list[Row]) -> list[str]:
    """Table 2 — the latest figures behind the moves."""
    head = f"{'Ticker':<{TICKER_W}}{'Value':>10}{'Chg':>9}"
    body = [f"{r.ticker[:TICKER_W]:<{TICKER_W}}{nz(r.value):>10}{nz(r.change):>9}" for r in rows]
    return _table(head, body)


def names_key(rows: list[Row]) -> str:
    return "\n".join(f"• {esc(r.name_fa)} — {esc(r.ticker)}" for r in rows)


EFFECT_LEGEND = "\n".join([
    "• Coef = ضریب اثر روی همین دارایی، از −۱۰ تا +۱۰ (w: اثر ۴ هفته، d: اثر ۲۴ ساعت پس از انتشار)",
    "• Move = حرکت فعلی بر حسب σ — T: شگفتی نسبت به روند، P: نسبت به مقدار قبلی، M: حرکت ۴ هفته (اجماع رایگان در دسترس نیست)",
    "• C = اطمینان ضریب (H زیاد، M متوسط، L کم) · D = اثر فعلی (▲ صعودی، ▼ نزولی، · ناچیز)"])
VALUE_LEGEND = "• Value = آخرین مقدار · Chg = تغییر نسبت به انتشار قبلی (داده‌های روزانه: نسبت به ۴ هفته قبل)"


def tables_section(rows: list[Row], asset: str, title: str) -> str:
    """Heading → legend → table, for both tables, then the name key; blocks separated by blank lines."""
    name = ASSET_FA[asset]
    parts = [title,
             f"<b>جدول ۱ — اثر بر {name}</b> (مرتب بر اساس قدر مطلق ضریب)", EFFECT_LEGEND, *effect_table(rows),
             "<b>جدول ۲ — آخرین مقدارها</b>", VALUE_LEGEND, *value_table(rows),
             "<b>نام شاخص‌ها</b>\n" + names_key(rows)]
    return "\n\n".join(parts)


# ───────────────────────────── causal chain (plain sentences) ─────────────────────────────

def _move_phrase(state: float) -> str:
    verb = "بالا رفته" if state > 0 else "پایین آمده"
    m = 2 * abs(state)  # the state is the 4-week change in σ, clipped to ±2 and halved
    if m >= 2:
        return f"به‌طور غیرعادی {verb} است (دست‌کم ۲ برابر نوسان معمول ۴ هفته)"
    size = "بیش از حد معمول" if m >= 1.5 else "در حد معمول" if m >= 0.7 else "کمی"
    return f"{size} {verb} است ({num(m, '{:.1f}')} برابر نوسان معمول ۴ هفته)"


def val(v: float) -> str:
    """A released value without noise decimals."""
    return num(v, "{:,.0f}" if abs(v) >= 100 else "{:,.2f}")


def chain_sentences(f: Facts, asset: str, now: pd.Timestamp) -> list[str]:
    """Cause → effect sentences from the numbers behind the score (template; also the fallback for Claude's text)."""
    cfg = indicators_cfg()
    ch = channel_fa(cfg)
    chans = {i["key"]: i.get("channels", []) for sec in ("releases", "state") for i in cfg[sec]}
    name = ASSET_FA[asset]
    rel_measure = {r["key"]: r["measure"] for r in cfg["releases"]}
    L = []
    for r in f.releases:
        if pd.Timestamp(r["release_utc"]) < now - pd.Timedelta(hours=24):
            continue
        k, conf = _coef(f, r["indicator"], asset, "24h")
        hot = "بالاتر" if r["z"] > 0 else "پایین‌تر"
        meas = MEASURE_FA.get(rel_measure.get(r["indicator"], ""), "")
        nm = esc(ind_name(r["indicator"]))
        s = (f"{nm}{f' ({meas})' if meas else ''} {hot} از انتظار آمد: {val(r['actual'])} در برابر {val(r['expected'])} "
             f"({basis_fa(r.get('basis'))})، یعنی شگفتی {num(r['z'], '{:+.1f}σ')} ({z_reading(r['z'])}).")
        if k is not None and abs(k) >= 1:
            s += (f" در گذشته عدد بالاتر از انتظار {nm} برای {name} در ۲۴ ساعت بعد {'مثبت' if k > 0 else 'منفی'} بوده است "
                  f"(ضریب {num(k, '{:.1f}', sign=True)} از ±۱۰، {coef_reading(k)}، اطمینان {CONF_FA.get(conf, conf)})؛ "
                  f"پس این عدد {hot} برای {name} {'صعودی' if k * r['z'] > 0 else 'نزولی'} ارزیابی می‌شود.")
        else:
            s += f" اثر تاریخی این شگفتی بر {name} در ۲۴ ساعت بعد ناچیز بوده است."
        L.append(s)
    med = (f.macro.get(asset) or {}).get("medium") or {}
    for t in med.get("top", [])[:4]:
        key, c = t["indicator"], t["contribution"]
        if abs(c) < 0.5:
            continue
        k, conf = _coef(f, key, asset, "4w")
        path = "، ".join(f"«{esc(ch.get(x, x))}»" for x in chans.get(key, []))
        hist = "افزایش" if (k or t["coef"]) > 0 else "کاهش"
        s = (f"{esc(ind_name(key))} در ۴ هفته‌ی اخیر {_move_phrase(t['state'])}."
             + (f" این تغییر از مسیر {path} اثر می‌گذارد." if path else "")
             + f" در گذشته بالا رفتن این شاخص با {hist} {name} در ۴ هفته‌ی بعد همراه بوده است"
             + (f" (ضریب {num(k, '{:.1f}', sign=True)} از ±۱۰، {coef_reading(k)}، اطمینان {CONF_FA.get(conf, conf)})" if k is not None else "")
             + f"؛ پس این عامل امتیاز ۴ هفته را {num(abs(c), '{:.1f}')} واحد {'بالا می‌برد' if c > 0 else 'پایین می‌آورد'}.")
        L.append(s)
    if not L:
        L.append(f"هیچ شاخصی در حال حاضر سهم محسوسی در امتیاز {name} ندارد؛ برآیند کلان نزدیک صفر است.")
    return L


# ───────────────────────────── sections ─────────────────────────────

def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def price_block(f: Facts, asset: str) -> list[str]:
    x = f.asset(asset)
    er = x.get("expected_range") or {}
    L = ["<b>💵 قیمت</b>"]
    fmt = "{:,.0f}" if asset == "BTC" else "{:,.2f}"
    if asset == "Gold":
        gc = f.gold_futures()
        px = x.get("price")
        if gc:
            L.append(f"• طلای واقعی — COMEX GC=F (قرارداد آتی نزدیک، معیار قیمت طلا؛ با تأخیر حدود ۱۵ دقیقه): "
                     f"<b>{num(gc[0], fmt)}</b> دلار/اونس ({tehran(gc[1])})")
        if px is not None:
            prem = f" — {num(abs(px / gc[0] - 1) * 100, '{:.2f}')}٪ {'بالاتر' if px > gc[0] else 'پایین‌تر'} از GC=F" if gc else ""
            bar = ((x.get("technical") or {}).get("1h") or {}).get("bar")
            when = f" ({tehran(bar)})" if bar else ""
            if gc and bar and abs(_utc(gc[1]) - _utc(bar)) > pd.Timedelta(minutes=30):
                prem += "؛ زمان دو قیمت یکی نیست، پس بخشی از این فاصله از اختلاف زمان است"
            L.append(f"• PAXG — توکن طلا (هر توکن = یک اونس)، نماینده‌ی بلادرنگ ۲۴ ساعته: <b>{num(px, fmt)}</b>{when}{prem}")
        L.append("• مبنای محاسبات: امتیاز کلان ۴ هفته و ۶ ماه و ضریب‌های آن‌ها ← GC=F (روزانه). ضریب‌های ۲۴ ساعته، "
                 "سطوح تکنیکال، دامنه‌های محتمل، سیگنال، بک‌تست و پیش‌موقعیت‌گیری ← PAXG (قیمت بلادرنگ). "
                 "قیمت اسپات XAUUSD منبع رایگان ندارد.")
    else:
        L.append(f"• قیمت فعلی: <b>{num(x.get('price'), fmt)}</b> دلار")
    base = " (بر پایه‌ی PAXG)" if asset == "Gold" else ""
    if er.get("1d"):
        L.append(f"• دامنه‌ی محتمل یک روز{base}: {num(min(er['1d']), fmt)} تا {num(max(er['1d']), fmt)} "
                 "(قیمت ± میانگین دامنه‌ی نوسان روزانه، ATR)")
    if er.get("4w_1sigma"):
        L.append(f"• دامنه‌ی محتمل ۴ هفته{base}: {num(min(er['4w_1sigma']), fmt)} تا {num(max(er['4w_1sigma']), fmt)} "
                 f"(±۱σ ≈ ۶۸٪ احتمال؛ نوسان ۴ هفته {num(er.get('4w_sigma_pct'), '{:.1f}')}٪)")
    return L


def scores_section(f: Facts, asset: str) -> list[str]:
    m = f.macro.get(asset) or {}
    L = ["<b>🧮 امتیاز کلان</b> — بازه‌ی −۱۰۰ (کاملاً نزولی) تا +۱۰۰ (کاملاً صعودی)؛ صفر = خنثی؛ آستانه‌ی سیگنال ±۱۵"]
    for b, h in HORIZONS:
        sc = (m.get(b) or {}).get("score")
        if b == "short" and sc is None:
            L.append(f"• {h}: — (در ۲۴ ساعت اخیر انتشار مهمی نبوده؛ این افق فقط از شگفتی‌های تازه ساخته می‌شود)")
            continue
        L.append(f"• {h}: <b>{num(sc, '{:.0f}', sign=True)}</b> از ±۱۰۰ → {score_reading(sc)}")
    return L


def signals_section(f: Facts, asset: str, names: dict | None) -> list[str]:
    x = f.asset(asset)
    med = (x.get("macro") or {}).get("medium")
    L = ["<b>🎯 سیگنال معاملاتی</b>" + (" (بر پایه‌ی PAXG)" if asset == "Gold" else "")]
    for s in x.get("signals", []):
        L += signal_card(s, asset, blocked=False)
    for s in x.get("watchlist", []):
        L += ["", *signal_card(s, asset, blocked=True)]
    if x.get("signals"):
        return L
    summ = (x.get("backtest") or {}).get("summary") or {}
    if not x.get("watchlist"):
        L.append("سیگنالی صادر نشد. به زبان ساده:")
        L.append("• قاعده‌ی تکنیکال سیستم (ورود در پولبک روند ۴ ساعته، هم‌جهت با روند روزانه) در کندل‌های اخیر موقعیت ورودی نداده است.")
    else:
        L.append("\nسیگنال صادر نشد؛ شرط‌ها:")
    if med is None or abs(med) < SCORE_GATE:
        L.append(f"• شرط کلان: امتیاز کلان ۴ هفته باید دست‌کم ±۱۵ و هم‌جهت معامله باشد؛ الان {num(med, '{:.0f}', sign=True)} "
                 "است، پس حتی با ستاپ تکنیکال هم سیگنال مسدود می‌شد.")
    else:
        L.append(f"• شرط کلان برقرار است: امتیاز ۴ هفته {num(med, '{:.0f}', sign=True)} فقط معامله‌های "
                 f"{'خرید' if med > 0 else 'فروش'} را مجاز می‌کند.")
    for side, fa_side in (("long", "خرید"), ("short", "فروش")):
        s = summ.get(side)
        if not s:
            continue
        n, r = s.get("n_trades", 0), s.get("avg_r")
        ok = n >= EDGE_TRADES and (r or -1) > 0
        verdict = ("برقرار" + (" ولی ناچیز" if r is not None and r < 0.05 else "")) if ok else \
            ("برقرار نیست — این قاعده در گذشته زیان‌ده بوده" if r is not None and r <= 0 else "برقرار نیست — نمونه کم است")
        L.append(f"• برتری اثبات‌شده برای {fa_side}: {num(n, '{:,.0f}')} معامله‌ی خارج از نمونه، میانگین "
                 f"{num(r, '{:+.2f}R')} → {verdict}")
    L.append("قاعده: سیگنال فقط وقتی ارسال می‌شود که همان قاعده در بک‌تست خارج از نمونه، پس از کارمزد و لغزش، "
             "میانگین R بزرگ‌تر از صفر در دست‌کم ۳۰ معامله داشته باشد.")
    return L


def prepos_section(f: Facts, asset: str) -> list[str]:
    rows = [r for r in (f.prepos or {}).get("live", []) if r.get("asset") == asset]
    if not rows:
        return []
    L = ["<b>⏱ پیش‌موقعیت‌گیری پیش از انتشارها</b>"]
    for r in sorted(rows, key=lambda r: r["release_utc"]):
        head = f"• {esc(r.get('name_fa') or r['release'])} ({tehran(r['release_utc'])}): "
        st, sc = r.get("status"), r.get("score")
        bt = ((f.prepos.get("backtest") or {}).get(f"{r['release']}:{asset}") or {})
        if st == "signal":
            L.append(head + "سیگنال فعال — در پیام جداگانه ارسال شده است.")
        elif st == "watchlist":
            why = "؛ ".join(esc(fa(x)) for x in bt.get("blocked_by", [])) or "برتری خارج از نمونه اثبات نشده"
            L.append(head + f"بازار پیش از انتشار نشانه‌ی موقعیت‌گیری {word(r.get('direction'))} داده "
                     f"(امتیاز {num(r.get('trigger_score', sc), '{:.2f}', sign=True)}، آستانه ±۱٫۵)، اما این ترکیب در "
                     f"بک‌تست برتری اثبات‌شده ندارد ({why})؛ سیگنالی ارسال نمی‌شود.")
        elif st == "quiet":
            L.append(head + f"نشانه‌ی معناداری از موقعیت‌گیری نیست (امتیاز {num(sc, '{:.2f}', sign=True)}، آستانه ±۱٫۵).")
        elif st == "waiting":
            L.append(head + f"پنجره‌ی بررسی {num(r.get('window_h'), '{:.0f}')} ساعت پیش از انتشار، از {tehran(r['window_opens'])} باز می‌شود.")
        else:
            L.append(head + PREPOS_STATUS_FA.get(st, str(st)))
        if st in ("quiet", "waiting") and bt and not bt.get("edge"):
            L[-1] += " این ترکیب در بک‌تست برتری اثبات‌شده ندارد؛ پس حتی با نشانه هم سیگنالی ارسال نمی‌شود."
    return L


def upcoming_events(f: Facts, asset: str, now: pd.Timestamp, days: int = 7, n: int = 3) -> list[str]:
    c = f.calendar
    if c.empty:
        return []
    up = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=days))]
    up = up[up.get("importance", 5) >= 4] if "importance" in up else up
    rel = releases_cfg()
    rcfg = {r["key"] for r in indicators_cfg()["releases"]}
    out = []
    for e in up.drop_duplicates("event_id").sort_values("scheduled_utc").head(n).itertuples():
        r = rel.get(e.release_id, {})
        keys = [k for k in r.get("series_keys", []) if k in rcfg]
        ks = [(k, _coef(f, k, asset, "24h")[0]) for k in keys]
        ks = [(k, v) for k, v in ks if v is not None]
        line = f"{esc(r.get('name_fa', e.release_id))} — {tehran(e.scheduled_utc)}"
        if ks:
            k, v = max(ks, key=lambda kv: abs(kv[1]))
            line += (f"؛ ضریب اثر شگفتی آن ({esc(ind_name(k))}) بر {ASSET_FA[asset]} در ۲۴ ساعت: "
                     f"{num(v, '{:.1f}', sign=True)} از ±۱۰ ({coef_reading(v)})")
        out.append(line)
    return out


def invalidation(x: dict, med: float | None, fmt: str, base: str) -> list[str]:
    """Price levels that would invalidate / change the 4-week conclusion, relative to where the price is now."""
    px = x.get("price")
    d1, h4 = (x.get("technical") or {}).get("1d") or {}, (x.get("technical") or {}).get("4h") or {}
    hi, lo = d1.get("last_swing_high"), d1.get("last_swing_low")
    hi4, lo4 = h4.get("last_swing_high"), h4.get("last_swing_low")
    if px is None or hi is None or lo is None:
        return ["• سطح ابطال قیمتی در دسترس نیست؛ فقط تغییر علامت امتیاز ۴ هفته نتیجه را عوض می‌کند."]
    lvl = lambda v, what: f"{num(v, fmt)} ({what}{base})"  # noqa: E731
    if med is not None and med >= 5:
        if px < lo:
            return [f"• هشدار: قیمت هم‌اکنون زیر آخرین کف سوئینگ روزانه {lvl(lo, 'کف سوئینگ روزانه')} است؛ یعنی قیمت با "
                    f"تمایل صعودی کلان هم‌جهت نیست. تا بسته‌شدن روزانه‌ی دوباره بالای آن، نتیجه‌ی صعودی تأیید نمی‌شود."]
        out = f"• این نتیجه باطل می‌شود اگر: قیمت روزانه زیر {lvl(lo, 'آخرین کف سوئینگ روزانه')} بسته شود"
        if lo4 and lo < lo4 < px:
            out += f"؛ هشدار زودتر: شکست {lvl(lo4, 'کف سوئینگ ۴ ساعته')}"
        return [out + "؛ یا امتیاز ۴ هفته منفی شود."]
    if med is not None and med <= -5:
        if px > hi:
            return [f"• هشدار: قیمت هم‌اکنون بالای آخرین سقف سوئینگ روزانه {lvl(hi, 'سقف سوئینگ روزانه')} است؛ یعنی قیمت با "
                    f"تمایل نزولی کلان هم‌جهت نیست. تا بسته‌شدن روزانه‌ی دوباره زیر آن، نتیجه‌ی نزولی تأیید نمی‌شود."]
        out = f"• این نتیجه باطل می‌شود اگر: قیمت روزانه بالای {lvl(hi, 'آخرین سقف سوئینگ روزانه')} بسته شود"
        if hi4 and px < hi4 < hi:
            out += f"؛ هشدار زودتر: شکست {lvl(hi4, 'سقف سوئینگ ۴ ساعته')}"
        return [out + "؛ یا امتیاز ۴ هفته مثبت شود."]
    if px < lo:
        return [f"• قیمت هم‌اکنون زیر آخرین کف سوئینگ روزانه {lvl(lo, 'کف سوئینگ روزانه')} است، پس ساختار قیمت نزولی است "
                f"در حالی که شواهد کلان خنثی‌اند. بسته‌شدن روزانه‌ی دوباره بالای آن این نشانه را خنثی می‌کند، بسته‌شدن بالای "
                f"{lvl(hi, 'سقف سوئینگ روزانه')} نشانه‌ی صعودی است؛ عبور امتیاز ۴ هفته از ±۱۵ هم جهت کلان را تعیین می‌کند."]
    if px > hi:
        return [f"• قیمت هم‌اکنون بالای آخرین سقف سوئینگ روزانه {lvl(hi, 'سقف سوئینگ روزانه')} است، پس ساختار قیمت صعودی "
                f"است در حالی که شواهد کلان خنثی‌اند. بسته‌شدن روزانه‌ی دوباره زیر آن این نشانه را خنثی می‌کند، بسته‌شدن زیر "
                f"{lvl(lo, 'کف سوئینگ روزانه')} نشانه‌ی نزولی است؛ عبور امتیاز ۴ هفته از ±۱۵ هم جهت کلان را تعیین می‌کند."]
    return [f"• جهت‌دار شدن: بسته‌شدن روزانه بالای {lvl(hi, 'سقف سوئینگ روزانه')} نشانه‌ی صعودی و زیر "
            f"{lvl(lo, 'کف سوئینگ روزانه')} نشانه‌ی نزولی است؛ یا عبور امتیاز ۴ هفته از ±۱۵."]


def conclusion(f: Facts, asset: str, now: pd.Timestamp) -> list[str]:
    x = f.asset(asset)
    m = f.macro.get(asset) or {}
    name = ASSET_FA[asset]
    fmt = "{:,.0f}" if asset == "BTC" else "{:,.2f}"
    L = [f"<b>✅ جمع‌بندی {name}</b>"]
    for b, h in HORIZONS:
        sc = (m.get(b) or {}).get("score")
        if b == "short" and sc is None:
            L.append(f"• {h}: جهت کلان ندارد (انتشار تازه‌ای نبوده)؛ حرکت کوتاه‌مدت را قیمت و رویدادهای پیش‌رو تعیین می‌کنند.")
            continue
        L.append(f"• {h}: <b>{score_direction(sc)}</b> ({num(sc, '{:.0f}', sign=True)} از ±۱۰۰) · اطمینان {score_confidence(sc)}")
    top = ((m.get("medium") or {}).get("top") or [])
    if top:
        t = top[0]
        L.append(f"• مهم‌ترین عامل الان: <b>{esc(ind_name(t['indicator']))}</b> — به‌تنهایی "
                 f"{num(t['contribution'], '{:.1f}', sign=True)} واحد از امتیاز ۴ هفته "
                 f"({'به نفع افزایش' if t['contribution'] > 0 else 'به نفع کاهش'} قیمت).")
    med = (m.get("medium") or {}).get("score")
    L += invalidation(x, med, fmt, "، قیمت PAXG" if asset == "Gold" else "")
    ev = upcoming_events(f, asset, now)
    if ev:
        L.append("• رویدادهایی که می‌توانند نتیجه را عوض کنند:")
        L += [f"   – {e}" for e in ev]
    return L


# ───────────────────────────── messages ─────────────────────────────

def asset_message(f: Facts, asset: str, now: pd.Timestamp, chain: list[str] | None = None, trigger: str | None = None,
                  names: dict | None = None) -> str:
    rows = indicator_rows(f, asset)
    L = [f"{ICON[asset]} <b>گزارش {ASSET_FA[asset]} ({'BTC' if asset == 'BTC' else 'طلا / XAU'})</b> — {tehran(now)} (تهران)"]
    if trigger:
        L.append(f"علت گزارش: {esc(trigger)}")
    L += ["", *price_block(f, asset), "", *scores_section(f, asset)]
    L += ["", "<b>🔗 زنجیره‌ی علّی — چرا این امتیاز</b>"]
    L += [f"{i}. {s}" for i, s in enumerate(chain or chain_sentences(f, asset, now), 1)]
    shown = [r for r in rows if r.coef is not None][:TOP_ROWS]
    title = (f"<b>📋 مؤثرترین شاخص‌ها برای {ASSET_FA[asset]}</b> — {num(len(shown), '{:.0f}')} از "
             f"{num(len(rows), '{:.0f}')} شاخص؛ جدول کامل در پیام بعد")
    out = "\n".join(L) + "\n\n" + tables_section(shown, asset, title) + "\n"
    L = ["", *signals_section(f, asset, names)]
    pp = prepos_section(f, asset)
    if pp:
        L += ["", *pp]
    L += ["", *conclusion(f, asset, now), "", GLOSSARY_NOTE]
    return out + "\n".join(L) + dashboard_link() + "\n\n" + DISCLAIMER


def table_message(f: Facts, asset: str) -> str:
    rows = indicator_rows(f, asset)
    title = f"📋 <b>جدول کامل اثر شاخص‌ها بر {ASSET_FA[asset]}</b> — {num(len(rows), '{:.0f}')} شاخص"
    return tables_section(rows, asset, title) + "\n\n" + DISCLAIMER


def overview_message(f: Facts, now: pd.Timestamp, trigger: str | None = None, weekly: bool = False) -> str:
    s = f.state
    title = "🗓 <b>خلاصه‌ی هفتگی — نمای کلی بازار</b>" if weekly else "🌐 <b>نمای کلی بازار</b>"
    L = [f"{title} — {tehran(now)} (تهران)"]
    if trigger:
        L.append(f"علت گزارش: {esc(trigger)}")
    L += ["", "<b>زمینه‌ی مشترک بیت‌کوین و طلا</b>"]
    reg = s.get("regime") or {}
    if reg.get("regime"):
        p = (reg.get("probabilities") or {}).get(reg["regime"], 0)
        L.append(f"🧭 فاز اقتصادی: <b>{esc(word(reg['regime']))}</b> — احتمال {num(p * 100, '{:.0f}')}٪ "
                 "(مدل ۴ فاز: بهبود، رونق، اوج، رکود؛ از رشد، تورم و نقدینگی)")
        for t in (reg.get("triggers") or [])[:2]:
            L.append(f"   {esc(fa(t))}")
    st = (s.get("fed_stance") or {}).get("score")
    if st is not None:
        L.append(f"🏦 موضع فدرال: <b>{num(st, '{:.0f}', sign=True)}</b> از بازه‌ی −۱۰۰ (کاملاً انبساطی) تا +۱۰۰ "
                 f"(کاملاً انقباضی) → {stance_reading(st)}")
    fw = s.get("fedwatch_next") or {}
    if fw.get("meeting"):
        pct = {k: num((fw.get(k) or 0) * 100, "{:.0f}") for k in ("p_cut", "p_hold", "p_hike")}
        L.append(f"📈 جلسه‌ی بعدی FOMC ({ltr(esc(fw['meeting']))}): احتمال کاهش نرخ {pct['p_cut']}٪ · "
                 f"بدون تغییر {pct['p_hold']}٪ · افزایش {pct['p_hike']}٪ (از قیمت قراردادهای آتی وجوه فدرال)")
    since = now - pd.Timedelta(days=7 if weekly else 1)
    rel = [r for r in f.releases if pd.Timestamp(r["release_utc"]) >= since]
    if rel:
        L += ["", f"<b>چه منتشر شد ({'۷ روز' if weekly else '۲۴ ساعت'} اخیر)</b>"]
        for r in rel:
            hot = "بالاتر از انتظار 🔼" if r["z"] > 0 else "پایین‌تر از انتظار 🔽"
            L.append(f"• {esc(ind_name(r['indicator']))} ({ltr(esc(r['obs_date']))}): {val(r['actual'])} در برابر "
                     f"{val(r['expected'])} ({basis_fa(r.get('basis'))}) — {hot}، شگفتی "
                     f"{num(r['z'], '{:+.1f}σ')} ({z_reading(r['z'])})")
    news = f.important_news(since)
    if not news.empty:
        L += ["", "<b>خبرهای مهم</b>"]
        L += [f"• {esc(r.title_fa)} ({'★' * int(r.importance)})" for r in news.head(5).itertuples()]
    c = f.calendar
    if not c.empty:
        nxt = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=7))]
        if not nxt.empty:
            relc = releases_cfg()
            L += ["", "<b>تقویم ۷ روز آینده</b>"]
            for e in nxt.drop_duplicates("event_id").sort_values("scheduled_utc").itertuples():
                imp = int(getattr(e, "importance", 0) or 0)
                L.append(f"• {tehran(e.scheduled_utc)} — {esc(relc.get(e.release_id, {}).get('name_fa', e.release_id))} "
                         f"{'★' * imp}")
    L += ["", "پیام‌های بعدی: گزارش کامل بیت‌کوین، گزارش کامل طلا، جدول کامل شاخص‌ها و راهنمای اصطلاحات."]
    return "\n".join(L) + dashboard_link() + "\n\n" + DISCLAIMER


GLOSSARY = [
    ("امتیاز کلان (Macro Score)",
     "عددی از −۱۰۰ تا +۱۰۰ برای هر افق (۲۴ ساعت، ۴ هفته، ۶ ماه). از جمع «ضریب اثر × تغییر فعلی» همه‌ی شاخص‌ها ساخته "
     "می‌شود. مثبت = شرایط کلان به نفع افزایش قیمت، منفی = به نفع کاهش. زیر ±۵ خنثی، ۵ تا ۱۵ ضعیف، ۱۵ تا ۳۵ متوسط، "
     "۳۵ تا ۶۰ قوی، بالای ۶۰ بسیار قوی. سیگنال معاملاتی دست‌کم ±۱۵ هم‌جهت لازم دارد."),
    ("ضریب اثر (Impact Coefficient)",
     "عددی از −۱۰ تا +۱۰ که می‌گوید در گذشته یک حرکت یک‌σ این شاخص، قیمت دارایی را چقدر و در چه جهتی جابه‌جا کرده است. "
     "مثبت یعنی «شاخص بالاتر ← قیمت بالاتر». ۱۰ یعنی اثر بزرگ، از نظر آماری معنادار و با دست‌کم ۳۰ مشاهده‌ی مستقل. "
     "زیر ۱ تقریباً بی‌اثر، ۱ تا ۳ ضعیف، ۳ تا ۶ متوسط، بالای ۶ قوی. این یک همبستگی تاریخی است، نه پیش‌بینی."),
    ("σ (سیگما، انحراف معیار)",
     "اندازه‌ی نوسان معمول. «+۲σ» یعنی حرکتی دو برابر حرکت معمول که کمتر پیش می‌آید. دامنه‌ی ±۱σ حدود ۶۸٪ احتمال را پوشش می‌دهد."),
    ("شگفتی (Surprise)",
     "فاصله‌ی عدد منتشرشده از «انتظار»، تقسیم بر اندازه‌ی معمول این فاصله (بر حسب σ). چون اجماع تحلیلگران منبع رایگان "
     "ندارد، انتظار = روند (میانگین انتشارهای قبلی، T) یا مقدار قبلی (P)."),
    ("R (واحد ریسک)",
     "فاصله‌ی ورود تا حد ضرر. +۱R یعنی سودی به اندازه‌ی ریسک، −۱R یعنی خوردن حد ضرر. «میانگین R» متوسط نتیجه‌ی هر معامله "
     "است؛ بزرگ‌تر از صفر یعنی قاعده در مجموع سودده بوده است."),
    ("R:R", "نسبت سود هدف به ریسک؛ R:R ۲ یعنی سود بالقوه دو برابر ضرر بالقوه."),
    ("خارج از نمونه (Out-of-sample)",
     "آزمون روی داده‌هایی که در ساختن یا تنظیم قاعده استفاده نشده‌اند (walk-forward: تنظیم روی گذشته، آزمون روی دوره‌ی بعد). "
     "فقط نتیجه‌ی خارج از نمونه نشانه‌ی برتری واقعی است."),
    ("نرخ برد / hit rate",
     "درصد معامله‌های سودده (نرخ برد) یا درصد دفعاتی که جهت پیش‌بینی درست بوده (hit rate). بدون میانگین R معنا ندارد: "
     "نرخ برد ۴۰٪ با سودهای بزرگ می‌تواند سودده باشد."),
    ("اطمینان ضریب (H/M/L)",
     "زیاد: آماره‌ی t دست‌کم ۲٫۵۸ و ۵۰ مشاهده‌ی مستقل؛ متوسط: t دست‌کم ۱٫۹۶ و ۳۰ مشاهده؛ کم: بقیه."),
    ("p (p-value)", "احتمال این‌که نتیجه‌ای به همین خوبی صرفاً از شانس بیاید؛ زیر ۰٫۰۵ معنادار حساب می‌شود."),
    ("ATR", "میانگین دامنه‌ی واقعی نوسان روزانه در ۱۴ روز اخیر؛ مبنای دامنه‌ی محتمل یک روز."),
    ("سقف/کف سوئینگ", "آخرین نقطه‌ی برگشت قیمت در نمودار (روزانه یا ۴ ساعته)؛ شکستن آن یعنی تغییر ساختار روند."),
    ("فاز اقتصادی", "یکی از ۴ فاز چرخه (بهبود، رونق، اوج، رکود) که مدل از سطح و شتاب رشد، تورم و نقدینگی تخمین می‌زند."),
    ("موضع فدرال", "از −۱۰۰ (کاملاً انبساطی) تا +۱۰۰ (کاملاً انقباضی)؛ از مسیر نرخ، قیمت آتی‌ها، ترازنامه و بیانیه‌ها."),
    ("GC=F و PAXG",
     "GC=F قرارداد آتی طلای COMEX (قیمت واقعی طلا، با تأخیر). PAXG توکنی است که هر واحد آن پشتوانه‌ی یک اونس طلا دارد و "
     "۲۴ ساعته معامله می‌شود؛ نماینده‌ی بلادرنگ است و می‌تواند کمی بالاتر یا پایین‌تر از طلای واقعی باشد."),
    ("پیش‌موقعیت‌گیری", "حرکت‌های یک‌طرفه‌ی بازار در ساعت‌های پیش از انتشار یک داده (قیمت، جریان سفارش، قراردادهای باز، نرخ‌ها)."),
]


def glossary_message() -> str:
    L = ["📖 <b>راهنمای اصطلاحات</b>"]
    for term, desc in GLOSSARY:
        L += ["", f"<b>{term}</b>: {desc}"]
    return "\n".join(L)


def report_messages(f: Facts, now: pd.Timestamp, chains: dict[str, list[str]] | None = None, trigger: str | None = None,
                    names: dict | None = None, weekly: bool = False) -> list[tuple[str, str]]:
    """(kind, html) in sending order: overview, BTC, gold, full tables, glossary."""
    out = [("overview", overview_message(f, now, trigger, weekly))]
    for a, k in (("BTC", "btc"), ("Gold", "gold")):
        if f.asset(a) or f.macro.get(a):
            out.append((f"report_{k}", asset_message(f, a, now, (chains or {}).get(a), trigger, names)))
    for a, k in (("BTC", "btc"), ("Gold", "gold")):
        if f.asset(a) or f.macro.get(a):
            out.append((f"table_{k}", table_message(f, a)))
    out.append(("glossary", glossary_message()))
    return out
