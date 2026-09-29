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
from reports.facts import Facts, channel_fa, ind_name, indicators_cfg, releases_cfg
from reports.messages import ASSET_FA, DISCLAIMER, PREPOS_STATUS_FA, dashboard_link, num, signal_card, tehran

SCORE_GATE = 15  # |medium Macro Score| a trade signal needs (jobs.signals.MACRO_THRESHOLD)
EDGE_TRADES = 30  # jobs.signals.MIN_EDGE_TRADES
TOP_ROWS = 6
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
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


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


def flagged(r: Row) -> bool:
    """A reading that matters on its own: an unusual move (≥ 2σ), a large surprise, or an inverted curve."""
    inverted = r.key.startswith("spread_") and r.value.startswith("-")
    return inverted or (r.x is not None and (abs(r.x) >= 1 if r.coef_tag == "w" else abs(r.x) >= 2))


def impact_table(rows: list[Row]) -> list[str]:
    """One row per indicator: its main reading and its impact on this asset (flag for decision-relevant anomalies)."""
    head = f"{'Ticker':<{TICKER_W}}{'Value':>10}{'Coef':>7}{'D':>2}{'!':>2}"
    body = [f"{r.ticker[:TICKER_W]:<{TICKER_W}}{nz(r.value):>10}"
            f"{nz('—' if r.coef is None else f'{r.coef:+.1f}{r.coef_tag}'):>7}"
            f"{ {1: '▲', -1: '▼', 0: '·'}[r.effect]:>2}{'!' if flagged(r) else '':>2}" for r in rows]
    return _table(head, body)


def names_key(rows: list[Row]) -> str:
    return "\n".join(f"• {esc(r.name_fa)} — {esc(r.ticker)}" for r in rows)


TABLE_LEGEND = ("• Value = آخرین مقدار · Coef = ضریب اثر از −۱۰ تا +۱۰ (w: ۴ هفته، d: ۲۴ ساعت پس از انتشار) · "
                "D = اثر فعلی (▲ صعودی، ▼ نزولی، · ناچیز) · ! = حرکت غیرعادی (≥۲σ) یا وارونگی منحنی")


def tables_section(rows: list[Row], asset: str, title: str) -> str:
    """Heading → one-line legend → the table → the name key; blocks separated by blank lines."""
    return "\n\n".join([title, TABLE_LEGEND, *impact_table(rows), "<b>نام شاخص‌ها</b>\n" + names_key(rows)])


# ───────────────────────────── causal chain (plain sentences) ─────────────────────────────

def _move_phrase(state: float) -> str:
    verb = "بالا رفته" if state > 0 else "پایین آمده"
    m = 2 * abs(state)  # the state is the 4-week change in σ, clipped to ±2 and halved
    size = "به‌طور غیرعادی" if m >= 2 else "بیش از حد معمول" if m >= 1.5 else "در حد معمول" if m >= 0.7 else "کمی"
    return f"{size} {verb}"


def val(v: float) -> str:
    """A released value without noise decimals."""
    return num(v, "{:,.0f}" if abs(v) >= 100 else "{:,.2f}")


def chain_sentences(f: Facts, asset: str, now: pd.Timestamp) -> list[str]:
    """Cause → effect sentences (template; also the fallback for Claude's text). Kept narrative: strengths in words,
    one number per driver (its share of the 4-week score); the table carries the coefficients."""
    cfg = indicators_cfg()
    ch = channel_fa(cfg)
    chans = {i["key"]: i.get("channels", []) for sec in ("releases", "state") for i in cfg[sec]}
    name = ASSET_FA[asset]
    L = []
    for r in f.releases:
        if pd.Timestamp(r["release_utc"]) < now - pd.Timedelta(hours=24):
            continue
        k, _ = _coef(f, r["indicator"], asset, "24h")
        hot = "بالاتر" if r["z"] > 0 else "پایین‌تر"
        nm = esc(ind_name(r["indicator"]))
        s = f"{nm} {hot} از انتظار آمد ({val(r['actual'])} در برابر {val(r['expected'])}؛ شگفتی {z_reading(r['z'])})."
        if k is not None and abs(k) >= 1:
            s += (f" در گذشته عدد بالاتر برای {name} {'مثبت' if k > 0 else 'منفی'} بوده (اثر {coef_reading(k)})، "
                  f"پس این عدد {hot} برای {name} {'صعودی' if k * r['z'] > 0 else 'نزولی'} است.")
        else:
            s += f" این شگفتی در گذشته اثر محسوسی بر {name} نداشته است."
        L.append(s)
    med = (f.macro.get(asset) or {}).get("medium") or {}
    for t in [t for t in med.get("top", []) if abs(t["contribution"]) >= 0.5][:3]:
        key, c = t["indicator"], t["contribution"]
        path = " و ".join(f"«{esc(ch.get(x, x))}»" for x in chans.get(key, [])[:2])
        s = (f"{esc(ind_name(key))} در ۴ هفته‌ی اخیر {_move_phrase(t['state'])} است"
             + (f" و از مسیر {path} اثر می‌گذارد" if path else "")
             + f"؛ این برای {name} {'مثبت' if c > 0 else 'منفی'} است و حدود {num(max(abs(c), 1), '{:.0f}')} واحد "
             + f"امتیاز ۴ هفته را {'بالا می‌برد' if c > 0 else 'پایین می‌آورد'}.")
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
    lv = "{:,.0f}"  # ranges and levels: whole units are enough to act on
    if asset == "Gold":
        gc = f.gold_futures()
        px = x.get("price")
        if gc:
            L.append(f"• طلای واقعی (COMEX GC=F، با تأخیر): <b>{num(gc[0], '{:,.1f}')}</b> دلار/اونس")
        if px is not None:
            prem = (f" — {num(abs(px / gc[0] - 1) * 100, '{:.1f}')}٪ {'بالاتر' if px > gc[0] else 'پایین‌تر'} از GC=F"
                    if gc else "")
            bar = ((x.get("technical") or {}).get("1h") or {}).get("bar")
            if gc and bar and abs(_utc(gc[1]) - _utc(bar)) > pd.Timedelta(minutes=30):
                prem += " (زمان دو قیمت یکی نیست)"
            L.append(f"• PAXG (نماینده‌ی بلادرنگ): <b>{num(px, '{:,.1f}')}</b>{prem}")
        L.append("• مبنا: امتیاز کلان ۴ هفته و ۶ ماه ← GC=F؛ دامنه‌ها، سطوح و سیگنال ← PAXG.")
    else:
        L.append(f"• قیمت فعلی: <b>{num(x.get('price'), lv)}</b> دلار")
    if er.get("1d") and er.get("4w_1sigma"):
        L.append(f"• دامنه‌ی محتمل{' (PAXG)' if asset == 'Gold' else ''}: یک روز {num(min(er['1d']), lv)} تا "
                 f"{num(max(er['1d']), lv)} · ۴ هفته {num(min(er['4w_1sigma']), lv)} تا {num(max(er['4w_1sigma']), lv)}")
    return L


def scores_section(f: Facts, asset: str) -> list[str]:
    m = f.macro.get(asset) or {}
    L = ["<b>🧮 امتیاز کلان</b> (از −۱۰۰ تا +۱۰۰؛ آستانه‌ی سیگنال ±۱۵)"]
    for b, h in HORIZONS:
        sc = (m.get(b) or {}).get("score")
        if b == "short" and sc is None:
            L.append(f"• {h}: — (انتشار تازه‌ای نبوده)")
            continue
        L.append(f"• {h}: <b>{num(sc, '{:.0f}', sign=True)}</b> از ±۱۰۰ → {score_reading(sc)}")
    return L


def signals_section(f: Facts, asset: str, names: dict | None) -> list[str]:
    x = f.asset(asset)
    med = (x.get("macro") or {}).get("medium")
    L = ["<b>🎯 سیگنال معاملاتی</b>" + (" (بر پایه‌ی PAXG)" if asset == "Gold" else "")]
    for s in x.get("signals", []):  # a live signal is actionable: its full card stays
        L += signal_card(s, asset, blocked=False)
    if x.get("signals"):
        return L
    zone = (lambda z: f"{num(min(z), '{:,.0f}')} تا {num(max(z), '{:,.0f}')}")  # noqa: E731
    for s in x.get("watchlist", []):
        why = "برتری اثبات‌شده ندارد" if "edge" in " ".join(s.get("blocked_by") or []) else "کلان هم‌جهت نیست"
        L.append(f"👀 ستاپ {word(s['direction'])} در فهرست نظارت (ورود {zone(s['entry_zone'])}) — مسدود: {why}")
    L.append("سیگنالی صادر نشد:")
    if not x.get("watchlist"):
        L.append("• ستاپ تکنیکال: در کندل‌های اخیر ورودی نداده است")
    ok = med is not None and abs(med) >= SCORE_GATE
    L.append(f"• شرط کلان (±۱۵): الان {num(med, '{:.0f}', sign=True)} → {'برقرار' if ok else 'برقرار نیست'}")
    summ = (x.get("backtest") or {}).get("summary") or {}
    parts = []
    for side, fa_side in (("long", "خرید"), ("short", "فروش")):
        st = summ.get(side)
        if not st:
            continue
        n, r = st.get("n_trades", 0), st.get("avg_r")
        verdict = ("نمونه کم" if n < EDGE_TRADES else "زیان‌ده" if r is None or r <= 0 else "ناچیز" if r < 0.05 else "برقرار")
        count = f"، {num(n, '{:.0f}')} معامله" if n < EDGE_TRADES else ""
        parts.append(f"{fa_side} {num(r, '{:+.2f}R')} ({verdict}{count})")
    if parts:
        L.append("• برتری اثبات‌شده (میانگین R خارج از نمونه): " + "، ".join(parts))
    return L


def prepos_section(f: Facts, asset: str) -> list[str]:
    rows = [r for r in (f.prepos or {}).get("live", []) if r.get("asset") == asset]
    if not rows:
        return []
    L = ["<b>⏱ پیش‌موقعیت‌گیری پیش از انتشارها</b>"]
    no_edge = True
    for r in sorted(rows, key=lambda r: r["release_utc"]):
        st = r.get("status")
        bt = ((f.prepos.get("backtest") or {}).get(f"{r['release']}:{asset}") or {})
        no_edge &= not bt.get("edge")
        what = {"signal": "سیگنال فعال (پیام جداگانه)",
                "watchlist": f"نشانه‌ی {word(r.get('direction'))}، بدون برتری اثبات‌شده",
                "quiet": "بدون نشانه",
                "waiting": f"بررسی از {tehran(r['window_opens'])}" if r.get("window_opens") else "هنوز شروع نشده"}
        L.append(f"• {esc(r.get('name_fa') or r['release'])} ({tehran(r['release_utc'])}): "
                 f"{what.get(st, PREPOS_STATUS_FA.get(st, str(st)))}")
    if no_edge:
        L.append("هیچ‌کدام برتری اثبات‌شده ندارند؛ سیگنالی ارسال نمی‌شود.")
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
            line += f" (اثر تاریخی شگفتی آن: {coef_reading(v)})"
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
    L = [f"<b>✅ جمع‌بندی {name}</b>"]
    top = ((m.get("medium") or {}).get("top") or [])
    if top:
        t = top[0]
        L.append(f"• مهم‌ترین عامل الان: <b>{esc(ind_name(t['indicator']))}</b> — حدود "
                 f"{num(max(abs(t['contribution']), 1), '{:.0f}')} واحد از امتیاز ۴ هفته، "
                 f"{'به نفع افزایش' if t['contribution'] > 0 else 'به نفع کاهش'} قیمت.")
    med = (m.get("medium") or {}).get("score")
    L += invalidation(x, med, "{:,.0f}", "، قیمت PAXG" if asset == "Gold" else "")
    ev = upcoming_events(f, asset, now)
    if ev:
        L.append("• رویدادهایی که می‌توانند نتیجه را عوض کنند:")
        L += [f"   – {e}" for e in ev]
    return L


# ───────────────────────────── decision summary (top of each asset message) ─────────────────────────────

def _px(v: float, asset: str) -> str:
    """Summary-precision price: BTC to the nearest 100, gold to the dollar (the body carries the exact figure)."""
    return num(round(v, -2) if asset == "BTC" else round(v), "{:,.0f}")


def _events(f: Facts, asset: str, now: pd.Timestamp, trigger: str | None) -> list[dict]:
    """What triggered this report, most impactful first: releases of the last 24 h ranked by |coefficient × surprise|
    for this asset, then flagged news (importance ≥ 4) with its classification for this asset."""
    cfg = indicators_cfg()
    chans = {i["key"]: i.get("channels", []) for i in cfg["releases"]}
    out = []
    for r in f.releases:
        if pd.Timestamp(r["release_utc"]) < now - pd.Timedelta(hours=24):
            continue
        k, _ = _coef(f, r["indicator"], asset, "24h")
        out.append({"kind": "release", "r": r, "k": k, "channels": chans.get(r["indicator"], []),
                    "rank": abs((k or 0) * r["z"])})
    out.sort(key=lambda e: -e["rank"])
    col = "btc" if asset == "BTC" else "gold"
    news = f.important_news(now - pd.Timedelta(hours=24))
    for n in (news.head(2).to_dict("records") if not news.empty else []):
        out.append({"kind": "news", "n": n, "effect": n.get(col)})
    return out[:2]


def _no_position_reason(x: dict) -> str:
    wl = x.get("watchlist") or []
    if wl:
        why = " ".join(wl[0].get("blocked_by") or [])
        if "edge" in why:
            return "ستاپ تکنیکال هست، اما این قاعده هنوز برتری آماری اثبات‌شده‌ی خارج از نمونه ندارد"
        return "ستاپ تکنیکال هست، اما شواهد کلان ۴ هفته هم‌جهت و به اندازه‌ی کافی قوی نیست"
    med = (x.get("macro") or {}).get("medium")
    tail = "، و شواهد کلان هم برای تأیید سیگنال کافی نیست" if med is None or abs(med) < SCORE_GATE else ""
    return "قاعده‌ی تکنیکال در حال حاضر ستاپ ورودی نداده است" + tail


def decision_summary(f: Facts, asset: str, now: pd.Timestamp, trigger: str | None = None) -> list[str]:
    """Compact top section: trigger, impact on this asset, reasoning, probable direction, position (edge-gated).
    Template only — every figure is a rounded copy of a number the system computed; the body below has the exact values."""
    name = ASSET_FA[asset]
    x, m = f.asset(asset), f.macro.get(asset) or {}
    ch = channel_fa(indicators_cfg())
    evs = _events(f, asset, now, trigger)
    L = [f"🧭 <b>خلاصه‌ی تصمیم — {name}</b>"]
    # 1. trigger
    names = []
    for e in evs:
        if e["kind"] == "release":
            r = e["r"]
            names.append(f"{esc(ind_name(r['indicator']))} {'بالاتر' if r['z'] > 0 else 'پایین‌تر'} از انتظار "
                         f"(شگفتی {z_reading(r['z'])})")
        else:
            names.append(f"خبر: {esc(e['n'].get('title_fa') or e['n'].get('title'))}")
    if names:
        L.append("• رویداد: " + "؛ ".join(names))
    elif trigger:
        L.append(f"• رویداد: {esc(trigger)}")
    else:
        L.append("• رویداد: رویداد تازه‌ای ثبت نشده؛ این گزارش دوره‌ای است")
    # 2. impact on this asset + 3. reasoning (the same logic as the causal chain)
    top = evs[0] if evs else None
    if top and top["kind"] == "release" and top["k"] is not None and abs(top["k"]) >= 1:
        r, k = top["r"], top["k"]
        eff = "صعودی" if k * r["z"] > 0 else "نزولی"
        L.append(f"• اثر بر {name}: {eff} — اثر تاریخی {coef_reading(k)}")
        path = "، ".join(f"«{esc(ch.get(c, c))}»" for c in top["channels"][:2])
        L.append(f"• استدلال: {'از مسیر ' + path + '؛ ' if path else ''}در گذشته عدد بالاتر از انتظار برای {name} "
                 f"{'مثبت' if k > 0 else 'منفی'} بوده، پس این عدد {'بالاتر' if r['z'] > 0 else 'پایین‌تر'} {eff} است.")
    elif top and top["kind"] == "news" and top.get("effect") in ("bullish", "bearish"):
        eff = "صعودی" if top["effect"] == "bullish" else "نزولی"
        L.append(f"• اثر بر {name}: {eff} (طبقه‌بندی خودکار خبر)")
        chans = [c for c in str(top["n"].get("channels") or "").split("|") if c]
        L.append("• استدلال: " + (f"از مسیر {'، '.join('«' + esc(ch.get(c, c)) + '»' for c in chans[:2])}."
                                   if chans else "بر پایه‌ی تیتر خبر؛ پیش از تصمیم متن کامل را بخوانید."))
    else:
        if top:
            L.append(f"• اثر بر {name}: ناچیز — این رویداد در گذشته اثر محسوسی بر {name} نداشته است")
        tc = ((m.get("medium") or {}).get("top") or [])
        t = next((t for t in tc if abs(t["contribution"]) >= 0.5), None)
        if t:
            L.append(f"• استدلال: مهم‌ترین عامل ۴ هفته {esc(ind_name(t['indicator']))} است که "
                     f"{'بالا رفته' if t['state'] > 0 else 'پایین آمده'} و برای {name} "
                     f"{'مثبت' if t['contribution'] > 0 else 'منفی'} ارزیابی می‌شود.")
        else:
            L.append(f"• استدلال: هیچ شاخصی الان سهم محسوسی در امتیاز {name} ندارد.")
    # 4. probable direction (the system's Macro Scores, as words)
    sh, med = (m.get("short") or {}).get("score"), (m.get("medium") or {}).get("score")
    near = f"۲۴ ساعت {score_direction(sh)}، " if sh is not None else ""
    L.append(f"• جهت محتمل: {near}۴ هفته <b>{score_direction(med)}</b> — اطمینان {score_confidence(med)}")
    # 5. position — only a signal that passed the out-of-sample edge gate
    sig = (x.get("signals") or [None])[0]
    pre = next((r for r in (f.prepos or {}).get("live", []) if r.get("asset") == asset and r.get("status") == "signal"
                and r.get("entry") is not None), None)
    if sig:
        L.append(f"• پیشنهاد موقعیت: <b>{word(sig['direction'])}</b> · ورود ≈ {_px(min(sig['entry_zone']), asset)} تا "
                 f"{_px(max(sig['entry_zone']), asset)} · هدف ۱ ≈ {_px(sig['targets'][0], asset)} · حد ضرر ≈ "
                 f"{_px(sig['stop_loss'], asset)} · اطمینان {word(sig['confidence'])} (جزئیات در بخش سیگنال)")
    elif pre:
        L.append(f"• پیشنهاد موقعیت (پیش از انتشار): <b>{word(pre['direction'])}</b> · ورود ≈ {_px(pre['entry'], asset)} · "
                 f"هدف ۱ ≈ {_px(pre['targets'][0], asset)} · حد ضرر ≈ {_px(pre['stop_loss'], asset)} (پیام جداگانه)")
    else:
        L.append(f"• پیشنهاد موقعیت: <b>فعلاً هیچ موقعیتی پیشنهاد نمی‌شود</b> — {_no_position_reason(x)}.")
    return L


# ───────────────────────────── messages ─────────────────────────────

def asset_message(f: Facts, asset: str, now: pd.Timestamp, chain: list[str] | None = None, trigger: str | None = None,
                  names: dict | None = None) -> str:
    rows = indicator_rows(f, asset)
    L = [*decision_summary(f, asset, now, trigger), "", "━━━━━━━━━━━━━━", "",
         f"{ICON[asset]} <b>گزارش {ASSET_FA[asset]} ({'BTC' if asset == 'BTC' else 'طلا / XAU'})</b> — {tehran(now)} (تهران)"]
    if trigger:
        L.append(f"علت گزارش: {esc(trigger)}")
    L += ["", *price_block(f, asset), "", *scores_section(f, asset)]
    L += ["", "<b>🔗 زنجیره‌ی علّی — چرا این امتیاز</b>"]
    L += [f"{i}. {s}" for i, s in enumerate(chain or chain_sentences(f, asset, now), 1)]
    shown = [r for r in rows if r.coef is not None and abs(r.coef) >= 1][:TOP_ROWS]
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
