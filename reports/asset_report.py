"""Per-asset Persian Telegram reports: market overview, one message per asset (BTC, gold), the full indicator tables
and a short glossary — only what changes a decision, readable on a phone.

Every number comes from `reports.facts.Facts` (the job outputs) or from simple arithmetic on the snapshot's own series;
nothing is estimated here. Dates are Persian (Jalali) in Tehran time (core.jalali). Tables are <pre> blocks at most
32 characters wide: ASCII labels and numbers first, so they align; Persian words and dates last, each in a
right-to-left isolate so it reads correctly inside the left-to-right table line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.fa import word
from core.jalali import jdate, jdate_short
from core.registry import load_registry
from notify.telegram import esc
from reports.facts import Facts, channel_fa, ind_name, indicators_cfg, releases_cfg
from reports.messages import ASSET_FA, DISCLAIMER, dashboard_link, num, signal_card, tehran

SCORE_GATE = 15  # |medium Macro Score| a trade signal needs (jobs.signals.MACRO_THRESHOLD)
EDGE_TRADES = 30  # jobs.signals.MIN_EDGE_TRADES
TOP_ROWS = 6
CHUNK_ROWS = 14
HORIZONS = (("short", "۲۴ ساعت"), ("medium", "۴ هفته"), ("long", "۶ ماه"))
CONF_FA = {"high": "زیاد", "medium": "متوسط", "low": "کم"}
ICON = {"BTC": "₿", "Gold": "🥇"}
# short ASCII labels (≤ 5 characters) for the phone-width tables; the Persian name is in the key under each table
SHORT = {"cpi": "CPI", "core_cpi": "CCPI", "ppi_final_demand": "PPI", "pce": "PCE", "core_pce": "CPCE",
         "nonfarm_payrolls": "NFP", "unemployment_rate": "UNEMP", "avg_hourly_earnings": "AHE", "jolts_openings": "JOLTS",
         "initial_claims": "CLAIM", "ust_2y": "US2Y", "ust_10y": "US10Y", "real_yield_10y": "RY10", "breakeven_10y": "BE10",
         "spread_10y_2y": "10-2Y", "fed_funds_eff_daily": "FFR", "dxy": "DXY", "baa_10y_spread": "BAA", "nfci": "NFCI",
         "vix": "VIX", "move_index": "MOVE", "spx": "SPX", "copper_futures": "COPR", "wti_futures": "WTI",
         "fed_balance_sheet": "FEDBS", "net_liquidity_weekly": "NLIQ", "m2": "M2", "philly_fed_activity": "PHIL"}
# what a higher / lower print means, in plain words (template text, no numbers)
MEANING_FA = {"cpi": ("تورم داغ‌تر", "تورم آرام‌تر"), "core_cpi": ("تورم پایه‌ی داغ‌تر", "تورم پایه‌ی آرام‌تر"),
              "ppi_final_demand": ("فشار قیمت تولیدکننده بیشتر", "فشار قیمت تولیدکننده کمتر"),
              "pce": ("تورم PCE داغ‌تر", "تورم PCE آرام‌تر"), "core_pce": ("تورم پایه‌ی PCE داغ‌تر", "تورم پایه‌ی PCE آرام‌تر"),
              "nonfarm_payrolls": ("بازار کار قوی‌تر", "بازار کار ضعیف‌تر"),
              "unemployment_rate": ("بازار کار ضعیف‌تر", "بازار کار قوی‌تر"),
              "avg_hourly_earnings": ("فشار دستمزد بیشتر", "فشار دستمزد کمتر"),
              "jolts_openings": ("تقاضای نیروی کار قوی‌تر", "سرد شدن بازار کار"),
              "initial_claims": ("بازار کار ضعیف‌تر", "بازار کار قوی‌تر")}
BASIS_CODE = {"mean3": "T", "mean4": "T", "prev": "P"}
LRM, RLI, PDI = "\u200e", "\u2067", "\u2069"
GLOSSARY_NOTE = "معنی اصطلاحات: پیام آخر همین گزارش."


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
        ticker = SHORT.get(key) or (meta.source_id if meta else key).lstrip("^")[:5]
        rows.append(Row(key, ind_name(key), ticker, value, change, surprise, k, "w" if in_state else "d", eff,
                        conf, x))
    return sorted(rows, key=lambda r: -abs(r.coef) if r.coef is not None else 1)


TABLE_WIDTH = 32  # characters per line; fits Telegram's monospace font on a phone without wrapping


def _pre(lines: list[str]) -> str:
    # a left-to-right mark opens every line, so clients that pick each line's direction from its first strong
    # character keep the columns left-to-right even under a right-to-left message
    return "<pre>" + esc("\n".join(LRM + ln for ln in lines)) + "</pre>"


def _rtl(cell: str) -> str:
    """A Persian cell inside a left-to-right table line: isolated so its words and digits read right-to-left."""
    return f"{RLI}{cell}{PDI}"


def visible(line: str) -> str:
    return re.sub("[‎⁦-⁩]", "", line)


def _table(head: str, body: list[str]) -> list[str]:
    """Header, a dashed divider, then the rows — in blocks of CHUNK_ROWS so a Telegram message never cuts a block."""
    width = max(len(visible(x)) for x in [head, *body]) if body else len(head)
    return [_pre([head, "-" * width, *body[c:c + CHUNK_ROWS]]) for c in range(0, len(body), CHUNK_ROWS)] or [_pre([head])]


_SIGNED_ZERO = re.compile(r"[+-](?=0(?:\.0+)?(?![\d.,]))")


def nz(cell: str) -> str:
    """Drop the sign of a value that rounds to zero ("-0.0M" → "0.0M")."""
    return _SIGNED_ZERO.sub("", cell)


# ───────────────────────────── release comparison (overview + decision summary) ─────────────────────────────

def _rel_cfg() -> dict[str, dict]:
    return {r["key"]: r for r in indicators_cfg()["releases"]}


def rel_value(key: str, v: float) -> str:
    """A released figure in its own measure, without rescaling (the number is exactly the computed one)."""
    m = _rel_cfg().get(key, {}).get("measure", "level")
    meta = load_registry().get(key)
    units = meta.units if meta else ""
    if m in ("mom_pct", "qoq_ann", "yoy_pct"):
        return nz(f"{v:+.2f}%")
    if m in ("diff", "mom_change"):
        return nz(f"{v:+,.0f}k" if units == "thousands" else f"{v:+,.2f}")
    if units == "%":
        return f"{v:.1f}%"
    if units == "thousands":
        return f"{v:,.0f}k"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def basis_word(basis: str | None) -> str:
    return "قبلی" if "prev" in str(basis or "") else "روند"


def release_table(rels: list[dict]) -> str:
    head = f"{'Name':<6}{'Actual':>9}{'Expect':>9}{'':>3}"
    body = [f"{SHORT.get(r['indicator'], r['indicator'][:5]):<6}{rel_value(r['indicator'], r['actual']):>9}"
            f"{rel_value(r['indicator'], r['expected']):>9}{'▲' if r['z'] > 0 else '▼':>3}" for r in rels]
    return _table(head, body)[0]


def release_line(r: dict) -> str:
    """One plain sentence per release: above/below expectation (basis in one word) and what it means."""
    hot = r["z"] > 0
    head = f"• {esc(ind_name(r['indicator']))}: {'بالاتر' if hot else 'پایین‌تر'} از انتظار ({basis_word(r.get('basis'))})"
    if abs(r["z"]) < 0.5:
        return head + " — تفاوت ناچیز؛ پیام خاصی ندارد"
    meaning = MEANING_FA.get(r["indicator"], ("عدد بالاتر", "عدد پایین‌تر"))[0 if hot else 1]
    return head + f" — {meaning} (شگفتی {z_reading(r['z'])})"


# ───────────────────────────── indicator table (sorted by next update) ─────────────────────────────

def impact_word(k: float | None) -> str:
    if k is None or not np.isfinite(k):
        return "—"
    a = abs(k)
    return "ناچیز" if a < 1 else "ضعیف" if a < 3 else "متوسط" if a < 6 else "قوی"


def next_updates(f: Facts, now: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Next scheduled release of each series, from the snapshot calendar (collectors.calendar.upcoming())."""
    c = f.calendar
    if c.empty:
        return {}
    rel, out = releases_cfg(), {}
    for e in c[c["scheduled_utc"] > now].sort_values("scheduled_utc").itertuples():
        for k in rel.get(e.release_id, {}).get("series_keys", []):
            out.setdefault(k, e.scheduled_utc)
    if "fed_balance_sheet" in out:  # net liquidity = Fed balance sheet − TGA − reverse repo: updates with the H.4.1
        out.setdefault("net_liquidity_weekly", out["fed_balance_sheet"])
    return out


def _compact(v: str) -> str:
    v = nz(v.replace("%/m", "%").replace("pp", "%"))
    return re.sub(r"(\d)\.(\d)\d", r"\1.\2", v) if len(v) > 6 else v


@dataclass
class TableRow:
    row: Row
    when: pd.Timestamp | None
    daily: bool


def table_rows(f: Facts, rows: list[Row], now: pd.Timestamp) -> list[TableRow]:
    nxt, reg = next_updates(f, now), load_registry()
    out = [TableRow(r, nxt.get(r.key), (reg[r.key].frequency == "D") if r.key in reg else False) for r in rows]
    # soonest dated update first, then the daily market series, then series with no known date; ties by impact
    return sorted(out, key=lambda t: (0 if t.when is not None else 1 if t.daily else 2,
                                      t.when.value if t.when is not None else 0, -abs(t.row.coef or 0)))


def indicator_table(trs: list[TableRow], now: pd.Timestamp) -> list[str]:
    head = f"{'Name':<5} {'Value':>6} {'Impact':<7} Next"
    body = []
    for t in trs:
        r = t.row
        arrow = {1: " ▲", -1: " ▼", 0: ""}[r.effect] if impact_word(r.coef) not in ("—", "ناچیز") else ""
        imp = impact_word(r.coef) + arrow
        when = jdate_short(t.when, now) if t.when is not None else "روزانه" if t.daily else "—"
        body.append(f"{r.ticker[:5]:<5} {_compact(r.value):>6} {_rtl(imp)}{' ' * (7 - len(imp))} {_rtl(when)}")
    return _table(head, body)


def names_key(rows: list[Row]) -> str:
    mom = {r["key"] for r in indicators_cfg()["releases"] if r["measure"] == "mom_pct"}
    return "\n".join(f"• {esc(r.ticker)}: {esc(r.name_fa)}{' (ماهانه)' if r.key in mom else ''}" for r in rows)


TABLE_LEGEND = "اثر = قدرت اثر تاریخی بر همین دارایی، ▲▼ = جهت اثر فعلی · Next = به‌روزرسانی بعدی"


def tables_section(f: Facts, rows: list[Row], title: str, now: pd.Timestamp) -> str:
    trs = table_rows(f, rows, now)
    return "\n\n".join([title, TABLE_LEGEND, *indicator_table(trs, now), names_key([t.row for t in trs])])


# ───────────────────────────── drivers (plain sentences) ─────────────────────────────

def _move_phrase(state: float) -> str:
    verb = "بالا رفته" if state > 0 else "پایین آمده"
    m = 2 * abs(state)  # the state is the 4-week change in σ, clipped to ±2 and halved
    size = "به‌طور غیرعادی" if m >= 2 else "بیش از حد معمول" if m >= 1.5 else "در حد معمول" if m >= 0.7 else "کمی"
    return f"{size} {verb}"


def val(v: float) -> str:
    """A released value without noise decimals."""
    return num(v, "{:,.0f}" if abs(v) >= 100 else "{:,.2f}")


def chain_sentences(f: Facts, asset: str, now: pd.Timestamp) -> list[str]:
    """The 2–3 main drivers of the 4-week score as plain cause → effect sentences, without numbers (template; also
    the fallback for Claude's text). Releases are covered by the decision summary's release table."""
    cfg = indicators_cfg()
    ch = channel_fa(cfg)
    chans = {i["key"]: i.get("channels", []) for sec in ("releases", "state") for i in cfg[sec]}
    name = ASSET_FA[asset]
    med = (f.macro.get(asset) or {}).get("medium") or {}
    L = []
    for t in [t for t in med.get("top", []) if abs(t["contribution"]) >= 0.5][:3]:
        key, c = t["indicator"], t["contribution"]
        path = " و ".join(f"«{esc(ch.get(x, x))}»" for x in chans.get(key, [])[:2])
        L.append(f"{esc(ind_name(key))} در ۴ هفته‌ی اخیر {_move_phrase(t['state'])} است"
                 + (f" و از مسیر {path}" if path else "") + f" برای {name} {'مثبت' if c > 0 else 'منفی'} است.")
    if not L:
        L.append(f"هیچ شاخصی الان سهم محسوسی در امتیاز {name} ندارد؛ برآیند کلان نزدیک صفر است.")
    return L


# ───────────────────────────── body sections ─────────────────────────────

def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _px(v: float, asset: str) -> str:
    """Summary-precision price: BTC to the nearest 100, gold to the dollar (the body carries the exact figure)."""
    return num(round(v, -2) if asset == "BTC" else round(v), "{:,.0f}")


def price_block(f: Facts, asset: str) -> list[str]:
    x = f.asset(asset)
    er = (x.get("expected_range") or {}).get("4w_1sigma")
    rng = f"دامنه‌ی محتمل ۴ هفته: {_px(min(er), asset)} تا {_px(max(er), asset)}" if er else ""
    if asset == "Gold":
        gc, px = f.gold_futures(), x.get("price")
        L = []
        if gc:
            L.append(f"💵 طلا (COMEX GC=F): <b>{num(gc[0], '{:,.0f}')}</b> دلار/اونس")
        if px is not None:
            prem = f" ({num(abs(px / gc[0] - 1) * 100, '{:.1f}')}٪ {'بالاتر' if px > gc[0] else 'پایین‌تر'})" if gc else ""
            L.append(f"• PAXG (بلادرنگ، مبنای سطوح و سیگنال): {num(px, '{:,.0f}')}{prem}")
        if rng:
            L.append(f"• {rng} (PAXG)")
        return L
    return [f"💵 قیمت: <b>{num(x.get('price'), '{:,.0f}')}</b> دلار" + (f" · {rng}" if rng else "")]


def scores_section(f: Facts, asset: str) -> list[str]:
    m = f.macro.get(asset) or {}
    L = ["🧮 <b>امتیاز کلان</b> (−۱۰۰ تا +۱۰۰)"]
    for b, h in HORIZONS:
        sc = (m.get(b) or {}).get("score")
        L.append(f"• {h}: " + ("— (انتشار تازه‌ای نبوده)" if sc is None else
                                f"{score_direction(sc)} ({nz(num(sc, '{:+.0f}'))})"))
    return L


def _no_position_reason(x: dict) -> str:
    wl = x.get("watchlist") or []
    if wl:
        if "edge" in " ".join(wl[0].get("blocked_by") or []):
            return "ستاپ تکنیکال هست، اما این قاعده برتری آماری اثبات‌شده‌ی خارج از نمونه ندارد"
        return "ستاپ تکنیکال هست، اما شواهد کلان ۴ هفته هم‌جهت و کافی نیست"
    med = (x.get("macro") or {}).get("medium")
    tail = "، و شواهد کلان هم برای تأیید سیگنال کافی نیست" if med is None or abs(med) < SCORE_GATE else ""
    return "قاعده‌ی تکنیکال ستاپ ورودی نداده است" + tail


def signals_section(f: Facts, asset: str, names: dict | None = None) -> list[str]:
    x = f.asset(asset)
    if x.get("signals"):  # a live signal is actionable: its full card stays
        L = ["🎯 <b>سیگنال معاملاتی</b>" + (" (PAXG)" if asset == "Gold" else "")]
        for s in x["signals"]:
            L += signal_card(s, asset, blocked=False)
        return L
    return [f"🎯 سیگنال: فعلاً هیچ — {_no_position_reason(x)}."]


def prepos_section(f: Facts, asset: str) -> list[str]:
    rows = [r for r in (f.prepos or {}).get("live", []) if r.get("asset") == asset]
    if not rows:
        return []
    live = [r for r in rows if r.get("status") == "signal"]
    if live:
        return [f"⏱ پیش‌موقعیت‌گیری: سیگنال فعال پیش از {esc(live[0].get('name_fa') or live[0]['release'])} (پیام جداگانه)."]
    names = "، ".join(dict.fromkeys(esc(r.get("name_fa") or r["release"]) for r in sorted(rows, key=lambda r: r["release_utc"])))
    return [f"⏱ پیش‌موقعیت‌گیری ({names}): نشانه‌ای با برتری اثبات‌شده نیست."]


def upcoming_events(f: Facts, asset: str, now: pd.Timestamp, days: int = 7, n: int = 1) -> list[str]:
    c = f.calendar
    if c.empty:
        return []
    up = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=days))]
    up = up[up["importance"] >= 4] if "importance" in up else up
    rel, rcfg, out = releases_cfg(), _rel_cfg(), []
    for e in up.drop_duplicates("event_id").sort_values("scheduled_utc").head(n).itertuples():
        r = rel.get(e.release_id, {})
        ks = [v for v in (_coef(f, k, asset, "24h")[0] for k in r.get("series_keys", []) if k in rcfg) if v is not None]
        line = f"{esc(r.get('name_fa', e.release_id))} — {tehran(e.scheduled_utc)}"
        if ks:
            line += f" (اثر تاریخی: {impact_word(max(ks, key=abs))})"
        out.append(line)
    return out


def invalidation(x: dict, med: float | None, fmt: str, base: str) -> list[str]:
    """The one price level that would invalidate / change the 4-week view, relative to where the price is now."""
    px = x.get("price")
    d1 = (x.get("technical") or {}).get("1d") or {}
    hi, lo = d1.get("last_swing_high"), d1.get("last_swing_low")
    if px is None or hi is None or lo is None:
        return ["• سطح ابطال قیمتی در دسترس نیست."]
    lv = lambda v: f"{num(v, fmt)}{base}"  # noqa: E731
    if med is not None and med >= 5:
        if px < lo:
            return [f"• ⚠️ قیمت زیر کف روزانه {lv(lo)} است؛ تا بسته‌شدن دوباره بالای آن، تمایل صعودی تأیید نمی‌شود."]
        return [f"• ابطال: بسته‌شدن روزانه زیر {lv(lo)} (کف روزانه)."]
    if med is not None and med <= -5:
        if px > hi:
            return [f"• ⚠️ قیمت بالای سقف روزانه {lv(hi)} است؛ تا بسته‌شدن دوباره زیر آن، تمایل نزولی تأیید نمی‌شود."]
        return [f"• ابطال: بسته‌شدن روزانه بالای {lv(hi)} (سقف روزانه)."]
    if px < lo:
        return [f"• قیمت زیر کف روزانه {lv(lo)} است (ساختار نزولی)؛ بسته‌شدن بالای {lv(hi)} آن را صعودی می‌کند."]
    if px > hi:
        return [f"• قیمت بالای سقف روزانه {lv(hi)} است (ساختار صعودی)؛ بسته‌شدن زیر {lv(lo)} آن را نزولی می‌کند."]
    return [f"• جهت‌دار شدن: بسته‌شدن روزانه بالای {lv(hi)} صعودی، زیر {lv(lo)} نزولی."]


def conclusion(f: Facts, asset: str, now: pd.Timestamp) -> list[str]:
    x, m = f.asset(asset), f.macro.get(asset) or {}
    med = (m.get("medium") or {}).get("score")
    top = ((m.get("medium") or {}).get("top") or [])
    driver = (f"؛ مهم‌ترین عامل: {esc(ind_name(top[0]['indicator']))} "
              f"({'به نفع افزایش' if top[0]['contribution'] > 0 else 'به نفع کاهش'})" if top else "")
    L = [f"✅ <b>جمع‌بندی</b>: ۴ هفته {score_direction(med)}{driver}."]
    L += invalidation(x, med, "{:,.0f}", " (PAXG)" if asset == "Gold" else "")
    ev = upcoming_events(f, asset, now)
    if ev:
        L.append(f"• رویداد مهم بعدی: {ev[0]}")
    return L


# ───────────────────────────── decision summary (top of each asset message; template only) ─────────────────────────────

def _events(f: Facts, asset: str, now: pd.Timestamp, trigger: str | None) -> list[dict]:
    """What triggered this report, most impactful first: releases of the last 24 h ranked by |coefficient × surprise|
    for this asset, then flagged news (importance ≥ 4) with its classification for this asset."""
    chans = {i["key"]: i.get("channels", []) for i in indicators_cfg()["releases"]}
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
    return out


def decision_summary(f: Facts, asset: str, now: pd.Timestamp, trigger: str | None = None) -> list[str]:
    """Trigger (release table for the releases that matter for this asset), impact, reasoning, likely direction and a
    position filled only by a signal that passed the out-of-sample edge gate. Template only (no Claude); every figure
    is a computed number or a rounding of one."""
    name = ASSET_FA[asset]
    x, m = f.asset(asset), f.macro.get(asset) or {}
    ch = channel_fa(indicators_cfg())
    evs = _events(f, asset, now, trigger)
    rels = [e for e in evs if e["kind"] == "release"]
    relevant = [e for e in rels if e["k"] is not None and abs(e["k"]) >= 1][:3]
    news = [e for e in evs if e["kind"] == "news"]
    L = [f"🧭 <b>خلاصه‌ی تصمیم — {name}</b>"]
    if relevant:
        L += ["• رویداد:", release_table([e["r"] for e in relevant])]
        for e in relevant:
            r, k = e["r"], e["k"]
            L.append(f"{release_line(r)} → برای {name} {'صعودی' if k * r['z'] > 0 else 'نزولی'}")
    elif rels:
        L.append(f"• رویداد: {'، '.join(esc(ind_name(e['r']['indicator'])) for e in rels[:3])} منتشر شد؛ "
                 f"اثر تاریخی آن بر {name} ناچیز است")
    elif news:
        L.append(f"• رویداد: خبر — {esc(news[0]['n'].get('title_fa') or news[0]['n'].get('title'))}")
    else:
        L.append(f"• رویداد: {esc(trigger)}" if trigger else "• رویداد: رویداد تازه‌ای ثبت نشده؛ گزارش دوره‌ای")
    top = relevant[0] if relevant else None
    if top:
        r, k = top["r"], top["k"]
        eff = "صعودی" if k * r["z"] > 0 else "نزولی"
        L.append(f"• اثر بر {name}: {eff} — اثر تاریخی {impact_word(k)}")
        path = " و ".join(f"«{esc(ch.get(c, c))}»" for c in top["channels"][:2])
        L.append(f"• استدلال: {'از مسیر ' + path + '؛ ' if path else ''}عدد بالاتر در گذشته برای {name} "
                 f"{'مثبت' if k > 0 else 'منفی'} بوده است.")
    elif not rels and news and news[0].get("effect") in ("bullish", "bearish"):
        L.append(f"• اثر بر {name}: {'صعودی' if news[0]['effect'] == 'bullish' else 'نزولی'} (طبقه‌بندی خودکار تیتر)")
    else:
        t = next((t for t in ((m.get("medium") or {}).get("top") or []) if abs(t["contribution"]) >= 0.5), None)
        L.append(f"• استدلال: مهم‌ترین عامل ۴ هفته {esc(ind_name(t['indicator']))} است که برای {name} "
                 f"{'مثبت' if t['contribution'] > 0 else 'منفی'} است." if t else
                 f"• استدلال: هیچ شاخصی الان سهم محسوسی در امتیاز {name} ندارد.")
    sh, med = (m.get("short") or {}).get("score"), (m.get("medium") or {}).get("score")
    near = f"۲۴ ساعت {score_direction(sh)}، " if sh is not None else ""
    L.append(f"• جهت محتمل: {near}۴ هفته <b>{score_direction(med)}</b> — اطمینان {score_confidence(med)}")
    sig = (x.get("signals") or [None])[0]
    pre = next((r for r in (f.prepos or {}).get("live", []) if r.get("asset") == asset and r.get("status") == "signal"
                and r.get("entry") is not None), None)
    if sig:
        L.append(f"• پیشنهاد موقعیت: <b>{word(sig['direction'])}</b> · ورود ≈ {_px(min(sig['entry_zone']), asset)} تا "
                 f"{_px(max(sig['entry_zone']), asset)} · هدف ۱ ≈ {_px(sig['targets'][0], asset)} · حد ضرر ≈ "
                 f"{_px(sig['stop_loss'], asset)} · اطمینان {word(sig['confidence'])}")
    elif pre:
        L.append(f"• پیشنهاد موقعیت (پیش از انتشار): <b>{word(pre['direction'])}</b> · ورود ≈ {_px(pre['entry'], asset)} · "
                 f"هدف ۱ ≈ {_px(pre['targets'][0], asset)} · حد ضرر ≈ {_px(pre['stop_loss'], asset)}")
    else:
        L.append(f"• پیشنهاد موقعیت: <b>فعلاً هیچ</b> — {_no_position_reason(x)}.")
    return L


# ───────────────────────────── messages ─────────────────────────────

def _block(lines: list[str]) -> str:
    return "\n".join(lines)


def asset_message(f: Facts, asset: str, now: pd.Timestamp, chain: list[str] | None = None, trigger: str | None = None,
                  names: dict | None = None) -> str:
    rows = indicator_rows(f, asset)
    shown = [r for r in rows if r.coef is not None and abs(r.coef) >= 1][:TOP_ROWS]
    title = "📋 <b>شاخص‌های مؤثر</b> (به ترتیب به‌روزرسانی بعدی؛ جدول کامل در پیام بعد)"
    parts = [_block(decision_summary(f, asset, now, trigger)), "━━━━━━━━━━━━━━",
             f"{ICON[asset]} <b>گزارش {ASSET_FA[asset]}</b> — {tehran(now)}",
             _block(price_block(f, asset)), _block(scores_section(f, asset)),
             _block(["🔗 <b>عوامل اصلی</b>", *[f"{i}. {s}" for i, s in enumerate((chain or chain_sentences(f, asset, now))[:3], 1)]]),
             tables_section(f, shown, title, now),
             _block(signals_section(f, asset, names) + prepos_section(f, asset)),
             _block(conclusion(f, asset, now)), GLOSSARY_NOTE]
    return "\n\n".join(parts)


def table_message(f: Facts, asset: str, now: pd.Timestamp | None = None) -> str:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    rows = indicator_rows(f, asset)
    title = f"📋 <b>جدول کامل شاخص‌ها — {ASSET_FA[asset]}</b> (به ترتیب به‌روزرسانی بعدی)"
    return tables_section(f, rows, title, now)


def outlook_sentence(reg: dict) -> str:
    """Plain-language outlook from the regime model's own outputs (signs of its axes and its phase probabilities)."""
    ax = reg.get("axes") or {}
    parts = []
    if ax.get("G") is not None:
        parts.append(f"رشد {'بالای' if ax['G'] > 0 else 'زیر'} میانگین تاریخی است"
                     + (f" و {'رو به بهبود' if ax['g'] > 0 else 'رو به کندی'}" if ax.get("g") is not None else ""))
    if ax.get("p") is not None:
        parts.append(f"فشار تورم و نقدینگی {'رو به افزایش' if ax['p'] > 0 else 'رو به کاهش'}")
    probs = reg.get("probabilities") or {}
    cur = reg.get("regime")
    others = sorted(((p, r) for r, p in probs.items() if r != cur), reverse=True)
    risk = f" با ریسک نزدیک شدن به {word(others[0][1])}" if others and others[0][0] >= 0.15 else ""
    head = "؛ ".join(parts)
    return (head + "؛ " if head else "") + f"چشم‌انداز: ادامه‌ی {word(cur)}{risk}"


def overview_message(f: Facts, now: pd.Timestamp, trigger: str | None = None, weekly: bool = False) -> str:
    s = f.state
    title = "🗓 <b>خلاصه‌ی هفتگی — نمای کلی بازار</b>" if weekly else "🌐 <b>نمای کلی بازار</b>"
    L = [f"{title} — {tehran(now)}"]
    if trigger:
        L.append(f"علت گزارش: {esc(trigger)}")
    reg = s.get("regime") or {}
    if reg.get("regime"):
        p = (reg.get("probabilities") or {}).get(reg["regime"], 0)
        L += ["", f"🧭 فاز اقتصادی: <b>{esc(word(reg['regime']))}</b> — احتمال {num(p * 100, '{:.0f}')}٪",
              f"   {outlook_sentence(reg)}"]
    st = (s.get("fed_stance") or {}).get("score")
    if st is not None:
        L.append(f"🏦 موضع فدرال: <b>{num(st, '{:.0f}', sign=True)}</b> از بازه‌ی −۱۰۰ (کاملاً انبساطی) تا +۱۰۰ "
                 f"(کاملاً انقباضی) → {stance_reading(st)}")
    fw = s.get("fedwatch_next") or {}
    if fw.get("meeting"):
        pct = {k: num((fw.get(k) or 0) * 100, "{:.0f}") for k in ("p_cut", "p_hold", "p_hike")}
        L.append(f"📈 جلسه‌ی بعدی FOMC ({jdate(str(fw['meeting']))}): احتمال کاهش نرخ {pct['p_cut']}٪ · "
                 f"بدون تغییر {pct['p_hold']}٪ · افزایش {pct['p_hike']}٪ (از قیمت قراردادهای آتی وجوه فدرال)")
    parts = [_block(L)]
    since = now - pd.Timedelta(days=7 if weekly else 1)
    rel = [r for r in f.releases if pd.Timestamp(r["release_utc"]) >= since]
    if rel:
        parts += [f"📊 <b>چه منتشر شد ({'۷ روز' if weekly else '۲۴ ساعت'} اخیر)</b>", release_table(rel),
                  _block([release_line(r) for r in rel])]
    news = f.important_news(since)
    if not news.empty:
        parts.append(_block(["📰 <b>خبرهای مهم</b>", *[f"• {esc(r.title_fa)} ({'★' * int(r.importance)})"
                                                     for r in news.head(5).itertuples()]]))
    c = f.calendar
    if not c.empty:
        nxt = c[(c["scheduled_utc"] > now) & (c["scheduled_utc"] <= now + pd.Timedelta(days=7))]
        if not nxt.empty:
            relc = releases_cfg()
            parts.append(_block(["📅 <b>تقویم ۷ روز آینده</b>", *[
                f"• {tehran(e.scheduled_utc)} — {esc(relc.get(e.release_id, {}).get('name_fa', e.release_id))} "
                f"{'★' * int(getattr(e, 'importance', 0) or 0)}"
                for e in nxt.drop_duplicates("event_id").sort_values("scheduled_utc").itertuples()]]))
    return "\n\n".join(parts) + dashboard_link()


GLOSSARY = [
    ("امتیاز کلان", "از −۱۰۰ تا +۱۰۰؛ برآیند اثر تاریخی شاخص‌ها با تغییرات فعلی‌شان. زیر ±۱۵ برای سیگنال کافی نیست."),
    ("اثر (قوی/متوسط/ضعیف/ناچیز)", "اندازه‌ی اثر تاریخی شاخص بر همین دارایی؛ ▲▼ جهت اثر فعلی آن است. همبستگی تاریخی است، نه پیش‌بینی."),
    ("انتظار (روند/قبلی)", "اجماع رایگان در دسترس نیست؛ انتظار = میانگین انتشارهای قبلی (روند) یا مقدار قبلی."),
    ("شگفتی", "فاصله‌ی عدد منتشرشده از انتظار، نسبت به اندازه‌ی معمول این فاصله (کوچک، متوسط، بزرگ)."),
    ("برتری اثبات‌شده", "قاعده در آزمون خارج از نمونه، پس از کارمزد، سود متوسط مثبت و معنادار داشته؛ بدون آن سیگنالی ارسال نمی‌شود."),
    ("R و R:R", "R = فاصله‌ی ورود تا حد ضرر؛ R:R = نسبت سود هدف به این ریسک (در کارت سیگنال)."),
    ("دامنه‌ی محتمل ۴ هفته", "بازه‌ای که قیمت با حدود ۶۸٪ احتمال در آن می‌ماند، بر پایه‌ی نوسان اخیر."),
    ("کف/سقف روزانه", "آخرین نقطه‌ی برگشت قیمت در نمودار روزانه؛ شکستن آن یعنی تغییر ساختار روند."),
    ("GC=F و PAXG", "GC=F قیمت آتی طلای COMEX (با تأخیر)؛ PAXG توکن طلا، نماینده‌ی بلادرنگ و مبنای سطوح و سیگنال."),
    ("پیش‌موقعیت‌گیری", "حرکت یک‌طرفه‌ی بازار در ساعت‌های پیش از انتشار یک داده."),
    ("فاز اقتصادی", "یکی از ۴ فاز چرخه (بهبود، رونق، اوج، رکود) از سطح و شتاب رشد، تورم و نقدینگی."),
    ("موضع فدرال", "از −۱۰۰ (کاملاً انبساطی) تا +۱۰۰ (کاملاً انقباضی)."),
]


def glossary_message() -> str:
    """The last message of every report — it carries the report's single disclaimer."""
    return "\n".join(["📖 <b>راهنمای اصطلاحات</b>", *[f"• <b>{t}</b>: {d}" for t, d in GLOSSARY]]) + "\n\n" + DISCLAIMER


def changes_block(prev: dict, cur: dict) -> list[str]:
    """After a release: price and macro-score changes per asset since the previous report (template, no new numbers)."""
    if not prev or not prev.get("assets"):
        return ["🔄 <b>چه تغییر کرد</b>", "گزارش قبلی برای مقایسه در دسترس نیست."]
    L = [f"🔄 <b>چه تغییر کرد</b> (نسبت به گزارش {tehran(prev['asof'])})"]
    for a in ("BTC", "Gold"):
        p, c = prev["assets"].get(a) or {}, cur["assets"].get(a) or {}
        parts = []
        if p.get("price") and c.get("price"):
            parts.append(f"قیمت {num((c['price'] / p['price'] - 1) * 100, '{:+.1f}')}٪")
        moved = []
        for b, h in HORIZONS[:2]:
            s0, s1 = (p.get("scores") or {}).get(b), (c.get("scores") or {}).get(b)
            if s0 is None or s1 is None:
                continue
            if score_direction(s0) != score_direction(s1) or abs(s1 - s0) >= 5:
                moved.append(f"{h}: از {score_direction(s0)} ({nz(num(s0, '{:+.0f}'))}) به "
                             f"{score_direction(s1)} ({nz(num(s1, '{:+.0f}'))})")
        L.append(f"• {ICON[a]} {ASSET_FA[a]}: " + "، ".join(parts + (moved or ["امتیاز کلان تغییر مهمی نکرد"])))
    return L


def report_messages(f: Facts, now: pd.Timestamp, chains: dict[str, list[str]] | None = None, trigger: str | None = None,
                    names: dict | None = None, weekly: bool = False) -> list[tuple[str, str]]:
    """(kind, html) in sending order: overview, BTC, gold, full tables, glossary (with the disclaimer, once)."""
    out = [("overview", overview_message(f, now, trigger, weekly))]
    assets = [(a, k) for a, k in (("BTC", "btc"), ("Gold", "gold")) if f.asset(a) or f.macro.get(a)]
    out += [(f"report_{k}", asset_message(f, a, now, (chains or {}).get(a), trigger, names)) for a, k in assets]
    out += [(f"table_{k}", table_message(f, a, now)) for a, k in assets]
    out.append(("glossary", glossary_message()))
    return out
