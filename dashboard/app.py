"""Macro Pulse dashboard (Persian, RTL, light/dark).   streamlit run dashboard/app.py

Data: see dashboard/data.py (snapshot directory/URL now, database later). Prices in the top bar are fetched
live on every refresh (cached 60 s) and fall back to the snapshot if the exchanges are unreachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.timeutil import TEHRAN  # noqa: E402
from dashboard import charts, data  # noqa: E402
from core.fa import fa, label, ltr, word  # noqa: E402
from dashboard.cards import REGIME_FA, SECTIONS, STATUS  # noqa: E402
from dashboard.theme import CSS, tokens  # noqa: E402
from features.fed_stance import FORMULA_FA  # noqa: E402
from technicals.indicators import resample  # noqa: E402

DISCLAIMER = "این داشبورد صرفاً تحلیلی و آموزشی است و توصیه مالی یا پیشنهاد خرید و فروش نیست."

st.set_page_config(page_title="نبض کلان | Macro Pulse", page_icon="📈", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
MODE = (getattr(st.context, "theme", None) and st.context.theme.type) or "light"


# ───────────────────────────── data ─────────────────────────────

@st.cache_data(ttl=900, show_spinner="در حال بارگذاری داده‌ها…")
def get_bundle() -> data.Bundle:
    return data.load()


@st.cache_data(ttl=900, show_spinner="در حال محاسبه شاخص‌ها…")
def get_analysis(as_of: str | None) -> dict:
    return data.analysis(get_bundle(), pd.Timestamp.now(tz="UTC").date())


@st.cache_data(ttl=60, show_spinner=False)
def live_prices() -> dict:
    from collectors import crypto

    out = {}
    if os.environ.get("MACRO_PULSE_NO_LIVE") or (B.root / "DEMO").exists():  # tests / demo data: never mix in live prices
        return out
    for asset, sym in (("BTC", "BTC"), ("Gold", "PAXG")):
        try:
            r = crypto.candles(sym, "1h", limit=26)
            c = r.data["close"]
            out[asset] = {"price": float(c.iloc[-1]), "chg24": float((c.iloc[-1] / c.iloc[-25] - 1) * 100), "venue": r.venue,
                          "ts": r.data["ts"].iloc[-1]}
        except Exception:  # noqa: BLE001 — shown as "snapshot price" instead
            pass
    return out


def fmt_tehran(ts) -> str:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts.tz_convert(TEHRAN).strftime("%Y-%m-%d %H:%M")


try:
    B = get_bundle()
except FileNotFoundError as exc:
    st.error(f"داده‌ای یافت نشد: {exc}")
    st.stop()
A = get_analysis(B.as_of)
T = tokens(MODE)


def series_or_feature(key: str, src: str) -> pd.Series | None:
    if src == "series":
        return B.series(key)
    f = A["features"].get(key)
    return None if f is None or f.empty else f.set_index("date")["value"].sort_index()


# ───────────────────────────── header ─────────────────────────────

st.title("نبض کلان — اثر اقتصاد کلان آمریکا بر بیت‌کوین و طلا")
if (B.root / "DEMO").exists():
    st.warning("⚠️ داده نمایشی (مصنوعی) — این اعداد واقعی نیستند؛ فقط برای آزمایش رابط کاربری.")
st.caption(f"آخرین به‌روزرسانی داده: {fmt_tehran(B.as_of) if B.as_of else 'نامشخص'} (به وقت تهران) · {DISCLAIMER}")


@st.fragment(run_every=60)
def fmt_price(asset: str, v: float) -> str:
    return f"{v:,.0f}" if asset == "BTC" else f"{v:,.2f}"


def span(asset: str, pair) -> str:
    return "—" if not pair else f"{fmt_price(asset, min(pair))} تا {fmt_price(asset, max(pair))}"


NAMES = (dict(zip(B.calendar["name_en"], B.calendar["name_fa"]))
         if B.calendar is not None and {"name_en", "name_fa"} <= set(B.calendar.columns) else {})


def tr(msg: str) -> str:
    return fa(msg, NAMES)


IND_FA = {c.key: c.title_fa for cs in SECTIONS.values() for c in cs}


def ind_name(k: str) -> str:
    return B.registry[k].name_fa if k in B.registry else IND_FA.get(k, k)


CURVE_FA = {"spread_10y_2y": "۱۰−۲", "spread_10y_3m": "۱۰−۳ماهه", "spread_30y_2y": "۳۰−۲"}


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Display copy: timestamps → dates, English labels → Persian, missing → blank."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]) or c in ("start", "end", "resteepen_date", "depth_date"):
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        elif out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else
                                (fa(v) if isinstance(v, str) else v))
    return out


TF_FA = {"1d": "روزانه", "4h": "۴ساعته", "1h": "ساعتی"}


def top_bar():
    live = live_prices()
    cols = st.columns(6)
    snap = {a: B.signals.get("assets", [{}] * 2)[i].get("price") if B.signals else None for i, a in enumerate(("BTC", "Gold"))}
    for col, (asset, title) in zip(cols[:2], (("BTC", "بیت‌کوین (BTC)"), ("Gold", "طلا (PAXG، نماینده)"))):
        if asset in live:
            p = live[asset]
            chg = f"{p['chg24']:+.2f}٪"
            col.metric(title, fmt_price(asset, p["price"]), f"{ltr(chg)} (۲۴ ساعت)", delta_color="off", delta_arrow="off",
                       help=f"قیمت زنده از {p['venue']} · {fmt_tehran(p['ts'])} تهران" +
                            (" · طلا: PAXG توکن طلا (هر توکن = یک اونس) نماینده بلادرنگ XAUUSD است" if asset == "Gold" else ""),
                       border=True)
        else:
            col.metric(title, fmt_price(asset, snap[asset]) if snap.get(asset) else "—", "اسنپ‌شات", delta_color="off", delta_arrow="off",
                       border=True, help="قیمت زنده در دسترس نبود؛ آخرین قیمت اسنپ‌شات نمایش داده شده است.")
    rn = A["regime_now"]
    p = rn["probabilities"][rn["regime"]]
    cols[2].metric("فاز چرخه اقتصادی", REGIME_FA[rn["regime"]], f"احتمال {ltr(f'{p * 100:.0f}٪')}", delta_color="off", delta_arrow="off", border=True,
                   help="مدل قاعده‌محور بدون نگاه به آینده: سطح رشد × شتاب (رشد در برابر فشار تورم/نقدینگی). برای جزئیات به زبانه «رژیم‌ها» بروید.")
    st_score = A["stance"].score
    cols[3].metric("موضع فدرال", "—" if st_score is None else f"{st_score:+.0f}", f"{len(A['stance'].used)}/۵ جزء",
                   delta_color="off", delta_arrow="off", border=True, help=FORMULA_FA)
    ms = B.macro_scores or {}
    for col, asset, title in ((cols[4], "BTC", "امتیاز کلان BTC"), (cols[5], "Gold", "امتیاز کلان طلا")):
        m = ms.get(asset, {})
        med, lng = (m.get("medium") or {}).get("score"), (m.get("long") or {}).get("score")
        col.metric(title, "—" if med is None else f"{med:+.0f}", f"بلندمدت {ltr(f'{lng:+.0f}')}" if lng is not None else "",
                   delta_color="off", delta_arrow="off", border=True,
                   help="−۱۰۰ تا +۱۰۰؛ میانگین وزنی وضعیت فعلی شاخص‌ها × ضریب اثر تاریخی آن‌ها. عدد اصلی = افق ۴ هفته، زیرنویس = ۶ ماه.")


top_bar()


# ───────────────────────────── cards ─────────────────────────────

def card(col, cd) -> None:
    s = series_or_feature(cd.key, cd.src)
    if s is None or s.empty:
        col.metric(cd.title_fa, "—", help=cd.desc_fa + "\n\nداده در دسترس نیست.", border=True)
        return
    v = float(s.iloc[-1])
    prev = float(s.iloc[-1 - cd.lookback]) if len(s) > cd.lookback else np.nan
    delta = None if np.isnan(prev) else f"{ltr(f'{v - prev:+,.2f}')} نسبت به {ltr(f'{s.index[-1 - cd.lookback]:%Y-%m-%d}')}"
    if cd.src == "series":
        rep = B.availability.get(cd.key, {})
        icon, status = STATUS.get(rep.get("status", "ok"), STATUS["ok"])
        meta = B.registry[cd.key]
        src = f"{meta.source.upper()} · {meta.source_id}" + (" · نماینده (proxy)" if meta.proxy else "")
    else:
        icon, status, src = "🧮", "مشتق‌شده", "محاسبه از سری‌های پایه (features/derived.py)"
    unit_in_value = cd.unit in ("%", "$")
    title = f"{icon} {cd.title_fa}" + (f" ({cd.unit})" if cd.unit and not unit_in_value and cd.unit not in ("شاخص", "نسبت") else "")
    col.metric(title, cd.fmt.format(v) + (cd.unit if unit_in_value else ""),
               delta, delta_color="off", delta_arrow="off", border=True,
               chart_data=s.tail(90).tolist(), chart_type="line",
               help=f"{cd.desc_fa}\n\nمنبع: {src}\n\nآخرین مشاهده: {s.index[-1]:%Y-%m-%d} · وضعیت: {status}")


def card_grid(cards, per_row: int = 4) -> None:
    for i in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for col, cd in zip(cols, cards[i:i + per_row]):
            card(col, cd)


def since(s: pd.Series | None, years: int = 10) -> pd.Series | None:
    return None if s is None else s[s.index >= s.index[-1] - pd.DateOffset(years=years)]


# ───────────────────────────── tabs ─────────────────────────────

TAB_NAMES = ["بیت‌کوین و طلا", *SECTIONS.keys(), "انتظارات نرخ", "تقویم و اخبار", "ضرایب اثر", "رژیم‌ها", "گزارش‌ها", "درباره داده‌ها"]
tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

with tabs["بیت‌کوین و طلا"]:
    st.info(DISCLAIMER)
    for res in (B.signals or {}).get("assets", []):
        asset = res["asset"]
        st.subheader({"BTC": "بیت‌کوین", "Gold": "طلا"}[asset] + (f" — {res['proxy']}" if res.get("proxy") else ""))
        bars = B.bars("BTC" if asset == "BTC" else "PAXG")
        levels = []
        tech = res.get("technical", {})
        for tf in ("4h", "1d"):
            t4 = tech.get(tf, {})
            if t4.get("last_swing_high"):
                levels.append({"price": t4["last_swing_high"], "label": f"سقف سوئینگ {TF_FA[tf]}", "dash": "dot"})
            if t4.get("last_swing_low"):
                levels.append({"price": t4["last_swing_low"], "label": f"کف سوئینگ {TF_FA[tf]}", "dash": "dot"})
        vp = res.get("context", {}).get("volume_profile_30d", {})
        if vp.get("poc"):
            levels.append({"price": vp["poc"], "label": "پرحجم‌ترین قیمت ۳۰ روزه", "dash": "dash"})
        for w in res.get("context", {}).get("orderbook_walls", [])[:3]:
            levels.append({"price": w["price"], "label": f"دیوار {'خرید' if w['side'] == 'bid' else 'فروش'} {w['usd_m']:,.0f} میلیون دلار", "dash": "dashdot"})
        for s in res.get("signals", []) + res.get("watchlist", []):
            levels.append({"price": s["stop_loss"], "label": "حد ضرر", "dash": "solid", "color": T["series"][7]})
            for k, tp in enumerate(s["targets"], 1):
                levels.append({"price": tp, "label": f"هدف {k}", "dash": "solid", "color": T["series"][2]})
        if bars is not None:
            b4 = resample(bars[bars["ts"] >= bars["ts"].iloc[-1] - pd.Timedelta(days=90)], "4h")
            st.plotly_chart(charts.price_levels(b4, asset, MODE, levels), width="stretch")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**وضعیت تکنیکال**")
            tdf = pd.DataFrame(tech).T[["trend", "structure", "last_event", "rsi", "macd_hist", "atr"]]
            tdf[["trend", "structure", "last_event"]] = tdf[["trend", "structure", "last_event"]].map(word)
            tdf.index = [TF_FA.get(i, i) for i in tdf.index]
            st.dataframe(tdf.rename(columns={"trend": "روند", "structure": "ساختار", "last_event": "آخرین رویداد", "rsi": "RSI",
                                          "macd_hist": "هیستوگرام MACD", "atr": "ATR"}), width="stretch")
            er = res.get("expected_range", {})
            if er:
                st.markdown(f"**دامنه محتمل:** یک روز {span(asset, er.get('1d'))} · چهار هفته (±۱σ) "
                            f"{span(asset, er.get('4w_1sigma'))} — نوسان ۴ هفته {er.get('4w_sigma_pct', 0):.1f}٪، "
                            f"رانش کلان {er.get('macro_drift_pct', 0):+.1f}٪")
            if res.get("context", {}).get("liquidation_clusters_estimate"):
                st.markdown("**خوشه‌های تخمینی لیکوئیدیشن** (تخمین از تغییرات OI × اهرم‌های رایج):  " + "، ".join(
                    f"{x['side']} {x['price']:,.0f} (${x['usd_m']}M)" for x in res["context"]["liquidation_clusters_estimate"][:6]))
        with c2:
            bt = res.get("backtest", {})
            st.markdown(f"**بک‌تست خارج از نمونه** ({bt.get('period', ['', ''])[0]} → {bt.get('period', ['', ''])[1]}، "
                        f"کارمزد و لغزش لحاظ شده)")
            rows = []
            for side, lab in (("all", "همه"), ("long", "خرید"), ("short", "فروش")):
                sm = bt.get("summary", {}).get(side, {})
                rows.append({"سمت": lab, "معاملات": sm.get("n_trades", 0),
                             "نرخ برد": f"{sm['win_rate']:.0%}" if sm.get("win_rate") is not None else "—",
                             "میانگین R": sm.get("avg_r"), "افت حداکثر (R)": sm.get("max_drawdown_r")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for title, items in (("سیگنال‌های فعال", res.get("signals", [])), ("فهرست نظارت (مسدودشده)", res.get("watchlist", []))):
            st.markdown(f"**{title}**")
            if not items:
                st.caption("هیچ‌کدام" + (" — ستاپ فعال با شرایط کامل وجود ندارد." if title.startswith("سیگنال") else ""))
            for s in items:
                with st.container(border=True):
                    wr = "—" if s.get("win_rate") is None else f"{s['win_rate']:.0%}"
                    st.markdown(f"**{'خرید' if s['direction'] == 'long' else 'فروش'}** · ورود {span(asset, s['entry_zone'])} · "
                                f"حد ضرر {fmt_price(asset, s['stop_loss'])} · اهداف "
                                f"{'، '.join(fmt_price(asset, t) for t in s['targets'])} · نسبت سود به زیان "
                                f"{'، '.join(f'{r:.2f}' for r in s['rr'])} · نرخ برد {wr} در {s['n_backtest']} معامله · "
                                f"اطمینان {word(s['confidence'])}")
                    if s.get("blocked_by"):
                        st.caption("دلیل مسدود شدن: " + "؛ ".join(map(tr, s["blocked_by"])))
                    st.caption("ابطال: " + fa(s["invalidation"]))
                    st.caption("دلایل تکنیکال: " + "؛ ".join(map(tr, s["reasons_technical"])))
                    if s.get("risks"):
                        st.caption("ریسک‌ها: " + "؛ ".join(map(tr, s["risks"])))
        if res.get("risks"):
            st.markdown("**ریسک‌های فعلی:** " + " · ".join(map(tr, res["risks"])))
        st.divider()

for section, cards_ in SECTIONS.items():
    with tabs[section]:
        card_grid(cards_)
        if section == "نرخ‌ها و فدرال":
            st.plotly_chart(charts.lines({"سقف هدف": since(B.series("fed_target_upper")), "EFFR": since(B.series("fed_funds_eff_daily")),
                                          "۲ ساله": since(B.series("ust_2y"))}, MODE, "نرخ سیاستی و بازده ۲ ساله", "%"),
                            width="stretch")
        elif section == "تورم":
            f = A["features"]
            fig = charts.lines({k: since(f[key].set_index("date")["value"]) for k, key in
                                (("CPI", "cpi_yoy"), ("CPI هسته", "core_cpi_yoy"), ("PCE هسته", "core_pce_yoy")) if key in f},
                               MODE, "تورم سالانه (٪) — خط‌چین: هدف ۲٪ فدرال", "%")
            fig.add_hline(y=2, line={"color": T["muted"], "width": 1, "dash": "dash"})
            st.plotly_chart(fig, width="stretch")
        elif section == "نقدینگی":
            nl = series_or_feature("net_liquidity_weekly", "feature")
            st.plotly_chart(charts.lines({"نقدینگی خالص": since(nl)}, MODE, "نقدینگی خالص = WALCL − TGA − RRP (میلیارد دلار)",
                                         "میلیارد دلار"), width="stretch")
            seg = A.get("qe_qt_segments")
            if seg is not None and not seg.empty:
                s2 = seg.assign(regime=seg["label"].map({1: "QE", 0: "خنثی", -1: "QT"}))[["start", "end", "regime", "weeks"]]
                st.markdown("**دوره‌های QE/QT (از روند ترازنامه فدرال)**")
                st.dataframe(pretty(s2.tail(10).iloc[::-1]).rename(columns={"start": "شروع", "end": "پایان", "regime": "وضعیت",
                                                                            "weeks": "هفته"}), hide_index=True, width="stretch")
        elif section == "بازار کار":
            st.plotly_chart(charts.lines({"نرخ بیکاری": since(B.series("unemployment_rate"), 25)}, MODE, "نرخ بیکاری", "%"),
                            width="stretch")
        elif section == "منحنی بازده":
            fig = charts.lines({"۱۰−۲": B.series("spread_10y_2y"), "۱۰−۳ماهه": B.series("spread_10y_3m")}, MODE,
                               "اسپردهای منحنی بازده — نواحی سایه‌دار: دوره‌های وارونگی ۱۰−۲؛ خط‌چین: بازگشت شیب", "واحد درصد",
                               zero_line=True, height=380)
            st.plotly_chart(charts.with_episodes(fig, A["episodes"]["spread_10y_2y"], MODE), width="stretch")
            st.caption("وضعیت فعلی: " + "، ".join(f"{CURVE_FA.get(k, k)}: {word(v)}" for k, v in A["curve_state"].items()))
            ep = A["episodes"]["spread_10y_2y"]
            st.dataframe(pretty(ep[["start", "end", "duration_days", "depth", "resteepen_date", "label", "brief"]]).rename(columns={
                "start": "شروع", "end": "پایان", "duration_days": "روز", "depth": "عمق (واحد درصد)", "resteepen_date": "بازگشت شیب",
                "label": "توضیح", "brief": "کوتاه (نادیده در نمودار)"}), hide_index=True, width="stretch")
        elif section == "اعتبار":
            st.plotly_chart(charts.lines({"HY OAS": B.series("hy_oas"), "Baa−10Y": since(B.series("baa_10y_spread"))}, MODE,
                                         "اسپردهای اعتباری", "واحد درصد"), width="stretch")
        elif section == "دلار و بین‌بازاری":
            f = A["features"]
            st.plotly_chart(charts.lines({lab: since(f[k].set_index("date")["value"], 3) for lab, k in
                                          (("مس/طلا", "copper_gold_ratio_z252"), ("S&P/طلا", "spx_gold_ratio_z252"),
                                           ("XLF/XLU", "xlf_xlu_ratio_z252"), ("نزدک/S&P", "ndx_spx_ratio_z252")) if k in f},
                                         MODE, "نسبت‌های بین‌بازاری — امتیاز z یک‌ساله", "z", zero_line=True), width="stretch")
        elif section == "کالاها":
            st.plotly_chart(charts.lines({"WTI": since(B.series("wti_futures"), 3), "برنت": since(B.series("brent_futures"), 3)},
                                         MODE, "نفت خام (قرارداد آتی جلو)", "دلار/بشکه"), width="stretch")

with tabs["انتظارات نرخ"]:
    fw = A["fedwatch"]
    st.markdown("**احتمال تصمیمات FOMC از قراردادهای آتی وجوه فدرال (روش CME FedWatch، بازسازی‌شده)**")
    if fw.empty:
        st.warning("داده آتی یا تقویم FOMC در دسترس نیست.")
    else:
        st.dataframe(pd.DataFrame({
            "جلسه": fw["meeting"], "کاهش": fw["p_cut"], "ثابت": fw["p_hold"], "افزایش": fw["p_hike"],
            "نرخ ضمنی پس از جلسه": fw["implied_rate_after"], "تغییر تجمعی (bp)": fw["cumulative_vs_today_bp"]}),
            hide_index=True, width="stretch",
            column_config={k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="percent")
                           for k in ("کاهش", "ثابت", "افزایش")})
        fi = A["fedwatch_inputs"] or {}
        tgt = fi.get("target") or [None, None]
        effr = fi.get("effr_now")
        st.caption("احتمال‌ها ضمنیِ بازار هستند (شامل صرف ریسک)، نه پیش‌بینی. " +
                   (f"نرخ مؤثر فعلی {ltr(f'{effr:.2f}٪')}، " if effr is not None else "") +
                   (f"محدوده هدف {ltr(f'{tgt[0]:.2f}–{tgt[1]:.2f}٪')}، " if tgt[0] is not None else "") +
                   f"{fi.get('contracts', 0)} قرارداد آتی.")
    st.markdown("**امتیاز موضع فدرال**")
    st.caption(FORMULA_FA)
    stn = A["stance"]
    st.dataframe(pd.DataFrame([{"جزء": label(k), "مقدار [−۱، +۱]": v, "ورودی‌ها": "، ".join(
        f"{label(ik)}: {round(iv, 3) if isinstance(iv, float) else iv}" for ik, iv in (stn.inputs.get(k) or {}).items())
        or "داده‌ای موجود نیست"} for k, v in stn.components.items()]), hide_index=True, width="stretch")

with tabs["تقویم و اخبار"]:
    cal = B.calendar.copy()
    now = pd.Timestamp.now(tz="UTC")
    if not cal.empty:
        cal = cal[(cal["scheduled_utc"] >= now - pd.Timedelta(hours=6)) & (cal["scheduled_utc"] <= now + pd.Timedelta(days=14))]
        rows = []
        from collectors.calendar import tracked_releases
        rel = {r["id"]: r for r in tracked_releases()}
        for e in cal.sort_values("scheduled_utc").drop_duplicates("event_id").itertuples():
            r = rel.get(e.release_id, {})
            prev = None
            for k in r.get("series_keys", []):
                s = B.series(k)
                if s is not None:
                    prev = f"{B.registry[k].name_fa if k in B.registry else k}: {s.iloc[-1]:,.2f} ({s.index[-1]:%Y-%m})"
                    break
            rows.append({"زمان (تهران)": fmt_tehran(e.scheduled_utc), "رویداد": r.get("name_fa", e.release_id),
                         "اهمیت": "★" * int(r.get("importance", 1)), "قبلی": prev or "—",
                         "پیش‌بینی (اجماع)": "رایگان در دسترس نیست"})
        st.markdown("**تقویم اقتصادی ۱۴ روز آینده**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**آخرین اخبار (فدرال، BEA، GDELT)** — تیترها به زبان اصلی؛ ترجمه و طبقه‌بندی اهمیت با Claude در فاز ۶")
    if not B.news.empty:
        n = B.news.head(40).copy()
        n["زمان (تهران)"] = n["published_utc"].map(lambda x: fmt_tehran(x) if pd.notna(x) else "—")
        n["source"] = n["source"].map(label)
        st.dataframe(n[["زمان (تهران)", "source", "title", "url"]].rename(columns={"source": "منبع", "title": "عنوان"}),
                     hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn("لینک")})
    else:
        st.caption("خبری در اسنپ‌شات نیست.")

with tabs["ضرایب اثر"]:
    co = B.coefficients
    if co.empty:
        st.warning("ضرایب اثر هنوز محاسبه نشده‌اند.")
    else:
        sample = st.radio("نمونه", ["full", "recent"], horizontal=True,
                          format_func=lambda x: {"full": "کل دوره", "recent": "۳ سال اخیر (فقط BTC)"}[x])
        heads = {"1h": "۱ ساعت", "24h": "۲۴ ساعت", "4w": "۴ هفته", "6m": "۶ ماه"}
        d = co[co["horizon"].isin(heads) & (co["sample"] == sample)].copy()
        d["col"] = d["asset"].map({"BTC": "BTC", "Gold": "طلا"}) + " · " + d["horizon"].map(heads)
        d["indicator"] = d["indicator"].map(ind_name)
        st.plotly_chart(charts.heatmap(d, MODE), width="stretch")
        st.caption("ضریب اثر: ۰ تا ۱۰ با علامت؛ ترکیبی از اندازه اثر، معنی‌داری آماری و اندازه نمونه. همبستگی تاریخی است نه پیش‌بینی. "
                   "افق‌های کوتاه (۱ و ۲۴ ساعت) مربوط به شگفتی انتشار داده‌ها هستند.")
        tbl = d[["indicator", "col", "coefficient", "t_stat", "n", "n_eff", "confidence", "conflict"]].sort_values(
            "coefficient", key=abs, ascending=False)
        tbl["confidence"] = tbl["confidence"].map(word)
        st.dataframe(tbl.rename(columns={"indicator": "شاخص", "col": "دارایی · افق", "coefficient": "ضریب", "t_stat": "آماره t",
                                         "n": "تعداد نمونه", "n_eff": "نمونه مؤثر", "confidence": "اطمینان",
                                         "conflict": "خلاف نظریه (نیم‌وزن)"}),
                     hide_index=True, width="stretch", column_config={"ضریب": st.column_config.NumberColumn(format="%+.1f"),
                                                                     "آماره t": st.column_config.NumberColumn(format="%.2f")})

with tabs["رژیم‌ها"]:
    rn = A["regime_now"]
    st.markdown(f"**فاز فعلی: {REGIME_FA[rn['regime']]}** (داده تا {ltr(rn['date'])}) — " +
                "، ".join(f"{REGIME_FA[k]} {ltr(f'{v * 100:.0f}٪')}" for k, v in rn["probabilities"].items()))
    st.markdown("**شرایط تغییر فاز:**")
    for t_ in rn["triggers"]:
        st.caption(fa(t_))
    labels = A["regime_rules"]["regime"]
    for asset, key in (("BTC", "btc_usd_daily"), ("Gold", "gold_futures")):
        s = B.series(key)
        if s is not None:
            st.plotly_chart(charts.regime_timeline(labels, s, asset, MODE), width="stretch")
    st.markdown("**بازده یک‌ماهه بعدی در هر فاز**")
    rp = A["regime_perf"].copy()
    rp["asset"], rp["regime"] = rp["asset"].map({"BTC": "BTC", "Gold": "طلا"}), rp["regime"].map(word)
    st.dataframe(rp.rename(columns={"asset": "دارایی", "regime": "فاز", "months": "ماه", "mean_fwd_1m_pct": "میانگین بازده ماه بعد (٪)",
                                    "median_fwd_1m_pct": "میانه بازده ماه بعد (٪)", "hit_rate": "درصد ماه‌های مثبت",
                                    "ann_vol_pct": "نوسان سالانه (٪)"}),
                 hide_index=True, width="stretch", column_config={"درصد ماه‌های مثبت": st.column_config.NumberColumn(format="percent")})
    st.caption("بازده ماه بعد از پایان هر ماه با فاز همان ماه (بدون نگاه به آینده) — آمار توصیفی است نه پیش‌بینی.")
    if A.get("regime_hmm"):
        h = A["regime_hmm"].probs.iloc[-1]
        st.caption(f"مدل HMM (برای مقایسه): {REGIME_FA[h['regime']]} — پارامترهای HMM روی کل تاریخچه برازش شده‌اند "
                   "(احتمال‌ها فقط رو به جلو فیلتر می‌شوند)؛ فقط برای مقایسه، نه بک‌تست.")
    if A.get("nber_check"):
        nb = A["nber_check"]
        hit = nb.get("classified_recession_or_peak", 0) * 100
        false_ = nb.get("expansion_months_classified_recession", 0) * 100
        st.caption(f"مقایسه با رکودهای رسمی NBER: از {ltr(nb.get('recession_months'))} ماه رکود رسمی، {ltr(f'{hit:.0f}٪')} "
                   f"به‌عنوان رکود یا اوج طبقه‌بندی شده؛ {ltr(f'{false_:.0f}٪')} از ماه‌های رونق رسمی به‌عنوان رکود. "
                   "فازهای این مدل با تاریخ‌گذاری رسمی رکود یکی نیستند.")

with tabs["گزارش‌ها"]:
    st.caption("گزارش‌های فنی خط لوله (به انگلیسی، برای بررسی دقیق اعداد). گزارش‌های فارسی روزانه/هفتگی با Claude در فاز ۶ "
               "اضافه می‌شوند؛ تاریخچه کامل پس از فعال شدن پایگاه داده (Supabase).")
    names = {"signals": "سیگنال‌ها و تکنیکال", "impact_summary": "ضرایب اثر", "features_summary": "شاخص‌ها و رژیم",
             "data_availability": "گزارش دسترس‌پذیری داده"}
    for n, md in B.reports.items():
        with st.expander(names.get(n, n)):
            st.markdown(md)

with tabs["درباره داده‌ها"]:
    rows = [{"کلید": k, "نام": B.registry[k].name_fa if k in B.registry else k, "وضعیت": STATUS.get(r.get("status"), ("", r.get("status")))[1],
             "منبع": f"{r.get('source')} · {r.get('source_id')}", "آخرین داده": r.get("last_date"), "تواتر": label(r.get("frequency") or ""),
             "نماینده": "بله" if r.get("proxy") else ""} for k, r in B.availability.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("شکاف‌های داده و منابع جایگزین: DATA_GAPS.md در مخزن.")
