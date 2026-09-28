"""Derived series from raw observations (latest values; the regime model uses the PIT loader instead).

All outputs are (date, value) frames keyed by a feature key. Units are stated in FEATURES.
Missing inputs never get filled: an output is produced only where all its inputs exist
(except explicit as-of alignment of slower series onto faster ones, e.g. weekly WALCL on daily TGA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.transforms import annualized_3m, as_series, mom, ratio, to_frame, trend_state, yoy, zscore


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    name_en: str
    name_fa: str
    units: str
    frequency: str


FEATURES: dict[str, FeatureSpec] = {}


def _spec(key, name_en, name_fa, units, freq):
    FEATURES[key] = FeatureSpec(key, name_en, name_fa, units, freq)


INFLATION = {"cpi": "CPI", "core_cpi": "Core CPI", "ppi_final_demand": "PPI", "pce": "PCE", "core_pce": "Core PCE"}
for k, n in INFLATION.items():
    _spec(f"{k}_mom", f"{n} MoM", f"{n} ماهانه", "%", "M")
    _spec(f"{k}_yoy", f"{n} YoY", f"{n} سالانه", "%", "M")
    _spec(f"{k}_3m_ann", f"{n} 3-month annualized", f"{n} سه‌ماهه سالانه‌شده", "%", "M")
_spec("wage_yoy", "Average hourly earnings YoY", "رشد سالانه دستمزد", "%", "M")
_spec("openings_per_unemployed", "Job openings per unemployed", "نسبت فرصت شغلی به بیکار", "ratio", "M")
_spec("payrolls_mom_change", "Nonfarm payrolls monthly change", "تغییر ماهانه اشتغال", "thousands", "M")
_spec("claims_4wk_avg", "Initial claims 4-week average", "میانگین ۴ هفته‌ای مدعیان بیکاری", "persons", "W")
_spec("real_rate_ffr_core_pce", "Real fed funds (EFFR − core PCE YoY)", "نرخ بهره واقعی (وجوه فدرال منهای PCE هسته)", "%", "M")
_spec("fed_target_mid", "Fed funds target midpoint", "میانه دامنه هدف نرخ بهره", "%", "D")
_spec("spread_30y_2y", "30Y − 2Y spread", "اسپرد ۳۰ ساله منهای ۲ ساله", "pp", "D")
_spec("net_liquidity", "Net liquidity (WALCL − TGA − RRP), daily", "نقدینگی خالص (روزانه)", "USD bn", "D")
_spec("net_liquidity_weekly", "Net liquidity (WALCL − TGA − RRP), Wednesday", "نقدینگی خالص (هفتگی)", "USD bn", "W")
_spec("net_liquidity_13w_change", "Net liquidity 13-week change", "تغییر ۱۳ هفته‌ای نقدینگی خالص", "USD bn", "W")
_spec("fed_balance_sheet_13w_ann", "Fed balance sheet 13-week change, annualized", "تغییر سالانه‌شده ترازنامه فدرال", "%", "W")
_spec("global_cb_assets_usd", "Fed + ECB + BoJ assets in USD", "دارایی بانک‌های مرکزی بزرگ (دلاری)", "USD bn", "M")
_spec("global_cb_assets_yoy", "Fed + ECB + BoJ assets in USD, YoY", "رشد سالانه دارایی بانک‌های مرکزی", "%", "M")
_spec("m2_yoy", "M2 YoY", "رشد سالانه M2", "%", "M")
_spec("gold_silver_ratio", "Gold / Silver", "نسبت طلا به نقره", "ratio", "D")
_spec("copper_gold_ratio", "Copper / Gold (×1000)", "نسبت مس به طلا", "ratio×1000", "D")
_spec("spx_gold_ratio", "S&P 500 / Gold", "نسبت S&P 500 به طلا", "ratio", "D")
_spec("xlf_xlu_ratio", "Financials / Utilities (XLF/XLU)", "نسبت مالی به خدمات عمومی", "ratio", "D")
_spec("ndx_spx_ratio", "Nasdaq 100 / S&P 500", "نسبت نزدک به S&P", "ratio", "D")
for r in ("gold_silver_ratio", "copper_gold_ratio", "spx_gold_ratio", "xlf_xlu_ratio", "ndx_spx_ratio"):
    _spec(f"{r}_z252", f"{FEATURES[r].name_en} z-score (1y)", f"{FEATURES[r].name_fa} (امتیاز z)", "z", "D")
    _spec(f"{r}_trend", f"{FEATURES[r].name_en} trend (50/200d)", f"{FEATURES[r].name_fa} (روند)", "+1/−1", "D")
_spec("qe_qt_regime", "Balance-sheet regime (+1 QE, 0 neutral, −1 QT)", "رژیم ترازنامه (QE/QT)", "+1/0/−1", "W")


def _s(data: dict[str, pd.DataFrame], key: str) -> pd.Series | None:
    df = data.get(key)
    return None if df is None or df.empty else as_series(df)


def net_liquidity(walcl: pd.Series, tga: pd.Series, rrp: pd.Series) -> pd.Series:
    """USD bn. WALCL/TGA in USD mn, RRP in USD bn. Aligned on TGA/RRP dates with WALCL carried forward
    from its latest Wednesday (the H.4.1 level stays the latest official figure until the next week)."""
    j = pd.concat([tga.rename("tga"), rrp.rename("rrp")], axis=1, join="inner").dropna()
    w = walcl.reindex(j.index.union(walcl.index)).ffill().reindex(j.index)
    first = walcl.index.min()
    out = w / 1000.0 - j["tga"] / 1000.0 - j["rrp"]
    return out[out.index >= first].dropna()


def qe_qt_regime(walcl: pd.Series, threshold_ann_pct: float = 5.0, min_weeks: int = 8) -> pd.Series:
    """Label weekly balance-sheet regimes: 13-week change annualized above +threshold → QE (+1),
    below −threshold → QT (−1), else 0. Runs shorter than `min_weeks` are absorbed into the prior regime
    so one-off operations (e.g. year-end repo swings) do not flip the label."""
    ann = ((walcl / walcl.shift(13)) ** 4 - 1.0) * 100.0
    raw = pd.Series(np.select([ann > threshold_ann_pct, ann < -threshold_ann_pct], [1, -1], 0), index=walcl.index)
    raw[ann.isna()] = np.nan
    lab = raw.copy()
    run_id = (raw != raw.shift()).cumsum()
    for _, grp in raw.groupby(run_id):
        if len(grp) < min_weeks:
            prev = lab.loc[:grp.index[0]].iloc[:-1].dropna()
            lab.loc[grp.index] = prev.iloc[-1] if len(prev) else grp.iloc[0]
    return lab.dropna()


def regime_segments(labels: pd.Series) -> pd.DataFrame:
    """Contiguous (start, end, label) segments of a regime label series."""
    run_id = (labels != labels.shift()).cumsum()
    rows = [{"start": g.index[0], "end": g.index[-1], "label": int(g.iloc[0]), "weeks": len(g)} for _, g in labels.groupby(run_id)]
    return pd.DataFrame(rows)


def compute(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.Series] = {}
    for k in INFLATION:
        s = _s(data, k)
        if s is not None:
            out[f"{k}_mom"], out[f"{k}_yoy"], out[f"{k}_3m_ann"] = mom(s), yoy(s), annualized_3m(s)
    if (s := _s(data, "avg_hourly_earnings")) is not None:
        out["wage_yoy"] = yoy(s)
    if (a := _s(data, "jolts_openings")) is not None and (b := _s(data, "unemployed_level")) is not None:
        out["openings_per_unemployed"] = ratio(a, b)
    if (s := _s(data, "nonfarm_payrolls")) is not None:
        from features.transforms import monthly_regular
        out["payrolls_mom_change"] = monthly_regular(s).diff()
    if (s := _s(data, "initial_claims")) is not None:
        out["claims_4wk_avg"] = s.rolling(4).mean()
    if (f := _s(data, "fed_funds_eff_monthly")) is not None and "core_pce_yoy" in out:
        j = pd.concat([f, out["core_pce_yoy"]], axis=1, join="inner").dropna()
        out["real_rate_ffr_core_pce"] = j.iloc[:, 0] - j.iloc[:, 1]
    if (u := _s(data, "fed_target_upper")) is not None and (lo := _s(data, "fed_target_lower")) is not None:
        out["fed_target_mid"] = pd.concat([u, lo], axis=1, join="inner").mean(axis=1)
    if (a := _s(data, "ust_30y")) is not None and (b := _s(data, "ust_2y")) is not None:
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        out["spread_30y_2y"] = j.iloc[:, 0] - j.iloc[:, 1]

    walcl, rrp = _s(data, "fed_balance_sheet"), _s(data, "reverse_repo")
    if walcl is not None and rrp is not None:
        if (tga_d := _s(data, "tga_daily")) is not None:
            out["net_liquidity"] = net_liquidity(walcl, tga_d, rrp)
        if (tga_w := _s(data, "tga_weekly")) is not None:
            # Wednesday levels; RRP for the same Wednesday (as-of if that day is missing)
            rrp_w = rrp.reindex(rrp.index.union(walcl.index)).ffill().reindex(walcl.index)
            j = pd.concat([walcl, tga_w, rrp_w], axis=1, join="inner").dropna()
            nl_w = j.iloc[:, 0] / 1000.0 - j.iloc[:, 1] / 1000.0 - j.iloc[:, 2]
            out["net_liquidity_weekly"] = nl_w
            out["net_liquidity_13w_change"] = nl_w.diff(13)
    if walcl is not None:
        out["fed_balance_sheet_13w_ann"] = ((walcl / walcl.shift(13)) ** 4 - 1.0) * 100.0
        out["qe_qt_regime"] = qe_qt_regime(walcl)

    ecb, boj = _s(data, "ecb_assets"), _s(data, "boj_assets")
    eur, jpy = _s(data, "usd_per_eur"), _s(data, "jpy_per_usd")
    if walcl is not None and ecb is not None and boj is not None and eur is not None and jpy is not None:
        mavg = lambda s: s.resample("MS").mean()  # noqa: E731 — monthly averages of weekly/daily data
        fed_m = mavg(walcl) / 1000.0
        ecb_m = mavg(ecb) * mavg(eur) / 1000.0
        boj_m = boj.resample("MS").last() * 0.1 / mavg(jpy)  # JPY 100mn → USD bn
        g = pd.concat([fed_m, ecb_m, boj_m], axis=1, join="inner").dropna().sum(axis=1)
        out["global_cb_assets_usd"] = g
        out["global_cb_assets_yoy"] = yoy(g)
    if (s := _s(data, "m2")) is not None:
        out["m2_yoy"] = yoy(s)

    pairs = {"gold_silver_ratio": ("gold_futures", "silver_futures", 1.0),
             "copper_gold_ratio": ("copper_futures", "gold_futures", 1000.0),
             "spx_gold_ratio": ("spx", "gold_futures", 1.0),
             "xlf_xlu_ratio": ("xlf", "xlu", 1.0),
             "ndx_spx_ratio": ("ndx", "spx", 1.0)}
    for key, (a, b, mult) in pairs.items():
        if (sa := _s(data, a)) is not None and (sb := _s(data, b)) is not None:
            r = ratio(sa, sb) * mult
            out[key] = r
            out[f"{key}_z252"] = zscore(r, window=252, min_periods=126)
            out[f"{key}_trend"] = trend_state(r)
    return {k: to_frame(v) for k, v in out.items() if v is not None and not v.dropna().empty}
