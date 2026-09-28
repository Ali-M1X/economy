"""Business/liquidity-cycle regime: Expansion · Peak · Recession · Recovery.

Rule-based model (transparent, primary)
---------------------------------------
Built on a point-in-time monthly panel (`features.pit`), so every historical label uses only data that
was public at that month-end. Inputs are z-scored on an expanding window (no look-ahead).

Axis 1 — growth momentum  g = 3-month change of the growth composite G, where G = mean z of:
    Philly Fed activity, Empire State activity, 10Y−3M slope, −Baa−10Y spread, −NFCI,
    −YoY change of 4-week initial claims, 3-month average payroll gain.
Axis 2 — inflation/liquidity pressure  p = mean of:
    z(core CPI 3m annualized − core CPI YoY)   (inflation accelerating → +)
    −z(liquidity momentum)                     (liquidity draining → +; M2 YoY, WALCL 13w change, −NFCI)
Quadrants (investment-clock style):
    Recovery  = g > 0, p ≤ 0      Expansion = g > 0, p > 0
    Peak      = g ≤ 0, p > 0      Recession = g ≤ 0, p ≤ 0
Probabilities: P(g>0) = logistic(g / s_g), P(p>0) = logistic(p / s_p) with s = expanding std, and each
regime's probability is the product of its two sides (they sum to 1). The label is the most likely regime.
Switch triggers are the distances of g and p from zero.

HMM (optional, comparison only)
-------------------------------
A 4-state Gaussian HMM on (g, p) fitted on the full sample (in-sample parameters — labelled as such),
with *filtered* (forward-only) state probabilities. States are mapped to regimes by their mean (g, p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.transforms import zscore

REGIMES = ("Expansion", "Peak", "Recession", "Recovery")
REGIMES_FA = {"Expansion": "رونق", "Peak": "اوج", "Recession": "رکود", "Recovery": "بهبود"}

GROWTH_INPUTS = {  # key → sign (+1 means higher = stronger growth)
    "philly_fed_activity": +1, "empire_state_activity": +1, "spread_10y_3m": +1, "baa_10y_spread": -1,
    "nfci": -1, "claims_yoy": -1, "payrolls_3m_avg": +1,
}
LIQUIDITY_INPUTS = {"m2_yoy": +1, "walcl_13w": +1, "nfci": -1}
MIN_Z_MONTHS = 60


def build_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: month-end PIT values of raw series (columns = registry keys). Adds transformed inputs."""
    p = panel.copy()
    if "initial_claims" in p:
        p["claims_yoy"] = (p["initial_claims"] / p["initial_claims"].shift(12) - 1) * 100
    if "nonfarm_payrolls" in p:
        p["payrolls_3m_avg"] = p["nonfarm_payrolls"].diff().rolling(3).mean()
    if "m2" in p:
        p["m2_yoy"] = (p["m2"] / p["m2"].shift(12) - 1) * 100
    if "fed_balance_sheet" in p:
        p["walcl_13w"] = (p["fed_balance_sheet"] / p["fed_balance_sheet"].shift(3) - 1) * 100
    if "core_cpi" in p:
        c = p["core_cpi"]
        p["core_cpi_accel"] = (((c / c.shift(3)) ** 4 - 1) - (c / c.shift(12) - 1)) * 100
    return p


def _composite(p: pd.DataFrame, inputs: dict[str, int], min_inputs: int) -> pd.Series:
    zs = pd.concat({k: sign * zscore(p[k], min_periods=MIN_Z_MONTHS) for k, sign in inputs.items() if k in p}, axis=1)
    comp = zs.mean(axis=1)
    comp[zs.notna().sum(axis=1) < min_inputs] = np.nan
    return comp


def axes(panel: pd.DataFrame) -> pd.DataFrame:
    p = build_inputs(panel)
    growth = _composite(p, GROWTH_INPUTS, min_inputs=3).rolling(3).mean()
    g = growth - growth.shift(3)
    liq = _composite(p, LIQUIDITY_INPUTS, min_inputs=2).rolling(3).mean()
    liq_mom = liq - liq.shift(3)
    infl = zscore(p["core_cpi_accel"], min_periods=MIN_Z_MONTHS) if "core_cpi_accel" in p else pd.Series(np.nan, index=p.index)
    pressure = pd.concat([infl, -zscore(liq_mom, min_periods=MIN_Z_MONTHS)], axis=1).mean(axis=1)
    return pd.DataFrame({"growth_level": growth, "g": g, "p": pressure})


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def rule_probabilities(ax: pd.DataFrame) -> pd.DataFrame:
    sg = ax["g"].expanding(min_periods=24).std()
    sp = ax["p"].expanding(min_periods=24).std()
    pg = _logistic(ax["g"] / (0.5 * sg))
    pp = _logistic(ax["p"] / (0.5 * sp))
    probs = pd.DataFrame({"Expansion": pg * pp, "Peak": (1 - pg) * pp, "Recession": (1 - pg) * (1 - pp),
                          "Recovery": pg * (1 - pp)}, index=ax.index).dropna()
    probs["regime"] = probs[list(REGIMES)].idxmax(axis=1)
    return probs


def switch_triggers(ax_row: pd.Series, regime: str) -> list[str]:
    g, p = float(ax_row["g"]), float(ax_row["p"])
    out = []
    target_g = {"Expansion": "Peak", "Recovery": "Recession", "Peak": "Expansion", "Recession": "Recovery"}[regime]
    target_p = {"Expansion": "Recovery", "Recovery": "Expansion", "Peak": "Recession", "Recession": "Peak"}[regime]
    out.append(f"→ {target_g} if growth momentum crosses zero (now {g:+.2f}σ)")
    out.append(f"→ {target_p} if inflation/liquidity pressure crosses zero (now {p:+.2f}σ)")
    return out


@dataclass
class HmmResult:
    probs: pd.DataFrame  # filtered probabilities per regime + regime label
    note: str


def hmm_regimes(ax: pd.DataFrame, seed: int = 7) -> HmmResult | None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None
    X = ax[["g", "p"]].dropna()
    if len(X) < 120:
        return None
    model = GaussianHMM(n_components=4, covariance_type="full", n_iter=500, random_state=seed)
    model.fit(X.to_numpy())
    # Forward (filtered) probabilities — no smoothing with future observations.
    from scipy.stats import multivariate_normal

    emis = np.column_stack([multivariate_normal(model.means_[k], model.covars_[k], allow_singular=True).pdf(X.to_numpy())
                            for k in range(4)])
    alpha = np.zeros_like(emis)
    a = model.startprob_ * emis[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ model.transmat_) * emis[t]
        alpha[t] = a / a.sum() if a.sum() > 0 else alpha[t - 1]
    # Map states to quadrants by the sign pattern of their means (each regime used once, by best fit).
    targets = {"Expansion": (1, 1), "Peak": (-1, 1), "Recession": (-1, -1), "Recovery": (1, -1)}
    from itertools import permutations

    best, best_score = None, -np.inf
    for perm in permutations(REGIMES):
        score = sum(np.dot(np.sign(model.means_[k]) + model.means_[k], targets[r]) for k, r in enumerate(perm))
        if score > best_score:
            best, best_score = perm, score
    probs = pd.DataFrame(alpha, index=X.index, columns=list(best))[list(REGIMES)]
    probs["regime"] = probs[list(REGIMES)].idxmax(axis=1)
    return HmmResult(probs, "HMM parameters are fitted in-sample on the full history; state probabilities are "
                            "filtered (forward-only). Use for comparison, not for backtests.")


def regime_performance(labels: pd.Series, prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Forward 1-month return of each asset after month-ends classified in each regime (label known at t)."""
    rows = []
    for asset, px in prices.items():
        m = px.resample("ME").last()
        fwd = (m.shift(-1) / m - 1) * 100
        j = pd.concat([labels.rename("regime"), fwd.rename("ret")], axis=1, join="inner").dropna()
        for reg in REGIMES:
            r = j.loc[j["regime"] == reg, "ret"]
            rows.append({"asset": asset, "regime": reg, "months": int(len(r)),
                         "mean_fwd_1m_pct": round(float(r.mean()), 2) if len(r) else None,
                         "median_fwd_1m_pct": round(float(r.median()), 2) if len(r) else None,
                         "hit_rate": round(float((r > 0).mean()), 2) if len(r) else None,
                         "ann_vol_pct": round(float(r.std() * np.sqrt(12)), 1) if len(r) > 1 else None})
    return pd.DataFrame(rows)
