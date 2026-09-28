"""Impact coefficients: how much each indicator moves BTC and gold, per horizon.

Horizons
- short  (1h, 4h, 24h): event study on standardized first-print release surprises (models.events);
- medium (1-4 weeks):   local projections — forward h-week log return on the indicator's 4-week change
                         (weekly, point-in-time panel), Newey-West errors (lag h);
- long   (3-12 months): forward h-month return on the indicator's 3-month change (monthly PIT panel),
                         Newey-West errors (lag h), plus the same regression within each regime.
The indicator change is scaled by its sample std, so β = % return per 1σ move of the indicator.

Impact Coefficient (0-10, signed)
    impact = 10 × min(1, |β| / (0.5 · sd_ret)) × min(1, |t| / 2) × min(1, n_eff / 30)
    sign   = sign(β): + means "indicator up / hotter surprise → asset up" (bullish)
i.e. a 1σ indicator move that moves the asset by half its typical horizon move, with |t| ≥ 2 and at least
30 independent observations, scores 10; weak, insignificant or small-sample effects shrink toward 0.
(The sample-size factor matters: with overlapping horizons, Newey-West t-stats from a handful of independent
observations are unreliable — on pure noise, 20 overlapping 6-month windows produced |t| > 4.)
Confidence: high if |t| ≥ 2.58 and n_eff ≥ 50; medium if |t| ≥ 1.96 and n_eff ≥ 30; else low
(n_eff = n / overlap for overlapping horizons).
These are descriptive, in-sample statistics about historical co-movement — not forecasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HEADLINE = {"short": "1h", "medium": "4w", "long": "6m"}


@dataclass
class Estimate:
    indicator: str
    asset: str
    horizon: str  # 1h|4h|24h|1w|2w|4w|3m|6m|12m
    bucket: str  # short|medium|long
    sample: str  # full|recent|regime:<name>
    beta: float  # % asset return per 1σ indicator move (or per 1σ surprise)
    t_stat: float
    n: int
    n_eff: float
    sd_ret: float
    hit_rate: float | None
    coefficient: float = 0.0  # signed 0..10
    confidence: str = "low"

    def as_row(self) -> dict:
        return asdict(self)


def ols_nw(y: np.ndarray, x: np.ndarray, lag: int = 0) -> tuple[float, float, int]:
    """Slope, Newey-West t-stat (Bartlett kernel, `lag` lags) and n for y = a + b·x."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 8 or np.std(x) == 0:
        return np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g = X * u[:, None]
    S = g.T @ g
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = g[l:].T @ g[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(V[1, 1]))
    return float(b[1]), float(b[1] / se) if se > 0 else np.nan, n


def normalize(e: Estimate) -> Estimate:
    if not np.isfinite(e.beta) or not np.isfinite(e.t_stat) or not e.sd_ret:
        e.coefficient, e.confidence = 0.0, "low"
        return e
    size = min(1.0, abs(e.beta) / (0.5 * e.sd_ret))
    sig = min(1.0, abs(e.t_stat) / 2.0)
    enough = min(1.0, e.n_eff / 30.0)
    e.coefficient = round(float(np.sign(e.beta) * 10 * size * sig * enough), 2)
    at = abs(e.t_stat)
    e.confidence = "high" if at >= 2.58 and e.n_eff >= 50 else "medium" if at >= 1.96 and e.n_eff >= 30 else "low"
    return e


# ───────────────────────────── short: event study ─────────────────────────────

Z_CAP = 4.0  # surprises beyond ±4 robust σ are capped in regressions (limits single-event leverage)
MIN_HIT_EVENTS = 20


def event_study(ev: pd.DataFrame, indicator: str, asset: str, sample: str = "full",
                horizons=("1h", "4h", "24h")) -> list[Estimate]:
    """Hit rate = share of events with |z| ≥ 0.5 where the asset moved in the direction implied by β;
    reported only when at least 20 such events exist."""
    out = []
    for h in horizons:
        col = f"ret_{h}"
        if col not in ev:
            continue
        d = ev[["z", col]].dropna()
        zc = d["z"].clip(-Z_CAP, Z_CAP)
        b, t, n = ols_nw(d[col].to_numpy(), zc.to_numpy(), lag=0)
        big = d[d["z"].abs() >= 0.5]
        hit = (float((np.sign(big[col]) == np.sign(b * big["z"])).mean())
               if len(big) >= MIN_HIT_EVENTS and np.isfinite(b) else None)
        out.append(normalize(Estimate(indicator, asset, h, "short", sample, b, t, n, float(n),
                                      float(d[col].std()) if n > 1 else np.nan, hit)))
    return out


# ───────────────────────────── medium / long: panels ─────────────────────────────

def change(s: pd.Series, periods: int, how: str) -> pd.Series:
    if how == "diff":
        return s - s.shift(periods)
    if how == "logdiff":
        return np.log(s / s.shift(periods)) * 100
    raise ValueError(how)


def forward_return(px: pd.Series, periods: int) -> pd.Series:
    return np.log(px.shift(-periods) / px) * 100


def projection(ind: pd.Series, px: pd.Series, how: str, change_periods: int, horizon: int,
               indicator: str, asset: str, horizon_label: str, bucket: str, sample: str = "full",
               mask: pd.Series | None = None) -> Estimate:
    """Forward `horizon`-period return on the `change_periods` change of the indicator (both on the same grid)."""
    x = change(ind, change_periods, how)
    y = forward_return(px, horizon)
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if mask is not None:
        j = j[mask.reindex(j.index).fillna(False).astype(bool)]
    sd_x = j["x"].std()
    xs = j["x"] / sd_x if sd_x else j["x"] * np.nan
    b, t, n = ols_nw(j["y"].to_numpy(), xs.to_numpy(), lag=horizon)
    return normalize(Estimate(indicator, asset, horizon_label, bucket, sample, b, t, n, n / max(horizon, 1),
                              float(j["y"].std()) if n > 1 else np.nan, None))


def weekly(px_or_panel: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return px_or_panel.resample("W-FRI").last()


def monthly(px_or_panel):
    return px_or_panel.resample("ME").last()


def estimates_frame(ests: list[Estimate]) -> pd.DataFrame:
    return pd.DataFrame([e.as_row() for e in ests])
