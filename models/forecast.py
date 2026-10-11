"""Probabilities for an upcoming release: hotter than the forecast, in line, or cooler.

The forecast is the consensus when one exists (ForexFactory), else the system's own expectation (mean of the last 3
prints or the previous print). "In line" means |actual − forecast| < IN_LINE_Z × σ, the same "roughly as expected"
band the reports use for past surprises.

Three ways to the odds, in order of preference:
1. market  — prediction-market brackets (Kalshi "above X" contracts). Results are published on a fixed decimal grid, so
             the survival curve is a step function: P(actual > t) = price of the highest strike ≤ t.
2. nowcast — a model nowcast against the consensus: actual ~ Normal(nowcast, σ), where σ is the historical std of
             actual − consensus (or, until enough consensus history exists, the historical surprise scale).
             P(actual > consensus) = Φ((nowcast − consensus) / σ).
3. even    — consensus (or system expectation) only: symmetric odds around the forecast; the system has no edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

IN_LINE_Z = 0.5
MIN_CONSENSUS_HISTORY = 12  # actual − consensus pairs needed before their std replaces the surprise scale


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


@dataclass
class Odds:
    hot: float      # actual above the forecast by more than the in-line band
    inline: float
    cool: float
    method: str     # market | nowcast | even

    @property
    def edge(self) -> bool:
        """Whether the odds lean (only market or nowcast odds can; even odds never do)."""
        return self.method != "even" and abs(self.hot - self.cool) >= 0.1


def normal_odds(center: float, forecast: float, sigma: float, method: str) -> Odds:
    band = IN_LINE_Z * sigma
    hot = 1 - phi((forecast + band - center) / sigma)
    cool = phi((forecast - band - center) / sigma)
    return Odds(hot, max(0.0, 1 - hot - cool), cool, method)


def even_odds(sigma: float = 1.0) -> Odds:
    """Forecast only: the actual is as likely to land above as below it."""
    return normal_odds(0.0, 0.0, sigma, "even")


def survival(points: list[tuple[float, float]], t: float) -> float | None:
    """P(actual > t) from (strike, P(actual > strike)) points, as a step function (highest strike ≤ t).
    Below the lowest strike the probability is taken as at that strike (≈ 1 in practice)."""
    if not points:
        return None
    pts = sorted(points)
    below = [p for s, p in pts if s <= t + 1e-9]
    return below[-1] if below else pts[0][1]


def clean_survival(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Market prices are noisy: force P(actual > strike) to fall as the strike rises (running minimum)."""
    out, lo = [], 1.0
    for s, p in sorted(points):
        lo = min(lo, max(0.0, min(1.0, p)))
        out.append((s, lo))
    return out


def grid_step(points: list[tuple[float, float]]) -> float | None:
    """The spacing of the market's strikes (results are published on this grid, e.g. 0.1 pp for CPI)."""
    xs = sorted({round(s, 9) for s, _ in points})
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 1e-9]
    return min(gaps) if gaps else None


def market_odds(points: list[tuple[float, float]], forecast: float, sigma: float) -> Odds | None:
    """Hot / in line / cool from the step survival curve. The in-line band is the usual ±IN_LINE_Z·σ, but never wider
    than half a strike step: on a published grid, "in line" then means printing exactly at the forecast."""
    pts = clean_survival(points)
    if len(pts) < 2:
        return None
    band = IN_LINE_Z * sigma
    step = grid_step(pts)
    if step:
        band = min(band, step / 2)
    hot = survival(pts, forecast + band)
    above_low = survival(pts, forecast - band - 1e-9)
    if hot is None or above_low is None:
        return None
    cool = 1 - above_low
    return Odds(hot, max(0.0, 1 - hot - cool), cool, "market")


def surprise_sigma(consensus_errors: list[float], fallback: float | None) -> float | None:
    """σ of actual − forecast: the std of past actual − consensus once there are enough, else the surprise scale."""
    e = [x for x in consensus_errors if x is not None and np.isfinite(x)]
    if len(e) >= MIN_CONSENSUS_HISTORY:
        s = float(np.std(e, ddof=1))
        if s > 0:
            return s
    return fallback if fallback and fallback > 0 else None


def event_odds(forecast: float | None, sigma: float | None, nowcast: float | None = None,
               market: list[tuple[float, float]] | None = None) -> Odds | None:
    """Best available odds for one release. None when there is no forecast or no σ to measure surprises with."""
    if forecast is None or not sigma:
        return None
    if market:
        o = market_odds(market, forecast, sigma)
        if o is not None:
            return o
    if nowcast is not None and np.isfinite(nowcast):
        return normal_odds(nowcast, forecast, sigma, "nowcast")
    return even_odds(sigma)
