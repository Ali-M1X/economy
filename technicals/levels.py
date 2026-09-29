"""Price levels: support/resistance, liquidity pools, volume profile, order-book walls, liquidation estimates.

Backtestable (used by the signal rules): prior day/week high/low, swing highs/lows, equal highs/lows.
Live context only (shown, not used for entries/targets because no free history exists): volume profile of
the recent window, order-book walls, estimated liquidation clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from technicals.indicators import resample

LEVERAGE_TIERS = {10: 0.4, 25: 0.3, 50: 0.2, 100: 0.1}  # assumed share of new OI per leverage tier
MAINT_MARGIN = 0.005


def prior_period_levels(bars: pd.DataFrame, at: pd.Series) -> pd.DataFrame:
    """For each timestamp in `at`, the high/low of the previous *completed* UTC day and week."""
    out = pd.DataFrame({"t": pd.to_datetime(at, utc=True)})
    for tf, name in (("1d", "pd"), ("1w", "pw")):
        agg = resample(bars, tf)
        agg["end"] = agg["ts"] + (pd.Timedelta(days=1) if tf == "1d" else pd.Timedelta(days=7))
        agg = agg.sort_values("end")
        m = pd.merge_asof(out.sort_values("t"), agg[["end", "high", "low"]].rename(columns={"end": "t"}),
                          on="t", direction="backward")
        out[f"{name}h"], out[f"{name}l"] = m["high"].to_numpy(), m["low"].to_numpy()
    return out.drop(columns="t")


def equal_levels(prices: np.ndarray, tolerance: float) -> list[float]:
    """Clusters of ≥ 2 swing prices within `tolerance` of each other (resting stop liquidity)."""
    if len(prices) < 2:
        return []
    p = np.sort(prices)
    out, cluster = [], [p[0]]
    for x in p[1:]:
        if x - cluster[-1] <= tolerance:
            cluster.append(x)
        else:
            if len(cluster) >= 2:
                out.append(float(np.mean(cluster)))
            cluster = [x]
    if len(cluster) >= 2:
        out.append(float(np.mean(cluster)))
    return out


def volume_profile(bars: pd.DataFrame, bins: int = 60, value_area: float = 0.70) -> dict:
    """Volume-at-price from OHLCV bars (each bar's volume spread evenly over its high-low range).
    Returns POC, VAH/VAL (the `value_area` share of volume around the POC), HVN/LVN (local extremes
    of the smoothed profile) and the profile itself."""
    lo, hi = float(bars["low"].min()), float(bars["high"].max())
    if not np.isfinite(lo) or hi <= lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol = np.zeros(bins)
    for l, h, v in bars[["low", "high", "volume"]].itertuples(index=False):
        a, b = np.searchsorted(edges, l, "right") - 1, np.searchsorted(edges, h, "left")
        a, b = max(a, 0), min(max(b, a + 1), bins)
        vol[a:b] += v / (b - a)
    poc = int(vol.argmax())
    lo_i = hi_i = poc
    total, acc = vol.sum(), vol[poc]
    while acc < value_area * total and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        dn = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    sm = np.convolve(vol, np.ones(3) / 3, mode="same")
    hvn = [float(centers[i]) for i in range(1, bins - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > np.median(sm)]
    lvn = [float(centers[i]) for i in range(1, bins - 1) if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] < np.median(sm)]
    return {"poc": float(centers[poc]), "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i]), "hvn": hvn, "lvn": lvn,
            "profile": pd.DataFrame({"price": centers, "volume": vol})}


def orderbook_walls(depth: pd.DataFrame, mid: float, within: float = 0.05, factor: float = 3.0) -> pd.DataFrame:
    """Aggregated order-book buckets (side, bucket, usd) with notional ≥ `factor` × the median bucket on that side,
    within ±`within` of the mid price."""
    d = depth[(depth["bucket"] >= mid * (1 - within)) & (depth["bucket"] <= mid * (1 + within))]
    out = []
    for side, g in d.groupby("side"):
        med = g["usd"].median()
        out.append(g[g["usd"] >= factor * med])
    return pd.concat(out).sort_values("usd", ascending=False) if out else d.iloc[0:0]


def liquidation_clusters(oi: pd.DataFrame, price: pd.Series, bucket: float) -> pd.DataFrame:
    """ESTIMATE of where leveraged positions opened in the OI window would be liquidated.

    For every hour where open interest rose, the added notional is split 50/50 long/short at that hour's
    price and across leverage tiers (LEVERAGE_TIERS); liquidation ≈ entry × (1 ∓ 1/L ± maintenance margin).
    Real positions, leverage and closes are unknown — this is a heuristic map, labelled as such."""
    j = pd.merge_asof(oi.sort_values("ts"), price.rename("px").rename_axis("ts").reset_index().sort_values("ts"), on="ts")
    j["add_usd"] = j["open_interest_usd"].diff().clip(lower=0)
    rows = []
    for r in j.dropna(subset=["add_usd", "px"]).itertuples():
        for lev, w in LEVERAGE_TIERS.items():
            n = r.add_usd * w / 2
            rows.append(("long", r.px * (1 - 1 / lev + MAINT_MARGIN), n))
            rows.append(("short", r.px * (1 + 1 / lev - MAINT_MARGIN), n))
    if not rows:
        return pd.DataFrame(columns=["side", "bucket", "usd"])
    d = pd.DataFrame(rows, columns=["side", "price", "usd"])
    d["bucket"] = (d["price"] // bucket) * bucket
    return d.groupby(["side", "bucket"], as_index=False)["usd"].sum().sort_values("usd", ascending=False)
