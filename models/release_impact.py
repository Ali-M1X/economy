"""The real impact of a release on Bitcoin and gold: how far the price moves, how fast, whether the direction can be
predicted, whether the move fades, and how much of it is traded before the release (front-running).

All returns are log returns in % from 15-minute bars: the price at time t is the close of the last bar that ended at or
before t (never filled across a gap longer than MAX_GAP).

Per release indicator and asset:
- shock:     median and 75th-percentile |return| at 15 min, 1 h, 4 h and 24 h after the release, whatever the direction;
             and the shock ratio = that median ÷ the median |return| over the same window at the same weekday and time
             of day on days without a tracked release (the BASE_WEEKS weeks before each release).
- direction: over big surprises (|z| ≥ BIG_Z) of releases with a textbook sign, how often the 1-hour move had the sign
             theory expects; a two-sided binomial p-value; and the average 1-hour move per 1σ of surprise (OLS slope).
- fade:      over releases whose first hour moved more than a normal hour, how often the 24-hour return ended on the
             other side of the release price, and the median share of the first-hour move given back by 24 h.
- pre:       median |drift| and median signed drift 6 / 24 / 48 h before the release, against normal days; how often the
             24-hour drift and the 24-hour post-release move point the same way; and, after a large drift (top
             tercile), the median post-release move measured in the drift's direction and how often it reversed.
Every statistic carries its n and is hidden (None) when n < MIN_N. The tier is a label for the measured numbers:
"volatile_directional", "volatile_undirected" or "low" (shock ratio below BIG_SHOCK).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HORIZONS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "24h": 24.0}
PRE_WINDOWS = (6, 24, 48)
BASE_WEEKS = 12
MAX_GAP = pd.Timedelta(minutes=45)
MIN_N = 12
BIG_Z = 1.0
BIG_SHOCK = 1.5         # 1-hour shock ratio from which a release counts as a volatility event
DIRECTION_P = 0.10      # two-sided binomial p below which the 1-hour direction counts as predictable
TIERS = ("volatile_directional", "volatile_undirected", "low")


@dataclass
class Prices:
    """Log close by bar end time, for fast as-of lookups."""
    ends: np.ndarray    # int64 ns, sorted
    logc: np.ndarray

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, bar_minutes: int = 15) -> "Prices":
        b = bars.dropna(subset=["close"]).sort_values("ts")
        ts = pd.to_datetime(b["ts"], utc=True).dt.as_unit("ns") + pd.Timedelta(minutes=bar_minutes)
        return cls(ts.astype("int64").to_numpy(), np.log(b["close"].astype(float).to_numpy()))

    def at(self, t: np.ndarray) -> np.ndarray:
        """log price as of each time (ns int64); NaN when the last bar ended more than MAX_GAP before."""
        t = np.asarray(t, dtype="int64")
        i = np.searchsorted(self.ends, t, side="right") - 1
        ok = (i >= 0) & (t - self.ends[np.clip(i, 0, None)] <= MAX_GAP.value)
        out = np.full(t.shape, np.nan)
        out[ok] = self.logc[i[ok]]
        return out

    def ret(self, start: np.ndarray, hours: float) -> np.ndarray:
        """% log return from each start time over `hours`."""
        s = np.asarray(start, dtype="int64")
        return 100 * (self.at(s + int(hours * 3.6e12)) - self.at(s))


def _ns(times) -> np.ndarray:
    return pd.to_datetime(pd.Series(times), utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def baseline_times(release_ns: np.ndarray, busy_days: set, weeks: int = BASE_WEEKS) -> np.ndarray:
    """Same weekday and time of day in the `weeks` weeks before each release, skipping days with any tracked release."""
    week = 7 * 86400 * 10**9
    cand = np.unique((release_ns[:, None] - week * np.arange(1, weeks + 1)[None, :]).ravel())
    days = pd.to_datetime(cand, utc=True, unit="ns").date
    return cand[np.array([d not in busy_days for d in days], dtype=bool)]


def _med(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def _r(v: float | None, d: int = 3) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def _hide(block: dict, n: int) -> dict:
    """Keep only n when the sample is too small to read."""
    return block if n >= MIN_N else {"n": int(n)}


def measure(px: Prices, events: pd.DataFrame, busy_days: set, theory: int = 0) -> dict:
    """events: release_utc and (optionally) z — one row per release of one indicator. theory: textbook sign of the
    asset's move after a hotter print (+1, −1, 0 = none)."""
    t = _ns(events["release_utc"])
    z = events["z"].to_numpy(dtype=float) if "z" in events else np.full(len(t), np.nan)
    base_t = baseline_times(t, busy_days)
    out: dict = {"n_events": int(len(t))}

    shock, post = {}, {}
    for name, h in HORIZONS.items():
        r = px.ret(t, h)
        post[name] = r
        a = np.abs(r[np.isfinite(r)])
        b = np.abs(px.ret(base_t, h))
        b = b[np.isfinite(b)]
        med, base = (float(np.median(a)) if len(a) else None), (float(np.median(b)) if len(b) else None)
        shock[name] = _hide({"n": int(len(a)), "median_abs": _r(med), "p75_abs": _r(np.percentile(a, 75)) if len(a) else None,
                             "normal_median_abs": _r(base), "n_normal": int(len(b)),
                             "ratio": _r(med / base, 2) if med is not None and base else None}, len(a))
    out["shock"] = shock

    # direction (1 hour)
    r1 = post["1h"]
    ok = np.isfinite(r1) & np.isfinite(z)
    big = ok & (np.abs(z) >= BIG_Z) & (theory != 0)
    nb = int(big.sum())
    direction = {"n": nb}
    if nb:
        hits = int((np.sign(r1[big]) == np.sign(theory * z[big])).sum())
        p = binomtest(hits, nb, 0.5).pvalue
        slope = float(np.polyfit(z[ok], r1[ok], 1)[0]) if ok.sum() >= 3 and np.ptp(z[ok]) > 0 else None
        direction.update({"hit_rate": _r(hits / nb, 3), "p_value": _r(p, 4), "per_sigma_1h": _r(slope),
                          "n_all": int(ok.sum()), "predictable": bool(p < DIRECTION_P),
                          "with_theory": bool(hits / nb > 0.5)})
    out["direction"] = _hide(direction, nb) if theory else {"n": 0, "note": "no textbook direction"}

    # fade: first-hour moves larger than a normal hour
    r24 = post["24h"]
    norm1 = shock["1h"].get("normal_median_abs")
    m = np.isfinite(r1) & np.isfinite(r24) & (np.abs(r1) > (norm1 or 0)) & (r1 != 0)
    nf = int(m.sum())
    fade = {"n": nf}
    if nf:
        fade.update({"reversal_rate": _r(float(np.mean(np.sign(r24[m]) != np.sign(r1[m]))), 3),
                     "median_retrace_pct": _r(float(np.median((1 - r24[m] / r1[m]) * 100)), 1)})
    out["fade"] = _hide(fade, nf)

    # front-running
    pre, drifts = {}, {}
    for W in PRE_WINDOWS:
        d = px.ret(t - int(W * 3.6e12), W)
        drifts[W] = d
        b = np.abs(px.ret(base_t - int(W * 3.6e12), W))
        b = b[np.isfinite(b)]
        a = np.abs(d[np.isfinite(d)])
        med, base = (float(np.median(a)) if len(a) else None), (float(np.median(b)) if len(b) else None)
        pre[str(W)] = _hide({"n": int(len(a)), "median_abs": _r(med), "median_signed": _r(_med(d)),
                             "normal_median_abs": _r(base), "ratio": _r(med / base, 2) if med is not None and base else None},
                            len(a))
    out["pre"] = pre
    d24 = drifts[24]
    m = np.isfinite(d24) & np.isfinite(r24) & (d24 != 0)
    npp = int(m.sum())
    pp = {"n": npp}
    if npp:
        pp["same_direction_rate"] = _r(float(np.mean(np.sign(d24[m]) == np.sign(r24[m]))), 3)
        # top tercile by rank: the largest third of the |24 h drifts|
        idx = np.flatnonzero(m)
        k = int(np.ceil(len(idx) / 3))
        top_idx = idx[np.argsort(-np.abs(d24[idx]), kind="stable")[:k]]
        top = np.zeros_like(m)
        top[top_idx] = True
        cut = float(np.abs(d24[top_idx]).min())
        cont = r24[top] * np.sign(d24[top])  # + = the move continued in the drift's direction
        pp["large_drift"] = _hide({"n": int(top.sum()), "threshold_abs": _r(cut),
                                   "post_median_in_drift_direction": _r(_med(cont)),
                                   "reversal_rate": _r(float(np.mean(cont < 0)), 3) if len(cont) else None}, int(top.sum()))
    out["pre_post"] = _hide(pp, npp)
    out["tier"] = tier(out)
    return out


def tier(m: dict) -> str | None:
    ratio = (m.get("shock", {}).get("1h") or {}).get("ratio")
    if ratio is None:
        return None
    if ratio < BIG_SHOCK:
        return "low"
    return "volatile_directional" if (m.get("direction") or {}).get("predictable") else "volatile_undirected"


# ───────────────────────────── live front-running read ─────────────────────────────

VERDICTS = ("not_priced", "partly_priced", "heavily_priced")


def frontrun_read(px: Prices, m: dict, release: pd.Timestamp, now: pd.Timestamp) -> dict | None:
    """Move since the pre-release window opened (24 h, else 48 h before), against this release's usual pre-release
    drift over the same elapsed time (the historical median |drift| scaled by √(elapsed / window)).
    ratio < 1 → not priced; 1–2 → partly priced; ≥ 2 → heavily priced."""
    pre = m.get("pre") or {}
    for W in (24, 48):
        start = release - pd.Timedelta(hours=W)
        hist = (pre.get(str(W)) or {}).get("median_abs")
        if now <= start or not hist or now >= release:
            continue
        elapsed = (now - start) / pd.Timedelta(hours=1)
        if elapsed < 1:
            continue
        move = float(px.ret(np.array([start.as_unit("ns").value]), elapsed)[0])
        if not np.isfinite(move):
            continue
        normal = hist * sqrt(min(elapsed, W) / W)
        ratio = abs(move) / normal if normal else None
        if ratio is None:
            return None
        verdict = VERDICTS[0] if ratio < 1 else VERDICTS[1] if ratio < 2 else VERDICTS[2]
        return {"window_h": W, "elapsed_h": round(elapsed, 1), "move_pct": round(move, 2), "normal_pct": round(normal, 2),
                "ratio": round(ratio, 1), "verdict": verdict, "large": (m.get("pre_post") or {}).get("large_drift")}
    return None
