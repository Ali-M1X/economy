"""FedWatch-style meeting probabilities from 30-Day Fed Funds futures (CME methodology, replicated).

For a meeting on day d of a month with N days, the contract's implied rate is the month average:
    avg = (d · R_start + (N − d) · R_end) / N          (the new rate applies from day d+1)
Meetings are processed in date order:
- R_start of the first meeting is the current EFFR;
- if the month after a meeting has no meeting, R_end is that month's implied rate (cleaner, CME's
  preferred route); otherwise R_end is solved from the meeting month's own contract;
- R_start of the next meeting is the previous meeting's R_end.
Per meeting, the expected change ΔR = R_end − R_start is split between the two nearest 25 bp outcomes:
k = floor(ΔR / 0.25), p(k+1 moves) = ΔR/0.25 − k, p(k moves) = 1 − that. Distributions are chained across
meetings to get the probability of each target range at every meeting (relative to today's range).
Caveat (shown in the dashboard): a futures-implied path also embeds a small risk premium; probabilities
are market-implied, not forecasts.
"""

from __future__ import annotations

import calendar
import math
from collections import defaultdict
from datetime import date

import pandas as pd

STEP = 0.25


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def meeting_path(implied: dict[date, float], meetings: list[date], effr_now: float) -> list[dict]:
    """implied: {contract month (1st of month): implied rate %}. Returns one dict per meeting that can be priced."""
    out = []
    r_start = effr_now
    meeting_months = {_month_start(m) for m in meetings}
    for m in sorted(meetings):
        ms = _month_start(m)
        if ms not in implied:
            break
        n = calendar.monthrange(m.year, m.month)[1]
        d = m.day
        nxt = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        if nxt in implied and nxt not in meeting_months:
            r_end, method = implied[nxt], "next-month contract"
        else:
            if n - d == 0:
                break
            r_end, method = (implied[ms] * n - r_start * d) / (n - d), "meeting-month contract"
        out.append({"meeting": m, "rate_start": r_start, "rate_end": r_end, "change": r_end - r_start, "method": method})
        r_start = r_end
    return out


def move_distribution(change: float) -> dict[int, float]:
    """Split an expected change into the two adjacent 25 bp outcomes (k and k+1 steps)."""
    x = change / STEP
    k = math.floor(x)
    frac = x - k
    dist = {k: 1.0 - frac}
    if frac > 1e-9:
        dist[k + 1] = frac
    return {s: p for s, p in dist.items() if p > 1e-9}


def probabilities(implied: dict[date, float], meetings: list[date], effr_now: float,
                  target_lower_now: float) -> pd.DataFrame:
    """One row per priced meeting: per-meeting P(cut/hold/hike), expected rate after the meeting, and the
    cumulative distribution over target ranges (JSON-able dict "lower-upper" → probability)."""
    rows = []
    cum: dict[int, float] = {0: 1.0}
    for step in meeting_path(implied, meetings, effr_now):
        dist = move_distribution(step["change"])
        new_cum: dict[int, float] = defaultdict(float)
        for a, pa in cum.items():
            for b, pb in dist.items():
                new_cum[a + b] += pa * pb
        cum = dict(new_cum)
        ranges = {f"{target_lower_now + s * STEP:.2f}-{target_lower_now + (s + 1) * STEP:.2f}": round(p, 4)
                  for s, p in sorted(cum.items())}
        rows.append({
            "meeting": step["meeting"],
            "implied_rate_after": round(step["rate_end"], 4),
            "expected_change_bp": round(step["change"] * 100, 1),
            "p_cut": round(sum(p for s, p in dist.items() if s < 0), 4),
            "p_hold": round(dist.get(0, 0.0), 4),
            "p_hike": round(sum(p for s, p in dist.items() if s > 0), 4),
            "cumulative_vs_today_bp": round(sum(s * p for s, p in cum.items()) * 25, 1),
            "target_range_probs": ranges,
            "method": step["method"],
        })
    return pd.DataFrame(rows)
