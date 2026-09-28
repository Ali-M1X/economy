"""Fed Stance Score: −100 (very dovish / expansionary) … +100 (very hawkish / contractionary).

Each component is mapped to [−1, +1] (positive = hawkish), then combined with fixed weights.
Missing components are dropped and the remaining weights re-normalised; the result lists which
components were used, so the dashboard can show "3 of 4 components".

| component | input | mapping to [−1, +1] | weight |
|---|---|---|---|
| a1 policy direction | last change of the target upper bound | sign(change) × exp(−months since / 6) | 0.15 |
| a2 market-implied path | fed funds futures: implied rate ~12 months ahead − current EFFR | clip(Δ / 1.00 pp) | 0.20 |
| b  balance sheet | WALCL 13-week change, annualized | clip(−Δ% / 10%)  (shrinking = QT = hawkish) | 0.20 |
| c  communications | mean Claude hawkishness of FOMC statements/minutes/speeches, last 60 days | already in [−1, +1] | 0.25 |
| d  dot plot | latest SEP median for the next year-end − current target midpoint | clip(Δ / 0.75 pp) | 0.20 |

Score = 100 × Σ wᵢ·xᵢ / Σ wᵢ over available components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

WEIGHTS = {"a1_policy_direction": 0.15, "a2_market_path": 0.20, "b_balance_sheet": 0.20,
           "c_communications": 0.25, "d_dot_plot": 0.20}
FORMULA_FA = (
    "امتیاز موضع فدرال = ۱۰۰ × میانگین وزنی پنج جزء (مثبت = انقباضی): "
    "جهت آخرین تغییر نرخ (۱۵٪)، مسیر نرخ از آتی‌ها (۲۰٪)، روند ترازنامه QE/QT (۲۰٪)، "
    "لحن بیانیه‌ها و سخنرانی‌ها با طبقه‌بندی Claude (۲۵٪)، و فاصله میانه Dot Plot تا نرخ فعلی (۲۰٪)."
)


def clip(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


@dataclass
class StanceResult:
    score: float | None
    components: dict[str, float | None]
    inputs: dict[str, object] = field(default_factory=dict)

    @property
    def used(self) -> list[str]:
        return [k for k, v in self.components.items() if v is not None]


def policy_direction(target_upper: pd.Series, today: date) -> tuple[float | None, dict]:
    ch = target_upper.diff()
    ch = ch[ch.abs() > 1e-9]
    if ch.empty:
        return None, {}
    last_date, last = ch.index[-1], float(ch.iloc[-1])
    months = (pd.Timestamp(today) - last_date).days / 30.44
    return clip(math.copysign(1.0, last) * math.exp(-months / 6.0)), {
        "last_change_date": last_date.date().isoformat(), "last_change_bp": round(last * 100)}


def market_path(curve: pd.DataFrame, effr_now: float, today: date, months_ahead: int = 12) -> tuple[float | None, dict]:
    """curve: contract_month (date), implied_rate. Uses the contract closest to `months_ahead` ahead."""
    if curve is None or curve.empty:
        return None, {}
    target = pd.Timestamp(today) + pd.DateOffset(months=months_ahead)
    c = curve.assign(dist=(pd.to_datetime(curve["contract_month"]) - target).abs()).sort_values("dist").iloc[0]
    delta = float(c["implied_rate"]) - effr_now
    return clip(delta / 1.0), {"contract": c.get("contract"), "implied_rate": float(c["implied_rate"]),
                               "delta_pp": round(delta, 3)}


def balance_sheet(walcl: pd.Series) -> tuple[float | None, dict]:
    if len(walcl.dropna()) < 14:
        return None, {}
    ann = ((walcl.iloc[-1] / walcl.iloc[-14]) ** 4 - 1.0) * 100.0
    return clip(-ann / 10.0), {"walcl_13w_ann_pct": round(float(ann), 2)}


def communications(docs: pd.DataFrame | None, today: date, days: int = 60) -> tuple[float | None, dict]:
    """docs: published_utc, hawkish_score (−1..+1). Unclassified documents are ignored."""
    if docs is None or docs.empty:
        return None, {"n_docs": 0}
    d = docs.dropna(subset=["hawkish_score"])
    d = d[pd.to_datetime(d["published_utc"], utc=True) >= pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=days)]
    if d.empty:
        return None, {"n_docs": 0}
    return clip(float(d["hawkish_score"].mean())), {"n_docs": int(len(d))}


def dot_plot(sep_latest: pd.DataFrame | None, target_mid: float, today: date) -> tuple[float | None, dict]:
    """sep_latest: SEP median projections as currently published (date = target year start)."""
    if sep_latest is None or sep_latest.empty:
        return None, {}
    nxt = sep_latest[sep_latest["date"].dt.year == today.year + 1]
    row = nxt.iloc[0] if not nxt.empty else sep_latest.iloc[-1]
    delta = float(row["value"]) - target_mid
    return clip(delta / 0.75), {"sep_year": int(row["date"].year), "sep_median": float(row["value"]),
                                "target_mid": target_mid, "delta_pp": round(delta, 3)}


def stance_score(*, today: date, target_upper: pd.Series, effr_now: float, target_mid: float,
                 curve: pd.DataFrame | None, walcl: pd.Series, docs: pd.DataFrame | None,
                 sep_latest: pd.DataFrame | None) -> StanceResult:
    parts, inputs = {}, {}
    for name, (val, info) in {
        "a1_policy_direction": policy_direction(target_upper, today),
        "a2_market_path": market_path(curve, effr_now, today),
        "b_balance_sheet": balance_sheet(walcl),
        "c_communications": communications(docs, today),
        "d_dot_plot": dot_plot(sep_latest, target_mid, today),
    }.items():
        parts[name] = None if val is None or np.isnan(val) else round(val, 4)
        inputs[name] = info
    used = {k: v for k, v in parts.items() if v is not None}
    if not used:
        return StanceResult(None, parts, inputs)
    w = sum(WEIGHTS[k] for k in used)
    score = 100.0 * sum(WEIGHTS[k] * v for k, v in used.items()) / w
    return StanceResult(round(score, 1), parts, inputs)
