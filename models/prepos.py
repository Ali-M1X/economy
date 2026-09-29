"""Pre-positioning (front-run) signals: is the market already positioning for a release before it drops, and does
that positioning predict the post-release move?

For a release at time T and a lead window W (hours), the pre-release window is [T−W, T). Features at a step t inside
the window use only data that existed at t (bars closed by t, metrics stamped by t, series values whose
`available_at` ≤ t); every baseline (normal volatility, flow, OI, funding and rate scales) is estimated from data
*before* the window opens. Features (all z-like, clipped to ±6):

  drift    log return over [T−W, t] ÷ the volatility normal for those hours of the week (15-min bars, 90-day baseline)
  steady   one-sample t-statistic of the window's bar returns — a directional move with abnormally low variance
           (steady one-way flow) has a large |t|; under "no drift" it is Student-t distributed
  flow     taker-buy imbalance (aggressive buys − sells) over the window vs its 90-day baseline
  rate     change of the 2-year Treasury yield known at t vs at T−W (market-implied policy path), ÷ its daily σ
  oi       change of log open interest in the direction of the price drift (BTC only), ÷ its 5-min σ
  funding  change of the perpetual funding rate (BTC only), ÷ its σ per funding print
  lead     surprise of an earlier related release already published by t (ADP before NFP; CPI/PPI before PCE)

Composite score(t) = Σ wₖ·fₖ(t) / √(Σ|wₖ| over available features), where the orientation wₖ ∈ {−1, 0, +1} is fitted
on *past* events only (sign of the correlation between the full-window feature and the post-release 24 h return,
kept if |t| ≥ 1). A signal is the first step at which |score| ≥ THRESHOLD; it trades in the score's direction from
the next bar (market entry, fees + slippage) with a 1σ stop and targets at 1R / 1.75R / 2.5R, closing by T + 24 h.

Evaluation is walk-forward over events: an event is predicted only from events whose outcome (T + 24 h) was known
before its window opened. Each lead window is evaluated on its own, and a nested choice of the window (made from past
events only) gives the one out-of-sample track record that the edge gate judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from backtest import engine

WINDOWS_H = (6, 12, 24, 48, 72)
THRESHOLD = 1.5
MIN_ELAPSED = pd.Timedelta(hours=1)      # the first hour of a window is too short to judge
ENTRY_GAP = pd.Timedelta(minutes=15)     # last signal step is one bar before the release
HOLD = pd.Timedelta(hours=24)            # post-release horizon (label and trade exit)
BASELINE = pd.Timedelta(days=90)
MIN_TRAIN = 12
MIN_ORIENT_T = 1.0
MIN_SELECT_EVENTS = 8
TP_R = (1.0, 1.75, 2.5)
FEATURES = ("drift", "steady", "flow", "rate", "oi", "funding", "lead")
BAR = pd.Timedelta(minutes=15)
EDGE_MIN_TRADES = 30
EDGE_MAX_P = 0.05


@dataclass
class Inputs:
    """Everything the model may read. Each frame carries the time at which a row became known."""
    bars: pd.DataFrame                     # ts (bar open, UTC), open, high, low, close, volume, taker_buy
    rate: pd.DataFrame | None = None       # available_at (UTC), value — 2-year yield, % (known_series of ust_2y)
    oi: pd.DataFrame | None = None         # ts (UTC, known at), open_interest
    funding: pd.DataFrame | None = None    # ts (UTC, known at), funding_rate
    lead: pd.DataFrame | None = None       # event_id, available_at (UTC), z  — lead surprises per target event

    def truncated(self, t: pd.Timestamp) -> Inputs:
        """Only what existed at t (the look-ahead test compares results against this)."""
        cut = lambda df, col: None if df is None else df[df[col] <= t].reset_index(drop=True)  # noqa: E731
        return Inputs(bars=self.bars[self.bars["ts"] + BAR <= t].reset_index(drop=True),
                      rate=cut(self.rate, "available_at"), oi=cut(self.oi, "ts"), funding=cut(self.funding, "ts"),
                      lead=cut(self.lead, "available_at"))


# ───────────────────────────── features ─────────────────────────────

def _slice(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows with start ≤ ts ≤ end by binary search (bars sorted by ts) — avoids full-length masks per event."""
    ns = bars["ts"].to_numpy(dtype="datetime64[ns]")
    i0 = np.searchsorted(ns, np.datetime64(start.tz_convert(None)), side="left")
    i1 = np.searchsorted(ns, np.datetime64(end.tz_convert(None)), side="right")
    return bars.iloc[i0:i1]


def _hour_of_week(ts: pd.Series) -> np.ndarray:
    return (ts.dt.dayofweek * 24 + ts.dt.hour).to_numpy()


def _asof(df: pd.DataFrame | None, col_t: str, col_v: str, times: pd.DatetimeIndex) -> np.ndarray:
    if df is None or df.empty:
        return np.full(len(times), np.nan)
    d = df[[col_t, col_v]].dropna().sort_values(col_t)
    idx = np.searchsorted(d[col_t].to_numpy(dtype="datetime64[ns]"), times.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    v = d[col_v].to_numpy(dtype=float)
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)


def feature_path(inp: Inputs, T: pd.Timestamp, W: float, event_id: str | None = None,
                 until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Features at every 15-min step of the window [T−W, min(T−15m, until)] (index = step time = bar close)."""
    s = T - pd.Timedelta(hours=W)
    last = T - ENTRY_GAP if until is None else min(T - ENTRY_GAP, until)
    b = _slice(inp.bars, s - BASELINE - BAR, last)
    closes = b["ts"] + BAR
    base = b[(closes <= s) & (closes > s - BASELINE)]
    win = b[(closes > s) & (closes <= last)]
    prev = b[closes <= s]
    if len(base) < 96 * 30 or win.empty or prev.empty:
        return pd.DataFrame(columns=FEATURES)
    # normal volatility by hour of week, from the 90 days before the window
    lr_base = np.log(base["close"].to_numpy() / base["open"].to_numpy())
    how = pd.Series(lr_base ** 2).groupby(_hour_of_week(base["ts"])).mean()
    var_all = float(np.mean(lr_base ** 2))
    c0 = float(prev["close"].iloc[-1])
    c = win["close"].to_numpy(dtype=float)
    r_bar = np.diff(np.log(np.r_[c0, c]))
    ev = pd.Series(_hour_of_week(win["ts"])).map(how).fillna(var_all).to_numpy()
    n = np.arange(1, len(c) + 1)
    cum_r, cum_r2, cum_ev = np.cumsum(r_bar), np.cumsum(r_bar ** 2), np.cumsum(ev)
    drift = cum_r / np.sqrt(cum_ev)
    mean = cum_r / n
    var = np.where(n > 1, (cum_r2 - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    steady = np.where((n >= 8) & (var > 0), mean / np.sqrt(np.maximum(var, 1e-18) / n), np.nan)
    # taker flow
    flow = np.full(len(c), np.nan)
    if "taker_buy" in b.columns:
        xb = ((2 * base["taker_buy"] - base["volume"]) / base["volume"].replace(0, np.nan)).to_numpy(dtype=float)
        xw = ((2 * win["taker_buy"] - win["volume"]) / win["volume"].replace(0, np.nan)).to_numpy(dtype=float)
        mu, sd = np.nanmean(xb), np.nanstd(xb)
        if np.isfinite(mu) and sd > 0 and np.isfinite(xw).any():
            xw = np.where(np.isfinite(xw), xw, mu)
            flow = (np.cumsum(xw) / n - mu) / (sd / np.sqrt(n))
    steps = pd.DatetimeIndex(win["ts"] + BAR)
    out = pd.DataFrame({"drift": drift, "steady": steady, "flow": flow}, index=steps)
    # 2-year yield (daily, known after its release lag)
    out["rate"] = np.nan
    if inp.rate is not None and not inp.rate.empty:
        rk = inp.rate[inp.rate["available_at"] <= s].sort_values("available_at")
        if len(rk) > 60:
            sd = float(rk["value"].diff().tail(250).std()) * 100
            v_s = _asof(inp.rate, "available_at", "value", pd.DatetimeIndex([s]))[0]
            v_t = _asof(inp.rate, "available_at", "value", steps)
            k = np.searchsorted(inp.rate.sort_values("available_at")["available_at"].to_numpy(dtype="datetime64[ns]"),
                                steps.to_numpy(dtype="datetime64[ns]"), side="right") - \
                np.searchsorted(inp.rate.sort_values("available_at")["available_at"].to_numpy(dtype="datetime64[ns]"),
                                np.datetime64(s.tz_convert(None)), side="right")
            if sd > 0 and np.isfinite(v_s):
                out["rate"] = np.where(k > 0, (v_t - v_s) * 100 / (sd * np.sqrt(np.maximum(k, 1))), 0.0)
    # open interest, in the direction of the drift
    out["oi"] = np.nan
    if inp.oi is not None and not inp.oi.empty:
        ob = inp.oi[(inp.oi["ts"] <= s) & (inp.oi["ts"] > s - pd.Timedelta(days=30))]
        if len(ob) > 48:  # 5-min archive in backtests, hourly OKX history live: the sampling step is inferred
            lo = np.log(ob["open_interest"].to_numpy(dtype=float))
            step = ob["ts"].diff().median()
            sd_step = float(np.nanstd(np.diff(lo)))
            o_s = _asof(inp.oi, "ts", "open_interest", pd.DatetimeIndex([s]))[0]
            o_t = _asof(inp.oi, "ts", "open_interest", steps)
            n_steps = np.maximum(((steps - s) / step).to_numpy(dtype=float), 1.0)
            if sd_step > 0 and o_s > 0:
                z = np.log(o_t / o_s) / (sd_step * np.sqrt(n_steps))
                out["oi"] = z * np.sign(cum_r)
    # funding
    out["funding"] = np.nan
    if inp.funding is not None and not inp.funding.empty:
        fb = inp.funding[(inp.funding["ts"] <= s) & (inp.funding["ts"] > s - BASELINE)]
        if len(fb) > 30:
            sdf = float(fb["funding_rate"].diff().std())
            f_s = _asof(inp.funding, "ts", "funding_rate", pd.DatetimeIndex([s]))[0]
            f_t = _asof(inp.funding, "ts", "funding_rate", steps)
            tsf = inp.funding.sort_values("ts")["ts"].to_numpy(dtype="datetime64[ns]")
            k = np.searchsorted(tsf, steps.to_numpy(dtype="datetime64[ns]"), side="right") - \
                np.searchsorted(tsf, np.datetime64(s.tz_convert(None)), side="right")
            if sdf > 0 and np.isfinite(f_s):
                out["funding"] = np.where(k > 0, (f_t - f_s) / (sdf * np.sqrt(np.maximum(k, 1))), 0.0)
    # leading data published before t
    out["lead"] = np.nan
    if inp.lead is not None and event_id is not None and not inp.lead.empty:
        ld = inp.lead[inp.lead["event_id"] == event_id]
        if not ld.empty:
            # unknown until published: NaN (not 0) — a 0 would count as an available feature in the score and
            # reveal that a lead release is still to come (caught by the truncation test)
            known = np.array([[steps[i] >= r.available_at for r in ld.itertuples()] for i in range(len(steps))])
            zs = ld["z"].to_numpy(dtype=float)
            cnt = known.sum(axis=1)
            out["lead"] = np.where(cnt > 0, (known * zs[None, :]).sum(axis=1) / np.sqrt(np.maximum(cnt, 1)), np.nan)
    return out[list(FEATURES)].clip(-6, 6)


# ───────────────────────────── score and trades ─────────────────────────────

def score(path: pd.DataFrame, w: dict[str, int]) -> pd.Series:
    if path.empty:
        return pd.Series(dtype=float)
    cols = [k for k, v in w.items() if v != 0]
    if not cols:
        return pd.Series(0.0, index=path.index)
    f = path[cols]
    avail = f.notna()
    num = (f.fillna(0) * pd.Series({k: w[k] for k in cols})).sum(axis=1)
    den = np.sqrt(avail.sum(axis=1).clip(lower=1))
    return (num / den).where(avail.any(axis=1), 0.0)


def fit_orientation(X: pd.DataFrame, y: np.ndarray) -> dict[str, int]:
    """Sign of each feature's relation to the post-release return, kept only if |t| ≥ MIN_ORIENT_T."""
    w = {}
    for k in FEATURES:
        x = X[k].to_numpy(dtype=float) if k in X else np.array([])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_TRAIN or np.nanstd(x[ok]) == 0:
            w[k] = 0
            continue
        r = np.corrcoef(x[ok], y[ok])[0, 1]
        t = r * np.sqrt((ok.sum() - 2) / max(1 - r ** 2, 1e-12))
        w[k] = int(np.sign(r)) if abs(t) >= MIN_ORIENT_T else 0
    return w


def post_return(bars: pd.DataFrame, T: pd.Timestamp) -> float:
    """log(close at T + 24 h / open of the bar starting at T); NaN if bars are missing."""
    w = _slice(bars, T, T + HOLD - BAR)
    if w.empty or w["ts"].iloc[0] != T or w["ts"].iloc[-1] + BAR != T + HOLD:
        return np.nan
    return float(np.log(w["close"].iloc[-1] / w["open"].iloc[0]))


def post_vol_ratio(bars: pd.DataFrame, T: pd.Timestamp, how: pd.Series, var_all: float) -> float:
    w = _slice(bars, T, T + HOLD - BAR)
    if len(w) < 48:
        return np.nan
    rv = float((np.log(w["close"] / w["open"]) ** 2).sum())
    ev = float(pd.Series(_hour_of_week(w["ts"])).map(how).fillna(var_all).sum())
    return rv / ev if ev > 0 else np.nan


def horizon_sigma(bars: pd.DataFrame, t: pd.Timestamp, until: pd.Timestamp, vol_mult: float) -> float:
    """Expected σ of the log return from t to `until`, from the 90-day hour-of-week baseline × post-release multiplier."""
    base = _slice(bars, t - BASELINE, t - BAR)
    lr = np.log(base["close"] / base["open"])
    how = (lr ** 2).groupby(_hour_of_week(base["ts"])).mean()
    grid = pd.Series(pd.date_range(t, until - BAR, freq=BAR))
    ev = pd.Series(_hour_of_week(grid)).map(how).fillna(float((lr ** 2).mean())).sum()
    return float(np.sqrt(ev * vol_mult))


def first_crossing(sc: pd.Series, start: pd.Timestamp) -> tuple[pd.Timestamp, int, float] | None:
    s = sc[(sc.index >= start) & (sc.abs() >= THRESHOLD)]
    if s.empty:
        return None
    return s.index[0], int(np.sign(s.iloc[0])), float(s.iloc[0])


def make_signal(bars: pd.DataFrame, t: pd.Timestamp, direction: int, T: pd.Timestamp, vol_mult: float,
                tag: str) -> engine.Signal | None:
    known = _slice(bars, t - pd.Timedelta(days=3), t - BAR)
    if known.empty:
        return None
    px = float(known["close"].iloc[-1])
    sig = horizon_sigma(bars, t, T + HOLD, vol_mult)
    if not np.isfinite(sig) or sig <= 0:
        return None
    stop = px * np.exp(-direction * sig)
    risk = abs(px - stop)
    targets = tuple(px + direction * k * risk for k in TP_R)
    n_bars = int((T + HOLD - t) / BAR)
    return engine.Signal(time=t, direction=direction, entry=None, stop=stop, targets=targets, max_bars=n_bars, tag=tag)


# ───────────────────────────── walk-forward evaluation ─────────────────────────────

@dataclass
class EventRow:
    event_id: str
    T: pd.Timestamp
    paths: dict[float, pd.DataFrame] = field(default_factory=dict)
    y: float = np.nan              # post-release 24 h log return
    vol_ratio: float = np.nan      # realized / normal variance after the release (for the stop size)


def prepare(inp: Inputs, events: pd.DataFrame, windows=WINDOWS_H) -> list[EventRow]:
    """events: event_id, release_utc. Paths for every window; label y; post-release volatility ratio."""
    rows = []
    b = inp.bars
    for e in events.sort_values("release_utc").itertuples():
        T = pd.Timestamp(e.release_utc)
        er = EventRow(e.event_id, T)
        for W in windows:
            er.paths[W] = feature_path(inp, T, W, e.event_id)
        er.y = post_return(b, T)
        base = _slice(b, T - BASELINE, T - BAR)
        if len(base) > 96 * 30:
            lr = np.log(base["close"] / base["open"])
            er.vol_ratio = post_vol_ratio(b, T, (lr ** 2).groupby(_hour_of_week(base["ts"])).mean(), float((lr ** 2).mean()))
        rows.append(er)
    return [r for r in rows if any(not p.empty for p in r.paths.values())]


def _train(rows: list[EventRow], i: int, W: float) -> list[EventRow]:
    """Events whose outcome was known before event i's window opened."""
    s = rows[i].T - pd.Timedelta(hours=W)
    return [r for r in rows[:i] if r.T + HOLD <= s and np.isfinite(r.y) and not r.paths[W].empty]


def _full(r: EventRow, W: float) -> pd.Series:
    return r.paths[W].iloc[-1]


def _fit(train: list[EventRow], W: float) -> tuple[dict[str, int], float]:
    X = pd.DataFrame([_full(r, W) for r in train])
    y = np.array([r.y for r in train])
    vr = np.array([r.vol_ratio for r in train])
    vol_mult = float(np.nanmedian(vr)) if np.isfinite(vr).any() else 1.0
    return fit_orientation(X, y), max(vol_mult, 0.5)


def _full_scores(train: list[EventRow], W: float, w: dict[str, int]) -> np.ndarray:
    """Full-window composite of each event, vectorised (same formula as score())."""
    X = np.array([_full(r, W).reindex(list(FEATURES)).to_numpy(dtype=float) for r in train])
    wv = np.array([w.get(k, 0) for k in FEATURES], dtype=float)
    use = (wv != 0)[None, :] & np.isfinite(X)
    num = np.where(use, np.nan_to_num(X) * wv[None, :], 0).sum(axis=1)
    return np.where(use.any(axis=1), num / np.sqrt(np.maximum(use.sum(axis=1), 1)), 0.0)


def _quality(train: list[EventRow], W: float, w: dict[str, int]) -> tuple[float, int]:
    """In-sample t-stat of the direction-signed post-release return on events whose full-window |score| ≥ threshold."""
    sc = _full_scores(train, W, w)
    y = np.array([r.y for r in train])
    big = np.abs(sc) >= THRESHOLD
    xs = np.sign(sc[big]) * y[big]
    if len(xs) < MIN_SELECT_EVENTS or np.std(xs) == 0:
        return -np.inf, len(xs)
    return float(np.mean(xs) / (np.std(xs, ddof=1) / np.sqrt(len(xs)))), len(xs)


def walk_forward(rows: list[EventRow], bars: pd.DataFrame, costs: engine.CostModel | None = None,
                 windows=WINDOWS_H, tag: str = "") -> dict:
    """Out-of-sample results per window and for the nested window choice."""
    costs = costs or engine.CostModel()
    arr = engine.as_arrays(bars)
    per_w = {W: {"trades": [], "scores": [], "ys": []} for W in windows}
    sel = {"trades": [], "choices": []}
    for i, r in enumerate(rows):
        fits = {}
        for W in windows:
            tr = _train(rows, i, W)
            if len(tr) < MIN_TRAIN or r.paths[W].empty:
                continue
            w, vm = _fit(tr, W)
            fits[W] = (w, vm, tr)
            sc = score(r.paths[W], w)
            if np.isfinite(r.y) and not sc.empty:
                per_w[W]["scores"].append(float(sc.iloc[-1]))
                per_w[W]["ys"].append(r.y)
            per_w[W]["trades"].append(_trade(r, W, sc, vm, arr, bars, costs, tag))
        # nested choice of the window from past events only
        best = max(((W, _quality(f[2], W, f[0])) for W, f in fits.items()), key=lambda x: x[1][0], default=None)
        if best is not None and np.isfinite(best[1][0]):
            W = best[0]
            w, vm, _ = fits[W]
            sel["choices"].append((r.event_id, W))
            sel["trades"].append(_trade(r, W, score(r.paths[W], w), vm, arr, bars, costs, tag))
    return {"per_window": {W: _summ(v["trades"], v["scores"], v["ys"]) for W, v in per_w.items()},
            "selected": _summ(sel["trades"], [], []), "choices": sel["choices"],
            "selected_trades": [t for t in sel["trades"] if t]}


def _trade(r: EventRow, W: float, sc: pd.Series, vol_mult: float, arr: dict, bars: pd.DataFrame,
           costs: engine.CostModel, tag: str) -> dict | None:
    if sc.empty:
        return None
    cross = first_crossing(sc, r.T - pd.Timedelta(hours=W) + MIN_ELAPSED)
    if cross is None:
        return None
    t, d, v = cross
    sig = make_signal(bars, t, d, r.T, vol_mult, f"{tag}{r.event_id}")
    if sig is None:
        return None
    res = engine.simulate(sig, arr, costs)
    return {"event_id": r.event_id, "signal_time": t, "direction": d, "score": v, "filled": res.filled,
            "r": res.r, "tps_hit": res.tps_hit, "stopped": res.stopped, "W": W}


def _summ(trades: list, scores: list, ys: list) -> dict:
    t = pd.DataFrame([x for x in trades if x])
    out: dict = {"n_events": len(trades), "n_signals": int(len(t))}
    if not t.empty:
        t = t[t["filled"]]
    rs = t["r"].to_numpy(dtype=float) if not t.empty else np.array([])
    out["n_trades"] = int(len(rs))
    if len(rs):
        out.update(win_rate=round(float((rs > 0).mean()), 3), avg_r=round(float(rs.mean()), 3),
                   total_r=round(float(rs.sum()), 2),
                   max_drawdown_r=round(float((np.maximum.accumulate(np.r_[0, np.cumsum(rs)])[1:] - np.cumsum(rs)).max()), 2))
        if len(rs) > 2 and rs.std(ddof=1) > 0:
            tstat = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
            out.update(t_r=round(float(tstat), 2), p_r=round(float(stats.t.sf(tstat, len(rs) - 1)), 4))
    if len(scores) >= 10:
        s, y = np.array(scores), np.array(ys)
        rho, p = stats.spearmanr(s, y)
        big = np.abs(s) >= THRESHOLD
        out.update(spearman=round(float(rho), 3), spearman_p=round(float(p), 4))
        if big.sum() >= 5:
            hits = int((np.sign(s[big]) == np.sign(y[big])).sum())
            signed = np.sign(s[big]) * y[big]
            out.update(n_strong=int(big.sum()), hit_rate=round(hits / int(big.sum()), 3),
                       hit_p=round(float(stats.binomtest(hits, int(big.sum()), 0.5, "greater").pvalue), 4),
                       effect_bp=round(float(signed.mean()) * 1e4, 1),
                       effect_d=round(float(signed.mean() / signed.std(ddof=1)), 2) if signed.std(ddof=1) > 0 else None)
    return out


def edge(summary: dict) -> tuple[bool, list[str]]:
    """Phase 4 gate (≥ 30 out-of-sample trades, average R > 0) plus one-sided significance of R (p < 0.05)."""
    why = []
    if summary.get("n_trades", 0) < EDGE_MIN_TRADES:
        why.append(f"too few out-of-sample trades ({summary.get('n_trades', 0)} < {EDGE_MIN_TRADES})")
    if (summary.get("avg_r") or -1) <= 0:
        why.append(f"average R not positive ({summary.get('avg_r')})")
    if summary.get("p_r") is None or summary["p_r"] >= EDGE_MAX_P:
        why.append(f"not significant (p = {summary.get('p_r')})")
    return not why, why


# ───────────────────────────── live model ─────────────────────────────

def live_model(rows: list[EventRow], now: pd.Timestamp, windows=WINDOWS_H) -> dict:
    """Orientation per window from every event whose outcome is known now, and the window a new event would use."""
    out = {"windows": {}, "selected_window": None}
    best = (-np.inf, None)
    for W in windows:
        tr = [r for r in rows if r.T + HOLD <= now and np.isfinite(r.y) and not r.paths[W].empty]
        if len(tr) < MIN_TRAIN:
            continue
        w, vm = _fit(tr, W)
        q, n = _quality(tr, W, w)
        out["windows"][W] = {"weights": w, "vol_mult": round(vm, 3), "quality_t": None if not np.isfinite(q) else round(q, 2),
                             "n_train": len(tr), "n_strong": n}
        if q > best[0]:
            best = (q, W)
    out["selected_window"] = best[1]
    return out
