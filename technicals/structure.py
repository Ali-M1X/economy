"""Swing points and market structure, computed causally.

A swing high at bar i is the highest high of bars [i−k, i+k]; it only becomes *known* at bar i+k (when the
k bars to its right have closed). Every structure field at bar t uses swings known at t — never later ones.

Structure labels: each new swing high is HH (above the previous swing high) or LH; each swing low HL or LL.
State at t: bullish if the last swing high is HH and the last swing low is HL; bearish if LH and LL; else range.
Events on bar closes:
- BOS (break of structure): close beyond the last swing extreme *in the direction of* the current state;
- CHOCH (change of character): close beyond the last swing extreme *against* the current state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def swings(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Rows: idx (bar of the swing), known_at (idx + k), kind ('H'/'L'), price."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    rows = []
    for i in range(k, n - k):
        win_h, win_l = h[i - k:i + k + 1], l[i - k:i + k + 1]
        if h[i] == win_h.max() and np.argmax(win_h) == k:
            rows.append((i, i + k, "H", h[i]))
        if l[i] == win_l.min() and np.argmin(win_l) == k:
            rows.append((i, i + k, "L", l[i]))
    return pd.DataFrame(rows, columns=["idx", "known_at", "kind", "price"]).sort_values(["known_at", "idx"]).reset_index(drop=True)


def structure(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Per-bar causal structure: last known swing high/low (+ previous), labels, state, BOS/CHOCH events."""
    sw = swings(df, k)
    n = len(df)
    close = df["close"].to_numpy()
    out = {c: np.full(n, np.nan) for c in ("sh", "sh_prev", "sl", "sl_prev")}
    state = np.zeros(n, dtype=int)
    event = np.array([""] * n, dtype=object)
    sh = sh_prev = sl = sl_prev = np.nan
    j = 0
    rows = sw.to_numpy()
    cur_state = 0
    for t in range(n):
        while j < len(rows) and rows[j][1] <= t:  # swing known at bar t
            _, _, kind, price = rows[j]
            if kind == "H":
                sh_prev, sh = sh, price
            else:
                sl_prev, sl = sl, price
            j += 1
        hh = sh > sh_prev if np.isfinite(sh_prev) else None
        hl = sl > sl_prev if np.isfinite(sl_prev) else None
        if hh is True and hl is True:
            cur_state = 1
        elif hh is False and hl is False:
            cur_state = -1
        # events on this bar's close relative to the swings known *before* it closed
        if np.isfinite(sh) and close[t] > sh and (t == 0 or close[t - 1] <= sh):
            event[t] = "BOS_UP" if cur_state >= 0 else "CHOCH_UP"
        elif np.isfinite(sl) and close[t] < sl and (t == 0 or close[t - 1] >= sl):
            event[t] = "BOS_DN" if cur_state <= 0 else "CHOCH_DN"
        state[t] = cur_state
        out["sh"][t], out["sh_prev"][t], out["sl"][t], out["sl_prev"][t] = sh, sh_prev, sl, sl_prev
    res = pd.DataFrame(out, index=df.index)
    res["state"] = state
    res["event"] = event
    return res


def recent_swings(sw: pd.DataFrame, t: int, kind: str, lookback: int = 60) -> np.ndarray:
    """Prices of swings of `kind` known at bar t whose bar is within `lookback` bars."""
    m = (sw["known_at"] <= t) & (sw["kind"] == kind) & (sw["idx"] >= t - lookback)
    return sw.loc[m, "price"].to_numpy()
