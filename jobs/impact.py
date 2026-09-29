"""Phase 3: impact coefficients (short/medium/long, BTC & gold) and the composite Macro Score.

    python -m jobs.impact --from-cache output/cache      # no database
    python -m jobs.impact                                 # from the database (when storage is enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.registry import load_registry
from core.settings import OUTPUT_DIR, ROOT
from core.timeutil import utcnow
from features import pit
from jobs.features import REGIME_KEYS, load_cache
from models import events, impact, regime

OUT_DIR = OUTPUT_DIR
BTC_START = pd.Timestamp("2017-01-01")
GOLD_START = pd.Timestamp("2000-01-01")
RECENT_YEARS = 3


def indicator_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text(encoding="utf-8"))


def load_bars(cache: Path, asset: str) -> pd.DataFrame | None:
    p = cache / f"candles_{asset}_15m.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# ───────────────────────────── short horizon ─────────────────────────────

def short_horizon(cfg: dict, vintages: dict, bars: dict[str, pd.DataFrame], now: pd.Timestamp):
    ests, all_events = [], []
    for rel in cfg["releases"]:
        v = vintages.get(rel["key"])
        if v is None or v.empty:
            continue
        freq = "W" if rel["key"] == "initial_claims" else "M"
        ev = events.surprises(v, rel["key"], rel["measure"], rel["expected"], rel["release"], freq=freq)
        all_events.append(ev)
        for asset, b in bars.items():
            if b is None:
                continue
            er = events.event_returns(ev, b)
            er = er[er["release_utc"] >= b["ts"].min()]
            ests += impact.event_study(er, rel["key"], asset, "full")
            if asset == "BTC":
                recent = er[er["release_utc"] >= now - pd.DateOffset(years=RECENT_YEARS)]
                ests += impact.event_study(recent, rel["key"], asset, "recent")
    evs = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return ests, evs


# ───────────────────────────── medium / long ─────────────────────────────

def state_panel(data: dict, vintages: dict, grid: pd.DatetimeIndex, keys: list[str]) -> pd.DataFrame:
    reg = load_registry()
    known = {k: pit.known_series(data[k], reg[k], vintages.get(k)) for k in keys if k in data and k in reg}
    panel = pit.asof_panel(known, grid)
    if {"fed_balance_sheet", "tga_weekly", "reverse_repo"} <= set(panel.columns):
        panel["net_liquidity_weekly"] = (panel["fed_balance_sheet"] / 1000 - panel["tga_weekly"] / 1000
                                         - panel["reverse_repo"]).where(panel.index >= "2013-09-30")
    return panel


def asset_prices(data: dict) -> dict[str, pd.Series]:
    s = lambda k: data[k].set_index("date")["value"].sort_index()  # noqa: E731
    return {"BTC": s("btc_usd_daily")[lambda x: x.index >= BTC_START],
            "Gold": s("gold_futures")[lambda x: x.index >= GOLD_START]}


def medium_long(cfg: dict, data: dict, vintages: dict, regimes: pd.Series, now: pd.Timestamp):
    keys = sorted({i["key"] for i in cfg["state"] if not i.get("derived")} | {"tga_weekly", "reverse_repo"})
    wk = state_panel(data, vintages, pd.date_range("1999-01-01", now, freq="W-FRI"), keys)
    mo = state_panel(data, vintages, pd.date_range("1999-01-31", now, freq="ME"), keys)
    ests = []
    for asset, px in asset_prices(data).items():
        pw, pm = impact.weekly(px), impact.monthly(px)
        samples = {"full": None}
        if asset == "BTC":
            samples["recent"] = now - pd.DateOffset(years=RECENT_YEARS)
        for ind in cfg["state"]:
            k, how = ind["key"], ind["change"]
            if k not in wk:
                continue
            for sample, since in samples.items():
                iw, pww = wk[k], pw if since is None else pw[pw.index >= since]
                im, pmm = mo[k], pm if since is None else pm[pm.index >= since]
                for h in (1, 2, 4):
                    ests.append(impact.projection(iw, pww, how, 4, h, k, asset, f"{h}w", "medium", sample))
                for h in (3, 6, 12):
                    ests.append(impact.projection(im, pmm, how, 3, h, k, asset, f"{h}m", "long", sample))
            # regime-conditional long-horizon effect (full sample)
            for reg_name in regime.REGIMES:
                mask = (regimes == reg_name)
                ests.append(impact.projection(mo[k], pm, how, 3, 6, k, asset, "6m", "long", f"regime:{reg_name}", mask=mask))
    return ests, wk, mo


# ───────────────────────────── macro score ─────────────────────────────

def theory_signs(cfg: dict) -> dict[tuple[str, str, str], int]:
    """(section, indicator, asset) → prior sign; section 'short' uses release priors, else state priors."""
    out = {}
    for sec, items in (("short", cfg["releases"]), ("state", cfg["state"])):
        for i in items:
            for asset, sign in (i.get("theory") or {}).items():
                out[(sec, i["key"], asset)] = int(sign)
    return out


def flag_conflicts(coefs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    th = theory_signs(cfg)
    sec = np.where(coefs["bucket"] == "short", "short", "state")
    coefs = coefs.copy()
    coefs["theory"] = [th.get((s_, i, a), 0) for s_, i, a in zip(sec, coefs["indicator"], coefs["asset"])]
    coefs["conflict"] = (coefs["theory"] != 0) & (np.sign(coefs["coefficient"]) * coefs["theory"] < 0) \
        & (coefs["coefficient"].abs() >= 1)
    return coefs


def headline(coefs: pd.DataFrame, asset: str, bucket: str) -> pd.Series:
    """Signed headline coefficient per indicator. BTC blends full-sample and recent (last 3y) equally,
    because its macro sensitivity changed over time."""
    h = impact.HEADLINE[bucket]
    c = coefs[(coefs["asset"] == asset) & (coefs["bucket"] == bucket) & (coefs["horizon"] == h)]
    # coefficients that contradict the theory prior count at half weight in the score
    c = c.assign(w=np.where(c["conflict"], 0.5, 1.0) * c["coefficient"])
    full = c[c["sample"] == "full"].set_index("indicator")["w"]
    if asset != "BTC":
        return full
    recent = c[c["sample"] == "recent"].set_index("indicator")["w"]
    return pd.concat([full, recent], axis=1).mean(axis=1)


def current_state(panel: pd.DataFrame, cfg: dict, periods: int) -> pd.Series:
    """Latest change of each indicator in σ units (std of that change over the sample), clipped to ±2 then /2."""
    out = {}
    for ind in cfg["state"]:
        k = ind["key"]
        if k not in panel:
            continue
        ch = impact.change(panel[k], periods, ind["change"]).dropna()
        if len(ch) < 30 or not ch.std():
            continue
        out[k] = float(np.clip(ch.iloc[-1] / ch.std(), -2, 2) / 2)
    return pd.Series(out)


SCORE_EVIDENCE_FLOOR = 30.0  # ≈ three full-strength (|coef| = 10) relationships


def macro_score(coef: pd.Series, state: pd.Series) -> tuple[float | None, pd.DataFrame]:
    """Score = 100 × Σ coefᵢ·stateᵢ / max(Σ|coefᵢ|, 30), stateᵢ ∈ [−1, 1].
    The floor keeps uniformly weak evidence from producing a strong-looking score: with only weak
    coefficients (e.g. gold's medium horizon, all |coef| < 1) the score stays near 0."""
    j = pd.concat([coef.rename("coef"), state.rename("state")], axis=1).dropna()
    j = j[j["coef"] != 0]
    if j.empty:
        return None, j
    denom = max(j["coef"].abs().sum(), SCORE_EVIDENCE_FLOOR)
    j["contribution"] = 100 * j["coef"] * j["state"] / denom
    return round(float(j["contribution"].sum()), 1), j.sort_values("contribution", key=abs, ascending=False)


def scores(coefs: pd.DataFrame, wk: pd.DataFrame, mo: pd.DataFrame, evs: pd.DataFrame, cfg: dict, now: pd.Timestamp) -> dict:
    out = {}
    recent_ev = evs[pd.to_datetime(evs["release_utc"], utc=True) >= now - pd.Timedelta(hours=24)] if not evs.empty else evs
    short_state = (recent_ev.groupby("indicator")["z"].last().clip(-2, 2) / 2) if not recent_ev.empty else pd.Series(dtype=float)
    for asset in ("BTC", "Gold"):
        out[asset] = {}
        for bucket, state in (("short", short_state), ("medium", current_state(wk, cfg, 4)), ("long", current_state(mo, cfg, 3))):
            sc, br = macro_score(headline(coefs, asset, bucket), state)
            out[asset][bucket] = {
                "score": sc if bucket != "short" or not short_state.empty else None,
                "note": "no tracked release in the last 24h" if bucket == "short" and short_state.empty else None,
                "top": [{"indicator": i, "coef": round(r.coef, 2), "state": round(r.state, 2),
                         "contribution": round(r.contribution, 1)} for i, r in br.head(6).iterrows()],
            }
    return out


def recent_releases(evs: pd.DataFrame, now: pd.Timestamp, days: int = 7) -> list[dict]:
    """First-print surprises of the last `days` (newest first) — the "what changed" input of the reports."""
    if evs.empty:
        return []
    e = evs[pd.to_datetime(evs["release_utc"], utc=True) >= now - pd.Timedelta(days=days)]
    e = e.sort_values("release_utc", ascending=False)
    return [{"indicator": r.indicator, "release_utc": pd.Timestamp(r.release_utc).isoformat(),
             "obs_date": f"{pd.Timestamp(r.obs_date):%Y-%m-%d}", "actual": round(float(r.actual), 4),
             "expected": round(float(r.expected), 4), "surprise": round(float(r.surprise), 4), "z": round(float(r.z), 2),
             "basis": r.surprise_basis} for r in e.itertuples()]


# ───────────────────────────── report ─────────────────────────────

def render(coefs: pd.DataFrame, evs: pd.DataFrame, sc: dict, now: pd.Timestamp) -> str:
    L = ["# Phase 3 — Impact coefficients & Macro Score", "", f"As of {now:%Y-%m-%d %H:%M} UTC", "",
         "Impact Coefficient: signed 0–10 (+ = indicator up / hotter surprise → asset up). "
         "Descriptive historical co-movement, not a forecast. See `models/impact.py` for the formula.",
         "Theory column: textbook sign from `config/indicators.yaml`; ⚠ = the estimate contradicts it "
         "(kept as estimated, but half weight in the Macro Score). Short-horizon gold uses PAXG (proxy, from 2020-09).", ""]
    L += ["## Macro Score (−100 … +100)", "", "| asset | short (24h events) | medium (4w) | long (6m) |", "|---|---|---|---|"]
    for a, d in sc.items():
        cell = lambda b: "—" if d[b]["score"] is None else f"{d[b]['score']:+.0f}"  # noqa: E731
        L.append(f"| {a} | {cell('short')} | {cell('medium')} | {cell('long')} |")
    for a, d in sc.items():
        for b in ("medium", "long"):
            if d[b]["top"]:
                L.append(f"\n**{a} {b}** top contributions: " + ", ".join(
                    f"{t['indicator']} {t['contribution']:+.1f} (coef {t['coef']:+.1f} × state {t['state']:+.2f})" for t in d[b]["top"]))
    for bucket, title in (("short", "Short horizon — release surprises (event study)"),
                          ("medium", "Medium horizon — 4-week local projections"),
                          ("long", "Long horizon — 6-month")):
        h = impact.HEADLINE[bucket]
        c = coefs[(coefs["bucket"] == bucket) & (coefs["horizon"] == h) & (~coefs["sample"].str.startswith("regime"))]
        L += ["", f"## {title} (headline {h})", "",
              "| indicator | asset | sample | coef | β per 1σ (%) | t | n | hit | conf | theory |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in c.sort_values("coefficient", key=abs, ascending=False).itertuples():
            hit = "" if r.hit_rate is None or pd.isna(r.hit_rate) else f"{r.hit_rate:.0%}"
            th = "⚠ conflicts" if r.conflict else {1: "+", -1: "−", 0: "?"}[r.theory]
            L.append(f"| {r.indicator} | {r.asset} | {r.sample} | {r.coefficient:+.1f} | {r.beta:+.3f} | {r.t_stat:+.2f} | "
                     f"{r.n} | {hit} | {r.confidence} | {th} |")
    rc = coefs[coefs["sample"].str.startswith("regime")]
    if not rc.empty:
        L += ["", "## Regime-conditional long-horizon coefficients (6m, |coef| ≥ 3)", "",
              "| indicator | asset | regime | coef | t | n |", "|---|---|---|---|---|---|"]
        for r in rc[rc["coefficient"].abs() >= 3].sort_values(["asset", "indicator"]).itertuples():
            L.append(f"| {r.indicator} | {r.asset} | {r.sample.split(':')[1]} | {r.coefficient:+.1f} | {r.t_stat:+.2f} | {r.n} |")
    if not evs.empty:
        L += ["", "## Release events used", "", "| indicator | releases | first | last | surprise basis |", "|---|---|---|---|---|"]
        for k, g in evs.groupby("indicator"):
            L.append(f"| {k} | {len(g)} | {g['release_utc'].min():%Y-%m-%d} | {g['release_utc'].max():%Y-%m-%d} | {g['surprise_basis'].iloc[0]} |")
    return "\n".join(L) + "\n"


def store(coefs: pd.DataFrame, sc: dict, now: pd.Timestamp, eng=None) -> dict[str, int]:
    """Append this run to the history tables (weekly recompute keeps a time series of coefficients)."""
    from db import schema
    from db.store import engine, init_db, upsert

    eng = eng or engine()
    init_db(eng)
    day = now.date()
    rows = [{"computed_on": day, "indicator": r.indicator, "asset": r.asset, "horizon": r.horizon, "sample": r.sample,
             "bucket": r.bucket, "coefficient": float(r.coefficient),
             "effect_per_sigma": None if pd.isna(r.beta) else float(r.beta),
             "t_stat": None if pd.isna(r.t_stat) else float(r.t_stat),
             "hit_rate": None if r.hit_rate is None or pd.isna(r.hit_rate) else float(r.hit_rate),
             "n": int(r.n), "confidence": r.confidence,
             "details": {"sd_ret": None if pd.isna(r.sd_ret) else float(r.sd_ret), "n_eff": float(r.n_eff),
                         "theory": int(r.theory), "conflict": bool(r.conflict)}}
            for r in coefs.itertuples()]
    n1 = upsert(schema.impact_coefficients, rows, eng)
    n2 = upsert(schema.macro_scores, [{"computed_at": now.to_pydatetime(), "asset": a, "bucket": b, "score": d["score"],
                                       "contributions": d["top"], "note": d["note"]}
                                      for a, buckets in sc.items() for b, d in buckets.items()], eng)
    return {"impact_coefficients": n1, "macro_scores": n2}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", required=True, help="snapshot dir from jobs.availability --cache-dir")
    ap.add_argument("--store", action="store_true", help="also write coefficients and scores to DATABASE_URL")
    args = ap.parse_args(argv)
    cache = Path(args.from_cache)
    now = pd.Timestamp(utcnow())
    cfg = indicator_config()
    data, vintages, *_ = load_cache(cache)
    bars = {"BTC": load_bars(cache, "BTC"), "Gold(PAXG)": load_bars(cache, "PAXG")}
    short_ests, evs = short_horizon(cfg, vintages, bars, now)
    # regimes (PIT) for the conditional estimates — same model as Phase 2
    reg = load_registry()
    known = {k: pit.known_series(data[k], reg[k], vintages.get(k)) for k in REGIME_KEYS if k in data}
    grid = pd.date_range("1991-01-31", now.tz_localize(None), freq="ME")
    rules = regime.rule_probabilities(regime.axes(pit.asof_panel(known, grid)))
    ml_ests, wk, mo = medium_long(cfg, data, vintages, rules["regime"], now.tz_localize(None))
    coefs = impact.estimates_frame(short_ests + ml_ests)
    coefs["asset"] = coefs["asset"].replace({"Gold(PAXG)": "Gold"})
    coefs = flag_conflicts(coefs, cfg)
    sc = scores(coefs, wk, mo, evs, cfg, now)
    OUT_DIR.mkdir(exist_ok=True)
    coefs.to_csv(OUT_DIR / "impact_coefficients.csv", index=False)
    (OUT_DIR / "macro_scores.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    (OUT_DIR / "recent_releases.json").write_text(json.dumps(recent_releases(evs, now), indent=1), encoding="utf-8")
    if args.store:
        print(f"stored: {store(coefs, sc, now)}")
    md = render(coefs, evs, sc, now)
    (OUT_DIR / "impact_summary.md").write_text(md, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n\n" + md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover
        traceback.print_exc()
        sys.exit(1)
