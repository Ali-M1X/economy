"""Phase 2: derived features, curve inversions, FedWatch, Fed Stance Score, regimes → DB + summary.

    python -m jobs.features            # reads DATABASE_URL, writes features/regimes/fedwatch/stance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

from core.registry import load_registry
from core.settings import OUTPUT_DIR
from core.timeutil import utcnow
from db import schema
from db.store import engine, init_db, load_all_series, load_table, load_vintages, upsert
from features import curve, derived, fed_stance, fedwatch, pit
from features.transforms import as_series
from models import regime

OUT_DIR = OUTPUT_DIR
REGIME_KEYS = ["philly_fed_activity", "empire_state_activity", "spread_10y_3m", "baa_10y_spread", "nfci",
               "initial_claims", "nonfarm_payrolls", "m2", "fed_balance_sheet", "core_cpi"]


def _d(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).date()


def compute_all(data: dict[str, pd.DataFrame], vintages: dict[str, pd.DataFrame], futures_q: pd.DataFrame,
                calendar_ev: pd.DataFrame, fed_docs: pd.DataFrame, today: date) -> dict:
    reg = load_registry()
    res: dict = {"today": today}

    # 1) derived series
    res["features"] = derived.compute(data)

    # 2) curve inversions
    res["episodes"] = {}
    for key in ("spread_10y_2y", "spread_10y_3m"):
        if key in data:
            res["episodes"][key] = curve.inversion_episodes(as_series(data[key]))
    if "spread_30y_2y" in res["features"]:
        res["episodes"]["spread_30y_2y"] = curve.inversion_episodes(as_series(res["features"]["spread_30y_2y"]))
    res["curve_state"] = {k: curve.curve_state(as_series(data[k]) if k in data else as_series(res["features"][k]), ep)
                          for k, ep in res["episodes"].items()}

    # 3) FedWatch from the latest quote of each ZQ contract
    effr = as_series(data["fed_funds_eff_daily"])
    effr_now = float(effr.iloc[-1])
    lower_now = float(as_series(data["fed_target_lower"]).iloc[-1])
    upper_now = float(as_series(data["fed_target_upper"]).iloc[-1])
    zq = futures_q[futures_q["root"] == "ZQ"].sort_values("quote_ts").drop_duplicates("contract", keep="last") \
        if not futures_q.empty else futures_q
    meetings = sorted(pd.to_datetime(calendar_ev.loc[calendar_ev["release_id"] == "fomc", "scheduled_utc"], utc=True)
                      .dt.tz_convert("America/New_York").dt.date.unique()) if not calendar_ev.empty else []
    upcoming = [m for m in meetings if m >= today]
    implied = {pd.Timestamp(r.contract_month).date(): float(r.implied_rate) for r in zq.itertuples()} if not zq.empty else {}
    res["fedwatch"] = fedwatch.probabilities(implied, upcoming, effr_now, lower_now) if implied and upcoming else pd.DataFrame()
    res["fedwatch_inputs"] = {"effr_now": effr_now, "target": [lower_now, upper_now], "contracts": len(implied),
                              "meetings": [m.isoformat() for m in upcoming[:8]]}

    # 4) Fed Stance Score
    sep = data.get("sep_fed_funds_median")
    zq_curve = zq.assign(contract_month=pd.to_datetime(zq["contract_month"])) if not zq.empty else None
    res["stance"] = fed_stance.stance_score(
        today=today, target_upper=as_series(data["fed_target_upper"]), effr_now=effr_now,
        target_mid=(lower_now + upper_now) / 2, curve=zq_curve, walcl=as_series(data["fed_balance_sheet"]),
        docs=fed_docs if not fed_docs.empty else None, sep_latest=sep)

    # 5) Regimes on a point-in-time monthly panel
    known = {k: pit.known_series(data[k], reg[k], vintages.get(k)) for k in REGIME_KEYS if k in data}
    grid = pd.date_range("1991-01-31", pd.Timestamp(today), freq="ME")
    if grid[-1] != pd.Timestamp(today):
        grid = grid.append(pd.DatetimeIndex([pd.Timestamp(today)]))  # current, partial month: data known today
    panel = pit.asof_panel(known, grid)
    ax = regime.axes(panel)
    rules = regime.rule_probabilities(ax)
    res["regime_axes"], res["regime_rules"] = ax, rules
    res["regime_hmm"] = regime.hmm_regimes(ax)
    last = rules.iloc[-1]
    res["regime_now"] = {"date": rules.index[-1].date(), "regime": last["regime"],
                         "probabilities": {r: round(float(last[r]), 3) for r in regime.REGIMES},
                         "triggers": regime.switch_triggers(ax.loc[rules.index[-1]], last["regime"]),
                         # latest axes (growth level G, momentum g, inflation/liquidity pressure p, outlook m) —
                         # the reports turn their signs into a plain-language outlook sentence
                         "axes": {k: (None if pd.isna(v) else round(float(v), 3))
                                  for k, v in ax.loc[rules.index[-1], ["G", "g", "p", "m"]].items()}}
    prices = {}
    if "btc_usd_daily" in data:
        prices["BTC"] = as_series(data["btc_usd_daily"])
    if "gold_futures" in data:
        prices["Gold"] = as_series(data["gold_futures"])
    res["regime_perf"] = regime.regime_performance(rules["regime"], prices)
    if "nber_recession" in data:
        nber = as_series(data["nber_recession"]).resample("ME").last()
        j = pd.concat([rules["regime"], nber.rename("rec")], axis=1, join="inner").dropna()
        rec = j[j["rec"] == 1]
        res["nber_check"] = {"recession_months": int(len(rec)),
                             "classified_recession_or_peak": round(float(rec["regime"].isin(["Recession", "Peak"]).mean()), 2)
                             if len(rec) else None,
                             "expansion_months_classified_recession": round(float((j.loc[j["rec"] == 0, "regime"] == "Recession").mean()), 2)}
    # QE/QT segments
    if "qe_qt_regime" in res["features"]:
        res["qe_qt_segments"] = derived.regime_segments(as_series(res["features"]["qe_qt_regime"]))
    return res


def store(res: dict, eng=None) -> dict[str, int]:
    now = utcnow()
    n = {}
    rows = [{"feature_key": k, "obs_date": d.date(), "value": float(v), "computed_at": now}
            for k, df in res["features"].items() for d, v in zip(df["date"], df["value"])]
    rows += [{"feature_key": "fed_stance_score", "obs_date": res["today"], "value": float(res["stance"].score),
              "computed_at": now}] if res["stance"].score is not None else []
    n["features"] = upsert(schema.features, rows, eng)
    ep_rows = [{"spread_key": k, "start_date": _d(e.start), "end_date": _d(e.end), "duration_days": int(e.duration_days),
                "depth": float(e.depth), "depth_date": _d(e.depth_date), "resteepen_date": _d(e.resteepen_date),
                "label": (f"(brief) {e.label or ''}".strip() if e.brief else e.label), "computed_at": now}
               for k, eps in res["episodes"].items() for e in eps.itertuples()]
    n["inversion_episodes"] = upsert(schema.inversion_episodes, ep_rows, eng)
    fw = res["fedwatch"]
    n["fedwatch"] = upsert(schema.fedwatch, [{
        "computed_at": now, "meeting_date": r.meeting, "implied_rate_after": r.implied_rate_after,
        "expected_change_bp": r.expected_change_bp, "p_cut": r.p_cut, "p_hold": r.p_hold, "p_hike": r.p_hike,
        "cumulative_vs_today_bp": r.cumulative_vs_today_bp, "target_range_probs": r.target_range_probs,
        "method": r.method} for r in fw.itertuples()], eng) if not fw.empty else 0
    st = res["stance"]
    n["fed_stance"] = upsert(schema.fed_stance, [{"obs_date": res["today"], "score": st.score, "components": st.components,
                                                  "inputs": json.loads(json.dumps(st.inputs, default=str)),
                                                  "computed_at": now}], eng)
    reg_rows = []
    for model, probs in (("rules", res["regime_rules"]), ("hmm", res["regime_hmm"].probs if res["regime_hmm"] else None)):
        if probs is None:
            continue
        reg_rows += [{"model": model, "obs_date": t.date(), "regime": r["regime"],
                      "probabilities": {k: round(float(r[k]), 4) for k in regime.REGIMES}, "computed_at": now}
                     for t, r in probs.iterrows()]
    n["regime_states"] = upsert(schema.regime_states, reg_rows, eng)
    return n


def render(res: dict, stored: dict | None) -> str:
    L = ["# Phase 2 — Features summary", "", f"As of {res['today']} (UTC)", ""]
    st = res["stance"]
    L += ["## Fed Stance Score", "",
          f"**{st.score:+.1f}** (−100 dovish … +100 hawkish) · components used: {len(st.used)}/5" if st.score is not None
          else "**n/a** (no component available)", "",
          "| component | value [−1, +1] | weight | inputs |", "|---|---|---|---|"]
    for k, v in st.components.items():
        L.append(f"| {k} | {'—' if v is None else f'{v:+.2f}'} | {fed_stance.WEIGHTS[k]:.2f} | "
                 f"{json.dumps(st.inputs.get(k), default=str)} |")
    L += ["", "## FedWatch (from ZQ futures)", "", f"Inputs: {json.dumps(res['fedwatch_inputs'])}", ""]
    fw = res["fedwatch"]
    if fw.empty:
        L.append("n/a — missing futures quotes or FOMC dates")
    else:
        L += ["| meeting | P(cut) | P(hold) | P(hike) | implied after | cum. vs today | method |", "|---|---|---|---|---|---|---|"]
        for r in fw.itertuples():
            L.append(f"| {r.meeting} | {r.p_cut:.0%} | {r.p_hold:.0%} | {r.p_hike:.0%} | {r.implied_rate_after:.3f}% | "
                     f"{r.cumulative_vs_today_bp:+.0f}bp | {r.method} |")
    rn = res["regime_now"]
    L += ["", "## Regime (rule-based, point-in-time)", "",
          "_Phases, not NBER dating: \"Recession\" = below-average and deteriorating growth._", "",
          f"**{rn['regime']}** ({regime.REGIMES_FA[rn['regime']]}) at {rn['date']} · probabilities "
          + ", ".join(f"{k} {v:.0%}" for k, v in rn["probabilities"].items()), ""]
    L += [f"- {t}" for t in rn["triggers"]]
    if res.get("regime_hmm"):
        h = res["regime_hmm"].probs.iloc[-1]
        L += ["", f"HMM (comparison): **{h['regime']}** · " + ", ".join(f"{k} {h[k]:.0%}" for k in regime.REGIMES),
              f"_{res['regime_hmm'].note}_"]
    else:
        L += ["", "HMM: not available (hmmlearn not installed or too little data)"]
    if res.get("nber_check"):
        L += ["", f"Sanity vs NBER: {json.dumps(res['nber_check'])}"]
    counts = res["regime_rules"]["regime"].value_counts().to_dict()
    L += ["", f"History {res['regime_rules'].index[0]:%Y-%m} → {res['regime_rules'].index[-1]:%Y-%m}: {counts}", "",
          "### BTC & Gold forward 1-month return by regime", "",
          "| asset | regime | months | mean % | median % | hit rate | ann. vol % |", "|---|---|---|---|---|---|---|"]
    for r in res["regime_perf"].itertuples():
        L.append(f"| {r.asset} | {r.regime} | {r.months} | {r.mean_fwd_1m_pct} | {r.median_fwd_1m_pct} | {r.hit_rate} | {r.ann_vol_pct} |")
    L += ["", "## Yield-curve inversions", ""]
    for k, eps in res["episodes"].items():
        L += [f"### {k} — now: {res['curve_state'][k]}", "",
              "| start | end | days | depth | depth date | re-steepened | note |", "|---|---|---|---|---|---|---|"]
        for e in eps.itertuples():
            L.append(f"| {_d(e.start)} | {_d(e.end) or 'ongoing'} | {e.duration_days} | {e.depth:+.2f} | {_d(e.depth_date)} | "
                     f"{_d(e.resteepen_date) or '—'} | {'(brief dip) ' if e.brief else ''}{e.label if isinstance(e.label, str) else ''} |")
        L.append("")
    if "qe_qt_segments" in res:
        seg = res["qe_qt_segments"]
        L += ["## Balance-sheet regimes (QE/QT)", "", "| start | end | regime | weeks |", "|---|---|---|---|"]
        name = {1: "QE", 0: "neutral", -1: "QT"}
        for s in seg.itertuples():
            L.append(f"| {s.start:%Y-%m-%d} | {s.end:%Y-%m-%d} | {name[s.label]} | {s.weeks} |")
    f = res["features"]
    latest = lambda k: f"{f[k]['value'].iloc[-1]:,.2f} ({f[k]['date'].iloc[-1]:%Y-%m-%d})" if k in f else "n/a"  # noqa: E731
    L += ["", "## Key derived values", "", "| feature | latest |", "|---|---|"]
    for k in ("net_liquidity", "net_liquidity_weekly", "net_liquidity_13w_change", "global_cb_assets_usd", "real_rate_ffr_core_pce",
              "core_cpi_yoy", "core_cpi_3m_ann", "core_pce_yoy", "wage_yoy", "openings_per_unemployed", "spread_30y_2y",
              "gold_silver_ratio", "copper_gold_ratio_z252", "xlf_xlu_ratio_trend", "m2_yoy"):
        L.append(f"| {k} | {latest(k)} |")
    L += ["", f"{len(f)} derived series computed.", ""]
    if stored is not None:
        L += ["## Stored", "", json.dumps(stored), ""]
    return "\n".join(L)


def current_state(res: dict) -> dict:
    """Small machine-readable snapshot for later phases (signals risk section, dashboard, reports)."""
    fw = res["fedwatch"]
    return {"as_of": res["today"], "regime": res["regime_now"],
            "fed_stance": {"score": res["stance"].score, "components": res["stance"].components},
            "fedwatch_next": fw.iloc[0][["meeting", "p_cut", "p_hold", "p_hike"]].to_dict() if not fw.empty else None,
            "curve_state": res["curve_state"]}


def load_cache(d: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(d / "observations.csv.gz", parse_dates=["date"])
    data = {k: g[["date", "value"]].reset_index(drop=True) for k, g in obs.groupby("series_key")}
    vintages = {}
    if (d / "vintages.csv.gz").exists():
        v = pd.read_csv(d / "vintages.csv.gz", parse_dates=["date", "realtime_start", "realtime_end"])
        vintages = {k: g.drop(columns="series_key").reset_index(drop=True) for k, g in v.groupby("series_key")}
    fq = pd.read_csv(d / "futures_quotes.csv", parse_dates=["quote_ts"]) if (d / "futures_quotes.csv").exists() else pd.DataFrame()
    if not fq.empty:
        fq["contract_month"] = pd.to_datetime(fq["contract_month"]).dt.date
    cal = pd.read_csv(d / "calendar_events.csv")
    cal["scheduled_utc"] = pd.to_datetime(cal["scheduled_utc"], utc=True)
    docs = pd.DataFrame()
    if (d / "fed_docs_scored.csv").exists():  # written by jobs.classify (Claude); absent without an API key
        docs = pd.read_csv(d / "fed_docs_scored.csv")
        if not docs.empty:
            docs["published_utc"] = pd.to_datetime(docs["published_utc"], utc=True, errors="coerce", format="mixed")
    return data, vintages, fq, cal, docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", help="read inputs from a jobs.availability --cache-dir snapshot (no database)")
    ap.add_argument("--no-store", action="store_true", help="do not write results to the database")
    args = ap.parse_args(argv)
    today = utcnow().date()
    eng = None
    if args.from_cache:
        data, vintages, fq, cal, docs = load_cache(Path(args.from_cache))
    else:
        eng = engine()
        init_db(eng)
        data = load_all_series(eng=eng)
        if not data:
            print("no observations in the database — run jobs.availability --store first")
            return 1
        reg = load_registry()
        vintages = {k: load_vintages(k, eng) for k in REGIME_KEYS if k in reg and reg[k].vintages}
        fq, cal, docs = (load_table(schema.futures_quotes, eng), load_table(schema.calendar_events, eng),
                         load_table(schema.fed_documents, eng))
    res = compute_all(data, vintages, fq, cal, docs, today)
    stored = None
    if not args.no_store and eng is not None:
        stored = store(res, eng)
    md = render(res, stored)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "features_summary.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "features_state.json").write_text(json.dumps(current_state(res), default=str, indent=1), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n\n" + md)
    print(md)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover — never let a traceback print credentials
        from db.store import redact
        print(redact(traceback.format_exc()), file=sys.stderr)
        sys.exit(1)
