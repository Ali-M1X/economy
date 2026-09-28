"""Phase 3 tests: surprises from vintages, event returns, estimators, normalization, macro score."""

import numpy as np
import pandas as pd
import pytest

from models import events, impact


def _vintages(n_months=60, start="2019-01-01", revise=True, seed=1):
    """Monthly index growing ~0.2%/month; each month is first published on the 12th of the next month;
    optionally the previous month is revised at the same time (as CPI/NFP often are)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    mom = 0.2 + rng.normal(0, 0.1, n_months)
    level = 100 * np.cumprod(1 + mom / 100)
    rows = []
    bulk = pd.Timestamp("2018-12-31")
    rows += [{"date": pd.Timestamp("2018-12-01"), "value": 99.8, "realtime_start": bulk, "realtime_end": pd.NaT}]
    for i, d in enumerate(dates):
        rel = d + pd.DateOffset(months=1, days=11)
        nxt = dates[i + 1] + pd.DateOffset(months=1, days=11) if i + 1 < n_months else pd.NaT
        rows.append({"date": d, "value": level[i], "realtime_start": rel,
                     "realtime_end": (nxt - pd.Timedelta(days=1)) if revise and pd.notna(nxt) else pd.NaT})
        if revise and pd.notna(nxt):  # revised print, known from the next release on
            rows.append({"date": d, "value": level[i] * 1.0005, "realtime_start": nxt, "realtime_end": pd.NaT})
    return pd.DataFrame(rows), mom, dates


def test_surprises_use_first_prints_and_past_std_only():
    v, mom, dates = _vintages()
    ev = events.surprises(v, "cpi", "mom_pct", "mean3", "cpi")
    assert (ev["n_new_obs"] == 1).all()
    assert pd.Timestamp("2018-12-31") not in set(ev["release_date"])  # bulk load skipped
    # the actual for month t equals the first-print MoM (t vs t-1 as revised in that same vintage)
    r = ev[ev["obs_date"] == dates[30]].iloc[0]
    assert r["release_utc"] == pd.Timestamp(f"{(dates[30] + pd.DateOffset(months=1, days=11)).date()} 12:30", tz="UTC") \
        or r["release_utc"].hour in (12, 13)  # 08:30 ET = 12:30 UTC (EDT) / 13:30 UTC (EST)
    assert np.isfinite(r["z"])
    # z uses only earlier surprises: changing a later surprise must not change an earlier z
    z_before = ev.set_index("obs_date")["z"].iloc[20]
    v2 = v.copy()
    v2.loc[v2["date"] == dates[-1], "value"] *= 1.05
    ev2 = events.surprises(v2, "cpi", "mom_pct", "mean3", "cpi")
    assert ev2.set_index("obs_date")["z"].iloc[20] == pytest.approx(z_before)


def test_release_time_is_dst_aware():
    assert events.release_time_utc("cpi", pd.Timestamp("2026-01-13")).hour == 13  # EST
    assert events.release_time_utc("cpi", pd.Timestamp("2026-07-14")).strftime("%H:%M") == "12:30"  # EDT
    assert events.release_time_utc("jolts", pd.Timestamp("2026-07-07")).strftime("%H:%M") == "14:00"


def test_event_returns_use_last_trade_before_release():
    ts = pd.date_range("2026-07-14 11:00", periods=30, freq="15min", tz="UTC")  # 7.5h of bars
    close = np.full(len(ts), 100.0)
    close[ts >= pd.Timestamp("2026-07-14 12:30", tz="UTC")] = 101.0  # reaction happens in the release bar
    bars = pd.DataFrame({"ts": ts, "open": close, "close": close})
    ev = pd.DataFrame({"release_utc": [pd.Timestamp("2026-07-14 12:30", tz="UTC")], "z": [1.0]})
    out = events.event_returns(ev, bars)
    assert out["ret_1h"].iloc[0] == pytest.approx(np.log(1.01) * 100)
    assert out["ret_24h"].isna().iloc[0]  # beyond the bars: dropped, not filled


def test_ols_nw_recovers_slope_and_widens_se_with_overlap():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    y = 0.5 * x + rng.normal(size=2000)
    b, t, n = impact.ols_nw(y, x, lag=0)
    assert b == pytest.approx(0.5, abs=0.05) and n == 2000 and t > 15
    # overlapping sums inflate naive t; NW corrects
    yo = pd.Series(y).rolling(8).sum().to_numpy()
    xo = pd.Series(x).rolling(8).sum().to_numpy()
    _, t0, _ = impact.ols_nw(yo, xo, lag=0)
    _, t8, _ = impact.ols_nw(yo, xo, lag=8)
    assert t8 < t0


def test_normalization_rules():
    e = impact.normalize(impact.Estimate("x", "BTC", "4w", "medium", "full", beta=2.0, t_stat=3.0, n=300,
                                         n_eff=75, sd_ret=4.0, hit_rate=None))
    assert e.coefficient == pytest.approx(10.0) and e.confidence == "high"
    e = impact.normalize(impact.Estimate("x", "BTC", "4w", "medium", "full", beta=-0.5, t_stat=-1.0, n=40,
                                         n_eff=10, sd_ret=4.0, hit_rate=None))
    assert e.coefficient == pytest.approx(-10 * 0.25 * 0.5 * (10 / 30), abs=0.01) and e.confidence == "low"


def test_event_study_detects_planted_reaction():
    rng = np.random.default_rng(3)
    z = rng.normal(size=120)
    ev = pd.DataFrame({"z": z, "ret_1h": -0.4 * z + rng.normal(0, 0.3, 120)})  # hot print → asset down
    est = impact.event_study(ev, "cpi", "BTC", horizons=("1h",))[0]
    assert est.beta == pytest.approx(-0.4, abs=0.08) and est.coefficient < -8 and est.hit_rate > 0.7


def test_projection_on_planted_relation():
    idx = pd.date_range("2005-01-07", periods=900, freq="W-FRI")
    rng = np.random.default_rng(5)
    ind = pd.Series(np.cumsum(rng.normal(size=900)), index=idx)
    # build prices whose forward 4-week log return (%) = −0.2 × indicator's past 4-week change + noise
    fwd = np.nan_to_num(-0.2 * ind.diff(4).to_numpy()) + rng.normal(0, 0.5, 900)
    logpx = np.zeros(900)
    for i in range(900 - 4):
        logpx[i + 4] = logpx[i] + fwd[i] / 100
    px = pd.Series(100 * np.exp(logpx), index=idx)
    e = impact.projection(ind, px, "diff", 4, 4, "x", "Gold", "4w", "medium")
    assert e.beta < 0 and abs(e.t_stat) > 3 and e.coefficient < 0 and e.n > 800


def test_macro_score_breakdown():
    from jobs.impact import macro_score
    coef = pd.Series({"dxy": -8.0, "net_liquidity_weekly": 6.0, "vix": 0.0})
    state = pd.Series({"dxy": 0.5, "net_liquidity_weekly": -0.5, "vix": 1.0})
    sc, br = macro_score(coef, state)
    assert sc == pytest.approx(100 * (-8 * 0.5 + 6 * -0.5) / 30, abs=0.1)  # Σ|coef| = 14 < floor 30
    assert list(br.index)[0] == "dxy" and "vix" not in br.index
    strong = pd.Series({f"i{k}": 10.0 for k in range(5)})
    sc2, _ = macro_score(strong, pd.Series({f"i{k}": 1.0 for k in range(5)}))
    assert sc2 == pytest.approx(100.0)  # strong, aligned evidence reaches +100
    weak, _ = macro_score(pd.Series({"a": 0.6, "b": 0.4}), pd.Series({"a": 1.0, "b": 1.0}))
    assert abs(weak) < 5  # uniformly weak evidence stays near zero


def test_impact_job_end_to_end_on_synthetic_snapshot(tmp_path, monkeypatch):
    import yaml
    from core.settings import ROOT
    from jobs import impact as job
    from jobs.features import REGIME_KEYS

    rng = np.random.default_rng(11)
    cfg = yaml.safe_load((ROOT / "config" / "indicators.yaml").read_text())
    days = pd.bdate_range("1995-01-02", "2026-09-25")
    months = pd.date_range("1995-01-01", "2026-08-01", freq="MS")
    weeks = pd.date_range("1995-01-07", "2026-09-19", freq="W-SAT")
    rows = []

    def add(key, idx, vals):
        rows.append(pd.DataFrame({"date": idx, "value": vals, "series_key": key}))
    for ind in cfg["state"]:
        if ind.get("derived"):
            continue
        add(ind["key"], days, 100 + np.cumsum(rng.normal(0, 0.5, len(days))))
    for k in ("tga_weekly", "reverse_repo"):
        add(k, days, np.abs(100 + np.cumsum(rng.normal(0, 1, len(days)))))
    for k in REGIME_KEYS:
        if k not in {i["key"] for i in cfg["state"]}:
            add(k, months, 100 + np.cumsum(rng.normal(0, 1, len(months))))
    add("btc_usd_daily", days, 10000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days)))))
    add("gold_futures", days, 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))))
    pd.concat(rows).to_csv(tmp_path / "observations.csv.gz", index=False)

    v, _, _ = _vintages(n_months=80, start="2019-01-01")
    v.assign(series_key="cpi").to_csv(tmp_path / "vintages.csv.gz", index=False)
    ts = pd.date_range("2019-01-01", "2026-09-28", freq="15min", tz="UTC")
    for a in ("BTC", "PAXG"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(ts))))
        pd.DataFrame({"ts": ts, "open": c, "high": c, "low": c, "close": c, "volume": 1.0}) \
            .to_csv(tmp_path / f"candles_{a}_15m.csv.gz", index=False)
    pd.DataFrame(columns=["event_id", "release_id", "scheduled_utc"]).to_csv(tmp_path / "calendar_events.csv", index=False)

    monkeypatch.setattr(job, "OUT_DIR", tmp_path / "out")
    assert job.main(["--from-cache", str(tmp_path)]) == 0
    coefs = pd.read_csv(tmp_path / "out" / "impact_coefficients.csv")
    assert {"short", "medium", "long"} <= set(coefs["bucket"])
    assert {"full", "recent"} <= set(coefs["sample"]) and coefs["sample"].str.startswith("regime:").any()
    assert coefs["coefficient"].between(-10, 10).all()
    short = coefs[(coefs["bucket"] == "short") & (coefs["indicator"] == "cpi")]
    assert set(short["asset"]) == {"BTC", "Gold"} and (short[short["sample"] == "full"]["n"] > 30).all()
    # null test: every series here is random noise, so large coefficients must be rare
    big = (coefs["coefficient"].abs() >= 5).mean()
    assert big < 0.05, f"{big:.1%} of coefficients ≥ 5 on pure noise"
    md = (tmp_path / "out" / "impact_summary.md").read_text()
    from db.store import engine, table_counts
    from jobs.impact import store
    coefs2 = job.flag_conflicts(coefs, job.indicator_config()) if "conflict" not in coefs else coefs
    eng = engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    import json as _json
    n = store(coefs2, _json.loads((tmp_path / "out" / "macro_scores.json").read_text()), pd.Timestamp("2026-09-28", tz="UTC"), eng)
    assert n["impact_coefficients"] == len(coefs2) and n["macro_scores"] == 6
    assert table_counts(eng)["impact_coefficients"] == len(coefs2)
    engine.cache_clear()
    assert "Macro Score" in md and "Release events used" in md


def test_theory_conflicts_are_flagged_and_half_weighted():
    from jobs.impact import flag_conflicts, headline
    cfg = {"releases": [], "state": [{"key": "real_yield_10y", "theory": {"BTC": -1, "Gold": -1}},
                                     {"key": "spread_10y_2y", "theory": {"BTC": 0, "Gold": 0}}]}
    rows = [{"indicator": "real_yield_10y", "asset": "BTC", "bucket": "medium", "horizon": "4w", "sample": s_, "coefficient": c}
            for s_, c in (("full", 0.0), ("recent", 6.9))]
    rows.append({"indicator": "spread_10y_2y", "asset": "BTC", "bucket": "medium", "horizon": "4w", "sample": "full", "coefficient": 3.0})
    c = flag_conflicts(pd.DataFrame(rows), cfg)
    assert c.loc[1, "conflict"] and not c.loc[0, "conflict"] and not c.loc[2, "conflict"]
    h = headline(c, "BTC", "medium")
    assert h["real_yield_10y"] == pytest.approx((0 + 6.9 * 0.5) / 2)


def test_robust_surprise_scale_survives_an_outlier():
    v, _, dates = _vintages(n_months=80)
    ev = events.surprises(v, "cpi", "mom_pct", "mean3", "cpi")
    base = ev["z"].abs().median()
    v2 = v.copy()
    v2.loc[v2["date"] >= dates[30], "value"] *= 0.8  # a COVID-style −20% level shock at month 30
    ev2 = events.surprises(v2, "cpi", "mom_pct", "mean3", "cpi")
    later = ev2[ev2["obs_date"] > dates[40]]["z"].abs().median()
    assert later == pytest.approx(base, rel=0.5)  # later z-scores are not crushed toward 0
