"""Phase 2 unit tests on synthetic data with known answers."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.registry import load_registry
from features import curve, derived, fed_stance, fedwatch, pit
from features.transforms import annualized_3m, mom, yoy
from models import regime


def frame(idx, vals):
    return pd.DataFrame({"date": pd.DatetimeIndex(idx), "value": np.asarray(vals, dtype="float64")})


# ───────────────────────────── transforms ─────────────────────────────

def test_inflation_transforms_do_not_bridge_missing_months():
    idx = pd.date_range("2024-01-01", periods=24, freq="MS").delete(21)  # drop Oct 2025-style month
    s = pd.Series(100 * 1.01 ** np.arange(24), index=pd.date_range("2024-01-01", periods=24, freq="MS")).drop(idx[0:0])
    s = s.drop(s.index[21])
    m = mom(s)
    assert np.isnan(m.iloc[21]) and np.isnan(m.iloc[22])  # the missing month and the month after
    assert m.iloc[20] == pytest.approx(1.0)
    assert yoy(s).iloc[-1] == pytest.approx((1.01 ** 12 - 1) * 100)
    assert annualized_3m(s).iloc[-1] == pytest.approx((1.01 ** 12 - 1) * 100)


# ───────────────────────────── liquidity ─────────────────────────────

def test_net_liquidity_units_and_alignment():
    days = pd.bdate_range("2026-09-14", "2026-09-25")
    walcl = pd.Series([6_750_000.0, 6_748_000.0], index=pd.to_datetime(["2026-09-16", "2026-09-23"]))  # USD mn, Wed
    tga = pd.Series(945_000.0, index=days)  # USD mn
    rrp = pd.Series(1.0, index=days)  # USD bn
    nl = derived.net_liquidity(walcl, tga, rrp)
    assert nl.index[0] == pd.Timestamp("2026-09-16")  # nothing before the first WALCL
    assert nl[pd.Timestamp("2026-09-22")] == pytest.approx(6750 - 945 - 1)  # carries 09-16 WALCL
    assert nl[pd.Timestamp("2026-09-23")] == pytest.approx(6748 - 945 - 1)


def test_qe_qt_labels():
    w = pd.date_range("2019-01-02", periods=200, freq="W-WED")
    level = np.r_[np.full(60, 4000.0), 4000 * 1.01 ** np.arange(1, 71), np.full(70, 4000 * 1.01 ** 70) * 0.995 ** np.arange(70)]
    lab = derived.qe_qt_regime(pd.Series(level, index=w))
    assert lab.iloc[40] == 0 and lab.iloc[100] == 1 and lab.iloc[-1] == -1
    seg = derived.regime_segments(lab)
    assert list(seg["label"]) == [0, 1, -1]


# ───────────────────────────── curve ─────────────────────────────

def test_inversion_episodes_merge_filter_and_resteepen():
    d = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    s = pd.Series(0.5, index=d)
    s["2022-03-01":"2022-03-05"] = -0.05  # 5-day blip → ignored
    s["2022-07-01":"2023-06-30"] = -1.0
    s["2023-07-01":"2023-07-10"] = 0.1  # 10-day gap → merged
    s["2023-07-11":"2024-06-30"] = -0.5
    s["2023-02-01"] = -1.2
    eps = curve.inversion_episodes(s)
    assert len(eps) == 1
    e = eps.iloc[0]
    assert e.start == pd.Timestamp("2022-07-01") and e.end == pd.Timestamp("2024-06-30")
    assert e.depth == -1.2 and e.depth_date == pd.Timestamp("2023-02-01")
    assert e.resteepen_date == pd.Timestamp("2024-07-01")
    assert e.label.startswith("Inflation shock")
    assert curve.curve_state(s, eps) == "re-steepening"


def test_ongoing_inversion():
    s = pd.Series([0.2] * 30 + [-0.3] * 40, index=pd.date_range("2026-01-01", periods=70, freq="D"))
    eps = curve.inversion_episodes(s)
    assert bool(eps.iloc[0].ongoing) and eps.iloc[0].end is None
    assert curve.curve_state(s, eps) == "inverted"


# ───────────────────────────── FedWatch ─────────────────────────────

def test_fedwatch_matches_hand_computation():
    # Oct meeting on the 28th (31-day month). Nov has no meeting → R_end from the Nov contract.
    implied = {date(2026, 10, 1): 3.895, date(2026, 11, 1): 4.055, date(2026, 12, 1): 4.205}
    fw = fedwatch.probabilities(implied, [date(2026, 10, 28), date(2026, 12, 9)], effr_now=3.88, target_lower_now=3.75)
    oct_ = fw.iloc[0]
    assert oct_.method == "next-month contract"
    assert oct_.expected_change_bp == pytest.approx(17.5)  # 4.055 − 3.88
    assert oct_.p_hike == pytest.approx(0.70) and oct_.p_hold == pytest.approx(0.30)
    dec = fw.iloc[1]  # Dec: solved from the Dec contract with R_start = 4.055, d=9, N=31
    r_end = (4.205 * 31 - 4.055 * 9) / 22
    assert dec.implied_rate_after == pytest.approx(r_end, abs=1e-4)
    probs = dec.target_range_probs
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(k.count("-") == 1 for k in probs)


def test_move_distribution_cut():
    d = fedwatch.move_distribution(-0.10)
    assert d == pytest.approx({-1: 0.4, 0: 0.6})


# ───────────────────────────── Fed stance ─────────────────────────────

def test_stance_score_weights_and_missing_components():
    today = date(2026, 9, 28)
    upper = pd.Series([3.75, 4.00], index=pd.to_datetime(["2026-06-01", "2026-09-17"]))
    walcl = pd.Series(np.linspace(7000, 6748, 30), index=pd.date_range("2026-03-04", periods=30, freq="W-WED"))
    curve_df = pd.DataFrame({"contract_month": pd.to_datetime(["2027-09-01"]), "implied_rate": [4.835], "contract": ["ZQU27"]})
    sep = frame(["2026-01-01", "2027-01-01"], [3.6, 3.9])
    r = fed_stance.stance_score(today=today, target_upper=upper, effr_now=3.88, target_mid=3.875, curve=curve_df,
                                walcl=walcl, docs=None, sep_latest=sep)
    assert r.components["c_communications"] is None and len(r.used) == 4
    assert r.components["a1_policy_direction"] > 0.9  # hike 11 days ago
    assert r.components["a2_market_path"] == pytest.approx(0.955)
    assert r.components["b_balance_sheet"] > 0  # shrinking → hawkish
    assert r.components["d_dot_plot"] == pytest.approx(0.0333, abs=1e-3)
    w = {k: fed_stance.WEIGHTS[k] for k in r.used}
    expected = 100 * sum(w[k] * r.components[k] for k in w) / sum(w.values())
    assert r.score == pytest.approx(expected, abs=0.05)
    assert -100 <= r.score <= 100


# ───────────────────────────── point in time ─────────────────────────────

def test_pit_first_print_and_asof():
    meta = load_registry()["cpi"]
    df = frame(["2026-07-01", "2026-08-01"], [322.5, 323.4])
    vint = pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-08-01"]), "value": [322.5, 323.0, 323.4],
                         "realtime_start": pd.to_datetime(["2026-08-12", "2026-09-11", "2026-10-15"]),
                         "realtime_end": pd.to_datetime(["NaT", "2026-10-14", "NaT"])})
    known = pit.known_series(df, meta, vint)
    panel = pit.asof_panel({"cpi": known}, pd.to_datetime(["2026-09-10", "2026-09-30", "2026-10-31"]))
    assert panel["cpi"].tolist() == [322.5, 323.0, 323.0]  # first print, never the later revision


def test_pit_release_lag_for_series_without_vintages():
    meta = load_registry()["philly_fed_activity"]  # monthly; lag −10 → treated as 0 after period end
    known = pit.known_series(frame(["2026-08-01", "2026-09-01"], [20.0, 37.8]), meta)
    panel = pit.asof_panel({"p": known}, pd.to_datetime(["2026-09-15", "2026-10-01"]))
    assert panel["p"].tolist() == [20.0, 37.8]


# ───────────────────────────── regime ─────────────────────────────

def _synthetic_panel(n=360):
    idx = pd.date_range("1995-01-31", periods=n, freq="ME")
    t = np.arange(n)
    cyc = np.sin(2 * np.pi * t / 96)  # 8-year cycle
    rng = np.random.default_rng(0)
    noise = lambda s=0.1: rng.normal(0, s, n)  # noqa: E731
    return pd.DataFrame({
        "philly_fed_activity": 20 * cyc + noise(2), "empire_state_activity": 15 * cyc + noise(2),
        "spread_10y_3m": np.roll(cyc, -12) + noise(), "baa_10y_spread": 2 - cyc + noise(),
        "nfci": -0.5 * cyc + noise(0.05), "initial_claims": 300_000 - 50_000 * cyc + noise(1000),
        "nonfarm_payrolls": np.cumsum(150 + 100 * cyc), "m2": 5000 * 1.005 ** t * (1 + 0.02 * np.roll(cyc, 6)),
        "fed_balance_sheet": 1000 * 1.004 ** t, "core_cpi": 150 * 1.002 ** t * (1 + 0.01 * np.roll(cyc, -6)),
    }, index=idx)


def test_regime_rules_probabilities_and_cycle():
    ax = regime.axes(_synthetic_panel())
    probs = regime.rule_probabilities(ax)
    assert np.allclose(probs[list(regime.REGIMES)].sum(axis=1), 1.0)
    assert set(probs["regime"]) == set(regime.REGIMES)  # a full cycle visits every quadrant
    trig = regime.switch_triggers(ax.loc[probs.index[-1]], probs["regime"].iloc[-1])
    assert len(trig) == 2 and all("σ" in t for t in trig)


def test_regime_has_no_lookahead():
    panel = _synthetic_panel()
    full = regime.rule_probabilities(regime.axes(panel))
    cut = regime.rule_probabilities(regime.axes(panel.iloc[:250]))
    common = cut.index
    pd.testing.assert_frame_equal(full.loc[common], cut, check_exact=False, atol=1e-12)


def test_regime_performance_uses_forward_returns():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    labels = pd.Series(["Expansion", "Recession", "Expansion", "Recession"], index=idx)
    px = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    perf = regime.regime_performance(labels, {"BTC": px}).set_index("regime")
    assert perf.loc["Expansion", "months"] == 2 and perf.loc["Expansion", "mean_fwd_1m_pct"] == pytest.approx(5.0)
    assert perf.loc["Recession", "mean_fwd_1m_pct"] == pytest.approx(-10.0)


# ───────────────────────────── features job end-to-end ─────────────────────────────

def test_features_job_end_to_end_on_sqlite(tmp_path):
    from db import schema
    from db.store import engine, init_db, table_counts
    from jobs import features as job

    panel = _synthetic_panel()
    data = {k: frame(panel.index, panel[k]) for k in panel}
    d = pd.date_range("2024-01-01", "2026-09-28", freq="D")
    data.update({
        "fed_funds_eff_daily": frame(d, np.where(d < "2026-09-17", 3.63, 3.88)),
        "fed_target_upper": frame(d, np.where(d < "2026-09-17", 3.75, 4.00)),
        "fed_target_lower": frame(d, np.where(d < "2026-09-17", 3.50, 3.75)),
        "spread_10y_2y": frame(d, np.linspace(-0.5, 0.36, len(d))),
        "spread_10y_3m": frame(d, np.linspace(-1.0, 0.93, len(d))),
        "sep_fed_funds_median": frame(["2026-01-01", "2027-01-01"], [3.6, 3.9]),
        "btc_usd_daily": frame(d, np.linspace(40000, 83500, len(d))),
        "gold_futures": frame(d, np.linspace(2000, 4153, len(d))),
    })
    fq = pd.DataFrame([{"contract": f"ZQ{m}", "root": "ZQ", "contract_month": date(2026, m, 1), "price": 100 - r,
                        "implied_rate": r, "quote_ts": pd.Timestamp("2026-09-28", tz="UTC")}
                       for m, r in ((9, 3.747), (10, 3.895), (11, 4.055), (12, 4.205))])
    cal = pd.DataFrame({"release_id": ["fomc", "fomc"],
                        "scheduled_utc": pd.to_datetime(["2026-10-28 18:00", "2026-12-09 19:00"], utc=True)})
    res = job.compute_all(data, {}, fq, cal, pd.DataFrame(), date(2026, 9, 28))
    assert res["regime_now"]["regime"] in regime.REGIMES
    assert list(res["fedwatch"]["meeting"]) == [date(2026, 10, 28), date(2026, 12, 9)]
    assert res["stance"].score is not None
    eng = engine(f"sqlite:///{tmp_path / 'f.sqlite'}")
    init_db(eng)
    stored = job.store(res, eng)
    counts = table_counts(eng)
    for t in ("features", "fedwatch", "fed_stance", "regime_states", "inversion_episodes"):
        assert counts[t] > 0, t
    md = job.render(res, stored)
    assert "Fed Stance Score" in md and "FedWatch" in md and "Regime" in md
    engine.cache_clear()


def test_hmm_filtered_probabilities():
    pytest.importorskip("hmmlearn")
    h = regime.hmm_regimes(regime.axes(_synthetic_panel()))
    assert h is not None and "in-sample" in h.note
    assert np.allclose(h.probs[list(regime.REGIMES)].sum(axis=1), 1.0)
    assert set(h.probs["regime"]) <= set(regime.REGIMES)
