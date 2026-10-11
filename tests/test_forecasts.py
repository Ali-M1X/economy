"""Forecasts for upcoming releases: source parsers (saved real payload samples), release matching, probability maths."""

from __future__ import annotations

import pytest

from models import forecast as fc


# ── probability maths ────────────────────────────────────────────────────

def test_normal_odds_sum_to_one_and_follow_the_nowcast():
    o = fc.normal_odds(center=0.4, forecast=0.3, sigma=0.1, method="nowcast")
    assert abs(o.hot + o.inline + o.cool - 1) < 1e-9
    assert o.hot > o.cool and o.edge
    # P(actual > consensus) is Φ((nowcast − consensus) / σ): the hot and in-line-above parts together
    assert abs((1 - fc.phi((0.3 - 0.4) / 0.1)) - fc.phi(1.0)) < 1e-12


def test_even_odds_have_no_edge():
    o = fc.even_odds()
    assert abs(o.hot - o.cool) < 1e-12 and not o.edge and o.method == "even"
    assert abs(o.hot - (1 - fc.phi(fc.IN_LINE_Z))) < 1e-12


def test_market_odds_use_the_step_survival_curve():
    # Kalshi "CPI above X%" prices (single-decimal results): P(>0.2)=0.80, P(>0.3)=0.35, P(>0.4)=0.10
    pts = [(0.1, 0.95), (0.2, 0.80), (0.3, 0.35), (0.4, 0.10)]
    o = fc.market_odds(pts, forecast=0.3, sigma=0.1)
    assert o.method == "market"
    assert abs(o.hot - 0.35) < 1e-9          # actual ≥ 0.4  ⇔  above 0.3
    assert abs(o.cool - 0.20) < 1e-9         # actual ≤ 0.2  ⇔  not above 0.2
    assert abs(o.inline - 0.45) < 1e-9       # exactly 0.3


def test_market_prices_are_made_monotone():
    pts = fc.clean_survival([(0.1, 0.9), (0.2, 0.95), (0.3, 0.4)])
    assert [p for _, p in pts] == [0.9, 0.9, 0.4]


def test_event_odds_prefers_market_then_nowcast_then_even():
    pts = [(0.2, 0.8), (0.3, 0.3)]
    assert fc.event_odds(0.3, 0.1, nowcast=0.5, market=pts).method == "market"
    assert fc.event_odds(0.3, 0.1, nowcast=0.5).method == "nowcast"
    assert fc.event_odds(0.3, 0.1).method == "even"
    assert fc.event_odds(None, 0.1) is None and fc.event_odds(0.3, None) is None


def test_surprise_sigma_switches_to_consensus_history_when_long_enough():
    assert fc.surprise_sigma([0.1, -0.1], fallback=0.2) == 0.2
    errs = [0.1, -0.1] * 6
    assert fc.surprise_sigma(errs, fallback=0.2) == pytest.approx(0.1044, abs=1e-3)
    assert fc.surprise_sigma([], fallback=None) is None


# ── source parsers (saved samples of the real feeds, October 2026) ───────

import json  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from collectors import forecasts as col  # noqa: E402
from jobs import forecasts as fjob  # noqa: E402
from models import events as ev_model  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "forecasts"
NOW = pd.Timestamp("2026-10-11 05:30", tz="UTC")


def _json(name):
    return json.loads((FX / name).read_text(encoding="utf-8"))


def test_parse_amount_units():
    assert col.parse_amount("0.3%") == 0.3 and col.parse_amount("195K") == 195000
    assert col.parse_amount("7.10M") == 7.1e6 and col.parse_amount("-25K") == -25000
    assert col.parse_amount("26.5") == 26.5 and col.parse_amount("") is None and col.parse_amount(None) is None
    assert col.to_unit(150000, "thousands") == 150 and col.to_unit(195000, "persons") == 195000


def test_forexfactory_usd_rows_map_to_releases():
    rows = col.parse_forexfactory(_json("ff_thisweek.json"))
    got = {(r["release_id"], r["key"]): r for r in rows}
    assert set(got) == {("cpi", "cpi"), ("cpi", "core_cpi"), ("ppi", "ppi_final_demand"), ("claims", "initial_claims"),
                        ("philly", "philly_fed_activity"), ("empire", "empire_state_activity")}  # no EUR, y/y, holiday
    cpi = got[("cpi", "cpi")]
    assert cpi["consensus"] == 0.6 and cpi["previous"] == 0.4 and cpi["et_date"] == date(2026, 10, 14)
    assert cpi["time_utc"] == pd.Timestamp("2026-10-14 12:30", tz="UTC")
    assert got[("cpi", "core_cpi")]["consensus"] == 0.2
    assert got[("claims", "initial_claims")]["consensus"] == 195000  # claims are counted in persons
    assert got[("philly", "philly_fed_activity")]["consensus"] == 26.5


def test_cleveland_latest_value_per_target_month():
    nc = col.parse_cleveland(_json("cleveland_nowcast_month.json"))
    assert nc[("cpi", "2026-09")] == {"value": pytest.approx(0.530663, abs=1e-6), "as_of": "2026-10-09"}
    assert nc[("cpi", "2026-10")]["value"] == pytest.approx(0.269347, abs=1e-6)  # trailing blank skipped
    assert nc[("core_pce", "2026-09")]["value"] == pytest.approx(0.247277, abs=1e-6)
    assert not any(k.startswith("Actual") for k, _ in nc)


def test_gdpnow_latest_quarter():
    assert col.parse_gdpnow((FX / "fred_GDPNOW.csv").read_text()) == {"quarter_start": date(2026, 7, 1), "value": 3.5862}
    assert col.parse_gdpnow("observation_date,GDPNOW\n") is None


def test_kalshi_above_contracts_become_survival_points():
    cpi = col.parse_kalshi("KXCPI", _json("kalshi_KXCPI.json"))[0]
    assert cpi["key"] == "cpi" and cpi["ticker"] == "KXCPI-26SEP" and cpi["et_date"] == date(2026, 10, 14)
    pts = dict(cpi["points"])
    assert len(pts) == 9 and pts[0.5] == pytest.approx(0.605) and pts[0.6] == pytest.approx(0.14)  # bid/ask mid
    pay = col.parse_kalshi("KXPAYROLLS", _json("kalshi_KXPAYROLLS.json"))[0]
    assert [s for s, _ in pay["points"]][:3] == [-25, 0, 20]  # jobs → thousands, like our payroll change
    assert col.parse_kalshi("UNKNOWN", _json("kalshi_KXCPI.json")) == []


def test_market_odds_on_the_real_cpi_brackets():
    pts = col.parse_kalshi("KXCPI", _json("kalshi_KXCPI.json"))[0]["points"]
    o = fc.market_odds(pts, forecast=0.6, sigma=0.3)  # a wide σ still means "exactly 0.6" on a 0.1 grid
    assert (round(o.hot, 3), round(o.inline, 3), round(o.cool, 3)) == (0.14, 0.465, 0.395)
    assert o.edge and o.cool > o.hot


def _calendar():
    rows = [("cpi-2026-10-14", "cpi", "2026-10-14 12:30"), ("ppi-2026-10-15", "ppi", "2026-10-15 12:30"),
            ("claims-2026-10-15", "claims", "2026-10-15 12:30"), ("philly-2026-10-15", "philly", "2026-10-15 12:30"),
            ("gdp-2026-10-29", "gdp", "2026-10-29 12:30"), ("nfp-2026-11-06", "nfp", "2026-11-06 13:30"),
            ("cpi-2026-09-11", "cpi", "2026-09-11 12:30")]
    c = pd.DataFrame(rows, columns=["event_id", "release_id", "scheduled_utc"])
    c["scheduled_utc"] = pd.to_datetime(c["scheduled_utc"], utc=True)
    return c


def test_release_matching_by_us_date():
    c = _calendar()
    assert col.match_event(c, "cpi", date(2026, 10, 14)) == "cpi-2026-10-14"
    assert col.match_event(c, "cpi", date(2026, 10, 15)) == "cpi-2026-10-14"  # within a day
    assert col.match_event(c, "cpi", date(2026, 10, 20)) is None
    assert col.match_event(c, "ppi", date(2026, 10, 14)) == "ppi-2026-10-15"
    assert col.prev_month(date(2026, 10, 14)) == "2026-09" and col.prev_month(date(2026, 1, 5)) == "2025-12"


def test_assemble_joins_every_source_to_the_calendar():
    ff = col.parse_forexfactory(_json("ff_thisweek.json"))
    cl = col.parse_cleveland(_json("cleveland_nowcast_month.json"))
    gd = col.parse_gdpnow((FX / "fred_GDPNOW.csv").read_text())
    ks = (col.parse_kalshi("KXCPI", _json("kalshi_KXCPI.json")) +
          col.parse_kalshi("KXPAYROLLS", _json("kalshi_KXPAYROLLS.json")))
    t = col.assemble(_calendar(), ff, cl, gd, ks, NOW, days=30).set_index(["event_id", "key"])
    cpi = t.loc[("cpi-2026-10-14", "cpi")]
    assert cpi["consensus"] == 0.6 and cpi["consensus_source"] == "forexfactory"
    assert cpi["nowcast"] == pytest.approx(0.5307, abs=1e-4) and cpi["nowcast_source"] == "cleveland"  # September CPI
    assert cpi["market_source"] == "kalshi" and len(json.loads(cpi["market_points"])) == 9
    assert t.loc[("cpi-2026-10-14", "core_cpi")]["nowcast"] == pytest.approx(0.2017, abs=1e-4)
    assert t.loc[("claims-2026-10-15", "initial_claims")]["consensus"] == 195000
    assert t.loc[("gdp-2026-10-29", "gdp")]["nowcast"] == 3.5862  # Q3 GDPNow → the Q3 advance estimate
    assert t.loc[("nfp-2026-11-06", "nonfarm_payrolls")]["market_source"] == "kalshi"
    assert "cpi-2026-09-11" not in t.index.get_level_values(0)  # past releases are not forecast


def test_collect_survives_every_source_failing(monkeypatch):
    from core.http import SourceError

    def boom(source, *a, **k):
        raise SourceError(source, "HTTP 403")
    monkeypatch.setattr(col, "get", boom)
    monkeypatch.setattr(col, "KALSHI_PAUSE_S", 0)
    t, status = col.collect(_calendar(), NOW)
    assert t.empty and status["cleveland"].startswith("failed") and status["forexfactory"]["thisweek"].startswith("failed")
    assert all(v.startswith("failed") for v in status["kalshi"].values())


def test_history_keeps_the_last_capture_before_each_release():
    def table(cons, when):
        return pd.DataFrame([{"event_id": "cpi-2026-10-14", "release_id": "cpi", "key": "cpi", "scheduled_utc": when,
                              "consensus": cons, "previous": 0.4, "consensus_source": "forexfactory"}])
    rel = "2026-10-14T12:30:00+00:00"
    h = fjob.update_history(pd.DataFrame(), table(0.5, rel), NOW)
    h = fjob.update_history(h, table(0.6, rel), pd.Timestamp("2026-10-14 11:00", tz="UTC"))  # newer, still before
    assert len(h) == 1 and h.iloc[0]["consensus"] == 0.6
    h = fjob.update_history(h, table(0.9, rel), pd.Timestamp("2026-10-14 13:00", tz="UTC"))  # after: frozen
    assert len(h) == 1 and h.iloc[0]["consensus"] == 0.6


def test_surprises_are_measured_against_the_stored_consensus():
    evs = pd.DataFrame({"indicator": ["cpi"] * 2, "release_utc": pd.to_datetime(["2026-09-11 12:30", "2026-10-14 12:30"], utc=True),
                        "obs_date": pd.to_datetime(["2026-08-01", "2026-09-01"]), "actual": [0.40, 0.70],
                        "expected": [0.05, 0.20], "surprise": [0.35, 0.50], "z": [1.4, 2.0],
                        "surprise_basis": ["vs mean3 (no free consensus)"] * 2})
    hist = pd.DataFrame({"key": ["cpi"], "release_utc": ["2026-10-14T12:30:00+00:00"], "consensus": [0.6]})
    out = ev_model.apply_consensus(evs, hist)
    assert list(out["surprise_basis"]) == ["vs mean3 (no free consensus)", "vs consensus"]
    last = out.iloc[1]
    assert last["expected"] == 0.6 and last["surprise"] == pytest.approx(0.1)
    assert last["z"] == pytest.approx(0.1 / 0.25)  # the indicator's own surprise scale (0.50 / 2.0)
    assert ev_model.apply_consensus(evs, None) is evs
