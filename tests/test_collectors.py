"""Offline parser tests against payloads in the sources' real formats."""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from collectors import calendar, crypto, fed, fred, futures, nyfed, rss, treasury, yahoo
from core.http import SourceError
from core.timeutil import TEHRAN, UTC, et_to_utc

FIX = Path(__file__).parent / "fixtures"


def test_fredgraph_csv_drops_missing_and_counts_it():
    df, missing = fred.parse_graph_csv((FIX / "fredgraph_DGS10.csv").read_text(), "DGS10")
    assert missing == 1
    assert list(df["value"]) == [4.12, 4.15, 4.18]
    assert df["date"].iloc[-1] == pd.Timestamp("2026-09-24")


def test_fredgraph_csv_legacy_header():
    df, _ = fred.parse_graph_csv("DATE,UNRATE\n2026-08-01,4.3\n", "UNRATE")
    assert df["value"].iloc[0] == 4.3


def test_fredgraph_csv_wrong_series_raises():
    with pytest.raises(SourceError):
        fred.parse_graph_csv("observation_date,OTHER\n2026-08-01,1\n", "UNRATE")


def test_fred_api_observations():
    df, missing = fred.parse_api_observations(json.loads((FIX / "fred_api_obs.json").read_text()))
    assert missing == 1 and len(df) == 3
    assert df["value"].dtype == "float64"


def test_alfred_value_as_of_has_no_lookahead():
    # CPI for Aug was first published 2026-09-11 as 323.0 and revised on 2026-10-15 to 323.4
    v = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-01", "2026-08-01", "2026-08-01"]),
        "value": [322.5, 323.0, 323.4],
        "realtime_start": pd.to_datetime(["2026-08-12", "2026-09-11", "2026-10-15"]),
        "realtime_end": pd.to_datetime(["NaT", "2026-10-14", "NaT"]),
    })
    before_release = fred.value_as_of(v, "2026-09-10")
    assert list(before_release["date"]) == [pd.Timestamp("2026-07-01")]
    first_print = fred.value_as_of(v, "2026-09-11")
    assert first_print.set_index("date").loc["2026-08-01", "value"] == 323.0
    revised = fred.value_as_of(v, "2026-10-20")
    assert revised.set_index("date").loc["2026-08-01", "value"] == 323.4


def test_treasury_curve_csv():
    curve = treasury.parse_curve_csv((FIX / "treasury_curve.csv").read_text())
    assert curve.loc[curve["Date"] == "2026-09-24", "10 Yr"].iloc[0] == "4.18"


def test_dts_tga_row_selection():
    rows = [
        {"record_date": "2026-09-24", "account_type": "Treasury General Account (TGA) Opening Balance",
         "open_today_bal": "800000", "close_today_bal": "null"},
        {"record_date": "2026-09-24", "account_type": "Treasury General Account (TGA) Closing Balance",
         "open_today_bal": "812345", "close_today_bal": "null"},
        {"record_date": "2021-09-01", "account_type": "Federal Reserve Account",
         "open_today_bal": "1", "close_today_bal": "300000"},
    ]
    df = treasury.parse_dts_tga(rows)
    assert dict(zip(df["date"], df["value"])) == {"2026-09-24": "812345", "2021-09-01": "300000"}


def test_nyfed_parse():
    df = nyfed.parse({"refRates": [{"effectiveDate": "2026-09-24", "type": "EFFR", "percentRate": 4.08}]})
    assert df.iloc[0]["value"] == 4.08
    with pytest.raises(SourceError):
        nyfed.parse({"error": "x"})


def test_yahoo_chart_uses_exchange_local_dates():
    df, meta = yahoo.parse_chart(json.loads((FIX / "yahoo_chart.json").read_text()))
    assert [t.date() for t in df["time"]] == [date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
    assert pd.isna(df["close"].iloc[1])  # a null close stays missing, never filled
    assert meta["regularMarketPrice"] == 3712.4


def test_yahoo_chart_error_raises():
    with pytest.raises(SourceError):
        yahoo.parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_fomc_calendar_parse():
    ms = fed.parse_fomc_calendar((FIX / "fomc_calendar.html").read_text())
    decisions = [m["decision_date"] for m in ms]
    assert decisions == [date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 1), date(2026, 10, 28), date(2026, 12, 9)]
    assert [m["sep"] for m in ms] == [False, True, False, False, True]
    assert ms[2]["start_date"] == date(2026, 4, 30)  # Apr/May meeting spans months
    # 2:00 pm ET on Oct 28 2026 (EDT, UTC−4) = 18:00 UTC
    assert ms[3]["statement_utc"] == datetime(2026, 10, 28, 18, 0, tzinfo=UTC)


def test_ics_parse_unfolds_lines_and_converts_eastern():
    evs = calendar.parse_ics((FIX / "bls.ics").read_text())
    assert evs[1]["summary"] == "Employment Situation for September 2026"
    assert evs[0]["start_utc"] == datetime(2026, 10, 14, 12, 30, tzinfo=UTC)


def test_bls_ics_matches_only_tracked(monkeypatch):
    class R:
        text = (FIX / "bls.ics").read_text()
    monkeypatch.setattr(calendar, "get", lambda *a, **k: R())
    ev = calendar.from_bls_ics(date(2026, 10, 1), date(2026, 10, 31))
    assert sorted(e["release_id"] for e in ev) == ["cpi", "nfp"]  # "Real Earnings" is not tracked
    cpi = next(e for e in ev if e["release_id"] == "cpi")
    assert cpi["scheduled_tehran"].strftime("%H:%M") == "16:00"  # 12:30 UTC = 16:00 Tehran (UTC+3:30)


def test_rss_parse():
    items = rss.parse_feed((FIX / "rss.xml").read_text(), "fed_monetary")
    assert items[0]["title"].startswith("Federal Reserve issues")
    assert items[0]["published_utc"] == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)


def test_crypto_candle_fallback(monkeypatch):
    def broken(*a):
        raise SourceError("binance_vision", "HTTP 451")

    def okx(symbol, interval, limit):
        return crypto._frame([["1790049600000", "100", "110", "90", "105", "3"]])

    monkeypatch.setitem(crypto.CANDLE_PROVIDERS, "binance_vision", broken)
    monkeypatch.setitem(crypto.CANDLE_PROVIDERS, "okx", okx)
    r = crypto.candles("BTC", "1h")
    assert r.venue == "okx" and "binance_vision" in r.errors
    assert r.data["close"].iloc[0] == 105.0


def test_crypto_all_venues_fail(monkeypatch):
    def broken(*a):
        raise SourceError("x", "down")
    for v in list(crypto.CANDLE_PROVIDERS):
        monkeypatch.setitem(crypto.CANDLE_PROVIDERS, v, broken)
    with pytest.raises(SourceError):
        crypto.candles("BTC", "1d")


def test_coinbase_4h_is_resampled_from_1h():
    ts = pd.date_range("2026-09-24", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "open": range(8), "high": range(1, 9), "low": range(8), "close": range(8), "volume": [1.0] * 8})
    out = crypto.resample(df, "4h")
    assert len(out) == 2
    assert out.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [0, 4, 0, 3, 4.0]


def test_aggregate_depth_buckets_across_venues():
    books = {"a": {"bids": [["100050", "1"]], "asks": [["100150", "2"]]},
             "b": {"bids": [["100099", "1"]], "asks": []}}
    d = crypto.aggregate_depth(books, bucket=100)
    bid = d[d.side == "bid"].iloc[0]
    assert bid.bucket == 100000 and bid.qty == 2 and bid.venues == 2


def test_futures_symbols_and_months():
    assert futures.contract_symbol("ZQ", 2026, 12, "CBT") == "ZQZ26.CBT"
    assert futures.upcoming_months(3, date(2026, 11, 5)) == [(2026, 11), (2026, 12), (2027, 1)]
    assert futures.upcoming_months(2, date(2026, 10, 1), quarterly=True) == [(2026, 12), (2027, 3)]


def test_et_to_utc_handles_dst():
    assert et_to_utc(datetime(2026, 1, 14, 8, 30)).hour == 13  # EST
    assert et_to_utc(datetime(2026, 7, 15, 8, 30)).hour == 12  # EDT
    assert et_to_utc(datetime(2026, 7, 15, 8, 30)).astimezone(TEHRAN).strftime("%H:%M") == "16:00"


def test_rss_bytes_with_bom_parse_but_misdecoded_text_would_not():
    raw = b"\xef\xbb\xbf" + (FIX / "rss.xml").read_bytes()
    assert rss.parse_feed(raw, "fed_all")[0]["title"].startswith("Federal Reserve issues")
    # what requests.Response.text produces for text/xml without a charset header:
    with pytest.raises(SourceError):
        rss.parse_feed(raw.decode("iso-8859-1"), "fed_all")


def test_rss_html_block_page_is_named():
    with pytest.raises(SourceError, match="HTML page"):
        rss.parse_feed(b"<!DOCTYPE HTML><html><title>Access Denied</title></html>", "bls")


def test_okx_oi_history_parse():
    df = crypto.parse_okx_oi_history([["1790290800000", "2775900", "27759", "2319000000"],
                                      ["1790287200000", "2770000", "27700", "2300000000"]])
    assert df["ts"].is_monotonic_increasing
    assert df["open_interest"].iloc[-1] == 27759 and df["open_interest_usd"].iloc[-1] == 2.319e9


def _effr_sep_2026_hike():
    days = pd.date_range("2026-08-01", "2026-09-27", freq="D")
    vals = [3.63 if d < pd.Timestamp("2026-09-17") else 3.88 for d in days]
    return pd.DataFrame({"date": days, "value": vals})


def test_realized_month_average_with_mid_month_hike():
    avg, known, n = futures.realized_month_average(_effr_sep_2026_hike(), 2026, 9, date(2026, 9, 28))
    assert (known, n) == (27, 30)
    assert round(avg, 3) == 3.747  # 16 days at 3.63 + 14 days at 3.88


def test_check_current_month_contract():
    effr = _effr_sep_2026_hike()
    ok = pd.DataFrame([{"contract": "ZQU26.CBT", "contract_month": date(2026, 9, 1), "implied_rate": 3.747}])
    assert futures.check_current_month(ok, effr, date(2026, 9, 28))[0] == "ok"
    bad = ok.assign(implied_rate=4.05)  # e.g. a mislabeled (Nov) contract
    assert futures.check_current_month(bad, effr, date(2026, 9, 28))[0] == "fail"


def test_yahoo_chart_keeps_utc_bar_times():
    df, _ = yahoo.parse_chart(json.loads((FIX / "yahoo_chart.json").read_text()))
    assert str(df["ts_utc"].dt.tz) == "UTC" and df["ts_utc"].iloc[0] == pd.Timestamp("2026-09-22 04:00", tz="UTC")


def test_candles_history_paginates_and_drops_forming_bar(monkeypatch):
    calls = []
    t0 = pd.Timestamp("2020-01-01", tz="UTC")

    def fake(source, url, params):
        calls.append(params["startTime"])
        start = pd.Timestamp(params["startTime"], unit="ms", tz="UTC")
        n = 1000 if len(calls) == 1 else 5
        return [[int((start + pd.Timedelta(hours=i)).timestamp() * 1000), "1", "2", "0.5", "1.5", "10"] for i in range(n)]
    monkeypatch.setattr(crypto, "get_json", fake)
    df = crypto.candles_history("BTC", "1h", start=str(t0.date()), end=t0 + pd.Timedelta(hours=2000))
    assert len(calls) == 2 and calls[1] == int((t0 + pd.Timedelta(hours=1000)).timestamp() * 1000)
    assert len(df) == 1005 and df["ts"].is_unique
    gaps = crypto.hourly_gaps(df.drop(index=[10, 11, 12]))
    assert gaps.iloc[0]["bars"] == 3
