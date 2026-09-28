"""End-to-end persistence of one availability run (every table branch), on SQLite and — when
TEST_POSTGRES_URL is set — on a real Postgres with the SQL migrations (incl. Supabase RLS)."""

import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from collectors import crypto
from collectors.base import SeriesResult
from core.registry import load_registry
from db import store
from jobs import availability

UTC = timezone.utc


def _stash():
    ts = pd.date_range("2026-09-28 00:00", periods=3, freq="h", tz="UTC")
    candles = pd.DataFrame({"ts": ts, "open": [1.0, 2, 3], "high": [2.0, 3, 4], "low": [0.5, 1, 2],
                            "close": [1.5, 2.5, 3.5], "volume": [10.0, 11, 12]})
    books = {"okx": {"bids": [["100050", "1"]], "asks": [["100150", "2"]]}}
    return {
        ("BTC", "1h"): crypto.VenueResult("okx", candles, {}),
        "book_BTC": (books, crypto.aggregate_depth(books, 100.0)),  # contains numpy int64 (venues)
        "deriv": {"okx": {"open_interest": 27759.0, "open_interest_usd": 2.32e9, "funding_rate": 2.5e-5,
                          "mark_price": 83557.9, "next_funding_ts": pd.Timestamp("2026-09-29", tz="UTC")}},
        "oi_hist": pd.DataFrame({"ts": ts, "open_interest": np.array([1.0, 2, 3]), "open_interest_usd": [1e9, 2e9, 3e9]}),
        "liq_okx": pd.DataFrame([{"venue": "okx", "asset": "BTC", "ts": ts[0], "side": "long", "price": 83000.0,
                                  "qty_contracts": 5.0, "raw_id": "okx-1"}]),
        "liq_bybit": [{"venue": "bybit", "asset": "BTC", "ts": ts[1], "side": "short", "price": 84000.0, "qty": 0.1,
                       "raw_id": "bybit-1"}],
        "fut_ZQ": pd.DataFrame([{"contract": "ZQU26.CBT", "root": "ZQ", "contract_month": date(2026, 9, 1),
                                 "price": 96.253, "implied_rate": 3.747, "quote_ts": ts[2]}]),
        "fomc": [{"decision_date": date(2026, 10, 28), "sep": False}],
        "calendar_fred": [],
        "news": [{"id": "abc", "source": "fed_monetary", "published_utc": datetime(2026, 9, 16, 18, tzinfo=UTC),
                  "title": "FOMC statement", "url": "https://x", "summary": ""}],
        "_vintages": {"cpi": pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-01"]), "value": [323.0, 323.4],
                                           "realtime_start": pd.to_datetime(["2026-09-11", "2026-10-15"]),
                                           "realtime_end": pd.to_datetime(["2026-10-14", pd.NaT])})},
    }


def _run(url, monkeypatch):
    store.engine.cache_clear()
    monkeypatch.setattr(store, "engine", lambda u=None, _e=store.engine(url): _e)
    reg = {k: v for k, v in load_registry().items() if k in ("ust_10y", "cpi")}
    results = {
        "ust_10y": SeriesResult("ust_10y", pd.DataFrame({"date": pd.to_datetime(["2026-09-24"]), "value": [5.18]}), "u"),
        "cpi": SeriesResult("cpi", pd.DataFrame({"date": pd.to_datetime(["2026-08-01"]), "value": [334.1]}), "u"),
    }
    series = {k: availability.SeriesReport(key=k, name_en="", category="", source="", source_id="", url="", units="",
                                           frequency="D", release_lag_days=0, proxy=False, proxy_for=None,
                                           status="ok", last_date="2026-09-24") for k in reg}
    rep = availability.store_all(reg, series, results, _stash(), datetime(2026, 9, 28, tzinfo=UTC))
    assert rep["error"] is None, rep["error"]
    assert all(c.startswith("✅") for c in rep["readback"])
    c = rep["counts"]
    for t in ("observations", "observation_vintages", "series_meta", "candles", "orderbook_snapshots",
              "derivatives_snapshots", "liquidations", "futures_quotes", "calendar_events", "news_items", "fed_documents"):
        assert c[t] > 0, t
    assert c["derivatives_snapshots"] == 4  # 1 snapshot + 3 hourly OI points
    # idempotent: a second run adds nothing new
    rep2 = availability.store_all(reg, series, results, _stash(), datetime(2026, 9, 28, tzinfo=UTC))
    assert rep2["counts"]["observations"] == c["observations"] and rep2["counts"]["candles"] == c["candles"]
    return rep


def test_store_all_sqlite(tmp_path, monkeypatch):
    _run(f"sqlite:///{tmp_path / 'db.sqlite'}", monkeypatch)


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_store_all_postgres_with_migrations_and_rls(monkeypatch):
    rep = _run(os.environ["TEST_POSTGRES_URL"], monkeypatch)
    eng = store.engine()
    with eng.connect() as conn:
        rls = conn.exec_driver_sql("SELECT count(*) FROM pg_tables WHERE schemaname='public' AND NOT rowsecurity "
                                   "AND tablename <> 'schema_migrations'").scalar_one()
        applied = [r[0] for r in conn.exec_driver_sql("SELECT filename FROM schema_migrations ORDER BY 1")]
    assert applied[:2] == ["001_init.sql", "002_security.sql"]
    assert rls == 0, "every data table must have row level security enabled"
    assert rep["target"].startswith("postgresql @ ")
