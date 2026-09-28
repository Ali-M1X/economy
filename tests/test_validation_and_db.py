from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.registry import SeriesMeta, load_registry
from db import schema
from db.store import engine, init_db, load_series, save_observations, save_series_meta, upsert
from validation import checks

ROOT = Path(__file__).resolve().parent.parent


def meta(**kw) -> SeriesMeta:
    base = dict(key="x", name_en="x", name_fa="x", category="rates", source="fred", source_id="X", url="u",
                units="%", frequency="D", release_lag_days=1, max_stale_days=7, valid_min=0, valid_max=10,
                start="2020-01-01")
    return SeriesMeta(**{**base, **kw})


def frame(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": np.array(values, dtype="float64")})


# ───────────────────────────── registry ─────────────────────────────

def test_registry_is_complete_and_consistent():
    reg = load_registry()
    assert len(reg) > 60
    for m in reg.values():
        assert m.url.startswith("https://"), m.key
        assert m.name_fa.strip(), m.key
        assert m.max_stale_days > 0, m.key
        assert isinstance(m.valid_min, float) and isinstance(m.valid_max, float), m.key
        if m.proxy:
            assert m.proxy_for, m.key


def test_registry_rejects_non_numeric_bounds(tmp_path):
    from core.registry import RegistryError
    p = tmp_path / "s.yaml"
    p.write_text('series:\n  - {key: a, name_en: a, name_fa: a, category: rates, source: fred, source_id: A, url: "https://x", '
                 'units: "%", frequency: D, release_lag_days: 1, max_stale_days: 7, valid_min: low, valid_max: 5, start: "2020-01-01"}\n')
    with pytest.raises(RegistryError):
        load_registry(p)


def test_registry_covers_the_spec():
    ids = {m.source_id for m in load_registry().values()}
    required = {"DFF", "FEDFUNDS", "DFEDTARU", "DFEDTARL", "DFII10", "CPIAUCSL", "CPILFESL", "PPIFIS", "PPIACO",
                "PCEPI", "PCEPILFE", "M2SL", "WALCL", "RRPONTSYD", "NFCI", "ANFCI", "UNRATE", "CES0500000003",
                "JTSJOL", "UNEMPLOY", "PAYEMS", "ICSA", "DGS2", "DGS10", "DGS30", "DGS3MO", "T10Y2Y", "T10Y3M",
                "T10YIE", "BAMLH0A0HYM2", "BAMLC0A0CM", "DX-Y.NYB", "DTWEXBGS", "HG=F", "GC=F", "^GSPC", "XLF",
                "XLU", "^NDX", "CL=F", "BZ=F", "SI=F", "NG=F", "DBC", "ZQ=F", "SR3=F"}
    assert required <= ids, required - ids


# ───────────────────────────── checks ─────────────────────────────

def test_schema_rejects_duplicates_and_unsorted():
    assert checks.check_schema(frame(["2026-01-02", "2026-01-02"], [1, 2])).status == "fail"
    assert checks.check_schema(frame(["2026-01-03", "2026-01-02"], [1, 2])).status == "fail"
    assert checks.check_schema(frame(["2026-01-02", "2026-01-03"], [1, 2])).ok


def test_range_check():
    r = checks.check_range(frame(["2026-01-02", "2026-01-05"], [5, 50]), meta())
    assert r.status == "fail" and "2026-01-05=50" in r.message


def test_staleness():
    m = meta(max_stale_days=7)
    df = frame(["2026-09-01"], [1])
    assert checks.check_staleness(df, m, date(2026, 9, 5)).ok
    assert checks.check_staleness(df, m, date(2026, 9, 20)).status == "fail"


def test_weekend_dates_catch_timezone_shift():
    sundays = pd.date_range("2025-01-05", periods=60, freq="W-SUN")
    assert checks.check_weekend_dates(frame(sundays, [1] * 60), meta()).status == "fail"
    bdays = pd.bdate_range("2025-01-01", periods=300)
    assert checks.check_weekend_dates(frame(bdays, [1] * 300), meta()).ok
    assert checks.check_weekend_dates(frame(sundays, [1] * 60), meta(category="crypto")).ok


def test_gap_check_warns():
    df = frame(["2026-01-02", "2026-01-05", "2026-03-01"], [1, 1, 1])
    assert checks.check_gaps(df, meta()).status == "warn"


def test_cross_check_abs_and_rel():
    d = pd.bdate_range("2026-06-01", periods=30)
    a = frame(d, [4.10] * 30)
    assert checks.check_cross(a, frame(d, [4.11] * 30), 0.011, "abs").ok
    assert checks.check_cross(a, frame(d, [4.20] * 30), 0.011, "abs").status == "fail"
    assert checks.check_cross(frame(d, [100.0] * 30), frame(d, [100.05] * 30), 0.001, "rel").ok
    assert checks.check_cross(a, frame(d[:2], [4.1, 4.1]), 0.01, "abs").status == "warn"


# ───────────────────────────── database ─────────────────────────────

@pytest.fixture()
def eng(tmp_path):
    engine.cache_clear()
    e = engine(f"sqlite:///{tmp_path / 't.sqlite'}")
    init_db(e)
    return e


def test_upsert_is_idempotent_and_updates(eng):
    df = frame(["2026-09-01", "2026-09-02"], [1.0, 2.0])
    save_observations("k", df, eng)
    save_observations("k", frame(["2026-09-02", "2026-09-03"], [2.5, 3.0]), eng)
    out = load_series("k", eng)
    assert list(out["value"]) == [1.0, 2.5, 3.0]


def test_series_meta_roundtrip(eng):
    reg = load_registry()
    assert save_series_meta(reg.values(), eng=eng) == len(reg)
    with eng.connect() as c:
        assert c.execute(schema.series_meta.select()).rowcount != 0


def test_json_columns(eng):
    upsert(schema.orderbook_snapshots, [{"asset": "BTC", "ts": pd.Timestamp("2026-09-28", tz="UTC").to_pydatetime(),
                                         "bucket_size": 100.0, "mid_price": None, "venues": ["okx"],
                                         "depth": [{"side": "bid", "bucket": 1.0}]}], eng)


def test_committed_migration_matches_schema():
    committed = (ROOT / "db" / "migrations" / "001_init.sql").read_text()
    assert committed == schema.postgres_ddl(), "run: python -m db.schema > db/migrations/001_init.sql"
