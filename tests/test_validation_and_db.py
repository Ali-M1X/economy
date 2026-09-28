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
                "XLU", "^NDX", "CL=F", "BZ=F", "SI=F", "NG=F", "DBC"}
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


def test_monthly_staleness_counts_from_period_end():
    # August data (dated 2026-08-01) is the latest possible until the September release in early October
    m = meta(frequency="M", max_stale_days=45)
    df = frame(["2026-08-01"], [4.1])
    assert checks.staleness_days(df, date(2026, 9, 28), "M") == 28
    assert checks.check_staleness(df, m, date(2026, 9, 28)).ok
    assert checks.check_staleness(df, m, date(2026, 10, 20)).status == "fail"
    assert checks.staleness_days(frame(["2026-04-01"], [1]), date(2026, 9, 28), "Q") == 90


def test_projection_staleness_uses_publication_time():
    m = meta(frequency="A", projection=True, max_stale_days=120)
    df = frame(["2026-01-01", "2027-01-01", "2028-01-01"], [3.6, 3.4, 3.1])  # target years, in the future
    from datetime import datetime, timezone
    pub = datetime(2026, 9, 16, 18, tzinfo=timezone.utc)
    assert checks.check_staleness(df, m, date(2026, 9, 28), pub).ok
    assert checks.check_staleness(df, m, date(2027, 3, 1), pub).status == "fail"
    assert checks.check_staleness(df, m, date(2026, 9, 28), None).status == "warn"
    assert checks.check_gaps(df, m).ok


def test_seven_day_series_skip_weekend_check():
    days = pd.date_range("2025-01-01", periods=400, freq="D")
    assert checks.check_weekend_dates(frame(days, [1] * 400), meta()).status == "fail"
    assert checks.check_weekend_dates(frame(days, [1] * 400), meta(calendar="7d")).ok


def test_registry_seven_day_and_projection_flags():
    reg = load_registry()
    assert {k for k, m in reg.items() if m.calendar == "7d"} == {
        "fed_funds_eff_daily", "fed_target_upper", "fed_target_lower", "iorb"}
    assert reg["sep_fed_funds_median"].projection and reg["sep_fed_funds_median"].vintages


def test_weekend_dates_catch_timezone_shift():
    sundays = pd.date_range("2025-01-05", periods=60, freq="W-SUN")
    assert checks.check_weekend_dates(frame(sundays, [1] * 60), meta()).status == "fail"
    bdays = pd.bdate_range("2025-01-01", periods=300)
    assert checks.check_weekend_dates(frame(bdays, [1] * 300), meta()).ok
    assert checks.check_weekend_dates(frame(sundays, [1] * 60), meta(category="crypto")).ok


def test_gap_check_reports_the_actual_gap():
    df = frame(["2026-01-02", "2026-01-05", "2026-03-01", "2026-03-02"], [1, 1, 1, 1])
    r = checks.check_gaps(df, meta())
    assert r.status == "warn" and "2026-01-05 → 2026-03-01" in r.message and "55d" in r.message


def test_gap_check_monthly_missing_month():
    # A month the source never published: Sep → Nov is a 61-day gap
    df = frame(["2025-08-01", "2025-09-01", "2025-11-01", "2025-12-01"], [1, 1, 1, 1])
    r = checks.check_gaps(df, meta(frequency="M"))
    assert r.status == "warn" and "2025-09-01 → 2025-11-01" in r.message


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
