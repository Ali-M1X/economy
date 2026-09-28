"""Thin persistence layer: idempotent upserts into Postgres (Supabase) or SQLite."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import Table, create_engine
from sqlalchemy.engine import Engine

from core.registry import SeriesMeta
from core.settings import get_settings
from core.timeutil import utcnow
from db import schema


@lru_cache(maxsize=2)
def engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    if url.startswith("postgres://"):  # Supabase shows this scheme; SQLAlchemy wants postgresql+psycopg
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(url, future=True)
    # Supabase's transaction pooler (port 6543) cannot keep server-side prepared statements, so disable
    # psycopg's automatic preparing; harmless on direct and session-pooler connections.
    return create_engine(url, pool_pre_ping=True, future=True, connect_args={"prepare_threshold": None})


def describe_target(eng: Engine) -> str:
    """Dialect + host, without credentials (safe to print in CI logs)."""
    u = eng.url
    return "sqlite (local file)" if u.get_backend_name() == "sqlite" else f"postgresql @ {u.host}:{u.port}/{u.database}"


def init_db(eng: Engine | None = None) -> list[str]:
    from db.migrate import migrate

    return migrate(eng or engine())


def upsert(table: Table, rows: Iterable[dict], eng: Engine | None = None) -> int:
    rows = list(rows)
    if not rows:
        return 0
    eng = eng or engine()
    if eng.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif eng.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover
        raise NotImplementedError(eng.dialect.name)
    pk = [c.name for c in table.primary_key.columns]
    # Postgres allows 65535 bind parameters per statement; SQLite (≥3.32) 32766.
    chunk = max(1, (30000 if eng.dialect.name == "sqlite" else 60000) // len(rows[0]))
    with eng.begin() as conn:
        for i in range(0, len(rows), chunk):
            stmt = insert(table).values(rows[i:i + chunk])
            update = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk}
            stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update) if update else stmt.on_conflict_do_nothing()
            conn.execute(stmt)
    return len(rows)


def table_counts(eng: Engine | None = None) -> dict[str, int]:
    from sqlalchemy import func, select

    eng = eng or engine()
    with eng.connect() as conn:
        return {t.name: conn.execute(select(func.count()).select_from(t)).scalar_one() for t in schema.metadata.sorted_tables}


def save_series_meta(metas: Iterable[SeriesMeta], status: dict[str, dict] | None = None,
                     eng: Engine | None = None) -> int:
    """Upsert registry metadata; `status` optionally adds per-key fetch status columns."""
    status = status or {}
    rows = [{
        "key": m.key, "name_en": m.name_en, "name_fa": m.name_fa, "category": m.category, "source": m.source,
        "source_id": m.source_id, "url": m.url, "units": m.units, "frequency": m.frequency,
        "release_lag_days": m.release_lag_days, "max_stale_days": m.max_stale_days, "proxy": m.proxy,
        "proxy_for": m.proxy_for, **status.get(m.key, {}),
    } for m in metas]
    return upsert(schema.series_meta, rows, eng)


def save_observations(key: str, df: pd.DataFrame, eng: Engine | None = None) -> int:
    now = utcnow()
    rows = [{"series_key": key, "obs_date": d.date(), "value": float(v), "fetched_at": now}
            for d, v in zip(df["date"], df["value"])]
    return upsert(schema.observations, rows, eng)


def save_vintages(key: str, df: pd.DataFrame, eng: Engine | None = None) -> int:
    rows = [{"series_key": key, "obs_date": r.date.date(), "realtime_start": r.realtime_start.date(),
             "realtime_end": None if pd.isna(r.realtime_end) else r.realtime_end.date(), "value": float(r.value)}
            for r in df.itertuples()]
    return upsert(schema.observation_vintages, rows, eng)


def load_series(key: str, eng: Engine | None = None) -> pd.DataFrame:
    eng = eng or engine()
    t = schema.observations
    with eng.connect() as conn:
        rows = conn.execute(t.select().where(t.c.series_key == key).order_by(t.c.obs_date)).fetchall()
    return pd.DataFrame({"date": pd.to_datetime([r.obs_date for r in rows]), "value": [r.value for r in rows]})
