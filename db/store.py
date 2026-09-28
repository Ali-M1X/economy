"""Thin persistence layer: idempotent upserts into Postgres (Supabase) or SQLite."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from urllib.parse import unquote

from sqlalchemy import Table, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.engine import Engine

from core.registry import SeriesMeta
from core.settings import get_settings
from core.timeutil import utcnow
from db import schema


def parse_database_url(raw: str) -> URL:
    """Parse a Postgres/SQLite URL, tolerating *unencoded* special characters in the password.

    Supabase passwords may contain '@', ':' or ']', which break standard URL parsing (the host would be
    taken from inside the password). The credentials end at the LAST '@' before the host, so split there
    and let SQLAlchemy encode the parts itself."""
    if raw.startswith("sqlite"):
        return make_url(raw)
    scheme, sep, rest = raw.partition("://")
    if not sep:
        raise ValueError("DATABASE_URL must look like postgresql://user:password@host:port/dbname")
    driver = "postgresql+psycopg" if scheme in ("postgres", "postgresql", "postgresql+psycopg") else scheme
    userinfo, at, hostpart = rest.rpartition("@")
    if not at:
        raise ValueError("DATABASE_URL has no user:password@ part")
    user, _, password = userinfo.partition(":")
    hostport, _, dbq = hostpart.partition("/")
    database, _, query = dbq.partition("?")
    host, _, port = hostport.rpartition(":") if hostport.count(":") == 1 else (hostport, "", "")
    params = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
    return URL.create(driver, username=unquote(user), password=unquote(password), host=host,
                      port=int(port) if port else None, database=database or "postgres", query=params)


def secret_fragments(raw: str | None) -> list[str]:
    """Pieces of a connection string that must never appear in logs (password, userinfo, the raw URL,
    and every '@'/':'-separated chunk of the credentials — what a mis-parse would print)."""
    if not raw or raw.startswith("sqlite"):
        return []
    rest = raw.partition("://")[2]
    userinfo = rest.rpartition("@")[0]
    frags = {raw, rest, userinfo, userinfo.partition(":")[2]}
    for chunk in re.split(r"[@:]", userinfo):
        frags.add(chunk)
    return sorted((f for f in frags if len(f) >= 4), key=len, reverse=True)


def redact(text: str, raw: str | None = None) -> str:
    raw = raw if raw is not None else get_settings().database_url
    for f in secret_fragments(raw):
        text = text.replace(f, "***")
    return text


def mask_in_ci(raw: str | None = None) -> None:
    """Ask GitHub Actions to mask every credential fragment in all later log output."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for f in secret_fragments(raw if raw is not None else get_settings().database_url):
            print(f"::add-mask::{f}", flush=True)


@lru_cache(maxsize=2)
def engine(url: str | None = None) -> Engine:
    raw = url or get_settings().database_url
    if raw.startswith("sqlite:///"):
        Path(raw[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(raw, future=True)
    mask_in_ci(raw)
    # Supabase's transaction pooler (port 6543) cannot keep server-side prepared statements, so disable
    # psycopg's automatic preparing; harmless on direct and session-pooler connections.
    return create_engine(parse_database_url(raw), pool_pre_ping=True, future=True,
                         connect_args={"prepare_threshold": None})


def describe_target(eng: Engine) -> str:
    """Dialect + host only — never user or password (safe for public CI logs)."""
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


def load_all_series(keys: Iterable[str] | None = None, eng: Engine | None = None) -> dict[str, pd.DataFrame]:
    eng = eng or engine()
    t = schema.observations
    q = t.select().order_by(t.c.series_key, t.c.obs_date)
    if keys is not None:
        q = q.where(t.c.series_key.in_(list(keys)))
    with eng.connect() as conn:
        df = pd.DataFrame(conn.execute(q).fetchall(), columns=[c.name for c in t.columns])
    if df.empty:
        return {}
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return {k: pd.DataFrame({"date": g["obs_date"].to_numpy(), "value": g["value"].astype("float64").to_numpy()})
            for k, g in df.groupby("series_key")}


def load_vintages(key: str, eng: Engine | None = None) -> pd.DataFrame:
    eng = eng or engine()
    t = schema.observation_vintages
    with eng.connect() as conn:
        rows = conn.execute(t.select().where(t.c.series_key == key).order_by(t.c.obs_date, t.c.realtime_start)).fetchall()
    df = pd.DataFrame(rows, columns=[c.name for c in t.columns])
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "realtime_start", "realtime_end"])
    return pd.DataFrame({"date": pd.to_datetime(df["obs_date"]), "value": df["value"].astype("float64"),
                         "realtime_start": pd.to_datetime(df["realtime_start"]),
                         "realtime_end": pd.to_datetime(df["realtime_end"])})


def load_table(table, eng: Engine | None = None) -> pd.DataFrame:
    eng = eng or engine()
    with eng.connect() as conn:
        rows = conn.execute(table.select()).fetchall()
    return pd.DataFrame(rows, columns=[c.name for c in table.columns])
