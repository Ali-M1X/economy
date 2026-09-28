"""Apply db/migrations/*.sql in order (Postgres/Supabase), recording each in `schema_migrations`.

SQLite (local/offline) has no RLS or DO-blocks, so it is initialised from the SQLAlchemy metadata instead.

    python -m db.migrate
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine

from db import schema

MIGRATIONS = Path(__file__).parent / "migrations"


def migrate(eng: Engine) -> list[str]:
    if eng.dialect.name != "postgresql":
        schema.metadata.create_all(eng)
        return ["sqlite: create_all"]
    applied_now: list[str] = []
    # A raw DB-API connection runs each file as one simple query: no parameter parsing, so `%I` in
    # format() calls and DO $$ blocks pass through untouched.
    raw = eng.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        raw.commit()
        cur.execute("SELECT filename FROM schema_migrations")
        done = {r[0] for r in cur.fetchall()}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name in done:
                continue
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            raw.commit()
            applied_now.append(path.name)
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return applied_now


if __name__ == "__main__":
    from db.store import engine

    print(migrate(engine()) or "up to date")
