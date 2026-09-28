"""Database schema (single source of truth).

`python -m db.schema > db/migrations/001_init.sql` regenerates the Postgres (Supabase) migration;
a test asserts the committed SQL matches this module. SQLite is supported for local/offline runs.
"""

from __future__ import annotations

import sys

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text,
                        UniqueConstraint)
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy import Index

metadata = MetaData()

TS = DateTime(timezone=True)

series_meta = Table(
    "series_meta", metadata,
    Column("key", String(64), primary_key=True),
    Column("name_en", Text, nullable=False),
    Column("name_fa", Text, nullable=False),
    Column("category", String(32), nullable=False),
    Column("source", String(32), nullable=False),
    Column("source_id", String(128), nullable=False),
    Column("url", Text, nullable=False),
    Column("units", String(64), nullable=False),
    Column("frequency", String(16), nullable=False),
    Column("release_lag_days", Integer, nullable=False),
    Column("max_stale_days", Integer, nullable=False),
    Column("proxy", Boolean, nullable=False, default=False),
    Column("proxy_for", Text),
    Column("last_fetched_at", TS),
    Column("last_obs_date", Date),
    Column("source_last_updated", TS),
    Column("last_status", String(16)),  # ok | warn | fail
    Column("last_message", Text),
)

# Latest (revised) value of every observation.
observations = Table(
    "observations", metadata,
    Column("series_key", String(64), primary_key=True),
    Column("obs_date", Date, primary_key=True),
    Column("value", Float, nullable=False),
    Column("fetched_at", TS, nullable=False),
)

# ALFRED vintages: the value as it was known between realtime_start and realtime_end (NULL = current).
observation_vintages = Table(
    "observation_vintages", metadata,
    Column("series_key", String(64), primary_key=True),
    Column("obs_date", Date, primary_key=True),
    Column("realtime_start", Date, primary_key=True),
    Column("realtime_end", Date),
    Column("value", Float, nullable=False),
)

candles = Table(
    "candles", metadata,
    Column("asset", String(16), primary_key=True),  # BTC | PAXG | GC=F ...
    Column("venue", String(32), primary_key=True),
    Column("interval", String(8), primary_key=True),
    Column("ts", TS, primary_key=True),  # bar open time, UTC
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float),
    Column("is_proxy", Boolean, nullable=False, default=False),
)

orderbook_snapshots = Table(
    "orderbook_snapshots", metadata,
    Column("asset", String(16), primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("bucket_size", Float, nullable=False),
    Column("mid_price", Float),
    Column("venues", JSON, nullable=False),  # list of venues aggregated
    Column("depth", JSON, nullable=False),  # [{side, bucket, qty, usd, venues}]
)

derivatives_snapshots = Table(
    "derivatives_snapshots", metadata,
    Column("asset", String(16), primary_key=True),
    Column("venue", String(32), primary_key=True),
    Column("ts", TS, primary_key=True),
    Column("open_interest", Float),  # base units (BTC)
    Column("open_interest_usd", Float),
    Column("funding_rate", Float),
    Column("mark_price", Float),
    Column("next_funding_ts", TS),
)

liquidations = Table(
    "liquidations", metadata,
    Column("raw_id", String(160), primary_key=True),
    Column("venue", String(32), nullable=False),
    Column("asset", String(16), nullable=False),
    Column("ts", TS, nullable=False),
    Column("side", String(8), nullable=False),  # long | short (the position that was liquidated)
    Column("price", Float, nullable=False),
    Column("qty", Float),
    Column("usd", Float),
)

futures_quotes = Table(
    "futures_quotes", metadata,
    Column("contract", String(32), primary_key=True),
    Column("quote_ts", TS, primary_key=True),
    Column("root", String(8), nullable=False),  # ZQ | SR3
    Column("contract_month", Date, nullable=False),
    Column("price", Float, nullable=False),
    Column("implied_rate", Float, nullable=False),
)

calendar_events = Table(
    "calendar_events", metadata,
    Column("event_id", String(64), primary_key=True),
    Column("release_id", String(32), nullable=False),
    Column("name_en", Text, nullable=False),
    Column("name_fa", Text, nullable=False),
    Column("scheduled_utc", TS, nullable=False),
    Column("importance", Integer, nullable=False),
    Column("date_source", String(32), nullable=False),
    Column("consensus", Float),
    Column("consensus_source", String(64)),  # NULL when no free consensus exists
    Column("previous", Float),
    Column("actual", Float),
    Column("actual_known_at", TS),
    Column("surprise_z", Float),
    Column("surprise_basis", String(32)),  # consensus | trend | previous
    Column("status", String(16), nullable=False, default="scheduled"),
)

news_items = Table(
    "news_items", metadata,
    Column("id", String(40), primary_key=True),
    Column("source", String(64), nullable=False),
    Column("published_utc", TS),
    Column("title", Text, nullable=False),
    Column("url", Text, nullable=False),
    Column("summary", Text),
    Column("relevance_btc", Float),
    Column("relevance_gold", Float),
    Column("direction_btc", Integer),  # −1 / 0 / +1
    Column("direction_gold", Integer),
    Column("importance", Integer),  # 1..5, only ≥4 enters the model
    Column("classified_at", TS),
    Column("classifier_model", String(64)),
)

fed_documents = Table(
    "fed_documents", metadata,
    Column("id", String(40), primary_key=True),
    Column("doc_type", String(32), nullable=False),  # statement | minutes | speech | testimony
    Column("published_utc", TS),
    Column("title", Text, nullable=False),
    Column("url", Text, nullable=False),
    Column("text", Text),
    Column("hawkish_score", Float),  # −1 (dovish) .. +1 (hawkish)
    Column("classified_at", TS),
    Column("classifier_model", String(64)),
)

# Derived series produced in Phase 2+ (net liquidity, spreads, z-scores, stance score, ...).
features = Table(
    "features", metadata,
    Column("feature_key", String(64), primary_key=True),
    Column("obs_date", Date, primary_key=True),
    Column("value", Float, nullable=False),
    Column("computed_at", TS, nullable=False),
)

regime_states = Table(
    "regime_states", metadata,
    Column("model", String(32), primary_key=True),  # rules | hmm
    Column("obs_date", Date, primary_key=True),
    Column("regime", String(16), nullable=False),
    Column("probabilities", JSON, nullable=False),
    Column("computed_at", TS, nullable=False),
)

impact_coefficients = Table(
    "impact_coefficients", metadata,
    Column("computed_on", Date, primary_key=True),
    Column("indicator", String(64), primary_key=True),
    Column("asset", String(8), primary_key=True),
    Column("horizon", String(16), primary_key=True),  # 1h | 4h | 24h | 1-4w | 3-12m
    Column("coefficient", Float, nullable=False),  # signed 0..10
    Column("effect_per_sigma", Float),
    Column("t_stat", Float),
    Column("hit_rate", Float),
    Column("n", Integer),
    Column("confidence", String(8)),
    Column("details", JSON),
)

backtests = Table(
    "backtests", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_at", TS, nullable=False),
    Column("strategy", String(64), nullable=False),
    Column("asset", String(8), nullable=False),
    Column("horizon", String(16), nullable=False),
    Column("win_rate", Float), Column("avg_r", Float), Column("max_drawdown", Float), Column("n_trades", Integer),
    Column("params", JSON),
)

signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", TS, nullable=False),
    Column("asset", String(8), nullable=False),
    Column("direction", String(8), nullable=False),
    Column("horizon", String(16), nullable=False),
    Column("entry_low", Float), Column("entry_high", Float), Column("stop_loss", Float),
    Column("tp1", Float), Column("tp2", Float), Column("tp3", Float),
    Column("rr", JSON), Column("win_rate", Float), Column("confidence", String(8)),
    Column("invalidation", Text), Column("reasons", JSON), Column("backtest_id", Integer),
    Column("status", String(16), nullable=False, default="active"),
)

reports = Table(
    "reports", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", TS, nullable=False),
    Column("kind", String(32), nullable=False),  # impact | heads_up | weekly | health
    Column("trigger", String(64)),
    Column("body_fa", Text, nullable=False),
    Column("data", JSON),
    Column("sent_telegram", Boolean, nullable=False, default=False),
)

fetch_log = Table(
    "fetch_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("job", String(32), nullable=False),
    Column("target", String(128), nullable=False),  # series key or probe name
    Column("started_at", TS, nullable=False),
    Column("finished_at", TS, nullable=False),
    Column("status", String(8), nullable=False),  # ok | warn | fail
    Column("rows", Integer),
    Column("message", Text),
    UniqueConstraint("run_id", "target", name="uq_fetch_log_run_target"),
)

Index("ix_candles_asset_interval_ts", candles.c.asset, candles.c.interval, candles.c.ts)
Index("ix_liquidations_asset_ts", liquidations.c.asset, liquidations.c.ts)
Index("ix_calendar_scheduled", calendar_events.c.scheduled_utc)
Index("ix_news_published", news_items.c.published_utc)
Index("ix_signals_created", signals.c.created_at)
Index("ix_reports_created", reports.c.created_at)


def postgres_ddl() -> str:
    from sqlalchemy.dialects import postgresql

    dialect = postgresql.dialect()
    parts = ["-- Generated by `python -m db.schema`. Do not edit by hand.\n"]
    for table in metadata.sorted_tables:
        parts.append(str(CreateTable(table, if_not_exists=True).compile(dialect=dialect)).strip() + ";\n")
        for idx in sorted(table.indexes, key=lambda i: i.name):
            parts.append(str(CreateIndex(idx, if_not_exists=True).compile(dialect=dialect)).strip() + ";\n")
    return "\n".join(parts)


if __name__ == "__main__":
    sys.stdout.write(postgres_ddl())
