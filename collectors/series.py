"""Dispatch a registry entry to its collector."""

from __future__ import annotations

from collectors import bls, fred, nyfed, treasury, yahoo
from collectors.base import SeriesResult
from core.registry import SeriesMeta

FETCHERS = {
    "fred": fred.fetch,
    "treasury": treasury.fetch,
    "fiscaldata": treasury.fetch_tga,
    "nyfed": nyfed.fetch,
    "yahoo": yahoo.fetch,
    "bls": bls.fetch,
}


def fetch_series(meta: SeriesMeta, start: str | None = None) -> SeriesResult:
    return FETCHERS[meta.source](meta, start)
