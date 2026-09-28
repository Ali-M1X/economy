"""Loads config/series.yaml into typed metadata objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from core.settings import ROOT

REGISTRY_PATH = ROOT / "config" / "series.yaml"

FREQUENCIES = {"D", "W", "M", "Q", "intraday"}
SOURCES = {"fred", "treasury", "fiscaldata", "nyfed", "yahoo", "bls"}


@dataclass(frozen=True)
class CrossCheck:
    key: str
    tolerance: float
    mode: str  # "abs" (absolute difference) or "rel" (relative difference)


@dataclass(frozen=True)
class SeriesMeta:
    key: str
    name_en: str
    name_fa: str
    category: str
    source: str
    source_id: str
    url: str
    units: str
    frequency: str
    release_lag_days: int
    max_stale_days: int
    valid_min: float
    valid_max: float
    start: str
    proxy: bool = False
    proxy_for: str | None = None
    vintages: bool = False
    cross_check: CrossCheck | None = None


class RegistryError(ValueError):
    pass


def _parse(raw: dict, defaults: dict) -> SeriesMeta:
    data = {**defaults, **raw}
    cc = data.pop("cross_check", None)
    required = {"key", "name_en", "name_fa", "category", "source", "source_id", "url", "units", "frequency",
                "release_lag_days", "max_stale_days", "valid_min", "valid_max"}
    missing = required - data.keys()
    if missing:
        raise RegistryError(f"{data.get('key', '?')}: missing fields {sorted(missing)}")
    if data["frequency"] not in FREQUENCIES:
        raise RegistryError(f"{data['key']}: bad frequency {data['frequency']}")
    if data["source"] not in SOURCES:
        raise RegistryError(f"{data['key']}: unknown source {data['source']}")
    if data["valid_min"] >= data["valid_max"]:
        raise RegistryError(f"{data['key']}: valid_min >= valid_max")
    if data.get("proxy") and not data.get("proxy_for"):
        raise RegistryError(f"{data['key']}: proxy series must say what it is a proxy for")
    return SeriesMeta(
        cross_check=CrossCheck(**cc) if cc else None,
        **{k: data[k] for k in SeriesMeta.__dataclass_fields__ if k != "cross_check" and k in data},
    )


@lru_cache(maxsize=4)
def load_registry(path: Path = REGISTRY_PATH) -> dict[str, SeriesMeta]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = doc.get("defaults", {})
    out: dict[str, SeriesMeta] = {}
    for raw in doc["series"]:
        meta = _parse(raw, defaults)
        if meta.key in out:
            raise RegistryError(f"duplicate key {meta.key}")
        out[meta.key] = meta
    for meta in out.values():
        if meta.cross_check and meta.cross_check.key not in out:
            raise RegistryError(f"{meta.key}: cross_check target {meta.cross_check.key} not in registry")
    return out
