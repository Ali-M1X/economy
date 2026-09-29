"""Where the dashboard gets its data.

Sources (first match wins):
1. MACRO_PULSE_SNAPSHOT_DIR — a local snapshot directory (the `output/` folder of a data-availability run:
   `cache/` + analysis outputs). Used for local runs, tests and the VPS.
2. MACRO_PULSE_SNAPSHOT_URL — a .tar.gz of that folder (e.g. the `snapshot-latest` GitHub release asset
   published by the workflow when the repository variable PUBLISH_SNAPSHOT is "true"). Cached for 15 min.
3. (Later) DATABASE_URL — once storage is enabled; not required for the dashboard to work.
Everything returned is plain pandas/dicts; the Streamlit layer caches it.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.registry import SeriesMeta, load_registry


@dataclass
class Bundle:
    root: Path
    as_of: str | None
    data: dict[str, pd.DataFrame]
    vintages: dict[str, pd.DataFrame]
    futures: pd.DataFrame
    calendar: pd.DataFrame
    news: pd.DataFrame
    registry: dict[str, SeriesMeta]
    availability: dict = field(default_factory=dict)  # series key → report row (status, source_url, …)
    macro_scores: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    reports: dict[str, str] = field(default_factory=dict)  # name → markdown

    def series(self, key: str) -> pd.Series | None:
        df = self.data.get(key)
        return None if df is None or df.empty else df.set_index("date")["value"].sort_index()

    def bars(self, name: str) -> pd.DataFrame | None:
        p = self.root / "cache" / f"candles_{name}_15m.csv.gz"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df


def resolve_root() -> Path:
    d = os.environ.get("MACRO_PULSE_SNAPSHOT_DIR")
    if d:
        return Path(d)
    url = os.environ.get("MACRO_PULSE_SNAPSHOT_URL")
    if url:
        return download_snapshot(url)
    default = Path(__file__).resolve().parent.parent / "output"
    if (default / "cache").exists():
        return default
    raise FileNotFoundError("No data snapshot: set MACRO_PULSE_SNAPSHOT_DIR or MACRO_PULSE_SNAPSHOT_URL "
                            "(see README → Dashboard).")


def download_snapshot(url: str) -> Path:
    from core.http import get

    raw = get("snapshot", url, timeout=120).content
    target = Path(tempfile.mkdtemp(prefix="macro_pulse_"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(target, filter="data")
    # the archive holds an `output/` folder
    return target / "output" if (target / "output").exists() else target


def load(root: Path | None = None) -> Bundle:
    from jobs.features import load_cache

    root = root or resolve_root()
    data, vintages, fq, cal, _ = load_cache(root / "cache")
    read_json = lambda n: json.loads((root / n).read_text(encoding="utf-8")) if (root / n).exists() else {}  # noqa: E731
    avail = read_json("data_availability.json")
    news_p = root / "cache" / "news.csv"
    news = pd.read_csv(news_p) if news_p.exists() else pd.DataFrame(columns=["source", "published_utc", "title", "url"])
    if not news.empty:
        news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, errors="coerce", format="mixed")
    coef_p = root / "impact_coefficients.csv"
    return Bundle(
        root=root, as_of=avail.get("generated_at"), data=data, vintages=vintages, futures=fq, calendar=cal, news=news,
        registry=load_registry(), availability={r["key"]: r for r in avail.get("series", [])},
        macro_scores=read_json("macro_scores.json"), signals=read_json("signals.json"),
        coefficients=pd.read_csv(coef_p) if coef_p.exists() else pd.DataFrame(),
        reports={n: (root / f"{n}.md").read_text(encoding="utf-8") for n in
                 ("data_availability", "features_summary", "impact_summary", "signals") if (root / f"{n}.md").exists()},
    )


def analysis(b: Bundle, today) -> dict:
    """Phase 2 results (features, FedWatch, stance, regimes) computed from the bundle."""
    from jobs.features import compute_all

    return compute_all(b.data, b.vintages, b.futures, b.calendar, pd.DataFrame(), today)
