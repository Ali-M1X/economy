"""Classify the snapshot's news and Fed documents with Claude and write the results next to the snapshot.

    python -m jobs.classify --cache output/cache [--state state/]

Writes cache/news_scored.csv and cache/fed_docs_scored.csv (read by jobs.features for the Fed Stance Score and by
the reports). `--state` keeps earlier classifications between runs (restored by the workflow from the Actions cache),
so an item is sent to Claude once. Without ANTHROPIC_API_KEY it does nothing and exits 0: the stance component
simply stays unavailable, exactly as before.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from llm import claude
from llm.classify import classify_fed_docs, classify_news

FED_SOURCES = {"fed_monetary": "statement", "fed_all": "press", "fed_speeches": "speech", "fed_testimony": "testimony"}
NEWS_DAYS, FED_DAYS = 3, 60


def fetch_fed_text(url: str) -> str | None:
    """Full text of a Fed press release (statements are short); None if the page can't be read."""
    from bs4 import BeautifulSoup

    from core.http import get

    try:
        soup = BeautifulSoup(get("fed", url).content, "html.parser")
    except Exception:  # noqa: BLE001 — fall back to the RSS summary
        return None
    node = soup.select_one("#article") or soup.select_one("#content") or soup.body
    return " ".join(node.get_text(" ").split()) if node else None


def load_state(path: Path | None, name: str) -> pd.DataFrame:
    p = path / name if path else None
    return pd.read_csv(p) if p and p.exists() else pd.DataFrame(columns=["id"])


def run(cache: Path, state: Path | None, now: pd.Timestamp) -> dict:
    news_p = cache / "news.csv"
    if not news_p.exists():
        return {"skipped": "no news.csv in the snapshot"}
    news = pd.read_csv(news_p)
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True, errors="coerce", format="mixed")
    if "summary" not in news:
        news["summary"] = None
    news = news.astype({"summary": object}).where(news.notna(), None)
    is_fed = news["source"].isin(FED_SOURCES)

    done_news, done_fed = load_state(state, "news_scored.csv"), load_state(state, "fed_docs_scored.csv")
    # Fed items go through both: the news classifier (an FOMC decision is the most important news there is) and the
    # stance classifier below
    new_news = news[(news["published_utc"] >= now - pd.Timedelta(days=NEWS_DAYS)) & ~news["id"].isin(done_news["id"])]
    new_fed = news[is_fed & (news["published_utc"] >= now - pd.Timedelta(days=FED_DAYS)) & ~news["id"].isin(done_fed["id"])]
    # the same document appears in several Fed feeds: classify it once
    new_fed = new_fed.drop_duplicates("url")

    scored_news = pd.DataFrame(classify_news(new_news.to_dict("records"))) if not new_news.empty else pd.DataFrame()
    docs = []
    for r in new_fed.to_dict("records"):
        text = fetch_fed_text(r["url"]) if r["source"] == "fed_monetary" else None
        docs.append({**r, "doc_type": FED_SOURCES[r["source"]], "text": text or r.get("summary") or ""})
    scored_fed = pd.DataFrame(classify_fed_docs(docs)) if docs else pd.DataFrame()

    meta = news.set_index("id")[["source", "published_utc", "title", "url"]]
    if not scored_news.empty:
        scored_news = scored_news.join(meta, on="id")
        scored_news["channels"] = scored_news["channels"].map(lambda c: "|".join(c))
    if not scored_fed.empty:
        scored_fed = scored_fed.join(meta, on="id")
        scored_fed["doc_type"] = scored_fed["source"].map(FED_SOURCES)
    all_news = pd.concat([done_news, scored_news], ignore_index=True).drop_duplicates("id", keep="last")
    all_fed = pd.concat([done_fed, scored_fed], ignore_index=True).drop_duplicates("id", keep="last")
    all_news.to_csv(cache / "news_scored.csv", index=False)
    all_fed.to_csv(cache / "fed_docs_scored.csv", index=False)
    if state:
        state.mkdir(parents=True, exist_ok=True)
        all_news.to_csv(state / "news_scored.csv", index=False)
        all_fed.to_csv(state / "fed_docs_scored.csv", index=False)
    return {"news_new": len(scored_news), "news_total": len(all_news), "fed_new": len(scored_fed), "fed_total": len(all_fed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--state")
    args = ap.parse_args(argv)
    if not claude.available():
        print("ANTHROPIC_API_KEY not set — skipping news / Fed document classification")
        return 0
    try:
        res = run(Path(args.cache), Path(args.state) if args.state else None, pd.Timestamp.now(tz="UTC"))
    except claude.LLMError as e:  # a failed classification must not fail the pipeline
        print(f"::warning::classification skipped: {e}")
        return 0
    print(f"classification: {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
