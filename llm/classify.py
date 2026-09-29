"""Claude classification of news items and Fed documents.

News: relevance and direction for BTC and gold, importance 1–5, transmission channels, a Persian one-line title.
Fed documents: a hawkish/dovish label mapped to −1…+1 for the Fed Stance Score (component c). Labels are discrete
enums so the mapping to numbers happens in code, not in the model.
"""

from __future__ import annotations

import json

from llm.claude import json_call

CHANNELS = ["rates", "real_yields", "policy", "dollar", "liquidity", "risk_appetite", "credit", "growth", "inflation",
            "safe_haven", "crypto_specific", "gold_specific", "geopolitics"]
EFFECT = ["bullish", "bearish", "neutral", "unrelated"]
STANCE = {"very_dovish": -1.0, "dovish": -0.5, "neutral": 0.0, "hawkish": 0.5, "very_hawkish": 1.0}
BATCH = 25

NEWS_SYSTEM = """You classify financial news for a system that tracks the effect of US macro news on Bitcoin (BTC) and gold.
For each item decide, from the headline and summary only (do not assume facts that are not in the text):
- btc / gold: the likely first-order effect on that asset — bullish, bearish, neutral (relevant but no clear direction) or unrelated.
- importance 1–5 for BTC/gold traders: 5 = market-moving (Fed decision or surprise, major data shock, war/escalation,
  major crypto regulation or ETF decision); 4 = clearly relevant and likely to move prices; 3 = relevant background;
  2 = minor; 1 = noise or unrelated.
- channels: the transmission channels involved (empty if unrelated).
- title_fa: a faithful Persian translation of the headline, keeping tickers and indicator names (CPI, FOMC, ETF…) in English.
Return one entry per input id."""

FED_SYSTEM = """You rate the monetary-policy stance expressed in Federal Reserve documents (statements, minutes, speeches,
testimony) relative to the current policy setting. very_hawkish = signals tightening or strongly resists easing;
hawkish = leans toward tighter/for-longer; neutral = balanced or not about policy; dovish = leans toward easing;
very_dovish = signals imminent or substantial easing. Rate only what the text says. rationale_fa: one short Persian sentence."""


def _news_schema() -> dict:
    item = {"type": "object", "additionalProperties": False,
            "required": ["id", "btc", "gold", "importance", "channels", "title_fa"],
            "properties": {"id": {"type": "string"}, "btc": {"type": "string", "enum": EFFECT},
                           "gold": {"type": "string", "enum": EFFECT}, "importance": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                           "channels": {"type": "array", "items": {"type": "string", "enum": CHANNELS}},
                           "title_fa": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": item}}}


def _fed_schema() -> dict:
    item = {"type": "object", "additionalProperties": False, "required": ["id", "stance", "rationale_fa"],
            "properties": {"id": {"type": "string"}, "stance": {"type": "string", "enum": list(STANCE)},
                           "rationale_fa": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "items": item}}}


def _batches(items: list[dict], n: int = BATCH):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _payload(items: list[dict], text_key: str = "summary") -> str:
    return json.dumps([{"id": i["id"], "source": i.get("source"), "published_utc": str(i.get("published_utc")),
                        "title": i.get("title"), "text": (i.get(text_key) or "")} for i in items], ensure_ascii=False)


def classify_news(items: list[dict]) -> list[dict]:
    """items: id, source, published_utc, title, summary → rows with the classification (ids the model skipped are dropped)."""
    out = []
    for chunk in _batches(items):
        known = {i["id"] for i in chunk}
        res = json_call(NEWS_SYSTEM, "Classify these items:\n" + _payload(chunk), _news_schema(), max_tokens=8000)
        out += [r for r in res["items"] if r["id"] in known]
    return out


def classify_fed_docs(docs: list[dict]) -> list[dict]:
    """docs: id, doc_type, published_utc, title, text → rows with stance label and hawkish_score (−1…+1)."""
    out = []
    for chunk in _batches(docs, 10):
        known = {d["id"] for d in chunk}
        res = json_call(FED_SYSTEM, "Rate these documents:\n" + _payload(chunk, "text"), _fed_schema(), max_tokens=6000)
        out += [{**r, "hawkish_score": STANCE[r["stance"]]} for r in res["items"] if r["id"] in known]
    return out
