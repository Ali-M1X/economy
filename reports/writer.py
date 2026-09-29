"""Claude-written causal chain for each asset's report — composed only from the facts and the template sentences.

The spec's rule "no invented figures" is enforced, not just requested: every number in Claude's text must appear in
the facts (after rounding), otherwise the text is rejected and the deterministic explanation is used instead.
"""

from __future__ import annotations

import json
import re

import numpy as np

from llm import claude
from notify.telegram import esc
from reports.facts import Facts

_NUM = re.compile(r"[-+−]?\d[\d,٬]*(?:[.٫]\d+)?")
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫٬−", "01234567890123456789.,-")
# numbers that name horizons/units rather than data ("24 ساعت", "4 هفته", "6 ماه", "10 ساله", "−100 … +100", "1σ")
_FREE = {0, 1, 2, 3, 4, 5, 6, 10, 12, 24, 30, 100}


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text.translate(_FA_DIGITS)):
        try:
            out.append(float(m.replace(",", "").replace("+", "")))
        except ValueError:
            pass
    return out


def _allowed(obj) -> set[float]:
    vals: set[float] = set()

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)) and np.isfinite(o):
            for d in (0, 1, 2, 3, 4):
                vals.add(round(float(o), d))
                vals.add(round(abs(float(o)), d))
                if abs(o) <= 1.0:  # probabilities are quoted as percentages
                    vals.add(round(float(o) * 100, d))
            return
        if isinstance(o, str):
            for n in _numbers(o):  # dates / meeting days inside strings
                vals.add(n)
                vals.add(abs(n))
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return vals


def unknown_numbers(text: str, facts: dict) -> list[float]:
    allowed = _allowed(facts) | _FREE
    bad = []
    for n in _numbers(text):
        if abs(n) in _FREE or n in allowed or abs(n) in allowed:
            continue
        if any(abs(abs(n) - a) < 1e-9 for a in allowed):
            continue
        bad.append(n)
    return bad


ASSET_SYSTEM = """You write the causal-chain section of a Persian (Farsi) report on ONE asset ({asset_fa}). You receive FACTS
(JSON, computed by code) and TEMPLATE sentences already built from those facts.

Rules:
- Write in Persian. Keep tickers and indicator names (CPI, DXY, FOMC …) in English where the facts do.
- Use ONLY numbers that appear in FACTS or TEMPLATE, written as they appear (you may round). Never add any other number.
- 3 to 6 short, complete sentences, one per line, numbered "1." "2." …; each sentence states cause → effect explicitly
  (what changed, through which channel, and what it means for {asset_fa}). No lists of tickers, no arrows.
- Cover only {asset_fa}. If the evidence is weak (scores near zero, coefficients below 1), say so plainly.
- Plain text only, no Markdown, no HTML, no investment advice."""


def asset_facts(f: Facts, asset: str) -> dict:
    s = f.summary()
    return {"as_of": s["as_of"], "regime": s["regime"], "fed_stance": s["fed_stance"], "releases": s["releases"],
            "asset": asset, **s["assets"].get(asset, {})}


def explain_asset(f: Facts, asset: str, template: list[str]) -> tuple[list[str], str]:
    """(sentences, source) for one asset — Claude's rewrite of the template when it passes the number check."""
    if not claude.available():
        return template, "template"
    from reports.messages import ASSET_FA

    facts = asset_facts(f, asset)
    user = ("FACTS:\n" + json.dumps(facts, ensure_ascii=False, default=str) + "\n\nTEMPLATE:\n" + "\n".join(template))
    try:
        text = claude.text_call(ASSET_SYSTEM.format(asset_fa=ASSET_FA[asset]), user, max_tokens=3000, effort="medium")
    except claude.LLMError as e:
        print(f"::warning::Claude explanation for {asset} unavailable ({e}); using the template")
        return template, "template"
    bad = unknown_numbers(text, {"facts": facts, "template": template})
    if bad:
        print(f"::warning::Claude explanation for {asset} used numbers not in the facts {bad[:5]}; using the template")
        return template, "template"
    lines = [re.sub(r"^\s*[\d۰-۹]+[.)]\s*", "", ln).strip() for ln in text.strip().splitlines()]
    lines = [esc(ln) for ln in lines if ln]  # the template is already HTML-safe; Claude's text is escaped here
    return (lines or template), ("claude" if lines else "template")
