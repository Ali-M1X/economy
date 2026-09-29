"""Claude-written causal explanation for the impact report — composed only from the facts and the channel table.

The spec's rule "no invented figures" is enforced, not just requested: every number in Claude's text must appear in
the facts (after rounding), otherwise the text is rejected and the deterministic explanation is used instead.
"""

from __future__ import annotations

import json
import re

import numpy as np

from llm import claude
from reports.facts import Facts, indicators_cfg

SYSTEM = """You write the causal-explanation paragraph of a Persian (Farsi) report on how US macro data affects Bitcoin
and gold. You receive FACTS (JSON) computed by code, and a CHANNELS table.

Rules:
- Write in Persian. Keep tickers and indicator names (CPI, DXY, FOMC, M2 …) in English.
- Use ONLY numbers that appear in FACTS, written exactly as they appear there (you may round to fewer decimals).
  Never compute, estimate or invent any other number, date or percentage.
- Explain effects ONLY through the transmission channels in CHANNELS, as chains such as
  "CPI بالاتر از انتظار ← احتمال افزایش نرخ ← بازده واقعی بالاتر ← دلار قوی‌تر ← منفی برای طلا و BTC".
- Cover: what changed (releases, if any), the chain of effects, then the implication for BTC and for gold in the short
  (24h), medium (4 weeks) and long (6 months) horizon, citing the macro scores and the largest contributors.
- If the evidence is weak or conflicting (scores near zero, coefficients opposite to theory), say so plainly.
- Plain text only, no Markdown, no HTML. At most 220 words. No investment advice."""

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


def fallback(summary: dict) -> str:
    """Deterministic explanation (used without an API key or when Claude's text fails validation)."""
    ch = summary.get("channels_fa", {})
    parts = []
    for r in summary.get("releases", [])[:3]:
        hot = "بالاتر" if r["z"] > 0 else "پایین‌تر"
        path = " ← ".join(ch.get(c, c) for c in r.get("channels", []))
        parts.append(f"{r['name_fa']} {hot} از انتظار آمد؛ مسیر اثر: {path}.")
    for a, fa_name in (("BTC", "بیت‌کوین"), ("Gold", "طلا")):
        x = summary["assets"].get(a, {})
        med = (x.get("macro_score") or {}).get("medium")
        top = (x.get("top_contributors") or {}).get("medium", [])[:2]
        if med is None:
            continue
        tone = "مثبت" if med >= 15 else "منفی" if med <= -15 else "خنثی و ضعیف"
        drivers = "، ".join(f"{t['name_fa']} ({t['contribution']:+.1f})" for t in top)
        parts.append(f"برای {fa_name} برآیند کلان ۴ هفته {tone} است ({med:+.0f})" + (f"؛ بیشترین سهم: {drivers}." if drivers else "."))
    return " ".join(parts)


def explain(f: Facts) -> tuple[str, str]:
    """(text, source) — source is "claude" or "template"."""
    summary = f.summary()
    if not claude.available():
        return fallback(summary), "template"
    cfg = indicators_cfg()
    user = ("FACTS:\n" + json.dumps(summary, ensure_ascii=False, default=str) +
            "\n\nCHANNELS:\n" + json.dumps({k: v["en"] + " | " + v["fa"] for k, v in cfg["channels"].items()}, ensure_ascii=False))
    try:
        text = claude.text_call(SYSTEM, user, max_tokens=4000, effort="medium")
    except claude.LLMError as e:
        print(f"::warning::Claude explanation unavailable ({e}); using the template")
        return fallback(summary), "template"
    bad = unknown_numbers(text, summary)
    if bad:
        print(f"::warning::Claude explanation used numbers not in the facts {bad[:5]}; using the template")
        return fallback(summary), "template"
    return text.strip(), "claude"

