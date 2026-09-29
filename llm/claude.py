"""Thin Claude API wrapper. The numbers always come from code; Claude only classifies text and writes explanations.

Model: CLAUDE_MODEL (default `claude-sonnet-5`, as specified for this project). Every call can fail — no key,
network, refusal, truncated output — and raises LLMError; callers fall back to deterministic output, so a report
is never blocked by the LLM.
"""

from __future__ import annotations

import json

from core.settings import get_settings


class LLMError(RuntimeError):
    pass


def available() -> bool:
    return bool(get_settings().anthropic_api_key)


def _client():
    import anthropic

    s = get_settings()
    if not s.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=s.anthropic_api_key, timeout=120.0, max_retries=3)


def _text(response) -> str:
    if response.stop_reason == "refusal":
        raise LLMError("model declined the request")
    if response.stop_reason == "max_tokens":
        raise LLMError("output truncated (max_tokens)")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMError("empty response")
    return text


def _call(**kwargs):
    import anthropic

    try:
        return _client().messages.create(model=get_settings().claude_model, **kwargs)
    except anthropic.APIStatusError as e:  # 4xx/5xx after the SDK's own retries
        raise LLMError(f"Claude API {e.status_code}: {e.message}") from None
    except anthropic.APIConnectionError:
        raise LLMError("Claude API unreachable") from None


def json_call(system: str, user: str, schema: dict, *, max_tokens: int = 4000, effort: str = "low") -> dict:
    """One request whose answer is JSON guaranteed to match `schema` (structured outputs)."""
    r = _call(max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
              output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    try:
        return json.loads(_text(r))
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON from model: {e}") from None


def text_call(system: str, user: str, *, max_tokens: int = 8000, effort: str = "medium") -> str:
    r = _call(max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
              output_config={"effort": effort})
    return _text(r)
