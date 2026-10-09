"""Free text generation through GitHub Models, used to make the experimental advisory report read like an analyst.

Inside GitHub Actions the workflow's own GITHUB_TOKEN works as the key when the job has `permissions: models: read`,
so there is no account, key or bill (the free tier allows a few dozen requests a day; the report needs three).
Every call can fail (no token, rate limit, network) and raises LLMError; callers keep their template text.
"""

from __future__ import annotations

import os

import requests

from llm.claude import LLMError

URL = "https://models.github.ai/inference/chat/completions"
MODELS = ("openai/gpt-4.1", "openai/gpt-4o", "openai/gpt-4.1-mini")


def available() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN")) and os.environ.get("ADVISOR_LLM", "github") == "github"


def chat(system: str, user: str, *, max_tokens: int = 3000, session: requests.Session | None = None) -> tuple[str, str]:
    """(text, model) from the first model that answers."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise LLMError("GITHUB_TOKEN not set")
    http = session or requests.Session()
    errors = []
    for model in [os.environ["ADVISOR_MODEL"]] if os.environ.get("ADVISOR_MODEL") else MODELS:
        try:
            r = http.post(URL, timeout=120, headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                                                     "Content-Type": "application/json"},
                          json={"model": model, "max_tokens": max_tokens, "temperature": 0.4,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:120]}")
                continue
            try:
                choice = r.json()["choices"][0]
            except ValueError:
                errors.append(f"{model}: not JSON ({r.headers.get('content-type')}, {r.url}, {r.text[:150]!r})")
                continue
            text = (choice.get("message") or {}).get("content") or ""
            if choice.get("finish_reason") == "length" or not text.strip():
                errors.append(f"{model}: truncated or empty")
                continue
            return text.strip(), model
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            errors.append(f"{model}: {type(e).__name__}")
    raise LLMError("; ".join(errors) or "no model answered")
