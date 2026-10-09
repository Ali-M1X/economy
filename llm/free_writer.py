"""Optional free text generation for the experimental advisory report (any OpenAI-compatible chat endpoint).

- With the secrets LLM_API_KEY (+ LLM_BASE_URL, LLM_MODEL) it uses that provider, e.g. a free OpenRouter key
  (LLM_BASE_URL=https://openrouter.ai/api/v1, LLM_MODEL=deepseek/deepseek-chat-v3-0324:free) or a free Gemini key
  (LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai, LLM_MODEL=gemini-2.5-flash).
- Otherwise it tries GitHub Models with the workflow's GITHUB_TOKEN (permissions: models: read). In October 2026 that
  endpoint answered every request with a bare "OK", so in practice the template text is used.
Every call can fail and raises LLMError; callers keep their template text.
"""

from __future__ import annotations

import os

import requests

from llm.claude import LLMError

# (endpoint, model ids): the current GitHub Models endpoint first, then the older Azure-hosted one that also accepts a
# GitHub token and names models without the publisher prefix.
ENDPOINTS = (("https://models.github.ai/inference/chat/completions", ("openai/gpt-4.1", "openai/gpt-4o", "openai/gpt-4.1-mini")),
             ("https://models.inference.ai.azure.com/chat/completions", ("gpt-4.1", "gpt-4o", "gpt-4o-mini")))


def _tries() -> list[tuple[str, str, str]]:
    """(url, model, key) in the order to try."""
    if os.environ.get("LLM_API_KEY"):
        base = (os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        model = os.environ.get("LLM_MODEL") or "deepseek/deepseek-chat-v3-0324:free"
        return [(f"{base}/chat/completions", model, os.environ["LLM_API_KEY"])]
    tok = os.environ.get("GITHUB_TOKEN")
    return [(url, m, tok) for url, models in ENDPOINTS for m in models] if tok else []


def available() -> bool:
    return bool(_tries())


def chat(system: str, user: str, *, max_tokens: int = 3000, session: requests.Session | None = None) -> tuple[str, str]:
    """(text, model) from the first model that answers."""
    tries = _tries()
    if not tries:
        raise LLMError("no LLM_API_KEY or GITHUB_TOKEN")
    http = session or requests.Session()
    errors = []
    for url, model, token in tries:
        try:
            r = http.post(url, timeout=120, headers={"Authorization": f"Bearer {token}",
                                                     "Accept": "application/vnd.github+json",
                                                     "X-GitHub-Api-Version": "2022-11-28",
                                                     "Content-Type": "application/json"},
                          json={"model": model, "max_tokens": max_tokens, "temperature": 0.4,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            if r.status_code != 200:
                errors.append(f"{model}: HTTP {r.status_code} {r.text[:120]}")
                continue
            try:
                choice = r.json()["choices"][0]
            except ValueError:
                errors.append(f"{model}: not JSON ({r.headers.get('content-type')}, {r.text[:60]!r})")
                continue
            text = (choice.get("message") or {}).get("content") or ""
            if choice.get("finish_reason") == "length" or not text.strip():
                errors.append(f"{model}: truncated or empty")
                continue
            return text.strip(), model
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            errors.append(f"{model}: {type(e).__name__}")
    raise LLMError("; ".join(errors) or "no model answered")


def probe() -> None:
    """Print what the endpoints answer (python -m llm.free_writer), for diagnosing the free tier from Actions."""
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"model": "openai/gpt-4.1-mini", "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 20}
    for name, call in (
            ("catalog", lambda: requests.get("https://models.github.ai/catalog/models", headers=h, timeout=30)),
            ("plain", lambda: requests.post(ENDPOINTS[0][0], headers=h, json=body, timeout=60)),
            ("stream-off", lambda: requests.post(ENDPOINTS[0][0], headers={**h, "Accept": "application/json"},
                                                 json={**body, "stream": False}, timeout=60)),
            ("no-redirect", lambda: requests.post(ENDPOINTS[0][0], headers=h, json=body, timeout=60,
                                                  allow_redirects=False))):
        try:
            r = call()
            hist = [f"{x.status_code}->{x.headers.get('location')}" for x in r.history]
            print(f"::notice title=probe {name}::{r.status_code} {r.headers.get('content-type')} hist={hist} "
                  f"hdr={ {k: v for k, v in r.headers.items() if k.lower().startswith(('x-', 'server', 'retry'))} } "
                  f"body={r.text[:300]!r}".replace("\n", " "))
        except requests.RequestException as e:
            print(f"::notice title=probe {name}::{type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    probe()
