"""Talking to the models, through any OpenAI-compatible endpoint (OpenRouter by default)."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("llm")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
# Long waits on purpose: a task can run for days, and an outage should stall
# it rather than kill it.
BACKOFF = [5, 15, 45, 120, 300, 600]


class LLMError(Exception):
    pass


@dataclass
class Reply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    cost: float = 0.0
    message: dict = field(default_factory=dict)  # raw assistant message, to append to history


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30))
    return _client


def _cost(agent: dict, usage: dict) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    price = agent.get("price") or {}
    return (usage.get("prompt_tokens", 0) * float(price.get("input_per_m", 0))
            + usage.get("completion_tokens", 0) * float(price.get("output_per_m", 0))) / 1_000_000


async def chat(agent: dict, messages: list[dict], tools: list[dict] | None = None,
               plugins: list[dict] | None = None, model: str | None = None) -> Reply:
    key = os.environ.get(agent.get("api_key_env", "OPENROUTER_API_KEY"), "")
    if not key:
        raise LLMError(f"No API key: set {agent.get('api_key_env', 'OPENROUTER_API_KEY')}")
    payload: dict = {
        "model": model or agent["model"],
        "messages": messages,
        "usage": {"include": True},
        **(agent.get("params") or {}),
    }
    if tools:
        payload["tools"] = tools
    if plugins:
        payload["plugins"] = plugins
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/klukemma/yourmma",
        "X-Title": "AI Orchestrator",
    }
    url = agent.get("base_url", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"

    last_err = ""
    for attempt, wait in enumerate([0, *BACKOFF]):
        if wait:
            log.warning("retrying %s in %ss (%s)", payload["model"], wait, last_err)
            await asyncio.sleep(wait)
        try:
            resp = await client().post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last_err = f"network error: {exc}"
            continue
        if resp.status_code in RETRY_STATUS:
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            continue
        if resp.status_code >= 400:
            raise LLMError(f"{payload['model']}: HTTP {resp.status_code}: {resp.text[:800]}")
        data = resp.json()
        if data.get("error"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            last_err = str(err)[:500]
            if isinstance(code, int) and code in RETRY_STATUS:
                continue
            raise LLMError(f"{payload['model']}: {last_err}")
        choices = data.get("choices") or []
        if not choices:
            last_err = "empty response"
            continue
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):  # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        clean = {"role": "assistant", "content": content}
        if msg.get("tool_calls"):
            clean["tool_calls"] = msg["tool_calls"]
        annotations = msg.get("annotations") or []
        if annotations:
            sources = [a.get("url_citation", {}).get("url") for a in annotations if a.get("type") == "url_citation"]
            sources = [s for s in sources if s]
            if sources:
                content += "\n\nSources:\n" + "\n".join(f"- {s}" for s in dict.fromkeys(sources))
        return Reply(content=content, tool_calls=msg.get("tool_calls") or [],
                     cost=_cost(agent, data.get("usage") or {}), message=clean)
    raise LLMError(f"{payload['model']}: gave up after retries ({last_err})")
