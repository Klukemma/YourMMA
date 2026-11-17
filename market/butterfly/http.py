"""One polite HTTP getter: honest User-Agent, timeout, a few retries on
rate limits and server errors, never on client errors."""
from __future__ import annotations

import time

import requests

from . import config


class FetchError(RuntimeError):
    pass


def get(url: str, *, params: dict | None = None, user_agent: str | None = None,
        tries: int = 3, timeout: float = 30.0) -> requests.Response:
    ua = user_agent or config.HTTP_USER_AGENT
    headers = {"User-Agent": ua} if ua else {}
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:          # network reset, timeout
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        time.sleep(2 * (attempt + 1))
    raise FetchError(f"{url}: {last}")
