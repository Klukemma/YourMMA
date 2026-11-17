"""World news from the GDELT DOC API (api.gdeltproject.org), free, no key.

Articles are appended to one JSONL file per UTC day, de-duplicated by URL,
each stamped with the time GDELT first saw it (`seen`) and the time we
fetched it (`fetched`). Reasoning may only use an article whose `fetched`
time is before the prediction it supports.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .. import config
from ..http import get

# Standing queries: themes whose effects travel through supply chains,
# currencies and commodities. Extend freely.
QUERIES = {
    "shipping": '("panama canal" OR "suez canal" OR "red sea shipping" OR "freight rates")',
    "energy": '(opec OR "oil supply" OR "natural gas prices" OR "refinery outage")',
    "rates": '("federal reserve" OR "interest rate decision" OR "inflation report")',
    "trade": '(tariff OR "export ban" OR "trade restrictions" OR sanctions)',
    "chips": '(semiconductor OR "chip export" OR tsmc OR "chip shortage")',
    "weather": '(drought OR hurricane OR "crop failure" OR "heat wave") AND (supply OR prices)',
    "fx": '("dollar index" OR "yen intervention" OR "currency devaluation")',
}


def parse_gdelt(payload: dict, query_name: str, fetched: str) -> list[dict]:
    out = []
    for a in payload.get("articles", []):
        seen = a.get("seendate", "")
        try:
            seen = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
        out.append({
            "url": a.get("url"),
            "title": a.get("title"),
            "domain": a.get("domain"),
            "country": a.get("sourcecountry"),
            "language": a.get("language"),
            "seen": seen,
            "fetched": fetched,
            "query": query_name,
        })
    return [x for x in out if x["url"]]


# GDELT allows one request every 5 seconds per address and answers 429 faster.
_PAUSE = 6.0
_LAST = [0.0]


def fetch(query_name: str, timespan: str = "1d", maxrecords: int = 75) -> list[dict]:
    wait = _PAUSE - (time.monotonic() - _LAST[0])
    if wait > 0:
        time.sleep(wait)
    _LAST[0] = time.monotonic()
    r = get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": f"{QUERIES[query_name]} sourcelang:english",
        "mode": "artlist", "format": "json", "maxrecords": maxrecords,
        "timespan": timespan, "sort": "datedesc",
    }, tries=2)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return parse_gdelt(r.json(), query_name, fetched)


def append(articles: list[dict]) -> int:
    """Append new articles to today's file; returns how many were new."""
    config.NEWS_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).date().isoformat()
    path = config.NEWS_DIR / f"{day}.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["url"] for line in path.read_text().splitlines() if line.strip()}
    lines = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            lines.append(json.dumps(a, ensure_ascii=False) + "\n")
    if lines:
        with path.open("a") as f:
            f.writelines(lines)
    return len(lines)
