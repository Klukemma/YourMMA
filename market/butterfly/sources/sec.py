"""SEC EDGAR: ticker -> CIK, each company's recent filings, and filing text.

8-K filings are the most reliable free "news" there is: the company itself
reports the event, and EDGAR stamps the exact acceptance time, so there is
never a doubt about when the market could have known.

SEC fair-access rules: identify yourself (config.SEC_USER_AGENT) and stay
under 10 requests a second; this module sleeps between calls.
"""
from __future__ import annotations

import html
import json
import re
import time
from html.parser import HTMLParser

from .. import config
from ..http import FetchError, get

_LAST_CALL = [0.0]

# 8-K item codes worth reasoning about, with a plain name.
ITEMS_8K = {
    "1.01": "material agreement entered",
    "1.02": "material agreement terminated",
    "1.05": "cybersecurity incident",
    "2.01": "acquisition or disposal completed",
    "2.02": "results of operations",
    "2.03": "new material debt",
    "2.05": "restructuring / exit costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "4.01": "auditor change",
    "4.02": "prior financials unreliable (restatement)",
    "5.02": "director/officer change",
    "7.01": "Reg FD disclosure",
    "8.01": "other events",
}


def _sec_get(url: str):
    if not config.SEC_USER_AGENT:
        raise FetchError('SEC_USER_AGENT is not set. SEC requires a name and contact email, '
                         'e.g. SEC_USER_AGENT="Your Name you@example.org" (see .env.example)')
    wait = 0.15 - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.monotonic()
    return get(url, user_agent=config.SEC_USER_AGENT)


def ticker_map(refresh: bool = False) -> dict[str, dict]:
    """{'AAPL': {'cik': 320193, 'title': 'Apple Inc.'}, ...}"""
    path = config.FILINGS_DIR / "company_tickers.json"
    if refresh or not path.exists():
        raw = _sec_get("https://www.sec.gov/files/company_tickers.json").json()
        out = {v["ticker"].upper(): {"cik": int(v["cik_str"]), "title": v["title"]} for v in raw.values()}
        config.FILINGS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=0, sort_keys=True))
    return json.loads(path.read_text())


def parse_recent_filings(payload: dict, forms: set[str] | None = None) -> list[dict]:
    rec = payload["filings"]["recent"]
    cik = int(payload["cik"])
    out = []
    for i, form in enumerate(rec["form"]):
        if forms and form not in forms:
            continue
        acc = rec["accessionNumber"][i]
        doc = rec["primaryDocument"][i]
        items = [x.strip() for x in (rec.get("items", [""] * len(rec["form"]))[i] or "").split(",") if x.strip()]
        out.append({
            "cik": cik,
            "form": form,
            "accession": acc,
            "filed": rec["filingDate"][i],
            "accepted": rec["acceptanceDateTime"][i],     # exact time EDGAR accepted it
            "items": items,
            "item_names": [ITEMS_8K.get(x, x) for x in items],
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
        })
    return out


def recent_filings(ticker: str, forms: set[str] | None = frozenset({"8-K", "10-K", "10-Q"})) -> list[dict]:
    info = ticker_map().get(ticker.upper())
    if info is None:
        return []                                   # ETFs and foreign issuers have no 10-K/8-K here
    r = _sec_get(f"https://data.sec.gov/submissions/CIK{info['cik']:010d}.json")
    rows = parse_recent_filings(r.json(), set(forms) if forms else None)
    for row in rows:
        row["ticker"] = ticker.upper()
    return rows


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(doc: str) -> str:
    p = _Text()
    p.feed(doc)
    text = html.unescape("".join(p.parts)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def filing_text(url: str) -> str:
    return html_to_text(_sec_get(url).text)


def section(text: str, start: str, end: str) -> str:
    """The longest stretch between a heading matching `start` and the next
    matching `end` - the longest, because the table of contents names every
    item too and is the short match."""
    best = ""
    for m in re.finditer(start, text, flags=re.I):
        e = re.search(end, text[m.end():], flags=re.I)
        chunk = text[m.start(): m.end() + (e.start() if e else len(text))]
        if len(chunk) > len(best):
            best = chunk
    return best.strip()


def risk_factors(text: str) -> str:
    return section(text, r"item\s*1a\.?\s*risk\s*factors", r"item\s*1b\.?|item\s*2\.?\s*properties")


def business(text: str) -> str:
    return section(text, r"item\s*1\.?\s*business", r"item\s*1a\.?")
