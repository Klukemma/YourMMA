"""Settings shared by every module. Nothing here is secret.

Override with environment variables rather than editing, so a run's settings
are visible in the command that started it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # market/


def _load_dotenv(path: Path) -> None:
    """KEY=value lines from market/.env (gitignored); real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
DATA = Path(os.environ.get("BUTTERFLY_DATA", ROOT / "data"))
PRICES_DIR = DATA / "prices"
MACRO_DIR = DATA / "macro"
FX_DIR = DATA / "fx"
FILINGS_DIR = DATA / "filings"
NEWS_DIR = DATA / "news"
LEDGER_PATH = Path(os.environ.get("BUTTERFLY_LEDGER", DATA / "ledger.jsonl"))
GRAPH_PATH = Path(os.environ.get("BUTTERFLY_GRAPH", DATA / "graph.json"))

# Kraken builds a 1.0% spread into every xStock purchase price (kraken.com/legal/xstocks).
# Assumed the same on the way out until measured, so a round trip costs ~2%.
SPREAD_PER_SIDE = float(os.environ.get("BUTTERFLY_SPREAD", "0.01"))

# Benchmark every prediction is measured against.
BENCHMARK = "SPY"

# SEC requires automated clients to name themselves *and a contact email* in
# the User-Agent (sec.gov/os/accessing-edgar-data) and answers 403 otherwise.
# There is deliberately no default: set e.g.
#   SEC_USER_AGENT="Your Name your@email"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT") or None
# Everything else is sent with the HTTP library's own User-Agent; some sites
# (FRED) refuse unfamiliar custom names.
HTTP_USER_AGENT = os.environ.get("HTTP_USER_AGENT") or None

# Underlying tickers we expect to trade as xStocks. Kraken's list changes;
# check it before trading a name that is not here (kraken.com xStocks page).
UNIVERSE = [
    "SPY", "QQQ", "GLD", "TLT",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "XOM", "CVX", "KO", "PG", "WMT", "COST", "UNH", "LLY",
    "AMD", "INTC", "AVGO", "NFLX", "PFE", "MRK", "BA", "DIS", "MCD", "CRM",
]

# Macro series pulled from FRED (fred.stlouisfed.org). Free, no key.
FRED_SERIES = {
    "DGS10": "10-year Treasury yield",
    "DGS2": "2-year Treasury yield",
    "DTB3": "3-month Treasury bill",
    "DCOILWTICO": "WTI crude oil",
    "DCOILBRENTEU": "Brent crude oil",
    "DHHNGSP": "Henry Hub natural gas",
    "DTWEXBGS": "Trade-weighted US dollar (broad)",
    "VIXCLS": "VIX",
    "BAMLH0A0HYM2": "High-yield credit spread",
    "T10YIE": "10-year breakeven inflation",
    "SP500": "S&P 500 index (price only, ~10 years)",
}

FX_BASE = "USD"
FX_SYMBOLS = ["EUR", "JPY", "GBP", "CNY", "CAD", "MXN", "BRL", "CHF", "AUD", "INR", "KRW"]
