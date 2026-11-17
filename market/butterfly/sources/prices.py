"""Daily prices, cached as one CSV per ticker: date, close, adj_close, source.

Sources, tried in order:
  tiingo  free account key (tiingo.com), dividend- and split-adjusted history
          back to the 1990s. Key from TIINGO_API_KEY; in the cloud dev
          container the network proxy adds it instead.
  yahoo   no key, adjusted - but refuses most scripts now (HTTP 429)
  stooq   no key, adjustment undocumented, so its rows carry source=stooq
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import pandas as pd

from .. import config
from ..http import FetchError, get

COLUMNS = ["date", "close", "adj_close", "source"]


def _path(ticker: str):
    return config.PRICES_DIR / f"{ticker.upper()}.csv"


def parse_tiingo(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame({"date": [r["date"][:10] for r in rows],
                       "close": [r["close"] for r in rows],
                       "adj_close": [r["adjClose"] for r in rows]})
    df["source"] = "tiingo"
    return df.dropna(subset=["close", "adj_close"])


def fetch_tiingo(ticker: str) -> pd.DataFrame:
    key = os.environ.get("TIINGO_API_KEY")
    headers = {"Authorization": f"Token {key}"} if key else None
    # Tiingo writes BRK-B style tickers with a dash
    r = get(f"https://api.tiingo.com/tiingo/daily/{ticker.lower().replace('.', '-')}/prices",
            params={"startDate": "1990-01-01", "format": "json"}, headers=headers)
    return parse_tiingo(r.json())


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    res = payload["chart"]["result"][0]
    offset = res["meta"].get("gmtoffset", 0)
    ts = res.get("timestamp") or []
    close = res["indicators"]["quote"][0]["close"]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", close)
    dates = [datetime.fromtimestamp(t + offset, tz=timezone.utc).date().isoformat() for t in ts]
    df = pd.DataFrame({"date": dates, "close": close, "adj_close": adj})
    df["source"] = "yahoo"
    return df.dropna(subset=["close", "adj_close"]).drop_duplicates("date", keep="last")


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = get(url, params={"period1": 0, "period2": 9_999_999_999, "interval": "1d",
                         "events": "div,splits", "includeAdjustedClose": "true"}, tries=1)
    return parse_yahoo_chart(r.json())


def parse_stooq_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    if "Close" not in raw.columns:
        raise FetchError(f"stooq returned no prices: {text[:80]!r}")
    df = pd.DataFrame({"date": raw["Date"], "close": raw["Close"], "adj_close": raw["Close"]})
    df["source"] = "stooq"
    return df


def fetch_stooq(ticker: str) -> pd.DataFrame:
    r = get("https://stooq.com/q/d/l/", params={"s": f"{ticker.lower()}.us", "i": "d"}, tries=1)
    return parse_stooq_csv(r.text)


def update(ticker: str) -> tuple[int, str]:
    """Refresh one ticker's cache. Returns (rows, source). Raises FetchError when
    every source fails, leaving the cache as it was."""
    errors = []
    for fetch in (fetch_tiingo, fetch_yahoo, fetch_stooq):
        try:
            df = fetch(ticker)
        except (FetchError, KeyError, IndexError, TypeError, ValueError) as e:
            errors.append(f"{fetch.__name__}: {e}")
            continue
        if len(df):
            config.PRICES_DIR.mkdir(parents=True, exist_ok=True)
            df[COLUMNS].to_csv(_path(ticker), index=False)
            return len(df), df["source"].iloc[-1]
    raise FetchError(f"{ticker}: " + "; ".join(errors))


def load(ticker: str) -> pd.DataFrame:
    """Cached prices indexed by date (DatetimeIndex), ascending."""
    p = _path(ticker)
    if not p.exists():
        raise FileNotFoundError(f"no cached prices for {ticker}; run `python -m butterfly fetch prices`")
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df


def adj_close(ticker: str) -> pd.Series:
    return load(ticker)["adj_close"].rename(ticker.upper())
