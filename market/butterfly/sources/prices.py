"""Daily prices, cached as one CSV per ticker: date, close, adj_close, source.

`adj_close` is adjusted for splits and dividends where the source says so
(Yahoo's adjclose). Stooq is a fallback whose adjustment is not documented,
so its rows carry source=stooq and the backtest says so when it uses them.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd

from .. import config
from ..http import FetchError, get

COLUMNS = ["date", "close", "adj_close", "source"]


def _path(ticker: str):
    return config.PRICES_DIR / f"{ticker.upper()}.csv"


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
                         "events": "div,splits", "includeAdjustedClose": "true"})
    return parse_yahoo_chart(r.json())


def parse_stooq_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    if "Close" not in raw.columns:
        raise FetchError(f"stooq returned no prices: {text[:80]!r}")
    df = pd.DataFrame({"date": raw["Date"], "close": raw["Close"], "adj_close": raw["Close"]})
    df["source"] = "stooq"
    return df


def fetch_stooq(ticker: str) -> pd.DataFrame:
    r = get("https://stooq.com/q/d/l/", params={"s": f"{ticker.lower()}.us", "i": "d"})
    return parse_stooq_csv(r.text)


def update(ticker: str) -> tuple[int, str]:
    """Refresh one ticker's cache. Returns (rows, source). Raises FetchError when
    every source fails, leaving the cache as it was."""
    errors = []
    for fetch in (fetch_yahoo, fetch_stooq):
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
