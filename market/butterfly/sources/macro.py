"""FRED macro series and Frankfurter (ECB) FX rates. Both free, no key."""
from __future__ import annotations

import io

import pandas as pd

from .. import config
from ..http import get


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    # FRED writes "." for a missing day
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = series
    return df.dropna(subset=["value"])


def update_fred(series: str) -> int:
    r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series})
    df = parse_fred_csv(r.text, series)
    config.MACRO_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MACRO_DIR / f"{series}.csv", index=False)
    return len(df)


def load_fred(series: str) -> pd.Series:
    df = pd.read_csv(config.MACRO_DIR / f"{series}.csv", parse_dates=["date"])
    return df.set_index("date")["value"].rename(series)


def parse_frankfurter(payload: dict) -> pd.DataFrame:
    rows = [{"date": d, **rates} for d, rates in payload["rates"].items()]
    return pd.DataFrame(rows).sort_values("date")


def update_fx(start: str = "1999-01-04") -> int:
    """Daily ECB reference rates, quoted as units of each currency per USD."""
    syms = ",".join(config.FX_SYMBOLS)
    r = get(f"https://api.frankfurter.dev/v1/{start}..",
            params={"base": config.FX_BASE, "symbols": syms})
    df = parse_frankfurter(r.json())
    config.FX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FX_DIR / f"{config.FX_BASE}.csv", index=False)
    return len(df)


def load_fx() -> pd.DataFrame:
    return pd.read_csv(config.FX_DIR / f"{config.FX_BASE}.csv", parse_dates=["date"]).set_index("date")
