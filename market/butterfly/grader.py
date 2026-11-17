"""Grade ledger predictions against what the market did afterwards.

Entry rule (no look-ahead): the entry price is the close of the first trading
day whose date (US Eastern) is *after* the day the prediction was made. A
prediction written at 10:00 on Monday enters at Tuesday's close - it never
gets Monday's close, which may already have been moving on the same news.

Exit: the close `horizon_days` trading days after entry.

Scores per prediction:
  abnormal     stock return - benchmark return over the holding window
  signed       direction * abnormal (positive = the call was right)
  hit          signed > 0
  brier        (confidence - hit)^2; 0.25 is what always saying 0.5 earns
  long_net     for +1 calls only: stock return - two spreads, the realistic
               xStocks trade (they cannot be shorted, so -1 calls are "avoid")

Baselines on exactly the same (ticker, entry, horizon) windows:
  always_long  direction +1 every time
  momentum     direction = sign of the stock's abnormal return over the
               21 trading days before entry
A real edge has to beat both, not just 50%.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

NY = ZoneInfo("America/New_York")


def entry_index(trading_days: pd.DatetimeIndex, created_at: str) -> int | None:
    t = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    made_on = pd.Timestamp(t.astimezone(NY).date())
    pos = int(trading_days.searchsorted(made_on, side="right"))
    return pos if pos < len(trading_days) else None


def grade_one(pred: dict, prices: pd.DataFrame, spread: float = config.SPREAD_PER_SIDE) -> dict | None:
    """`prices`: adjusted closes, one column per ticker incl. the benchmark,
    indexed by trading date. None while the window has not closed yet."""
    tk, bm = pred["ticker"], pred.get("benchmark", config.BENCHMARK)
    if tk not in prices or bm not in prices:
        return None
    px = prices[[tk, bm]].dropna()
    i = entry_index(px.index, pred["created_at"])
    if i is None or i + pred["horizon_days"] >= len(px):
        return None
    j = i + pred["horizon_days"]
    r_stock = px[tk].iloc[j] / px[tk].iloc[i] - 1
    r_bm = px[bm].iloc[j] / px[bm].iloc[i] - 1
    abnormal = r_stock - r_bm
    signed = pred["direction"] * abnormal
    hit = signed > 0
    # momentum baseline: sign of abnormal over the 21 days before entry
    mom = None
    if i >= 21:
        prior = (px[tk].iloc[i] / px[tk].iloc[i - 21]) - (px[bm].iloc[i] / px[bm].iloc[i - 21])
        mom = 1 if prior > 0 else -1
    return {
        "id": pred["id"],
        "ticker": tk,
        "direction": pred["direction"],
        "horizon_days": pred["horizon_days"],
        "confidence": pred["confidence"],
        "entry_date": px.index[i].date().isoformat(),
        "exit_date": px.index[j].date().isoformat(),
        "r_stock": r_stock,
        "r_benchmark": r_bm,
        "abnormal": abnormal,
        "signed": signed,
        "hit": bool(hit),
        "brier": (pred["confidence"] - float(hit)) ** 2,
        "long_net": (r_stock - 2 * spread) if pred["direction"] == 1 else None,
        "base_long_signed": abnormal,
        "base_mom_signed": (mom * abnormal) if mom is not None else None,
    }


def grade(preds: list[dict], prices: pd.DataFrame, spread: float = config.SPREAD_PER_SIDE):
    graded, pending = [], []
    for p in preds:
        g = grade_one(p, prices, spread)
        (graded if g else pending).append(g or p)
    return pd.DataFrame(graded), pending


def _t(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def summarize(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n": 0}
    longs = g["long_net"].dropna()
    out = {
        "n": int(len(g)),
        "hit_rate": float(g["hit"].mean()),
        "mean_signed_abnormal": float(g["signed"].mean()),
        "t_stat": _t(g["signed"]),
        "brier": float(g["brier"].mean()),
        "brier_coinflip": 0.25,
        "long_calls": int(len(longs)),
        "long_net_mean": float(longs.mean()) if len(longs) else None,
        "baseline_always_long": float(g["base_long_signed"].mean()),
        "baseline_momentum": float(g["base_mom_signed"].dropna().mean()) if g["base_mom_signed"].notna().any() else None,
    }
    out["beats_baselines"] = bool(
        out["mean_signed_abnormal"] > out["baseline_always_long"]
        and (out["baseline_momentum"] is None or out["mean_signed_abnormal"] > out["baseline_momentum"])
    )
    buckets = pd.cut(g["confidence"], [0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    out["calibration"] = [
        {"confidence": str(b), "n": int(len(s)), "said": float(s["confidence"].mean()), "hit": float(s["hit"].mean())}
        for b, s in g.groupby(buckets, observed=True)
    ]
    out["by_horizon"] = {
        int(h): {"n": int(len(s)), "hit_rate": float(s["hit"].mean()), "mean_signed": float(s["signed"].mean())}
        for h, s in g.groupby("horizon_days")
    }
    return out


def load_prices(tickers: set[str]) -> pd.DataFrame:
    from .sources import prices as src
    cols = {}
    for t in sorted(tickers | {config.BENCHMARK}):
        try:
            cols[t] = src.adj_close(t)
        except FileNotFoundError:
            pass
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
