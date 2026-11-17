"""Monthly trend switch vs. buy-and-hold, with the xStocks spread charged.

Rule: at each month's last close, hold the asset if its close is above the
average of the last `months` month-end closes, otherwise hold cash. The order
goes in the next trading day and fills at that day's close, so the new
position earns returns from the day after - the decision never uses a price
it could not have seen.

Costs: `spread` per side on every change of position, including the first
purchase (buy-and-hold pays it once).
"""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd

TRADING_DAYS = 252


def month_end_closes(px: pd.Series) -> pd.Series:
    return px.groupby([px.index.year, px.index.month]).tail(1)


def trend_signal(months: int) -> Callable[[pd.Series], pd.Series]:
    def sig(me: pd.Series) -> pd.Series:
        return (me > me.rolling(months).mean()).astype(float).where(me.rolling(months).count() == months)
    sig.__name__ = f"trend_{months}m"
    return sig


def always_in(me: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=me.index)


def run(px: pd.Series, signal: Callable[[pd.Series], pd.Series], spread: float = 0.01,
        cash_rate: pd.Series | None = None) -> pd.DataFrame:
    """Daily frame: position, gross asset return, cash return, cost, strategy return, equity.
    `cash_rate`: annual % yield (e.g. FRED DTB3), forward-filled; None = cash earns 0."""
    px = px.dropna().astype(float)
    me = month_end_closes(px)
    target = signal(me).reindex(px.index).ffill()
    # decided at close t, filled at close t+1, earns from t+2
    pos = target.shift(2)
    first = pos.first_valid_index()
    if first is None:
        raise ValueError("not enough history for the signal")
    df = pd.DataFrame({"px": px, "pos": pos}).loc[first:]
    df["asset_ret"] = df["px"].pct_change().fillna(0.0)
    if cash_rate is not None:
        cr = cash_rate.reindex(df.index).ffill().fillna(0.0)
        df["cash_ret"] = cr / 100.0 / TRADING_DAYS
    else:
        df["cash_ret"] = 0.0
    prev = df["pos"].shift(1).fillna(0.0)          # start in cash: the first buy pays the spread
    df["cost"] = (df["pos"] - prev).abs() * spread
    # on the first day of a position the fill was at the prior close, so the day's
    # return already belongs to the new position
    df["ret"] = df["pos"] * df["asset_ret"] + (1 - df["pos"]) * df["cash_ret"] - df["cost"]
    df.iloc[0, df.columns.get_loc("ret")] = -df["cost"].iloc[0]   # no return before the first fill
    df["equity"] = (1 + df["ret"]).cumprod()
    return df


def metrics(df: pd.DataFrame) -> dict:
    r = df["ret"]
    years = len(r) / TRADING_DAYS
    eq = df["equity"]
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    vol = r.std() * math.sqrt(TRADING_DAYS)
    dd = (eq / eq.cummax() - 1).min()
    switches = int((df["pos"].diff().abs() > 0).sum()) + int(df["pos"].iloc[0] != 0)
    return {
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "years": round(years, 1),
        "cagr": cagr,
        "vol": vol,
        "sharpe0": (r.mean() * TRADING_DAYS) / vol if vol > 0 else float("nan"),
        "max_drawdown": float(dd),
        "time_in_market": float(df["pos"].mean()),
        "trades": switches,
        "trades_per_year": switches / years if years else float("nan"),
        "total_cost": float(df["cost"].sum()),
        "final_multiple": float(eq.iloc[-1]),
    }


def rebase(df: pd.DataFrame, start, spread: float) -> pd.DataFrame:
    """The same run as if the account opened, in cash, on `start`: a position
    held that day is bought then and pays the spread, so every strategy in a
    comparison starts from the same place."""
    d = df.loc[start:].copy()
    d.iloc[0, d.columns.get_loc("cost")] = abs(d["pos"].iloc[0]) * spread
    d.iloc[0, d.columns.get_loc("ret")] = -d["cost"].iloc[0]
    d["equity"] = (1 + d["ret"]).cumprod()
    return d


def compare(px: pd.Series, months_list=(6, 8, 10, 12), spread: float = 0.01,
            cash_rate: pd.Series | None = None) -> pd.DataFrame:
    """Buy-and-hold and each trend window over the same dates."""
    runs = {"buy_and_hold": run(px, always_in, spread, cash_rate)}
    runs |= {f"trend_{m}m": run(px, trend_signal(m), spread, cash_rate) for m in months_list}
    start = max(d.index[0] for d in runs.values())
    return pd.DataFrame({name: metrics(rebase(d, start, spread)) for name, d in runs.items()}).T


def subperiods(px: pd.Series, months: int, spread: float, cash_rate=None, years: int = 5) -> pd.DataFrame:
    """Trend vs buy-and-hold CAGR in consecutive blocks: does it win often, or once?"""
    tr = run(px, trend_signal(months), spread, cash_rate)
    bh = rebase(run(px, always_in, spread, cash_rate), tr.index[0], spread)
    out = []
    for y0 in range(tr.index[0].year, tr.index[-1].year + 1, years):
        a, b = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0 + years}-01-01")
        t, h = tr.loc[a:b - pd.Timedelta(days=1), "ret"], bh.loc[a:b - pd.Timedelta(days=1), "ret"]
        if len(t) < TRADING_DAYS // 2:
            continue
        ann = lambda r: (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        dd = lambda r: float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        out.append({"period": f"{y0}-{min(y0 + years - 1, tr.index[-1].year)}",
                    "trend_cagr": ann(t), "hold_cagr": ann(h),
                    "trend_maxdd": dd(t), "hold_maxdd": dd(h)})
    return pd.DataFrame(out)

