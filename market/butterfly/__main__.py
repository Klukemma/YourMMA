"""Command line.  Run from market/:  python -m butterfly <command>

  fetch prices|macro|fx|filings|news|all   refresh the data caches
  backtest [--ticker SPY]                  trend switch vs buy-and-hold, spread charged
  ripple <node> [--shock -1] [--hops 3]    what an event moves, through the dependency map
  nodes                                    list the map's nodes
  predict --ticker X --direction 1 ...     write a prediction to the ledger
  verify                                   check the ledger was never edited
  grade                                    score every prediction whose window has closed
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config


def _fetch(what: str) -> int:
    from .http import FetchError
    from .sources import macro, news, prices, sec
    failed = 0

    def attempt(label, fn):
        nonlocal failed
        try:
            print(f"  {label}: {fn()}")
        except (FetchError, OSError, ValueError, KeyError) as e:
            failed += 1
            print(f"  {label}: FAILED {e}", file=sys.stderr)

    if what in ("prices", "all"):
        print("prices")
        for t in config.UNIVERSE:
            attempt(t, lambda t=t: "%d rows from %s" % prices.update(t))
    if what in ("macro", "all"):
        print("macro (FRED)")
        for s in config.FRED_SERIES:
            attempt(s, lambda s=s: f"{macro.update_fred(s)} rows")
    if what in ("fx", "all"):
        print("fx (ECB via Frankfurter)")
        attempt(config.FX_BASE, lambda: f"{macro.update_fx()} days")
    if what in ("filings", "all"):
        print("filings (SEC EDGAR)")
        before = failed
        attempt("ticker map", lambda: f"{len(sec.ticker_map(refresh=True))} tickers")
        if failed > before:
            print("  skipping companies: no ticker map", file=sys.stderr)
        rows = [] if failed == before else None

        def one(t):
            got = sec.recent_filings(t)
            rows.extend(got)
            return f"{len(got)} filings" if got else "no SEC filer (ETF or foreign issuer)"

        for t in (config.UNIVERSE if rows is not None else []):
            attempt(t, lambda t=t: one(t))
        if rows:
            config.FILINGS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_json(config.FILINGS_DIR / "recent.jsonl", orient="records", lines=True)
    if what in ("news", "all"):
        print("news (GDELT)")
        for q in news.QUERIES:
            attempt(q, lambda q=q: f"{news.append(news.fetch(q))} new articles")
    return 1 if failed else 0


def _pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.1f}%"


def _backtest(a) -> int:
    from . import backtest as bt
    from .sources import macro, prices
    df = prices.load(a.ticker)
    px = df["adj_close"]
    if (df["source"] != "yahoo").any():
        print(f"note: some {a.ticker} rows come from a source whose dividend adjustment is undocumented")
    cash = None
    if a.cash:
        try:
            cash = macro.load_fred(a.cash)
        except FileNotFoundError:
            print(f"note: {a.cash} not cached, cash earns 0")
    months = [int(m) for m in a.months.split(",")]
    table = bt.compare(px, months, a.spread, cash)
    show = table[["start", "end", "cagr", "vol", "sharpe0", "max_drawdown", "time_in_market", "trades_per_year", "total_cost", "final_multiple"]].copy()
    for c in ("cagr", "vol", "max_drawdown", "time_in_market", "total_cost"):
        show[c] = show[c].map(lambda v: f"{v * 100:.1f}%")
    for c in ("sharpe0", "trades_per_year", "final_multiple"):
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    print(f"{a.ticker}: spread {a.spread:.2%} per side, cash = {a.cash or '0%'}\n")
    print(show.to_string())
    sub = bt.subperiods(px, a.subperiod_months, a.spread, cash)
    print(f"\ntrend_{a.subperiod_months}m vs buy-and-hold in 5-year blocks:")
    print(sub.to_string(index=False, formatters={c: _pct for c in sub.columns if c != "period"}))
    wins = int((sub["trend_cagr"] > sub["hold_cagr"]).sum())
    print(f"\ntrend beat buy-and-hold in {wins} of {len(sub)} blocks")
    return 0


def _ripple(a) -> int:
    from .graph import Graph, describe
    g = Graph.load(config.GRAPH_PATH)
    res = g.ripple(a.node, a.shock, a.hops, a.min_strength, a.verified_only)
    print(f"ripple from {a.node} ({'rise' if a.shock > 0 else 'fall'}), {a.hops} hops:\n")
    for r in res:
        if a.tickers_only and not r["ticker"]:
            continue
        print("  " + describe(g, r))
    return 0


def _predict(a) -> int:
    from .ledger import Ledger
    ev = json.loads(a.evidence) if a.evidence else []
    rec = Ledger(config.LEDGER_PATH).add(ticker=a.ticker, direction=a.direction, horizon_days=a.horizon,
                                         confidence=a.confidence, event=a.event, chain=a.chain,
                                         evidence=ev, expected_move=a.expected_move, author=a.author, tags=a.tag)
    print(f"{rec['id']} written at {rec['created_at']}")
    return 0


def _verify(a) -> int:
    from .ledger import Ledger
    led = Ledger(config.LEDGER_PATH)
    problems = led.verify()
    print(f"{len(led.records())} predictions; " + ("intact" if not problems else "DAMAGED"))
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


def _grade(a) -> int:
    from . import grader
    from .ledger import Ledger
    preds = Ledger(config.LEDGER_PATH).records()
    prices = grader.load_prices({p["ticker"] for p in preds})
    g, pending = grader.grade(preds, prices, a.spread)
    print(f"{len(g)} graded, {len(pending)} still open")
    s = grader.summarize(g)
    if s["n"]:
        print(json.dumps(s, indent=1, default=str))
        if s["n"] < 150:
            print(f"\n{s['n']} graded is too few to judge; the gate is 150+.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="butterfly")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("what", choices=["prices", "macro", "fx", "filings", "news", "all"])

    b = sub.add_parser("backtest")
    b.add_argument("--ticker", default="SPY")
    b.add_argument("--months", default="6,8,10,12")
    b.add_argument("--subperiod-months", type=int, default=10)
    b.add_argument("--spread", type=float, default=config.SPREAD_PER_SIDE)
    b.add_argument("--cash", default="", help="FRED series for cash yield, e.g. DTB3; default 0%%")

    r = sub.add_parser("ripple")
    r.add_argument("node")
    r.add_argument("--shock", type=int, default=1, choices=[1, -1])
    r.add_argument("--hops", type=int, default=3)
    r.add_argument("--min-strength", type=float, default=0.05)
    r.add_argument("--verified-only", action="store_true")
    r.add_argument("--tickers-only", action="store_true")

    sub.add_parser("nodes")

    pr = sub.add_parser("predict")
    pr.add_argument("--ticker", required=True)
    pr.add_argument("--direction", type=int, required=True, choices=[1, -1])
    pr.add_argument("--horizon", type=int, required=True)
    pr.add_argument("--confidence", type=float, required=True)
    pr.add_argument("--event", required=True)
    pr.add_argument("--chain", nargs="+", required=True, help="one argument per reasoning step")
    pr.add_argument("--evidence", help='JSON list: [{"url": ..., "fetched": ...}]')
    pr.add_argument("--expected-move", type=float)
    pr.add_argument("--author", default="manual")
    pr.add_argument("--tag", action="append", default=[])

    sub.add_parser("verify")
    gr = sub.add_parser("grade")
    gr.add_argument("--spread", type=float, default=config.SPREAD_PER_SIDE)

    a = p.parse_args(argv)
    if a.cmd == "fetch":
        return _fetch(a.what)
    if a.cmd == "backtest":
        return _backtest(a)
    if a.cmd == "ripple":
        return _ripple(a)
    if a.cmd == "nodes":
        from .graph import Graph
        for nid, n in Graph.load(config.GRAPH_PATH).nodes.items():
            print(f"  {nid:<30} {n['type']:<10} {n['name']}")
        return 0
    if a.cmd == "predict":
        return _predict(a)
    if a.cmd == "verify":
        return _verify(a)
    if a.cmd == "grade":
        return _grade(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
