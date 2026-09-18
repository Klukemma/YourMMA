"""Run all three strategies and the parlay backtest over the logged history.

    python engine/backtest.py
    python engine/backtest.py --from 2026-01-01 --min-confidence 0.2

Everything here settles against real prices from engine/data/odds.csv, and a
bet that cannot be priced is reported rather than dropped quietly - a strategy
that looks good only because its losing bets had no odds is not a strategy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from failure_model import flag_quality, walk_forward_flags
from parlay import (build_parlays, report as report_parlays, settle_parlays,
                    summarise_parlays, break_even_hit_rate)
from roi import HISTORY, build_bets, load_odds
from strategies import (report as report_strategies, running_return, settle_all,
                        strategy_combined, strategy_fade, strategy_model,
                        summarise)

DATA_DIR = ENGINE / "data"
ODDS = Path(os.environ.get("ODDS_CSV", DATA_DIR / "odds.csv"))


def load_graded(path=None):
    """Logged predictions whose result is known."""
    blob = json.loads(Path(path or HISTORY).read_text())
    rows = blob["predictions"] if isinstance(blob, dict) else blob
    graded = [r for r in rows if r.get("correct") is not None]
    for row in graded:
        row.setdefault("actual_date", row.get("event_date"))
    return graded, len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="2026-01-01")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    graded, total = load_graded()
    bets = build_bets(graded, pd.Timestamp(args.start))
    odds_index = load_odds(ODDS)
    priced, unpriced = settle_all(strategy_model(bets), odds_index)

    print("=" * 79)
    print("BACKTEST")
    print("=" * 79)
    print(f"  logged predictions      {total}")
    print(f"  with a known result     {len(graded)}")
    print(f"  from {args.start}          {len(bets)}")
    print(f"  priced against odds     {len(priced)}")
    print(f"  no price available      {len(unpriced)}")
    if not priced:
        print("\nNothing could be priced; there is no return to report.")
        return

    print("\n" + "=" * 79)
    print("THE FLAGGING MODEL")
    print("=" * 79)
    flagged, rows = walk_forward_flags(bets, odds_index)
    quality = flag_quality(rows)
    print(f"  picks scored            {len(rows)}")
    print(f"  flagged as likely wrong {len(flagged)}")
    print(f"  flag quality            {quality:.3f}  (0.5 is a coin toss)")
    if rows:
        actually = sum(1 for r in rows if r["actually_failed"])
        print(f"  picks that did fail     {actually} of {len(rows)} "
              f"({actually / len(rows):.1%})")

    print("\n" + "=" * 79)
    print("THREE STRATEGIES, ONE UNIT PER BET")
    print("=" * 79)
    runs = {
        "MODEL": strategy_model(bets, args.min_confidence),
        "FADE (flagged only)": strategy_fade(bets, flagged),
        "COMBINED": strategy_combined(bets, flagged, args.min_confidence),
    }
    summaries, priced_by_label = [], {}
    for label, selection in runs.items():
        run_priced, _ = settle_all(selection, odds_index)
        priced_by_label[label] = run_priced
        summaries.append(summarise(run_priced, label))
    report_strategies(summaries)

    print("\n  Running return, MODEL strategy (every 20th bet):")
    curve = running_return(priced_by_label["MODEL"])
    for i, point in enumerate(curve):
        if i % 20 == 0 or i == len(curve) - 1:
            print(f"    bet {i + 1:>4}  {point['date'].date()}  "
                  f"cumulative {point['cumulative']:+.2f}u")

    print("\n" + "=" * 79)
    print("PARLAYS THE ENGINE WOULD HAVE SUGGESTED")
    print("=" * 79)
    parlay_rows, settled_by_label = [], {}
    for legs in (2, 3, 4):
        settled = settle_parlays(build_parlays(priced_by_label["MODEL"], legs))
        label = f"{legs} legs"
        settled_by_label[label] = settled
        parlay_rows.append(summarise_parlays(settled, label))
    report_parlays(parlay_rows, settled_by_label)

    out = ENGINE / "experiments" / "backtest.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(summaries + [
        {**r, "bets": r["parlays"]} for r in parlay_rows
    ]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
