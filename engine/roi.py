"""Flat-stake ROI on graded predictions, and what odds would be needed to profit.

    python engine/roi.py                      # from 2026-01-01
    python engine/roi.py --from 2026-03-01
    python engine/roi.py --odds engine/data/odds.csv
    python engine/roi.py --min-confidence 0.4

One unit on every pick, win or lose, from the start date. With an odds file it
reports the real running return. Without one it reports the break-even price -
the average odds each strategy would have needed to break even - which is the
honest version of "is this worth betting" when no prices are available.

An odds file is CSV with: date, fighter_a, fighter_b, odds_a, odds_b
in American format (-150, +130). Corner order does not matter.
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade import (DATE_TOLERANCE, deduplicate, grade, load_results)
from name_resolution import norm_name

ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
HISTORY = Path(os.environ.get("PREDICTIONS_LOG", DATA_DIR / "prediction_history.json"))

STAKE = 1.0


def american_to_profit(odds):
    """Profit on a 1-unit winning stake. -150 -> 0.667, +130 -> 1.30."""
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def american_to_implied(odds):
    """Implied probability, including the bookmaker's margin."""
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def break_even_probability(odds):
    """The win rate needed to break even at these odds. Same as implied."""
    return american_to_implied(odds)


def probability_to_american(p):
    """The fair American price for a probability - the break-even line."""
    if not 0 < p < 1:
        return None
    if p >= 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def load_odds(path):
    """{(norm_a, norm_b): [(date, odds_a, odds_b)]}, both corner orders."""
    df = pd.read_csv(path)
    required = {'date', 'fighter_a', 'fighter_b', 'odds_a', 'odds_b'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"odds file is missing columns: {sorted(missing)}")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    index = {}
    for row in df.itertuples(index=False):
        a, b = norm_name(row.fighter_a), norm_name(row.fighter_b)
        index.setdefault((a, b), []).append((row.date, row.odds_a, row.odds_b))
        index.setdefault((b, a), []).append((row.date, row.odds_b, row.odds_a))
    return index


def find_odds(bet, odds_index):
    """Odds on the predicted fighter, or None."""
    if not odds_index:
        return None
    key = (norm_name(bet['pick']), norm_name(bet['opponent']))
    candidates = odds_index.get(key)
    if not candidates:
        return None
    date = pd.to_datetime(bet['date'])
    within = [c for c in candidates if abs(c[0] - date) <= DATE_TOLERANCE]
    if not within:
        return None
    return min(within, key=lambda c: abs(c[0] - date))[1]


def build_bets(graded, start, min_confidence=0.0):
    """One bet per graded pick from the start date, most recent first excluded."""
    bets = []
    for row in graded:
        date = pd.to_datetime(row['actual_date'])
        if pd.isna(date) or date < start:
            continue
        if row['win_probability'] is None:
            continue
        confidence = abs(2 * row['win_probability'] - 1)
        if confidence < min_confidence:
            continue
        opponent = (row['blue_corner']
                    if norm_name(row['predicted_winner']) == norm_name(row['red_corner'])
                    else row['red_corner'])
        bets.append({
            'date': date,
            'event': row['event_name'],
            'pick': row['predicted_winner'],
            'opponent': opponent,
            'win_probability': row['win_probability'],
            'confidence': confidence,
            'won': bool(row['correct']),
        })
    bets.sort(key=lambda b: b['date'])
    return bets


def settle(bets, odds_index):
    """Attach odds and profit. Bets with no price are returned separately."""
    priced, unpriced = [], []
    for bet in bets:
        odds = find_odds(bet, odds_index)
        if odds is None:
            unpriced.append(bet)
            continue
        bet = dict(bet)
        bet['odds'] = float(odds)
        bet['profit'] = american_to_profit(odds) * STAKE if bet['won'] else -STAKE
        priced.append(bet)
    return priced, unpriced


def report_roi(priced):
    staked = STAKE * len(priced)
    profit = sum(b['profit'] for b in priced)
    wins = sum(b['won'] for b in priced)
    print("=" * 66)
    print("FLAT-STAKE ROI")
    print("=" * 66)
    print(f"  bets           : {len(priced)}")
    print(f"  record         : {wins}-{len(priced)-wins}  ({wins/len(priced):.1%})")
    print(f"  staked         : {staked:.2f} units")
    print(f"  profit         : {profit:+.2f} units")
    print(f"  ROI            : {profit/staked:+.1%}")
    print(f"  final bankroll : {100 + profit*1:.2f}  (from 100, 1 unit per bet)")

    print("\n  running by event:")
    running = 0.0
    by_event = {}
    for b in priced:
        by_event.setdefault((b['date'], b['event']), []).append(b)
    print(f"    {'date':<12}{'event':<42}{'n':>3}{'P/L':>9}{'running':>10}")
    for (date, event), rows in sorted(by_event.items()):
        pl = sum(r['profit'] for r in rows)
        running += pl
        print(f"    {str(date)[:10]:<12}{event[:40]:<42}{len(rows):>3}{pl:>+9.2f}{running:>+10.2f}")


def report_break_even(bets, label="ALL PICKS"):
    """With no prices, say what the market would have had to offer."""
    if not bets:
        print(f"  {label}: no bets")
        return
    wins = sum(b['won'] for b in bets)
    rate = wins / len(bets)
    fair = probability_to_american(rate) if 0 < rate < 1 else None
    stated = float(np.mean([b['win_probability'] for b in bets]))
    print(f"  {label:<26}{len(bets):>5}{rate:>9.1%}{stated:>10.1%}"
          f"{(str(fair) if fair else '-'):>12}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start', default='2026-01-01',
                    help='count from this date (default 2026-01-01)')
    ap.add_argument('--odds', help='CSV of historical odds')
    ap.add_argument('--min-confidence', type=float, default=0.0)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    history = json.loads(HISTORY.read_text())
    graded, _ = grade(deduplicate(history['predictions']), load_results(UFC_CSV))
    bets = build_bets(graded, start, args.min_confidence)
    print(f"graded picks from {start.date()}: {len(bets)}")
    if not bets:
        sys.exit("Nothing to settle.")

    odds_index = load_odds(args.odds) if args.odds else None
    priced, unpriced = settle(bets, odds_index)

    if priced:
        report_roi(priced)
        if unpriced:
            print(f"\n  {len(unpriced)} picks had no odds and were excluded.")
        return

    # No prices available - report what would have been needed instead.
    print("\n" + "=" * 66)
    print("NO ODDS AVAILABLE - BREAK-EVEN ANALYSIS")
    print("=" * 66)
    print("  The average price each strategy needed just to break even.")
    print("  Beat that price on average and it profits; miss it and it loses.\n")
    print(f"  {'strategy':<26}{'bets':>5}{'won':>9}{'model said':>10}{'break-even':>12}")
    print("  " + "-" * 62)
    report_break_even(bets, "every pick")
    for threshold in (0.2, 0.4, 0.6):
        subset = [b for b in bets if b['confidence'] >= threshold]
        if subset:
            report_break_even(subset, f"confidence >= {threshold}")

    print("\n  A break-even line of -300 means you need better than -300 on average.")
    print("  Favourites at that price are common, so a high hit rate is not")
    print("  automatically profitable. Supply --odds to settle this properly.")


if __name__ == '__main__':
    main()
