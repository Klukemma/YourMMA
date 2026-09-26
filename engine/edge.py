"""Does the model know anything the market does not?

ROI told us the picks lose. This asks the sharper question: on the same fights,
is the model's probability closer to the truth than the market's? If the market
is better everywhere, no amount of further modelling on public stats will help,
because the market already prices those stats. If the model is better on some
segment, that segment is where an edge could live.

Market prices are de-vigged before comparison. A -150/+130 pair implies 60% and
43.5%, summing to 103.5% - the bookmaker's margin. Comparing a model
probability against a raw implied probability would credit the model for
beating a number that was never meant to be a fair estimate.

    python engine/edge.py
    python engine/edge.py --from 2026-01-01
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade import deduplicate, deduplicate_graded, grade, load_results
from name_resolution import norm_name
from roi import (american_to_implied, american_to_profit, build_bets,
                 find_odds, load_odds)

ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
HISTORY = Path(os.environ.get("PREDICTIONS_LOG", DATA_DIR / "prediction_history.json"))
ODDS = DATA_DIR / "odds.csv"


def devig(implied_pick, implied_other):
    """Strip the bookmaker's margin, proportionally."""
    total = implied_pick + implied_other
    return implied_pick / total if total > 0 else np.nan


def brier(probs, outcomes):
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def log_loss(probs, outcomes):
    p = np.clip(np.asarray(probs), 1e-15, 1 - 1e-15)
    o = np.asarray(outcomes)
    return float(np.mean(-(o * np.log(p) + (1 - o) * np.log(1 - p))))


def build_rows(bets, odds_index):
    """Pair each bet with the market's own estimate."""
    rows = []
    for bet in bets:
        odds_pick = find_odds(bet, odds_index)
        if odds_pick is None:
            continue
        mirrored = dict(bet, pick=bet['opponent'], opponent=bet['pick'])
        odds_other = find_odds(mirrored, odds_index)
        if odds_other is None:
            continue
        market = devig(american_to_implied(odds_pick), american_to_implied(odds_other))
        if not np.isfinite(market):
            continue
        rows.append({
            'date': bet['date'], 'event': bet['event'],
            'pick': bet['pick'], 'opponent': bet['opponent'],
            'model': float(bet['win_probability']),
            'market': float(market),
            'odds': float(odds_pick),
            'won': 1.0 if bet['won'] else 0.0,
            'profit': american_to_profit(odds_pick) if bet['won'] else -1.0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['disagreement'] = df['model'] - df['market']
        df['is_favourite'] = df['odds'] < 0
    return df


def segment(df, mask, label):
    sub = df[mask]
    if len(sub) < 5:
        print(f"  {label:<30}{len(sub):>5}   too few to read")
        return
    roi = sub['profit'].sum() / len(sub)
    print(f"  {label:<30}{len(sub):>5}{sub['won'].mean():>9.1%}"
          f"{sub['model'].mean():>10.1%}{sub['market'].mean():>10.1%}{roi:>+9.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start', default='2026-01-01')
    args = ap.parse_args()

    history = json.loads(HISTORY.read_text())
    graded = deduplicate_graded(
        grade(deduplicate(history['predictions']), load_results(UFC_CSV))[0])
    bets = build_bets(graded, pd.Timestamp(args.start))
    df = build_rows(bets, load_odds(ODDS))
    if df.empty:
        sys.exit("No fights had prices on both sides.")

    print("=" * 72)
    print(f"MODEL vs MARKET   {len(df)} fights priced on both sides, "
          f"from {args.start}")
    print("=" * 72)
    print(f"  {'':<22}{'Brier':>10}{'log loss':>11}")
    print(f"  {'model':<22}{brier(df['model'], df['won']):>10.4f}"
          f"{log_loss(df['model'], df['won']):>11.4f}")
    print(f"  {'market (de-vigged)':<22}{brier(df['market'], df['won']):>10.4f}"
          f"{log_loss(df['market'], df['won']):>11.4f}")
    gap = brier(df['model'], df['won']) - brier(df['market'], df['won'])
    verdict = ("the market is better" if gap > 0 else "the model is better")
    print(f"\n  Brier difference: {gap:+.4f}  -> {verdict}")

    # Whose favourite wins when they disagree about who wins at all?
    flip = df[(df['model'] >= 0.5) != (df['market'] >= 0.5)]
    if len(flip) >= 5:
        print(f"\n  outright disagreements (model and market pick different "
              f"fighters): {len(flip)}")
        print(f"    model's pick won {flip['won'].mean():.1%} of them "
              f"({int(flip['won'].sum())}/{len(flip)})")

    print("\n" + "-" * 72)
    print(f"  {'segment':<30}{'n':>5}{'won':>9}{'model':>10}{'market':>10}{'ROI':>9}")
    print("-" * 72)
    segment(df, df.index == df.index, 'every priced pick')
    segment(df, df['is_favourite'], 'backing a favourite')
    segment(df, ~df['is_favourite'], 'backing an underdog')
    print()
    segment(df, df['disagreement'] > 0.20, 'model 20pt more bullish')
    segment(df, (df['disagreement'] > 0.10) & (df['disagreement'] <= 0.20),
            'model 10-20pt more bullish')
    segment(df, df['disagreement'].abs() <= 0.10, 'model and market agree')
    segment(df, df['disagreement'] < -0.10, 'model less bullish')
    print()
    segment(df, df['odds'] <= -300, 'heavy favourites (<= -300)')
    segment(df, (df['odds'] > -300) & (df['odds'] < 0), 'modest favourites')
    segment(df, (df['odds'] > 0) & (df['odds'] < 200), 'short underdogs')
    segment(df, df['odds'] >= 200, 'long underdogs (>= +200)')

    out = ENGINE_DIR / 'experiments' / 'edge_rows.csv'
    df.to_csv(out, index=False)
    print(f"\nper-fight rows written to {out}")


if __name__ == '__main__':
    main()
