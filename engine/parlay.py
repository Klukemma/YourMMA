"""Parlays the engine would have suggested, and what they would have returned.

A parlay multiplies the odds of several picks and needs every one of them to
land. That trade is much worse than it looks: at a 62% hit rate a three-leg
parlay lands about 24% of the time, so the payout has to more than quadruple
just to break even, and the bookmaker's margin is applied to every leg.

This builds the parlays the engine's own confidence ordering implies - the
most confident legs on each card - and settles them against real prices. It
is a backtest of a suggestion that was never made rather than a claim that
parlays are a good idea; if they lose money here, that is the finding.
"""

import numpy as np
import pandas as pd

from roi import american_to_profit, find_odds
from strategies import STAKE

DEFAULT_LEGS = (2, 3, 4)


def decimal_odds(american):
    """American price as a decimal multiplier, so legs can be multiplied."""
    return american_to_profit(american) + 1.0


def card_key(bet):
    """Bets on the same event, which is what a parlay combines."""
    return (bet.get("event") or "", pd.to_datetime(bet["date"]).strftime("%Y-%m-%d"))


def build_parlays(priced, legs=3, per_card=1):
    """The `legs` most confident priced picks on each card.

    per_card caps how many parlays one event contributes, so a large card
    cannot dominate the record.
    """
    cards = {}
    for bet in priced:
        cards.setdefault(card_key(bet), []).append(bet)

    out = []
    for key, card in sorted(cards.items()):
        ranked = sorted(card, key=lambda b: -b["confidence"])
        for start in range(per_card):
            chosen = ranked[start:start + legs]
            if len(chosen) < legs:
                break
            out.append({
                "event": key[0],
                "date": key[1],
                "legs": chosen,
                "n_legs": legs,
            })
    return out


def settle_parlays(parlays):
    """A parlay pays only if every leg wins."""
    settled = []
    for parlay in parlays:
        multiplier = float(np.prod([decimal_odds(leg["odds"])
                                    for leg in parlay["legs"]]))
        won = all(leg["won"] for leg in parlay["legs"])
        profit = (multiplier - 1.0) * STAKE if won else -STAKE
        settled.append({**parlay, "multiplier": multiplier,
                        "won": won, "profit": profit})
    return settled


def summarise_parlays(settled, label):
    if not settled:
        return {"label": label, "parlays": 0, "wins": 0,
                "hit_rate": float("nan"), "profit": 0.0, "roi": float("nan"),
                "mean_multiplier": float("nan")}
    wins = sum(1 for p in settled if p["won"])
    profit = sum(p["profit"] for p in settled)
    return {
        "label": label,
        "parlays": len(settled),
        "wins": wins,
        "hit_rate": wins / len(settled),
        "profit": profit,
        "roi": profit / (len(settled) * STAKE),
        "mean_multiplier": float(np.mean([p["multiplier"] for p in settled])),
    }


def break_even_hit_rate(settled):
    """The hit rate these parlays needed to break even, given their prices."""
    if not settled:
        return float("nan")
    return float(np.mean([1.0 / p["multiplier"] for p in settled]))


def report(rows, settled_by_label=None):
    print(f"  {'parlay':<16}{'n':>6}{'hit':>8}{'needed':>9}"
          f"{'avg payout':>12}{'profit':>10}{'ROI':>9}")
    print("  " + "-" * 70)
    for row in rows:
        if not row["parlays"]:
            print(f"  {row['label']:<16}{'-':>6}  none could be priced")
            continue
        needed = float("nan")
        if settled_by_label:
            needed = break_even_hit_rate(settled_by_label.get(row["label"], []))
        print(f"  {row['label']:<16}{row['parlays']:>6}{row['hit_rate']:>8.1%}"
              f"{needed:>9.1%}{row['mean_multiplier']:>12.2f}"
              f"{row['profit']:>+10.2f}{row['roi']:>+9.1%}")
