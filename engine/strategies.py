"""Three betting strategies, and what each would have returned.

The engine's picks are right about 62% of the time, which alone does not beat
a bookmaker's margin. The question worth answering is narrower: when the model
disagrees with the market, who is right?

That needs three separate counts, not one.

  MODEL     back every pick the engine makes, above a confidence floor.
            This is the strategy the engine implies.

  FADE      back the OTHER fighter, but only on the picks a second model
            flags as likely wrong. If the flagging model knows anything, this
            makes money on exactly the fights the first model gets wrong.

  COMBINED  back the pick when it is confident and unflagged, and back the
            opponent when it is flagged. The two models working together.

A staked unit is 1.0 on every bet, so the return is directly comparable
across strategies and no bankroll rule quietly does the work.

Fading is not free: a flagged pick that was going to win costs a unit, so
FADE only pays if the flags are better than noise. Reporting all three
separately is what makes that visible instead of hidden inside one number.
"""

import numpy as np
import pandas as pd

from roi import american_to_profit, find_odds

STAKE = 1.0


def fade(bet):
    """The same fight, backing the other fighter.

    `won` flips because the bet is now on the opponent. The graded history
    holds only wins and losses - no draws or no-contests - so there is no
    third outcome to mishandle here.
    """
    return {
        **bet,
        "pick": bet["opponent"],
        "opponent": bet["pick"],
        "won": not bet["won"],
        "faded": True,
    }


def strategy_model(bets, min_confidence=0.0):
    """Back every pick at or above the confidence floor."""
    return [dict(b, faded=False) for b in bets
            if b["confidence"] >= min_confidence]


def strategy_fade(bets, flagged):
    """Back the opponent on flagged picks only.

    `flagged` is a set of fight keys the second model expects to go wrong.
    """
    return [fade(b) for b in bets if key_of(b) in flagged]


def strategy_combined(bets, flagged, min_confidence=0.0):
    """Back the pick when confident and unflagged; back the opponent when
    flagged. A confident pick that is flagged is faded, not skipped: the
    flag is a statement that the confidence is misplaced."""
    out = []
    for bet in bets:
        if key_of(bet) in flagged:
            out.append(fade(bet))
        elif bet["confidence"] >= min_confidence:
            out.append(dict(bet, faded=False))
    return out


def key_of(bet):
    """Identity of a fight, stable whichever side is being backed."""
    names = tuple(sorted([str(bet["pick"]).lower().strip(),
                          str(bet["opponent"]).lower().strip()]))
    return (pd.to_datetime(bet["date"]).strftime("%Y-%m-%d"), names)


def settle_all(bets, odds_index):
    """Attach the price and the profit for each bet it can price."""
    priced, unpriced = [], []
    for bet in bets:
        odds = find_odds(bet, odds_index)
        if odds is None:
            unpriced.append(bet)
            continue
        profit = american_to_profit(odds) * STAKE if bet["won"] else -STAKE
        priced.append({**bet, "odds": odds, "profit": profit})
    return priced, unpriced


def running_return(priced):
    """Cumulative profit after each bet, in date order."""
    ordered = sorted(priced, key=lambda b: pd.to_datetime(b["date"]))
    total = 0.0
    out = []
    for bet in ordered:
        total += bet["profit"]
        out.append({"date": bet["date"], "profit": bet["profit"],
                    "cumulative": total})
    return out


def bootstrap_roi(priced, samples=2000, seed=0):
    """A 95% interval for the return, by resampling the bets.

    131 bets is a small sample and a flat-looking return can hide a wide
    range. Reporting the interval stops a couple of lucky underdogs reading
    as an edge.
    """
    if not priced:
        return (float("nan"), float("nan"))
    profits = np.array([b["profit"] for b in priced], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(profits, size=(samples, len(profits)), replace=True)
    rois = draws.mean(axis=1) / STAKE
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


def summarise(priced, label):
    """Everything worth knowing about one strategy's run."""
    if not priced:
        return {"label": label, "bets": 0, "wins": 0, "hit_rate": float("nan"),
                "profit": 0.0, "roi": float("nan"),
                "roi_low": float("nan"), "roi_high": float("nan")}
    wins = sum(1 for b in priced if b["won"])
    profit = sum(b["profit"] for b in priced)
    low, high = bootstrap_roi(priced)
    return {
        "label": label,
        "bets": len(priced),
        "wins": wins,
        "hit_rate": wins / len(priced),
        "profit": profit,
        "roi": profit / (len(priced) * STAKE),
        "roi_low": low,
        "roi_high": high,
    }


def report(rows):
    print(f"  {'strategy':<22}{'bets':>6}{'hit':>8}{'profit':>10}{'ROI':>9}"
          f"{'95% interval':>22}")
    print("  " + "-" * 77)
    for r in rows:
        if not r["bets"]:
            print(f"  {r['label']:<22}{'-':>6}{'no bets priced':>30}")
            continue
        interval = f"[{r['roi_low']:+.1%}, {r['roi_high']:+.1%}]"
        print(f"  {r['label']:<22}{r['bets']:>6}{r['hit_rate']:>8.1%}"
              f"{r['profit']:>+10.2f}{r['roi']:>+9.1%}{interval:>22}")
