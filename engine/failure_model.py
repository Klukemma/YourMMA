"""The second layer: a model that predicts when the first model is wrong.

The engine already reports a confidence number, and that number is close to
useless as a guide to correctness - across 136 graded bouts the relationship
between "how confident was it" and "was it right" was barely better than a
coin toss. So a flag has to be built from more than confidence.

What is available at the moment a bet would be placed:

    the model's own probability and confidence
    the market's price on both fighters
    how far the model is from the market
    whether the model is backing the underdog

The last two are the interesting ones. A model that disagrees sharply with a
market that prices thousands of fights a year is either finding something or
making a mistake, and which one it is can be learned from history.

Training is walk-forward: for each bet, the model is fitted only on bets that
had already settled. A flag on the first few dozen is therefore unavailable
rather than guessed, which is why those come back unflagged.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from roi import american_to_implied, find_odds
from strategies import key_of

MIN_TRAIN = 40      # too few settled bets to fit anything honest
FLAG_THRESHOLD = 0.5


def features(bet, odds_index):
    """What the flagger sees. None when the fight has no price."""
    pick_odds = find_odds(bet, odds_index)
    if pick_odds is None:
        return None
    other = {**bet, "pick": bet["opponent"], "opponent": bet["pick"]}
    opp_odds = find_odds(other, odds_index)
    if opp_odds is None:
        return None

    market = american_to_implied(pick_odds)
    opp_market = american_to_implied(opp_odds)
    total = market + opp_market
    market_fair = market / total if total > 0 else np.nan  # margin removed

    model_p = float(bet["win_probability"])
    return {
        "model_p": model_p,
        "confidence": float(bet["confidence"]),
        "market_fair": market_fair,
        "disagreement": model_p - market_fair,
        "backing_underdog": 1.0 if market_fair < 0.5 else 0.0,
    }


COLUMNS = ["model_p", "confidence", "market_fair", "disagreement",
           "backing_underdog"]


def walk_forward_flags(bets, odds_index, threshold=FLAG_THRESHOLD,
                       min_train=MIN_TRAIN):
    """Flag the picks a model trained on earlier bets expects to fail.

    Returns (flagged_keys, rows) where rows carries the probability of failure
    for every bet that could be scored.
    """
    ordered = sorted(bets, key=lambda b: pd.to_datetime(b["date"]))
    usable = []
    for bet in ordered:
        feat = features(bet, odds_index)
        if feat is not None:
            usable.append((bet, feat))

    flagged, rows = set(), []
    for i, (bet, feat) in enumerate(usable):
        history = usable[:i]
        # Only bets that had settled before this one, and only if there are
        # enough of them and both outcomes are present.
        if len(history) < min_train:
            continue
        X = pd.DataFrame([f for _, f in history])[COLUMNS]
        y = np.array([0.0 if b["won"] else 1.0 for b, _ in history])
        if len(np.unique(y)) < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        p_fail = float(model.predict_proba(
            pd.DataFrame([feat])[COLUMNS])[0, 1])
        rows.append({"key": key_of(bet), "date": bet["date"],
                     "p_fail": p_fail, "actually_failed": not bet["won"]})
        if p_fail >= threshold:
            flagged.add(key_of(bet))
    return flagged, rows


def flag_quality(rows):
    """How well the flags track actual failures.

    0.5 is a coin toss. Reported because a strategy built on flags that carry
    no information will still produce a return, and that return will be luck.
    """
    from sklearn.metrics import roc_auc_score

    if len(rows) < 20:
        return float("nan")
    y = np.array([1.0 if r["actually_failed"] else 0.0 for r in rows])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, [r["p_fail"] for r in rows]))
