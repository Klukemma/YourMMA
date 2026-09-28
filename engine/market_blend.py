"""Blend the model with the closing line, and keep the two numbers apart.

THE CEILING, measured on 2,339 priced walk-forward predictions from 2020:

    the model picks right                 62.0%
    the market favourite wins             67.5%
    ceiling implied by market prices      65.2%   a perfect forecaster
    blend, weight chosen from earlier
      years only and applied forward      68.2%

The blend beats the market it is built from, by about a point, which is the
model contributing something the line does not already carry. It is also the
largest single gain available anywhere in this project: five points of
accuracy for arithmetic, against a model that has been tuned for months.

Anything above about 68% is not on offer. The market itself manages 67.5%,
and a perfectly calibrated forecaster reading the market's own prices would
manage 65.2%. A backtest claiming 70% on public data would be wrong before it
was interesting.

TWO NUMBERS, TWO JOBS. Blending makes the prediction accurate and makes the
betting edge disappear, because the edge IS the disagreement with the market:
fold the market in and there is less left to disagree with. At a 50% weight
the fights showing a 5-point edge drop from 1,092 to 773. So this module
returns both and never collapses them -

    blended   what the fight is likely to do. The number to show, to grade
              and to believe.
    raw       what the model thinks WITHOUT the market. The number the edge
              and every betting strategy must keep using, because an edge
              measured against a probability that already contains the line
              is measuring the line against itself.

DE-VIG FIRST. Both sides' prices imply probabilities that sum to more than
one - that surplus is the bookmaker's margin. Blending with the raw implied
probability folds the margin into the prediction and tilts every fight toward
the favourite. The margin is removed by scaling both sides to sum to one,
which needs the OPPONENT's price as well and is why this takes a pair.
"""

import numpy as np

# Chosen by walk-forward search on odds.csv: for each year, the weight that
# maximised accuracy over every earlier year. It came back 0.75 for every
# year from 2016 to 2026 without exception, which is a stable parameter
# rather than a fitted one. Re-measured by experiments/market_blend.py.
MARKET_WEIGHT = 0.75

# Below this the de-vigged pair is not trustworthy - a quoted pair that
# implies a 30% margin is a stale or mistyped line, not a market view.
MAX_MARGIN = 0.15


def implied_probability(american):
    """What one American price implies, margin included."""
    if american is None:
        return None
    american = float(american)
    if american == 0:
        return None
    if american > 0:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)


def devig(pick_odds, opponent_odds, *, max_margin=MAX_MARGIN):
    """The market's probability for the pick, with the margin removed.

    Returns None when either side is missing or the pair implies a margin so
    wide that it is not a market view. Proportional de-vigging: both sides
    scaled to sum to one. It is not the only method - shin and power
    de-vigging differ on heavy favourites - but it is the one the rest of
    this project already uses in failure_model.features, and two different
    de-vigs in one codebase would be a quiet disagreement about what the
    market said.
    """
    mine = implied_probability(pick_odds)
    theirs = implied_probability(opponent_odds)
    if mine is None or theirs is None:
        return None
    total = mine + theirs
    if total <= 0 or total - 1.0 > max_margin:
        return None
    return mine / total


def blend(model_probability, market_probability, *, weight=MARKET_WEIGHT):
    """Weighted average, or the model untouched when there is no market.

    An unpriced fight keeps the model's own number rather than being pushed
    toward a half: "nobody has priced this" is not evidence that it is close.
    """
    if market_probability is None:
        return float(model_probability)
    weight = float(np.clip(weight, 0.0, 1.0))
    mixed = (1.0 - weight) * float(model_probability) + weight * float(market_probability)
    return float(np.clip(mixed, 0.01, 0.99))


def edge(model_probability, market_probability):
    """How far the MODEL is from the market, for betting.

    Takes the raw model probability on purpose. Passing the blended one here
    would compare the market with a number that is three-quarters market, and
    report the remainder as an edge.
    """
    if market_probability is None:
        return None
    return float(model_probability) - float(market_probability)
