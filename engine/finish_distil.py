"""Learn P(finish) from the market's price rather than from the result.

A binary outcome is a terrible target. One fight gives one bit, and "this was
a 55/45 fight that happened to end early" and "this was a 90/10 fight that
ended early" arrive looking identical. The market answers the same question
on a continuous scale, with money behind it, and its answer is published
before the bell - so training toward it is not reading a result.

MEASURED, walk-forward over 4,764 fights carrying both a price and an
outcome, each year predicted from earlier years only, three seeds, judged on
what actually happened:

    trained on the outcome         accuracy 55.7%   AUC 0.577   Brier 0.2671
    trained on 25% market          55.4%            0.585       0.2494
    trained on 50% market          56.8%            0.599       0.2441
    trained on 75% market          57.6%            0.616       0.2408
    trained on the market alone    58.8%            0.630       0.2403
    the market itself              60.2%            0.661       0.2334

Monotone the whole way, and the weight chosen from earlier years came back
1.00 in every year - pure distillation, no share of the raw outcome at all.
The AUC gain over training on the outcome is +0.053, 95% [+0.038, +0.068],
and it closes 63% of the distance to the market.

WHY THIS MATTERS MORE THAN THE BLEND. The method market does not exist live:
every method market on this plan returns 422 INVALID_MARKET, so blending was
never available. A distilled model needs no price at prediction time. It has
learned to predict the market FROM THE RECORD, which is the only form in
which this information can reach a card.

WHAT IT ANSWERED. The gap to the market had two possible explanations - the
features carry it and the model was not extracting it, or the features do not
carry it. Two thirds of the gap closed without one new feature, so it was
mostly the first. The remaining third may still be the second: camps,
injuries, styles, the things a record does not hold.

THE STUDENT INHERITS THE TEACHER'S BIAS. The market says 56.5% of fights are
finished where 50.8% are, and a model trained on that says so too. The
isotonic calibration predict_card already applies runs afterwards and fixes
the price without touching the order.

WHAT THAT +0.053 IS AND IS NOT MEASURED AGAINST. The baseline in that table
is a binary classifier trained on the outcome over the priced fights only. It
is NOT the model this student replaces, which is a three-way classifier
trained with class weights on every fight in the archive - a stronger model.
So the experiment establishes that distilling beats outcome-training ON THE
SAME FEATURES. It does not, by itself, establish a gain over what ships.

AGAINST WHAT SHIPS, on the 841-fight calibration split - which the three-way
model has seen once as its early-stopping eval set and the student has not,
so the comparison is tilted against the student - both calibrated the same
way:

                              AUC    Brier     acc   bal acc
    three-way + isotonic    0.644   0.2315   60.3%     0.602
    this student            0.661   0.2283   61.7%     0.612

Better on all four, and a paired bootstrap over those fights puts every
difference inside the noise: Brier +0.0033 [-0.0026, +0.0089], accuracy
+0.015 [-0.014, +0.045], balanced accuracy +0.010 [-0.019, +0.041]. 841
fights cannot resolve a gap this size; it would take several thousand. So the
honest statement is that the student is better on every measure taken and
worse on none, that the mechanism has strong walk-forward support, and that
the direct comparison with the shipped model is consistent with it rather
than proof of it.

predict_card runs that comparison on every build and will not use a student
that loses it. A measurement in an experiment file says the idea works. It
does not say this build wired it up right, and the first one did not: it
fitted on the raw feature frame and predicted on the scaled one, which cost
0.069 of AUC and squeezed every fight on a card into 44-54%. See
`_check_footprint` for the part of that failure this module can catch itself.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# The same shape as the method model it stands beside, so a difference between
# them is the target and not the hyperparameters.
PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
              reg_lambda=1.0, random_state=42, verbosity=0)

PROP_COLUMNS = ("dec_a", "dec_b", "ko_a", "ko_b", "sub_a", "sub_b")
MIN_TRAIN = 400          # below this the student has not seen enough market
MAX_OVERROUND = 1.60     # a pair of prices implying more than this is broken


def implied(american):
    """What one American price implies, margin included."""
    american = np.asarray(american, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(american > 0, 100.0 / (american + 100.0),
                        -american / (-american + 100.0))


def market_finish_probability(frame):
    """P(finish) from the six props, with the bookmaker's margin removed.

    The six outcomes cover the space between them, so they de-vig together:
    scale all six to sum to one, then add the two that are not a decision.
    Returns (probability, overround) so a caller can throw away rows whose
    prices cannot be right - these props imply about 122% between them
    normally, and a row implying 160% is a mis-scrape, not a market.
    """
    prices = np.column_stack([implied(frame[c]) for c in PROP_COLUMNS])
    total = prices.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fair = prices / total[:, None]
    return 1.0 - (fair[:, 0] + fair[:, 1]), total


def load_targets(path, key_of):
    """{fight key: market P(finish)} from a method_odds.csv.

    `key_of(fighter_a, fighter_b, date)` is supplied by the caller so this
    module does not need to know how the rest of the project spells a fight.
    """
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "fighter_a", "fighter_b",
                                 *PROP_COLUMNS])
    if frame.empty:
        return {}
    probability, overround = market_finish_probability(frame)
    keep = (overround > 1.0) & (overround < MAX_OVERROUND) \
        & np.isfinite(probability)
    return {key_of(a, b, d): float(p) for a, b, d, p, ok
            in zip(frame.fighter_a, frame.fighter_b, frame.date,
                   probability, keep) if ok}


# How far a prediction matrix may sit from the one the student was fitted on
# before this module refuses to answer. See `_check_footprint` for why.
DRIFT_SCALES = 6.0        # robust scales away before a column counts as moved
DRIFT_SHARE = 0.25        # share of columns that must have moved to refuse
SCALE_FACTOR = 4.0        # how far a column's spread may differ, either way
SCALE_SHARE = 0.50        # share of columns whose spread must differ to refuse
MIN_ROWS_FOR_SCALE = 30   # below this a column's spread cannot be estimated


def _footprint(X):
    """Where the training data lived, per column, robustly.

    A median and an inter-quartile range rather than a mean and a standard
    deviation, because one absurd row should not move the description of where
    the data sits.
    """
    X = np.asarray(X, dtype=float)
    q1, median, q3 = np.nanpercentile(X, [25, 50, 75], axis=0)
    return median, np.maximum(q3 - q1, 1e-9)


def _check_footprint(model, X):
    """Refuse a matrix that is not in the units the student was fitted in.

    THE BUG THIS EXISTS FOR. The first version wired into predict_card fitted
    on the raw feature frame and predicted on the standardised one. Trees do
    not fail loudly on that: every split threshold is simply in the wrong
    units, so most rows fall down the same few branches and the model returns
    a confident-looking number in a narrow band. Held-out AUC fell from 0.630
    to 0.582 and the card printed 44-54% for nine fights in a row. Nothing
    anywhere said the input was wrong.

    TWO CHECKS, AND THE SECOND IS THE ONE THAT WORKS. Location alone is weak:
    standardising a column whose median is 50 and whose inter-quartile range
    is 13 moves it less than four of its own ranges, which is inside any
    tolerance loose enough not to reject an unusual fighter. The FIRST version
    of this guard used location alone and a test proved it would have missed
    the very bug it was written for. Spread is the reliable signal - a
    standardised column has an inter-quartile range near 1.3 where a raw one
    has whatever its units give, and those differ by an order of magnitude -
    but it needs enough rows to estimate. So:

        many rows   spread, which catches a units mismatch outright
        few rows    location only, which is weak and known to be weak

    A single card is the second case, so this is NOT the last line of defence.
    That is the held-out comparison in predict_card, which refuses to use a
    student that does not beat the model it replaces.
    """
    footprint = getattr(model, "_distil_footprint", None)
    if footprint is None:
        return
    median, scale = footprint
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(median):
        raise ValueError(
            f"the student was fitted on {len(median)} features and was given "
            f"{X.shape[1]}")

    if len(X) >= MIN_ROWS_FOR_SCALE:
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        here = np.maximum(q3 - q1, 1e-9)
        ratio = here / scale
        moved = (ratio > SCALE_FACTOR) | (ratio < 1.0 / SCALE_FACTOR)
        share = float(np.mean(moved))
        if share > SCALE_SHARE:
            raise ValueError(
                f"{share:.0%} of columns are spread more than "
                f"{SCALE_FACTOR:g}x differently from the data this student "
                f"was fitted on. That is a units mismatch, not unusual "
                f"fighters - the likeliest cause is fitting on the raw "
                f"feature frame and predicting on the scaled one, or the "
                f"reverse.")

    off = np.abs(np.nanmedian(X, axis=0) - median) > DRIFT_SCALES * scale
    if float(np.mean(off)) > DRIFT_SHARE:
        raise ValueError(
            f"{np.mean(off):.0%} of columns sit more than {DRIFT_SCALES:g} "
            f"inter-quartile ranges from where this student was fitted, which "
            f"is a different feature space, not a strange fighter.")


def fit(X, targets, *, min_train=MIN_TRAIN, params=None):
    """Train the student. None when there is too little market to learn from.

    Returning None rather than a model matters: a student fitted on eighty
    fights would still produce a confident number for every bout on a card,
    and there would be nothing in the output to say it was guessing.
    """
    X = np.asarray(X, dtype=float)
    targets = np.asarray(targets, dtype=float)
    usable = np.isfinite(targets)
    if usable.sum() < min_train:
        return None
    model = XGBRegressor(objective="reg:squarederror",
                         **(params or PARAMS))
    model.fit(X[usable], targets[usable], verbose=False)
    model._distil_footprint = _footprint(X[usable])
    return model


def predict(model, X):
    """P(finish), or None when there is no student.

    Raises when the matrix is not in the units the student was fitted in;
    see `_check_footprint`. A wrong answer that looks right is worse than a
    crash, and this particular wrong answer looked right for a whole card.
    """
    if model is None:
        return None
    _check_footprint(model, X)
    return np.clip(model.predict(np.asarray(X, dtype=float)), 0.001, 0.999)
