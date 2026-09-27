"""Make P(finish) worth its face value, and decide where to call it.

Two jobs that look like one and are not.

CALIBRATION. The method model ranks fights well and prices them badly. Over
the confirm period it said 79% and 64% happened, said 31% and 36% happened -
monotone, so the ordering is real, and stretched outward at both ends by the
class weighting, so the numbers are not. An isotonic fit on held-out
predictions pulls the ends back without touching the order, because isotonic
regression is monotone by construction: it can change what a score is worth,
never which of two fights is likelier to end early.

THE THRESHOLD. Where to stop saying "distance" and start saying "finish".
The obvious answer is a half and the obvious temptation is to move it down
until the app says "finish" more often, because that feels closer to how
fights go. THAT IS THE ONE THING THIS MODULE MUST NOT DO. A threshold chosen
to match a prior is a knob turned until the output flatters the person
turning it.

So it is chosen against a stated objective, on data the model did not train
on, and it is allowed to come back as 0.5:

    objective   balanced accuracy - the mean of the two classes' recall,
                which a threshold cannot game by simply calling everything
                a finish, the way plain accuracy can be gamed by calling
                everything a decision
    fitted on   a holdout carved out of the training window
    applied to  the period after it, never overlapping

If the chosen threshold does not beat a half on the period after the one that
chose it, the module says so and a half is what ships.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

# A threshold is only searched inside this band. Outside it the call stops
# being a call: at 0.2 everything is a finish, at 0.8 nothing is.
THRESHOLD_RANGE = (0.30, 0.70)
THRESHOLD_STEPS = 41
MIN_CALIBRATION_ROWS = 200   # below this an isotonic fit is memorising


def fit_calibrator(p_finish, finished, *, min_rows=MIN_CALIBRATION_ROWS):
    """Isotonic map from a raw P(finish) to one worth its face value.

    Returns None rather than a calibrator when there is too little to fit, or
    when the outcomes are all one class - either would produce a mapping that
    looks authoritative and means nothing.
    """
    p_finish = np.asarray(p_finish, dtype=float)
    finished = np.asarray(finished, dtype=float)
    if len(p_finish) < min_rows or len(np.unique(finished)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p_finish, finished)
    return model


def apply_calibrator(calibrator, p_finish):
    """Calibrated P(finish), or the raw value when there is no calibrator."""
    if calibrator is None:
        return np.asarray(p_finish, dtype=float)
    return np.clip(calibrator.predict(np.asarray(p_finish, dtype=float)), 0.0, 1.0)


def balanced_accuracy(p_finish, finished, threshold):
    """Mean of the two recalls.

    Plain accuracy is maximised near the base rate by a model that barely
    discriminates, and a threshold search against it drifts toward calling
    everything the commoner class. This one is 0.5 for any constant call,
    whichever constant it is.
    """
    said = np.asarray(p_finish, dtype=float) > threshold
    real = np.asarray(finished, dtype=float) == 1
    pos, neg = real.sum(), (~real).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(0.5 * ((said & real).sum() / pos
                        + (~said & ~real).sum() / neg))


def choose_threshold(p_finish, finished, *, low=THRESHOLD_RANGE[0],
                     high=THRESHOLD_RANGE[1], steps=THRESHOLD_STEPS):
    """The threshold that maximises balanced accuracy on THIS data.

    Ties go to the value nearest a half, so a flat objective leaves the
    default in place instead of drifting to the edge of the search band on
    rounding noise.
    """
    grid = np.linspace(low, high, steps)
    scored = [(balanced_accuracy(p_finish, finished, t), -abs(t - 0.5), t)
              for t in grid]
    scored = [row for row in scored if row[0] == row[0]]
    if not scored:
        return 0.5
    return float(max(scored)[2])


def rebuild_three_way(proba, calibrated_finish):
    """Put a calibrated P(finish) back into KO / Submission / Decision.

    The split between the two finish classes is left exactly as the model had
    it: calibration was measured on the binary and says nothing about whether
    a finish arrives by knockout or by submission, so moving that split would
    be inventing a correction nobody measured.
    """
    proba = np.asarray(proba, dtype=float)
    calibrated_finish = np.asarray(calibrated_finish, dtype=float)
    raw_finish = proba[:, 1] + proba[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        share_ko = np.where(raw_finish > 0, proba[:, 1] / raw_finish, 0.5)
    out = np.column_stack([
        1.0 - calibrated_finish,
        calibrated_finish * share_ko,
        calibrated_finish * (1.0 - share_ko),
    ])
    return out
