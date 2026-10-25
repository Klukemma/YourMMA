"""What the soccer app shows, as the walk-forward research decided it.

Three goals models were built and measured against the market, 2006-2015
to choose and 2016-2025 to confirm (experiments/; the numbers are in
soccer/README.md). None beat the market's pre-match prices, and none added
to them: pooled with the market, every model lost to the market pooled with
itself on the choosing period. So:

  the pick   the market's home/draw/away chances (the pre-match prices with
             the bookmakers' cut removed), RECALIBRATED: p_k is proportional
             to q_k^beta * exp(c_k), each season's (beta, c_draw, c_away)
             fitted on every earlier season. The market's favourites win a
             little more often than its prices say, and the draw is priced a
             little short; this is the correction, measured out of sample.
  goals      the over/under 2.5 market with the cut removed. Recalibrating
             it the same way did not improve it over the choosing period.
  the model  the best of the three, on both periods: shots-based ratings
             (experiments/shots_ratings.py) - each club's attack and defence
             from goals, shots on target and shots, turned into the chance of
             every score. SHOWN beside the pick as the model's own view,
             never blended into it. Where no price was posted, it is all
             there is, and the app says so.

The same procedure produces the backtest the app prints and this week's
numbers, so the record shown is the record of this procedure.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import soccer_backtest as bt
import soccer_data as sd
from experiments import shots_ratings

MODEL = "shots_ratings"
MIN_TRAIN = 1000


def predict(frame):
    """The model's own p_h, p_d, p_a, p_o25 for every top-league row of
    `frame` (finished matches and fixtures still to play), each from matches
    on earlier days; club ratings in .attrs["ratings"]."""
    return shots_ratings.predict_walk_forward(frame)


# ------------------------------------------------------------- the pick

def _log3(q):
    return np.log(np.clip(q[["p_h", "p_d", "p_a"]].to_numpy(float), 1e-9, 1))


def recalibrate(q, w):
    """p_k proportional to q_k^beta * exp(c_k); w = (beta, c_draw, c_away)."""
    beta, c_d, c_a = w
    z = beta * _log3(q) + np.array([0.0, c_d, c_a])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return pd.DataFrame(e / e.sum(axis=1, keepdims=True), index=q.index, columns=["p_h", "p_d", "p_a"])


def fit_recalibration(q, result):
    """(beta, c_draw, c_away) minimising log loss on these matches, with an
    analytic gradient and tight tolerances (the general pooling fit stopped
    early on this surface)."""
    idx = bt.outcome_index(result).to_numpy()
    lq = _log3(q)
    onehot = np.zeros_like(lq)
    onehot[np.arange(len(idx)), idx] = 1.0

    def loss(w):
        beta, c_d, c_a = w
        z = beta * lq + np.array([0.0, c_d, c_a])
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        p = np.exp(logp)
        n = len(idx)
        g_beta = -((onehot - p) * lq).sum() / n
        g_c = -(onehot - p).sum(axis=0) / n
        return -logp[np.arange(n), idx].mean(), np.array([g_beta, g_c[1], g_c[2]])

    fit = minimize(loss, x0=np.array([1.0, 0.0, 0.0]), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 1000})
    return tuple(float(v) for v in fit.x)


def pick(frame, preds):
    """The chances the app shows for every top-league row: the recalibrated
    market where all three prices were posted, the model's own where not
    (flagged model_only). Each season uses weights fitted on earlier seasons
    only. Returns (frame of p_h, p_d, p_a, model_only; {season: weights})."""
    q = bt.no_vig(frame, "pre")
    priced = q.notna().all(axis=1)
    out = preds[["p_h", "p_d", "p_a"]].copy()
    out["model_only"] = True
    weights = {}
    top = frame["league"].isin(sd.TOP)
    for season in sorted(frame.loc[top & priced, "season"].unique()):
        train = bt.scored(frame) & (frame["season"] < season)
        if train.sum() < MIN_TRAIN:
            continue
        w = fit_recalibration(q[train], frame.loc[train, "result"])
        weights[int(season)] = w
        rows = top & priced & (frame["season"] == season)
        out.loc[rows, ["p_h", "p_d", "p_a"]] = recalibrate(q[rows], w).to_numpy()
        out.loc[rows, "model_only"] = False
    out.loc[~top, ["p_h", "p_d", "p_a"]] = np.nan
    return out, weights


# ------------------------------------------------------------- goals

def goals(frame, preds):
    """The over-2.5 chance the app shows: the over/under market with the
    cut removed - recalibrating it, as the pick is, did not improve it over
    the whole choosing period - and the model's own where no over/under
    price was posted."""
    q = bt.no_vig_total(frame)
    out = preds["p_o25"].copy()
    priced = q.notna() & frame["league"].isin(sd.TOP)
    out[priced] = q[priced]
    return out
