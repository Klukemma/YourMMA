"""The soccer model the app runs, and the blend with the market.

  model   CHOSEN by the walk-forward research in soccer/engine/experiments
          (see soccer/README.md for the numbers): every match - finished or
          still to play - predicted only from matches on earlier days.
  blend   logarithmic pooling of the model's home/draw/away chances with the
          market's (the pre-match prices with the bookmakers' cut removed),
          each season's weights fitted on the model's OUT-OF-SAMPLE
          predictions for earlier seasons only (soccer_backtest.fit_pool).
  goals   the chance of three or more goals, pooled the same way with the
          over/under 2.5 market where one was posted.

The same procedure produces the backtest the app prints and this week's
picks, so the record shown is the record of this procedure.
"""

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd
from experiments import dixon_coles, elo_goals

# Which model the app runs: "dixon_coles", "elo_goals" or "ensemble" (the
# two averaged in log space). Set from the research verdict.
MODEL = "ensemble"
# Constrain the model's pooling weight to be >= 0: a negative weight is a
# contrarian signal, not a forecast. Set from the research verdict.
NONNEGATIVE = True


def _ensemble(frames):
    """Equal-weight average of log-probabilities, renormalised; the mean of
    the over-2.5 chances."""
    logs = np.mean([np.log(np.clip(f[["p_h", "p_d", "p_a"]].to_numpy(float), 1e-9, 1)) for f in frames], axis=0)
    p = np.exp(logs)
    p /= p.sum(axis=1, keepdims=True)
    out = pd.DataFrame(p, index=frames[0].index, columns=["p_h", "p_d", "p_a"])
    out["p_o25"] = np.mean([f["p_o25"].to_numpy(float) for f in frames], axis=0)
    missing = np.any([f[["p_h", "p_d", "p_a"]].isna().any(axis=1).to_numpy() for f in frames], axis=0)
    out.loc[missing, ["p_h", "p_d", "p_a"]] = np.nan
    return out


def predict(frame, model=MODEL):
    """p_h, p_d, p_a, p_o25 for every top-league row of `frame` (finished
    matches and fixtures still to play), each from matches on earlier days."""
    if model == "dixon_coles":
        return dixon_coles.predict_walk_forward(frame)
    if model == "elo_goals":
        return elo_goals.predict_walk_forward(frame)
    return _ensemble([dixon_coles.predict_walk_forward(frame), elo_goals.predict_walk_forward(frame)])


# ------------------------------------------------------------- the blend

def fit_weights(frame, preds, before_season, nonnegative=NONNEGATIVE):
    """Pooling weights from every scored match of seasons before
    `before_season` (model out of sample, market pre-match)."""
    mkt = bt.no_vig(frame, "pre")
    ok = bt.scored(frame) & preds[["p_h", "p_d", "p_a"]].notna().all(axis=1) & (frame["season"] < before_season)
    if ok.sum() < 1000:
        return None
    return bt.fit_pool(preds[ok], mkt[ok], frame.loc[ok, "result"], nonnegative=nonnegative)


def blend(frame, preds, nonnegative=NONNEGATIVE):
    """Blended home/draw/away for every row with a model number and a full
    pre-match price, each season pooled with weights from earlier seasons.
    Rows without a price keep the model's own chances and are flagged
    model_only. Returns (frame of p_h, p_d, p_a, model_only; {season: w})."""
    mkt = bt.no_vig(frame, "pre")
    out = preds[["p_h", "p_d", "p_a"]].copy()
    out["model_only"] = True
    weights = {}
    has = preds[["p_h", "p_d", "p_a"]].notna().all(axis=1)
    for season in sorted(frame.loc[has, "season"].unique()):
        w = fit_weights(frame, preds, season, nonnegative)
        if w is None:
            continue
        weights[int(season)] = w
        rows = has & (frame["season"] == season) & mkt.notna().all(axis=1)
        if rows.any():
            out.loc[rows, ["p_h", "p_d", "p_a"]] = bt.pool(preds[rows], mkt[rows], w).to_numpy()
            out.loc[rows, "model_only"] = False
    out.loc[~has, ["p_h", "p_d", "p_a"]] = np.nan
    return out, weights


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def goals(frame, preds):
    """The over-2.5 chance used: model and market pooled on the logit scale,
    each season fitted on earlier seasons; the model alone where no
    over/under price was posted."""
    from scipy.optimize import minimize
    mkt = bt.no_vig_total(frame)
    total = frame["hg"] + frame["ag"]
    out = preds["p_o25"].copy()
    has = preds["p_o25"].notna() & mkt.notna()
    for season in sorted(frame.loc[has, "season"].unique()):
        train = has & (frame["season"] < season) & total.notna() & frame["league"].isin(sd.TOP)
        if train.sum() < 1000:
            continue
        a, b = _logit(preds.loc[train, "p_o25"]), _logit(mkt[train])
        y = (total[train] > 2.5).astype(float).to_numpy()

        def loss(w):
            z = w[0] * a + w[1] * b + w[2]
            return float(np.mean(np.logaddexp(0, z) - y * z))

        w = minimize(loss, x0=np.array([0.1, 0.9, 0.0]), method="L-BFGS-B").x
        rows = has & (frame["season"] == season)
        z = w[0] * _logit(preds.loc[rows, "p_o25"]) + w[1] * _logit(mkt[rows]) + w[2]
        out[rows] = 1 / (1 + np.exp(-z))
    return out
