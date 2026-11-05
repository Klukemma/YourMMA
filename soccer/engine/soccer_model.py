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
  scores     both teams to score, over 1.5 / 3.5 and the likeliest exact
             scores, from one score matrix per match (soccer_scores): the
             one that best reproduces the pick and the over/under price,
             the model's own where those are not both priced.
  the model  the best of the three, on both periods: shots-based ratings
             (experiments/shots_ratings.py) - each club's attack and defence
             from goals, shots on target and shots, turned into the chance of
             every score. SHOWN beside the pick as the model's own view,
             never blended into it. Where no price was posted, it is all
             there is, and the app says so.

The same procedure produces the backtest the app prints and this week's
numbers, so the record shown is the record of this procedure.

EVERY LEAGUE (experiments/leagues.py measured them; soccer_leagues holds what
it decided). predict(frame, leagues=[...]) runs the model over every league
in the frame (shots_ratings.predict_leagues: one rating pool per country, a
regression per league group) and gives numbers for the leagues asked for;
pick_leagues() is the pick for any league: its market price (market_price:
the pre-match average, or for an extra league whose history carries only the
close, the closing average where no pre-match price is posted), recalibrated
with weights fitted, by one of WEIGHT_OPTIONS, only on matches dated before
the league-season's first match - or the raw no-vig price for a league whose
recalibration lost to it on its own development period - and the model's
own where no price was posted. Called with the defaults, predict and pick
are what they were, bit for bit.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import soccer_backtest as bt
import soccer_data as sd
from experiments import shots_ratings

MODEL = "shots_ratings"
MIN_TRAIN = 1000


def predict(frame, leagues=None, england="all"):
    """The model's own p_h, p_d, p_a, p_o25 for every top-league row of
    `frame` (finished matches and fixtures still to play), each from matches
    on earlier days; club ratings in .attrs["ratings"]. Also each match's
    expected goals lam, nu and the Dixon-Coles rho (score_params; p_h..p_o25
    are bit for bit the same with them or without) - the whole score matrix,
    which soccer_scores reads the goals and scores numbers from.

    leagues: which leagues get numbers - None is the five top leagues on the
    frame as it is (the shipped model, one pass); a list runs every league in
    the frame by shots_ratings.predict_leagues, with England's pool `england`
    ("all": E0-EC; "top-two": E0-E1 as before - soccer_leagues.ENGLAND_POOL
    is the measured choice)."""
    if leagues is None:
        return shots_ratings.predict_walk_forward(frame, score_params=True)
    return shots_ratings.predict_leagues(frame, list(leagues), score_params=True, england=england)


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


# ------------------------------------------------------------- every league

# How a league's recalibration weights are fitted (experiments/leagues.py
# chose one on its development periods; soccer_leagues.PICK_WEIGHTS):
#   per_league  on the league's own earlier matches
#   per_price   on every league priced the same way: the European divisions'
#               pre-match prices, or the extra leagues' closing prices
#   top5        the top five's weights (pick(), fitted on their earlier
#               seasons), for every league
WEIGHT_OPTIONS = ("per_league", "per_price", "top5")
IDENTITY = (1.0, 0.0, 0.0)          # beta 1, no draw or away shift: the raw no-vig price


def market_price(frame):
    """Each row's no-vig home/draw/away from the price its league is picked
    on: the pre-match average where all three were posted; for an extra
    league (soccer_data.EXTRA), whose history has only the closing price,
    the closing average where no pre-match price was. -> (frame p_h, p_d,
    p_a - NaN where neither; Series "pre" / "close" / None)."""
    pre = bt.no_vig(frame, "pre")
    close = bt.no_vig(frame, "close")
    has_pre = pre.notna().all(axis=1)
    use_close = frame["league"].isin(sd.EXTRA) & ~has_pre & close.notna().all(axis=1)
    q = pre.where(has_pre)
    q[use_close] = close[use_close]
    kind = pd.Series(np.where(has_pre, "pre", np.where(use_close, "close", None)), index=frame.index, dtype=object)
    return q, kind


def season_starts(frame):
    """{(league, season): the date of its first match on file} - results and
    fixtures alike (a fixture list is a fact before a ball is kicked)."""
    return frame.groupby(["league", "season"])["date"].min().to_dict()


def _top5_fits(frame, q_pre):
    """pick()'s weights: per top-five season, fitted on every earlier
    top-five season - with the date of the last match they saw."""
    top = frame["league"].isin(sd.TOP)
    done = top & frame["result"].isin(bt.RESULTS) & q_pre.notna().all(axis=1)
    fits = []
    for season in sorted(frame.loc[top, "season"].dropna().unique()):
        train = done & (frame["season"] < season)
        if train.sum() < MIN_TRAIN:
            continue
        w = fit_recalibration(q_pre[train], frame.loc[train, "result"])
        fits.append((int(season), frame.loc[train, "date"].max(), w))
    return fits


def pick_leagues(frame, preds, leagues, option="top5", raw=(), min_train=None):
    """The pick for every row of `leagues` (other rows NaN): the league's
    market price (market_price) recalibrated - p proportional to
    q^beta exp(c) - with the weights `option` fits, each league-season's
    from matches dated before its first match only; the raw no-vig price for
    a league in `raw` (its recalibration lost to the raw price on its own
    development period), or where too few earlier matches to fit on
    (`min_train`, default MIN_TRAIN); the model's own where no price was
    posted (model_only). With option "top5", a top-five league-season's
    weights are pick()'s for its season, to the bit.

    Returns (frame p_h, p_d, p_a, model_only, price, basis; {(league, season):
    weights or None}). basis: "recalibrated" / "raw" / "model"."""
    if option not in WEIGHT_OPTIONS:
        raise ValueError(f"unknown weight option {option!r}")
    min_train = MIN_TRAIN if min_train is None else min_train
    leagues = list(leagues)
    q, kind = market_price(frame)
    priced = q.notna().all(axis=1).to_numpy()
    done = frame["result"].isin(bt.RESULTS).to_numpy() & priced
    league = frame["league"].to_numpy()
    dates = frame["date"].to_numpy()
    qv = q.to_numpy(float)
    p3 = preds[["p_h", "p_d", "p_a"]].to_numpy(float).copy()
    model_only = np.ones(len(frame), bool)
    basis = np.full(len(frame), "model", dtype=object)
    weights, cache = {}, {}
    top5 = _top5_fits(frame, bt.no_vig(frame, "pre")) if option == "top5" else None
    groups = frame.groupby(["league", "season"]).indices
    for (lg, season), pos in sorted(groups.items()):
        if lg not in leagues:
            continue
        first = dates[pos].min()
        rows = pos[priced[pos]]
        if not len(rows):
            continue
        w = None
        if lg in raw:
            pass
        elif option == "top5":
            if lg in sd.TOP:
                w = next((f[2] for f in top5 if f[0] == season), None)
            else:
                ok = [f for f in top5 if f[1] < first]
                w = ok[-1][2] if ok else None
        else:
            pool = ([lg] if option == "per_league" else
                    list(sd.EXTRA) if lg in sd.EXTRA else list(sd.MAIN))
            key = (tuple(pool), first)
            if key not in cache:
                train = done & np.isin(league, pool) & (dates < first)
                cache[key] = (fit_recalibration(q[train], frame.loc[train, "result"])
                              if train.sum() >= min_train else None)
            w = cache[key]
        weights[(lg, int(season))] = w
        qq = pd.DataFrame(qv[rows], columns=["p_h", "p_d", "p_a"])
        p3[rows] = recalibrate(qq, w).to_numpy() if w is not None else qv[rows]
        model_only[rows] = False
        basis[rows] = "recalibrated" if w is not None else "raw"
    mine = np.isin(league, leagues)
    p3[~mine] = np.nan
    out = pd.DataFrame(p3, index=frame.index, columns=["p_h", "p_d", "p_a"])
    out["model_only"] = model_only
    out["price"] = kind.where(mine, None)
    out["basis"] = np.where(mine, basis, None)
    return out, weights


# ------------------------------------------------------------- goals

def goals(frame, preds, leagues=None):
    """The over-2.5 chance the app shows: the over/under market with the
    cut removed - recalibrating it, as the pick is, did not improve it over
    the whole choosing period - and the model's own where no over/under
    price was posted (every match of the extra leagues, which have none).
    leagues: whose rows (default the top five)."""
    q = bt.no_vig_total(frame)
    out = preds["p_o25"].copy()
    priced = q.notna() & frame["league"].isin(sd.TOP if leagues is None else list(leagues))
    out[priced] = q[priced]
    return out
