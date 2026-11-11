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
recalibration lost to it on its own development period - and, where no price
was posted, the model's own (or no pick, for a league whose model did not
beat its base rates). Measured and chosen on the development periods:
England's pool stays E0-E1 (soccer_leagues.ENGLAND_POOL) and the weights are
pooled per price type (PICK_WEIGHTS = "per_price": every European division's
pre-match prices, every extra league's closing prices) - which changes the
top five's weights too, and makes the Bundesliga's pick the raw market.
shipped_pick() is pick_leagues with those decisions; predict(frame,
leagues=...) takes England's pool from soccer_leagues by default. Both
refuse a frame without every measured league (require_every_league): the
pools and regressions learn from leagues the app does not show (RUS), so a
frame of the app's leagues alone gives numbers nobody measured, silently.
goals(frame, preds, leagues) is the over 2.5 beside each card, NaN where
the league did not ship it from that source (rule c) and outside `leagues`;
latest_weights() is each league's newest weights, as the Record tab prints
them. Called without leagues, predict, pick and goals are what they were,
bit for bit: the running app is unchanged until the pipeline switches to
these.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import soccer_backtest as bt
import soccer_data as sd
from experiments import shots_ratings

MODEL = "shots_ratings"
MIN_TRAIN = 1000


def predict(frame, leagues=None, england=None):
    """The model's own p_h, p_d, p_a, p_o25 for every row of `leagues` in
    `frame` (finished matches and fixtures still to play; NaN elsewhere),
    each from matches on earlier days; club ratings in .attrs["ratings"].
    Also each match's expected goals lam, nu and the Dixon-Coles rho
    (score_params; p_h..p_o25 are bit for bit the same with them or without)
    - the whole score matrix, which soccer_scores reads the goals and scores
    numbers from.

    leagues: which leagues get numbers - None is the five top leagues on the
    frame as it is (the shipped model, one pass); a list runs every league in
    the frame by shots_ratings.predict_leagues, with England's pool `england`
    ("all": E0-EC; "top-two": E0-E1 as before; None, the default: the
    measured choice, soccer_leagues.ENGLAND_POOL). With a list, the frame
    must hold every league that was measured (require_every_league): a
    country's pool and a group's regression learn from every league of them
    in the frame, so a league left out moves the others' numbers to ones
    nobody measured."""
    if leagues is None:
        return shots_ratings.predict_walk_forward(frame, score_params=True)
    if england is None:
        import soccer_leagues
        england = soccer_leagues.ENGLAND_POOL
    if england not in ("all", "top-two"):
        raise ValueError(f"England's pool is not decided ({england!r}): soccer_leagues has no measured decisions")
    require_every_league(frame)
    return shots_ratings.predict_leagues(frame, list(leagues), score_params=True, england=england)


def require_every_league(frame):
    """Raise unless `frame` holds a match of every league experiments/leagues.py
    measured (soccer_leagues.SHIPPED) - those not shown included: RUS's
    matches train the extra leagues' regression and, priced at the close,
    their recalibration weights. Measured on the frame of all 38: without
    RUS the 2026 weights of all twelve recalibrated extra leagues move (ARG
    (1.106, 0.0297, -0.0015) became (1.1068, 0.0322, -0.0009)), with no
    error - numbers that were never measured. Load every league
    (soccer_data.load_matches(leagues="all"))."""
    import soccer_leagues as sl
    have = set(frame["league"].unique())
    missing = [c for c in sl.SHIPPED if c not in have]
    if missing:
        raise ValueError(f"the frame lacks measured leagues {missing}: their matches train the model and the "
                         "pick's weights, so without them neither is the one measured - load every league")


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


def fit_recalibration(q, result, x0=None):
    """(beta, c_draw, c_away) minimising log loss on these matches, with an
    analytic gradient and tight tolerances (the general pooling fit stopped
    early on this surface). x0: where to start (default (1, 0, 0), no
    recalibration) - the log loss is convex in the weights, so a start only
    changes how many steps it takes, within the tolerances."""
    return _fit_log3(_log3(q), bt.outcome_index(result).to_numpy(), x0)


def _fit_log3(lq, idx, x0=None):
    """fit_recalibration on log prices lq (n, 3) and outcome indices idx."""
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

    start = np.array([1.0, 0.0, 0.0]) if x0 is None else np.array(x0, float)
    fit = minimize(loss, x0=start, jac=True, method="L-BFGS-B",
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


# The price each league is picked on, and scored against: the pre-match
# average, or for an extra league's history (closing prices only) the close.
# It lives with the backtest, which scores every league against it.
market_price = bt.market_price


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


def _pool_fits(frame, q, done, min_train, keys):
    """{(pool, first date): weights fitted on the pool's priced, finished
    matches dated before that day (None: fewer than min_train)}. Per pool,
    in date order, each fit started from the one before it - fitted on
    earlier matches only, so a start never carries anything from the day on
    (the log loss is convex: the start changes only the steps taken)."""
    league = frame["league"].to_numpy()
    dates = frame["date"].to_numpy()
    lq = _log3(q)
    res = bt.outcome_index(frame["result"]).to_numpy()
    out = {}
    for pool in sorted({k[0] for k in keys}):
        pos = np.flatnonzero(done & np.isin(league, list(pool)))
        pos = pos[np.argsort(dates[pos], kind="stable")]
        on = dates[pos]
        prev = None
        for first in sorted(k[1] for k in keys if k[0] == pool):
            n = int(np.searchsorted(on, first, side="left"))
            if n < min_train:
                out[(pool, first)] = None
                continue
            tr = pos[:n]
            prev = out[(pool, first)] = _fit_log3(lq[tr], res[tr].astype(int), prev)
    return out


def pick_leagues(frame, preds, leagues, option="top5", raw=(), min_train=None, model_alone=None):
    """The pick for every row of `leagues` (other rows NaN): the league's
    market price (market_price) recalibrated - p proportional to
    q^beta exp(c) - with the weights `option` fits, each league-season's
    from matches dated before its first match only; the raw no-vig price for
    a league in `raw` (its recalibration lost to the raw price on its own
    development period), or where too few earlier matches to fit on
    (`min_train`, default MIN_TRAIN); where no price was posted (model_only),
    the model's own for a league in `model_alone` (default: every league) and
    no pick at all (NaN, basis "unpriced": "no pick until priced") for the
    others - soccer_leagues.MODEL_ALONE holds which leagues' model beat their
    base rates. With option "top5", a top-five league-season's weights are
    pick()'s for its season, to the bit.

    Returns (frame p_h, p_d, p_a, model_only, price, basis; {(league, season):
    weights or None}). basis: "recalibrated" / "raw" / "model" /
    "unpriced"."""
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
    weights = {}
    top5 = _top5_fits(frame, bt.no_vig(frame, "pre")) if option == "top5" else None
    groups = frame.groupby(["league", "season"]).indices
    jobs = []
    for (lg, season), pos in sorted(groups.items()):
        if lg not in leagues:
            continue
        rows = pos[priced[pos]]
        if len(rows):
            pool = ([lg] if option == "per_league" else
                    tuple(sd.EXTRA) if lg in sd.EXTRA else tuple(sd.MAIN))
            jobs.append((lg, int(season), dates[pos].min(), rows, tuple(pool)))
    fits = {} if option == "top5" else _pool_fits(frame, q, done, min_train,
                                                  {(j[4], j[2]) for j in jobs if j[0] not in raw})
    for lg, season, first, rows, pool in jobs:
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
            w = fits[(pool, first)]
        weights[(lg, season)] = w
        qq = pd.DataFrame(qv[rows], columns=["p_h", "p_d", "p_a"])
        p3[rows] = recalibrate(qq, w).to_numpy() if w is not None else qv[rows]
        model_only[rows] = False
        basis[rows] = "recalibrated" if w is not None else "raw"
    mine = np.isin(league, leagues)
    p3[~mine] = np.nan
    if model_alone is not None:
        unpriced = mine & model_only & ~np.isin(league, list(model_alone))
        p3[unpriced] = np.nan
        basis[unpriced] = "unpriced"
    out = pd.DataFrame(p3, index=frame.index, columns=["p_h", "p_d", "p_a"])
    out["model_only"] = model_only
    out["price"] = kind.where(mine, None)
    out["basis"] = np.where(mine, basis, None)
    return out, weights


def shipped_pick(frame, preds, leagues=None):
    """The pick as experiments/leagues.py decided it, for `leagues` (default
    every league in the app, soccer_leagues.in_app(); each must be one it
    measured): pick_leagues with the measured weight option, the raw price
    where rule (a) found the recalibration lost, and the model's own where
    no price was posted only in a league whose model beat its base rates in
    both periods (rule b) - "unpriced" elsewhere. The frame must hold every
    measured league (require_every_league): the weights are pooled over
    every league priced the same way, shown or not."""
    import soccer_leagues as sl
    if sl.PICK_WEIGHTS not in WEIGHT_OPTIONS:
        raise ValueError(f"the weight option is not decided ({sl.PICK_WEIGHTS!r}): "
                         "soccer_leagues has no measured decisions")
    leagues = sl.in_app() if leagues is None else list(leagues)
    unmeasured = [c for c in leagues if c not in sl.SHIPPED]
    if unmeasured:
        raise ValueError(f"not measured, so never picked: {unmeasured}")
    require_every_league(frame)
    return pick_leagues(frame, preds, leagues, sl.PICK_WEIGHTS, raw=sl.raw_leagues(leagues),
                        model_alone=sl.model_alone(leagues))


def latest_weights(weights, leagues=None):
    """{code: {"season", "w"} or None} from pick_leagues' {(league, season):
    weights or None}: each league's newest season with fitted weights, to 4
    decimals - what the Record tab prints, and what soccer_leagues.SHIPPED
    stores beside each league ("weights"). None for a league picked on the
    raw price (rule a), or with no season fitted yet. pick()'s {season: w}
    is a different shape: one set for the top five, keyed by season alone.
    leagues: which codes (default every league in `weights`, in
    soccer_data's order)."""
    have = {lg for lg, _ in weights}
    codes = [c for c in sd.ALL_LEAGUES if c in have] if leagues is None else list(leagues)
    out = {}
    for c in codes:
        fitted = sorted((s, w) for (lg, s), w in weights.items() if lg == c and w is not None)
        out[c] = {"season": int(fitted[-1][0]), "w": [round(float(v), 4) for v in fitted[-1][1]]} if fitted else None
    return out


# ------------------------------------------------------------- goals

def goals(frame, preds, leagues=None):
    """The over-2.5 chance the app shows, for the rows of `leagues` (default
    the top five; NaN for every other row): the over/under market with the
    cut removed - recalibrating it, as the pick is, did not improve it over
    the whole choosing period - and the model's own where no over/under
    price was posted (every match of the extra leagues, which have none).

    NaN, too, where the league did not ship over 2.5 from that source (rule
    c, soccer_leagues.withheld(code, source): "market" where the over/under
    is posted, "model" where the number is the model's): the card withholds
    it, and a number shown beside the card must not bring it back - in the
    run of 2026-11 that was ARG, NOR and POL's model number and SC2's
    market one. NaN outside `leagues` rather than the model's own, so a
    caller that forgets to pass the leagues shows nothing instead of the
    model's number where the market's was posted. Neither changes the top
    five: every one ships over 2.5 from both sources, and the model has no
    numbers outside them when predict() runs without leagues."""
    import soccer_leagues as sl
    q = bt.no_vig_total(frame)
    mine = frame["league"].isin(sd.TOP if leagues is None else list(leagues))
    posted = q.notna()
    out = preds["p_o25"].copy()
    out[posted & mine] = q[posted & mine]
    hide = ~mine
    for code in frame.loc[mine, "league"].unique():
        here = mine & (frame["league"] == code)
        for source, rows in (("market", here & posted), ("model", here & ~posted)):
            if "o25" in sl.withheld(code, source):
                hide |= rows
    out[hide] = np.nan
    return out
