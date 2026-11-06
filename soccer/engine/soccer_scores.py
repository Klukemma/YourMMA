"""The numbers beside the pick, read off one score matrix per match: both
teams to score, over 1.5 and over 3.5 goals, the three likeliest exact
scores - and how sure the pick is (its band).

experiments/score_markets.py built and measured these (walk-forward, chosen
on 2006-2015, confirmed once on 2016-2025; its docstring and soccer/README.md
hold the numbers). This module is what it measured, moved here so the app
and the experiment run the same code: the experiment imports every function
below and reproduces its numbers bit for bit.

THE MATRIX. A Dixon-Coles matrix tau_rho(x, y) Pois(x; lam) Pois(y; nu),
0-10 goals a side, renormalised (dc_matrix). Two sources:

  market  (K) per match, the (lam, nu) whose matrix reproduces, in least
          squares, the pick's home/draw/away chances (the recalibrated
          market, soccer_model.pick) and the over/under 2.5 price with the
          cut removed, with the model's rho for that match. Chosen on
          2006-2015 exact-score log loss over the model's own matrix and
          the market total split the model's way.
  model   (M) the model's own matrix: shots_ratings' lam, nu and rho for the
          match. Measured too; a little worse than K in both periods.

WHAT THE APP SHOWS (shown_matrices), for every match:
  K       where the home/draw/away AND the over/under 2.5 are priced - with
          the fit's largest miss against its four targets on the card, and
          FLAGGED where that is more than FIT_MAX_MISS;
  M       where only the home/draw/away is priced (K's 1X2-only path was
          measured on 1 match a period - effectively unmeasured) or where
          nothing is priced - each said on the card;
  nothing where the model has no rates for the match (missing, said so).

FIT_MAX_MISS = 0.04 flags a fit; it never swaps the matrix. It was set on
2006-2015 from the distribution of K's per-match largest miss against its
four targets - a body (median 0.006, 99th percentile 0.0387) and a long tail
(0.056 at the 99.5th, 0.17 at the 99.9th) of matches whose prices contradict
any Poisson matrix, mostly Serie A 2006-2010 run-ins: the 99th percentile,
rounded up to the next 0.01. It catches 168 of 18,258 fits on 2006-2015 and 8
of 17,935 on 2016-2025 (none of 2026-27's first 346). DEV outcomes per
candidate threshold (0.03-0.10) were looked at too, and an earlier rule that
showed M on those matches was dropped on them: there, M misses the card's
prices by MORE than K (median largest miss 0.10 against 0.06; in 163 of the
176 such matches over both periods) and scores worse (2006-2015 exact-score
log loss 2.797 against K's 2.753; 2016-2025 2.858 against 2.857). So K is
shown and the card says how far it misses and on which price - in 4 of those
168 DEV matches K is near-degenerate (its likeliest score above 30%; worst:
over 2.5 at 0.9% from a matrix that reproduced a 71% draw, against a 43%
over/under price). measure() reports the rule as shipped.

Over 2.5 on the card stays the over/under market's (K's over 2.5 scores as
the market's does - 2016-2025 log loss 0.67067 against 0.67066: nothing to gain). The pick's own home/draw/away stays as shown;
the matrix's is shown beside it, never in its place (K's draw is on average
0.005 below the pick's).

THE BANDS. How sure the pick is, by its top probability, at the 2006-2015
quartiles (0.423, 0.497, 0.616) rounded to 0.05: STRONG >= 0.60, SOLID
0.50-0.60, LEAN 0.40-0.50, TOSS-UP below. Each band is shown with the hit rate
MEASURED for it (measure): the market pick's where the pick is the market's,
the model's own picks' at the SAME edges where it is the model's alone.

CAVEATS (never deleted): both teams to score runs LOW (2016-2025: K's mean
0.511 against 0.537, low in every decile); exact scores are long shots (the
likeliest happens about one match in eight); the numbers from M where no
price is posted are the model's alone.
"""

import math

import numpy as np
import pandas as pd

import soccer_backtest as bt
import soccer_data as sd

MAX_GOALS = 10                    # the matrix runs 0-10 each way, as the model's does (shots_ratings.MAX_GOALS)
GRID = 5                          # exact scores 0-0 .. 5-5, then one "other" bucket
N_CELLS = (GRID + 1) ** 2 + 1     # 37 categories
LAM_RANGE = (0.05, 8.0)           # as the model clips its own expected goals
# Edges at the DEV quartiles of the pick's top probability, rounded to 0.05
# (score_markets.band_edges re-derives them from DEV on every run).
BANDS = (("STRONG", 0.60), ("SOLID", 0.50), ("LEAN", 0.40), ("TOSS-UP", 0.0))
# The largest miss of the market-implied matrix against any of its four
# targets beyond which the card flags it (see the notes): a flag, never a swap.
FIT_MAX_MISS = 0.04
TARGET_NAMES = ("home win", "draw", "away win", "over 2.5")
SOURCE_MARKET, SOURCE_MODEL = "market", "model"
# Why a card's score numbers are the model's own (BAD_FIT: an earlier rule's,
# kept so a pick logged under it still reads).
NO_PRICE, NO_OVER_UNDER, BAD_FIT, NO_MODEL = "no_price", "no_over_under", "fit", "no_model"


# ----------------------------------------------------------- the matrix

def dc_matrix(lam, nu, rho, max_goals=MAX_GOALS):
    """The Dixon-Coles score matrix, (n, G+1, G+1), [x, y] = P(home x, away
    y): independent Poissons with the low-score correction tau_rho, truncated
    at `max_goals` each way and renormalised - the matrix behind
    shots_ratings.score_probs. lam, nu, rho: scalars or arrays of length n;
    a NaN anywhere in a row gives a NaN matrix for that row."""
    lam = np.atleast_1d(np.asarray(lam, float))
    nu = np.atleast_1d(np.asarray(nu, float))
    rho = np.broadcast_to(np.asarray(rho, float), lam.shape)
    g = np.arange(max_goals + 1)
    fact = np.array([math.factorial(k) for k in g], float)
    ph = np.exp(-lam)[:, None] * lam[:, None] ** g / fact
    pa = np.exp(-nu)[:, None] * nu[:, None] ** g / fact
    mat = ph[:, :, None] * pa[:, None, :]
    mat[:, 0, 0] *= 1 - lam * nu * rho
    mat[:, 0, 1] *= 1 + lam * rho
    mat[:, 1, 0] *= 1 + nu * rho
    mat[:, 1, 1] *= 1 - rho
    return mat / mat.sum(axis=(1, 2), keepdims=True)


def _masks(size):
    x = np.arange(size)[:, None]
    y = np.arange(size)[None, :]
    return x > y, x == y, x < y, x + y


def markets(mats):
    """Everything the app could show from score matrices (n, G+1, G+1), as
    arrays of length n: p_h, p_d, p_a, btts, o15, o25, o35, and `cells`
    (n, 37): the chance of each score 0-0 .. 5-5 (index 6 * home + away)
    then of any other score."""
    mats = np.ascontiguousarray(mats, float)
    if mats.ndim == 2:
        mats = mats[None]
    home, draw, away, tot = _masks(mats.shape[1])
    both = np.zeros_like(home)
    both[1:, 1:] = True

    def total(mask):
        # every row summed over its own whole contiguous matrix: the rounding
        # never depends on how many rows are in the batch (fancy-indexed sums do)
        return (mats * mask).sum(axis=(1, 2))
    cells = np.ascontiguousarray(mats[:, :GRID + 1, :GRID + 1]).reshape(len(mats), -1)
    other = np.clip(1.0 - total(np.pad(np.ones((GRID + 1, GRID + 1), bool), (0, mats.shape[1] - GRID - 1))), 0.0, 1.0)
    return {
        "p_h": total(home), "p_d": total(draw), "p_a": total(away), "btts": total(both),
        "o15": total(tot >= 2), "o25": total(tot >= 3), "o35": total(tot >= 4),
        "cells": np.column_stack([cells, other]),
    }


def score_label(k):
    """Category index -> '2-1' (home-away), or 'other'."""
    return "other" if k == N_CELLS - 1 else f"{k // (GRID + 1)}-{k % (GRID + 1)}"


def score_index(hg, ag):
    """The category of each final score: 6 * home + away, or 36 (other)."""
    hg, ag = np.asarray(hg, float), np.asarray(ag, float)
    inside = (hg <= GRID) & (ag <= GRID)
    return np.where(inside, hg * (GRID + 1) + ag, N_CELLS - 1).astype(int)


def summary(mat, top=3):
    """One match's score matrix (G+1, G+1) -> the numbers a match card can
    show: both teams to score, over 1.5 / 2.5 / 3.5, the `top` most likely
    exact scores with their chances, and the chance of a score beyond 5-5.
    None for a missing (NaN) matrix."""
    mat = np.asarray(mat, float)
    if not np.isfinite(mat).all():
        return None
    mk = markets(mat[None])
    cells = mk["cells"][0]
    order = np.argsort(-cells[:-1], kind="stable")[:top]
    return {
        "btts": float(mk["btts"][0]), "o15": float(mk["o15"][0]), "o25": float(mk["o25"][0]),
        "o35": float(mk["o35"][0]),
        "scores": [{"score": score_label(k), "p": float(cells[k])} for k in order],
        "other": float(cells[-1]),
    }


def band_of(p_top, bands=BANDS):
    """The confidence band of a pick from its top probability (None if missing)."""
    if p_top is None or p_top != p_top:
        return None
    for name, floor in bands:
        if p_top >= floor:
            return name
    return bands[-1][0]


# ----------------------------------------------------------- market-implied rates

def _targets(mats):
    mk = markets(mats)
    return np.column_stack([mk["p_h"], mk["p_d"], mk["p_a"], mk["o25"]])


def _normal_step(J, r):
    """The Gauss-Newton step -(J'J)^-1 J'r for every row, J (n, m, k), r (n, m),
    in elementwise arithmetic only: a row's step never depends on which other
    rows share the batch. (A batched einsum or solve rounds differently with
    the batch's size; on an ill-posed fit that moved a number in the tenth
    decimal when prices AFTER the match changed - the point-in-time check
    caught it.)"""
    m, k = J.shape[1], J.shape[2]
    A = [[sum(J[:, i, a] * J[:, i, b] for i in range(m)) for b in range(k)] for a in range(k)]
    g = [sum(J[:, i, a] * r[:, i] for i in range(m)) for a in range(k)]
    if k == 2:
        a, b, c, d = A[0][0] + 1e-12, A[0][1], A[1][0], A[1][1] + 1e-12
        det = a * d - b * c
        return np.column_stack([-(d * g[0] - b * g[1]) / det, -(a * g[1] - c * g[0]) / det])
    M = np.stack([np.stack(row, axis=-1) for row in A], axis=1) + 1e-12 * np.eye(k)
    return -np.linalg.solve(M, np.stack(g, axis=1)[:, :, None])[:, :, 0]     # variants only


def _gauss_newton(th, model, target, w, project, iters, tol, max_step):
    """Vectorised Gauss-Newton: each row's parameters th[i] moved to minimise
    sum_k (w_ik (model(th)_ik - target_ik))^2, forward-difference Jacobian.
    model(th_rows, row_index) -> (rows, 4); project(th_rows) keeps them legal.
    Rows stop once their step is below tol."""
    eps = 1e-6
    live = np.ones(len(th), dtype=bool)
    k_par = th.shape[1]
    for _ in range(iters):
        if not live.any():
            break
        idx = np.flatnonzero(live)
        t = th[idx]
        f = model(t, idx)
        r = (f - target[idx]) * w[idx]
        J = np.empty((len(idx), target.shape[1], k_par))
        for k in range(k_par):
            tk = t.copy()
            tk[:, k] += eps
            J[:, :, k] = (model(tk, idx) - f) / eps * w[idx]
        step = np.clip(_normal_step(J, r), -max_step, max_step)
        th[idx] = project(t + step)
        live[idx] = np.abs(step).max(axis=1) > tol
    return th


def implied_rates(p3, p_o25, rho, lam0=None, nu0=None, iters=60, tol=1e-10):
    """(lam, nu) whose Dixon-Coles matrix (with the given rho) comes closest,
    in least squares, to the home/draw/away chances p3 (n, 3) and the
    over-2.5 chance p_o25 (n,; NaN where no over/under price was posted: then
    the 1X2 alone). Gauss-Newton on (log lam, log nu), started from (lam0,
    nu0) where given. Returns (lam, nu, rmse): rmse is the root mean squared
    miss over the targets used, in probability. Rows with a NaN in p3 or rho
    come back NaN."""
    p3 = np.asarray(p3, float).reshape(-1, 3)
    n = len(p3)
    po = np.broadcast_to(np.asarray(np.nan if p_o25 is None else p_o25, float), (n,))
    rho = np.broadcast_to(np.asarray(rho, float), (n,))
    lam = np.full(n, np.nan)
    nu = np.full(n, np.nan)
    rmse = np.full(n, np.nan)
    ok = np.isfinite(p3).all(axis=1) & np.isfinite(rho)
    if not ok.any():
        return lam, nu, rmse
    p3, po, rho_ok = p3[ok], po[ok], rho[ok]
    has_o = np.isfinite(po)
    target = np.column_stack([p3, np.where(has_o, po, 0.0)])
    w = np.column_stack([np.ones((len(p3), 3)), has_o.astype(float)])
    lo, hi = np.log(LAM_RANGE[0]), np.log(LAM_RANGE[1])

    def start(x, default):
        if x is None:
            return np.full(len(p3), default)
        x = np.broadcast_to(np.asarray(x, float), (n,))[ok]
        return np.where(np.isfinite(x), x, default)
    th = np.clip(np.log(np.column_stack([start(lam0, 1.45), start(nu0, 1.15)])), lo, hi)

    def model(t, idx):
        return _targets(dc_matrix(np.exp(t[:, 0]), np.exp(t[:, 1]), rho_ok[idx]))
    th = _gauss_newton(th, model, target, w, lambda t: np.clip(t, lo, hi), iters, tol, 0.5)
    f = model(th, np.arange(len(th)))
    lam[ok], nu[ok] = np.exp(th[:, 0]), np.exp(th[:, 1])
    rmse[ok] = np.sqrt((((f - target) * w) ** 2).sum(axis=1) / w.sum(axis=1))
    return lam, nu, rmse


def score_matrix(p_h, p_d, p_a, p_o25=None, rho=None, lam=None, nu=None):
    """The matrix as score_markets MEASURED it. Score matrices (n, 11, 11)
    for n matches:

      priced      (p_h, p_d, p_a finite) the market-implied matrix - the
                  (lam, nu) that best reproduce the recalibrated market 1X2
                  (soccer_model.pick) and the no-vig over 2.5 (where posted;
                  pass NaN or None where not), with the model's Dixon-Coles
                  rho for that match (shots_ratings, score_params=True)
      not priced  (p_h NaN) the model's own matrix from its lam, nu, rho

    Returns (mats, info): info["from_market"] (bool), info["lam"], info["nu"]
    (the rates behind each matrix), info["rmse"] (how closely the market was
    reproduced; NaN for the model's own). A row with neither a price nor the
    model's rates is NaN - shown as missing. The app uses shown_matrices,
    which adds the rules for a missing over/under price and a bad fit."""
    p3 = np.column_stack([np.atleast_1d(np.asarray(v, float)) for v in (p_h, p_d, p_a)])
    n = len(p3)
    rho = np.broadcast_to(np.asarray(np.nan if rho is None else rho, float), (n,))
    lam_m = np.broadcast_to(np.asarray(np.nan if lam is None else lam, float), (n,))
    nu_m = np.broadcast_to(np.asarray(np.nan if nu is None else nu, float), (n,))
    priced = np.isfinite(p3).all(axis=1)
    lam_k, nu_k, rmse = implied_rates(np.where(priced[:, None], p3, np.nan), p_o25, rho, lam_m, nu_m)
    lam_out = np.where(priced, lam_k, lam_m)
    nu_out = np.where(priced, nu_k, nu_m)
    mats = dc_matrix(lam_out, nu_out, rho)
    return mats, {"from_market": priced & np.isfinite(lam_k), "lam": lam_out, "nu": nu_out, "rmse": rmse}


# ----------------------------------------------------------- what the app shows

def fit_miss(mats, p_h, p_d, p_a, p_o25):
    """Each row's largest absolute miss of its matrix against the targets it
    was fitted to - home, draw, away and over 2.5, the over 2.5 only where it
    is given - and which one (an index into TARGET_NAMES); NaN / -1 where
    the matrix or the 1X2 is missing."""
    got = _targets(np.asarray(mats, float))
    want = np.column_stack([np.atleast_1d(np.asarray(v, float)) for v in (p_h, p_d, p_a, p_o25)])
    err = np.abs(got - want)
    ok = np.isfinite(err[:, :3]).all(axis=1)
    filled = np.where(np.isfinite(err), err, -1.0)
    return np.where(ok, filled.max(axis=1), np.nan), np.where(ok, filled.argmax(axis=1), -1)


def shown_matrices(p_h, p_d, p_a, p_o25, rho, lam, nu, max_miss=FIT_MAX_MISS, need_o25=True):
    """THE PRODUCTION CALL: the score matrix each card shows, by the rule in
    the notes, for n matches. p_h..p_a: the pick's chances where the market
    priced the match (NaN where the pick is the model's alone); p_o25: the
    over/under price with the cut removed (NaN where not posted); rho, lam,
    nu: the model's (shots_ratings, score_params=True; NaN where it has none).

    Returns (mats, info): info["source"] "market" / "model" / None (no
    numbers), info["reason"] why the model's (no_price, no_over_under,
    no_model), info["miss"] K's largest miss against its targets wherever K
    is shown, info["miss_on"] which target, info["flagged"] where that miss
    is above max_miss. need_o25=False gives score_markets' measured K
    exactly."""
    p3 = np.column_stack([np.atleast_1d(np.asarray(v, float)) for v in (p_h, p_d, p_a)])
    n = len(p3)
    po = np.broadcast_to(np.asarray(np.nan if p_o25 is None else p_o25, float), (n,)).copy()
    rho = np.broadcast_to(np.asarray(np.nan if rho is None else rho, float), (n,))
    lam = np.broadcast_to(np.asarray(np.nan if lam is None else lam, float), (n,))
    nu = np.broadcast_to(np.asarray(np.nan if nu is None else nu, float), (n,))
    priced = np.isfinite(p3).all(axis=1)
    has_o = np.isfinite(po)
    use_k = priced & (has_o if need_o25 else True)
    k3 = np.where(use_k[:, None], p3, np.nan)
    mats, info = score_matrix(k3[:, 0], k3[:, 1], k3[:, 2], po, rho, lam, nu)
    market = info["from_market"]
    miss, which = fit_miss(mats, k3[:, 0], k3[:, 1], k3[:, 2], po)
    miss = np.where(market, miss, np.nan)
    which = np.where(market, which, -1)
    model_ok = np.isfinite(lam) & np.isfinite(nu) & np.isfinite(rho)
    source = np.where(market, SOURCE_MARKET, np.where(model_ok, SOURCE_MODEL, None))
    reason = np.where(market, None, np.where(~model_ok, NO_MODEL, np.where(priced, NO_OVER_UNDER, NO_PRICE)))
    mats = mats.copy()
    mats[~(market | model_ok)] = np.nan
    return mats, {"source": source, "reason": reason, "miss": miss, "miss_on": which,
                  "flagged": market & (miss > max_miss),
                  "lam": np.where(market, info["lam"], lam), "nu": np.where(market, info["nu"], nu),
                  "rho": np.asarray(rho, float)}


# The numbers a card can carry that a league may withhold (soccer_leagues:
# each ships only where its source beat the league's own rates in both
# periods). "exact" is the likeliest scores and the beyond-5-5 chance.
NUMBERS = ("btts", "o15", "o25", "o35", "exact")


def card(mat, source, reason, miss=None, miss_on=-1, max_miss=FIT_MAX_MISS, withhold=()):
    """One match's numbers for the app and the log: the summary of its shown
    matrix, the matrix's own home/draw/away (shown beside the pick's, never
    in its place), where it came from and - for the market's - how far it
    misses the prices it was fitted to ("flagged" above max_miss). None
    where there is no matrix. withhold: NUMBERS this card must not show
    (its league's measurement did not ship them): each is None, listed in
    "withheld" - a missing number is shown as missing."""
    s = summary(mat)
    if s is None or source is None:
        return None
    mk = markets(np.asarray(mat, float)[None])
    fit = miss is not None and not math.isnan(miss)
    out = dict(s, source=source, reason=reason,
               matrix={k: float(mk[k][0]) for k in ("p_h", "p_d", "p_a")},
               miss=float(miss) if fit else None,
               miss_on=TARGET_NAMES[int(miss_on)] if fit and int(miss_on) >= 0 else None,
               flagged=bool(fit and miss > max_miss))
    unknown = set(withhold) - set(NUMBERS)
    if unknown:
        raise ValueError(f"unknown numbers to withhold: {sorted(unknown)}")
    withhold = [k for k in NUMBERS if k in set(withhold)]
    if withhold:
        for k in withhold:
            if k == "exact":
                out["scores"], out["other"] = None, None
            else:
                out[k] = None
        out["withheld"] = withhold
    return out


# ----------------------------------------------------------- the league baseline

def baseline(matches, leagues=None):
    """B: each league's rates from EARLIER SEASONS only (every finished match
    of the league dated before the season's first match, back to 2005-06) -
    BTTS, over 1.5 / 2.5 / 3.5, the empirical frequency of each of the 37
    score categories (plus half a match in each, so no score is impossible)
    and of home / draw / away (p_h, p_d, p_a). Rows of a league's first
    season are NaN. leagues: whose rows (default the top five). Cut on
    dates, not season numbers: a match filed under an earlier season but
    played after a later one's start (ARG 2013/14's of 29/01/2015) is not in
    that later season's rates; for the five top leagues, whose seasons never
    overlap, the two are the same."""
    m = matches
    hg = pd.to_numeric(m["hg"], errors="coerce").to_numpy(float)
    ag = pd.to_numeric(m["ag"], errors="coerce").to_numpy(float)
    leagues = list(sd.TOP) if leagues is None else list(leagues)
    done = m["league"].isin(leagues).to_numpy() & np.isfinite(hg) & np.isfinite(ag)
    cat = score_index(np.nan_to_num(hg), np.nan_to_num(ag))
    ys = {"btts": (hg > 0) & (ag > 0), "o15": hg + ag > 1.5, "o25": hg + ag > 2.5, "o35": hg + ag > 3.5}
    res = {"p_h": hg > ag, "p_d": hg == ag, "p_a": hg < ag}
    cols = list(ys) + [f"c{k}" for k in range(N_CELLS)] + list(res)
    vals = np.full((len(m), len(cols)), np.nan)
    seasons = m["season"].to_numpy(float)
    league = m["league"].to_numpy()
    dates = m["date"].astype(str).to_numpy()
    for lg in leagues:
        in_lg = league == lg
        for s in np.unique(seasons[in_lg & np.isfinite(seasons)]):
            rows = in_lg & (seasons == s)
            train = done & in_lg & (dates < dates[rows].min())
            if not train.any():
                continue
            rates = [float(ys[k][train].mean()) for k in ys]
            cnt = np.bincount(cat[train], minlength=N_CELLS) + 0.5
            vals[rows] = np.r_[rates, cnt / cnt.sum(), [float(res[k][train].mean()) for k in res]]
    return pd.DataFrame(vals, index=m.index, columns=cols)


# ----------------------------------------------------------- the run's numbers

def walk(frame, preds, pick, q_o25=None, leagues=None, **rule):
    """The shown matrix for every row the model predicts (finished or still
    to play) of `leagues` (default the top five), each from the model's
    walk-forward rates, the pick (recalibrated on earlier seasons) and the
    match's own pre-match prices. preds: soccer_model.predict (lam, nu, rho
    columns; without them the model's matrix is missing). Returns (index,
    mats, info frame)."""
    q_o25 = bt.no_vig_total(frame) if q_o25 is None else q_o25
    rows = frame["league"].isin(sd.TOP if leagues is None else list(leagues)) & preds["p_h"].notna()
    idx = frame.index[rows]

    def col(c):
        return preds.loc[rows, c].to_numpy(float) if c in preds else np.full(len(idx), np.nan)
    p3 = pick.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy(float).copy()
    p3[pick.loc[rows, "model_only"].to_numpy(bool)] = np.nan
    mats, info = shown_matrices(p3[:, 0], p3[:, 1], p3[:, 2], q_o25[rows].to_numpy(float),
                                col("rho"), col("lam"), col("nu"), **rule)
    return idx, mats, pd.DataFrame(info, index=idx)


def _bin_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _gap(a, b):
    """Mean of a - b per match (lower is better for a) and its standard error."""
    d = np.asarray(a, float) - np.asarray(b, float)
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
    return {"vs_base": round(float(d.mean()), 5), "vs_base_se": round(se, 5) if se is not None else None}


def _band_table(p_top, hit, bands=BANDS):
    names = np.array([band_of(p) for p in p_top], dtype=object)
    out = {}
    for b, floor in bands:
        s = names == b
        out[b] = {"floor": floor, "picks": int(s.sum()),
                  "hit_rate": round(float(hit[s].mean()), 4) if s.any() else None,
                  "mean_prob": round(float(p_top[s].mean()), 4) if s.any() else None}
    out["all"] = {"picks": int(len(hit)), "hit_rate": round(float(hit.mean()), 4) if len(hit) else None}
    return out


def measure(frame, preds, pick, walked=None, periods=(bt.DEV, bt.CONFIRM), leagues=None):
    """What the Method tab prints, from the same walk-forward the cards come
    from - the rule as shipped (`walked`, from walk()), the model's own
    matrix and each league's earlier-season rates, on the same matches of
    each period (top-league matches with a result where all three have a
    number): log loss of both teams to score, over 1.5, over 3.5 and the
    exact score (37-way), how often the likeliest score and one of the three
    likeliest happened, the mean BTTS chance against the rate and by decile
    (the shown matrix's and the model's own), how often the side each BTTS
    chance favoured happened (against always "yes"), the matrix's own
    home/draw/away against the pick's where the matrix is the market's, and
    the bands - the pick's and the model's own picks', same edges."""
    idx, mats, info = walked if walked is not None else walk(frame, preds, pick, leagues=leagues)
    if not len(idx):
        return None
    shown = markets(mats)
    has_m = info["source"].notna().to_numpy()

    def col(c):
        return preds.loc[idx, c].to_numpy(float) if c in preds else np.full(len(idx), np.nan)
    own = markets(dc_matrix(col("lam"), col("nu"), col("rho")))
    base = baseline(frame, leagues).loc[idx]
    cells = [f"c{k}" for k in range(N_CELLS)]
    src = {"shown": shown, "model": own,
           "base": {k: base[k].to_numpy(float) for k in ("btts", "o15", "o25", "o35")}
           | {"cells": base[cells].to_numpy(float)}}
    m = frame.loc[idx]
    hg, ag = m["hg"].to_numpy(float), m["ag"].to_numpy(float)
    seasons = m["season"].to_numpy(float)
    done = m["result"].isin(bt.RESULTS).to_numpy()
    every = np.ones(len(idx), bool)
    for s in src.values():
        every &= np.isfinite(s["btts"]) & np.isfinite(s["o15"]) & np.isfinite(s["o35"]) & np.isfinite(s["cells"][:, 0])
    every &= has_m
    pick_p = pick.loc[idx, ["p_h", "p_d", "p_a"]].to_numpy(float)
    model_p = preds.loc[idx, ["p_h", "p_d", "p_a"]].to_numpy(float)
    result = m["result"].to_numpy()
    out = {"periods": {}, "fit_max_miss": FIT_MAX_MISS,
           "band_edges": [f for _, f in BANDS if f > 0][::-1], "bands": {"pick": {}, "model": {}}}
    for lo, hi in periods:
        key = f"{lo}-{hi}"
        sel = done & (seasons >= lo) & (seasons <= hi)
        rows = sel & every
        y = {"btts": ((hg > 0) & (ag > 0))[rows].astype(float), "o15": (hg + ag > 1.5)[rows].astype(float),
             "o35": (hg + ag > 3.5)[rows].astype(float)}
        cat = score_index(np.nan_to_num(hg[rows]), np.nan_to_num(ag[rows]))
        reasons = info["reason"].to_numpy()[rows]
        rep = {"matches": int(rows.sum()),
               "market": int((info["source"].to_numpy()[rows] == SOURCE_MARKET).sum()),
               "model": {r: int((reasons == r).sum()) for r in (NO_PRICE, NO_OVER_UNDER)},
               "flagged": int(info["flagged"].to_numpy()[rows].sum()) if "flagged" in info else 0}
        if rows.any():
            for t, yt in y.items():
                rep[t] = {"rate": round(float(yt.mean()), 4)}
                ll = {name: _bin_loss(s[t][rows], yt) for name, s in src.items()}
                for name, s in src.items():
                    rep[t][name] = {"log_loss": round(float(ll[name].mean()), 5),
                                    "mean_p": round(float(s[t][rows].mean()), 4)}
                rep[t]["shown"].update(_gap(ll["shown"], ll["base"]))
            rep["exact"] = {}
            ll = {}
            for name, s in src.items():
                C = s["cells"][rows]
                pe = C[np.arange(len(C)), cat]
                ll[name] = -np.log(np.clip(pe, 1e-12, 1))
                order = np.argsort(-C[:, :-1], axis=1, kind="stable")
                rep["exact"][name] = {
                    "log_loss": round(float(ll[name].mean()), 5),
                    "top1_hit": round(float((order[:, 0] == cat).mean()), 4),
                    "top1_mean_p": round(float(C[np.arange(len(C)), order[:, 0]].mean()), 4),
                    "top3_hit": round(float((order[:, :3] == cat[:, None]).any(axis=1).mean()), 4)}
            rep["exact"]["shown"].update(_gap(ll["shown"], ll["base"]))
            # both teams to score by decile of its chance: predicted against happened
            p = shown["btts"][rows]
            edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
            which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
            gaps = [float(y["btts"][which == b].mean() - p[which == b].mean()) for b in range(len(edges) - 1)
                    if (which == b).any()]
            rep["btts"]["deciles"] = len(gaps)
            rep["btts"]["deciles_low"] = sum(g > 0 for g in gaps)
            # the same for the model's own matrix: the caveat a card from it carries
            pm = own["btts"][rows]
            edges = np.unique(np.quantile(pm, np.linspace(0, 1, 11)))
            which = np.clip(np.searchsorted(edges, pm, side="right") - 1, 0, len(edges) - 2)
            gaps = [float(y["btts"][which == b].mean() - pm[which == b].mean()) for b in range(len(edges) - 1)
                    if (which == b).any()]
            rep["btts"]["model"].update(deciles=len(gaps), deciles_low=sum(g > 0 for g in gaps))
            # the side each chance favoured (the Record tab's call) against
            # always "yes" and the leagues' own rates
            rep["btts"]["side"] = {name: round(float(((s["btts"][rows] >= 0.5) == (y["btts"] > 0)).mean()), 4)
                                   for name, s in src.items()}
            rep["btts"]["side"]["always_yes"] = round(float(y["btts"].mean()), 4)
            # the matrix's own home/draw/away (the card's "Results" row, beside
            # the pick on a market card) against the pick's, on those matches
            mk = rows & (info["source"].to_numpy() == SOURCE_MARKET)
            if mk.any():
                lab = np.array([bt.RESULTS.index(r) for r in result[mk]])
                q = np.column_stack([shown[k][mk] for k in ("p_h", "p_d", "p_a")])
                pp = pick_p[mk]

                def ll3(prob):
                    return round(float(-np.log(np.clip(prob[np.arange(len(prob)), lab], 1e-12, 1)).mean()), 5)
                rep["matrix_1x2"] = {"matches": int(mk.sum()), "log_loss": ll3(q), "pick_log_loss": ll3(pp)}
        out["periods"][key] = rep
        # the bands: the pick (market where priced) and the model's own picks
        ok = sel & np.isfinite(pick_p).all(axis=1)
        side = np.array(bt.RESULTS)[np.nan_to_num(pick_p[ok]).argmax(axis=1)]
        out["bands"]["pick"][key] = _band_table(pick_p[ok].max(axis=1), side == result[ok])
        ok = sel & np.isfinite(model_p).all(axis=1)
        side = np.array(bt.RESULTS)[model_p[ok].argmax(axis=1)]
        out["bands"]["model"][key] = _band_table(model_p[ok].max(axis=1), side == result[ok])
    return out
