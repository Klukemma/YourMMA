"""Richer numbers beside the pick: the chance of every exact score, both
teams to score (BTTS), over 1.5 and 3.5 goals, and how sure the pick is.

THE IDEA. Behind home/draw/away and over 2.5 there can be a whole score
matrix, P(home x, away y) for x, y = 0..10, and everything else is read off
it (markets, summary): BTTS, every over/under line, the likeliest scores.
Which matrix, and does it beat the plainest honest alternative?

THE SOURCES. Each is a Dixon-Coles matrix tau_rho(x, y) Pois(x; lam)
Pois(y; nu), truncated at 10 goals a side and renormalised (dc_matrix):
  M   the model alone: shots_ratings' own walk-forward lam, nu and rho
      (calibrate(..., score_params=True); p_h..p_o25 are bit for bit the
      same with the flag on or off)
  K   market-implied: per match, the (lam, nu) whose matrix reproduces, in
      least squares (Gauss-Newton), the RECALIBRATED 1X2 the app picks with
      (soccer_model.pick) and the no-vig over 2.5 (no_vig_total), with the
      model's rho for that match. No 1X2 price: M, flagged
  KM  the market's total, the model's split: lam + nu from the no-vig over
      2.5 (Poisson), divided as the model divides it, the model's rho
  B   the baseline: each league's rates over every earlier season - BTTS,
      over 1.5 / 3.5, the 37 exact-score frequencies (+0.5 each)
The Dixon-Coles correction moves chance between 0-0 / 1-1 and 1-0 / 0-1 and
leaves the under-2.5 total alone, so in K the over 2.5 fixes lam + nu, the
1X2 the split, and the draw is what is left: K's draw is on average 0.005
below the pick's (mean absolute gap 0.007).

THE TARGETS. Top-five-league matches with a result: BTTS, over 1.5, over
3.5 (binary log loss, Brier); the exact score (37-way: 0-0 .. 5-5 and
"other"; log loss, and how often the single likeliest score and the three
likeliest happened); confidence bands for the shipped pick.

THE PROTOCOL. Chosen on DEV 2006-2015 only; CONFIRM 2016-2025 reported once.
  source  one matrix for every number on a card (so they agree with each
          other): the lowest DEV exact-score log loss - K (M 2.83389,
          K 2.82672, KM 2.83133)
  ships   a target ships only if K beats B on log loss in BOTH periods
  bands   four; edges at the DEV quartiles of the pick's top probability
          (0.423, 0.497, 0.616) rounded to 0.05: 0.40 / 0.50 / 0.60
  rejected on DEV (python score_markets.py --variants), exact-score log loss
          on the same DEV matches: K 2.82674; rho refitted each season on
          earlier seasons' exact scores (K_r) 2.82677 - it lands on -0.05 to
          -0.07, where the model's rho already is; rho fitted per match so
          the draw is hit exactly too (K3) 2.82798 (per-match rho -0.5 to
          +0.28). KM had the best DEV BTTS (0.68932 vs K 0.68972) but loses
          the exact score.

MEASURED (walk-forward ~24 s; ~115 s with the point-in-time checks). Log
loss, every source on the same matches: DEV 18,260, CONFIRM 17,937.

                     DEV 2006-2015                    CONFIRM 2016-2025
               M       K       KM      B          M       K       KM      B
  BTTS       0.68979 0.68972 0.68932 0.69312    0.68456 0.68319 0.68303 0.69047
  over 1.5   0.56468 0.56332 0.56342 0.57117    0.52792 0.52554 0.52553 0.53889
  over 3.5   0.57977 0.57823 0.57825 0.59079    0.59869 0.59508 0.59509 0.61558
  exact      2.83389 2.82672 2.83133 2.94606    2.87252 2.85793 2.86684 3.00865

  K - B (paired se)   DEV               CONFIRM
  BTTS            -0.0034 (0.0009)   -0.0073 (0.0010)
  over 1.5        -0.0079 (0.0008)   -0.0134 (0.0011)
  over 3.5        -0.0126 (0.0011)   -0.0205 (0.0015)
  exact           -0.1194 (0.0039)   -0.1507 (0.0042)
  likeliest score happened  K 12.9% (its mean chance 13.2%), B 11.5%  |  K 13.1% (12.6%), B 11.9%
  one of the top three      K 34.9% (34.6%), B 31.1%                  |  K 34.2% (33.0%), B 29.6%

ALL FOUR SHIP. K's over 2.5 is the market's own number to within the fit
(DEV 0.67957 vs the no-vig market 0.67975; CONFIRM 0.67067 vs 0.67066): no
gain there, and the app keeps showing the market's.

THE PICK'S BANDS (hit rate, mean top probability):
                        DEV 2006-2015            CONFIRM 2016-2025
  STRONG   >= 0.60    5087  69.8% (71.0%)      5659  70.9% (71.6%)
  SOLID   0.50-0.60   3880  53.0% (54.6%)      3957  55.0% (55.0%)
  LEAN    0.40-0.50   6259  44.8% (44.8%)      5445  44.9% (44.7%)
  TOSS-UP  < 0.40     3034  39.2% (37.7%)      2876  36.8% (37.7%)

CAVEATS.
  - BTTS is UNDER-predicted in both periods: K's mean 0.488 against 0.510
    on DEV, 0.511 against 0.537 on CONFIRM, low in every CONFIRM decile
    (0.394 -> 0.446 at the bottom, 0.625 -> 0.658 at the top). It beats B in
    both periods all the same. A recalibration fitted on earlier seasons
    might fix it; it is not shipped, and since CONFIRM has now been read it
    would need its own honest out-of-sample check.
  - The matrix cannot always reproduce the pick. The least-squares miss is
    0.004 (median rmse) on DEV and 0.003 on CONFIRM; 505 DEV and 19 CONFIRM
    matches miss by more than 0.02 - mostly Serie A 2006-07 and 2007-08,
    where the 1X2 and over/under prices disagree more than any Poisson
    matrix allows. The worst miss, 0.21, is an end-of-season match with the
    draw priced at 71%. The card should show the pick's own 1X2 with the
    matrix's numbers beside it; info["rmse"] says how far apart they are.
  - Exact scores are long shots: the likeliest one happens about one match
    in eight.
  - Calibration on CONFIRM by decile, K: over 1.5 within 0.013 of the
    observed rate everywhere, over 3.5 within 0.021, the likeliest score's
    chance a little low in the upper half (0.125 -> 0.144), every score
    cell pooled within 0.003.

POINT IN TIME. Each number for a match comes from matches on earlier days
(the model, its rho), the match's own pre-match prices, and weights and
rates fitted on earlier seasons (the recalibration, B). Checked on the whole
pipeline at 2011-02-05 and 2021-02-06 (point_in_time):
soccer_backtest.check_point_in_time on K's 1X2; with every result from the
cut on scrambled, none of the 1.49M / 4.69M numbers dated on or before the
cut moves (exact equality); with prices from the cut on shuffled as well,
nothing dated before the cut moves. The first run failed that last check by
1e-10 in one match: numpy's fancy-indexed sums round differently with the
batch's size, and an ill-posed fit magnified it. Every sum is now row-local
(markets, _normal_step), and a test holds every row to what it gets when
solved alone, bit for bit.

THE PRODUCTION CALLS (for soccer_model.py when this is integrated):
  score_matrix(p_h, p_d, p_a, p_o25, rho, lam, nu) -> (mats, info)
      the pick's 1X2 and the no-vig over 2.5 where priced (NaN / None
      where not), the model's lam, nu, rho (shots_ratings with
      score_params=True) as the fallback
  summary(mat) -> {btts, o15, o25, o35, scores: the three likeliest with
      their chances, other}
  band_of(p_top) -> STRONG / SOLID / LEAN / TOSS-UP
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_backtest as bt  # noqa: E402
import soccer_data as sd  # noqa: E402
import soccer_model as smod  # noqa: E402
from experiments import shots_ratings as sr  # noqa: E402

MAX_GOALS = sr.MAX_GOALS          # the matrix runs 0-10 each way, as the model's does
GRID = 5                          # exact scores 0-0 .. 5-5, then one "other" bucket
N_CELLS = (GRID + 1) ** 2 + 1     # 37 categories
LAM_RANGE = (0.05, 8.0)           # as the model clips its own expected goals

# The source of the score matrix, chosen on DEV (2006-2015) exact-score log
# loss: K, the market-implied matrix (the notes above have the numbers).
CHOSEN = "K"
SOURCES = ("M", "K", "KM", "B")
TARGETS = ("btts", "o15", "o35")
# How sure the pick is, by its top probability. The edges are the DEV
# quartiles of that probability (0.423, 0.497, 0.616) rounded to the nearest
# 0.05 - band_edges() re-derives them from DEV on every run. Each band's hit
# rate is MEASURED and shown beside the label: "STRONG" means "picks like
# this won N% of 2016-2025 matches", never "bet this".
BANDS = (("STRONG", 0.60), ("SOLID", 0.50), ("LEAN", 0.40), ("TOSS-UP", 0.0))
OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/score_markets.json")


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
    """THE PRODUCTION CALL. Score matrices (n, 11, 11) for n matches:

      priced      (p_h, p_d, p_a finite) the market-implied matrix - the
                  (lam, nu) that best reproduce the recalibrated market 1X2
                  (soccer_model.pick) and the no-vig over 2.5 (where posted;
                  pass NaN or None where not), with the model's Dixon-Coles
                  rho for that match (shots_ratings, score_params=True)
      not priced  (p_h NaN) the model's own matrix from its lam, nu, rho

    Returns (mats, info): info["from_market"] (bool), info["lam"], info["nu"]
    (the rates behind each matrix), info["rmse"] (how closely the market was
    reproduced; NaN for the model's own). A row with neither a price nor the
    model's rates is NaN - shown as missing."""
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


def total_rate(p_o25):
    """The Poisson total goals mu with P(total >= 3) = p_o25 (bisection)."""
    q = np.asarray(p_o25, float)
    lo, hi = np.full(q.shape, 1e-3), np.full(q.shape, 12.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        over = 1 - np.exp(-mid) * (1 + mid + mid ** 2 / 2)
        hi = np.where(over > q, mid, hi)
        lo = np.where(over > q, lo, mid)
    return np.where(np.isfinite(q), (lo + hi) / 2, np.nan)


# ----------------------------------------------------------- the walk-forward

def baseline(matches):
    """B: each top league's rates from EARLIER SEASONS only (every finished
    match back to 2005-06) - BTTS, over 1.5 / 2.5 / 3.5, and the empirical
    frequency of each of the 37 score categories (plus half a match in each,
    so no score is impossible). Rows of a league's first season are NaN."""
    m = matches
    hg = pd.to_numeric(m["hg"], errors="coerce").to_numpy(float)
    ag = pd.to_numeric(m["ag"], errors="coerce").to_numpy(float)
    done = m["league"].isin(sd.TOP).to_numpy() & np.isfinite(hg) & np.isfinite(ag)
    cat = score_index(np.nan_to_num(hg), np.nan_to_num(ag))
    ys = {"btts": (hg > 0) & (ag > 0), "o15": hg + ag > 1.5, "o25": hg + ag > 2.5, "o35": hg + ag > 3.5}
    out = pd.DataFrame(np.nan, index=m.index, columns=list(ys) + [f"c{k}" for k in range(N_CELLS)])
    seasons = m["season"].to_numpy(float)
    league = m["league"].to_numpy()
    for lg in sd.TOP:
        in_lg = league == lg
        for s in np.unique(seasons[in_lg & np.isfinite(seasons)]):
            train = done & in_lg & (seasons < s)
            rows = in_lg & (seasons == s)
            if not train.any():
                continue
            rates = [float(ys[k][train].mean()) for k in ys]
            cnt = np.bincount(cat[train], minlength=N_CELLS) + 0.5
            out.loc[rows, :] = np.r_[rates, cnt / cnt.sum()]
    return out


def _frame(mk, index):
    data = {k: mk[k] for k in ("p_h", "p_d", "p_a", "btts", "o15", "o25", "o35")}
    data.update({f"c{k}": mk["cells"][:, k] for k in range(N_CELLS)})
    return pd.DataFrame(data, index=index)


def walk_forward(matches, preds=None, model_params=None):
    """Every source's numbers for every top-league row the model predicts,
    each from matches on earlier days and weights fitted on earlier seasons:

      M   the model's own matrix (shots_ratings lam, nu, rho)
      K   the market-implied matrix (score_matrix), the model's where unpriced
      KM  the market's total (no-vig over 2.5 -> Poisson mu) split as the
          model splits it (lam / (lam + nu)), the model's rho; the model's
          own where no over/under price
      B   league rates from earlier seasons

    Returns {source: frame of p_h..o35, c0..c36} plus "pick" (the shipped
    pick, its top probability) and "info" (K's fit: from_market, rmse, lam, nu).
    model_params go to shots_ratings (tests run it on small made-up leagues)."""
    if preds is None:
        preds = sr.predict_walk_forward(matches, score_params=True, **(model_params or {}))
    pick, _ = smod.pick(matches, preds)
    q_o = bt.no_vig_total(matches)
    rows = matches["league"].isin(sd.TOP) & preds["p_h"].notna()
    idx = matches.index[rows]
    lam, nu, rho = (preds.loc[rows, c].to_numpy(float) for c in ("lam", "nu", "rho"))
    out = {"M": _frame(markets(dc_matrix(lam, nu, rho)), idx)}

    p3 = pick.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy(float).copy()
    p3[pick.loc[rows, "model_only"].to_numpy(bool)] = np.nan
    po = q_o[rows].to_numpy(float)
    mats, info = score_matrix(p3[:, 0], p3[:, 1], p3[:, 2], po, rho, lam, nu)
    out["K"] = _frame(markets(mats), idx)
    out["info"] = pd.DataFrame({"from_market": info["from_market"], "rmse": info["rmse"],
                                "lam": info["lam"], "nu": info["nu"], "rho": rho}, index=idx)

    mu = total_rate(po)
    share = lam / (lam + nu)
    has_o = np.isfinite(mu)
    out["KM"] = _frame(markets(dc_matrix(np.where(has_o, mu * share, lam), np.where(has_o, mu * (1 - share), nu), rho)), idx)

    out["B"] = baseline(matches).loc[idx]
    top_p = pick.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy(float)
    out["pick"] = pd.DataFrame({"p_top": top_p.max(axis=1), "side": np.array(list("HDA"))[np.nan_to_num(top_p).argmax(axis=1)],
                                "model_only": pick.loc[rows, "model_only"].to_numpy(bool)}, index=idx)
    out["preds"] = preds
    return out


# ----------------------------------------------------------- the scores

def _bin_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration(p, y, bins=10):
    """Mean predicted vs observed, by deciles of the predicted probability."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        s = which == b
        if s.any():
            rows.append({"lo": round(float(edges[b]), 4), "hi": round(float(edges[b + 1]), 4), "n": int(s.sum()),
                         "predicted": round(float(p[s].mean()), 4), "observed": round(float(y[s].mean()), 4)})
    return rows


def band_edges(p_top, quantiles=(0.25, 0.5, 0.75), step=0.05):
    """The band rule, applied to DEV picks only: quartiles of the pick's top
    probability, rounded to the nearest `step`."""
    q = np.quantile(np.asarray(p_top, float), quantiles)
    return [round(round(v / step) * step, 2) for v in q], [round(float(v), 4) for v in q]


def band_report(matches, wf, sel, bands=BANDS):
    pk = wf["pick"].loc[sel]
    hit = (pk["side"].to_numpy() == matches.loc[sel, "result"].to_numpy())
    name = pk["p_top"].map(band_of).to_numpy()
    out = {}
    for b, floor in bands:
        s = name == b
        out[b] = {"floor": floor, "picks": int(s.sum()), "share": round(float(s.mean()), 4),
                  "hit_rate": round(float(hit[s].mean()), 4) if s.any() else None,
                  "mean_prob": round(float(pk["p_top"].to_numpy()[s].mean()), 4) if s.any() else None}
    out["all"] = {"picks": int(len(hit)), "hit_rate": round(float(hit.mean()), 4),
                  "mean_prob": round(float(pk["p_top"].mean()), 4), "model_only": int(pk["model_only"].sum())}
    return out


def evaluate(matches, wf, period):
    """Every source on the same matches of one period: binary log loss and
    Brier (BTTS, over 1.5, over 2.5 for reference, over 3.5), the exact score
    (37-way log loss, hits of the single most likely score and of the top
    three), and the paired standard error of each source's log-loss gap to B."""
    lo, hi = period
    idx = wf["K"].index
    m = matches.loc[idx]
    sel = m["result"].isin(bt.RESULTS) & m["season"].between(lo, hi)
    for s in SOURCES:
        sel &= wf[s][["btts", "o15", "o35", "c0"]].notna().all(axis=1)
    rows = idx[sel.to_numpy()]
    m = matches.loc[rows]
    hg, ag = m["hg"].to_numpy(float), m["ag"].to_numpy(float)
    ys = {"btts": ((hg > 0) & (ag > 0)).astype(float), "o15": (hg + ag > 1.5).astype(float),
          "o25": (hg + ag > 2.5).astype(float), "o35": (hg + ag > 3.5).astype(float)}
    cat = score_index(hg, ag)
    info = wf["info"].loc[rows]
    q_o = bt.no_vig_total(m).to_numpy(float)
    has_o = np.isfinite(q_o)
    rep = {"matches": int(len(rows)), "k_from_market": int(info["from_market"].sum()),
           "k_model_fallback": int((~info["from_market"]).sum()), "k_no_over_under_price": int((~has_o).sum()),
           "k_fit_rmse": {"median": round(float(info["rmse"].median()), 5),
                          "p95": round(float(info["rmse"].quantile(0.95)), 5),
                          "over_0.02": int((info["rmse"] > 0.02).sum())}}
    losses = {}
    for t, y in ys.items():
        rep[t] = {"rate": round(float(y.mean()), 4)}
        for s in SOURCES:
            p = wf[s].loc[rows, t].to_numpy(float)
            ll = _bin_loss(p, y)
            losses[(t, s)] = ll
            rep[t][s] = {"log_loss": round(float(ll.mean()), 5), "brier": round(float(np.mean((p - y) ** 2)), 5),
                         "mean_p": round(float(p.mean()), 4)}
        if t == "o25":                           # the number the app already shows
            ll = _bin_loss(q_o[has_o], y[has_o])
            rep[t]["market"] = {"log_loss": round(float(ll.mean()), 5), "matches": int(has_o.sum()),
                                "K_same_matches": round(float(losses[(t, "K")][has_o].mean()), 5)}
    rep["exact"] = {}
    for s in SOURCES:
        C = wf[s].loc[rows, [f"c{k}" for k in range(N_CELLS)]].to_numpy(float)
        pe = C[np.arange(len(C)), cat]
        ll = -np.log(np.clip(pe, 1e-12, 1))
        losses[("exact", s)] = ll
        order = np.argsort(-C[:, :-1], axis=1, kind="stable")
        rep["exact"][s] = {"log_loss": round(float(ll.mean()), 5),
                           "top1_hit": round(float((order[:, 0] == cat).mean()), 4),
                           "top1_mean_p": round(float(C[np.arange(len(C)), order[:, 0]].mean()), 4),
                           "top3_hit": round(float((order[:, :3] == cat[:, None]).any(axis=1).mean()), 4),
                           "top3_mean_p": round(float(np.take_along_axis(C, order[:, :3], 1).sum(axis=1).mean()), 4)}
    for t in ("btts", "o15", "o25", "o35", "exact"):
        for s in SOURCES:
            if s == "B":
                continue
            d = losses[(t, s)] - losses[(t, "B")]
            rep[t][s]["vs_B"] = round(float(d.mean()), 5)
            rep[t][s]["vs_B_se"] = round(float(d.std(ddof=1) / math.sqrt(len(d))), 5)
    rep["_rows"] = rows
    rep["_ys"] = ys
    rep["_cat"] = cat
    return rep


# ----------------------------------------------------------- point in time

def _scramble(matches, cut, prices=False):
    """Results (and match statistics) from `cut` on replaced by random ones,
    as soccer_backtest.check_point_in_time does; with prices=True every price
    from `cut` on is also shuffled among those rows."""
    later = (matches["date"] >= str(cut)).to_numpy()
    out = matches.copy()
    rng = np.random.default_rng(1)
    for c in ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"):
        if c in out:
            out.loc[later, c] = rng.integers(0, 6, int(later.sum()))
    hg, ag = out.loc[later, "hg"].astype(float), out.loc[later, "ag"].astype(float)
    out.loc[later, "result"] = np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))
    if prices:
        cols = [c for c in out.columns if c.startswith(("avg", "max", "ps", "b365"))]
        perm = rng.permutation(np.flatnonzero(later))
        out.loc[later, cols] = out.iloc[perm][cols].to_numpy()
    return out


def _flat(wf):
    parts = [wf[s].add_prefix(f"{s}_") for s in SOURCES]
    parts += [wf["info"][["lam", "nu", "rmse"]].add_prefix("K_"), wf["pick"][["p_top"]]]
    return pd.concat(parts, axis=1)


def point_in_time(matches, cut, base, model_params=None):
    """Three checks at `cut`, using the full pipeline (model, pick, every
    source): (1) soccer_backtest.check_point_in_time on K's 1X2 for the
    matches ON the cut day; (2) with every result from the cut on scrambled,
    NO number dated on or before the cut moves; (3) with prices scrambled as
    well, no number dated before the cut moves."""
    cache = {}

    def predict(df):
        wf = base if df is matches else walk_forward(df, model_params=model_params)
        cache["last"] = wf
        k = wf["K"][["p_h", "p_d", "p_a"]]
        return k.reindex(df.index)
    ok1 = bt.check_point_in_time(predict, matches, cut)
    scr = cache["last"]
    a, b = _flat(base), _flat(scr)
    upto = matches.loc[a.index, "date"] <= str(cut)
    ok2 = bool(np.allclose(a[upto].to_numpy(float), b.loc[a.index][upto].to_numpy(float), equal_nan=True, rtol=0, atol=0))
    c = _flat(walk_forward(_scramble(matches, cut, prices=True), model_params=model_params))
    before = matches.loc[a.index, "date"] < str(cut)
    ok3 = bool(np.allclose(a[before].to_numpy(float), c.loc[a.index][before].to_numpy(float), equal_nan=True, rtol=0, atol=0))
    n_numbers = int(np.isfinite(a[upto].to_numpy(float)).sum())
    return {"cut": str(cut), "check_point_in_time": bool(ok1), "unmoved_on_or_before_cut": ok2,
            "unmoved_before_cut_prices_scrambled": ok3, "rows_on_or_before_cut": int(upto.sum()),
            "numbers_compared": n_numbers}


# ----------------------------------------------------------- rejected on DEV

def variants(matches, wf, rho_grid=np.round(np.arange(-0.12, 0.0001, 0.01), 2)):
    """DEV ONLY (seasons up to 2015 are all it reads): two other ways to set
    K's rho, both rejected on DEV. Prints and returns their DEV scores beside K.

      K_r  one rho per season, the grid value whose market-implied matrices
           scored the actual exact scores best over EARLIER seasons (2006 has
           none: the model's rho)
      K3   rho fitted per match with lam and nu, so the matrix reproduces the
           market's draw as well (three targets, three unknowns)"""
    info = wf["info"]
    keep = (matches.loc[info.index, "season"] <= bt.DEV[1]).to_numpy() & info["from_market"].to_numpy()
    idx = info.index[keep]
    m = matches.loc[idx]
    pick, _ = smod.pick(matches, wf["preds"])
    p3 = pick.loc[idx, ["p_h", "p_d", "p_a"]].to_numpy(float)
    po = bt.no_vig_total(m).to_numpy(float)
    hg, ag = m["hg"].to_numpy(float), m["ag"].to_numpy(float)
    done = np.isfinite(hg) & np.isfinite(ag)
    cat = score_index(np.nan_to_num(hg), np.nan_to_num(ag))
    lam0, nu0, rho_m = (info.loc[idx, c].to_numpy(float) for c in ("lam", "nu", "rho"))

    ll = np.empty((len(idx), len(rho_grid)))
    for j, r in enumerate(rho_grid):
        lam, nu, _ = implied_rates(p3, po, r, lam0, nu0)
        cells = markets(dc_matrix(lam, nu, r))["cells"]
        ll[:, j] = np.log(np.clip(cells[np.arange(len(idx)), cat], 1e-12, 1))
    seasons = m["season"].to_numpy(int)
    rho_r = rho_m.copy()
    chosen = {}
    for s in np.unique(seasons):
        train = done & (seasons < s) & (seasons >= bt.DEV[0])
        if train.any():
            chosen[int(s)] = float(rho_grid[ll[train].sum(axis=0).argmax()])
            rho_r[seasons == s] = chosen[int(s)]
    lam_r, nu_r, _ = implied_rates(p3, po, rho_r, lam0, nu0)

    has_o = np.isfinite(po)
    target = np.column_stack([p3, np.where(has_o, po, 0.0)])
    w = np.column_stack([np.ones((len(idx), 3)), has_o.astype(float)])

    def model3(t, rows):
        return _targets(dc_matrix(np.exp(t[:, 0]), np.exp(t[:, 1]), t[:, 2]))

    def project3(t):
        t = t.copy()
        t[:, :2] = np.clip(t[:, :2], np.log(LAM_RANGE[0]), np.log(LAM_RANGE[1]))
        lam, nu = np.exp(t[:, 0]), np.exp(t[:, 1])      # keep every tau positive
        t[:, 2] = np.clip(t[:, 2], np.maximum(-0.5, -0.95 / np.maximum(lam, nu)), np.minimum(0.5, 0.95 / (lam * nu)))
        return t
    th = _gauss_newton(np.column_stack([np.log(lam0), np.log(nu0), rho_m]), model3, target, w, project3, 80, 1e-10, 0.3)

    sel = done & (seasons >= bt.DEV[0])
    ys = {"btts": (hg > 0) & (ag > 0), "o15": hg + ag > 1.5, "o35": hg + ag > 3.5}
    out = {"rho_by_season": chosen, "matches": int(sel.sum()),
           "rho_k3": {"median": round(float(np.median(th[:, 2])), 4), "min": round(float(th[:, 2].min()), 4),
                      "max": round(float(th[:, 2].max()), 4)}}
    for name, mats in (("K", dc_matrix(lam0, nu0, rho_m)),           # info holds K's own rates
                       ("K_r", dc_matrix(lam_r, nu_r, rho_r)),
                       ("K3", dc_matrix(np.exp(th[:, 0]), np.exp(th[:, 1]), th[:, 2]))):
        mk = markets(mats[sel])
        r = {t: round(float(_bin_loss(mk[t], ys[t][sel].astype(float)).mean()), 5) for t in ys}
        pe = mk["cells"][np.arange(sel.sum()), cat[sel]]
        r["exact"] = round(float(-np.log(np.clip(pe, 1e-12, 1)).mean()), 5)
        out[name] = r
        print(f"   DEV {name:>3}: " + "  ".join(f"{k} {v:.5f}" for k, v in r.items()))
    print(f"   K_r rho by season: {chosen}; K3 rho {out['rho_k3']}")
    return out


# ----------------------------------------------------------- the run

def _strip(rep):
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def main():
    t_all = time.time()
    matches = sd.load_matches()
    t0 = time.time()
    wf = walk_forward(matches)
    t_wf = time.time() - t0
    print(f"walk-forward (model + pick + every source) 2005-2026: {t_wf:.1f}s")

    # -- choose on DEV only: the source, then the band edges
    dev = evaluate(matches, wf, bt.DEV)
    print(f"\nDEV {bt.DEV[0]}-{bt.DEV[1]}: {dev['matches']} matches (K from the market {dev['k_from_market']}, "
          f"model fallback {dev['k_model_fallback']})")
    chosen = min((s for s in SOURCES if s != "B"), key=lambda s: dev["exact"][s]["log_loss"])
    print(f"   chosen on DEV exact-score log loss: {chosen}")
    assert chosen == CHOSEN, f"DEV now chooses {chosen}; CHOSEN says {CHOSEN} - revisit before reading CONFIRM"
    pk = wf["pick"]
    dev_rows = pk.index[(matches.loc[pk.index, "season"].between(*bt.DEV)
                         & matches.loc[pk.index, "result"].isin(bt.RESULTS) & pk["p_top"].notna()).to_numpy()]
    edges, raw_q = band_edges(pk.loc[dev_rows, "p_top"])
    print(f"   band edges from DEV quartiles {raw_q} -> {edges}")
    assert edges == sorted(f for _, f in BANDS if f > 0), "the band constants no longer match the DEV rule"

    conf = evaluate(matches, wf, bt.CONFIRM)
    reps = {f"{bt.DEV[0]}-{bt.DEV[1]}": dev, f"{bt.CONFIRM[0]}-{bt.CONFIRM[1]}": conf}

    def line(t, rep, s):
        r = rep[t][s]
        gap = f"  vs B {r['vs_B']:+.5f} (se {r['vs_B_se']:.5f})" if "vs_B" in r else ""
        if t == "exact":
            return (f"{s:>3} log loss {r['log_loss']:.5f}  top-1 hit {r['top1_hit']:.4f} (mean p {r['top1_mean_p']:.4f})"
                    f"  top-3 hit {r['top3_hit']:.4f} (mean p {r['top3_mean_p']:.4f}){gap}")
        return f"{s:>3} log loss {r['log_loss']:.5f}  brier {r['brier']:.5f}  mean p {r['mean_p']:.4f}{gap}"
    for t in ("btts", "o15", "o35", "exact", "o25"):
        print(f"\n{t}" + ("" if t == "exact" else f"   (rate DEV {dev[t]['rate']:.4f}, CONFIRM {conf[t]['rate']:.4f})"))
        for period, rep in reps.items():
            print(f"  {period} ({rep['matches']} matches)")
            for s in SOURCES:
                print("    " + line(t, rep, s))
            if t == "o25":
                mk = rep["o25"]["market"]
                print(f"    market (no-vig, shipped) {mk['log_loss']:.5f} on {mk['matches']}; K on the same {mk['K_same_matches']:.5f}")

    ships = {}
    for t in TARGETS + ("exact",):
        both = all(rep[t][CHOSEN]["log_loss"] < rep[t]["B"]["log_loss"] for rep in reps.values())
        ships[t] = both
    print("\nships (chosen source beats B on log loss in BOTH periods): "
          + ", ".join(f"{t} {'YES' if v else 'NO'}" for t, v in ships.items()))
    for period, rep in reps.items():
        r = rep["k_fit_rmse"]
        print(f"   K reproduces the pick and the over 2.5 {period}: rmse median {r['median']:.4f}, p95 {r['p95']:.4f}, "
              f"{r['over_0.02']} matches miss by more than 0.02")

    # -- calibration on CONFIRM, the chosen source
    calib = {}
    rows = conf["_rows"]
    for t in TARGETS:
        calib[t] = calibration(wf[CHOSEN].loc[rows, t], conf["_ys"][t])
    C = wf[CHOSEN].loc[rows, [f"c{k}" for k in range(N_CELLS)]].to_numpy(float)
    top1 = C[:, :-1].argmax(axis=1)
    calib["top_score"] = calibration(C[np.arange(len(C)), top1], (top1 == conf["_cat"]).astype(float))
    onehot = np.zeros_like(C)
    onehot[np.arange(len(C)), conf["_cat"]] = 1.0
    calib["every_score_cell"] = calibration(C.ravel(), onehot.ravel())
    print(f"\ncalibration by decile, CONFIRM, {CHOSEN} (predicted -> observed)")
    for t, rows_ in calib.items():
        print(f"  {t:>16}: " + "  ".join(f"{r['predicted']:.3f}->{r['observed']:.3f}" for r in rows_))

    # -- confidence bands for the shipped pick
    bands = {}
    for period in (bt.DEV, bt.CONFIRM):
        sel = (matches.loc[pk.index, "season"].between(*period) & matches.loc[pk.index, "result"].isin(bt.RESULTS)
               & pk["p_top"].notna()).to_numpy()
        bands[f"{period[0]}-{period[1]}"] = band_report(matches, wf, pk.index[sel])
    print("\nconfidence bands of the pick (top probability; edges chosen on DEV)")
    for period, rep in bands.items():
        print(f"  {period}: " + "  ".join(
            f"{b} {rep[b]['picks']} picks hit {rep[b]['hit_rate']:.3f} (p {rep[b]['mean_prob']:.3f})" for b, _ in BANDS)
            + f"  | all {rep['all']['hit_rate']:.3f}")

    # -- point in time
    t0 = time.time()
    pit = [point_in_time(matches, cut, wf) for cut in ("2011-02-05", "2021-02-06")]
    t_pit = time.time() - t0
    for r in pit:
        print(f"   point in time {r}")

    # one match through the production calls, as the app would make them
    last = wf["info"].index[wf["info"]["from_market"].to_numpy()][-1]
    pick, _ = smod.pick(matches, wf["preds"])
    pr = wf["preds"].loc[last]
    mats, _ = score_matrix(pick.loc[last, "p_h"], pick.loc[last, "p_d"], pick.loc[last, "p_a"],
                           bt.no_vig_total(matches.loc[[last]]).iloc[0], pr["rho"], pr["lam"], pr["nu"])
    example = {"match": matches.loc[last, ["date", "league", "home", "away"]].astype(str).to_dict(),
               "pick": {k: round(float(pick.loc[last, k]), 4) for k in ("p_h", "p_d", "p_a")},
               "band": band_of(float(pick.loc[last, ["p_h", "p_d", "p_a"]].max())),
               "summary": summary(mats[0])}
    print(f"\nexample {example}")

    result = {
        "protocol": {"dev": list(bt.DEV), "confirm": list(bt.CONFIRM), "chosen_source": CHOSEN,
                     "choice_rule": "lowest DEV exact-score log loss among M, K, KM",
                     "ship_rule": "the chosen source beats B on log loss in both periods",
                     "band_rule": "DEV quartiles of the pick's top probability, rounded to 0.05",
                     "band_edges": edges, "band_dev_quartiles": raw_q},
        "periods": {p: _strip(r) for p, r in reps.items()},
        "ships": ships,
        "calibration_confirm": calib,
        "bands": bands,
        "point_in_time": pit,
        "example_summary": example,
        "runtime_s": {"walk_forward": round(t_wf, 1), "point_in_time": round(t_pit, 1),
                      "total": round(time.time() - t_all, 1)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {OUT}; total {time.time() - t_all:.1f}s")
    return result


if __name__ == "__main__":
    if "--variants" in sys.argv:
        _m = sd.load_matches()
        variants(_m, walk_forward(_m))
    else:
        main()
