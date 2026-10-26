"""DOCSTRING PLACEHOLDER"""

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
from experiments import shots_ratings as sr  # noqa: E402

MAX_GOALS = sr.MAX_GOALS          # the matrix runs 0-10 each way, as the model's does
GRID = 5                          # exact scores 0-0 .. 5-5, then one "other" bucket
N_CELLS = (GRID + 1) ** 2 + 1     # 37 categories
LAM_RANGE = (0.05, 8.0)           # as the model clips its own expected goals


# ----------------------------------------------------------- the matrix

def dc_matrix(lam, nu, rho, max_goals=MAX_GOALS):
    """The Dixon-Coles score matrix, (n, G+1, G+1), [x, y] = P(home x, away
    y): independent Poissons with the low-score correction tau_rho, truncated
    at `max_goals` each way and renormalised - exactly the matrix behind
    shots_ratings.score_probs. lam, nu, rho: scalars or arrays of length n."""
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
    mats = np.asarray(mats, float)
    if mats.ndim == 2:
        mats = mats[None]
    home, draw, away, tot = _masks(mats.shape[1])
    cells = mats[:, :GRID + 1, :GRID + 1].reshape(len(mats), -1)
    other = np.clip(1.0 - cells.sum(axis=1), 0.0, 1.0)
    return {
        "p_h": mats[:, home].sum(axis=1), "p_d": mats[:, draw].sum(axis=1), "p_a": mats[:, away].sum(axis=1),
        "btts": mats[:, 1:, 1:].sum(axis=(1, 2)),
        "o15": mats[:, tot >= 2].sum(axis=1), "o25": mats[:, tot >= 3].sum(axis=1),
        "o35": mats[:, tot >= 4].sum(axis=1),
        "cells": np.column_stack([cells, other]),
    }


def score_label(k):
    """Category index -> '2-1' (or 'other')."""
    return "other" if k == N_CELLS - 1 else f"{k // (GRID + 1)}-{k % (GRID + 1)}"


def score_index(hg, ag):
    """The category of each final score: 6 * home + away, or 36 (other)."""
    hg, ag = np.asarray(hg, float), np.asarray(ag, float)
    inside = (hg <= GRID) & (ag <= GRID)
    return np.where(inside, hg * (GRID + 1) + ag, N_CELLS - 1).astype(int)


def summary(mat, top=3):
    """One match's score matrix (G+1, G+1) -> the numbers a match card
    shows: both teams to score, over 1.5 / 2.5 / 3.5, the `top` most likely
    exact scores with their chances, and the chance of a score beyond 5-5."""
    mk = markets(np.asarray(mat, float)[None])
    cells = mk["cells"][0]
    order = np.argsort(-cells[:-1], kind="stable")[:top]
    return {
        "btts": float(mk["btts"][0]), "o15": float(mk["o15"][0]), "o25": float(mk["o25"][0]),
        "o35": float(mk["o35"][0]),
        "scores": [{"score": score_label(k), "p": float(cells[k])} for k in order],
        "other": float(cells[-1]),
    }


# ----------------------------------------------------------- market-implied rates

def _targets(mats):
    mk = markets(mats)
    return np.column_stack([mk["p_h"], mk["p_d"], mk["p_a"], mk["o25"]])


def implied_rates(p3, p_o25, rho, lam0=None, nu0=None, iters=60, tol=1e-10):
    """(lam, nu) whose Dixon-Coles matrix (with the given rho) comes closest,
    in least squares, to the home/draw/away chances p3 (n, 3) and the
    over-2.5 chance p_o25 (n,; NaN where no over/under price: then the fit
    uses the 1X2 alone). Vectorised Gauss-Newton on (log lam, log nu) with a
    forward-difference Jacobian, started from (lam0, nu0) where given.
    Returns (lam, nu, rmse) - rmse the root mean squared miss over the
    targets used, in probability."""
    p3 = np.asarray(p3, float).reshape(-1, 3)
    n = len(p3)
    po = np.broadcast_to(np.asarray(p_o25 if p_o25 is not None else np.nan, float), (n,))
    rho = np.broadcast_to(np.asarray(rho, float), (n,)).copy()
    has_o = np.isfinite(po)
    target = np.column_stack([p3, np.where(has_o, po, 0.0)])
    w = np.column_stack([np.ones((n, 3)), has_o.astype(float)])
    lo, hi = np.log(LAM_RANGE[0]), np.log(LAM_RANGE[1])
    th = np.empty((n, 2))
    th[:, 0] = np.log(1.45) if lam0 is None else np.log(np.where(np.isfinite(lam0), lam0, 1.45))
    th[:, 1] = np.log(1.15) if nu0 is None else np.log(np.where(np.isfinite(nu0), nu0, 1.15))
    th = np.clip(th, lo, hi)
    eps = 1e-6
    live = np.ones(n, dtype=bool)
    for _ in range(iters):
        if not live.any():
            break
        idx = np.flatnonzero(live)
        t = th[idx]
        f = _targets(dc_matrix(np.exp(t[:, 0]), np.exp(t[:, 1]), rho[idx]))
        r = (f - target[idx]) * w[idx]
        J = np.empty((len(idx), 4, 2))
        for k in range(2):
            tk = t.copy()
            tk[:, k] += eps
            J[:, :, k] = (_targets(dc_matrix(np.exp(tk[:, 0]), np.exp(tk[:, 1]), rho[idx])) - f) / eps * w[idx]
        JtJ = np.einsum("nik,nil->nkl", J, J) + 1e-12 * np.eye(2)
        Jtr = np.einsum("nik,ni->nk", J, r)
        step = -np.linalg.solve(JtJ, Jtr[:, :, None])[:, :, 0]
        step = np.clip(step, -0.5, 0.5)
        th[idx] = np.clip(t + step, lo, hi)
        live[idx] = np.abs(step).max(axis=1) > tol
    lam, nu = np.exp(th[:, 0]), np.exp(th[:, 1])
    f = _targets(dc_matrix(lam, nu, rho))
    miss = ((f - target) * w) ** 2
    rmse = np.sqrt(miss.sum(axis=1) / w.sum(axis=1))
    return lam, nu, rmse
