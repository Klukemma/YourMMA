"""Candidate: DRAWS FROM THE GOALS LINE - the over/under 2.5 price read beside
the home/draw/away price in the pick's recalibration.

THE IDEA. Draws are the hardest outcome, and the 1X2 market prices them a
little short (today's pick, soccer_model.pick, adds a constant to the draw:
p_k proportional to q_k^beta * exp(c_k), three weights per season fitted on
earlier seasons). But expected goals drive the draw rate - a 1.9-goal match
draws far more often than a 3.2-goal one - and the fixtures file carries a
second, separately traded price that measures expected goals directly: the
over/under 2.5. If the 1X2 market under-reacts to the goals line when it
prices the draw, the over/under price tells the pick how much to move it.

THE MODEL. A conditional (multinomial) logit, three outcomes, c_home = 0,
fitted per season on every earlier season (soccer_model.pick's training
rows: top-five matches with a result and a full pre-match average price):

  z_k = beta * log q_k + c_k + sum_t w_t * X_t[k],   p = softmax(z)

with q the no-vig pre-match average 1X2 and the extra terms X_t (cfg
"terms") built from the no-vig over/under 2.5 average q_o:
  lo_d     x * [draw]         x = logit q_o, standardised on the training
  lo_a     x * [away]             rows (mean, sd), clipped to +-4
  lo_beta  x * log q_k        (the favourite's tilt moving with goals)
  lo2_d    (x^2 - mean) * [draw]
  kd       (log K_d - log q_d) * [draw], standardised: K is the GOALS-IMPLIED
           Dixon-Coles matrix - lam + nu from q_o (a Poisson total; the
           Dixon-Coles correction leaves under 2.5 alone), split so that its
           home share H / (H + A) equals the 1X2's, with a fixed rho (cfg
           "rho") - so K_d is the draw the goals line and the favourite
           imply, and kd how far the 1X2's own draw sits from it
  kdir     log(K_k / q_k) per outcome, one weight (a log pool of q and K)
  free     log q_j * [draw], log q_j * [away] for j = h, d, a: the
           unconstrained multinomial logit on the three log prices
Where q_o is missing every extra term is 0 (the training mean): the pick's
model. With no terms it IS the pick (the same chances to 1e-6 on a made-up
league, tests/test_draws_from_goals.py). The extra weights can be shrunk
towards 0 - towards the pick - by a ridge of kappa pseudo-matches.

  over 2.5 ("ou"): a binary logit on the over/under price,
  logit p = a * logit q_o + c + extra terms: kd (the 1X2's draw against the
  goals-implied draw: a draw priced above what the line implies says fewer
  goals), ld (logit q_d, standardised), lop (|log q_h - log q_a|),
  measured against today's raw no-vig over/under market.

THE PROTOCOL (house rules). Point in time: a match's features are its own
PRE-MATCH prices only (avg_h/d/a and avg_o25/u25 - collected Friday /
Tuesday; never closing prices, never Pinnacle, never the best price).
Every weight, mean and standard deviation for season s is fitted on top-five
matches of seasons before s (at least MIN_TRAIN of them, as for the pick;
refitted every season); the Dixon-Coles rho of K is a fixed constant. The
live fixtures file carries avg_* and avg_o25/u25 on its top-five rows (the
same columns price_dispersion found on 48 of 48 on 2026-11-02).

  pool ("ten"): the same model fitted on all ten divisions' priced, finished
  matches of earlier seasons - the second tiers with their own beta / draw /
  away shifts (unpenalised, so the top five's base is fitted by the top five
  alone), sharing the extra weights. Predictions are for the top five only.

Choosing, DEV only (2006-2015; CONFIRM 2016-2025 is never scored by this
module - `compare` refuses it, `tune` never even predicts a season after
2015, and the CSV's later seasons were not looked at). `grid()` (round 1:
23 1X2 variants, one of them the pick itself), `grid2()` (round 2: 9, one
the pick itself) and `ou_grid()` (6 over-2.5 variants) were each
walk-forwarded over DEV and compared with the pick (the no-vig over/under
for over 2.5) on the same matches. Since the variant is chosen on DEV its
DEV number is optimistic, so the nested check is reported too: the variant
best on 2006-2010 (TUNE) alone, scored on 2011-2015 (CHECK).

MEASURED - DEV 2006-2015, top five, the same 18,259 matches for both
(python experiments/draws_from_goals.py, ~35 s with the scramble test;
--tune prints the grid, ~2.5 min).

  CHOSEN: "lo_d ten" - ONE extra weight, the draw's shift moving with the
  goals line, fitted on the ten divisions. 2015's weights (fitted on
  2005-2014): beta 1.088, c_d 0.001, c_a -0.028, the goals-line weight
  -0.062 per standard deviation of logit q_o over the ten divisions (sd
  0.253, so -0.24 per unit of logit) - a 2.5-sd low-goals match gets its
  draw logit +0.15, a 2.5-sd high-goals one -0.15.

                    candidate   today's pick   paired diff (cand - pick)
    log loss        0.98112     0.98133        -0.00022  SE 0.00015
    Brier           0.58516     0.58522        -0.00006  SE 0.00008
    accuracy        52.55%      52.57%
    the draw alone (binary log loss of p_d)
                    0.56424     0.56439        -0.00015  SE 0.00014
    seasons better: 7 of 10 (season-level SE of the mean diff 0.00011)
    per league (log loss cand / pick, diff, SE):
      Premier League  0.96757 / 0.96760  -0.00003 (0.00028)
      La Liga         0.95090 / 0.95155  -0.00066 (0.00034)
      Serie A         0.97500 / 0.97492  +0.00009 (0.00026)
      Bundesliga      1.00253 / 1.00298  -0.00045 (0.00046)
      Ligue 1         1.01374 / 1.01383  -0.00008 (0.00031)
    TUNE 2006-2010 -0.00010 (SE 0.00011), CHECK 2011-2015 -0.00033 (SE
    0.00027). It moves the pick by a median 0.4 points (largest of the three
    chances; 99th percentile 2.4) and changes the favourite in 41 matches.

  THE DRAW'S CALIBRATION, DEV, by decile of the goals line (no-vig over
  2.5): observed draw rate / pick / candidate
    lowest-goals decile  0.307 / 0.304 / 0.315    9th  0.211 / 0.224 / 0.215
    highest-goals decile 0.151 / 0.170 / 0.157
  The pick's draw is too flat in the goals line: priced long by 0.013-0.019
  in the two highest-goals deciles (the market's draw does not fall as fast
  as the goals line says), about right or short at the low end. The
  candidate fixes the top and overshoots the bottom decile by 0.007. By
  decile of each's own p_d it is mixed: its lowest decile 0.131 against an
  observed 0.135 (the pick's lowest, 0.140 against 0.134), its highest
  0.329 against 0.317 (the pick's 0.323 against 0.316).

  ROUND 1 (top five only), DEV diff (SE) / TUNE / CHECK:
    lo_d                      -0.00013 (0.00015) / +0.00008 / -0.00035
    lo_d+lo_a                 -0.00001 (0.00016) / +0.00037 / -0.00039
    lo_d+lo_a+lo_beta         +0.00021 (0.00021) / +0.00076 / -0.00034
    lo_d+lo2_d                -0.00007 (0.00018) / +0.00025 / -0.00038
    kd (rho 0/-0.06/-0.12)    +0.00009/+0.00012/+0.00015 (0.00008-9)
    kdir (rho -0.06)          +0.00014 (0.00009)
    free (the 1X2's own unconstrained logit, no goals line)
                              -0.00006 (0.00024) / +0.00030 / -0.00041
    the richest, ridge 0/300/1000: +0.00043 / +0.00004 / -0.00005
  Every extra weight lost on TUNE (2006-2010; fitted on one to five
  seasons - lo_d's weight was +0.03 when fitted on 2005 alone, -0.04 to
  -0.06 from 2008 on) and gained on CHECK, the unconstrained logit with no
  goals line in it as much as any. THE GOALS-IMPLIED DRAW (kd: the
  Dixon-Coles draw the line and the favourite imply, against the 1X2's)
  made the draw WORSE (draw diff +0.0001): the market's draw is already
  where a Poisson/Dixon-Coles matrix puts it; its error is in the slope.
  ROUND 2 (after round 1): lo_d with a ridge of 100/300 pseudo-matches
  -0.00016 / -0.00016 (TUNE +0.00003 / -0.00000); fitted on the ten
  divisions: lo_d -0.00022 (chosen), lo_d+lo2_d -0.00022 (0.00021),
  kd+lo_d -0.00020, lo_d+lo_a +0.00008, lo_d+lo_a+lo_beta +0.00029, kd
  -0.00001. Nested: the variant best on TUNE of all 30 is "lo_d ten"
  (-0.00010); on CHECK it gains -0.00033 (SE 0.00027). That nested check is
  NOT clean: round 2 was designed after round 1's CHECK numbers were seen.

  OVER 2.5: no variant beat the raw no-vig over/under on DEV (the plain
  recalibration +0.00010, SE 0.00027; + the 1X2's draw against the
  goals-implied one +0.00016; + the draw price +0.00035), so the candidate
  has no over-2.5 number of its own (CHOSEN["ou"] is None: no p_o25 column).

  Point in time: the scramble test at 2011-02-05 (every result, statistic
  and price - average, best, Bet365, Pinnacle, over/under, closing - of
  every division from that day on scrambled, seasons to 2011): 8,375
  earlier predictions unmoved (largest move 0.0); the cut day's 30 unmoved
  with its own prices kept; later ones moved. Runtime: ~23 s walk-forward
  over every season (most of it the goals-implied matrices, computed for
  every row though only round 1's kd uses them), ~35 s the default run.

CAVEATS.
  - The gain is small next to its noise: 1.5 standard errors on DEV after
    choosing among 30 variants on DEV; the nested check is 1.2 standard
    errors and was not clean. Two leagues carry it (La Liga, the
    Bundesliga); Serie A and the Premier League are flat.
  - The over/under average is football-data's aggregator's (Betbrain to
    2018-19, Oddsportal since): a different set of books from 2019-20 on, as
    for the 1X2 average. Refitting each season adapts only season by season.
  - The goals-implied matrix is a Poisson/Dixon-Coles idealisation; where
    the 1X2 and the over/under contradict any such matrix (Serie A run-ins
    of 2006-2010: soccer_scores' notes) kd is large, and is clipped at 4
    standard deviations, not removed.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_backtest as bt  # noqa: E402
import soccer_data as sd  # noqa: E402
import soccer_model  # noqa: E402
from soccer_scores import dc_matrix  # noqa: E402

NAME = "draws_from_goals"
OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/engine") / f"{NAME}.csv.gz"
MIN_TRAIN = soccer_model.MIN_TRAIN          # the pick's own minimum
TUNE = (2006, 2010)                         # the nested check: choose here ...
CHECK = (2011, 2015)                        # ... measure here
PIT_CUT = "2011-02-05"                      # the scramble test's DEV cut (a Saturday, 2010-11)
CLIP = 4.0
MAX_GOALS = 10

# The pick's model plus nothing: the starting point every variant extends.
BASE = {"terms": (), "rho": -0.06, "kappa": 0.0, "pool": "top5", "ou": None, "ou_terms": (), "ou_kappa": 0.0}
# This candidate's configuration (chosen on DEV; MEASURED in the docstring).
CHOSEN = dict(BASE, terms=("lo_d",), pool="ten")


# ----------------------------------------------------------------- prices

def _num(m, c):
    if c not in m:
        return np.full(len(m), np.nan)
    return pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)


def _no_vig(m, cols):
    """No-vig chances (n, k) from decimal prices; NaN where any is missing or
    not a price."""
    p = np.column_stack([_num(m, c) for c in cols])
    p[~(p > 0)] = np.nan
    inv = 1.0 / p
    return inv / inv.sum(axis=1)[:, None]


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def total_rate(q_o):
    """The Poisson total lam + nu with P(total >= 3) = q_o (bisection; the
    Dixon-Coles correction moves chance only among 0-0, 1-0, 0-1 and 1-1, so
    it leaves this alone). NaN where q_o is."""
    q = np.asarray(q_o, float)
    lo, hi = np.full(q.shape, 1e-3), np.full(q.shape, 12.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        over = 1 - np.exp(-mid) * (1 + mid + mid ** 2 / 2)
        hi = np.where(over > q, mid, hi)
        lo = np.where(over > q, lo, mid)
    return np.where(np.isfinite(q), (lo + hi) / 2, np.nan)


def _hda(mat):
    g = np.arange(mat.shape[1])
    x, y = g[:, None], g[None, :]
    return np.stack([(mat * (x > y)).sum(axis=(1, 2)), (mat * (x == y)).sum(axis=(1, 2)),
                     (mat * (x < y)).sum(axis=(1, 2))], axis=1)


def goals_implied(q3, q_o, rho, iters=45):
    """Per match, the Dixon-Coles matrix whose total matches the over/under
    (total_rate) and whose home share H / (H + A) matches the 1X2's,
    q_h / (q_h + q_a), with the given rho (bisection on lam / (lam + nu)).
    Returns its (H, D, A) (n, 3) and the total; NaN rows where q3 or q_o is
    missing. Every row is solved on its own numbers only."""
    q3 = np.asarray(q3, float)
    q_o = np.asarray(q_o, float)
    n = len(q3)
    out = np.full((n, 3), np.nan)
    mu_all = total_rate(q_o)
    ok = np.isfinite(q3).all(axis=1) & np.isfinite(mu_all)
    if not ok.any():
        return out, mu_all
    mu = mu_all[ok]
    share = q3[ok, 0] / (q3[ok, 0] + q3[ok, 2])
    lo, hi = np.full(len(mu), 1e-4), np.full(len(mu), 1 - 1e-4)
    for _ in range(iters):
        t = (lo + hi) / 2
        h = _hda(dc_matrix(t * mu, (1 - t) * mu, rho, MAX_GOALS))
        s = h[:, 0] / (h[:, 0] + h[:, 2])
        hi = np.where(s > share, t, hi)
        lo = np.where(s > share, lo, t)
    t = (lo + hi) / 2
    out[ok] = _hda(dc_matrix(t * mu, (1 - t) * mu, rho, MAX_GOALS))
    return out, mu_all


def features(m, rho):
    """Every raw feature of every row, from that row's own pre-match prices
    only (standardised later, on training rows)."""
    q = _no_vig(m, ["avg_h", "avg_d", "avg_a"])
    q_o = _no_vig(m, ["avg_o25", "avg_u25"])[:, 0]
    lq = np.log(np.clip(q, 1e-9, 1.0))
    K, mu = goals_implied(q, q_o, rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = _logit(q_o)
        lo = np.where(np.isfinite(q_o), lo, np.nan)
        kdir = np.log(np.clip(K, 1e-9, 1.0)) - lq
        return {"q": q, "lq": lq, "q_o": q_o, "lo": lo, "lo2": lo * lo, "mu": mu, "K": K,
                "kd": kdir[:, 1], "kdir": kdir, "ld": _logit(q[:, 1]), "lop": np.abs(lq[:, 0] - lq[:, 2])}


def _scaler(v, rows):
    """(mean, sd) of the training rows' finite values."""
    v = v[rows]
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    mu = float(v.mean())
    s = float(np.sqrt(((v - mu) ** 2).mean()))
    return mu, s if s > 0 else 1.0


def _std(v, ms):
    z = (v - ms[0]) / ms[1]
    return np.clip(np.where(np.isfinite(z), z, 0.0), -CLIP, CLIP)


def _scales(f, train):
    sc = {k: _scaler(f[k], train) for k in ("lo", "lo2", "kd", "ld", "lop")}
    kd = f["kdir"][train]
    kd = kd[np.isfinite(kd).all(axis=1)]
    sc["kdir"] = (0.0, float(np.sqrt((kd ** 2).mean())) if len(kd) else 1.0)
    return sc


def _design(f, rows, terms, sc, tier2=None):
    """The 1X2 design (n, 3, P): beta, c_d, c_a (the pick's model); with
    `tier2` (pool "ten": a bool per row, True for a second-tier match) the
    second tiers' own shifts of those three; then the extra terms in order."""
    lq = f["lq"][rows]
    n = len(lq)
    e_d = np.zeros((n, 3)); e_d[:, 1] = 1.0
    e_a = np.zeros((n, 3)); e_a[:, 2] = 1.0
    cols = [lq, e_d, e_a]
    if tier2 is not None:
        t2 = np.asarray(tier2, float)[:, None]
        cols += [t2 * lq, t2 * e_d, t2 * e_a]
    x = _std(f["lo"][rows], sc["lo"])[:, None]
    for t in terms:
        if t == "lo_d":
            cols.append(x * e_d)
        elif t == "lo_a":
            cols.append(x * e_a)
        elif t == "lo_beta":
            cols.append(x * lq)
        elif t == "lo2_d":
            cols.append(_std(f["lo2"][rows], sc["lo2"])[:, None] * e_d)
        elif t == "kd":
            cols.append(_std(f["kd"][rows], sc["kd"])[:, None] * e_d)
        elif t == "kdir":
            k = f["kdir"][rows] / sc["kdir"][1]
            cols.append(np.clip(np.where(np.isfinite(k), k, 0.0), -CLIP, CLIP))
        elif t == "free":
            for j in range(3):
                cols += [lq[:, j:j + 1] * e_d, lq[:, j:j + 1] * e_a]
        else:
            raise ValueError(f"unknown term {t!r}")
    return np.stack(cols, axis=2)


def _design_ou(f, rows, terms, sc):
    """The over-2.5 design (n, P): a * logit q_o, c, then the extra terms."""
    lo = _logit(f["q_o"][rows])
    cols = [lo, np.ones_like(lo)]
    for t in terms:
        if t not in ("kd", "ld", "lop"):
            raise ValueError(f"unknown over-2.5 term {t!r}")
        cols.append(_std(f[t][rows], sc[t]))
    return np.column_stack(cols)


# ----------------------------------------------------------------- fitting

def _fit_multi(X, y, kappa=0.0, n_base=3):
    """Weights minimising the total log loss + kappa * |extra|^2 (every
    column after the pick's three), over n. Convex; L-BFGS with an analytic
    gradient, tight tolerances, started at the raw price (1, 0, 0, 0...)."""
    n, _, P = X.shape
    Y = np.zeros((n, 3)); Y[np.arange(n), y] = 1.0
    mask = np.zeros(P); mask[n_base:] = 1.0

    def loss(t):
        z = X @ t
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        g = -np.einsum("nk,nkp->p", Y - np.exp(lp), X)
        return (-lp[np.arange(n), y].sum() + kappa * (mask * t * t).sum()) / n, (g + 2 * kappa * mask * t) / n

    t0 = np.zeros(P); t0[0] = 1.0
    return minimize(loss, t0, jac=True, method="L-BFGS-B",
                    options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000}).x


def _fit_binary(X, y, kappa=0.0, n_base=2):
    n, P = X.shape
    mask = np.zeros(P); mask[n_base:] = 1.0

    def loss(t):
        z = X @ t
        lp1, lp0 = -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z)
        g = -X.T @ (y - np.exp(lp1))
        return (-(y * lp1 + (1 - y) * lp0).sum() + kappa * (mask * t * t).sum()) / n, (g + 2 * kappa * mask * t) / n

    t0 = np.zeros(P); t0[0] = 1.0
    return minimize(loss, t0, jac=True, method="L-BFGS-B",
                    options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000}).x


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------- walk forward

def predict_walk_forward(matches, cfg=None, feats=None):
    """Walk-forward p_h, p_d, p_a (and p_o25 when cfg["ou"] == "fit") for
    every top-five row of `matches` with a full pre-match average 1X2 price
    (NaN elsewhere; p_o25 NaN where no over/under average). Season s uses
    weights, means and standard deviations fitted on top-five matches of
    seasons before s with a result and a full average price (at least
    MIN_TRAIN, as for the pick; NaN before). .attrs["weights"]: {season:
    {"1x2": [...], "ou": [...]}}. feats: features(matches, rho), if already
    computed for this very frame."""
    cfg = dict(CHOSEN, **(cfg or {}))
    m = matches
    f = features(m, cfg["rho"]) if feats is None else feats
    league = m["league"].to_numpy()
    season = pd.to_numeric(m["season"], errors="coerce").to_numpy(float)
    top = np.isin(league, sd.TOP)
    priced = np.isfinite(f["q"]).all(axis=1)
    finished = priced & m["result"].isin(bt.RESULTS).to_numpy()
    done = top & finished
    ten = cfg.get("pool", "top5") == "ten"
    if ten:     # every one of the ten divisions trains (second tiers with their own base shifts)
        rated = finished & np.isin(league, list(sd.LEAGUES))
        n_base = 6
    else:
        rated, n_base = done, 3
    y = bt.outcome_index(m["result"]).to_numpy()
    tot = _num(m, "hg") + _num(m, "ag")
    ou_ok = np.isfinite(f["q_o"])
    fit_ou = cfg.get("ou") == "fit"
    out = np.full((len(m), 4), np.nan)
    weights = {}
    for s in sorted(np.unique(season[top & priced & np.isfinite(season)])):
        train = done & (season < s)
        if train.sum() < MIN_TRAIN:
            continue
        fit_rows = rated & (season < s)
        sc = _scales(f, fit_rows)
        tr = np.flatnonzero(fit_rows)
        te = np.flatnonzero(top & priced & (season == s))
        t2_tr, t2_te = (~top[tr], np.zeros(len(te), bool)) if ten else (None, None)
        theta = _fit_multi(_design(f, tr, cfg["terms"], sc, t2_tr), y[tr].astype(int), cfg["kappa"], n_base)
        out[te, :3] = _softmax(_design(f, te, cfg["terms"], sc, t2_te) @ theta)
        rec = {"1x2": [float(v) for v in theta]}
        if fit_ou:
            tro = np.flatnonzero(train & ou_ok & np.isfinite(tot))
            teo = te[ou_ok[te]]
            if len(tro) >= MIN_TRAIN and len(teo):
                sco = _scales(f, train & ou_ok)
                th = _fit_binary(_design_ou(f, tro, cfg["ou_terms"], sco), (tot[tro] > 2.5).astype(float),
                                 cfg["ou_kappa"])
                out[teo, 3] = 1.0 / (1.0 + np.exp(-(_design_ou(f, teo, cfg["ou_terms"], sco) @ th)))
                rec["ou"] = [float(v) for v in th]
        weights[int(s)] = rec
    res = pd.DataFrame(out, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    if not fit_ou:
        res = res.drop(columns="p_o25")
    res.attrs["weights"] = weights
    return res


def shipped_pick(matches):
    """Today's pick on the same frame: soccer_model.pick (only priced rows
    are compared, so the model's own chances are not needed - NaN)."""
    nan = pd.DataFrame(np.nan, index=matches.index, columns=["p_h", "p_d", "p_a"])
    p, _ = soccer_model.pick(matches, nan)
    return p[["p_h", "p_d", "p_a"]].where(~p["model_only"])


# ----------------------------------------------------------------- measurement

def _guard(lo, hi):
    if lo < bt.DEV[0] or hi > bt.DEV[1]:
        raise ValueError("this module reports DEV (2006-2015) only")


def _paired(a, b):
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def _bin_ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(m, cand, pick, lo, hi):
    """Candidate vs today's pick on the same scored top-five matches of
    seasons lo..hi (DEV only): log loss, Brier, accuracy, the paired
    difference (candidate - pick, negative = better) with its standard
    error, per season and per league; the draw alone (binary log loss and
    Brier of p_d); over 2.5 against the no-vig over/under market."""
    _guard(lo, hi)
    sel = (bt.scored(m) & m["season"].between(lo, hi)
           & cand[["p_h", "p_d", "p_a"]].notna().all(axis=1) & pick[["p_h", "p_d", "p_a"]].notna().all(axis=1))
    r = m.loc[sel, "result"]
    y = bt.outcome_index(r).to_numpy()
    pc = cand.loc[sel, ["p_h", "p_d", "p_a"]].to_numpy(float)
    pp = pick.loc[sel, ["p_h", "p_d", "p_a"]].to_numpy(float)
    llc = -np.log(np.clip(pc[np.arange(len(y)), y], 1e-12, 1))
    llp = -np.log(np.clip(pp[np.arange(len(y)), y], 1e-12, 1))
    onehot = np.eye(3)[y]
    brc, brp = ((pc - onehot) ** 2).sum(axis=1), ((pp - onehot) ** 2).sum(axis=1)
    d, se = _paired(llc, llp)
    db, seb = _paired(brc, brp)
    seasons = m.loc[sel, "season"].to_numpy()
    by_season = pd.Series(llc - llp).groupby(seasons).mean()
    yd = (y == 1).astype(float)
    dlc, dlp = _bin_ll(pc[:, 1], yd), _bin_ll(pp[:, 1], yd)
    dd, dse = _paired(dlc, dlp)
    dbd, dbse = _paired((pc[:, 1] - yd) ** 2, (pp[:, 1] - yd) ** 2)
    out = {
        "matches": int(sel.sum()),
        "log_loss": float(llc.mean()), "pick_log_loss": float(llp.mean()),
        "brier": float(brc.mean()), "pick_brier": float(brp.mean()),
        "accuracy": bt.accuracy(cand[sel], r), "pick_accuracy": bt.accuracy(pick[sel], r),
        "diff": d, "se": se, "brier_diff": db, "brier_se": seb,
        "se_by_season": float(by_season.std(ddof=1) / math.sqrt(len(by_season))) if len(by_season) > 1 else None,
        "seasons_better": f"{int((by_season < 0).sum())} of {len(by_season)}",
        "by_season": {int(k): round(float(v), 5) for k, v in by_season.items()},
        "draw": {"rate": float(yd.mean()), "mean_p_d": float(pc[:, 1].mean()), "pick_mean_p_d": float(pp[:, 1].mean()),
                 "log_loss": float(dlc.mean()), "pick_log_loss": float(dlp.mean()), "diff": dd, "se": dse,
                 "brier": float(((pc[:, 1] - yd) ** 2).mean()), "pick_brier": float(((pp[:, 1] - yd) ** 2).mean()),
                 "brier_diff": dbd, "brier_se": dbse},
        "moved": {"median_max_abs": float(np.median(np.abs(pc - pp).max(axis=1))),
                  "p99_max_abs": float(np.quantile(np.abs(pc - pp).max(axis=1), 0.99)),
                  "favourite_changed": int((pc.argmax(axis=1) != pp.argmax(axis=1)).sum())},
    }
    lg = m.loc[sel, "league"].to_numpy()
    per = {}
    for code in sd.TOP:
        k = lg == code
        if k.sum() < 2:
            continue
        a, s_ = _paired(llc[k], llp[k])
        per[code] = {"matches": int(k.sum()), "log_loss": round(float(llc[k].mean()), 5),
                     "pick_log_loss": round(float(llp[k].mean()), 5),
                     "brier": round(float(brc[k].mean()), 5), "pick_brier": round(float(brp[k].mean()), 5),
                     "accuracy": round(float((pc[k].argmax(axis=1) == y[k]).mean()), 4),
                     "pick_accuracy": round(float((pp[k].argmax(axis=1) == y[k]).mean()), 4),
                     "diff": round(a, 5), "se": round(s_, 5)}
    out["per_league"] = per
    if "p_o25" in cand:
        ou = (sel & cand["p_o25"].notna() & bt.no_vig_total(m).notna()).to_numpy()
        if ou.sum() > 1:
            yo = ((m.loc[ou, "hg"] + m.loc[ou, "ag"]) > 2.5).to_numpy(float)
            co = cand.loc[ou, "p_o25"].to_numpy(float)
            mo = bt.no_vig_total(m)[ou].to_numpy(float)
            lc, lm = _bin_ll(co, yo), _bin_ll(mo, yo)
            a, s_ = _paired(lc, lm)
            out["over_2_5"] = {"matches": int(ou.sum()), "log_loss": float(lc.mean()), "market_log_loss": float(lm.mean()),
                               "brier": float(((co - yo) ** 2).mean()), "market_brier": float(((mo - yo) ** 2).mean()),
                               "diff": a, "se": s_}
    return out


def draw_calibration(m, cand, pick, lo, hi, by="p_d", bins=10):
    """The draw's calibration on DEV: matches in deciles of the candidate's
    p_d (by="p_d"; the pick's own deciles under "pick") or of the no-vig
    over 2.5 (by="q_o": does the draw follow the goals line?), with each
    decile's mean candidate p_d, mean pick p_d and observed draw rate."""
    _guard(lo, hi)
    sel = (bt.scored(m) & m["season"].between(lo, hi)
           & cand[["p_h", "p_d", "p_a"]].notna().all(axis=1) & pick[["p_h", "p_d", "p_a"]].notna().all(axis=1))
    d = pd.DataFrame({"cand": cand.loc[sel, "p_d"], "pick": pick.loc[sel, "p_d"],
                      "q_o": bt.no_vig_total(m)[sel], "draw": (m.loc[sel, "result"] == "D").astype(float)})
    out = {}
    for key in (["cand", "pick"] if by == "p_d" else ["q_o"]):
        dd = d[d[key].notna()].copy()
        dd["bin"] = pd.qcut(dd[key].rank(method="first"), bins, labels=False)
        g = dd.groupby("bin").agg(n=("draw", "size"), lo=(key, "min"), hi=(key, "max"), cand=("cand", "mean"),
                                  pick=("pick", "mean"), observed=("draw", "mean"))
        se = np.sqrt(g["observed"] * (1 - g["observed"]) / g["n"])
        g["se"] = se
        out[f"deciles of {key}"] = g.round(4)
    return out


# ----------------------------------------------------------------- choosing (DEV only)

def grid():
    """The 1X2 variants compared on DEV (round 1)."""
    v = [("pick (no terms)", {"terms": ()}),
         ("lo_d", {"terms": ("lo_d",)}),
         ("lo_d+lo_a", {"terms": ("lo_d", "lo_a")}),
         ("lo_d+lo_a+lo_beta", {"terms": ("lo_d", "lo_a", "lo_beta")}),
         ("lo_d+lo2_d", {"terms": ("lo_d", "lo2_d")}),
         ("kd rho 0", {"terms": ("kd",), "rho": 0.0}),
         ("kd rho -0.06", {"terms": ("kd",), "rho": -0.06}),
         ("kd rho -0.12", {"terms": ("kd",), "rho": -0.12}),
         ("kdir rho -0.06", {"terms": ("kdir",), "rho": -0.06}),
         ("kd+lo_d", {"terms": ("kd", "lo_d")}),
         ("kd+lo_d+lo_a", {"terms": ("kd", "lo_d", "lo_a")}),
         ("kd+lo_d+lo_a+lo_beta", {"terms": ("kd", "lo_d", "lo_a", "lo_beta")}),
         ("free", {"terms": ("free",)}),
         ("free+lo_d+lo_a", {"terms": ("free", "lo_d", "lo_a")}),
         ("free+kd+lo_d+lo_a+lo_beta", {"terms": ("free", "kd", "lo_d", "lo_a", "lo_beta")})]
    for name, cfg in list(v):
        if name in ("lo_d+lo_a+lo_beta", "kd+lo_d+lo_a+lo_beta", "free+lo_d+lo_a", "free+kd+lo_d+lo_a+lo_beta"):
            v += [(f"{name} k300", dict(cfg, kappa=300.0)), (f"{name} k1000", dict(cfg, kappa=1000.0))]
    return [(n, dict(BASE, **c)) for n, c in v]


def grid2():
    """Round 2, after round 1 showed every extra weight losing on 2006-2010
    (fitted on one to five seasons; lo_d's draw weight was +0.03 when fitted
    on 2005 alone and -0.04 to -0.06 from 2008 on) and gaining on 2011-2015:
    a light ridge on lo_d, and the same terms fitted on all ten divisions
    (the second tiers' priced matches too, with their own beta / draw / away
    shifts) - more matches for the extra weights, the same earlier seasons."""
    v = [("lo_d k100", {"terms": ("lo_d",), "kappa": 100.0}),
         ("lo_d k300", {"terms": ("lo_d",), "kappa": 300.0}),
         ("pick ten (no terms)", {"terms": (), "pool": "ten"}),
         ("lo_d ten", {"terms": ("lo_d",), "pool": "ten"}),
         ("lo_d+lo_a ten", {"terms": ("lo_d", "lo_a"), "pool": "ten"}),
         ("lo_d+lo2_d ten", {"terms": ("lo_d", "lo2_d"), "pool": "ten"}),
         ("lo_d+lo_a+lo_beta ten", {"terms": ("lo_d", "lo_a", "lo_beta"), "pool": "ten"}),
         ("kd ten", {"terms": ("kd",), "pool": "ten"}),
         ("kd+lo_d ten", {"terms": ("kd", "lo_d"), "pool": "ten"})]
    return [(n, dict(BASE, **c)) for n, c in v]


def ou_grid():
    """The over-2.5 variants (binary logit on logit q_o), against the raw
    no-vig over/under market."""
    v = [("ou recal only", ()), ("ou kd", ("kd",)), ("ou ld", ("ld",)), ("ou kd+ld", ("kd", "ld")),
         ("ou kd+ld+lop", ("kd", "ld", "lop"))]
    out = [(n, dict(BASE, ou="fit", ou_terms=t)) for n, t in v]
    out.append(("ou kd+ld+lop k1000", dict(BASE, ou="fit", ou_terms=("kd", "ld", "lop"), ou_kappa=1000.0)))
    return out


def tune(matches, pick, log=print, variants=None):
    """Every variant walk-forwarded over DEV and compared with the pick on
    the same matches - full DEV, TUNE and CHECK. Never scores (nor even
    predicts) a season after 2015. Returns a frame sorted by DEV
    difference."""
    rows = []
    dev = matches[matches["season"] <= bt.DEV[1]]
    pk = pick.loc[dev.index]
    feats = {}
    for name, cfg in (grid() + grid2() + ou_grid()) if variants is None else variants:
        if cfg["rho"] not in feats:
            feats[cfg["rho"]] = features(dev, cfg["rho"])
        c = predict_walk_forward(dev, cfg, feats=feats[cfg["rho"]])
        ou = cfg.get("ou") == "fit"
        r = {"variant": name, "market": "over 2.5 (vs no-vig market)" if ou else "1x2"}
        for tag, (lo, hi) in (("dev", bt.DEV), ("tune", TUNE), ("check", CHECK)):
            x = compare(dev, c, pk, lo, hi)
            x = x["over_2_5"] if ou else x
            r[f"{tag}_diff"], r[f"{tag}_se"] = x["diff"], x["se"]
            if not ou:
                r[f"{tag}_draw_diff"] = x["draw"]["diff"]
        rows.append(r)
        log(f"  {name:>34}: DEV {r['dev_diff']:+.5f} ({r['dev_se']:.5f})  tune {r['tune_diff']:+.5f} "
            f"({r['tune_se']:.5f})  check {r['check_diff']:+.5f} ({r['check_se']:.5f})"
            + ("" if ou else f"  draw {r['dev_draw_diff']:+.5f}"))
    return pd.DataFrame(rows).sort_values(["market", "dev_diff"])


# ----------------------------------------------------------------- point in time

PRICE_PREFIXES = ("avg_", "max_", "ps_", "b365_", "avgc_", "psc_")
STATS = ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar")


def pit_check(matches, cut=PIT_CUT, cfg=None, base=None):
    """Scramble test at `cut` (a DEV date): every result, match statistic AND
    price (pre-match average / best / Bet365 / Pinnacle, over/under,
    closing) of matches dated on or after the cut is scrambled; every
    candidate prediction dated before the cut must not move. A second
    scramble keeps the cut day's own prices (results and statistics from the
    cut on, prices from the next day on): the cut day's predictions must not
    move either. Runs on seasons up to the cut's season + 1 (the later
    season is fitted on scrambled rows by design: it shows the scramble
    bites). Returns a dict of checks."""
    rng = np.random.default_rng(1)
    cut_season = int(matches.loc[matches["date"] >= cut, "season"].min())
    frame = matches[matches["season"] <= cut_season + 1]
    base = predict_walk_forward(frame, cfg) if base is None else base.loc[frame.index]
    later = (frame["date"] >= cut).to_numpy()
    after = (frame["date"] > cut).to_numpy()
    before = (frame["date"] < cut).to_numpy()
    on = (frame["date"] == cut).to_numpy()

    def scramble(df, res_rows, price_rows):
        s = df.copy()
        k = int(res_rows.sum())
        for c in STATS:
            if c in s:
                s.loc[res_rows, c] = rng.integers(0, 6 if c in ("hg", "ag", "hthg", "htag") else 25, k)
        s.loc[res_rows, "result"] = np.where(s.loc[res_rows, "hg"] > s.loc[res_rows, "ag"], "H",
                                             np.where(s.loc[res_rows, "hg"] < s.loc[res_rows, "ag"], "A", "D"))
        k = int(price_rows.sum())
        for c in [c for c in s.columns if c.startswith(PRICE_PREFIXES)]:
            s.loc[price_rows, c] = rng.uniform(1.05, 15.0, k)
        return s

    x0 = base.to_numpy(float)
    a = predict_walk_forward(scramble(frame, later, later), cfg).to_numpy(float)
    b = predict_walk_forward(scramble(frame, later, after), cfg).to_numpy(float)

    def same(x, y):
        return bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x, y, equal_nan=True, rtol=0, atol=1e-12))
    return {
        "cut": cut, "frame": f"seasons to {cut_season + 1}",
        "before_cut_predictions": int(np.isfinite(x0[before, 0]).sum()),
        "before_cut_unmoved": same(x0[before], a[before]),
        "max_move_before_cut": float(np.nanmax(np.abs(x0[before] - a[before]))) if before.any() else None,
        "cut_day_predictions": int(np.isfinite(x0[on, 0]).sum()),
        "cut_day_unmoved_results_scrambled": same(x0[on | before], b[on | before]),
        "after_cut_moved": not same(x0[later], a[later]),
    }


def live_features_present(fixtures=None):
    """Every price column this candidate reads, and the share of the live
    fixtures' top-five rows that carry it."""
    fx = sd.load_fixtures() if fixtures is None else fixtures
    fx = fx[fx["league"].isin(sd.TOP)]
    cols = ["avg_h", "avg_d", "avg_a", "avg_o25", "avg_u25"]
    return {"rows": int(len(fx)), **{c: (float(pd.to_numeric(fx[c], errors="coerce").notna().mean())
                                         if c in fx and len(fx) else 0.0) for c in cols}}


# ----------------------------------------------------------------- run

def _print_cal(cal):
    for k, g in cal.items():
        print(f"  {k}:")
        print("    " + g.to_string().replace("\n", "\n    "))


def main():
    args = sys.argv[1:]
    t0 = time.time()
    matches = sd.load_matches()
    cand = predict_walk_forward(matches)
    t_pred = time.time() - t0
    pick = shipped_pick(matches)
    top = matches["league"].isin(sd.TOP)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([matches.loc[top, ["match_id"]], cand.loc[top]], axis=1).to_csv(OUT, index=False, compression="gzip")
    print(f"walk-forward, every season: {t_pred:.1f}s; wrote {OUT} ({int(top.sum())} top-five matches, "
          f"{int(cand.loc[top, 'p_h'].notna().sum())} priced)")
    rep = {"DEV": compare(matches, cand, pick, *bt.DEV), "TUNE": compare(matches, cand, pick, *TUNE),
           "CHECK": compare(matches, cand, pick, *CHECK)}
    print(json.dumps(rep, indent=1))
    print("draw calibration, DEV, by decile of each's own p_d:")
    _print_cal(draw_calibration(matches, cand, pick, *bt.DEV, by="p_d"))
    print("draw by decile of the goals line (no-vig over 2.5), DEV:")
    _print_cal(draw_calibration(matches, cand, pick, *bt.DEV, by="q_o"))
    w = cand.attrs["weights"]
    print("weights (last DEV season, fitted on 2005-2014):", json.dumps(w.get(2015)))
    print("live fixtures carry:", live_features_present())
    if "--tune" in args:
        t = tune(matches, pick)
        t.to_csv(OUT.parent / f"{NAME}_tune.csv", index=False)
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    if "--no-pit" not in args:
        t1 = time.time()
        print("point in time:", pit_check(matches, PIT_CUT, base=cand), f"({time.time() - t1:.1f}s)")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
