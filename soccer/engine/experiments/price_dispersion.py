"""Candidate: the bookmakers' disagreement - a recalibration of the market
that can read the best price and Bet365's beside the average.

THE IDEA. Today's pick (soccer_model.pick) is the Friday/Tuesday average
price with the cut removed, q, recalibrated: p_k proportional to
q_k^beta * exp(c_k), three weights per season fitted on earlier seasons. One
beta for every match. But the fixtures file carries more than the average:
the BEST price any book posted (max_*) and Bet365's (b365_*), for every
outcome. Where the best price is far above the average, or Bet365 differs
from the average, the books disagree - the market is less sure, and its
price might deserve a different beta, or a different draw / away shift.
And WHERE the best price sits (which outcome some book is long on) may say
which way the average is wrong.

THE MODEL. A conditional (multinomial) logit, three outcomes, c_home = 0:

  z_k = beta(x) * log q_k + c_k(x) + sum_j a_j * dir_jk,   p = softmax(z)
  beta(x) = beta0 + b . x,   c_d(x) = c_d0 + g_d . x,   c_a(x) = c_a0 + g_a . x

  x, per match (cfg "scalars"; each standardised by its training rows' mean and
  standard deviation, clipped to +-4, 0 where a price is missing):
    over     the average's overround, sum 1/avg - 1
    spread   the average's overround minus the best prices' overround -
             how much better the best book on each outcome is than the
             average, summed (the books' disagreement)
    b365     sum over outcomes of |Bet365's no-vig chance - the average's|
    lop      |log q_h - log q_a|, how lopsided the match is - NOT
             disagreement, a control: the best-vs-average gap is wider on a
             long price, so a beta that moves with "spread" could just be a
             beta that moves with the price (measured with and without)
  dir, per outcome (cfg "directional"; scaled by the training rows' standard
  deviation, clipped to +-4, 0 where missing):
    dmax     log(best prices' no-vig chance / average's no-vig chance)
    d365     log(Bet365's no-vig chance / average's no-vig chance)
  options: a directional weight per outcome (dir_per_outcome), a
  directional weight that moves with the disagreement (dir_by: x * dir_k).

  With b = g = a = 0 it is the pick's model exactly (checked on the whole
  history: the same chances to 3e-9, the same weights). The extra weights
  can be shrunk towards 0 - towards the pick - by a ridge of kappa
  pseudo-matches (penalty kappa * |extra|^2 on the total log loss), so a
  season fitted on few earlier matches stays near the pick; optionally the
  training rows are down-weighted by age (halflife, in seasons).

  over 2.5: the same on the over/under 2.5 price, binary logit:
  logit p = a0 * logit q_o + c0 + interactions with the over/under's own
  overround and spread, plus the best price's direction (logit of the best
  prices' no-vig over minus logit q_o), measured against today's raw no-vig
  over/under market.

THE PROTOCOL (house rules). Point in time: a match's features are its own
PRE-MATCH prices only (avg_*, max_*, b365_*, avg/max over-under - collected
Friday/Tuesday; never closing prices, never Pinnacle). Every weight, mean and
standard deviation for season s is fitted on top-five matches of seasons
before s with a result and a full average price - the pick's own training
rows, at least MIN_TRAIN of them - refitted every season. The live fixtures
file (soccer_data.load_fixtures, read 2026-11-02) carries avg_*, max_*,
b365_* and the average and best over/under on every row (48 of 48 top-five
rows; 99 of 99 over the ten divisions): every feature here is one the live
pick has. (Pinnacle's ps_* is empty there, which is why it is not used.)

Choosing, DEV only (2006-2015; CONFIRM 2016-2025 is never scored by this
module - `compare` refuses it, and the CSV's later seasons were not looked
at). `grid()` (55 1X2 variants: which scalars, which directional terms, the
ridge 0/100/300/1000/3000, half-life none/5/8; two rounds, the second after
the first showed the best price's direction carrying the gain) and
`ou_grid()` (6 over-2.5 variants) were each walk-forwarded over DEV and
compared with the pick (or the no-vig over/under) on the same matches.
Because the variant is CHOSEN on DEV, its DEV number is optimistic, so the
nested check is reported too: the variant that is best on 2006-2010 alone,
scored on 2011-2015.

MEASURED - DEV 2006-2015, top five, the same 18,259 matches for both
(python experiments/price_dispersion.py; --tune prints the whole grid):

  CHOSEN: dmax only, no ridge, no half-life - ONE extra weight. That is a
  log pool of the average's and the best price's no-vig chances:
  p_k ~ q_avg_k^(beta - a) * q_max_k^a * exp(c_k); fitted each season, a is
  1.15-1.48 and beta - a is -0.50 to 0.00 (2015's, fitted on 2005-2014:
  beta 1.006, a 1.334, c_d -0.034, c_a -0.017) - lean on the best price,
  and past it, away from the average.

                    candidate   today's pick   paired diff (cand - pick)
    log loss        0.98105     0.98133        -0.00028  SE 0.00017
    Brier           0.58513     0.58522        -0.00009  SE 0.00010
    accuracy        52.63%      52.57%
    seasons better: 7 of 10 (season-level SE of the mean diff 0.00012)
    per league (log loss, cand / pick, diff, SE):
      Premier League  0.96778 / 0.96760  +0.00018 (0.00036)
      La Liga         0.95140 / 0.95155  -0.00016 (0.00041)
      Serie A         0.97499 / 0.97492  +0.00007 (0.00037)
      Bundesliga      1.00217 / 1.00298  -0.00082 (0.00047)
      Ligue 1         1.01305 / 1.01383  -0.00077 (0.00036)
  nested: chosen on 2006-2010 (-0.00041, SE 0.00027) - the same variant
  (with or without an 8-season half-life, -0.00042 / -0.00041) - and on
  2011-2015 it gains -0.00015 (SE 0.00022), 3 seasons of 5 better.
  It moves the pick by a median 0.7 points (largest of the three chances;
  99th percentile 2.2) and changes the favourite in 143 of 18,259 matches.

  THE DISPERSION ITSELF did nothing: beta / draw / away moving with the
  overround, the spread and Bet365's disagreement ("disp|none") lost to the
  pick unshrunk (+0.00025, SE 0.00033) and gained at most -0.00007 (SE
  0.00010) with a ridge; adding them to dmax never beat dmax alone on the
  nested check; nor did dmax's weight moving with the spread (-0.00023,
  check -0.00008), a weight per outcome (-0.00027) or Bet365's direction
  (-0.00025). The lopsidedness control alone: +0.00009 unshrunk.
  over 2.5: no variant beat the raw no-vig over/under on DEV (best +0.00002,
  SE 0.00026; the plain recalibration +0.00010), so this candidate has no
  over-2.5 number of its own (CHOSEN["ou"] is None: no p_o25 column).

  Point in time: the scramble test at 2011-02-05 (every result, statistic and
  price from that day on scrambled, seasons to 2011): 8,375 earlier
  predictions unmoved (largest move 0.0), the cut day's 30 unmoved with its
  own prices kept, later ones moved. Runtime: ~3 s walk-forward over every
  season, ~6 s the whole default run, ~2 min with --tune.

CAVEATS.
  - The gain is small next to its noise: 1.6 standard errors on DEV after
    choosing on DEV; 0.7 on the nested check. It is weaker in 2011-2015
    than in 2006-2010, as if the average were catching up with the best.
  - The best price is football-data's aggregator's maximum, and the
    aggregator changed in 2019-20 (Betbrain's BbMx to Oddsportal's Max, a
    different and larger set of books), and from 2025-07-23 Pinnacle is out
    of its maxima (soccer_data's notes). On DEV the best price often WAS the
    sharpest book's; live, it may not be. Walk-forward refitting adapts only
    season by season, so CONFIRM's later seasons and the live season test a
    feature whose meaning has moved.
  - A single book's error in the maximum (on file: a draw at 7.6x the
    average) is clipped at 4 standard deviations, not removed.
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

OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/engine/price_dispersion.csv.gz")
MIN_TRAIN = soccer_model.MIN_TRAIN          # the pick's own minimum
TUNE = (2006, 2010)                         # the nested check: choose here ...
CHECK = (2011, 2015)                        # ... measure here
PIT_CUT = "2011-02-05"                      # the scramble test's DEV cut (a Saturday, 2010-11)
CLIP = 4.0

# This candidate's configuration (chosen on DEV; MEASURED in the docstring).
# 1X2: no dispersion interaction survived; the best price's direction did.
# Over 2.5: no variant beat the raw no-vig market on DEV, so the candidate
# has no over-2.5 number of its own (ou None: no p_o25 column).
CHOSEN = {"scalars": (), "directional": ("dmax",), "kappa": 0.0, "halflife": None,
          "dir_per_outcome": False, "dir_by": (),
          "ou": None, "ou_scalars": ("ou_over", "ou_spread"), "ou_directional": ("ou_dmax",), "ou_kappa": 1000.0}


# ----------------------------------------------------------------- features

def _num(m, c):
    if c not in m:
        return np.full(len(m), np.nan)
    return pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)


def _no_vig(m, cols):
    """(no-vig chances (n, k), overround) from decimal prices; NaN where any
    is missing or not a price."""
    p = np.column_stack([_num(m, c) for c in cols])
    p[~(p > 0)] = np.nan
    inv = 1.0 / p
    tot = inv.sum(axis=1)
    return inv / tot[:, None], tot - 1.0


def price_features(m):
    """Every feature of every row, from that row's own pre-match prices only
    (never closing, never Pinnacle). Raw - standardised later on training
    rows. Returns a dict of arrays."""
    q, over = _no_vig(m, ["avg_h", "avg_d", "avg_a"])
    qm, over_m = _no_vig(m, ["max_h", "max_d", "max_a"])
    qb, _ = _no_vig(m, ["b365_h", "b365_d", "b365_a"])
    lq = np.log(np.clip(q, 1e-9, 1.0))
    qo, ou_over = _no_vig(m, ["avg_o25", "avg_u25"])
    qmo, ou_over_m = _no_vig(m, ["max_o25", "max_u25"])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = {
            "q": q, "lq": lq,
            "over": over,
            "spread": over - over_m,
            "b365": np.abs(qb - q).sum(axis=1),
            "lop": np.abs(lq[:, 0] - lq[:, 2]),
            "dmax": np.log(qm / q),
            "d365": np.log(qb / q),
            "q_o": qo[:, 0],
            "ou_over": ou_over,
            "ou_spread": ou_over - ou_over_m,
            "ou_dmax": _logit(qmo[:, 0]) - _logit(qo[:, 0]),
        }
    return f


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _scaler(f, names, rows, center):
    """{name: (mean, sd)} from the training rows' finite values only."""
    out = {}
    for n in names:
        v = f[n][rows]
        v = v[np.isfinite(v)]
        mu = float(v.mean()) if center and len(v) else 0.0
        sd_ = float(np.sqrt(((v - mu) ** 2).mean())) if len(v) else 1.0
        out[n] = (mu, sd_ if sd_ > 0 else 1.0)
    return out


def _std(v, ms):
    z = (v - ms[0]) / ms[1]
    return np.clip(np.where(np.isfinite(z), z, 0.0), -CLIP, CLIP)


def _design(f, rows, cfg, scale):
    """The 1X2 design tensor (n, 3, P): beta, c_d, c_a, then per scalar
    x*log q, x*[draw], x*[away], then the directional terms. The first three
    columns are the pick's model."""
    lq = f["lq"][rows]
    n = len(lq)
    e_h = np.zeros((n, 3)); e_h[:, 0] = 1.0
    e_d = np.zeros((n, 3)); e_d[:, 1] = 1.0
    e_a = np.zeros((n, 3)); e_a[:, 2] = 1.0
    cols = [lq, e_d, e_a]
    for s in cfg["scalars"]:
        x = _std(f[s][rows], scale[s])[:, None]
        cols += [x * lq, x * e_d, x * e_a]
    for d in cfg["directional"]:
        v = _std(f[d][rows], scale[d])
        if cfg.get("dir_per_outcome"):
            cols += [v * e_h, v * e_d, v * e_a]
        else:
            cols.append(v)
        for s in cfg.get("dir_by", ()):
            cols.append(_std(f[s][rows], scale[s])[:, None] * v)
    return np.stack(cols, axis=2)


def _design_ou(f, rows, cfg, scale):
    """The over-2.5 design (n, P): a0 * logit q_o, c0, per scalar
    x*logit q_o and x, then the directional terms."""
    lo = _logit(f["q_o"][rows])
    cols = [lo, np.ones_like(lo)]
    for s in cfg["ou_scalars"]:
        x = _std(f[s][rows], scale[s])
        cols += [x * lo, x]
    for d in cfg["ou_directional"]:
        cols.append(_std(f[d][rows], scale[d]))
    return np.column_stack(cols)


# ----------------------------------------------------------------- fitting

def _fit_multi(X, y, w=None, kappa=0.0, n_base=3):
    """Weights minimising sum_i w_i * log loss_i + kappa * |extra|^2 (the
    extra weights: every column after the pick's three), over sum w.
    Convex; L-BFGS with an analytic gradient, tight tolerances."""
    n, _, P = X.shape
    Y = np.zeros((n, 3)); Y[np.arange(n), y] = 1.0
    w = np.ones(n) if w is None else np.asarray(w, float)
    W = w.sum()
    mask = np.zeros(P); mask[n_base:] = 1.0

    def loss(t):
        z = X @ t
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        P_ = np.exp(lp)
        nll = -(w * lp[np.arange(n), y]).sum()
        g = -np.einsum("nk,nkp->p", (Y - P_) * w[:, None], X)
        pen = kappa * (mask * t * t).sum()
        return (nll + pen) / W, (g + 2 * kappa * mask * t) / W

    t0 = np.zeros(P); t0[0] = 1.0
    fit = minimize(loss, t0, jac=True, method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000})
    return fit.x


def _fit_binary(X, y, w=None, kappa=0.0, n_base=2):
    n, P = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    W = w.sum()
    mask = np.zeros(P); mask[n_base:] = 1.0

    def loss(t):
        z = X @ t
        lp1 = -np.logaddexp(0.0, -z)
        lp0 = -np.logaddexp(0.0, z)
        nll = -(w * (y * lp1 + (1 - y) * lp0)).sum()
        p = np.exp(lp1)
        g = -X.T @ (w * (y - p))
        return (nll + kappa * (mask * t * t).sum()) / W, (g + 2 * kappa * mask * t) / W

    t0 = np.zeros(P); t0[0] = 1.0
    fit = minimize(loss, t0, jac=True, method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 3000})
    return fit.x


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------- walk forward

def predict_walk_forward(matches, cfg=None):
    """Walk-forward p_h, p_d, p_a, p_o25 for every top-five row of `matches`
    with a full pre-match average price (NaN elsewhere: no price, no
    candidate; p_o25 NaN where no over/under average). Season s uses weights,
    means and standard deviations fitted on top-five matches of seasons
    before s with a result and a full average price - the pick's own
    training rows (at least MIN_TRAIN of them, as for the pick; NaN before).
    .attrs["weights"]: {season: {"1x2": [...], "ou": [...]}}."""
    cfg = dict(CHOSEN, **(cfg or {}))
    m = matches
    f = price_features(m)
    league = m["league"].to_numpy()
    season = pd.to_numeric(m["season"], errors="coerce").to_numpy(float)
    top = np.isin(league, sd.TOP)
    priced = np.isfinite(f["q"]).all(axis=1)
    done = top & priced & m["result"].isin(bt.RESULTS).to_numpy()
    y = bt.outcome_index(m["result"]).to_numpy()
    tot = (_num(m, "hg") + _num(m, "ag"))
    ou_ok = np.isfinite(f["q_o"])
    out = np.full((len(m), 4), np.nan)
    weights = {}
    for s in sorted(np.unique(season[top & priced & np.isfinite(season)])):
        train = done & (season < s)
        if train.sum() < MIN_TRAIN:
            continue
        test = top & priced & (season == s)
        tr = np.flatnonzero(train)
        w = None if cfg["halflife"] is None else 0.5 ** ((s - 1 - season[tr]) / cfg["halflife"])
        scale = _scaler(f, tuple(cfg["scalars"]) + tuple(cfg.get("dir_by", ())), train, center=True)
        scale.update(_scaler(f, cfg["directional"], train, center=False))
        theta = _fit_multi(_design(f, tr, cfg, scale), y[tr].astype(int), w, cfg["kappa"])
        te = np.flatnonzero(test)
        out[te, :3] = _softmax(_design(f, te, cfg, scale) @ theta)
        rec = {"1x2": [float(v) for v in theta], "scale": {k: [round(a, 6), round(b, 6)] for k, (a, b) in scale.items()}}
        # over 2.5
        tro = np.flatnonzero(train & ou_ok & np.isfinite(tot))
        teo = np.flatnonzero(test & ou_ok)
        if cfg.get("ou") == "fit" and len(tro) >= MIN_TRAIN and len(teo):
            sc = _scaler(f, cfg["ou_scalars"], train & ou_ok, center=True)
            sc.update(_scaler(f, cfg["ou_directional"], train & ou_ok, center=False))
            wo = None if cfg["halflife"] is None else 0.5 ** ((s - 1 - season[tro]) / cfg["halflife"])
            th = _fit_binary(_design_ou(f, tro, cfg, sc), (tot[tro] > 2.5).astype(float), wo, cfg["ou_kappa"])
            out[teo, 3] = 1.0 / (1.0 + np.exp(-(_design_ou(f, teo, cfg, sc) @ th)))
            rec["ou"] = [float(v) for v in th]
        weights[int(s)] = rec
    res = pd.DataFrame(out, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    if cfg.get("ou") != "fit":
        res = res.drop(columns="p_o25")
    res.attrs["weights"] = weights
    return res


def shipped_pick(matches):
    """Today's pick on the same frame: soccer_model.pick (the model's own
    chances are not needed - only priced rows are compared - so NaN is
    passed for them)."""
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


def compare(m, cand, pick, lo, hi):
    """Candidate vs today's pick on the same scored top-five matches of
    seasons lo..hi (DEV only): log loss, Brier, accuracy, the paired
    difference (candidate - pick, negative = better) with its standard
    error, per season and per league; and over 2.5 against the no-vig
    over/under market on the same matches."""
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
    out = {
        "matches": int(sel.sum()),
        "log_loss": float(llc.mean()), "pick_log_loss": float(llp.mean()),
        "brier": float(brc.mean()), "pick_brier": float(brp.mean()),
        "accuracy": bt.accuracy(cand[sel], r), "pick_accuracy": bt.accuracy(pick[sel], r),
        "diff": d, "se": se, "brier_diff": db, "brier_se": seb,
        "se_by_season": float(by_season.std(ddof=1) / math.sqrt(len(by_season))) if len(by_season) > 1 else None,
        "seasons_better": f"{int((by_season < 0).sum())} of {len(by_season)}",
        "by_season": {int(k): round(float(v), 5) for k, v in by_season.items()},
    }
    lg = m.loc[sel, "league"].to_numpy()
    per = {}
    for code in sd.TOP:
        k = lg == code
        if k.sum() < 2:
            continue
        dd, ss = _paired(llc[k], llp[k])
        per[code] = {"matches": int(k.sum()), "log_loss": round(float(llc[k].mean()), 5),
                     "pick_log_loss": round(float(llp[k].mean()), 5),
                     "brier": round(float(brc[k].mean()), 5), "pick_brier": round(float(brp[k].mean()), 5),
                     "diff": round(dd, 5), "se": round(ss, 5)}
    out["per_league"] = per
    if "p_o25" in cand:
        ou = (sel & cand["p_o25"].notna() & bt.no_vig_total(m).notna()).to_numpy()
        if ou.sum() > 1:
            yo = ((m.loc[ou, "hg"] + m.loc[ou, "ag"]) > 2.5).to_numpy(float)
            co = np.clip(cand.loc[ou, "p_o25"].to_numpy(float), 1e-12, 1 - 1e-12)
            mo = np.clip(bt.no_vig_total(m)[ou].to_numpy(float), 1e-12, 1 - 1e-12)
            lc = -(yo * np.log(co) + (1 - yo) * np.log(1 - co))
            lm = -(yo * np.log(mo) + (1 - yo) * np.log(1 - mo))
            dd, ss = _paired(lc, lm)
            out["over_2_5"] = {"matches": int(ou.sum()), "log_loss": float(lc.mean()), "market_log_loss": float(lm.mean()),
                               "brier": float(((co - yo) ** 2).mean()), "market_brier": float(((mo - yo) ** 2).mean()),
                               "diff": dd, "se": ss}
    return out


# ----------------------------------------------------------------- choosing (DEV only)

def grid():
    """The variants compared on DEV: which scalars, which directional terms,
    the ridge (pseudo-matches), the training half-life (seasons)."""
    scal = {"none": (), "disp": ("over", "spread", "b365"), "disp+lop": ("over", "spread", "b365", "lop"),
            "lop": ("lop",)}
    direc = {"none": (), "dmax": ("dmax",), "dmax+d365": ("dmax", "d365")}
    out = []
    for sk, sv in scal.items():
        for dk, dv in direc.items():
            if not sv and not dv:
                continue
            for kappa in (0.0, 300.0, 1000.0, 3000.0):
                out.append((f"{sk}|{dk}|k{kappa:.0f}", {"scalars": sv, "directional": dv, "kappa": kappa}))
    for name, cfg in list(out):
        if name in ("disp|dmax|k1000", "disp+lop|dmax+d365|k1000", "none|dmax|k1000"):
            out.append((name + "|h5", dict(cfg, halflife=5.0)))
    # round 2, after round 1 showed the direction of the best price carrying
    # the gain: its weight per outcome, its weight moving with the
    # disagreement (dir_by), a light ridge, a long half-life
    d = {"scalars": (), "directional": ("dmax",)}
    out += [("none|dmax|k100", dict(d, kappa=100.0)),
            ("none|dmax3|k0", dict(d, kappa=0.0, dir_per_outcome=True)),
            ("none|dmax3|k300", dict(d, kappa=300.0, dir_per_outcome=True)),
            ("none|dmax x spread|k0", dict(d, kappa=0.0, dir_by=("spread",))),
            ("none|dmax x spread,over|k0", dict(d, kappa=0.0, dir_by=("spread", "over"))),
            ("none|dmax x spread,over,b365|k300", dict(d, kappa=300.0, dir_by=("spread", "over", "b365"))),
            ("disp|dmax x spread|k1000", dict(scalars=("over", "spread", "b365"), directional=("dmax",), kappa=1000.0,
                                              dir_by=("spread",))),
            ("none|dmax|k0|h8", dict(d, kappa=0.0, halflife=8.0))]
    return out


def ou_grid():
    """The over-2.5 variants (binary logit on logit q_o), against the raw
    no-vig over/under market."""
    return [("ou recal only", dict(ou_scalars=(), ou_directional=(), ou_kappa=0.0)),
            ("ou dmax k0", dict(ou_scalars=(), ou_directional=("ou_dmax",), ou_kappa=0.0)),
            ("ou disp k1000", dict(ou_scalars=("ou_over", "ou_spread"), ou_directional=(), ou_kappa=1000.0)),
            ("ou full k0", dict(ou_scalars=("ou_over", "ou_spread"), ou_directional=("ou_dmax",), ou_kappa=0.0)),
            ("ou full k1000", dict(ou_scalars=("ou_over", "ou_spread"), ou_directional=("ou_dmax",), ou_kappa=1000.0)),
            ("ou full k3000", dict(ou_scalars=("ou_over", "ou_spread"), ou_directional=("ou_dmax",), ou_kappa=3000.0))]


def tune(matches, pick, log=print):
    """Every variant walk-forwarded over DEV and compared with the pick on
    the same matches - full DEV, TUNE and CHECK. Never scores a season after
    2015. Returns a frame sorted by DEV difference."""
    rows = []
    dev = matches[matches["season"] <= bt.DEV[1]]
    pk = pick.loc[dev.index]
    for name, cfg in grid():
        c = predict_walk_forward(dev, dict(cfg, ou=None))
        r = {"variant": name, "market": "1x2"}
        for tag, (lo, hi) in (("dev", bt.DEV), ("tune", TUNE), ("check", CHECK)):
            x = compare(dev, c, pk, lo, hi)
            r[f"{tag}_diff"], r[f"{tag}_se"] = x["diff"], x["se"]
        rows.append(r)
        log(f"  {name:>34}: DEV {r['dev_diff']:+.5f} ({r['dev_se']:.5f})  tune {r['tune_diff']:+.5f}  "
            f"check {r['check_diff']:+.5f}")
    for name, cfg in ou_grid():
        c = predict_walk_forward(dev, dict(cfg, scalars=(), directional=(), kappa=0.0, ou="fit"))
        r = {"variant": name, "market": "over 2.5 (vs no-vig market)"}
        for tag, (lo, hi) in (("dev", bt.DEV), ("tune", TUNE), ("check", CHECK)):
            x = compare(dev, c, pk, lo, hi)["over_2_5"]
            r[f"{tag}_diff"], r[f"{tag}_se"] = x["diff"], x["se"]
        rows.append(r)
        log(f"  {name:>34}: DEV {r['dev_diff']:+.5f} ({r['dev_se']:.5f})  tune {r['tune_diff']:+.5f}  "
            f"check {r['check_diff']:+.5f}")
    return pd.DataFrame(rows).sort_values(["market", "dev_diff"])


# ----------------------------------------------------------------- point in time

PRICE_PREFIXES = ("avg_", "max_", "ps_", "b365_", "avgc_", "psc_")


def pit_check(matches, cut=PIT_CUT, cfg=None, base=None):
    """Scramble test at `cut` (a DEV date): every result, match statistic AND
    price (pre-match average / best / Bet365 / Pinnacle, over/under,
    closing) of matches dated on or after the cut is scrambled; every
    candidate prediction dated before the cut must not move. A second
    scramble keeps the cut day's own prices (results and statistics from the
    cut on, prices from the next day on): the cut day's predictions must not
    move either. Runs on seasons up to the cut's season + 1 (later seasons
    are fitted on scrambled rows by design; nothing before the cut can read
    them). Returns a dict of checks."""
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
        for c in ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"):
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
        "cut_day_predictions": int(np.isfinite(x0[on, 0]).sum()),
        "cut_day_unmoved_results_scrambled": same(x0[on | before], b[on | before]),
        "max_move_before_cut": float(np.nanmax(np.abs(x0[before] - a[before]))) if before.any() else None,
        "after_cut_moved": not same(x0[later], a[later]),
    }


def live_features_present(fixtures=None):
    """Every price column this candidate reads, and the share of the live
    fixtures' top-five rows that carry it."""
    fx = sd.load_fixtures() if fixtures is None else fixtures
    fx = fx[fx["league"].isin(sd.TOP)]
    cols = ["avg_h", "avg_d", "avg_a", "max_h", "max_d", "max_a", "b365_h", "b365_d", "b365_a",
            "avg_o25", "avg_u25", "max_o25", "max_u25"]
    return {"rows": int(len(fx)), **{c: (float(pd.to_numeric(fx[c], errors="coerce").notna().mean()) if c in fx and len(fx)
                                         else 0.0) for c in cols}}


# ----------------------------------------------------------------- run

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
    w = cand.attrs["weights"]
    print("weights (last DEV season fitted on 2005-2014):", json.dumps(w.get(2015)))
    print("live fixtures carry:", live_features_present())
    if "--tune" in args:
        t = tune(matches, pick)
        t.to_csv(OUT.parent / "price_dispersion_tune.csv", index=False)
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    if "--no-pit" not in args:
        t1 = time.time()
        print("point in time:", pit_check(matches, PIT_CUT, base=cand), f"({time.time() - t1:.1f}s)")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
