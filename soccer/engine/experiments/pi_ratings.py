"""Candidate: pi-ratings (Constantinou & Fenton 2013), alone and as a second
input to the market recalibration. VERDICT ON DEV: DOES NOT SHIP - pi-ratings
carry no information the pre-match market lacks.

THE IDEA. Every club carries a HOME rating and an AWAY rating (goal-difference
scale; 0 = an average side). Before a match on day D, home club i v away club j:

    f(r) = sign(r) (10^(|r|/c) - 1)          goals r is expected to win by
    gd   = f(R_i^home) - f(R_j^away)         expected goal difference
    e    = (goals_i - goals_j) - gd          the error
    psi  = sign(e) c log10(1 + |e|)          large wins count less than linearly

and after every match of day D has its number:

    R_i^home += lam_i psi        R_i^away += gamma lam_i psi
    R_j^away -= lam_j psi        R_j^home -= gamma lam_j psi
    lam_k = lam + lam_new exp(-games_k / n_new)    (learns faster while new to a league)

One rating pool per league (the top five AND their second tiers, rated so
promoted clubs arrive with a history). At a club's first match of a season:
same league - shrunk toward the league's end-of-last-season mean by `shrink`;
moved tier - ref_new + up_a / down_a + tier_b (rating - ref_old); new to the
data - ref_new + new_a. At each season's end every league's clubs are shifted
together so its mean home + away rating is 0 (`recentre`; f is nonlinear, so
the level matters, and promotions would otherwise drift it).

THE MAPPING (pi alone). An ordered logit on gd and cross = f(R_i^away) -
f(R_j^home) (the other venues' ratings; Constantinou's split is noisy and the
other venue carries strength too), with a draw band of half-width
exp(g0 + g1 |gd|); refitted every season on the top five's matches of EVERY
earlier season (from 2005-10-01, after the warm-up's first weeks), dated
before the season's first match.

AS A SECOND INPUT. p_k proportional to q_k^beta * m_k^alpha * exp(c_k): q the
no-vig PRE-MATCH average (avg_*), m pi alone; (beta, alpha, c_draw, c_away)
fitted each season on earlier seasons' priced top-five matches for which pi
had an out-of-sample number (2006 on), so the first pooled season is 2007.
Analytic gradient, tight tolerances (as soccer_model.fit_recalibration). The
CONTROL is the same fit with alpha = 0 on the same rows - the shipped pick's
form - so "pooled vs control" isolates what pi adds.

predict_walk_forward(matches) -> p_h, p_d, p_a (the pooled candidate; NaN
without a full pre-match price or before 2007), pi_h, pi_d, pi_a (pi alone,
from 2006), gd; second-tier rows NaN. main() writes every season to
/tmp/.../scratchpad/engine/pi_ratings.csv.gz and prints DEV only.

POINT IN TIME. A day's numbers are taken before any of the day's results
enter; season means are frozen before the next season's first match; the
mapping and the pooling see only matches dated before the season's first
match; the only price read is the match's own pre-match average (never a
closing price, never Pinnacle). Scramble test at 2011-02-05 (every result,
match statistic AND price - avg/max/b365/ps/closing/over-under - from the cut
on replaced by noise): all 8,376 top-five predictions dated before the cut
unchanged (pooled and pi alone, to 1e-12); pi alone ON the cut day (30
matches) unchanged; the pooled number on the cut day moves, as it must (it
reads that match's own pre-match price); pi after the cut moves (the test
bites). tests/test_pi_ratings.py repeats it on a made-up league and checks
the update against the paper by hand.

CHOSEN ON DEV (2006-2015) ONLY, by pi alone's DEV log loss (CONFIRM never
computed here):
  start     the paper's lam .035, gamma .7, c 3, gd only, no recentre: 0.99800
  grids     recentre on/off x lam {.035,.06,.09} x gamma {.5,.7};
            lam {.015,.02,.025,.03} x gamma {.7,.85,1} x c {2,3,4.5};
            lam {.03,.035,.045} x gamma {.8,.9} x c {1,1.5,2};
            up_a {-.1,-.3,-.5} x tier_b {.3,.6,.9} x shrink {0,.15};
            shrink {.15,.3} x loc {gd, gd+cross} x draw band {const, |gd|} x lam_new {0,.04};
            lam {.03,.045} x gamma {.6,.8,.9} x lam_new {.04,.08} x n_new {10,25};
            lam {.025,.03} x loc {gd+cross, four free weights} x n_new {25,50} x down_a {.1,.3}
  chosen    lam .03, gamma .6, c 2, lam_new .04 over 25 matches, shrink .15,
            up_a -.3, down_a .3, tier_b .3, new_a -.3, recentre, gd+cross,
            draw band on |gd|: 0.99362. The surface is flat near it (0.9936-0.9940).
  pooling   chosen a priori (all earlier seasons, alpha free); two variants
            checked on DEV and reported below, neither chosen.

MEASURED ON DEV (~10 s walk-forward, ~50 s with the variants and the
scramble test). Candidate - shipped pick (soccer_model.pick on the same
frame), paired per match, se = sd / sqrt(n):

  POOLED v PICK, 2007-2015, 16,433 matches
              log loss            d (se)              Brier               d (se)              accuracy
  all     0.97846 v 0.97849  -0.00003 (0.00016)  0.58321 v 0.58321  -0.00000 (0.00010)  .5291 v .5286
  E0      0.96712 v 0.96682  +0.00031 (0.00036)  0.57472 v 0.57454  +0.00018 (0.00022)  .5365 v .5368
  SP1     0.94319 v 0.94349  -0.00031 (0.00036)  0.55797 v 0.55818  -0.00021 (0.00022)  .5564 v .5547
  I1      0.97764 v 0.97776  -0.00012 (0.00034)  0.58378 v 0.58385  -0.00007 (0.00021)  .5332 v .5341
  D1      0.99783 v 0.99797  -0.00015 (0.00040)  0.59538 v 0.59536  +0.00003 (0.00025)  .5127 v .5120
  F1      1.01030 v 1.01021  +0.00010 (0.00028)  0.60656 v 0.60651  +0.00005 (0.00018)  .5035 v .5023
  (raw no-vig market on these rows 0.97911)
  pooled v control (alpha 0, same rows): -0.00003 (0.00007) log loss, -0.00003 (0.00005) Brier
  control v pick: -0.00000 (0.00014): the control is the pick, to the noise.

  PI ALONE v PICK, 2006-2015, 18,259 matches (every DEV scored match)
  all     0.99362 v 0.98133  +0.01228 (0.00116)  0.59347 v 0.58522  +0.00825 (0.00076)  .5160 v .5257
  E0      0.97402 v 0.96760  +0.00642 (0.00228)  0.57999 v 0.57526  +0.00474 (0.00147)  .5300 v .5353
  SP1     0.96838 v 0.95155  +0.01683 (0.00271)  0.57480 v 0.56374  +0.01106 (0.00175)  .5418 v .5497
  I1      0.99351 v 0.97492  +0.01859 (0.00288)  0.59494 v 0.58229  +0.01266 (0.00192)  .5170 v .5333
  D1      1.01202 v 1.00298  +0.00904 (0.00281)  0.60397 v 0.59879  +0.00518 (0.00181)  .5013 v .5082
  F1      1.02373 v 1.01383  +0.00991 (0.00221)  0.61569 v 0.60867  +0.00702 (0.00147)  .4871 v .4987
  Pi alone ranks below every model already measured on DEV (shots 0.9878,
  Elo 0.9917, Dixon-Coles 0.9920); with the paper's own settings 0.99800.

  alpha by season (fitted on earlier seasons): 2007 -0.002, 2008 +0.050,
  2009 +0.033, 2010 -0.017, 2011 -0.076, 2012 -0.076, 2013 -0.063, 2014 -0.104,
  2015 -0.098 - from 2010 on the fit BETS AGAINST pi (beta rises 1.05 -> 1.19
  to compensate): the market already over-reacts, if anything, to what a
  recent-goals rating sees.
  in sample (diagnostic, all 18,259 DEV matches, one fit): alpha -0.088, log
  loss gain 0.000078 over alpha 0, LR statistic 2.84 (chi2(1), p ~ 0.09) - the
  most ANY use of pi could have gained on DEV with hindsight, and contrarian.
  variants, pooled v pick on the same 16,433: alpha >= 0 +0.00002 (0.00014)
  (alpha 0 from 2010); last 4 seasons only +0.00003 (0.00020).
  The paper's settings, pooled: +0.00004 (0.00015).

VERDICT. Pi-ratings do not beat the market (+0.0123 log loss) and add nothing
to it: pooled, -0.00003 +- 0.00016 against today's pick (-0.00003 +- 0.00007
against the same-rows control), well inside one standard error, no league
clear of noise, and the weight on pi turns negative as soon as it has data.
Same finding as the shots, Elo and Dixon-Coles models. Do not ship.

CAVEATS. The ratings' settings were chosen on DEV log loss, so pi alone's
0.99362 is mildly optimistic for DEV itself (the grids moved it 0.0044 from
the paper's settings). Pooled numbers start in 2007 (pi needs an out-of-sample
season to fit the pooling on), so they are on 16,433 of DEV's 18,259 matches.
The standard errors treat matches as independent (same-day matches share
news; a clustered se would be a little larger). goals only: pi never sees
shots, cards or line-ups.
"""

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

TOP = list(sd.TOP)
COUNTRY = {code: meta[1] for code, meta in sd.ALL_LEAGUES.items()}
OUT_DIR = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/engine")
PREDS_OUT = OUT_DIR / "pi_ratings.csv.gz"

# Chosen on DEV (2006-2015) only; the grids are in the module notes.
RATING_DEFAULTS = dict(
    lam=0.03,         # learning rate: home team's home / away team's away rating per unit of psi
    gamma=0.6,        # cross-venue learning: the other venue's rating follows by gamma
    c=2.0,            # rating scale: expected goal difference = sign(r) (10^(|r|/c) - 1)
    lam_new=0.04,     # extra learning rate for a team new to its league (decays over n_new matches)
    n_new=25.0,
    h0=0.25,          # warm-up: every club starts at +h0 home, -h0 away
    shrink=0.15,      # each new season: rating -> ref + (1 - shrink) (rating - ref), same league
    up_a=-0.30,       # promoted: rating = ref_new + up_a + tier_b (rating - ref_old)
    down_a=0.30,      # relegated: rating = ref_new + down_a + tier_b (rating - ref_old)
    tier_b=0.3,
    new_a=-0.30,      # a club new to the data (or back after a season away): ref_new + new_a
    recentre=True,    # at each season's end, shift a league's clubs so its mean home + away rating is 0
)
MAP_DEFAULTS = dict(
    loc="gd+cross",   # ordered-logit location: "gd" (expected goal difference),
                      # "gd+cross" (+ the other venues' expected goal difference),
                      # "four" (each of the four ratings' expected goal difference, free weights)
    draw_width="ax",  # "const" or "ax": draw band half-width exp(g0 + g1 |gd|)
    window=0,         # seasons of history for the mapping fit (0 = every earlier season)
    warm_from="2005-10-01",   # the warm-up's first weeks are never fitted on
    ridge=1e-4,
    min_map=500,      # fewer earlier matches than this: no number from pi alone (NaN)
)
POOL_DEFAULTS = dict(
    pool_first=2006,  # pooling weights fitted on pi's out-of-sample seasons from here
    min_pool=1000,    # fewer earlier matches than this: no pooled number (NaN)
    nonneg=False,     # True: pi's weight alpha bounded at 0
    pool_window=0,    # seasons of history for the pooling fit (0 = every earlier one)
)
DEFAULTS = {**RATING_DEFAULTS, **MAP_DEFAULTS, **POOL_DEFAULTS}


# ------------------------------------------------------------- the ratings

def expected_gd(r, c=3.0):
    """A rating -> goals it is expected to win by against an average side."""
    r = np.asarray(r, float)
    return np.sign(r) * (np.power(10.0, np.abs(r) / c) - 1.0)


def _egd(r, c):
    return math.copysign(10.0 ** (abs(r) / c) - 1.0, r)


def rate(matches, **params):
    """Pi-ratings for every row of `matches`, each taken BEFORE that day's
    results enter. One rating pool per league (a club's home and away
    ratings); a club changing league carries its rating across by the tier
    step. Returns a frame indexed like `matches`: r_hh (home club's home
    rating), r_ha (its away rating), r_ah, r_aa (the away club's), gd (the
    expected goal difference f(r_hh) - f(r_aa)), cross (f(r_ha) - f(r_ah)),
    n_h, n_a (matches each club has played in its league), moved_h, moved_a
    (new to the league this season)."""
    p = {**RATING_DEFAULTS, **{k: v for k, v in params.items() if k in RATING_DEFAULTS}}
    lam, gamma, c = p["lam"], p["gamma"], p["c"]
    order = np.argsort(matches["date"].astype(str).to_numpy(), kind="stable")
    dates = matches["date"].astype(str).to_numpy()[order]
    league = matches["league"].to_numpy()[order]
    season = matches["season"].to_numpy(int)[order]
    home = matches["home"].to_numpy()[order]
    away = matches["away"].to_numpy()[order]
    hg = pd.to_numeric(matches["hg"], errors="coerce").to_numpy(float)[order]
    ag = pd.to_numeric(matches["ag"], errors="coerce").to_numpy(float)[order]
    n = len(order)
    out = np.full((n, 6), np.nan)
    nplayed = np.zeros((n, 2))
    moved = np.zeros((n, 2), bool)

    R = {}            # (country, club) -> [home rating, away rating]
    where = {}        # (country, club) -> (league, season) of its last match
    games = {}        # (country, club) -> matches played in its current league
    ref = {}          # (league, season) -> (mean home rating, mean away rating) at season's end
    snap_season = None
    is_new = {}       # (country, club) -> new to its league this season

    def snapshot(s):
        """Each league's mean ratings over its clubs of season s, taken
        before any match of a later season (clubs' ratings as they ended)."""
        acc = {}
        for key, (lg, ss) in where.items():
            if ss == s:
                acc.setdefault(lg, []).append(R[key])
        for lg, rs in acc.items():
            mh, ma = np.mean(np.array(rs), axis=0)
            shift = (mh + ma) / 2 if p["recentre"] else 0.0
            if shift:
                for key, (lg2, ss) in where.items():
                    if ss == s and lg2 == lg:
                        R[key][0] -= shift
                        R[key][1] -= shift
            ref[(lg, s)] = (mh - shift, ma - shift)

    def enter(key, lg, s):
        """A club's first match of season s in league lg: its rating carried
        in. True when it is new to the league (or to the data)."""
        refn = ref.get((lg, s - 1))
        if key not in R:
            R[key] = [p["h0"], -p["h0"]] if refn is None else [refn[0] + p["new_a"], refn[1] + p["new_a"]]
            games[key] = 0
            return refn is not None
        lg0, s0 = where[key]
        if refn is None:
            return False
        if lg0 == lg and s0 == s - 1:
            R[key] = [refn[i] + (1 - p["shrink"]) * (R[key][i] - refn[i]) for i in range(2)]
            return False
        ref0 = ref.get((lg0, s0))
        if s0 != s - 1 or ref0 is None or COUNTRY[lg0] != COUNTRY[lg]:
            R[key] = [refn[0] + p["new_a"], refn[1] + p["new_a"]]
        else:
            up = sd.ALL_LEAGUES[lg].tier < sd.ALL_LEAGUES[lg0].tier
            a = p["up_a"] if up else p["down_a"]
            R[key] = [refn[i] + a + p["tier_b"] * (R[key][i] - ref0[i]) for i in range(2)]
        games[key] = 0
        return True

    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        s_day = int(season[i:j].max())
        if snap_season is None:
            snap_season = int(season[i:j].min())
        while snap_season < s_day:         # a new season begins: freeze last season's means
            snapshot(snap_season)
            snap_season += 1
        for k in range(i, j):
            cty = COUNTRY[league[k]]
            kh, ka = (cty, home[k]), (cty, away[k])
            for side, key in ((0, kh), (1, ka)):
                if where.get(key) != (league[k], int(season[k])):
                    is_new[key] = enter(key, league[k], int(season[k]))
                    where[key] = (league[k], int(season[k]))
                moved[k, side] = is_new[key]
            rh, ra = R[kh], R[ka]
            out[k] = (rh[0], rh[1], ra[0], ra[1], 0.0, 0.0)
            nplayed[k] = (games[kh], games[ka])
        for k in range(i, j):              # the day's results, after every number of the day
            if not (np.isfinite(hg[k]) and np.isfinite(ag[k])):
                continue
            cty = COUNTRY[league[k]]
            kh, ka = (cty, home[k]), (cty, away[k])
            rh, ra = R[kh], R[ka]
            ghat = _egd(rh[0], c) - _egd(ra[1], c)
            err = (hg[k] - ag[k]) - ghat
            psi = c * math.log10(1.0 + abs(err)) * (1.0 if err > 0 else -1.0)
            lh = lam + p["lam_new"] * math.exp(-games[kh] / p["n_new"])
            la = lam + p["lam_new"] * math.exp(-games[ka] / p["n_new"])
            d_h = psi * lh
            rh[0] += d_h
            rh[1] += gamma * d_h
            d_a = -psi * la
            ra[1] += d_a
            ra[0] += gamma * d_a
            games[kh] += 1
            games[ka] += 1
        i = j

    feats = pd.DataFrame(out, columns=["r_hh", "r_ha", "r_ah", "r_aa", "gd", "cross"])
    feats["gd"] = expected_gd(feats["r_hh"], c) - expected_gd(feats["r_aa"], c)
    feats["cross"] = expected_gd(feats["r_ha"], c) - expected_gd(feats["r_ah"], c)
    feats["n_h"], feats["n_a"] = nplayed[:, 0], nplayed[:, 1]
    feats["moved_h"], feats["moved_a"] = moved[:, 0], moved[:, 1]
    back = np.empty(n, int)
    back[order] = np.arange(n)
    feats = feats.iloc[back].reset_index(drop=True)
    feats.index = matches.index
    feats.attrs["ratings"] = {k: tuple(v) for k, v in R.items()}
    feats.attrs["c"] = c
    return feats


# ------------------------------------------------------------- the mapping

def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _design(feats, kind):
    gd = feats["gd"].to_numpy(float)
    if kind == "four":
        c = feats.attrs.get("c", 3.0)
        cols = [expected_gd(feats[k].to_numpy(float), c) for k in ("r_hh", "r_aa", "r_ha", "r_ah")]
    else:
        cols = [gd] if kind == "gd" else [gd, feats["cross"].to_numpy(float)]
    return np.column_stack(cols), np.abs(gd)


def _ordered_probs(X, ax, theta, draw_width):
    k = X.shape[1]
    b = theta[:k]
    m = theta[k]
    g0 = theta[k + 1]
    g1 = theta[k + 2] if draw_width == "ax" else 0.0
    eta = X @ b
    w = np.exp(g0 + g1 * ax)
    p_h = _sig(eta - m - w)
    p_a = _sig(m - w - eta)
    p_d = np.clip(1.0 - p_h - p_a, 1e-12, 1.0)
    return np.column_stack([p_h, p_d, p_a])


def _fit_ordered(X, ax, y, draw_width, ridge, x0=None):
    k = X.shape[1]
    npar = k + 2 + (1 if draw_width == "ax" else 0)
    start = np.zeros(npar) if x0 is None else np.array(x0, float)
    if x0 is None:
        start[0] = 0.8
        start[k] = -0.3
        start[k + 1] = math.log(0.55)

    def nll(theta):
        P = _ordered_probs(X, ax, theta, draw_width)
        return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean() + ridge * float(theta[:k] @ theta[:k])

    fit = minimize(nll, start, method="L-BFGS-B", options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 2000})
    return fit.x


def map_walk_forward(matches, feats, **params):
    """Pi-ratings alone -> home/draw/away: an ordered logit on the expected
    goal difference, refitted each season on the top five's matches of
    earlier seasons only (dated before the season's first match). Returns
    (frame p_h, p_d, p_a indexed like matches - NaN where no fit; {season: theta})."""
    p = {**MAP_DEFAULTS, **{k: v for k, v in params.items() if k in MAP_DEFAULTS}}
    top = matches["league"].isin(TOP).to_numpy()
    res = bt.outcome_index(matches["result"]).to_numpy()
    done = top & np.isfinite(res.astype(float)) & feats["gd"].notna().to_numpy()
    dates = matches["date"].astype(str).to_numpy()
    seasons = matches["season"].to_numpy(int)
    X, ax = _design(feats, p["loc"])
    out = np.full((len(matches), 3), np.nan)
    thetas = {}
    prev = None
    for s in sorted(set(seasons[top])):
        rows = top & (seasons == s)
        first = dates[rows].min()
        train = done & (dates < first) & (dates >= p["warm_from"])
        if p["window"]:
            train &= seasons >= s - p["window"]
        if train.sum() < p["min_map"]:
            continue
        theta = _fit_ordered(X[train], ax[train], res[train].astype(int), p["draw_width"], p["ridge"], prev)
        prev = theta
        thetas[int(s)] = theta
        out[rows] = _ordered_probs(X[rows], ax[rows], theta, p["draw_width"])
    return pd.DataFrame(out, index=matches.index, columns=["p_h", "p_d", "p_a"]), thetas


# ------------------------------------------------------------- with the market

def _log3(p):
    return np.log(np.clip(np.asarray(p, float), 1e-9, 1))


def pooled(lq, lm, w):
    """p_k proportional to q_k^beta * m_k^alpha * exp(c_k); w = (beta, alpha, c_d, c_a)."""
    beta, alpha, c_d, c_a = w
    z = beta * lq + alpha * lm + np.array([0.0, c_d, c_a])
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_pooled(lq, lm, idx, fix_alpha=None, nonneg=False, x0=None):
    """(beta, alpha, c_d, c_a) minimising log loss, analytic gradient and
    tight tolerances (as soccer_model.fit_recalibration). fix_alpha=0 gives
    the shipped recalibration's form, fitted on the same rows."""
    onehot = np.zeros_like(lq)
    onehot[np.arange(len(idx)), idx] = 1.0
    n = len(idx)

    def loss(v):
        beta, alpha, c_d, c_a = v if fix_alpha is None else (v[0], fix_alpha, v[1], v[2])
        z = beta * lq + alpha * lm + np.array([0.0, c_d, c_a])
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        r = onehot - np.exp(logp)
        g = [-(r * lq).sum() / n]
        if fix_alpha is None:
            g.append(-(r * lm).sum() / n)
        gc = -r.sum(axis=0) / n
        g += [gc[1], gc[2]]
        return -logp[np.arange(n), idx].mean(), np.array(g)

    if fix_alpha is None:
        start = np.array([1.0, 0.0, 0.0, 0.0]) if x0 is None else np.array(x0, float)
        bounds = [(None, None), (0.0, None) if nonneg else (None, None), (None, None), (None, None)]
    else:
        start = np.array([1.0, 0.0, 0.0])
        bounds = None
    fit = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 2000})
    v = fit.x
    return tuple(float(x) for x in (v if fix_alpha is None else (v[0], fix_alpha, v[1], v[2])))


def pool_walk_forward(matches, pi, **params):
    """The market recalibrated with pi as a second input: each season's
    (beta, alpha, c_d, c_a) fitted on earlier seasons' top-five matches with
    a result, a full pre-match price and pi's out-of-sample number (seasons
    from pool_first on, dated before the season's first match). Also the
    control: the same fit with alpha fixed at 0, on the same rows.
    Returns (pooled frame, control frame, {season: weights}, {season: control weights})."""
    p = {**POOL_DEFAULTS, **{k: v for k, v in params.items() if k in POOL_DEFAULTS}}
    q = bt.no_vig(matches, "pre")
    priced = q.notna().all(axis=1).to_numpy()
    has_pi = pi[["p_h", "p_d", "p_a"]].notna().all(axis=1).to_numpy()
    top = matches["league"].isin(TOP).to_numpy()
    res = bt.outcome_index(matches["result"]).to_numpy()
    done = top & priced & has_pi & np.isfinite(res.astype(float))
    dates = matches["date"].astype(str).to_numpy()
    seasons = matches["season"].to_numpy(int)
    lq = _log3(q.to_numpy(float))
    lm = _log3(pi[["p_h", "p_d", "p_a"]].to_numpy(float))
    out = np.full((len(matches), 3), np.nan)
    ctl = np.full((len(matches), 3), np.nan)
    weights, cweights = {}, {}
    prev = None
    for s in sorted(set(seasons[top])):
        rows = top & (seasons == s)
        first = dates[rows].min()
        train = done & (dates < first) & (seasons >= p["pool_first"])
        if p["pool_window"]:
            train &= seasons >= s - p["pool_window"]
        if train.sum() < p["min_pool"]:
            continue
        w = fit_pooled(lq[train], lm[train], res[train].astype(int), nonneg=p["nonneg"], x0=prev)
        prev = w
        wc = fit_pooled(lq[train], lm[train], res[train].astype(int), fix_alpha=0.0)
        weights[int(s)], cweights[int(s)] = w, wc
        go = rows & priced & has_pi
        out[go] = pooled(lq[go], lm[go], w)
        goc = rows & priced
        ctl[goc] = pooled(lq[goc], np.zeros_like(lq[goc]), wc)
    cols = ["p_h", "p_d", "p_a"]
    return (pd.DataFrame(out, index=matches.index, columns=cols),
            pd.DataFrame(ctl, index=matches.index, columns=cols), weights, cweights)


def predict_walk_forward(matches, **params):
    """The candidate for every top-five row: p_h, p_d, p_a = the pre-match
    market recalibrated with pi-ratings as a second input (NaN where no full
    pre-match price, or before the pooling has earlier out-of-sample seasons
    to fit on); pi_h, pi_d, pi_a = pi-ratings alone (the ordered logit), NaN
    for second-tier rows. Weights in .attrs."""
    feats = rate(matches, **params)
    pi, thetas = map_walk_forward(matches, feats, **params)
    pool, ctl, weights, cweights = pool_walk_forward(matches, pi, **params)
    top = matches["league"].isin(TOP)
    out = pool.where(top)
    out["pi_h"], out["pi_d"], out["pi_a"] = (pi[c].where(top) for c in ("p_h", "p_d", "p_a"))
    out["gd"] = feats["gd"].where(top)
    out.attrs.update(thetas=thetas, weights=weights, control=ctl, control_weights=cweights)
    return out


# ------------------------------------------------------------- measurement

def _per_match(p, result):
    idx = bt.outcome_index(result).to_numpy().astype(int)
    P = p[["p_h", "p_d", "p_a"]].to_numpy(float)
    ll = -np.log(np.clip(P[np.arange(len(idx)), idx], 1e-12, 1))
    oh = np.zeros_like(P)
    oh[np.arange(len(idx)), idx] = 1
    br = ((P - oh) ** 2).sum(axis=1)
    acc = (P.argmax(axis=1) == idx).astype(float)
    return ll, br, acc


def compare(matches, cand, ref, sel):
    """Candidate vs reference on the rows `sel`: log loss, Brier, accuracy of
    each, the paired log-loss and Brier differences (candidate - reference)
    with standard errors, overall and per league."""
    def block(s):
        m = matches[s]
        llc, brc, acc_c = _per_match(cand[s], m["result"])
        llr, brr, acc_r = _per_match(ref[s], m["result"])
        d, db = llc - llr, brc - brr
        k = len(d)
        return dict(n=k, ll=llc.mean(), ll_ref=llr.mean(), brier=brc.mean(), brier_ref=brr.mean(),
                    acc=acc_c.mean(), acc_ref=acc_r.mean(),
                    d_ll=d.mean(), se_ll=d.std(ddof=1) / math.sqrt(k),
                    d_brier=db.mean(), se_brier=db.std(ddof=1) / math.sqrt(k))
    out = {"all": block(sel)}
    for lg in TOP:
        s = sel & (matches["league"] == lg)
        if s.sum() > 1:
            out[lg] = block(s)
    return out


def _fmt(name, refname, rep):
    lines = [f"{name} vs {refname}"]
    for k, r in rep.items():
        lines.append(f"  {k:>4} n={r['n']:>5}  LL {r['ll']:.5f} vs {r['ll_ref']:.5f}  "
                     f"d {r['d_ll']:+.5f} (se {r['se_ll']:.5f})  Brier {r['brier']:.5f} vs {r['brier_ref']:.5f} "
                     f"d {r['d_brier']:+.5f} (se {r['se_brier']:.5f})  acc {r['acc']:.4f} vs {r['acc_ref']:.4f}")
    return "\n".join(lines)


PRICE_COLS = ("avg_h", "avg_d", "avg_a", "max_h", "max_d", "max_a", "ps_h", "ps_d", "ps_a",
              "b365_h", "b365_d", "b365_a", "avg_o25", "avg_u25", "max_o25", "max_u25",
              "avgc_h", "avgc_d", "avgc_a", "psc_h", "psc_d", "psc_a", "avgc_o25", "avgc_u25")
STAT_COLS = ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar")


def scramble(matches, cut, seed=0):
    """Every result, match statistic and price from `cut` on replaced by noise."""
    later = (matches["date"].astype(str) >= str(cut)).to_numpy()
    df = matches.copy()
    rng = np.random.default_rng(seed)
    k = int(later.sum())
    for c in STAT_COLS:
        if c in df:
            df.loc[later, c] = rng.integers(0, 6, k)
    for c in PRICE_COLS:
        if c in df:
            df[c] = df[c].astype(float)
            df.loc[later, c] = rng.uniform(1.05, 12.0, k)
    df.loc[later, "result"] = np.where(df.loc[later, "hg"] > df.loc[later, "ag"], "H",
                                       np.where(df.loc[later, "hg"] < df.loc[later, "ag"], "A", "D"))
    return df


def point_in_time(matches, cut, **params):
    """Predictions dated before `cut` must not move when results, statistics
    and prices from the cut on are scrambled; pi alone must not move ON the
    cut day either (it reads no price of the day). Returns a dict."""
    a = predict_walk_forward(matches, **params)
    b = predict_walk_forward(scramble(matches, cut), **params)
    d = matches["date"].astype(str)
    before, on = (d < str(cut)).to_numpy(), (d == str(cut)).to_numpy()
    cols = ["p_h", "p_d", "p_a", "pi_h", "pi_d", "pi_a"]
    A, B = a[cols].to_numpy(float), b[cols].to_numpy(float)
    after = (d > str(cut)).to_numpy()
    return dict(cut=str(cut),
                before_rows=int(np.isfinite(A[before, 3]).sum()),
                before_unchanged=bool(np.allclose(A[before], B[before], equal_nan=True, rtol=0, atol=1e-12)),
                on_day_rows=int(np.isfinite(A[on, 3]).sum()),
                on_day_pi_unchanged=bool(np.allclose(A[on, 3:], B[on, 3:], equal_nan=True, rtol=0, atol=1e-12)),
                on_day_pooled_moved=bool(not np.allclose(A[on, :3], B[on, :3], equal_nan=True)),
                later_pi_moved=bool(not np.allclose(A[after, 3:], B[after, 3:], equal_nan=True)))


def dev_report(matches, preds, period=bt.DEV, control=None):
    """DEV ONLY: the candidate (pooled), pi alone and the same-rows control
    (default preds.attrs["control"]), each against today's shipped pick
    (soccer_model.pick) on the same matches."""
    import soccer_model as smod
    pi = preds[["pi_h", "pi_d", "pi_a"]].set_axis(["p_h", "p_d", "p_a"], axis=1)
    pick, pick_w = smod.pick(matches, pi)
    lo, hi = period
    base = bt.scored(matches) & matches["season"].between(lo, hi)
    has = lambda f: f[["p_h", "p_d", "p_a"]].notna().all(axis=1)  # noqa: E731
    sel_pool = base & has(preds) & ~pick["model_only"]
    sel_pi = base & has(pi) & ~pick["model_only"]
    ctl = preds.attrs["control"] if control is None else control
    rep = {"pooled_vs_pick": compare(matches, preds, pick, sel_pool),
           "pi_vs_pick": compare(matches, pi, pick, sel_pi),
           "pooled_vs_control": compare(matches, preds, ctl, sel_pool),
           "control_vs_pick": compare(matches, ctl, pick, sel_pool)}
    mkt = bt.no_vig(matches, "pre")
    rep["market_on_pool_rows"] = bt.log_loss(mkt[sel_pool], matches.loc[sel_pool, "result"])
    rep["market_on_pi_rows"] = bt.log_loss(mkt[sel_pi], matches.loc[sel_pi, "result"])
    return rep, pick_w


def in_sample_information(matches, preds, period=bt.DEV):
    """DIAGNOSTIC, in sample: on every priced period match with pi's
    out-of-sample number, the recalibration fitted once with and without pi
    (alpha free vs alpha 0). The log-likelihood-ratio statistic 2 n (LL0 - LL1)
    is chi-squared(1) if pi adds nothing - an upper bound on what any
    walk-forward use of pi could gain, never a shippable number."""
    lo, hi = period
    sel = (bt.scored(matches) & matches["season"].between(lo, hi)
           & preds[["pi_h", "pi_d", "pi_a"]].notna().all(axis=1)).to_numpy()
    lq = _log3(bt.no_vig(matches, "pre").to_numpy(float)[sel])
    lm = _log3(preds[["pi_h", "pi_d", "pi_a"]].to_numpy(float)[sel])
    idx = bt.outcome_index(matches["result"]).to_numpy()[sel].astype(int)

    def ll(w):
        P = pooled(lq, lm, w)
        return float(-np.log(P[np.arange(len(idx)), idx]).mean())
    w1 = fit_pooled(lq, lm, idx)
    w0 = fit_pooled(lq, lm, idx, fix_alpha=0.0)
    l1, l0 = ll(w1), ll(w0)
    return dict(n=int(sel.sum()), weights=w1, weights_alpha0=w0, ll=l1, ll_alpha0=l0,
                gain=l0 - l1, lr_stat=2 * len(idx) * (l0 - l1))


VARIANTS = {                       # DEV-only checks of the pooling, each reported
    "alpha >= 0": dict(nonneg=True),
    "last 4 seasons": dict(pool_window=4),
}


def main():
    t0 = time.time()
    matches = sd.load_matches()
    preds = predict_walk_forward(matches)
    t_run = time.time() - t0
    print(f"walk-forward, every season: {t_run:.1f}s")
    rep, pick_w = dev_report(matches, preds)
    print(_fmt("POOLED (market + pi)", "pick", rep["pooled_vs_pick"]))
    print(_fmt("PI ALONE", "pick", rep["pi_vs_pick"]))
    print(_fmt("POOLED", "control (alpha=0, same rows)", rep["pooled_vs_control"]))
    print(_fmt("CONTROL", "pick", rep["control_vs_pick"]))
    print(f"raw market on the pooled rows {rep['market_on_pool_rows']:.5f}, on the pi rows {rep['market_on_pi_rows']:.5f}")
    print("pooling weights (beta, alpha, c_draw, c_away), DEV seasons:")
    for s, w in sorted(preds.attrs["weights"].items()):
        if s <= bt.DEV[1]:
            print(f"   {s}: " + ", ".join(f"{v:+.4f}" for v in w))
    info = in_sample_information(matches, preds)
    print(f"in sample, DEV (diagnostic): n={info['n']} alpha={info['weights'][1]:+.4f} "
          f"LL {info['ll']:.5f} vs alpha=0 {info['ll_alpha0']:.5f} gain {info['gain']:.6f} LR {info['lr_stat']:.2f}")
    feats = rate(matches)
    pi, _ = map_walk_forward(matches, feats)
    for name, kw in VARIANTS.items():
        pool, ctl, w, _ = pool_walk_forward(matches, pi, **kw)
        r, _ = dev_report(matches, pool.join(preds[["pi_h", "pi_d", "pi_a"]]), control=ctl)
        a = r["pooled_vs_pick"]["all"]
        print(f"variant {name}: n={a['n']} LL {a['ll']:.5f} vs pick {a['ll_ref']:.5f} d {a['d_ll']:+.5f} (se {a['se_ll']:.5f})"
              f"  alphas " + ", ".join(f"{w[s][1]:+.3f}" for s in sorted(w) if s <= bt.DEV[1]))
    for cut in ("2011-02-05",):
        print("point in time:", point_in_time(matches, cut))
    top = matches["league"].isin(TOP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["p_h", "p_d", "p_a", "pi_h", "pi_d", "pi_a", "gd"]
    out = pd.concat([matches.loc[top, ["match_id", "league", "season", "date"]], preds.loc[top, cols]], axis=1)
    out.to_csv(PREDS_OUT, index=False, compression="gzip")
    print(f"wrote {PREDS_OUT} ({len(out)} rows); total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
