"""Candidate model: a time-decayed Dixon-Coles rating, refitted twice a week.

THE MODEL. Per country, the top tier and the second tier are rated together
(names are stable across tiers, so a promoted team arrives with the matches
it played on the way up). For a match in league l between home team i and
away team j:

    log lambda (home goals) = mu + h_l + att_i - def_j
    log nu     (away goals) = mu       + att_j - def_i

    P(x, y) = tau(x, y) * Pois(x; lambda) * Pois(y; nu)

with the Dixon-Coles low-score correction

    tau(0,0) = 1 - lambda*nu*rho    tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + nu*rho           tau(1,1) = 1 - rho        (else 1)

mu is per country, h_l and rho are fitted per country (h per tier).

FITTING. Weighted maximum likelihood with exponential time decay,
w = 2 ** (-age_days / half_life), and a ridge split into a team's QUALITY
(att + def, what decides 1X2) and its OPENNESS (att - def, what decides the
goal total):

    max  sum_m w_m log P(x_m, y_m)
         - ridge/2      * sum_i (da_i + dd_i)^2
         - ridge_open/2 * sum_i (da_i - dd_i)^2

where da_i, dd_i are the deviations from the prior mean. With
tier_prior=True the prior mean is the average of the team's tier (the tier
of its latest match before the fit; A_2, D_2 free, top tier = 0); with
tier_prior=False it is 0. Solved by damped Newton steps on the exact
Hessian, warm-started from the previous fit (3-5 steps a fit).

POINT IN TIME. Fixtures are predicted in blocks, each from one fit on
matches dated strictly before the block's first day:
  refit="market" (chosen)  Fri-Mon from a fit before Friday, Tue-Thu from a fit
                           before Tuesday - the moments football-data.co.uk
                           collects the pre-match price the app picks against
  refit="week"             Mon-Sun from a fit before Monday
  refit="half"             Mon-Thu before Monday, Fri-Sun before Friday
  refit="day"              every match day from a fit before that day
Nothing but dates, teams, leagues, seasons and the goals of earlier matches
is read: no prices, no match statistics.

DEV CHOICES (2006-2015 log loss; market 0.9816). Grids, in order:
  1. half_life {120,180,270,365,540} x ridge {1,3,10,30} (one ridge, tier
     prior, weekly): best 365 / 1 -> 0.99279; O/U wanted a bigger ridge
  2. half_life {300,400,540} x ridge {0.1,0.3,1} x ridge_open {1,3,10,30}:
     best 300 / 0.3 / 30 -> 0.99230
  3. at that point: ridge_open {100,300} worse; refit half 0.99222; tier
     prior off 0.99228; season_discount {0.8,0.6} worse (0.99283, 0.99429);
     tier2_weight 0.5 worse (0.99264); half_life 220 worse
  4. half_life {250,300,350} x ridge {0.1,0.3,1} x ridge_open {20,40,60} x
     tier_prior {on,off}, refit half: best 350 / 0.1 / 40 / off -> 0.99205
  5. half_life {350,420,500} x ridge {0.03,0.1} x tier_prior; refit
     {market, half, week, day} at the best: market 0.99204, half 0.99205,
     week 0.99213, day 0.99197 (day not taken: it would read Saturday's
     results for Sunday games, which the Friday price the app picks against
     cannot)
The tier prior helps at a strong ridge (ridge 1: 0.9923 vs 0.9927) but
not at the chosen weak one, so it is off.

OUTPUT. p_h, p_d, p_a and p_o25 from the 0-10 x 0-10 score matrix
(renormalised), for top-tier matches from the second season of the data on.
"""

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_data as sd  # noqa: E402

MAX_GOALS = 10
# Chosen on DEV (2006-2015) only - see tune_dev and the grid in the module notes.
DEFAULTS = dict(half_life=350.0, ridge=0.1, ridge_open=40.0, season_discount=1.0, tier2_weight=1.0,
                refit="market", tier_prior=False, window_half_lives=8.0,
                rho_bounds=(-0.25, 0.25))
COUNTRY = {code: meta[1] for code, meta in sd.LEAGUES.items()}
TIER = {code: meta[2] for code, meta in sd.LEAGUES.items()}


# ----------------------------------------------------------------- the fit
# theta = [att (n), def (n), mu, h_top, h_second, rho, A_2, D_2]

def _unpack(theta, n):
    return theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1:2 * n + 3], theta[2 * n + 3], theta[2 * n + 4], theta[2 * n + 5]


def _objective(theta, n, hi, ai, lg, x, y, w, lgam, tt, ridge, tier_prior, masks):
    att, dfn, mu, h, rho, A2, D2 = _unpack(theta, n)
    eh = mu + h[lg] + att[hi] - dfn[ai]
    ea = mu + att[ai] - dfn[hi]
    lam, nu = np.exp(eh), np.exp(ea)
    m00, m01, m10, m11 = masks

    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * nu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + nu[m10] * rho
    tau[m11] = 1.0 - rho
    tau = np.maximum(tau, 1e-10)

    ll = x * eh - lam + y * ea - nu - lgam + np.log(tau)

    gh = x - lam
    ga = y - nu
    grho = np.zeros_like(lam)
    t00 = tau[m00]
    gh[m00] -= lam[m00] * nu[m00] * rho / t00
    ga[m00] -= lam[m00] * nu[m00] * rho / t00
    grho[m00] = -lam[m00] * nu[m00] / t00
    gh[m01] += lam[m01] * rho / tau[m01]
    grho[m01] = lam[m01] / tau[m01]
    ga[m10] += nu[m10] * rho / tau[m10]
    grho[m10] = nu[m10] / tau[m10]
    grho[m11] = -1.0 / tau[m11]
    gh *= w
    ga *= w

    A = np.where(tt == 1, A2, 0.0) if tier_prior else 0.0
    D = np.where(tt == 1, D2, 0.0) if tier_prior else 0.0
    ra, rd = att - A, dfn - D
    rq, ro = ridge
    sq, so = ra + rd, ra - rd                       # quality and openness deviations (x2)
    f = -float(w @ ll) + 0.5 * rq * float(sq @ sq) + 0.5 * ro * float(so @ so) + 1e-3 * (A2 * A2 + D2 * D2)
    pa, pd_ = rq * sq + ro * so, rq * sq - ro * so  # penalty gradient wrt att, def

    g = np.empty_like(theta)
    g[:n] = -(np.bincount(hi, gh, n) + np.bincount(ai, ga, n)) + pa
    g[n:2 * n] = (np.bincount(ai, gh, n) + np.bincount(hi, ga, n)) + pd_
    g[2 * n] = -(gh.sum() + ga.sum())
    g[2 * n + 1:2 * n + 3] = -np.bincount(lg, gh, 2)
    g[2 * n + 3] = -float(w @ grho)
    if tier_prior:
        sel = tt == 1
        g[2 * n + 4] = -pa[sel].sum() + 2e-3 * A2
        g[2 * n + 5] = -pd_[sel].sum() + 2e-3 * D2
    else:
        g[2 * n + 4] = 2e-3 * A2
        g[2 * n + 5] = 2e-3 * D2
    return f, g


def _hessian(theta, n, hi, ai, lg, w, tt, ridge, tier_prior, masks, struct):
    """Exact Hessian of the objective (negative penalised log likelihood)."""
    att, dfn, mu, h, rho, A2, D2 = _unpack(theta, n)
    P = 2 * n + 6
    lam = np.exp(mu + h[lg] + att[hi] - dfn[ai])
    nu = np.exp(mu + att[ai] - dfn[hi])
    m00, m01, m10, m11 = masks
    # second derivatives of the per-match log likelihood in (eh, ea, rho)
    chh, caa = -lam.copy(), -nu.copy()
    cha = np.zeros_like(lam)
    chr_, car, crr = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
    u = lam[m00] * nu[m00] * rho
    q = np.maximum(1 - u, 1e-10) ** 2
    chh[m00] -= u / q
    caa[m00] -= u / q
    cha[m00] = -u / q
    chr_[m00] = -(lam[m00] * nu[m00]) / q
    car[m00] = -(lam[m00] * nu[m00]) / q
    crr[m00] = -(lam[m00] * nu[m00]) ** 2 / q
    v = lam[m01] * rho
    q = np.maximum(1 + v, 1e-10) ** 2
    chh[m01] += v / q
    chr_[m01] = lam[m01] / q
    crr[m01] = -lam[m01] ** 2 / q
    v = nu[m10] * rho
    q = np.maximum(1 + v, 1e-10) ** 2
    caa[m10] += v / q
    car[m10] = nu[m10] / q
    crr[m10] = -nu[m10] ** 2 / q
    crr[m11] = -1.0 / (1 - rho) ** 2

    blocks = (w * chh, w * caa, w * cha, w * cha, w * chr_, w * chr_, w * car, w * car, w * crr)
    vals = np.concatenate([(sgn[None, :] * c[:, None]).ravel() for sgn, c in zip(struct[1], blocks)])
    H = np.bincount(struct[0], vals, P * P).reshape(P, P)
    H = -H
    idx = np.arange(2 * n)
    rq, ro = ridge
    H[idx, idx] += rq + ro
    H[np.arange(n), n + np.arange(n)] += rq - ro
    H[n + np.arange(n), np.arange(n)] += rq - ro
    H[2 * n + 4, 2 * n + 4] += 2e-3
    H[2 * n + 5, 2 * n + 5] += 2e-3
    if tier_prior:
        sel = np.flatnonzero(tt == 1)
        k = len(sel)
        a, b = 2 * n + 4, 2 * n + 5
        H[sel, a] -= rq + ro
        H[a, sel] -= rq + ro
        H[n + sel, b] -= rq + ro
        H[b, n + sel] -= rq + ro
        H[sel, b] -= rq - ro
        H[b, sel] -= rq - ro
        H[n + sel, a] -= rq - ro
        H[a, n + sel] -= rq - ro
        H[a, a] += (rq + ro) * k
        H[b, b] += (rq + ro) * k
        H[a, b] += (rq - ro) * k
        H[b, a] += (rq - ro) * k
    return H


def _structure(hi, ai, lg, n):
    """Index pattern of the Hessian: each match touches 4 parameters through
    log lambda (att_home, def_away, mu, h_league), 3 through log nu
    (att_away, def_home, mu) and rho. Returns (flat P*P indices, signs per block)."""
    M, P, r = len(hi), 2 * n + 6, 2 * n + 3
    Ih = np.column_stack([hi, n + ai, np.full(M, 2 * n), 2 * n + 1 + lg])
    Ia = np.column_stack([ai, n + hi, np.full(M, 2 * n)])
    Sh, Sa = np.array([1.0, -1.0, 1.0, 1.0]), np.array([1.0, -1.0, 1.0])
    R = np.full((M, 1), r)
    pairs = [(Ih, Ih, Sh, Sh), (Ia, Ia, Sa, Sa), (Ih, Ia, Sh, Sa), (Ia, Ih, Sa, Sh),
             (Ih, R, Sh, np.ones(1)), (R, Ih, np.ones(1), Sh), (Ia, R, Sa, np.ones(1)), (R, Ia, np.ones(1), Sa),
             (R, R, np.ones(1), np.ones(1))]
    idx = np.concatenate([(I1[:, :, None] * P + I2[:, None, :]).ravel() for I1, I2, _, _ in pairs])
    signs = [np.outer(S1, S2).ravel() for _, _, S1, S2 in pairs]
    return idx, signs


def fit(hi, ai, lg, x, y, w, tt, theta0, ridge, tier_prior=True, rho_bounds=(-0.25, 0.25),
        tol=1e-8, max_iter=50):
    """One weighted Dixon-Coles fit by damped Newton steps on the exact
    Hessian (a handful of steps from a warm start). hi/ai: local team indices
    (0..n-1), lg: 0 top tier / 1 second tier, tt: each team's prior tier."""
    n = len(tt)
    x, y = x.astype(float), y.astype(float)
    masks = ((x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1))
    lgam = gammaln(x + 1.0) + gammaln(y + 1.0)
    M, P = len(x), 2 * n + 6
    struct = _structure(hi, ai, lg, n)
    args = (n, hi, ai, lg, x, y, w, lgam, tt, ridge, tier_prior, masks)
    theta = theta0.copy()
    theta[2 * n + 3] = np.clip(theta[2 * n + 3], *rho_bounds)
    f, g = _objective(theta, *args)
    for _ in range(max_iter):
        H = _hessian(theta, n, hi, ai, lg, w, tt, ridge, tier_prior, masks, struct)
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(H, g, rcond=None)[0]
        if g @ step >= 0:                       # not a descent direction: fall back to gradient
            step = -g / (np.abs(np.diag(H)).max() + 1.0)
        t = 1.0
        while True:
            cand = theta + t * step
            cand[2 * n + 3] = np.clip(cand[2 * n + 3], *rho_bounds)
            fc, gc = _objective(cand, *args)
            if fc <= f + 1e-4 * t * (g @ step) or t < 1e-6:
                break
            t *= 0.5
        moved = np.abs(cand - theta).max()
        theta, f, g = cand, fc, gc
        if moved < tol:
            break
    return theta


def score_matrix_probs(lam, nu, rho, max_goals=MAX_GOALS):
    """(p_h, p_d, p_a, p_o25) for arrays of expected goals."""
    k = np.arange(max_goals + 1)
    ph = np.exp(k[None, :] * np.log(lam[:, None]) - lam[:, None] - gammaln(k + 1)[None, :])
    pa = np.exp(k[None, :] * np.log(nu[:, None]) - nu[:, None] - gammaln(k + 1)[None, :])
    M = ph[:, :, None] * pa[:, None, :]
    M[:, 0, 0] *= np.maximum(1 - lam * nu * rho, 1e-10)
    M[:, 0, 1] *= np.maximum(1 + lam * rho, 1e-10)
    M[:, 1, 0] *= np.maximum(1 + nu * rho, 1e-10)
    M[:, 1, 1] *= max(1 - rho, 1e-10)
    M /= M.sum(axis=(1, 2), keepdims=True)
    p_h = np.tril(M, -1).sum(axis=(1, 2))
    p_d = np.trace(M, axis1=1, axis2=2)
    p_a = np.triu(M, 1).sum(axis=(1, 2))
    tot = k[:, None] + k[None, :]
    p_u = (M * (tot <= 2)[None]).sum(axis=(1, 2))
    return p_h, p_d, p_a, 1 - p_u


# ----------------------------------------------------------- walk forward

def _ridges(p):
    """(quality, openness) ridge. `ridge` alone = the same on both, which is
    ridge * sum(att^2 + def^2)."""
    r = float(p["ridge"])
    ro = p.get("ridge_open")
    return (r, r if ro is None else float(ro))


def _blocks(days, refit):
    """The cut (an epoch day) for each match: its week's Monday, or, with
    refit='half', the Friday for Fri-Sun matches."""
    weekday = (days + 3) % 7                       # 1970-01-01 was a Thursday; Monday = 0
    monday = days - weekday
    if refit == "half":
        return np.where(weekday >= 4, monday + 4, monday)
    if refit == "market":                          # Fri-Mon from Friday, Tue-Thu from Tuesday
        return np.where(weekday >= 4, monday + 4, np.where(weekday == 0, monday - 3, monday + 1))
    if refit == "day":
        return days
    return monday


def _country_walk(args):
    """Walk forward over one country. Returns (row positions, probs, fit log)."""
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return _country_walk_inner(*args)
    except ImportError:
        return _country_walk_inner(*args)


def _country_walk_inner(df, p, first_season):
    days = df["day"].to_numpy()
    teams = pd.Index(sorted(set(df["home"]) | set(df["away"])))
    H = teams.get_indexer(df["home"])
    Aw = teams.get_indexer(df["away"])
    lg = (df["tier"].to_numpy() == 2).astype(int)
    x = df["hg"].to_numpy(float)
    y = df["ag"].to_numpy(float)
    played = ~(np.isnan(x) | np.isnan(y))
    season = df["season"].to_numpy().astype(int)
    target = (lg == 0) & (df["season"].to_numpy() >= first_season)
    cuts = _blocks(days, p["refit"])
    hl = float(p["half_life"])
    window = hl * float(p["window_half_lives"])

    n_all = len(teams)
    last_tier = np.full(n_all, -1)                 # tier of each team's latest match before the cut
    att_g = np.zeros(n_all)                        # warm starts by team
    def_g = np.zeros(n_all)
    glob = np.array([math.log(1.3), 0.25, 0.25, -0.05, 0.0, 0.0])
    order = np.argsort(days, kind="mergesort")
    seen_ptr = 0

    out_rows, out_p, log = [], [], []
    for cut in np.unique(cuts[target]):
        # advance the "latest tier" bookkeeping to matches strictly before the cut
        while seen_ptr < len(order) and days[order[seen_ptr]] < cut:
            r = order[seen_ptr]
            last_tier[H[r]] = lg[r]
            last_tier[Aw[r]] = lg[r]
            seen_ptr += 1
        tr = played & (days < cut) & (days >= cut - window)
        pred = target & (cuts == cut)
        if not tr.any():
            continue
        active = np.unique(np.concatenate([H[tr], Aw[tr]]))
        loc = np.full(n_all, -1)
        loc[active] = np.arange(len(active))
        tt = np.maximum(last_tier[active], 0)
        n = len(active)
        theta0 = np.concatenate([att_g[active], def_g[active], glob])
        w = np.power(2.0, -(cut - days[tr]) / hl)
        if p["season_discount"] != 1.0:             # extra discount per season boundary crossed
            w = w * np.power(float(p["season_discount"]), season[pred].max() - season[tr])
        if p["tier2_weight"] != 1.0:
            w = np.where(lg[tr] == 1, w * float(p["tier2_weight"]), w)
        theta = fit(loc[H[tr]], loc[Aw[tr]], lg[tr], x[tr], y[tr], w, tt, theta0,
                    _ridges(p), bool(p["tier_prior"]), tuple(p["rho_bounds"]))
        att, dfn, mu, h, rho, A2, D2 = _unpack(theta, n)
        att_g[active], def_g[active] = att, dfn
        glob = theta[2 * n:].copy()
        # teams with no match in the window: average of their tier (or of the
        # tier they are about to play in, if never seen)
        prior_a = np.where(last_tier == 1, A2, 0.0) if p["tier_prior"] else np.zeros(n_all)
        prior_d = np.where(last_tier == 1, D2, 0.0) if p["tier_prior"] else np.zeros(n_all)
        a_all, d_all = prior_a.copy(), prior_d.copy()
        a_all[active], d_all[active] = att, dfn
        ph_, pa_ = H[pred], Aw[pred]
        lam = np.exp(mu + h[0] + a_all[ph_] - d_all[pa_])
        nu = np.exp(mu + a_all[pa_] - d_all[ph_])
        probs = np.column_stack(score_matrix_probs(lam, nu, rho))
        out_rows.append(np.flatnonzero(pred))
        out_p.append(probs)
        log.append((int(cut), n, int(tr.sum()), float(mu), float(h[0]), float(h[1]), float(rho), float(A2), float(D2)))
    rows = df["pos"].to_numpy()[np.concatenate(out_rows)] if out_rows else np.array([], int)
    probs = np.vstack(out_p) if out_p else np.empty((0, 4))
    return rows, probs, log


def predict_walk_forward(matches, return_log=False, workers=None, **params):
    """Walk-forward Dixon-Coles probabilities for every top-tier match from
    the second season in `matches` on. Returns a frame indexed like `matches`
    with p_h, p_d, p_a, p_o25 (NaN for second tiers and the warm-up season)."""
    p = dict(DEFAULTS)
    p.update(params)
    first_season = int(p.get("first_season", int(matches["season"].min()) + 1))
    parts = _prepare(matches)
    jobs = [(g, p, first_season) for _, g in parts]
    if workers is None:
        workers = min(4, len(jobs))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_country_walk, jobs))
    else:
        results = [_country_walk(j) for j in jobs]
    out = np.full((len(matches), 4), np.nan)
    logs = {}
    for (country, _), (rows, probs, log) in zip(parts, results):
        out[rows] = probs
        logs[country] = log
    frame = pd.DataFrame(out, index=matches.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    return (frame, logs) if return_log else frame


# ------------------------------------------------------------- the runner

def _prepare(matches):
    df = matches[["league", "season", "date", "home", "away", "hg", "ag"]].copy()
    df["pos"] = np.arange(len(df))
    df["day"] = pd.to_datetime(df["date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
    df["tier"] = df["league"].map(TIER)
    df["country"] = df["league"].map(COUNTRY)
    df["hg"] = pd.to_numeric(df["hg"], errors="coerce").astype(float)
    df["ag"] = pd.to_numeric(df["ag"], errors="coerce").astype(float)
    df = df[df["country"].notna()]
    return [(c, g.sort_values(["day", "pos"], kind="mergesort")) for c, g in sorted(df.groupby("country"), key=lambda kv: kv[0])]


def tune_dev(matches, grid, workers=4):
    """Score every parameter set in `grid` (a list of dicts) on DEV only.
    Fits run on seasons up to DEV's last; nothing later is read."""
    import soccer_backtest as bt
    dev = matches[matches["season"] <= bt.DEV[1]]
    first_season = int(dev["season"].min()) + 1
    parts = _prepare(dev)
    params = [dict(DEFAULTS, **g) for g in grid]
    jobs = [(g, p, first_season) for p in params for _, g in parts]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_country_walk, jobs, chunksize=1))
    out = []
    for k, g in enumerate(grid):
        arr = np.full((len(dev), 4), np.nan)
        for rows, probs, _ in results[k * len(parts):(k + 1) * len(parts)]:
            arr[rows] = probs
        preds = pd.DataFrame(arr, index=dev.index, columns=["p_h", "p_d", "p_a", "p_o25"])
        r = bt.report(dev, preds, periods=(bt.DEV,))["model"][f"{bt.DEV[0]}-{bt.DEV[1]}"]
        out.append(dict(g, log_loss=r["log_loss"], ou=r["over_2_5"]["log_loss"], mean_draw=r["mean_draw"],
                        brier=r["brier"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    import soccer_backtest as bt
    m = sd.load_matches()
    t0 = time.time()
    preds = predict_walk_forward(m[m["season"] <= 2015])
    print(f"{time.time() - t0:.1f}s")
    print(bt.report(m[m["season"] <= 2015], preds, periods=(bt.DEV,)))
