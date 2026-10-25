"""Candidate soccer model: a goal-based Elo (ClubElo / World Football Elo in
spirit), mapped to home/draw/away by a logit fitted on earlier seasons only,
with over/under 2.5 goals from a Poisson on exponentially weighted scoring
and conceding rates.

RATINGS. One Elo pool per country spanning its top and second tier: a
promoted or relegated team carries its rating, so the gap between the tiers
is set by the teams that move between them. Before a match on day D

    E_home = 1 / (1 + 10^(-(R_home + H_league - R_away) / 400))

and after every match of day D (never before all of day D is predicted)

    R_home += K_home * G * (S - E_home),   R_away -= K_away * G * (S - E_home)
    G = max(1, |goal margin|)^lam          (S = 1 / 0.5 / 0)
    H_league += k_hfa * G * (S - E_home)   (home advantage learns per league)

K is boosted for a team's first n_new matches in a tier it has just entered.
At a team's first match of a new season its rating regresses toward the
end-of-last-season mean of the tier it now plays in:
    R += rho * (M_tier - R),  then  + d_up / d_down if it changed tier.
A team new to the data starts at M_tier + offset, the offset MEASURED as the
mean (end-of-first-season rating - tier mean) of earlier newcomers to that
tier (shrunk to a prior with weight prior_n). The warm-up season 2005-06
starts every top-tier team at 1500 and every second-tier team at 1500 - gap0.

GOALS. Per team, exponentially weighted (half-life hl matches) attack and
defence ratios relative to the league's EW home/away scoring averages,
optionally opponent-adjusted:
    lam_h = L_home * A_home * D_away,   lam_a = L_away * A_away * D_home
    P(over 2.5) = 1 - Poisson(lam_h + lam_a) CDF at 2, then a logistic
    recalibration fitted on earlier seasons.

MAPPING. A multinomial logit (or ordered logit) of H/D/A on
x = (R_home + H_league - R_away)/400, x^2 and the expected total goals,
with league draw intercepts, refitted every season on earlier seasons only
(recency-weighted).

Every number for a match on day D comes from matches on earlier days only.
"""

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_data as sd  # noqa: E402

LEAGUE_INFO = {code: (meta[1], meta[2]) for code, meta in sd.LEAGUES.items()}
TOP = list(sd.TOP)

ELO_DEFAULTS = dict(
    k=20.0,            # base K
    lam=0.5,           # goal-margin exponent in G
    hfa0=70.0,         # starting home advantage (Elo points)
    k_hfa=0.0,         # learning rate of the per-league home advantage
    rho=0.2,           # between-season regression toward the tier mean
    d_up=0.0,          # extra shift for a promoted team
    d_down=0.0,        # extra shift for a relegated team
    gap0=150.0,        # warm-up: second tier starts this far below the top tier
    new_off0=-50.0,    # prior for a newcomer's offset from its tier mean
    prior_n=10.0,      # weight of that prior, in newcomers
    k_new=1.0,         # K multiplier for a team new to its tier ...
    n_new=0,           # ... for its first n_new matches there
    ac=0.0,            # autocorrelation damping (538 style); 0 = off
    warm_k=1.0,        # K multiplier during the warm-up season
    k_early=1.0,       # K multiplier for every team's first n_early matches of a season
    n_early=0,
    returning="regress",  # a team back after a season away: 'regress' or 'new'
    k_ht=0.0,          # learning rate of a team's own home edge (0 = off)
    gmode="max1",      # G = max(1,|m|)^lam ('max1') or ((1+|m|)/2)^lam ('plus1')
    tau=0.0,           # >0: soft target S = 0.5 + 0.5 tanh(m / tau) instead of W/D/L
    # goals
    hl=20.0,           # half-life (matches) of a team's attack/defence ratios
    hl_league=400.0,   # half-life (matches) of the league scoring averages
    adj=1,             # opponent-adjust the attack/defence ratios
    phi=0.6,           # tier change: ratios shrink toward 1 by this factor
)
MAP_DEFAULTS = dict(
    mapping="ordered", # 'ordered' (ordered logit) or 'mnl' (multinomial)
    feats="x,dead|",   # location features | draw-width features
    window=8,          # seasons of history for the mapping fit
    season_hl=4.0,     # recency weight half-life (seasons) inside the window
    ridge=1e-3,
    warm_from="2005-10-01",  # warm-up matches before this are not fitted on
    ou_calib=2,        # P(over 2.5) recalibration: 0 none, 1 logistic, 2 logistic + |x|
)
DEFAULTS = {**ELO_DEFAULTS, **MAP_DEFAULTS}


# ------------------------------------------------------------- data prep

def prepare(matches):
    """Arrays in date order (stable), plus the permutation back."""
    dates = matches["date"].astype(str).to_numpy()
    order = np.argsort(dates, kind="stable")
    d = dates[order]
    league = matches["league"].to_numpy()[order]
    country = np.array([LEAGUE_INFO[l][0] for l in league])
    tier = np.array([LEAGUE_INFO[l][1] for l in league])
    hg = pd.to_numeric(matches["hg"], errors="coerce").to_numpy(float)[order]
    ag = pd.to_numeric(matches["ag"], errors="coerce").to_numpy(float)[order]
    day = pd.to_datetime(pd.Series(d)).to_numpy().astype("datetime64[D]").astype(np.int64)
    bounds = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    return dict(
        n=len(d), order=order, date=d, day=day, bounds=bounds,
        league=league, country=country, tier=tier,
        season=matches["season"].to_numpy()[order].astype(int),
        home=matches["home"].astype(str).to_numpy()[order],
        away=matches["away"].astype(str).to_numpy()[order],
        hg=hg, ag=ag, index=matches.index,
        result=matches["result"].to_numpy()[order],
    )


# ------------------------------------------------------------- the pass

def _stakes(tab, gp, home, away, min_games=5):
    """Whether each team has nothing left to play for, from the league table
    as of the day before: no line (title = 1st, Champions League = 4th,
    Europe = 6th, safety = N-3rd) is still within reach of 3 points per
    game left, in either direction."""
    if len(gp) < 10 or min(gp.values()) < min_games:
        return 0.0, 0.0
    N = len(gp)
    out = []
    for team in (home, away):
        p_t = tab.get(team, 0)
        rem = 2 * (N - 1) - gp.get(team, 0)
        others = sorted((v for k2, v in tab.items() if k2 != team), reverse=True)
        others += [0] * (N - 1 - len(others))
        live = False
        for k in (1, 4, 6, N - 3):
            if abs(p_t - others[k - 1]) <= 3 * max(rem, 0):
                live = True
                break
        out.append(0.0 if live else 1.0)
    return out[0], out[1]


def run_ratings(P, **params):
    """One chronological pass. Returns pre-match features per match (in the
    prepared date order): x (rating difference incl. home advantage, /400),
    lam_h, lam_a, and diagnostics."""
    p = {**ELO_DEFAULTS, **params}
    k, lam, k_hfa, rho = p["k"], p["lam"], p["k_hfa"], p["rho"]
    ac = p["ac"]
    tau = p["tau"]
    plus1 = p["gmode"] == "plus1"
    first_season = int(P["season"].min())
    base = {1: 1500.0, 2: 1500.0 - p["gap0"]}
    alpha = 1.0 - 0.5 ** (1.0 / p["hl"])
    alpha_l = 1.0 - 0.5 ** (1.0 / p["hl_league"])
    phi, adj = p["phi"], p["adj"]

    n = P["n"]
    x = np.full(n, np.nan)
    rh_out = np.full(n, np.nan)
    ra_out = np.full(n, np.nan)
    lam_h = np.full(n, np.nan)
    lam_a = np.full(n, np.nan)
    moved = np.zeros(n, dtype=np.int8)   # 1 if either team is new to its tier this season
    dead_h = np.zeros(n)                 # 1 if the home team has nothing left to play for
    dead_a = np.zeros(n)
    pts = defaultdict(dict)              # (league, season) -> team -> points so far
    gp = defaultdict(dict)               # (league, season) -> team -> games played so far

    R = {}
    last_season, last_tier = {}, {}
    n_tier = {}
    n_season = defaultdict(int)
    H = defaultdict(lambda: p["hfa0"])
    HT = defaultdict(float)             # a team's own home edge on top of the league's
    k_ht = p["k_ht"]
    members = defaultdict(set)          # (country, tier, season) -> teams
    snap = {}                           # (country, season) -> {tier: mean rating}
    entrants = defaultdict(list)        # (country, season) -> [(team, tier)]
    off_sum = {1: 0.0, 2: 0.0}
    off_n = {1: 0.0, 2: 0.0}
    A, D = {}, {}
    Lh = defaultdict(lambda: 1.5)
    La = defaultdict(lambda: 1.1)

    def off_new(t):
        return (off_sum[t] + p["prior_n"] * p["new_off0"]) / (off_n[t] + p["prior_n"])

    def snapshot(c, s):
        means = {}
        for t in (1, 2):
            mem = members.get((c, t, s - 1))
            if mem:
                means[t] = float(np.mean([R[m] for m in mem]))
        # measure last season's newcomers against their tier's mean
        for team, t in entrants.get((c, s - 1), []):
            if t in means and team in R:
                off_sum[t] += R[team] - means[t]
                off_n[t] += 1.0
        snap[(c, s)] = means
        return means

    def start(team, c, s, t):
        if s == first_season:
            return base[t]
        means = snap.get((c, s)) or {}
        return means.get(t, base[t]) + off_new(t)

    def enter_season(team, c, s, t):
        key = team
        if key not in R:
            R[key] = start(team, c, s, t)
            if s != first_season:
                entrants[(c, s)].append((team, t))
            n_tier[key] = 0
            A[key], D[key] = 1.0, 1.0
        else:
            prev_s, prev_t = last_season[key], last_tier[key]
            means = snap.get((c, s)) or {}
            if prev_s == s - 1 or p["returning"] == "regress":
                target = means.get(t)
                if target is not None:
                    R[key] += rho * (target - R[key])
                if t < prev_t:
                    R[key] += p["d_up"]
                elif t > prev_t:
                    R[key] += p["d_down"]
            else:
                R[key] = start(team, c, s, t)
                entrants[(c, s)].append((team, t))
            if t != prev_t or prev_s != s - 1:
                n_tier[key] = 0
                A[key] = 1.0 + phi * (A[key] - 1.0)
                D[key] = 1.0 + phi * (D[key] - 1.0)
        last_season[key] = s
        last_tier[key] = t
        n_season[key] = 0

    def kmult(key):
        m = 1.0
        if n_tier[key] < p["n_new"]:
            m *= p["k_new"]
        if n_season[key] < p["n_early"]:
            m *= p["k_early"]
        return m

    bounds = P["bounds"]
    league, country, tier, season = P["league"], P["country"], P["tier"], P["season"]
    home, away, HG, AG = P["home"], P["away"], P["hg"], P["ag"]
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        # 1) everything known before this day -> features for every match on it
        for i in range(lo, hi):
            c, s, t, l = country[i], season[i], tier[i], league[i]
            if (c, s) not in snap:
                snapshot(c, s)
            hk, ak = (c, home[i]), (c, away[i])
            for key in (hk, ak):
                if last_season.get(key) != s:
                    enter_season(key, c, s, t)
            rh, ra = R[hk], R[ak]
            rh_out[i], ra_out[i] = rh, ra
            x[i] = (rh + H[l] + HT[hk] - ra) / 400.0
            lam_h[i] = Lh[l] * A[hk] * D[ak]
            lam_a[i] = La[l] * A[ak] * D[hk]
            moved[i] = int(n_tier[hk] < 19 or n_tier[ak] < 19)
            if t == 1:
                dead_h[i], dead_a[i] = _stakes(pts[(l, s)], gp[(l, s)], home[i], away[i])
        # 2) then the day's results
        for i in range(lo, hi):
            hgi, agi = HG[i], AG[i]
            if not (hgi == hgi and agi == agi):
                continue
            c, s, t, l = country[i], season[i], tier[i], league[i]
            hk, ak = (c, home[i]), (c, away[i])
            gd = hgi - agi
            if tau > 0:
                S = 0.5 + 0.5 * math.tanh(gd / tau)
            else:
                S = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            dr = R[hk] + H[l] + HT[hk] - R[ak]
            E = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
            if plus1:
                G = ((1.0 + abs(gd)) / 2.0) ** lam
            else:
                G = max(1.0, abs(gd)) ** lam
            if ac and gd != 0:
                wdr = dr if gd > 0 else -dr
                G *= ac / (ac + max(wdr, -0.9 * ac * 1000.0) * 0.001)
            kk = k * (p["warm_k"] if s == first_season else 1.0)
            delta = G * (S - E)
            R[hk] += kk * kmult(hk) * delta
            R[ak] -= kk * kmult(ak) * delta
            H[l] += k_hfa * delta
            if k_ht:
                HT[hk] += k_ht * delta
            # goals: EW attack / defence ratios against the league's averages
            lh, la = Lh[l], La[l]
            if adj:
                ah, dh, aa, da = A[hk], D[hk], A[ak], D[ak]
                A[hk] = (1 - alpha) * ah + alpha * hgi / (lh * da)
                D[ak] = (1 - alpha) * da + alpha * hgi / (lh * ah)
                A[ak] = (1 - alpha) * aa + alpha * agi / (la * dh)
                D[hk] = (1 - alpha) * dh + alpha * agi / (la * aa)
            else:
                A[hk] = (1 - alpha) * A[hk] + alpha * hgi / lh
                D[ak] = (1 - alpha) * D[ak] + alpha * hgi / lh
                A[ak] = (1 - alpha) * A[ak] + alpha * agi / la
                D[hk] = (1 - alpha) * D[hk] + alpha * agi / la
            for key in (hk, ak):
                A[key] = min(max(A[key], 0.25), 4.0)
                D[key] = min(max(D[key], 0.25), 4.0)
            Lh[l] = (1 - alpha_l) * lh + alpha_l * hgi
            La[l] = (1 - alpha_l) * la + alpha_l * agi
            tab, g = pts[(l, s)], gp[(l, s)]
            g[home[i]] = g.get(home[i], 0) + 1
            g[away[i]] = g.get(away[i], 0) + 1
            ph, pa = (3, 0) if gd > 0 else ((1, 1) if gd == 0 else (0, 3))
            tab[home[i]] = tab.get(home[i], 0) + ph
            tab[away[i]] = tab.get(away[i], 0) + pa
            members[(c, t, s)].add(hk)
            members[(c, t, s)].add(ak)
            n_tier[hk] += 1
            n_tier[ak] += 1
            n_season[hk] += 1
            n_season[ak] += 1
    return dict(x=x, rh=rh_out, ra=ra_out, lam_h=lam_h, lam_a=lam_a, moved=moved,
                dead_h=dead_h, dead_a=dead_a,
                hfa=dict(H), off={t: off_new(t) for t in (1, 2)},
                off_n=dict(off_n))


# ------------------------------------------------------------- the mapping

def _columns(P, F, names, sel):
    cols = []
    x = F["x"][sel]
    for name in names:
        if name == "x":
            cols.append(x)
        elif name == "x2":
            cols.append(x * x)
        elif name == "ax":
            cols.append(np.abs(x))
        elif name == "x3":
            cols.append(x ** 3)
        elif name == "xb":
            cols.append(F["xb"][sel])
        elif name == "tot":
            cols.append(np.log(F["lam_h"][sel] + F["lam_a"][sel]) - 1.0)
        elif name == "sup":
            cols.append(np.log(F["lam_h"][sel] / F["lam_a"][sel]))
        elif name == "dead":
            cols.append(F["dead_h"][sel] - F["dead_a"][sel])
        elif name == "dead2":
            cols.append(F["dead_h"][sel] * F["dead_a"][sel])
        elif name == "deadany":
            cols.append(np.maximum(F["dead_h"][sel], F["dead_a"][sel]))
        elif name == "moved":
            cols.append(F["moved"][sel].astype(float))
        elif name == "league":
            lg = P["league"][sel]
            for code in TOP[1:]:
                cols.append((lg == code).astype(float))
        elif name:
            raise ValueError(f"unknown feature {name}")
    return cols


def _design(P, F, feats, sel):
    """feats = 'location features | draw-width features'. For the ordered
    logit the two parts are separate; the multinomial uses their union.
    Returns (X_location, X_width), each with a leading column of ones."""
    loc, _, wid = feats.partition("|")
    ones = [np.ones(int(np.sum(sel)) if sel.dtype == bool else len(sel))]
    Xl = np.column_stack(ones + _columns(P, F, loc.split(","), sel))
    Xw = np.column_stack(ones + _columns(P, F, wid.split(","), sel))
    return Xl, Xw


def _fit_mnl(X, y, w, ridge):
    n, k = X.shape
    Y = np.zeros((n, 3))
    Y[np.arange(n), y] = 1.0
    wsum = w.sum()
    pen = np.full(k, ridge)
    pen[0] = 0.0

    def f(theta):
        W = np.zeros((k, 3))
        W[:, 1:] = theta.reshape(k, 2)
        U = X @ W
        U -= U.max(axis=1, keepdims=True)
        lse = np.log(np.exp(U).sum(axis=1, keepdims=True))
        logp = U - lse
        loss = -(w * logp[np.arange(n), y]).sum() / wsum
        Pm = np.exp(logp)
        g = X.T @ ((Pm - Y) * w[:, None]) / wsum
        loss += 0.5 * (pen[:, None] * W[:, 1:] ** 2).sum()
        g = g[:, 1:] + pen[:, None] * W[:, 1:]
        return loss, g.ravel()

    res = minimize(f, np.zeros(k * 2), jac=True, method="L-BFGS-B")
    W = np.zeros((k, 3))
    W[:, 1:] = res.x.reshape(k, 2)
    return W


def _pred_mnl(X, W):
    U = X @ W
    U -= U.max(axis=1, keepdims=True)
    e = np.exp(U)
    return e / e.sum(axis=1, keepdims=True)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ordered_probs(Xl, Xw, theta):
    """Ordered logit: home - away latent = Xl @ beta (column 0 is the home
    intercept); draw band of half-width exp(Xw @ gamma) around zero."""
    kl = Xl.shape[1]
    loc = Xl @ theta[:kl]
    w = np.exp(Xw @ theta[kl:])
    pa = _sig(-w - loc)
    pad = _sig(w - loc)
    return np.column_stack([1 - pad, pad - pa, pa])


def _fit_ordered(Xl, Xw, y, w, ridge):
    kl, kw = Xl.shape[1], Xw.shape[1]
    wsum = w.sum()
    n = len(y)
    pen = np.full(kl + kw, ridge)
    pen[0] = pen[kl] = 0.0

    def f(theta):
        loc = Xl @ theta[:kl]
        hw = np.exp(Xw @ theta[kl:])
        a = _sig(-hw - loc)          # P(A)
        b = _sig(hw - loc)           # P(A or D)
        pr = np.column_stack([1 - b, b - a, a])
        pr = np.clip(pr, 1e-12, 1)
        py = pr[np.arange(n), y]
        loss = -(w * np.log(py)).sum() / wsum + 0.5 * (pen * theta ** 2).sum()
        # gradient
        da = a * (1 - a)
        db = b * (1 - b)
        # d/dloc: dA/dloc = -da, dB/dloc = -db ; d/dhw: dA/dhw = -da, dB/dhw = db
        dloc = np.where(y == 0, db / py, np.where(y == 1, (-db + da) / py, -da / py))
        dhw = np.where(y == 0, -db / py, np.where(y == 1, (db + da) / py, -da / py))
        gl = -(Xl.T @ (w * dloc)) / wsum
        gw = -(Xw.T @ (w * dhw * hw)) / wsum
        return loss, np.r_[gl, gw] + pen * theta

    th0 = np.zeros(kl + kw)
    th0[0] = 0.2
    if kl > 1:
        th0[1] = 1.0
    th0[kl] = np.log(0.6)
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    return res.x


def map_walk_forward(P, F, **params):
    """H/D/A probabilities (date order). Each season's mapping is fitted on
    top-league matches of earlier seasons only."""
    p = {**MAP_DEFAULTS, **params}
    n = P["n"]
    out = np.full((n, 3), np.nan)
    y_all = pd.Series(P["result"]).map({"H": 0, "D": 1, "A": 2}).to_numpy()
    top = np.isin(P["league"], TOP)
    has_y = ~pd.isna(y_all)
    seasons = P["season"]
    fit_ok = top & has_y & np.isfinite(F["x"]) & (P["date"] >= p["warm_from"])
    first = int(seasons.min())
    coefs = {}
    for s in sorted(set(seasons.tolist())):
        if s <= first:
            continue
        tr = fit_ok & (seasons < s) & (seasons >= s - p["window"])
        te = top & (seasons == s)
        if tr.sum() < 500 or not te.any():
            continue
        wts = 0.5 ** ((s - 1 - seasons[tr]) / p["season_hl"])
        Ltr, Wtr = _design(P, F, p["feats"], tr)
        ytr = y_all[tr].astype(int)
        Lte, Wte = _design(P, F, p["feats"], te)
        if p["mapping"] == "mnl":
            Xtr = np.column_stack([Ltr, Wtr[:, 1:]])
            Xte = np.column_stack([Lte, Wte[:, 1:]])
            W = _fit_mnl(Xtr, ytr, wts, p["ridge"])
            out[te] = _pred_mnl(Xte, W)
        else:
            W = _fit_ordered(Ltr, Wtr, ytr, wts, p["ridge"])
            out[te] = _ordered_probs(Lte, Wte, W)
        coefs[s] = W
    return out, coefs


def _logit(q):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


def over25_walk_forward(P, F, **params):
    """P(over 2.5 goals): Poisson on lam_h + lam_a, then (ou_calib >= 1) a
    logistic recalibration fitted on earlier seasons only; ou_calib = 2 adds
    the Elo mismatch |x| to it."""
    p = {**MAP_DEFAULTS, **params}
    lam = F["lam_h"] + F["lam_a"]
    raw = 1.0 - np.exp(-lam) * (1.0 + lam + lam * lam / 2.0)
    top = np.isin(P["league"], TOP)
    seasons = P["season"]
    first = int(seasons.min())
    if not p["ou_calib"]:
        return np.where(top & (seasons > first), raw, np.nan)
    n = P["n"]
    out = np.full(n, np.nan)
    tot = P["hg"] + P["ag"]
    ok = top & np.isfinite(tot) & np.isfinite(raw) & (P["date"] >= p["warm_from"])
    cols = [np.ones(n), _logit(raw)]
    if p["ou_calib"] == 2:
        cols.append(np.abs(F["x"]))
    Z = np.column_stack(cols)
    for s in sorted(set(seasons.tolist())):
        if s <= first:
            continue
        tr = ok & (seasons < s) & (seasons >= s - p["window"])
        te = top & (seasons == s)
        if tr.sum() < 500 or not te.any():
            continue
        wts = 0.5 ** ((s - 1 - seasons[tr]) / p["season_hl"])
        yy = (tot[tr] > 2.5).astype(float)
        Ztr = Z[tr]

        def f(beta):
            q = np.clip(_sig(Ztr @ beta), 1e-12, 1 - 1e-12)
            loss = -(wts * (yy * np.log(q) + (1 - yy) * np.log(1 - q))).sum() / wts.sum()
            grad = -(Ztr.T @ (wts * (yy - q))) / wts.sum()
            return loss, grad

        b0 = np.zeros(Z.shape[1])
        b0[1] = 1.0
        beta = minimize(f, b0, jac=True, method="L-BFGS-B").x
        out[te] = _sig(Z[te] @ beta)
    return out


# ------------------------------------------------------------- public

def predict_walk_forward(matches, **params):
    """p_h, p_d, p_a, p_o25 for every top-league match after the warm-up
    season, indexed like `matches`; NaN elsewhere."""
    p = {**DEFAULTS, **params}
    P = prepare(matches)
    F = run_ratings(P, **{k: p[k] for k in ELO_DEFAULTS})
    probs, _ = map_walk_forward(P, F, **{k: p[k] for k in MAP_DEFAULTS})
    o25 = over25_walk_forward(P, F, **{k: p[k] for k in MAP_DEFAULTS})
    out = np.full((P["n"], 4), np.nan)
    out[P["order"], :3] = probs
    out[P["order"], 3] = o25
    df = pd.DataFrame(out, index=matches.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    df.loc[df[["p_h", "p_d", "p_a"]].isna().any(axis=1), ["p_h", "p_d", "p_a"]] = np.nan
    return df
