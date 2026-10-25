"""Candidate model: process-based ratings - shots as well as goals.

THE IDEA. Goals are rare and noisy; shots and shots on target come five to
ten times as often. Each team carries an attack and a defence rating (log
scale) updated after every match towards what it PRODUCED, measured by a
goal proxy rather than by goals alone:

    proxy = (w_g goals + w_sot c_sot SOT + w_s c_s shots) / (sum of the w present)

c_sot (goals per shot on target) and c_s (goals per shot) are each league's
running conversion rates from earlier days, so every part is in goals. A
part that is missing (second tiers before 2017, Bundesliga 2005-06) drops
out and the weights renormalise - with no shots, the proxy is goals.

THE RATINGS (per country, top and second tier in one pool). For a match in
league l between home i and away j:

    eta_home = MU + hadv_l + att_i - def_j        (log expected home proxy)
    eta_away = MU          + att_j - def_i

and after the day's matches, with r = proxy - exp(eta) (the Poisson score):

    att_i += k_i r_home    def_j -= k_j r_home    att_j += k_j r_away    def_i -= k_i r_away
    hadv_l += k_hadv (r_home - r_away) / 2
    k_i = k (1 + boost e^(-n_i / boost_tau) + season_boost e^(-g_i / season_tau)) [x red_k if a red card]

n_i = matches since the team entered its tier, g_i = its matches this season:
an exponentially weighted, opponent-adjusted average of the proxy that
learns faster while a team is new to a tier or a season is young, and
slower from a match played 10 v 11.

SEASONS AND TIERS. At a team's first match of a new season, with ref_l the
mean rating of league l's teams last season:
  same league   rating = ref_l + (1 - shrink) (rating - ref_l)
  moved tier    rating = ref_new + A + B (rating - ref_old)   (per att / def, per direction)
  new to data   rating = ref_new + A_new
The tier step (A, B) is MEASURED, point in time: for every team that changed
tier in an earlier season, its realised strength in the new tier is
y = log(sum proxy / sum expected proxy of an average team of that league)
over its first season there, regressed on x = its rating relative to the old
league when it left (ridge, pulled to a prior worth `prior_n` teams; the
prior alone in 2006, when nothing has been measured yet). Measured by 2026
from 281 promotions: promoted att -0.22 + 0.42 x, def -0.20 + 0.44 x.

GOALS. A Poisson regression, refitted every `refit_days` days on top-tier
matches dated strictly before the block, maps ratings to expected goals:

    log lambda = a_l^H + b1 eta_home + b2 eta_away + stakes terms
    log nu     = a_l^A + c1 eta_away + c2 eta_home + stakes terms

The stakes come from the league table on the morning of the match (points
and goal difference from earlier days; the division's roster is a fixture
fact): for each side, how close it is to a boundary it can still cross in
the games left - title / top 4 / top 6 ("up") and the drop ("rel") - as
max(0, 1 - points gap / (3 games left + 1)), and "dead" when no boundary can
be crossed; all scaled by lateness max(0, 1 - games left / late_rounds).
Then P(x, y) = tau_rho(x, y) Pois(x; lambda) Pois(y; nu), Dixon-Coles rho
fitted on the same rows; p_o25 from the same 0-10 x 0-10 matrix.

POINT IN TIME. Every number for a match on day D comes from matches on days
strictly before D: a day's features are taken before any of that day's
results enter the ratings or the table; regressions and the tier step see
only earlier rows. Second tiers are rated, never predicted; the first season
passed (2005-06 on the full file) only warms up, and a refit block with fewer
than `min_train` earlier top-tier matches gets no number (only on a slice of
history). Deterministic (no randomness).

MEASURED (full walk-forward, ~14 s). Raw: DEV 0.9878 vs market 0.9816,
CONFIRM 0.9754 vs 0.9652. The pooled blend beats the market alone by ~0.0004
in both periods, but pooling the market with ITSELF (same form, model weight
0) does better still (CONFIRM 0.9645 vs blend 0.9647): the gain is the
market's own recalibration, and the pooling weight on this model falls to ~0
by 2025. Shots-based ratings carry nothing the Friday price lacks.

CHOSEN ON DEV (2006-2015) ONLY, by grids on DEV log loss:
  proxy     w_sot {0,.3,.5,.65,.8,1} x w_shots {0,.2} x k {.02,.03,.045,.06} (w_g .35);
            then w_g {.25,.35,.45} x w_sot {.2,.3,.4} x w_shots {.1,.2,.3} x k {.017,.023};
            k {.008,.011,.015,.02}
  ratings   boost {.5,1.5,2.5,3} x boost_tau {5,10,20} x shrink {0,.1,.15,.2,.3};
            season_boost {0,.5,1} x season_tau {5,10} x red_k {1,.6};
            k_hadv {.001,.002,.004} x conv_decay {.998,.9995} x prior_n {5,15,50};
            tier step measured / fixed; k_def_mult {.7,1,1.4}
  goals     reg_half_life {730,1460,2920,none} x refit_days {7,14,28} x min_played {5,10,20};
            rho on/off x draw inflation on/off x ridge {1e-3,1}; promoted flags on/off;
            stakes on/off, late_rounds {5,8,12}
  rejected  extra rating channels (goals-only, SOT-only, fast + slow) beside the mix,
            an early-season slope, league-specific slopes.
Ablation on DEV (same pipeline, everything else as chosen): the mix 0.98783;
goals-only proxy 0.99119 (0.9904 with its own k); SOT-only 0.99321; goals +
SOT without shots 0.98806; no stakes terms 0.98948; no rho 0.98883 (draws
0.242 vs 0.260 actual); draw inflation 0.98785; fixed tier step 0.98781.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_data as sd  # noqa: E402

MAX_GOALS = 10
MU = math.log(1.3)
LEAGUES = list(sd.LEAGUES)
L_IDX = {code: i for i, code in enumerate(LEAGUES)}
L_TIER = np.array([sd.LEAGUES[c][2] for c in LEAGUES])
COUNTRIES = sorted({meta[1] for meta in sd.LEAGUES.values()})
L_COUNTRY = np.array([COUNTRIES.index(sd.LEAGUES[c][1]) for c in LEAGUES])

# Chosen on DEV (2006-2015) only; the grids are listed in the module notes.
DEFAULTS = dict(
    # the proxy: goals, shots on target and shots, in goals (shares 0.41 / 0.35 / 0.24)
    w_goals=0.35, w_sot=0.30, w_shots=0.20, conv_decay=0.998,
    # the ratings
    k=0.02, k_hadv=0.002, hadv0=0.25, boost=0.5, boost_tau=20.0, shrink=0.10,
    season_boost=0.5, season_tau=10.0, red_k=0.6, k_def_mult=1.0,
    # the tier step: priors (A_att, A_def, B) and their weight in teams
    prior_up=(-0.2, -0.2, 0.5), prior_down=(0.2, 0.2, 0.5), prior_new=(-0.1, -0.1),
    prior_n=50.0, measure_min=10, tier_step="measured",
    # the goals regression
    reg_half_life=1e9, refit_days=14, min_played=5, promo_feature=0.0,
    ridge=1.0, draw_inflate=False, rho=True, min_train=500,
    # the table: what is still at stake late in the season
    stakes=True, n_rel=3, cl_place=4, eur_place=6, late_rounds=8,
    first_season=2006,
)


# ----------------------------------------------------------- the ratings

def _fit_step(pairs, prior, prior_n, season, has_b=True):
    """Ridge regression y = A + B x pulled to the prior, from pairs measured
    in seasons before `season`, per rating channel. pairs: list of
    (season, x_att[C], y_att[C], x_def[C], y_def[C]). Returns arrays
    A_att, B_att, A_def, B_def of shape (C,)."""
    rows = [p for p in pairs if p[0] < season]
    out = []
    for part in (0, 1):
        a0 = prior[part]
        b0 = prior[2] if has_b else 0.0
        if not rows:
            out += [None, None]
            continue
        x = np.array([r[1 + 2 * part] for r in rows])          # (n, C)
        y = np.array([r[2 + 2 * part] for r in rows])
        A, B = [], []
        for c in range(x.shape[1]):
            if has_b:
                X = np.column_stack([np.ones(len(x)), x[:, c]])
                Pm = prior_n * np.diag([1.0, 0.04])            # B's prior as if x had sd 0.2
                th = np.linalg.solve(X.T @ X + Pm, X.T @ y[:, c] + Pm @ np.array([a0, b0]))
                A.append(th[0])
                B.append(th[1])
            else:
                A.append((y[:, c].sum() + prior_n * a0) / (len(y) + prior_n))
                B.append(0.0)
        out += [np.array(A), np.array(B)]
    return out


def channels_of(P):
    """Rating channels: (w_goals, w_sot, w_shots, k) each. One channel from
    the flat parameters unless `channels` is given."""
    ch = P.get("channels") or [(P["w_goals"], P["w_sot"], P["w_shots"], P["k"])]
    return np.array(ch, dtype=float)


def rate(matches, **params):
    """One chronological pass. Returns a frame indexed like `matches` with the
    point-in-time features of every match (taken before its day's results):
    eta_h_c / eta_a_c per rating channel c, and bookkeeping columns."""
    P = {**DEFAULTS, **params}
    m = matches
    order = np.lexsort((np.arange(len(m)), m["date"].to_numpy()))
    lg = m["league"].map(L_IDX).to_numpy()[order]
    season = m["season"].to_numpy(dtype=float).astype(int)[order]
    dates = m["date"].to_numpy()[order]
    country = L_COUNTRY[lg]
    keys = pd.Series(list(zip(country, m["home"].to_numpy()[order])) + list(zip(country, m["away"].to_numpy()[order])))
    codes, uniq = pd.factorize(keys)
    N = len(m)
    hi, ai = codes[:N], codes[N:]
    T = len(uniq)
    L = len(LEAGUES)

    def col(c):
        return pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)[order]
    hg, ag, hst, ast, hs, as_ = (col(c) for c in ("hg", "ag", "hst", "ast", "hs", "as"))
    red = np.nan_to_num(col("hr")) + np.nan_to_num(col("ar"))
    has_sot = ~np.isnan(hst) & ~np.isnan(ast)
    has_sh = ~np.isnan(hs) & ~np.isnan(as_)

    CH = channels_of(P)
    C = len(CH)
    W = CH[:, :3]                                  # (C, 3) proxy weights
    kvec = CH[:, 3]
    k_hadv, boost, btau, shrink = P["k_hadv"], P["boost"], P["boost_tau"], P["shrink"]
    sboost, stau, red_k = P["season_boost"], P["season_tau"], P["red_k"]
    kd = P["k_def_mult"]
    dconv = P["conv_decay"]

    att = np.zeros((T, C))
    dfn = np.zeros((T, C))
    n_tier = np.zeros(T)
    t_league = np.full(T, -1)
    t_season = np.full(T, -1)
    hadv = np.full((L, C), P["hadv0"])
    # running conversion rates per league, seeded with a weak prior
    sg_sot = np.full(L, 0.30 * 50)
    s_sot = np.full(L, 50.0)
    sg_sh = np.full(L, 0.11 * 50)
    s_sh = np.full(L, 50.0)

    members = {}                       # (league, season) -> list of teams
    # the league table, top tiers only: rosters are fixture facts (who is in
    # the division), points and goal difference come from earlier days only
    roster = {}
    for key, grp in pd.DataFrame({"l": lg, "s": season, "h": hi, "a": ai}).groupby(["l", "s"]):
        if L_TIER[key[0]] == 1:
            roster[key] = np.unique(np.r_[grp["h"].to_numpy(), grp["a"].to_numpy()])
    pts = np.zeros(T)
    gdf = np.zeros(T)
    gp = np.zeros(T)
    ref_att = np.zeros((L, C))
    ref_def = np.zeros((L, C))
    has_ref = np.zeros(L, dtype=bool)
    country_season = np.full(len(COUNTRIES), -1)
    global_season = -1
    pending = {}                       # team -> measurement in progress
    pairs = {"up": [], "down": [], "new": []}
    steps = {}
    step_log = {}

    f_eh = np.full((N, C), np.nan)
    f_ea = np.full((N, C), np.nan)
    f_nh = np.zeros(N)
    f_na = np.zeros(N)
    f_ph = np.zeros(N)                 # 1 while the team is in its first season after promotion
    f_pa = np.zeros(N)
    f_play_h = np.zeros(N)             # matches seen in the data (warm-up filter)
    f_play_a = np.zeros(N)
    f_dead_h, f_dead_a, f_up_h, f_up_a, f_rel_h, f_rel_a = (np.zeros(N) for _ in range(6))
    f_rem_h, f_rem_a = np.full(N, np.nan), np.full(N, np.nan)
    f_gp_h, f_gp_a = np.zeros(N), np.zeros(N)
    played = np.zeros(T)
    promoted_season = np.full(T, -1)

    def finalize(before):
        for t in [t for t, p in pending.items() if p["season"] < before]:
            p = pending.pop(t)
            ok = (p["n"] >= P["measure_min"] and np.all(p["sEf"] > 0) and np.all(p["sEa"] > 0)
                  and np.all(p["sPf"] > 0) and np.all(p["sPa"] > 0))
            if ok:
                y_att = np.log(p["sPf"] / p["sEf"])
                y_def = -np.log(p["sPa"] / p["sEa"])
                pairs[p["dir"]].append((p["season"], p["x_att"], y_att, p["x_def"], y_def))

    def start_season(t, l, s):
        prev_l, prev_s = t_league[t], t_season[t]
        if prev_l >= 0 and prev_s == s - 1 and prev_l == l:
            att[t] = ref_att[l] + (1 - shrink) * (att[t] - ref_att[l])
            dfn[t] = ref_def[l] + (1 - shrink) * (dfn[t] - ref_def[l])
        else:
            if prev_l >= 0 and prev_s == s - 1 and L_TIER[prev_l] != L_TIER[l]:
                d = "up" if L_TIER[l] < L_TIER[prev_l] else "down"
                x_att, x_def = att[t] - ref_att[prev_l], dfn[t] - ref_def[prev_l]
            else:                                 # new to the data, or back from below
                d, x_att, x_def = "new", np.zeros(C), np.zeros(C)
            A_a, B_a, A_d, B_d = steps[d]
            att[t] = ref_att[l] + A_a + B_a * x_att
            dfn[t] = ref_def[l] + A_d + B_d * x_def
            n_tier[t] = 0
            if d == "up":
                promoted_season[t] = s
            if has_ref[l]:                        # nothing to measure against in the first season
                pending[t] = dict(dir=d, season=s, league=l, x_att=x_att.copy(), x_def=x_def.copy(),
                                  sPf=np.zeros(C), sEf=np.zeros(C), sPa=np.zeros(C), sEa=np.zeros(C), n=0)
        t_league[t] = l
        t_season[t] = s
        pts[t] = gdf[t] = gp[t] = 0.0
        members.setdefault((l, s), []).append(t)

    def new_steps(s):
        out = {}
        pn = P["prior_new"]
        for d, prior, has_b in (("up", P["prior_up"], True), ("down", P["prior_down"], True),
                                ("new", (pn[0], pn[1], 0.0), False)):
            fit = _fit_step(pairs[d], prior, P["prior_n"], s, has_b)
            if P["tier_step"] != "measured" or fit[0] is None:
                fit[0], fit[1] = np.full(C, prior[0]), np.full(C, prior[2] if has_b else 0.0)
            if P["tier_step"] != "measured" or fit[2] is None:
                fit[2], fit[3] = np.full(C, prior[1]), np.full(C, prior[2] if has_b else 0.0)
            out[d] = fit
        return out

    day_starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    day_ends = np.r_[day_starts[1:], N]
    for d0, d1 in zip(day_starts, day_ends):
        # -- season boundaries (fixture facts only: who plays where)
        for i in range(d0, d1):
            s, l, c = season[i], lg[i], country[i]
            if s > global_season:
                finalize(s)
                steps = new_steps(s)
                step_log[int(s)] = {d: [np.round(v, 3).tolist() for v in steps[d]] for d in steps}
                step_log[int(s)]["n_pairs"] = {d: len(pairs[d]) for d in pairs}
                global_season = s
            if s > country_season[c]:
                for ll in np.flatnonzero(L_COUNTRY == c):
                    mem = members.get((ll, s - 1))
                    if mem:
                        ref_att[ll] = att[mem].mean(axis=0)
                        ref_def[ll] = dfn[mem].mean(axis=0)
                        has_ref[ll] = True
                country_season[c] = s
            for t in (hi[i], ai[i]):
                if t_season[t] != s:
                    start_season(t, l, s)
        # -- the table before today, for the top leagues playing today
        stake = {}
        for key in {(lg[i], season[i]) for i in range(d0, d1) if L_TIER[lg[i]] == 1}:
            teams = roster[key]
            n = len(teams)
            total = 2 * (n - 1)
            order_t = np.lexsort((-gdf[teams], -pts[teams]))
            ranked = pts[teams][order_t]
            rem = np.maximum(total - gp[teams], 0)
            rmax = rem.max()
            late = np.clip(1.0 - rem / P["late_rounds"], 0.0, 1.0)
            pos = np.empty(n, dtype=int)
            pos[order_t] = np.arange(n)
            live = np.zeros((n, 4))
            for j, k_ in enumerate((1, P["cl_place"], P["eur_place"], n - P["n_rel"])):
                # closeness to the boundary between places k_ and k_+1: 1 on it,
                # 0 when it can no longer be crossed in the games left
                above = pos < k_
                other = np.where(above, ranked[k_], ranked[k_ - 1])
                margin = np.abs(pts[teams] - other)
                live[:, j] = np.clip(1.0 - margin / (3.0 * rem + 1.0), 0.0, 1.0)
            c_up = live[:, :3].max(axis=1) * late
            c_rel = live[:, 3] * late
            dead = ((live.max(axis=1) == 0) * late).astype(float)
            for t, a1, a2, a3, r_ in zip(teams, c_up, c_rel, dead, rem):
                stake[t] = (a1, a2, a3, float(r_))
        # -- features, before any of today's results
        for i in range(d0, d1):
            h, a, l = hi[i], ai[i], lg[i]
            f_eh[i] = MU + hadv[l] + att[h] - dfn[a]
            f_ea[i] = MU + att[a] - dfn[h]
            f_nh[i], f_na[i] = n_tier[h], n_tier[a]
            f_ph[i] = float(promoted_season[h] == season[i])
            f_pa[i] = float(promoted_season[a] == season[i])
            f_play_h[i], f_play_a[i] = played[h], played[a]
            f_gp_h[i], f_gp_a[i] = gp[h], gp[a]
            if L_TIER[l] == 1:
                f_up_h[i], f_rel_h[i], f_dead_h[i], f_rem_h[i] = stake[h]
                f_up_a[i], f_rel_a[i], f_dead_a[i], f_rem_a[i] = stake[a]
        # -- today's results into the ratings
        for i in range(d0, d1):
            if hg[i] != hg[i] or ag[i] != ag[i]:
                continue
            h, a, l = hi[i], ai[i], lg[i]
            avail = np.array([1.0, float(has_sot[i]), float(has_sh[i])])
            c_sot = sg_sot[l] / s_sot[l]
            c_sh = sg_sh[l] / s_sh[l]
            comp_h = np.array([hg[i], c_sot * hst[i] if has_sot[i] else 0.0, c_sh * hs[i] if has_sh[i] else 0.0])
            comp_a = np.array([ag[i], c_sot * ast[i] if has_sot[i] else 0.0, c_sh * as_[i] if has_sh[i] else 0.0])
            den = W @ avail
            nothing = den <= 0                      # a channel with none of its parts: goals
            den = np.where(nothing, 1.0, den)
            px_h = np.where(nothing, hg[i], (W @ comp_h) / den)
            px_a = np.where(nothing, ag[i], (W @ comp_a) / den)
            r_h = px_h - np.exp(f_eh[i])
            r_a = px_a - np.exp(f_ea[i])
            kh = kvec * (1.0 + boost * math.exp(-n_tier[h] / btau) + sboost * math.exp(-gp[h] / stau))
            ka = kvec * (1.0 + boost * math.exp(-n_tier[a] / btau) + sboost * math.exp(-gp[a] / stau))
            if red[i] > 0:                           # 10 v 11 says less about either side
                kh = kh * red_k
                ka = ka * red_k
            # measurement of teams in their first season in a new tier
            for t, o, home in ((h, a, True), (a, h, False)):
                p = pending.get(t)
                if p is not None and p["season"] == season[i] and p["league"] == l:
                    ha = hadv[l]
                    if home:
                        p["sPf"] += px_h
                        p["sEf"] += np.exp(MU + ha + ref_att[l] - dfn[o])
                        p["sPa"] += px_a
                        p["sEa"] += np.exp(MU + att[o] - ref_def[l])
                    else:
                        p["sPf"] += px_a
                        p["sEf"] += np.exp(MU + ref_att[l] - dfn[o])
                        p["sPa"] += px_h
                        p["sEa"] += np.exp(MU + ha + att[o] - ref_def[l])
                    p["n"] += 1
            att[h] += kh * r_h
            dfn[a] -= kd * ka * r_h
            att[a] += ka * r_a
            dfn[h] -= kd * kh * r_a
            hadv[l] += k_hadv * (r_h - r_a) * 0.5
            n_tier[h] += 1
            n_tier[a] += 1
            played[h] += 1
            played[a] += 1
            pts[h] += 3.0 if hg[i] > ag[i] else (1.0 if hg[i] == ag[i] else 0.0)
            pts[a] += 3.0 if ag[i] > hg[i] else (1.0 if hg[i] == ag[i] else 0.0)
            gdf[h] += hg[i] - ag[i]
            gdf[a] += ag[i] - hg[i]
            gp[h] += 1
            gp[a] += 1
            if has_sot[i]:
                sg_sot[l] = dconv * sg_sot[l] + hg[i] + ag[i]
                s_sot[l] = dconv * s_sot[l] + hst[i] + ast[i]
            if has_sh[i]:
                sg_sh[l] = dconv * sg_sh[l] + hg[i] + ag[i]
                s_sh[l] = dconv * s_sh[l] + hs[i] + as_[i]

    data = {"n_h": f_nh, "n_a": f_na, "prom_h": f_ph, "prom_a": f_pa, "played_h": f_play_h, "played_a": f_play_a,
            "dead_h": f_dead_h, "dead_a": f_dead_a, "up_h": f_up_h, "up_a": f_up_a, "rel_h": f_rel_h, "rel_a": f_rel_a,
            "rem_h": f_rem_h, "rem_a": f_rem_a, "gp_h": f_gp_h, "gp_a": f_gp_a}
    for c in range(C):
        data[f"eta_h_{c}"] = f_eh[:, c]
        data[f"eta_a_{c}"] = f_ea[:, c]
    feats = pd.DataFrame(data)
    out = pd.DataFrame(index=m.index[order], data=feats.to_numpy(), columns=feats.columns).loc[m.index]
    out.attrs["steps"] = step_log
    out.attrs["channels"] = C
    # Each club's rating after every result in `matches`, for display only:
    # the goal difference a match it would expect against an average club of
    # its own league on a neutral pitch (channel 0).
    ratings = {}
    for t in range(T):
        l = t_league[t]
        if l < 0 or not has_ref[l]:
            continue
        egd = (math.exp(MU + att[t, 0] - ref_def[l, 0]) - math.exp(MU + ref_att[l, 0] - dfn[t, 0]))
        ratings[uniq[t][1]] = {"league": LEAGUES[l], "egd": float(egd)}
    out.attrs["ratings"] = ratings
    return out


# ----------------------------------------------------------- the goals

def _glm(X, y, w, ridge, iters=30):
    """Weighted Poisson regression (log link) by Newton's method; no penalty
    on the first column (the intercept)."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(np.average(y, weights=w), 1e-3))
    pen = np.full(X.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(iters):
        mu = np.exp(X @ beta)
        g = X.T @ (w * (y - mu)) - pen * beta
        H = (X * (w * mu)[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _design(feats, league, home, promo=False, stakes=True):
    """Columns: 1, top-league dummies (E0 is the base), own eta and opposing
    eta per rating channel [, promoted flags fading over 19 matches]
    [, the stakes: dead / chasing a place up the table / near the drop, own and opposing]."""
    n = len(feats)
    cols = [np.ones(n)]
    for code in sd.TOP[1:]:
        cols.append((league == code).astype(float))
    own, opp = ("eta_h", "eta_a") if home else ("eta_a", "eta_h")
    for c in range(feats.attrs.get("channels", 1)):
        cols += [feats[f"{own}_{c}"].to_numpy(), feats[f"{opp}_{c}"].to_numpy()]
    o, p = ("h", "a") if home else ("a", "h")
    if promo:
        cols += [feats[f"prom_{o}"].to_numpy() * np.exp(-feats[f"n_{o}"].to_numpy() / 19.0),
                 feats[f"prom_{p}"].to_numpy() * np.exp(-feats[f"n_{p}"].to_numpy() / 19.0)]
    if stakes:
        for name in ("dead", "up", "rel"):
            cols += [feats[f"{name}_{o}"].to_numpy(), feats[f"{name}_{p}"].to_numpy()]
    return np.column_stack(cols)


def _tau(lam, nu, rho):
    """Dixon-Coles corrections for 0-0, 0-1, 1-0, 1-1 (rows x 4)."""
    return np.column_stack([1 - lam * nu * rho, 1 + lam * rho, 1 + nu * rho, np.full(len(lam), 1 - rho)])


def _fit_rho(lam, nu, x, y, w):
    low = (x <= 1) & (y <= 1)
    lam, nu, x, y, w = lam[low], nu[low], x[low], y[low], w[low]
    cell = (x * 2 + y).astype(int)                    # 0:0-0 1:0-1 2:1-0 3:1-1
    hi = float(min(0.99, 1.0 / np.max(lam * nu)))

    def nll(rho):
        t = _tau(lam, nu, rho)[np.arange(len(lam)), cell]
        return -float(w @ np.log(np.maximum(t, 1e-12)))
    return float(minimize_scalar(nll, bounds=(-0.5, hi - 1e-6), method="bounded").x)


def _pois(lam):
    out = np.empty((len(lam), MAX_GOALS + 1))
    out[:, 0] = np.exp(-lam)
    for g in range(1, MAX_GOALS + 1):
        out[:, g] = out[:, g - 1] * lam / g
    return out


def score_probs(lam, nu, rho):
    """p_h, p_d, p_a, p_o25 from the truncated (0-10 each), Dixon-Coles
    corrected, renormalised score matrix."""
    lam, nu = np.asarray(lam, float), np.asarray(nu, float)
    ph, pa = _pois(lam), _pois(nu)
    t = _tau(lam, nu, rho) - 1.0                      # the four corrections
    c00, c01 = ph[:, 0] * pa[:, 0] * t[:, 0], ph[:, 0] * pa[:, 1] * t[:, 1]
    c10, c11 = ph[:, 1] * pa[:, 0] * t[:, 2], ph[:, 1] * pa[:, 1] * t[:, 3]
    cdf_a = np.cumsum(pa, axis=1)
    cdf_h = np.cumsum(ph, axis=1)
    p_h = (ph[:, 1:] * cdf_a[:, :-1]).sum(axis=1) + c10
    p_a = (pa[:, 1:] * cdf_h[:, :-1]).sum(axis=1) + c01
    p_d = (ph * pa).sum(axis=1) + c00 + c11
    under = (ph[:, 0] * (pa[:, 0] + pa[:, 1] + pa[:, 2]) + ph[:, 1] * (pa[:, 0] + pa[:, 1])
             + ph[:, 2] * pa[:, 0] + c00 + c01 + c10 + c11)
    tot = p_h + p_d + p_a
    return p_h / tot, p_d / tot, p_a / tot, 1.0 - under / tot


def _inflate(p, delta):
    q = p.copy()
    q[:, 1] *= math.exp(delta)
    return q / q.sum(axis=1, keepdims=True)


def _fit_delta(p, res, w):
    lp = np.log(p[np.arange(len(p)), res])
    isd = (res == 1).astype(float)
    pdr = p[:, 1]

    def nll(dl):
        return -float(w @ (lp + dl * isd - np.log1p(pdr * (math.exp(dl) - 1.0))))
    return float(minimize_scalar(nll, bounds=(-0.5, 0.5), method="bounded").x)


def calibrate(matches, feats, **params):
    """Walk-forward: Poisson regressions + draw terms refitted every
    refit_days on earlier top-tier matches; probabilities for the top tiers."""
    P = {**DEFAULTS, **params}
    m = matches
    top = m["league"].isin(sd.TOP).to_numpy()
    seas = m["season"].to_numpy(dtype=float)
    day = (pd.to_datetime(m["date"]) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    hg = pd.to_numeric(m["hg"], errors="coerce").to_numpy(dtype=float)
    ag = pd.to_numeric(m["ag"], errors="coerce").to_numpy(dtype=float)
    has = ~np.isnan(hg) & ~np.isnan(ag)
    warm = (feats["played_h"].to_numpy() >= P["min_played"]) & (feats["played_a"].to_numpy() >= P["min_played"])
    league = m["league"].to_numpy()
    promo = P["promo_feature"] > 0
    Xh = _design(feats, league, True, promo, P["stakes"])
    Xa = _design(feats, league, False, promo, P["stakes"])
    res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

    # the first season on file only warms up, whatever slice of history is passed
    first = max(P["first_season"], int(np.nanmin(seas)) + 1) if len(seas) else P["first_season"]
    target = top & (seas >= first)
    out = np.full((len(m), 4), np.nan)
    tdays = np.unique(day[target])
    fits = {}
    if len(tdays) == 0:
        return pd.DataFrame(out, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    start = tdays[0]
    while start <= tdays[-1]:
        end = start + P["refit_days"]
        block = target & (day >= start) & (day < end)
        if block.any():
            train = top & has & warm & (day < start)
            if train.sum() < P["min_train"]:          # nothing measured yet: no number
                nxt = tdays[tdays >= end]
                if len(nxt) == 0:
                    break
                start = nxt[0]
                continue
            w = 0.5 ** ((start - day[train]) / P["reg_half_life"])
            bh = _glm(Xh[train], hg[train], w, P["ridge"])
            ba = _glm(Xa[train], ag[train], w, P["ridge"])
            lam_t, nu_t = np.exp(Xh[train] @ bh), np.exp(Xa[train] @ ba)
            rho = _fit_rho(lam_t, nu_t, hg[train], ag[train], w) if P["rho"] else 0.0
            delta = 0.0
            if P["draw_inflate"]:
                pt = np.column_stack(score_probs(lam_t, nu_t, rho)[:3])
                delta = _fit_delta(pt, res[train], w)
            lam = np.clip(np.exp(Xh[block] @ bh), 0.05, 8.0)
            nu = np.clip(np.exp(Xa[block] @ ba), 0.05, 8.0)
            p_h, p_d, p_a, p_o = score_probs(lam, nu, rho)
            p3 = _inflate(np.column_stack([p_h, p_d, p_a]), delta)
            out[block, :3] = p3
            out[block, 3] = p_o
            fits[int(start)] = dict(bh=bh, ba=ba, rho=rho, delta=delta, n=int(train.sum()))
        nxt = tdays[tdays >= end]
        if len(nxt) == 0:
            break
        start = nxt[0]
    preds = pd.DataFrame(out, index=m.index, columns=["p_h", "p_d", "p_a", "p_o25"])
    preds.attrs["fits"] = fits
    return preds


def predict_walk_forward(matches, **params):
    feats = rate(matches, **params)
    preds = calibrate(matches, feats, **params)
    preds.attrs["steps"] = feats.attrs.get("steps")
    preds.attrs["ratings"] = feats.attrs.get("ratings")
    return preds


# ----------------------------------------------------------- the run

PREDS_OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/soccer_preds/shots_ratings.csv.gz")


def _pit(matches, cut):
    """bt.check_point_in_time, plus the stronger check that NO prediction
    dated on or before the cut moves when results from the cut on are scrambled."""
    import soccer_backtest as bt
    seen = []

    def predict(df):
        out = predict_walk_forward(df)
        seen.append(out)
        return out
    ok = bt.check_point_in_time(predict, matches, cut)
    upto = matches["date"] <= cut
    a, b = (x.loc[upto, ["p_h", "p_d", "p_a", "p_o25"]].to_numpy(float) for x in seen)
    return ok, bool(np.allclose(a, b, equal_nan=True)), int(np.isfinite(a[:, 0]).sum())


def main():
    import json
    import soccer_backtest as bt

    matches = sd.load_matches()
    t0 = time.time()
    preds = predict_walk_forward(matches)
    runtime = time.time() - t0
    print(f"walk-forward 2005-2026: {runtime:.1f}s")

    rep = bt.report(matches, preds, "shots_ratings")
    print(json.dumps(rep, indent=1))
    blend, weights = bt.blend_walk_forward(matches, preds)
    print(json.dumps(bt.report(matches, blend, "blend"), indent=1))
    print("pooling weights (a model, b market, c_draw, c_away):")
    for season in sorted(weights)[-6:]:
        print(f"   {season}: " + ", ".join(f"{v:.3f}" for v in weights[season]))
    for lo, hi in (bt.DEV, bt.CONFIRM):
        per = matches["season"].between(lo, hi)
        for name, probs in (("raw", preds), ("blend", blend)):
            for e in (0.02, 0.05):
                r = bt.flat_roi(matches[per], probs[per], "best", e)
                print(f"   ROI {lo}-{hi} {name:>5} edge>={e:.2f}: {r['bets']} bets, roi {r['roi']:+.4f}")
    mkt = bt.no_vig(matches)
    conf = bt.scored(matches) & matches["season"].between(*bt.CONFIRM) & preds["p_h"].notna()
    for lg in sd.TOP:
        sel = conf & (matches["league"] == lg)
        print(f"   CONFIRM {lg}: {int(sel.sum())} matches, log loss {bt.log_loss(preds[sel], matches.loc[sel, 'result']):.4f}"
              f" market {bt.log_loss(mkt[sel], matches.loc[sel, 'result']):.4f}")
    for cut in ("2011-02-05", "2021-02-06"):
        print(f"   point in time {cut}:", _pit(matches, cut))
    top = matches["league"].isin(sd.TOP) & preds["p_h"].notna()
    PREDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([matches.loc[top, ["match_id"]], preds.loc[top]], axis=1)
    out.to_csv(PREDS_OUT, index=False, compression="gzip")
    print(f"wrote {PREDS_OUT} ({len(out)} matches)")


if __name__ == "__main__":
    main()
