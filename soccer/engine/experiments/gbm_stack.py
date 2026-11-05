"""Candidate: a gradient-boosted stack over the pick (XGBoost, multi:softprob).

THE IDEA. Today's pick is the Friday/Tuesday market price, recalibrated
(soccer_model.pick: p ~ q^beta exp(c), weights from earlier seasons). Can a
gradient-boosted tree model, fed everything the engine knows before kickoff,
correct it where it is still wrong - by price region, by league, for clubs
whose results ran ahead of their shots, promoted clubs, late in the season?

  target     home / draw / away (multi:softprob, three classes)
  base       boosting starts FROM the pick: each match's base margin is
             log(pick), the pick's own walk-forward chances, so with no trees
             the candidate IS the pick and every tree is a correction the data
             had to pay for. Also measured: "market" (from the raw no-vig
             price) and "none" (from nothing: the trees learn the price too)
  features   all pre-match (FEATURE_SETS):
    market   the no-vig pre-match average (log p for home, draw, away), its
             overround, the best-vs-average gap per outcome log(max/avg),
             Bet365 against the average (log of the no-vig ratio), the no-vig
             over 2.5, its overround and best-vs-average gaps
    model    the shots-based ratings (experiments/shots_ratings, read-only):
             their home/draw/away and over-2.5 chances, expected goals lam and
             nu, log(model / market) per outcome, the rating predictors eta_h
             and eta_a (MU + home advantage + attack - opposing defence, each
             way: shots_ratings.rate keeps attack and defence only in these
             sums), promoted flags, matches since entering the tier, what is
             at stake in the table (a place up / the drop / nothing)
    form     per club over its last 5 and last 10 league matches (any of the
             ten divisions; earlier days only): goal difference, points, shots
             on target and shots difference, points minus the points its own
             pre-match prices expected; a goal-margin Elo (one pool per
             country, both tiers; difference with home advantage, and its
             win expectancy); rest days since each club's previous league
             match; the league's recent market miss (exponentially weighted
             mean of result minus no-vig price for home and draw, half-life
             380 matches, earlier days only); season stage; month; midweek
    league   one indicator per top-five league
  trees      learning rate 0.02, L2 10, row / column subsampling 0.8 / 0.7
             (seeded, single-threaded: deterministic), depth and
             min_child_weight as chosen; rounds by early stopping: fitted on
             seasons < s-1, the round count (0 allowed: the pick itself) with
             the lowest log loss on season s-1's top-five matches, then
             refitted on every season < s with that count.

THE PROTOCOL. Point in time: a match's features read only its own pre-match
prices (average, best, Bet365 - never a closing price, never Pinnacle) and
results / statistics of matches on EARLIER days; the shots model and the pick
are walk-forward themselves. Walk forward by season: the trees for season s
are fitted on the top-five matches of seasons 2006..s-1 (the shots model's
first season with chances is 2006) and refitted every season. Early stopping
needs a season to stop on, so the candidate starts in 2008: 2005-2007 are
NaN, as is any match without all three pre-match prices.

CHOSEN ON DEV ONLY. Every option was run walk-forward on the TUNE seasons
2008-2010 (5,477 matches, each season fitted on its own earlier seasons) and
compared with the pick on the same matches; the lowest log loss was chosen
(CHOSEN) and then checked on 2011-2015 (CHECK, 9,130 matches). 2016-2025 was
never scored by this module (compare() refuses anything outside DEV).
  grid  base {pick, market, none} x features {market, market+model, all}
        x depth {2, 3} x min_child_weight {50, 200} x rounds {fixed 25..600,
        early stopping}; then training rows {top five, + second tiers (their
        no-vig price recalibrated with the top five's weights as base, model
        features missing)} x features {market, market+form, all} x depth
        {2, 3}, early stopping (gbm_stack_tune*.csv in the scratchpad)
  TUNE, candidate - pick, log loss (paired se), early stopping:
        from the pick     -0.00037 (0.00030) .. +0.00024: market d3 mcw200
                          chosen; market+model -0.00012 .. +0.00020; all
                          -0.00015 .. +0.00024
        from the market   -0.00022 .. +0.00089 (the raw market itself +0.00089)
        from nothing      +0.0053 .. +0.0066 (1000 rounds: up to +0.022)
        + second tiers    -0.00029 .. +0.00015 (none better than top only)
        fixed rounds      every option worse than the pick from 100-200 rounds
                          on (+0.002 to +0.013 at 600): more trees, more noise
  CHOSEN: from the pick, market features, depth 3, min_child_weight 200.

MEASURED, DEV (python experiments/gbm_stack.py; candidate and pick on the
same matches, 2008-2015; paired se per match):

                       matches  log loss          Brier             accuracy
                                cand     pick     cand     pick     cand   pick
  DEV 2008-2015        14,607   0.97686  0.97698  0.58204  0.58212  53.02% 53.02%
    TUNE 2008-2010      5,477   0.98197  0.98234  0.58617  0.58644  53.06% 53.17%
    CHECK 2011-2015     9,130   0.97380  0.97376  0.57957  0.57952  53.00% 52.92%
  difference DEV -0.00012 (se 0.00027; by season 0.00023), 6 of 8 seasons
  better; TUNE -0.00037 (0.00030); CHECK +0.00004 (0.00039).
  per league, DEV (cand - pick, se): E0 +0.00006 (0.00063), SP1 +0.00023
  (0.00059), I1 +0.00019 (0.00056), D1 -0.00059 (0.00062), F1 -0.00057
  (0.00058). The raw no-vig market on the same matches: 0.97750.
  early-stopping rounds 2008-2015: 3, 37, 43, 157, 57, 60, 0, 35.

  The stack as briefed, every feature family (EVERY_FEATURE: market + model
  + form, otherwise as chosen) - NOT chosen, it lost on TUNE; its DEV numbers
  were looked at only after the choice: DEV -0.00040 (se 0.00023), TUNE
  -0.00015 (0.00016), CHECK -0.00055 (0.00036); log loss 0.97658, Brier
  0.58188, accuracy 52.97%; per league E0 -0.00025, SP1 -0.00057, I1 -0.00002,
  D1 -0.00123 (0.00060), F1 -0.00009. Switching to it now would be choosing on
  the check period.

  Point in time (pit_check, DEV cut 2011-02-05): results, statistics and
  every price from the cut on scrambled - all 4,723 predictions dated before
  the cut unmoved (max move 0.0); results and statistics scrambled from the
  cut on, prices from the next day - the 30 cut-day predictions unmoved; the
  predictions after the cut moved. Passed both ways: on the whole history
  (--pit-full, ~2 min) and, the default, on seasons to 2012 compared with the
  whole-history run (~25 s).
  Runtime: the walk-forward over every season 60-65 s on a shared 4-core box
  (most of it the shots model's own walk-forward); the whole run with the
  every-feature variant and the default scramble test ~2 min (~4 min with
  --pit-full). Single-threaded xgboost on purpose: with 4 threads on a busy
  box one fit took 300 s instead of 4.

VERDICT ON DEV. The trees do not beat the pick: -0.00012 is half a paired
standard error, and the option that won TUNE gave its edge back on CHECK -
the mark of choosing among ~150 noise-level options. Started from nothing,
the GBM does not even reproduce the market (+0.005 on TUNE); started from the
market it mostly re-learns the pick's recalibration; started from the pick
it stays within +-0.0004 of it. The briefed every-feature stack's -0.00040
(1.7 se, most of it the Bundesliga, where the pick's pooled recalibration
loses to the raw price on DEV) is a lead for a per-league recalibration, not
a result: it lost the pre-registered choice.

CAVEATS. A paired difference of 0.0003-0.0004 is one standard error on these
matches; the TUNE seasons train on 2-4 seasons only. Second tiers enter only
through form and Elo (and as optional training rows, with the model's
chances missing - predict_leagues for their chances costs ~45 s more). Rest
days count league matches only (cups and Europe are not on file).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import soccer_backtest as bt  # noqa: E402
import soccer_data as sd  # noqa: E402
import soccer_model  # noqa: E402
from experiments import shots_ratings as sr  # noqa: E402

OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/engine/gbm_stack.csv.gz")
FIRST_MODEL_SEASON = 2006           # shots_ratings' first season with chances (2005 warms up)
TUNE = (2008, 2010)                 # test seasons the options were compared on
PIT_CUT, PIT_UPTO = "2011-02-05", 2012   # the scramble test's DEV cut; its frame: seasons to 2012
CHECK = (2011, 2015)                # the chosen option checked on these, untouched while choosing
NTHREAD = 1
SEED = 7
MIN_TRAIN = 1000
RES3 = ("H", "D", "A")

# ----------------------------------------------------------------- features

MARKET = ["lq_h", "lq_d", "lq_a", "over", "gap_h", "gap_d", "gap_a", "b365_h", "b365_d", "b365_a",
          "q_o25", "ou_over", "gap_o", "gap_u"]
MODEL = ["mp_h", "mp_d", "mp_a", "mp_o25", "lam", "nu", "dm_h", "dm_d", "dm_a", "eta_h", "eta_a",
         "prom_h", "prom_a", "ntier_h", "ntier_a", "up_h", "up_a", "rel_h", "rel_a", "dead_h", "dead_a"]
FORM = ([f"{c}{w}_{s}" for c in ("gd", "pts", "sotd", "shd", "pvx") for w in (5, 10) for s in ("h", "a")]
        + ["elo_d", "elo_p", "rest_h", "rest_a", "lbias_h", "lbias_d", "stage", "month", "midweek"])
LEAGUE = [f"lg_{c}" for c in sd.TOP]
FEATURE_SETS = {
    "market": MARKET + LEAGUE,
    "market+model": MARKET + MODEL + LEAGUE,
    "market+form": MARKET + FORM + LEAGUE,
    "all": MARKET + MODEL + FORM + LEAGUE,
}


def _num(m, c):
    return pd.to_numeric(m[c], errors="coerce").to_numpy(dtype=float)


def _no_vig(m, cols):
    """(no-vig probabilities (n, k), overround) from decimal prices; NaN
    where any is missing."""
    p = np.column_stack([_num(m, c) for c in cols])
    p[~(p > 0)] = np.nan
    inv = 1.0 / p
    tot = inv.sum(axis=1)
    return inv / tot[:, None], tot - 1.0


def _form(m, q):
    """Per club (country | name), over its previous 5 and 10 league matches
    on earlier days: goal difference, points, shots on target and shots
    difference, points minus market-expected points; rest days. A club plays
    at most once a day (checked), so its previous entries are earlier days."""
    n = len(m)
    country = m["league"].map(lambda c: sd.ALL_LEAGUES[c].country).to_numpy()
    day = pd.to_datetime(m["date"]).to_numpy()
    hg, ag = _num(m, "hg"), _num(m, "ag")
    hst, ast, hs, as_ = _num(m, "hst"), _num(m, "ast"), _num(m, "hs"), _num(m, "as")
    pts_h = np.where(np.isnan(hg) | np.isnan(ag), np.nan, np.where(hg > ag, 3.0, np.where(hg == ag, 1.0, 0.0)))
    pts_a = np.where(np.isnan(pts_h), np.nan, np.where(pts_h == 1.0, 1.0, 3.0 - pts_h))
    xp_h = 3 * q[:, 0] + q[:, 1]
    xp_a = 3 * q[:, 2] + q[:, 1]
    long = pd.DataFrame({
        "pos": np.r_[np.arange(n), np.arange(n)],
        "side": np.r_[np.zeros(n, int), np.ones(n, int)],
        "team": np.r_[country + "|" + m["home"].to_numpy(), country + "|" + m["away"].to_numpy()],
        "day": np.r_[day, day],
        "gd": np.r_[hg - ag, ag - hg], "pts": np.r_[pts_h, pts_a],
        "sotd": np.r_[hst - ast, ast - hst], "shd": np.r_[hs - as_, as_ - hs],
        "pvx": np.r_[pts_h - xp_h, pts_a - xp_a],
    })
    if long.duplicated(["team", "day"]).any():
        raise ValueError("a club with two matches on one day: form would see the same day")
    long = long.sort_values(["team", "day", "pos"], kind="stable").reset_index(drop=True)
    g = long.groupby("team", sort=False)
    out = {}
    for c in ("gd", "pts", "sotd", "shd", "pvx"):
        prev = g[c].shift(1)
        for w in (5, 10):
            long[f"{c}{w}"] = prev.groupby(long["team"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    long["rest"] = g["day"].diff().dt.days.clip(upper=60).astype(float)
    for side, s in ((0, "h"), (1, "a")):
        part = long[long["side"] == side].set_index("pos").sort_index()
        for c in ("gd", "pts", "sotd", "shd", "pvx"):
            for w in (5, 10):
                out[f"{c}{w}_{s}"] = part[f"{c}{w}"].to_numpy()
        out[f"rest_{s}"] = part["rest"].to_numpy()
    return out


def _elo(m, k=20.0, hfa=60.0, start=(1500.0, 1350.0)):
    """A goal-margin Elo, one pool per country (both tiers): each match's
    rating difference (home - away + hfa) and its home-win expectancy, read
    before the day's results enter."""
    n = len(m)
    order = np.lexsort((np.arange(n), m["date"].to_numpy()))
    country = m["league"].map(lambda c: sd.ALL_LEAGUES[c].country).to_numpy()[order]
    tier = m["league"].map(lambda c: sd.ALL_LEAGUES[c].tier).to_numpy()[order]
    home = (country + "|" + m["home"].to_numpy()[order])
    away = (country + "|" + m["away"].to_numpy()[order])
    hg, ag = _num(m, "hg")[order], _num(m, "ag")[order]
    dates = m["date"].to_numpy()[order]
    R = {}
    diff = np.empty(n)
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], n]
    for d0, d1 in zip(starts, ends):
        for i in range(d0, d1):
            for t in (home[i], away[i]):
                if t not in R:
                    R[t] = start[0] if tier[i] == 1 else start[1]
            diff[i] = R[home[i]] - R[away[i]] + hfa
        for i in range(d0, d1):
            if hg[i] != hg[i] or ag[i] != ag[i]:
                continue
            we = 1.0 / (1.0 + 10 ** (-diff[i] / 400.0))
            w = 1.0 if hg[i] > ag[i] else (0.5 if hg[i] == ag[i] else 0.0)
            gd = abs(hg[i] - ag[i])
            mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
            delta = k * mult * (w - we)
            R[home[i]] += delta
            R[away[i]] -= delta
    out = np.empty(n)
    out[order] = diff
    return out


def _league_bias(m, q, half=380.0):
    """Per league, an exponentially weighted mean over matches on EARLIER days
    of (result - no-vig price) for home and draw: the market's recent miss in
    that league. NaN until 50 matches' worth of weight."""
    n = len(m)
    res = m["result"].to_numpy()
    y = np.column_stack([(res == r).astype(float) for r in ("H", "D")])
    ok = m["result"].isin(RES3).to_numpy() & np.isfinite(q).all(axis=1)
    resid = np.where(ok[:, None], y - q[:, :2], 0.0)
    out = np.full((n, 2), np.nan)
    dec = 0.5 ** (1.0 / half)
    dates = m["date"].to_numpy()
    for lg, pos in m.groupby("league").indices.items():
        pos = pos[np.lexsort((pos, dates[pos]))]
        d = dates[pos]
        starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
        ends = np.r_[starts[1:], len(pos)]
        A = np.zeros(2)
        B = 0.0
        for s0, s1 in zip(starts, ends):
            rows = pos[s0:s1]
            if B >= 50:
                out[rows] = A / B
            k = ok[rows]
            nk = int(k.sum())
            A = A * dec ** nk + resid[rows][k].sum(axis=0)
            B = B * dec ** nk + nk
    return out


def build_features(m, preds, feats):
    """The feature frame (index like m) for every row; NaN where a part is
    missing (xgboost routes missing values)."""
    n = len(m)
    q, over = _no_vig(m, ["avg_h", "avg_d", "avg_a"])
    qb, _ = _no_vig(m, ["b365_h", "b365_d", "b365_a"])
    qo, ou_over = _no_vig(m, ["avg_o25", "avg_u25"])
    X = {}
    lq = np.log(np.clip(q, 1e-6, 1))
    for j, k in enumerate("hda"):
        X[f"lq_{k}"] = lq[:, j]
        X[f"gap_{k}"] = np.log(_num(m, f"max_{k}") / _num(m, f"avg_{k}"))
        X[f"b365_{k}"] = np.log(qb[:, j] / q[:, j])
    X["over"] = over
    X["q_o25"] = qo[:, 0]
    X["ou_over"] = ou_over
    X["gap_o"] = np.log(_num(m, "max_o25") / _num(m, "avg_o25"))
    X["gap_u"] = np.log(_num(m, "max_u25") / _num(m, "avg_u25"))
    mp = preds[["p_h", "p_d", "p_a"]].to_numpy(float)
    for j, k in enumerate("hda"):
        X[f"mp_{k}"] = mp[:, j]
        X[f"dm_{k}"] = np.log(np.clip(mp[:, j], 1e-6, 1)) - lq[:, j]
    X["mp_o25"] = preds["p_o25"].to_numpy(float)
    X["lam"] = preds["lam"].to_numpy(float)
    X["nu"] = preds["nu"].to_numpy(float)
    X["eta_h"] = feats["eta_h_0"].to_numpy(float)
    X["eta_a"] = feats["eta_a_0"].to_numpy(float)
    for s in ("h", "a"):
        X[f"prom_{s}"] = feats[f"prom_{s}"].to_numpy(float)
        X[f"ntier_{s}"] = np.minimum(feats[f"n_{s}"].to_numpy(float), 100.0)
        for c in ("up", "rel", "dead"):
            X[f"{c}_{s}"] = feats[f"{c}_{s}"].to_numpy(float)
    X.update(_form(m, q))
    X["elo_d"] = _elo(m)
    X["elo_p"] = 1.0 / (1.0 + 10 ** (-X["elo_d"] / 400.0))
    lb = _league_bias(m, q)
    X["lbias_h"], X["lbias_d"] = lb[:, 0], lb[:, 1]
    # season stage: the club's games played over the season's games (rosters are fixture facts)
    teams = pd.concat([m[["league", "season", "home"]].rename(columns={"home": "t"}),
                       m[["league", "season", "away"]].rename(columns={"away": "t"})])
    size = teams.drop_duplicates().groupby(["league", "season"]).size()
    games = 2.0 * (pd.MultiIndex.from_frame(m[["league", "season"]]).map(size).to_numpy(float) - 1.0)
    X["stage"] = np.clip(0.5 * (feats["gp_h"].to_numpy(float) + feats["gp_a"].to_numpy(float)) / games, 0, 1)
    day = pd.to_datetime(m["date"])
    X["month"] = ((day.dt.month - 7) % 12).to_numpy(float)
    X["midweek"] = day.dt.dayofweek.isin([1, 2, 3]).to_numpy(float)
    for c in sd.TOP:
        X[f"lg_{c}"] = (m["league"] == c).to_numpy(float)
    out = pd.DataFrame(X, index=m.index)
    return out.where(np.isfinite(out))           # a zero or missing price: missing, never infinite


def inputs(matches):
    """Everything the trees need, config-free: the shots model (rate +
    calibrate = shots_ratings.predict_walk_forward with its score
    parameters), today's pick on the same frame, the features."""
    m = matches
    feats = sr.rate(m)
    preds = sr.calibrate(m, feats, score_params=True)
    pick, weights = soccer_model.pick(m, preds)
    X = build_features(m, preds, feats)
    return dict(m=m, preds=preds, pick=pick, weights=weights, X=X, pick_ext=_pick_ext(m, pick, weights))


def _pick_ext(m, pick, weights):
    """The pick, and for a second-tier row (a training row only, with
    rows="ten") its own no-vig price recalibrated with the top five's weights
    for its season - fitted on earlier seasons, as the pick's are."""
    out = pick[["p_h", "p_d", "p_a"]].copy()
    q = bt.no_vig(m, "pre")
    second = ~m["league"].isin(sd.TOP) & q.notna().all(axis=1)
    for season, w in weights.items():
        rows = second & (m["season"] == season)
        if rows.any():
            out.loc[rows] = soccer_model.recalibrate(q[rows], w).to_numpy()
    return out


# ----------------------------------------------------------------- the trees

# Chosen on TUNE (2008-2010) by walk-forward log loss, every option beside
# the pick on the same 5,477 matches (the grid is in the module notes).
CHOSEN = dict(base="pick", features="market", rows="top", max_depth=3, eta=0.02, min_child_weight=200.0,
              reg_lambda=10.0, subsample=0.8, colsample_bytree=0.7, max_rounds=600,
              rounds_by="early_stop", patience=100, fallback_rounds=None)
# The stack as briefed - every feature family - reported beside it on DEV.
EVERY_FEATURE = dict(CHOSEN, features="all")


def _params(cfg):
    return {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
            "max_depth": cfg["max_depth"], "eta": cfg["eta"], "min_child_weight": cfg["min_child_weight"],
            "lambda": cfg["reg_lambda"], "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"], "tree_method": "hist", "nthread": NTHREAD,
            "seed": SEED}


def _base(inp, cfg):
    """(n, 3) base margins: log(pick), log(no-vig market) or None."""
    if cfg["base"] == "none":
        return None
    src = inp["pick_ext"] if cfg["base"] == "pick" else bt.no_vig(inp["m"], "pre")
    return np.log(np.clip(src[["p_h", "p_d", "p_a"]].to_numpy(float), 1e-9, 1))


def _dm(X, y, base, rows):
    return xgb.DMatrix(X[rows], label=None if y is None else y[rows],
                       base_margin=None if base is None else base[rows], missing=np.nan, nthread=NTHREAD)


def _ll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def walk_forward(inp, cfg, seasons=None, log=None, all_rounds=None):
    """Season by season: trees fitted on top-five matches of every earlier
    season with model features, predictions for the season's top-five
    priced matches. Returns (n, 3) probabilities (NaN where none) and
    {season: rounds used}. all_rounds: a list of round counts - then a dict
    {rounds: (n, 3)} with the fixed counts (tuning; no early stopping)."""
    m = inp["m"]
    ten = cfg.get("rows", "top") == "ten"
    cols = FEATURE_SETS[cfg["features"]]
    X = inp["X"][cols].to_numpy(float)
    if ten:                                     # second tiers as training rows: a flag for them
        X = np.column_stack([X, (~m["league"].isin(sd.TOP)).to_numpy(float)])
    base = _base(inp, cfg)
    y = bt.outcome_index(m["result"]).to_numpy()
    y = np.where(np.isnan(y.astype(float)), -1, y).astype(int)
    season = m["season"].to_numpy(dtype=float)
    top = m["league"].isin(sd.TOP).to_numpy()
    priced = bt.no_vig(m, "pre").notna().all(axis=1).to_numpy()
    has_model = inp["preds"]["p_h"].notna().to_numpy()
    target = top & priced & has_model
    labeled = target & (y >= 0)
    if ten:
        labeled = labeled | (~top & priced & (season >= FIRST_MODEL_SEASON) & (y >= 0)
                             & (np.isfinite(base).all(axis=1) if base is not None else True))
    val_rows = target & (y >= 0)                 # early stopping is judged on top-five matches only
    n = len(m)
    out = np.full((n, 3), np.nan)
    outs = {r: np.full((n, 3), np.nan) for r in (all_rounds or [])}
    used = {}
    params = _params(cfg)
    for s in sorted(set(season[target].astype(int))):
        if seasons is not None and s not in seasons:
            continue
        tr = labeled & (season < s)
        te = target & (season == s)
        if tr.sum() < MIN_TRAIN or not te.any():
            continue
        dte = _dm(X, None, base, te)
        if all_rounds:
            bst = xgb.train(params, _dm(X, y, base, tr), num_boost_round=max(all_rounds))
            for r in all_rounds:
                if r == 0:
                    outs[r][te] = _softmax(base[te]) if base is not None else np.nan
                else:
                    outs[r][te] = bst.predict(dte, iteration_range=(0, r))
            continue
        rounds = cfg["fallback_rounds"]
        inner = labeled & (season < s - 1)
        val = val_rows & (season == s - 1)
        if cfg["rounds_by"] == "early_stop" and inner.sum() >= MIN_TRAIN and val.any():
            # the round count (0: the base alone) with the lowest log loss on
            # the last training season, trees fitted on the seasons before it;
            # stops once `patience` rounds pass without a new low
            ev = {}
            bst = xgb.train(params, _dm(X, y, base, inner), num_boost_round=cfg["max_rounds"],
                            evals=[(_dm(X, y, base, val), "val")], evals_result=ev, verbose_eval=False,
                            early_stopping_rounds=cfg["patience"])
            curve = np.r_[_ll(_softmax(base[val]), y[val]) if base is not None else np.inf,
                          ev["val"]["mlogloss"]]
            rounds = int(np.argmin(curve))
        if rounds is None:                      # no season to stop on (2007): no number
            continue
        used[s] = rounds
        if rounds == 0:
            out[te] = _softmax(base[te])
        else:
            bst = xgb.train(params, _dm(X, y, base, tr), num_boost_round=rounds)
            out[te] = bst.predict(dte)
        if log:
            log(f"   season {s}: train {int(tr.sum())}, rounds {rounds}")
    return (outs if all_rounds else out), used


def predict_walk_forward(matches, cfg=None, return_inputs=False):
    """The candidate's p_h, p_d, p_a for every top-five match it can price
    (NaN where it cannot: 2005-06, 2006-07, any match without a full
    pre-match price), each from trees fitted on earlier seasons only."""
    cfg = {**CHOSEN, **(cfg or {})}
    inp = inputs(matches)
    p, used = walk_forward(inp, cfg)
    out = pd.DataFrame(p, index=matches.index, columns=["p_h", "p_d", "p_a"])
    out.attrs["rounds"] = used
    if return_inputs:
        return out, inp
    return out


# ----------------------------------------------------------------- measurement

def _guard(lo, hi):
    if hi > bt.DEV[1] or lo < bt.DEV[0]:
        raise ValueError("this module reports DEV (2006-2015) only")


def compare(m, cand, pick, lo, hi):
    """Candidate vs the pick on the same scored matches of seasons lo-hi."""
    _guard(lo, hi)
    sel = (bt.scored(m) & m["season"].between(lo, hi)
           & cand[["p_h", "p_d", "p_a"]].notna().all(axis=1) & pick[["p_h", "p_d", "p_a"]].notna().all(axis=1))
    r = m.loc[sel, "result"]
    y = bt.outcome_index(r).to_numpy()
    pc = cand.loc[sel, ["p_h", "p_d", "p_a"]].to_numpy(float)
    pp = pick.loc[sel, ["p_h", "p_d", "p_a"]].to_numpy(float)
    llc = -np.log(np.clip(pc[np.arange(len(y)), y], 1e-12, 1))
    llp = -np.log(np.clip(pp[np.arange(len(y)), y], 1e-12, 1))
    d = llc - llp
    by_season = pd.Series(d).groupby(m.loc[sel, "season"].to_numpy()).mean()
    out = {
        "matches": int(sel.sum()),
        "log_loss": float(llc.mean()), "pick_log_loss": float(llp.mean()),
        "brier": bt.brier(cand[sel], r), "pick_brier": bt.brier(pick[sel], r),
        "accuracy": bt.accuracy(cand[sel], r), "pick_accuracy": bt.accuracy(pick[sel], r),
        "diff": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "se_by_season": float(by_season.std(ddof=1) / math.sqrt(len(by_season))) if len(by_season) > 1 else None,
        "seasons_better": f"{int((by_season < 0).sum())} of {len(by_season)}",
    }
    per = {}
    for lg in sd.TOP:
        k = (m.loc[sel, "league"] == lg).to_numpy()
        per[lg] = {"matches": int(k.sum()), "log_loss": float(llc[k].mean()), "pick_log_loss": float(llp[k].mean()),
                   "diff": float(d[k].mean()), "se": float(d[k].std(ddof=1) / math.sqrt(k.sum()))}
    out["per_league"] = per
    return out


def tune(inp, log=print):
    """The DEV-only option comparison: walk-forward log loss on TUNE seasons
    (each fitted on its own earlier seasons) for every option, beside the
    pick on the same matches. Never touches seasons after DEV."""
    m = inp["m"]
    seasons = set(range(TUNE[0], TUNE[1] + 1))
    pick = inp["pick"]
    rows = []
    grid = []
    for base in ("pick", "market", "none"):
        for fs in (FEATURE_SETS if base != "none" else ["all"]):
            for depth in (2, 3):
                for mcw in (50.0, 200.0):
                    grid.append(dict(CHOSEN, base=base, features=fs, max_depth=depth, min_child_weight=mcw))
    rounds = [0, 25, 50, 100, 200, 400, 600]
    for cfg in grid:
        res = walk_forward(inp, cfg, seasons, all_rounds=rounds if cfg["base"] != "none" else [100, 200, 400, 600, 1000])[0]
        for r, p in res.items():
            cand = pd.DataFrame(p, index=m.index, columns=["p_h", "p_d", "p_a"])
            c = compare(m, cand, pick, *TUNE)
            rows.append(dict(base=cfg["base"], features=cfg["features"], depth=cfg["max_depth"],
                             mcw=cfg["min_child_weight"], rounds=r, n=c["matches"],
                             ll=c["log_loss"], pick=c["pick_log_loss"], diff=c["diff"], se=c["se"]))
        cfg_es = dict(cfg, rounds_by="early_stop", max_rounds=600 if cfg["base"] != "none" else 1500)
        p, used = walk_forward(inp, cfg_es, seasons)
        c = compare(m, pd.DataFrame(p, index=m.index, columns=["p_h", "p_d", "p_a"]), pick, *TUNE)
        rows.append(dict(base=cfg["base"], features=cfg["features"], depth=cfg["max_depth"],
                         mcw=cfg["min_child_weight"], rounds="es " + ",".join(str(used[s]) for s in sorted(used)),
                         n=c["matches"], ll=c["log_loss"], pick=c["pick_log_loss"], diff=c["diff"], se=c["se"]))
        log(f"   {cfg['base']:>6} {cfg['features']:>12} d{cfg['max_depth']} mcw{cfg['min_child_weight']:.0f}: "
            + "  ".join(f"{r['rounds']}:{r['diff']:+.5f}" for r in rows[-len(res) - 1:]))
    return pd.DataFrame(rows)


def pit_check(matches, cut, cfg=None, base_preds=None, upto=None):
    """Scramble test at `cut`: every result, match statistic and price
    (pre-match average / best / Bet365 / Pinnacle, over/under, closing) of
    matches dated on or after the cut is scrambled; every candidate
    prediction dated before the cut must not move. A second scramble keeps
    the cut day's own prices (results and statistics scrambled from the cut
    on, prices from the next day on): predictions ON the cut day must not
    move either. upto: run the scrambled frames on seasons <= upto only (the
    rest of history dropped as well) - compared with `base_preds`, the
    predictions on the WHOLE, unscrambled frame. Returns a dict of checks."""
    rng = np.random.default_rng(1)
    base_preds = predict_walk_forward(matches, cfg) if base_preds is None else base_preds
    frame = matches if upto is None else matches[matches["season"] <= upto]
    later = (frame["date"] >= cut).to_numpy()
    after = (frame["date"] > cut).to_numpy()

    def scramble(df, res_rows, price_rows):
        s = df.copy()
        k = int(res_rows.sum())
        for c in ("hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"):
            s.loc[res_rows, c] = rng.integers(0, 6 if c in ("hg", "ag", "hthg", "htag") else 25, k)
        s.loc[res_rows, "result"] = np.where(s.loc[res_rows, "hg"] > s.loc[res_rows, "ag"], "H",
                                             np.where(s.loc[res_rows, "hg"] < s.loc[res_rows, "ag"], "A", "D"))
        k = int(price_rows.sum())
        for c in [c for c in s.columns if c.startswith(("avg_", "max_", "ps_", "b365_", "avgc_", "psc_"))]:
            s.loc[price_rows, c] = rng.uniform(1.2, 9.0, k)
        return s

    before = (frame["date"] < cut).to_numpy()
    on = (frame["date"] == cut).to_numpy()
    a = predict_walk_forward(scramble(frame, later, later), cfg).to_numpy(float)
    b = predict_walk_forward(scramble(frame, later, after), cfg).to_numpy(float)
    x0 = base_preds.loc[frame.index].to_numpy(float)

    def same(x, y):
        return bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x, y, equal_nan=True, rtol=0, atol=1e-12))
    return {
        "cut": cut, "frame": "every season" if upto is None else f"seasons to {upto}",
        "before_cut_predictions": int(np.isfinite(x0[before, 0]).sum()),
        "before_cut_unmoved": same(x0[before], a[before]),
        "cut_day_predictions": int(np.isfinite(x0[on, 0]).sum()),
        "cut_day_unmoved_results_scrambled": same(x0[on | before], b[on | before]),
        "max_move_before_cut": float(np.nanmax(np.abs(x0[before] - a[before]))) if before.any() else None,
        "after_cut_moved": not same(x0[later], a[later]),
    }


def main():
    args = sys.argv[1:]
    matches = sd.load_matches()
    t0 = time.time()
    cand, inp = predict_walk_forward(matches, return_inputs=True)
    runtime = time.time() - t0
    print(f"walk-forward (shots model + pick + features + trees), every season: {runtime:.1f}s")
    dev_rounds = {int(k): v for k, v in cand.attrs["rounds"].items() if k <= bt.DEV[1]}
    print("rounds per DEV season (early stopping on the season before):", dev_rounds)
    m, pick = matches, inp["pick"]
    top = m["league"].isin(sd.TOP)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([m.loc[top, ["match_id"]], cand.loc[top]], axis=1).to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT} ({int(top.sum())} top-five matches, {int(cand.loc[top, 'p_h'].notna().sum())} priced)")
    rep = {"DEV": compare(m, cand, pick, *bt.DEV), "TUNE": compare(m, cand, pick, *TUNE),
           "CHECK": compare(m, cand, pick, *CHECK)}
    print(json.dumps(rep, indent=1))
    p_all, used_all = walk_forward(inp, EVERY_FEATURE)
    every = pd.DataFrame(p_all, index=m.index, columns=["p_h", "p_d", "p_a"])
    print("every feature family (market + model + form), rounds per DEV season:",
          {int(k): v for k, v in used_all.items() if k <= bt.DEV[1]})
    print(json.dumps({k: compare(m, every, pick, *v) for k, v in
                      (("DEV", bt.DEV), ("TUNE", TUNE), ("CHECK", CHECK))}, indent=1))
    mkt = bt.no_vig(m, "pre")
    mkt.loc[~cand.notna().all(axis=1)] = np.nan
    r = compare(m, mkt, pick, *bt.DEV)
    print(f"the raw no-vig market on the same DEV matches: {r['log_loss']:.5f} vs the pick {r['pick_log_loss']:.5f}"
          f" ({r['matches']} matches)")
    if "--tune" in args:
        t = tune(inp)
        t.to_csv(OUT.parent / "gbm_stack_tune.csv", index=False)
        print(t.sort_values("diff").head(25).to_string())
    if "--no-pit" not in args:
        t1 = time.time()
        full = "--pit-full" in args
        print("point in time:", pit_check(matches, PIT_CUT, base_preds=cand, upto=None if full else PIT_UPTO),
              f"({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
