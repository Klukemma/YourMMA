"""Candidate: COACHING IN THE GAME MODEL. Steps 1-5 and 7 of the measurement
plan in nfl_approach's docstring, for the winner / margin / total models and
the pick (their blend with the market). The touchdown model (step 6) is not
touched here.

    python nfl/engine/experiments/coach_game.py             # chosen config: DEV report, scramble, CSV
    python nfl/engine/experiments/coach_game.py --search    # the DEV search that chose it

WHAT A SIDE READS BEFORE KICKOFF. For the offence (team T, head coach C) and
the defence it faces (the opponent's team D):

  coach state     keyed by the coach's name (nfl_approach.COACH_NAMES applied;
                  key="team" keys the same state by the franchise instead -
                  the baseline the plan requires). Exponentially weighted sums
                  (half_life in the key's games, `keep` of the weight carried
                  into a new season) of each measure's TILT - the game's value
                  minus the league's walk-forward value at that game's kickoff,
                  times its count:
                    pass      neutral pass rate (pass_measure="npr", every
                              season) or neutral PROE ("proe", 2006 on)   neutral plays
                    early     early-down pass rate / early-down PROE      early-down plays
                    rz_rush   red-zone rush rate                          red-zone plays
                    gz        go-zone go rate (gz="last" / "first" decision) go-zone decisions
                    pace      neutral seconds between snaps               gaps
                    nohuddle  neutral no-huddle rate                      neutral plays
                    shotgun   shotgun rate                                plays
                    pass_epa  EPA per dropback                            dropbacks
                    rush_epa  EPA per designed run                        runs
                  A tilt is read shrunk toward the league: sum(count x tilt) /
                  (sum(count) + k), k = k_mult x K_BASE[measure] in the
                  measure's own units. A coach with no history reads the league.
  defence state   keyed by the defence's franchise (its players stay when a
                  coach goes), from nfl_approach.allowed() rows, shrunk the
                  same way: EPA allowed per dropback and per run, EPA and
                  success allowed per play, the neutral pass rate it drew.
  defence type    rmp = rush EPA allowed tilt - pass EPA allowed tilt (above 0:
                  softer against the run than the pass).
  the shift       the coach's own weighted regression (own state) of his game
                  pass tilt on x = [rmp of the defence he faced, as it stood
                  before that game; his closing spread / 7], centred on his own
                  history, shrunk to the league's within-coach slope by k_slope
                  games of the league's x-variance (ridge toward the prior):
                    xtilt = base + beta . (x_now - xbar)
                  slopes=False reads xtilt = base.
  the payoff      "how much the expected approach benefits him": his game
                  EPA/play (and success rate) minus what the league would gain
                  against that defence (league + the defence's allowed tilt, as
                  it stood before that game), regressed (own state) on the
                  defence type rmp, shrunk the same way (intercept toward 0 -
                  the league - with k_pay games; slope toward the league's):
                    payoff = a + b (rmp_now - xbar)
  the matchup     plan step 4: p = league neutral pass rate + xtilt;
                    xepa = p (L_pass + off pass tilt + def pass allowed tilt)
                         + (1 - p) (L_rush + off rush tilt + def rush allowed tilt)
                    fit  = xtilt x (pass matchup - rush matchup): what his
                           expected tilt is worth against this defence beyond
                           a league-average call mix.
  experience      head-coach games before this one (always by the coach),
                  new_coach (under 17 of them), new_here (under 17 with this
                  franchise).

POINT IN TIME (nfl_approach's rule). One pass over the schedule in kickoff
order. A game reads both sides as they stood, then folds its two approach
rows into the coaches' and defences' own state at once (each plays one game at
a time). League means and the league slopes are SHARED: a game's rows fold
into them only once its kickoff + nfl_features.FINAL_AFTER is at or before the
reading game's kickoff. The x of a folded game (defence type, spread, league
values) are those read at ITS kickoff. scramble_test() checks it the direct
way.

THE MODEL. The features (home minus away, and sums for the total) are added
to nfl_features.FEATURES / TOTAL_FEATURES for one walk_forward call of
nfl_model (patched in and restored), so the winner L1, the margin ridge, the
total ridge and the market blend are the shipped code, refitted exactly as
the shipped pick is. Choices are made on DEV 2006-2015 only; nothing here
computes a CONFIRM number.

RESULT ON DEV (2,666 decided games, the shipped pick on the same games).
18 configurations searched (--search). Chosen: set "full", half_life 8,
keep 1.0, k_mult 0.4, k_slope 10, gz "first", coach-keyed.
  pick          log loss 0.61111 vs 0.61130: -0.00019, SE 0.00056 (by season
                0.00060), better in 4 of 10 seasons; 2006 alone is -0.0048,
                and the other nine average +0.00032. Accuracy 66.47% vs
                66.35% (3 games). The closing market: 0.61077.
  model alone   0.62451 vs 0.62068: +0.00383 worse, SE 0.00214, worse in 8
                of 10 seasons; accuracy 64.85% vs 65.34%.
  team-keyed    pick -0.00003 (SE 0.00038); coach minus team -0.00016, SE
                0.00024.
  margin/total  margin MAE 10.94 vs 10.86, against the spread 51.2% vs
                51.9%; total MAE 10.64 vs 10.65.
  vs the market no coach column has |z| > 1.1 beside logit(closing price)
                on DEV (16 columns).
  scramble      2011 week 10, every kickoff: features and predictions
                identical (max gap 0.0); two weeks on they move.
Verdict: no measurable gain; nothing here should ship on this evidence.
"""

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import nfl_approach as na  # noqa: E402
import nfl_backtest as bt  # noqa: E402
import nfl_data as nd  # noqa: E402
import nfl_features as nf  # noqa: E402
import nfl_model as nm  # noqa: E402

OUT = Path("/tmp/claude-0/-home-user-YourMMA/1c56e1e4-4b1a-5ef7-aa98-0e6e32f58cbb/scratchpad/coach/coach_game.csv.gz")


@dataclass(frozen=True)
class CoachParams:
    key: str = "coach"           # "coach" or "team" (the baseline)
    half_life: float = 16.0      # games of the key
    keep: float = 0.75           # weight carried into a new season
    k_mult: float = 1.0          # x K_BASE: the shrink toward the league
    k_slope: float = 40.0        # games of league x-variance behind each slope prior
    k_pay: float = 16.0          # games behind the payoff intercept's prior (0 = league)
    slopes: bool = True
    pass_measure: str = "npr"    # "npr" (neutral pass rate over league) or "proe"
    gz: str = "last"             # "last" or "first" fourth-down decision of a series
    league_half_life: float = 600.0   # team-games


# Measures: name -> (value column, count column). pass/early/gz resolved by params.
def measures(p):
    pas = ("neutral_pass_rate", "neutral_plays") if p.pass_measure == "npr" else ("neutral_proe", "neutral_plays")
    early = ("early_pass_rate", "early_plays") if p.pass_measure == "npr" else ("early_proe", "early_plays")
    gz = ("gz_go_rate", "gz_downs") if p.gz == "last" else ("gz_first_go_rate", "gz_first_downs")
    return {
        "pass": pas, "early": early, "rz_rush": ("rz_rush_rate", "rz_plays"), "gz": gz,
        "pace": ("pace_sec", "pace_snaps"), "nohuddle": ("neutral_no_huddle_rate", "neutral_plays"),
        "shotgun": ("shotgun_rate", "plays"), "pass_epa": ("pass_epa_play", "pass_plays"),
        "rush_epa": ("rush_epa_play", "rush_plays"), "epa": ("epa_play", "plays"),
        "sr": ("success_rate", "plays"),
        # the plain neutral pass rate, always: p in the matchup is a pass rate
        "npr": ("neutral_pass_rate", "neutral_plays"),
    }


NAMES = ["pass", "early", "rz_rush", "gz", "pace", "nohuddle", "shotgun", "pass_epa", "rush_epa",
         "epa", "sr", "npr"]
IX = {n: i for i, n in enumerate(NAMES)}
# About five games' worth of each count: the shrink at k_mult = 1, in the
# measure's own units (plays, decisions, gaps). Fixed before any result.
K_BASE = np.array([120.0, 250.0, 50.0, 8.0, 120.0, 120.0, 320.0, 200.0, 130.0, 320.0, 320.0, 120.0])
PASS_W = 25.0      # neutral plays in a game's regression weight of 1
PLAYS_W = 62.0     # plays in a game's payoff weight of 1
NEW_GAMES = 17


class _Reg:
    """Exponentially weighted sufficient statistics of y (q) on x (p)."""

    __slots__ = ("w", "x", "y", "xx", "xy")

    def __init__(self, p, q):
        self.w = 0.0
        self.x, self.y = np.zeros(p), np.zeros(q)
        self.xx, self.xy = np.zeros((p, p)), np.zeros((p, q))

    def scale(self, f):
        self.w *= f
        self.x *= f
        self.y *= f
        self.xx *= f
        self.xy *= f

    def add(self, w, x, y):
        self.w += w
        self.x += w * x
        self.y += w * y
        self.xx += w * np.outer(x, x)
        self.xy += w * np.outer(x, y)

    def centred(self):
        if self.w <= 0:
            return None, None, None
        xbar = self.x / self.w
        cxx = self.xx - self.w * np.outer(xbar, xbar)
        cxy = self.xy - np.outer(xbar, self.y / self.w) * self.w
        return xbar, cxx, cxy


class _State:
    """One key's (coach / team / defence) weighted tilt sums and regressions."""

    __slots__ = ("num", "den", "season", "reg_pass", "reg_pay")

    def __init__(self):
        self.num, self.den = np.zeros(len(NAMES)), np.zeros(len(NAMES))
        self.season = None
        self.reg_pass, self.reg_pay = _Reg(2, 1), _Reg(1, 2)

    def roll(self, season, keep):
        if self.season is not None and self.season != season:
            self.num *= keep
            self.den *= keep
            self.reg_pass.scale(keep)
            self.reg_pay.scale(keep)
        self.season = season

    def decay(self, d):
        self.num *= d
        self.den *= d
        self.reg_pass.scale(d)
        self.reg_pay.scale(d)

    def tilt(self, k):
        return self.num / (self.den + k)


def _row_values(r, meas):
    """value and count arrays (NAMES order) of one approach row (a dict)."""
    v, n = np.full(len(NAMES), np.nan), np.zeros(len(NAMES))
    for name, (vc, nc) in meas.items():
        val, cnt = r.get(vc), r.get(nc)
        if val is None or cnt is None or not np.isfinite(val) or not np.isfinite(cnt) or cnt <= 0:
            continue
        v[IX[name]], n[IX[name]] = float(val), float(cnt)
    return v, n


def _ridge(cxx, cxy, k_mat, prior):
    """(cxx + K)^-1 (cxy + K prior) - a weighted slope pulled toward `prior`."""
    a = cxx + k_mat
    try:
        return np.linalg.solve(a, cxy + k_mat @ prior)
    except np.linalg.LinAlgError:
        return prior.copy()


def build(games, approach, params=CoachParams()):
    """One row per game (game_id plus home_/away_ coach readings), every
    value read before that game's kickoff."""
    p = params
    meas = measures(p)
    d = 0.5 ** (1.0 / p.half_life)
    dL = 0.5 ** (1.0 / p.league_half_life)
    k = p.k_mult * K_BASE

    rows_by_game = {}
    for r in approach.to_dict("records"):
        rows_by_game.setdefault(r["game_id"], {})[r["team"]] = r

    home_coach = na.coach_names(games["home_coach"], games["season"])
    away_coach = na.coach_names(games["away_coach"], games["season"])

    off, dfn = {}, {}               # key -> _State ; franchise -> _State
    hc_games, here_games = {}, {}   # coach -> n ; (coach, team) -> n
    # league: weighted raw means (shared), within-coach pass regression and payoff regression
    lnum, lden = np.zeros(len(NAMES)), np.zeros(len(NAMES))
    lreg_pass, lreg_pay = _Reg(2, 1), _Reg(1, 2)
    # league priors before any game: plain fixed constants (no look-ahead)
    L0 = np.array([0.48, 0.50, 0.45, 0.45, 31.5, 0.02, 0.15, 0.0, -0.1, -0.03, 0.43, 0.48])
    if p.pass_measure == "proe":
        L0[IX["pass"]] = L0[IX["early"]] = 0.0
    pending = []                    # (kickoff, [(v, n, reg_pass item, reg_pay item)])
    out = []

    def league():
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(lden > 0, lnum / np.where(lden > 0, lden, 1), L0)
        return m

    def league_slopes():
        xb, cxx, cxy = lreg_pass.centred()
        if xb is None or lreg_pass.w < 50:
            bp, vp = np.zeros((2, 1)), np.array([0.1 ** 2, 0.8 ** 2])
        else:
            bp = np.linalg.solve(cxx + 1e-6 * np.eye(2), cxy)
            vp = np.maximum(np.diag(cxx) / lreg_pass.w, 1e-6)
        xb2, cxx2, cxy2 = lreg_pay.centred()
        if xb2 is None or lreg_pay.w < 50:
            bq, vq = np.zeros((1, 2)), np.array([0.1 ** 2])
        else:
            bq = np.linalg.solve(cxx2 + 1e-6 * np.eye(1), cxy2)
            vq = np.maximum(np.diag(cxx2) / lreg_pay.w, 1e-6)
        return bp, vp, bq, vq

    for i, g in enumerate(games.itertuples(index=False)):
        while pending and pending[0][0] + nf.FINAL_AFTER <= g.kickoff:
            for v, n, rp, rq in pending.pop(0)[1]:
                ok = n > 0
                lnum[:] = lnum * dL
                lden[:] = lden * dL
                lnum[ok] += n[ok] * v[ok]
                lden[ok] += n[ok]
                if rp is not None:
                    lreg_pass.scale(dL)
                    lreg_pass.add(*rp)
                if rq is not None:
                    lreg_pay.scale(dL)
                    lreg_pay.add(*rq)
        L = league()
        bp, vp, bq, vq = league_slopes()
        Kp = p.k_slope * np.diag(vp)
        Kq = p.k_slope * np.diag(vq)
        spread_home = g.spread_line if np.isfinite(g.spread_line) else 0.0
        row = {"game_id": g.game_id}
        reads = {}
        for side, team, opp, coach, sp in (("home", g.home, g.away, home_coach[i], spread_home),
                                           ("away", g.away, g.home, away_coach[i], -spread_home)):
            key = coach if p.key == "coach" else team
            st = off.get(key)
            if st is None:
                st = off[key] = _State()
            st.roll(g.season, p.keep)
            ds = dfn.get(opp)
            if ds is None:
                ds = dfn[opp] = _State()
            ds.roll(g.season, p.keep)
            t_off, t_def = st.tilt(k), ds.tilt(k)
            rmp = t_def[IX["rush_epa"]] - t_def[IX["pass_epa"]]
            x = np.array([rmp, sp / 7.0])
            # the shift
            base = t_off[IX["pass"]]
            xb, cxx, cxy = st.reg_pass.centred()
            if p.slopes and xb is not None:
                xb_s = st.reg_pass.x / (st.reg_pass.w + p.k_slope)
                beta = _ridge(cxx, cxy, Kp, bp)[:, 0]
                xtilt = base + float(beta @ (x - xb_s))
            elif p.slopes:
                xtilt = base + float(bp[:, 0] @ x)
            else:
                xtilt = base
            # the payoff
            xb2, cxx2, cxy2 = st.reg_pay.centred()
            if xb2 is not None:
                a = st.reg_pay.y / (st.reg_pay.w + p.k_pay)
                xb2_s = st.reg_pay.x / (st.reg_pay.w + p.k_slope)
                b = _ridge(cxx2, cxy2, Kq, bq)[0] if p.slopes else np.zeros(2)
                pay = a + b * (rmp - xb2_s[0])
            else:
                pay = (bq[0] * rmp) if p.slopes else np.zeros(2)
            # the matchup
            Lnpr = L[IX["npr"]]
            pr = float(np.clip(Lnpr + xtilt, 0.05, 0.95))
            pass_m = L[IX["pass_epa"]] + t_off[IX["pass_epa"]] + t_def[IX["pass_epa"]]
            rush_m = L[IX["rush_epa"]] + t_off[IX["rush_epa"]] + t_def[IX["rush_epa"]]
            xepa = pr * pass_m + (1 - pr) * rush_m
            fit = xtilt * (pass_m - rush_m)
            n_hc = hc_games.get(coach, 0)
            n_here = here_games.get((coach, team), 0)
            row.update({
                f"{side}_cg_xtilt": xtilt, f"{side}_cg_base": base,
                f"{side}_cg_xepa": xepa, f"{side}_cg_fit": fit,
                f"{side}_cg_pay_epa": float(pay[0]), f"{side}_cg_pay_sr": float(pay[1]),
                f"{side}_cg_early": t_off[IX["early"]], f"{side}_cg_rz_rush": t_off[IX["rz_rush"]],
                f"{side}_cg_gz": t_off[IX["gz"]], f"{side}_cg_pace": t_off[IX["pace"]],
                f"{side}_cg_nohuddle": t_off[IX["nohuddle"]], f"{side}_cg_shotgun": t_off[IX["shotgun"]],
                f"{side}_cg_rmp_faced": rmp,
                f"{side}_cg_hc_games": n_hc, f"{side}_new_coach": int(n_hc < NEW_GAMES),
                f"{side}_new_here": int(n_here < NEW_GAMES),
            })
            reads[team] = (key, coach, opp, x, L.copy(), t_def.copy(), base, xb_s if (p.slopes and xb is not None) else np.zeros(2),
                           rmp, (xb2_s[0] if xb2 is not None else 0.0))
        out.append(row)

        if not g.played:
            continue
        mine = rows_by_game.get(g.game_id)
        if not mine or len(mine) != 2:
            continue
        folded = []
        for team in (g.home, g.away):
            r = mine.get(team)
            if r is None:
                continue
            key, coach, opp, x, Lr, t_def, base, xbar, rmp, xbar2 = reads[team]
            v, n = _row_values(r, meas)
            tilt = v - Lr
            ok = n > 0
            st = off[key]
            st.decay(d)
            st.num[ok] += n[ok] * tilt[ok]
            st.den[ok] += n[ok]
            rp = rq = None
            if ok[IX["pass"]]:
                w = n[IX["pass"]] / PASS_W
                y = np.array([tilt[IX["pass"]]])
                st.reg_pass.add(w, x, y)
                rp = (w, x - xbar, y - base)      # the league's slope is within-coach
            if ok[IX["epa"]] and ok[IX["sr"]]:
                w = n[IX["epa"]] / PLAYS_W
                y = np.array([v[IX["epa"]] - (Lr[IX["epa"]] + t_def[IX["epa"]]),
                              v[IX["sr"]] - (Lr[IX["sr"]] + t_def[IX["sr"]])])
                st.reg_pay.add(w, np.array([rmp]), y)
                rq = (w, np.array([rmp - xbar2]), y)
            ds = dfn[opp]          # the allowed() view of this row: the defence's own state
            ds.decay(d)
            ds.num[ok] += n[ok] * tilt[ok]
            ds.den[ok] += n[ok]
            hc_games[coach] = hc_games.get(coach, 0) + 1
            here_games[(coach, team)] = here_games.get((coach, team), 0) + 1
            folded.append((v, n, rp, rq))
        pending.append((g.kickoff, folded))

    f = pd.DataFrame(out)
    return derive(f)


DIFFS = ["xtilt", "base", "xepa", "fit", "pay_epa", "pay_sr", "early", "rz_rush", "gz", "pace",
         "nohuddle", "shotgun"]


def derive(f):
    for c in DIFFS:
        f[f"cg_{c}_diff"] = f[f"home_cg_{c}"] - f[f"away_cg_{c}"]
    f["cg_exp_diff"] = np.log1p(f["home_cg_hc_games"]) - np.log1p(f["away_cg_hc_games"])
    f["cg_pace_sum"] = f["home_cg_pace"] + f["away_cg_pace"]
    f["cg_xtilt_sum"] = f["home_cg_xtilt"] + f["away_cg_xtilt"]
    f["cg_xepa_sum"] = f["home_cg_xepa"] + f["away_cg_xepa"]
    f["cg_nohuddle_sum"] = f["home_cg_nohuddle"] + f["away_cg_nohuddle"]
    return f


FLAGS = ["home_new_coach", "away_new_coach", "home_new_here", "away_new_here", "cg_exp_diff"]
SETS = {
    # the plan's step 5: expected pass tilt, matchup EPA, go-zone, pace, new coach, experience
    "plan": (["cg_xtilt_diff", "cg_xepa_diff", "cg_gz_diff", "cg_pace_diff", "home_new_coach",
              "away_new_coach", "cg_exp_diff"],
             ["cg_pace_sum", "cg_xtilt_sum"]),
    # plan + the payoff and the fit (how much the expected approach benefits him)
    "plan_payoff": (["cg_xtilt_diff", "cg_xepa_diff", "cg_fit_diff", "cg_pay_epa_diff", "cg_pay_sr_diff",
                     "cg_gz_diff", "cg_pace_diff", "home_new_coach", "away_new_coach", "cg_exp_diff"],
                    ["cg_pace_sum", "cg_xtilt_sum", "cg_xepa_sum"]),
    # everything read
    "full": (["cg_xtilt_diff", "cg_xepa_diff", "cg_fit_diff", "cg_pay_epa_diff", "cg_pay_sr_diff",
              "cg_early_diff", "cg_rz_rush_diff", "cg_gz_diff", "cg_pace_diff", "cg_nohuddle_diff",
              "cg_shotgun_diff"] + FLAGS,
             ["cg_pace_sum", "cg_xtilt_sum", "cg_xepa_sum", "cg_nohuddle_sum"]),
    # the approach alone, nothing EPA-based
    "approach": (["cg_xtilt_diff", "cg_gz_diff", "cg_pace_diff", "cg_early_diff", "cg_rz_rush_diff"] + FLAGS,
                 ["cg_pace_sum", "cg_xtilt_sum"]),
    # the payoff alone
    "payoff": (["cg_xepa_diff", "cg_fit_diff", "cg_pay_epa_diff", "cg_pay_sr_diff"], ["cg_xepa_sum"]),
}


@contextmanager
def features(extra, extra_total):
    """nfl_features' lists with the candidate's columns added, for one call."""
    old, old_t = list(nf.FEATURES), list(nf.TOTAL_FEATURES)
    nf.FEATURES[:] = old + [c for c in extra if c not in old]
    nf.TOTAL_FEATURES[:] = old_t + [c for c in extra_total if c not in old_t]
    try:
        yield
    finally:
        nf.FEATURES[:] = old
        nf.TOTAL_FEATURES[:] = old_t


def frame_with(base, games, approach, params):
    cg = build(games, approach, params)
    return base.merge(cg, on="game_id", how="left", validate="one_to_one")


def run(base, games, approach, params=CoachParams(), fset="plan_payoff", end=None):
    """Walk-forward predictions of the shipped code with the candidate's features."""
    frame = frame_with(base, games, approach, params)
    extra, extra_t = SETS[fset]
    with features(extra, extra_t):
        preds = nm.walk_forward(frame, end=end).astype(float)
    return frame, preds


# ---------------------------------------------------------------- DEV metrics only

def _ll_each(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dev_mask(frame, *ps):
    m = bt.decided(frame) & frame["season"].between(*bt.DEV) & frame["market_prob"].notna()
    for p in ps:
        m &= p.notna()
    return m


def paired(frame, cand, ship, sel):
    y = (frame.loc[sel, "result"] > 0).astype(float).to_numpy()
    lc, ls = _ll_each(cand[sel], y), _ll_each(ship[sel], y)
    dlt = lc - ls
    acc_c = ((cand[sel].to_numpy() > 0.5) == (y == 1))
    acc_s = ((ship[sel].to_numpy() > 0.5) == (y == 1))
    seasons = frame.loc[sel, "season"].to_numpy()
    per, season_means = [], []
    for s in sorted(set(seasons)):
        m = seasons == s
        season_means.append(float(dlt[m].mean()))
        per.append({"season": int(s), "games": int(m.sum()), "ll": round(float(lc[m].mean()), 4),
                    "ll_shipped": round(float(ls[m].mean()), 4), "diff": round(float(dlt[m].mean()), 5),
                    "acc": round(float(acc_c[m].mean()), 3), "acc_shipped": round(float(acc_s[m].mean()), 3)})
    season_means = np.array(season_means)
    return {
        "games": int(sel.sum()), "ll": round(float(lc.mean()), 5), "ll_shipped": round(float(ls.mean()), 5),
        "diff": round(float(dlt.mean()), 5), "se": round(float(dlt.std(ddof=1) / math.sqrt(len(dlt))), 5),
        "se_by_season": round(float(season_means.std(ddof=1) / math.sqrt(len(season_means))), 5),
        "seasons_better": int((season_means < 0).sum()), "seasons": len(season_means),
        "acc": round(float(acc_c.mean()), 4), "acc_shipped": round(float(acc_s.mean()), 4),
        "acc_diff_games": int(acc_c.sum() - acc_s.sum()),
        "per_season": per,
    }


def dev_report(frame, preds, shipped):
    pc, ps = preds["blend_prob"], shipped["blend_prob"]
    sel = dev_mask(frame, pc, ps)
    y = (frame.loc[sel, "result"] > 0).astype(float)
    mkt = frame.loc[sel, "market_prob"]
    out = {"pick": paired(frame, pc, ps, sel),
           "model_alone": paired(frame, preds["model_prob"], shipped["model_prob"], sel),
           "market_ll": round(bt.log_loss(mkt, y), 5),
           "market_acc": round(float(((mkt > 0.5) == (y == 1)).mean()), 4)}
    for name, col, line, outcome in (("spread", "margin", "spread_line", "result"),
                                     ("total", "total_pred", "total_line", "total")):
        r = {}
        for lab, pr in (("candidate", preds[col]), ("shipped", shipped[col])):
            rep = bt.line_report(frame, pr, frame[line], frame[outcome], periods=(bt.DEV,))
            r[lab] = {e: v for e, v in next(iter(rep.values())).items()}
            s = frame["played"] & frame["season"].between(*bt.DEV) & pr.notna()
            r[lab]["mae"] = round(float((pr[s] - frame.loc[s, outcome]).abs().mean()), 4)
        out[name] = r
    return out


def summary(rep):
    p, m = rep["pick"], rep["model_alone"]
    return (f"pick ll {p['ll']:.5f} vs {p['ll_shipped']:.5f} (diff {p['diff']:+.5f} se {p['se']:.5f}, "
            f"{p['seasons_better']}/{p['seasons']} seasons better) acc {p['acc']:.4f} vs {p['acc_shipped']:.4f} | "
            f"model ll {m['ll']:.5f} vs {m['ll_shipped']:.5f} ({m['diff']:+.5f} se {m['se']:.5f}) "
            f"acc {m['acc']:.4f} vs {m['acc_shipped']:.4f}")


# ---------------------------------------------------------------- point in time

def scramble_test(games, team_games, qb_games, approach, params, fset, season=2011, week=10, seed=7):
    """At each kickoff of one DEV week: scramble every result, team-game and
    QB line and approach row not final at that kickoff (kickoff +
    FINAL_AFTER after it) and every price of a game kicking off after it,
    rebuild everything and re-run the walk-forward through `season`. The
    games kicking off then must read identical features, model, margin,
    total and pick. Control: a game two weeks on must change."""
    rng = np.random.default_rng(seed)
    full_base = nm.feature_frame(games, team_games, qb_games)
    full, full_p = run(full_base, games, approach, params, fset, end=season)
    wk = games[(games["season"] == season) & (games["week"] == week)]
    cols = nf.FEATURES + nf.TOTAL_FEATURES + SETS[fset][0] + SETS[fset][1] + [
        c for c in full.columns if c.startswith(("home_cg_", "away_cg_"))]
    cols = list(dict.fromkeys(cols))
    report = []
    for kick in sorted(wk["kickoff"].unique()):
        g = games.copy()
        later = g["kickoff"] + nf.FINAL_AFTER > kick
        after = g["kickoff"] > kick
        n = int(later.sum())
        idx = g.index[later]
        hs = rng.integers(0, 45, n)
        as_ = rng.integers(0, 45, n)
        g.loc[idx, "home_score"] = hs
        g.loc[idx, "away_score"] = as_
        g.loc[idx, "result"] = hs - as_
        g.loc[idx, "total"] = hs + as_
        g.loc[idx, "played"] = True
        aidx = g.index[after]
        for c in ("spread_line", "total_line", "home_ml", "away_ml"):
            g.loc[aidx, c] = rng.permutation(g.loc[aidx, c].to_numpy())
        late_ids = set(g.loc[later, "game_id"])
        tg = team_games.copy()
        m = tg["game_id"].isin(late_ids)
        for c in ("plays", "epa", "dropbacks", "passing_epa", "rushing_epa"):
            if c in tg:
                tg.loc[m, c] = rng.permutation(tg.loc[m, c].to_numpy())
        qg = qb_games.copy()
        m = qg["game_id"].isin(late_ids)
        for c in ("dropbacks", "epa"):
            qg.loc[m, c] = rng.permutation(qg.loc[m, c].to_numpy())
        ap = approach.copy()
        m = ap["game_id"].isin(late_ids)
        for c in na.STATS + ["spread"]:
            ap.loc[m, c] = rng.permutation(ap.loc[m, c].to_numpy())
        base = nm.feature_frame(g, tg, qg)
        fr, pr = run(base, g, ap, params, fset, end=season)
        at = full["kickoff"] == kick
        a = full.loc[at, ["game_id"] + cols].set_index("game_id").astype(float)
        b = fr.loc[fr["kickoff"] == kick, ["game_id"] + cols].set_index("game_id").loc[a.index].astype(float)
        pa = full_p.loc[at].set_index(full.loc[at, "game_id"])
        pb = pr.loc[fr["kickoff"] == kick].set_index(fr.loc[fr["kickoff"] == kick, "game_id"]).loc[pa.index]
        feat_gap = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if len(a) else 0.0
        nan_same = bool((a.isna().to_numpy() == b.isna().to_numpy()).all())
        pred_gap = float(np.nanmax(np.abs(pa.to_numpy() - pb.to_numpy())))
        # control: two weeks on, the same season
        nxt = full[(full["season"] == season) & (full["week"] == week + 2)]
        ctrl_a = nxt.set_index("game_id")[[c for c in cols if c.startswith(("home_cg_", "away_cg_"))]].astype(float)
        ctrl_b = fr.set_index("game_id").loc[ctrl_a.index, ctrl_a.columns].astype(float)
        ctrl = float(np.nanmax(np.abs(ctrl_a.to_numpy() - ctrl_b.to_numpy())))
        report.append({"kickoff": str(kick), "games": int(at.sum()), "scrambled_games": n,
                       "max_feature_gap": feat_gap, "nan_pattern_same": nan_same,
                       "max_prediction_gap": pred_gap, "control_gap_week_plus_2": ctrl})
    return report


# ---------------------------------------------------------------- the search (DEV only)

def search(base, games, approach, shipped, log=print):
    """Stage 1: feature sets at the default parameters. Stage 2: one sweep
    per parameter around the best, keeping each improvement (coordinate
    search, one pass). Every score is the DEV pick log loss."""
    tried = []

    def score(params, fset):
        t = time.time()
        frame, preds = run(base, games, approach, params, fset)
        rep = dev_report(frame, preds, shipped)
        tried.append({"params": asdict(params), "set": fset, "pick_ll": rep["pick"]["ll"],
                      "pick_diff": rep["pick"]["diff"], "pick_se": rep["pick"]["se"],
                      "pick_acc": rep["pick"]["acc"], "model_ll": rep["model_alone"]["ll"],
                      "model_acc": rep["model_alone"]["acc"]})
        log(f"  {fset:12s} {_short(params)}  {summary(rep)}  ({time.time() - t:.0f}s)")
        return rep["pick"]["ll"]

    best_p, best_set = CoachParams(), None
    best = float("inf")
    for fset in SETS:
        s = score(best_p, fset)
        if s < best:
            best, best_set = s, fset
    sweeps = [("half_life", [8.0, 32.0]), ("keep", [0.5, 1.0]), ("k_mult", [0.4, 2.5]),
              ("k_slope", [10.0, 160.0]), ("k_pay", [4.0, 64.0]), ("slopes", [False]),
              ("pass_measure", ["proe"]), ("gz", ["first"])]
    for name, values in sweeps:
        for v in values:
            cand = replace(best_p, **{name: v})
            s = score(cand, best_set)
            if s < best:
                best, best_p = s, cand
    return best_p, best_set, tried


def _short(p):
    d = CoachParams()
    diff = {k: v for k, v in asdict(p).items() if getattr(d, k) != v}
    return ",".join(f"{k}={v}" for k, v in diff.items()) or "defaults"


# The configuration the DEV search chose (main --search: 18 configurations,
# feature sets at the defaults, then one coordinate sweep per parameter).
CHOSEN = CoachParams(half_life=8.0, keep=1.0, k_mult=0.4, k_slope=10.0, gz="first")
CHOSEN_SET = "full"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--no-scramble", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out", type=Path, default=OUT, help="the predictions, every season (gzip csv)")
    args = ap.parse_args(argv)
    games, tg, qg = nd.load_games(), nd.load_team_games(), nd.load_qb_games()
    approach = na.load_approach()
    base = nm.feature_frame(games, tg, qg)
    shipped = nm.walk_forward(base).astype(float)
    result = {}
    if args.search:
        params, fset, tried = search(base, games, approach, shipped)
        result["search"] = {"tried": tried, "configs": len(tried)}
    else:
        params, fset = CHOSEN, CHOSEN_SET
    print(f"chosen: {fset} {asdict(params)}")
    frame, preds = run(base, games, approach, params, fset)
    rep = dev_report(frame, preds, shipped)
    print("DEV", summary(rep))
    team = replace(params, key="team")
    tframe, tpreds = run(base, games, approach, team, fset)
    trep = dev_report(tframe, tpreds, shipped)
    print("DEV team-keyed", summary(trep))
    paired_ct = paired(frame, preds["blend_prob"], tpreds["blend_prob"],
                       dev_mask(frame, preds["blend_prob"], tpreds["blend_prob"]))
    result.update({"params": asdict(params), "set": fset, "features": SETS[fset],
                   "dev": rep, "dev_team_keyed": trep,
                   "coach_minus_team_pick": {k: paired_ct[k] for k in ("diff", "se", "seasons_better", "seasons")}})
    # the L1 winner's weights on the candidate columns, for the last DEV fit
    extra, extra_t = SETS[fset]
    with features(extra, extra_t):
        sm = nm.fit_season(frame, bt.DEV[1], preds["model_prob"])
        coefs = dict(zip(nf.FEATURES, sm.winner[-1].coef_[0]))
    result["winner_coefs_fit_for_2015"] = {k: round(float(v), 4) for k, v in coefs.items()}
    if not args.no_scramble:
        result["scramble"] = scramble_test(games, tg, qg, approach, params, fset)
        for r in result["scramble"]:
            print("scramble", r)
    # predictions, every season (metrics are DEV only)
    keep = ["game_id", "season", "week", "game_type", "kickoff", "home", "away", "home_coach", "away_coach",
            "result", "total", "spread_line", "total_line", "market_prob"]
    csv = frame[keep + [c for c in frame.columns if c.startswith(("home_cg_", "away_cg_", "cg_"))
                        or c in ("home_new_coach", "away_new_coach", "home_new_here", "away_new_here")]].copy()
    for c in ("model_prob", "margin", "total_pred", "blend_prob"):
        csv[f"cand_{c}"] = preds[c].to_numpy()
        csv[f"shipped_{c}"] = shipped[c].to_numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv.to_csv(args.out, index=False, compression="gzip", float_format="%.6f")
    print(f"wrote {args.out} ({len(csv)} games)")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str))
    return result


if __name__ == "__main__":
    main()
