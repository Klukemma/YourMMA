"""EXPERIMENT - coaching in the touchdown model. Not shipped; nothing here
reaches nfl_td, the pick or the app. It builds the plan's step 6
(nfl_approach's MEASUREMENT PLAN) and measures it on the touchdown model's
development period, 2014-2019, only. 2020-2025 is predicted and written
out but never scored here: the confirmation period is opened once, by the
judge.

    python nfl/engine/experiments/coach_td.py OUT.csv.gz [report.json]   # chosen config, DEV report
    python nfl/engine/experiments/coach_td.py --grid                     # the DEV choice of parameters
    python nfl/engine/experiments/coach_td.py --scramble                 # the point-in-time test

THE QUESTION. A coach who runs it in near the goal line, against a
defence that gives up rushing touchdowns, should lift his backs and his
sneaking quarterback and lower his receivers. nfl_td hands a team's
market-implied touchdowns out with a fixed league split (RUSH_TD_SHARE,
0.34 rushing) and the player's carry and target shares. Here the split,
the red-zone trips and the position hand-out are the coach's, against
this defence.

THE STATE. One pass over the schedule in kickoff order, the engine's rule
(nfl_features.build_with_state):

  coach (own)    keyed by his name (nfl_approach.coach_names applied): the
                 offence rows of every game he was head coach, for any team.
                 key="team" keys the same numbers by the franchise instead -
                 the plan's baseline.
  defence (own)  keyed by the franchise: the same rows from the defence's
                 side (what it allowed), since its players stay when a
                 coach goes.
  player (own)   his share of his team's red-zone and goal-to-go carries
                 and targets (approach_players), weighted as nfl_td weights
                 his carry share (decay 0.25, offseason keep 0.5).
  league (SHARED) the same sums over every team-game, folded only once its
                 kickoff + nfl_features.FINAL_AFTER is at or before the
                 reading game's kickoff.

Each entity keeps exponentially weighted sums (numerator, denominator) per
rate - half-life in its own games, an offseason keep - and reads
value = (num + k x league) / (den + k), k = prior_games x the league's mean
denominator per team-game. A coach with no game on file reads the league
(new_coach = 1).

THE RATES (numerator / denominator, per team-game):
  rz_rush   red-zone designed runs / red-zone plays
  gtg_rush  goal-to-go runs / goal-to-go plays
  td_rush   rushing TDs / offensive TDs              (the split)
  rz_trip   red-zone trips / drives
  rz_conv   red-zone trips ending in a TD / trips
  td_drive  offensive TDs / drives
  drives    drives per game                          (additive)
  rush_epa, pass_epa  EPA per run, per dropback      (additive)
  pace      neutral seconds per snap                 (additive)
  by position (QB, RB, FB, WR, TE): his share of the team's rushing TDs and
  receiving TDs (box score, player_games), and of its red-zone carries and
  targets (approach_players). Position: the player's listing in the game
  (nfl_td.player_lines), else his most common one.

THE MATCHUP. For a rate, logit(expected) = logit(coach) + logit(defence
allowed) - logit(league); for an additive one, coach + defence - league.

THE SPLIT s - the expected rushing share of the team's offensive TDs:
  e     the matchup's td_rush (coach + defence allowed - league), as above
  fit   a binomial regression of the realised split on logit e_td_rush,
        e_rz_rush, e_gtg_rush, the league's split, the spread and the implied
        points, fitted walk-forward by season on team-games of SPLIT_FROM to
        S - 1 (split_fit)
  mkt   the same regression on the league's split, the spread and the implied
        points alone - the control, with no coach in it

THE FEATURES (beside nfl_td.FEATURES; T = implied points / 7):
  halves(s)  xtd_<s>, log_xtd_<s>: T x (s x carry share + (1 - s) x target
             share) - nfl_td's xtd with s in place of its fixed 0.34 - and
             log_xtd_<s>_rush, log_xtd_<s>_rec, the two halves, logged
  lifts      rush_lift, rec_lift: log(the matchup's expected share of the
             team's rushing / receiving TDs that go to his position / the
             league's); rz_car_lift, rz_tgt_lift: the same for red-zone
             carries and targets. A run-first goal-line coach whose QB
             sneaks lifts the QB's rush_lift and lowers the WRs'.
  COACH_FEATURES = halves(e) + lifts: the candidate, chosen on DEV over
             halves(fit) + lifts ("coach_fit") and the sets in TRIED.
  player family (PLAYER_RZ_FEATURES): his own weighted red-zone and goal-to-go
    carry and target shares, and xtd_rz = T x (s_league x rz carry share +
    (1 - s_league) x rz target share), logged - no coach in it
  both: the two, plus xtd_rz_e (the same with the matchup's s)
  TRIED and dropped on DEV (each made DEV no better or worse):
    lean_<pos>    logit(e) - logit(league s), on the player's position only
    rzlean_<pos>  the same for the expected red-zone rush rate
    td_ratio      log(expected TDs from drives x TD per drive / T)
    rz_trips_dev  log(expected red-zone trips (drives x trip rate) / league)
    rz_conv_dev   logit(expected TD per trip) - logit(league)
    new_coach, coach_games_log

THE MODEL. nfl_td's learner (StandardScaler + LogisticRegression,
C = nfl_td.LOGISTIC_C), walk-forward by season from nfl_td.TRAIN_FROM:
season S is predicted by a fit on 2013 to S - 1. The baseline is that very
learner on nfl_td.FEATURES on the same rows - the shipped model's numbers.

THE MEASUREMENT (DEV 2014-2019 only): log loss per player-game, paired
against the baseline (mean difference, SE naive and clustered by game),
by position; calibration (mean said against scored, and bins) by position;
each team-game's top three (each scored, one or more scored) and top pick;
the same feature set keyed by team (the plan's baseline); and at team-game
level, how well each s predicts the realised split (split_check).

POINT IN TIME. scramble_test() hides every result not final at one cut
(2018 week 10, the late Sunday window, whose 1 pm games are still being
played), scrambles those games' box score, snap counts and both
play-by-play files, rebuilds, and requires identical features for every
player-game of the week up to the cut, and different ones two weeks later.
"""

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import nfl_approach as na   # noqa: E402
import nfl_backtest as bt   # noqa: E402
import nfl_data as nd       # noqa: E402
import nfl_features as nf   # noqa: E402
import nfl_td as td         # noqa: E402

DEV = td.DEV
POS = tuple(td.POSITIONS)                 # QB, RB, FB, WR, TE (nfl_data.OFFENSE_POSITIONS)
REPORT_POS = ("RB", "WR", "TE", "QB", "FB")


@dataclass(frozen=True)
class CoachParams:
    half_life: float = 48.0       # in the entity's own games
    offseason_keep: float = 0.9   # share of the weight carried into a new season
    prior_games: float = 16.0     # k, in games' worth of the league's mean count
    league_half_life: float = 544.0   # team-games (about one season)
    key: str = "coach"            # "coach", or "team" (the plan's baseline keying)
    player_decay: float = 0.25    # as nfl_td.TdParams
    player_keep: float = 0.5


# ---------------------------------------------------------------- the rates

LOGIT_RATES = ["rz_rush", "gtg_rush", "td_rush", "rz_trip", "rz_conv", "td_drive"]
ADDITIVE = ["drives", "rush_epa", "pass_epa", "pace"]
SPLITS = ("rush_td", "rec_td", "rz_car", "rz_tgt")
SPLIT_RATES = [f"{s}_{p}" for s in SPLITS for p in POS]
RATES = LOGIT_RATES + ADDITIVE + SPLIT_RATES
IDX = {r: i for i, r in enumerate(RATES)}
LOGIT_SET = set(LOGIT_RATES) | set(SPLIT_RATES)
# What a league with no game yet reads (1999 week 1 only; by 2013 every
# league value is fifteen seasons of games).
DEFAULTS = {"rz_rush": 0.45, "gtg_rush": 0.55, "td_rush": 0.37, "rz_trip": 0.29, "rz_conv": 0.54,
            "td_drive": 0.21, "drives": 11.2, "rush_epa": -0.1, "pass_epa": 0.0, "pace": 31.7}
DEFAULT_SPLIT = {"rush_td": {"QB": 0.12, "RB": 0.82, "FB": 0.03, "WR": 0.02, "TE": 0.01},
                 "rec_td": {"QB": 0.0, "RB": 0.10, "FB": 0.02, "WR": 0.62, "TE": 0.26},
                 "rz_car": {"QB": 0.12, "RB": 0.82, "FB": 0.03, "WR": 0.02, "TE": 0.01},
                 "rz_tgt": {"QB": 0.0, "RB": 0.14, "FB": 0.02, "WR": 0.52, "TE": 0.32}}
EPS = 1e-4


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def team_game_vectors(approach, lines, players):
    """{(game_id, team): (num, den)} - the RATES of one offence in one game,
    as numerators and denominators. `lines` is nfl_td.player_lines();
    `players` is nfl_approach.load_players() with a `position` column."""
    a = approach
    n = len(a)
    num = np.zeros((n, len(RATES)))
    den = np.zeros((n, len(RATES)))
    f = lambda c: a[c].fillna(0).to_numpy(float)   # noqa: E731

    def put(rate, x, d):
        num[:, IDX[rate]], den[:, IDX[rate]] = x, d

    rz, gtg = f("rz_plays"), f("gtg_plays")
    put("rz_rush", f("rz_rush_rate") * rz, rz)
    put("gtg_rush", f("gtg_rush_rate") * gtg, gtg)
    tds = f("rush_td") + f("pass_td")
    put("td_rush", f("rush_td"), tds)
    put("rz_trip", f("rz_drives"), f("drives"))
    put("rz_conv", f("rz_td_drives"), f("rz_drives"))
    put("td_drive", tds, f("drives"))
    put("drives", f("drives"), np.ones(n))
    put("rush_epa", f("rush_epa_play") * f("rush_plays") * a["rush_epa_play"].notna(),
        f("rush_plays") * a["rush_epa_play"].notna())
    put("pass_epa", f("pass_epa_play") * f("pass_plays") * a["pass_epa_play"].notna(),
        f("pass_plays") * a["pass_epa_play"].notna())
    put("pace", f("pace_sec") * f("pace_snaps") * a["pace_sec"].notna(),
        f("pace_snaps") * a["pace_sec"].notna())

    key = pd.MultiIndex.from_frame(a[["game_id", "team"]])
    # Box score: each position's rushing and receiving TDs.
    bl = lines[lines["position"].isin(POS) & lines["team"].notna()]
    box = bl.groupby(["game_id", "team", "position"])[["rushing_tds", "receiving_tds"]].sum()
    for stat, col in (("rush_td", "rushing_tds"), ("rec_td", "receiving_tds")):
        wide = box[col].unstack("position").reindex(columns=list(POS)).fillna(0.0)
        wide = wide.reindex(key).fillna(0.0).to_numpy(float)
        total = wide.sum(axis=1)
        for j, p in enumerate(POS):
            put(f"{stat}_{p}", wide[:, j], total)
    pl = players[players["position"].isin(POS)]
    rzp = pl.groupby(["game_id", "team", "position"])[["rz_carries", "rz_targets"]].sum()
    for stat, col in (("rz_car", "rz_carries"), ("rz_tgt", "rz_targets")):
        wide = rzp[col].unstack("position").reindex(columns=list(POS)).fillna(0.0)
        wide = wide.reindex(key).fillna(0.0).to_numpy(float)
        total = wide.sum(axis=1)
        for j, p in enumerate(POS):
            put(f"{stat}_{p}", wide[:, j], total)
    return {k: (num[i], den[i]) for i, k in enumerate(zip(a["game_id"], a["team"]))}


def player_positions(players, lines):
    """approach_players with each row's position: his listing in the game's
    line (nfl_td.player_lines), else his most common listing, else None."""
    game_pos = lines.dropna(subset=["position"]).drop_duplicates(["game_id", "gsis_id"]).set_index(
        ["game_id", "gsis_id"])["position"]
    usual = lines.dropna(subset=["position"]).groupby("gsis_id")["position"].agg(
        lambda s: s.value_counts().index[0])
    out = players.copy()
    idx = pd.MultiIndex.from_frame(out[["game_id", "player_id"]])
    out["position"] = game_pos.reindex(idx).to_numpy()
    miss = out["position"].isna()
    out.loc[miss, "position"] = out.loc[miss, "player_id"].map(usual)
    return out


# ---------------------------------------------------------------- the state

class _Sums:
    """Exponentially weighted (numerator, denominator) per rate, in the
    entity's own games, cut to `keep` at a new season."""

    __slots__ = ("num", "den", "games", "season")

    def __init__(self):
        self.num = np.zeros(len(RATES))
        self.den = np.zeros(len(RATES))
        self.games = 0
        self.season = None

    def add(self, season, num, den, decay, keep):
        if self.season is not None and self.season != season:
            self.num *= keep
            self.den *= keep
        self.season = season
        self.num = (1 - decay) * self.num + num
        self.den = (1 - decay) * self.den + den
        self.games += 1


class _League:
    """The league's sums: SHARED, so a game's rows wait in `pending` until
    its kickoff + FINAL_AFTER is at or before the reading kickoff."""

    def __init__(self, decay):
        self.decay = decay
        self.num = np.zeros(len(RATES))
        self.den = np.zeros(len(RATES))
        self.n = 0.0
        self.pending = deque()
        d = np.zeros(len(RATES))
        for r, v in DEFAULTS.items():
            d[IDX[r]] = v
        for s, by in DEFAULT_SPLIT.items():
            for p, v in by.items():
                d[IDX[f"{s}_{p}"]] = v
        self.default = d

    def release(self, kickoff):
        while self.pending and self.pending[0][0] + nf.FINAL_AFTER <= kickoff:
            _, num, den = self.pending.popleft()
            self.num = (1 - self.decay) * self.num + num
            self.den = (1 - self.decay) * self.den + den
            self.n = (1 - self.decay) * self.n + 1.0

    def value(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            v = self.num / self.den
        return np.where(self.den > 0, v, self.default)

    def per_game(self):
        """The league's mean denominator per team-game (k's unit)."""
        if self.n <= 0:
            return np.ones(len(RATES))
        return np.maximum(self.den / self.n, 1e-3)


def _shrunk(sums, league_value, k):
    if sums is None:
        return league_value.copy()
    return (sums.num + k * league_value) / (sums.den + k)


_LOGIT_MASK = np.array([r in LOGIT_SET for r in RATES])


def _matchup(off, dfn, lg):
    """Expected rates of this offence against this defence: logit-additive
    for a rate, additive for the rest."""
    with np.errstate(over="ignore"):
        logit = _sigmoid(_logit(off) + _logit(dfn) - _logit(lg))
    return np.where(_LOGIT_MASK, logit, off + dfn - lg)


class _PlayerRz:
    __slots__ = ("share", "w", "season")

    def __init__(self):
        self.share = np.zeros(4)   # rz carries, rz targets, gtg carries, gtg targets
        self.w = 0.0
        self.season = None


TEAM_COLUMNS = (["game_id", "team", "coach_key", "new_coach", "coach_games_log"]
                + [f"e_{r}" for r in RATES] + [f"L_{r}" for r in RATES])
PLAYER_RZ_RAW = ["ew_rz_car_share", "ew_rz_tgt_share", "ew_gtg_car_share", "ew_gtg_tgt_share"]


def build_state(games, approach, players, lines, params=CoachParams()):
    """One kickoff-ordered pass. Returns (team-game frame: TEAM_COLUMNS, one
    row per side of every game; player frame: game_id, gsis_id and
    PLAYER_RZ_RAW for every snapped player line) - every value read BEFORE
    its game is folded in."""
    vec = team_game_vectors(approach, lines, players)
    coach_of = {(g, t): c for g, t, c in zip(approach["game_id"], approach["team"], approach["coach"])}
    # Player near-goal-line counts per team-game, and the team totals.
    pl = players.groupby(["game_id", "team", "player_id"])[
        ["rz_carries", "rz_targets", "gtg_carries", "gtg_targets"]].sum()
    pl_by = {}
    for (gid, team, pid), row in zip(pl.index, pl.to_numpy(float)):
        pl_by.setdefault((gid, team), {})[pid] = row
    on_field = {}
    for gid, gsis, team, snapped, stats in zip(lines["game_id"], lines["gsis_id"], lines["team"],
                                                lines["in_snaps"], lines["has_stats"]):
        if snapped or stats:
            on_field.setdefault(gid, []).append((gsis, team, bool(snapped)))

    decay = 1 - 0.5 ** (1 / params.half_life)
    league = _League(1 - 0.5 ** (1 / params.league_half_life))
    offences, defences, prz = {}, {}, {}
    team_rows, player_rows = [], []

    g = games.sort_values(["kickoff", "game_id"], kind="mergesort").copy()
    g["home_coach"] = na.coach_names(g["home_coach"], g["season"])
    g["away_coach"] = na.coach_names(g["away_coach"], g["season"])
    for row in g.itertuples(index=False):
        league.release(row.kickoff)
        lg = league.value()
        k = params.prior_games * league.per_game()
        sides = ((row.home, row.away, row.home_coach), (row.away, row.home, row.away_coach))
        for team, opp, coach in sides:
            key = coach if params.key == "coach" else team
            off = offences.get(key)
            e = _matchup(_shrunk(off, lg, k), _shrunk(defences.get(opp), lg, k), lg)
            games_hc = off.games if off is not None else 0
            team_rows.append([row.game_id, team, key, float(games_hc == 0), math.log1p(games_hc),
                              *e, *lg])
        for gsis, team, snapped in on_field.get(row.game_id, ()):
            if not snapped:
                continue
            s = prz.get(gsis)
            player_rows.append([row.game_id, gsis, *(s.share if s is not None else np.zeros(4))])
        if not row.played:
            continue
        # Fold the game in: own state now, the league once it is final.
        for team, opp, _ in sides:
            v = vec.get((row.game_id, team))
            if v is None:
                continue
            num, den = v
            name = coach_of[(row.game_id, team)]
            key = name if params.key == "coach" else team
            offences.setdefault(key, _Sums()).add(row.season, num, den, decay, params.offseason_keep)
            defences.setdefault(opp, _Sums()).add(row.season, num, den, decay, params.offseason_keep)
            league.pending.append((row.kickoff, num, den))
        for team, _, _ in sides:
            if (row.game_id, team) not in vec:
                continue
            counts = pl_by.get((row.game_id, team), {})
            tot = np.sum(list(counts.values()), axis=0) if counts else np.zeros(4)
            if not (tot > 0).any():
                continue
            seen = set()
            for gsis, t, _ in on_field.get(row.game_id, ()):
                if t != team or gsis in seen:
                    continue
                seen.add(gsis)
                c = counts.get(gsis, np.zeros(4))
                share = np.where(tot > 0, c / np.maximum(tot, 1), 0.0)
                s = prz.setdefault(gsis, _PlayerRz())
                if s.season is not None and s.season != row.season:
                    s.w *= params.player_keep
                s.season = row.season
                d = params.player_decay
                s.w = (1 - d) * s.w + d
                s.share += (d / s.w) * (share - s.share)
    teams = pd.DataFrame(team_rows, columns=TEAM_COLUMNS)
    plays = pd.DataFrame(player_rows, columns=["game_id", "gsis_id"] + PLAYER_RZ_RAW)
    return teams, plays.drop_duplicates(["game_id", "gsis_id"])


# ---------------------------------------------------------------- features

# The split of a team-game's TDs into rushing and receiving, s, three ways:
#   e    the matchup's expected rushing share of TDs (coach + defence - league)
#   fit  a binomial regression of the realised split on the coach's matchup
#        rates (TD split, red-zone and goal-to-go rush rates), the league's
#        split, the spread and the implied points - fitted walk-forward by
#        season on team-games of SPLIT_FROM to S - 1 (seasons before S are
#        final before any of S kicks off, so the SHARED rule holds)
#   mkt  the same regression on the league's split, the spread and the
#        implied points alone: the control with no coach in it
SPLIT_FROM = 2002
SPLIT_TERMS = {"fit": ["l_e_td_rush", "l_e_rz_rush", "l_e_gtg_rush", "l_L_td_rush", "spread", "implied"],
               "mkt": ["l_L_td_rush", "spread", "implied"]}
SPLIT_C = 1.0


def _split_frame(teams, games, approach):
    g = games[["game_id", "season", "home", "spread_line", "total_line"]]
    m = teams.merge(g, on="game_id", how="left")
    home = (m["team"] == m["home"]).to_numpy()
    m["spread"] = np.where(home, m["spread_line"], -m["spread_line"])
    m["spread"] = m["spread"].fillna(0.0)
    m["implied"] = [td.implied_points(t, s, h) for t, s, h in zip(m["total_line"], m["spread_line"], home)]
    for c in ("e_td_rush", "e_rz_rush", "e_gtg_rush", "L_td_rush"):
        m[f"l_{c}"] = _logit(m[c].to_numpy(float))
    return m.merge(approach[["game_id", "team", "rush_td", "pass_td"]], on=["game_id", "team"], how="left")


def split_fit(teams, games, approach, start=td.TRAIN_FROM):
    """teams with s_fit and s_mkt: each season's from a fit on the seasons
    before it (SPLIT_FROM on). Seasons before `start` read NaN."""
    m = _split_frame(teams, games, approach)
    n = (m["rush_td"] + m["pass_td"]).to_numpy(float)
    r = m["rush_td"].to_numpy(float)
    has = np.isfinite(n) & (n > 0)
    out = teams.copy()
    for tag in SPLIT_TERMS:
        out[f"s_{tag}"] = np.nan
    for season in range(start, int(m["season"].max()) + 1):
        test = (m["season"] == season).to_numpy()
        train = has & m["season"].between(SPLIT_FROM, season - 1).to_numpy()
        if not test.any() or not train.any():
            continue
        for tag, cols in SPLIT_TERMS.items():
            x = m.loc[train, cols].to_numpy(float)
            y = np.r_[np.ones(len(x)), np.zeros(len(x))]
            w = np.r_[r[train], n[train] - r[train]]
            est = LogisticRegression(C=SPLIT_C, max_iter=2000).fit(np.vstack([x, x]), y, sample_weight=w)
            out.loc[test, f"s_{tag}"] = est.predict_proba(m.loc[test, cols].to_numpy(float))[:, 1]
    return out


def _halves(tag):
    return [f"xtd_{tag}", f"log_xtd_{tag}", f"log_xtd_{tag}_rush", f"log_xtd_{tag}_rec"]


LIFTS = ["rush_lift", "rec_lift", "rz_car_lift", "rz_tgt_lift"]
COACH_FEATURES = _halves("e") + LIFTS            # the candidate (chosen on DEV; see TRIED)
PLAYER_RZ_FEATURES = PLAYER_RZ_RAW + ["xtd_rz", "log_xtd_rz"]
BOTH_EXTRA = ["xtd_rz_e", "log_xtd_rz_e"]
LEANS = [f"lean_{p}" for p in POS]
RZLEANS = [f"rzlean_{p}" for p in POS]
TEAM_LEVEL = ["td_ratio", "rz_trips_dev", "rz_conv_dev"]
NEW_COACH = ["new_coach", "coach_games_log"]
B = list(td.FEATURES)
FEATURE_SETS = {
    "base": B,
    "split_mkt": B + _halves("mkt"),                  # control: the split with no coach in it
    "split_e": B + _halves("e"),                      # the raw matchup split
    "split_fit": B + _halves("fit"),                  # the fitted coach split
    "coach": B + COACH_FEATURES,                      # the raw split + the coach's position hand-out
    "coach_fit": B + _halves("fit") + LIFTS,          # the same on the fitted split
    "player_rz": B + PLAYER_RZ_FEATURES,              # the player's own red-zone shares, no coach
    "both": B + COACH_FEATURES + PLAYER_RZ_FEATURES + BOTH_EXTRA,
}
# Tried on DEV on the way and dropped (each added to the raw split's xtd):
TRIED = {
    "lean": B + ["xtd_e", "log_xtd_e"] + LEANS,
    "rzlean": B + ["xtd_e", "log_xtd_e"] + LEANS + RZLEANS,
    "team_level": B + TEAM_LEVEL,
    "new_coach": B + NEW_COACH,
    "all_first_try": B + _halves("e") + LEANS + RZLEANS + LIFTS + TEAM_LEVEL + NEW_COACH,
}
SCRAMBLE_COLUMNS = (B + _halves("e") + _halves("fit") + _halves("mkt") + LIFTS + PLAYER_RZ_FEATURES
                    + BOTH_EXTRA + LEANS + RZLEANS + TEAM_LEVEL + NEW_COACH)


def features(base, teams, plays):
    """nfl_td's frame with the coach and player near-goal-line features.
    `teams` is split_fit()'s (build_state()'s team frame with s_fit, s_mkt)."""
    f = base.merge(teams, on=["game_id", "team"], how="left", validate="many_to_one")
    f = f.merge(plays, on=["game_id", "gsis_id"], how="left", validate="one_to_one")
    f[PLAYER_RZ_RAW] = f[PLAYER_RZ_RAW].fillna(0.0)
    T = f["implied"] / td.POINTS_PER_TD
    car, tgt = f["ew_carry_share"], f["ew_target_share"]
    for tag, s in (("e", f["e_td_rush"]), ("fit", f["s_fit"]), ("mkt", f["s_mkt"])):
        s = s.fillna(f["L_td_rush"])     # seasons before the split's first fit (never scored)
        f[f"xtd_{tag}"] = T * (s * car + (1 - s) * tgt)
        f[f"log_xtd_{tag}"] = np.log(f[f"xtd_{tag}"] + 0.01)
        f[f"log_xtd_{tag}_rush"] = np.log(T * s * car + 0.01)
        f[f"log_xtd_{tag}_rec"] = np.log(T * (1 - s) * tgt + 0.01)
    sl = f["L_td_rush"]
    lean = _logit(f["e_td_rush"]) - _logit(sl)
    rzlean = _logit(f["e_rz_rush"]) - _logit(f["L_rz_rush"])
    for p in POS:
        f[f"lean_{p}"] = lean * f[f"pos_{p}"]
        f[f"rzlean_{p}"] = rzlean * f[f"pos_{p}"]
    for stat, name in (("rush_td", "rush_lift"), ("rec_td", "rec_lift"), ("rz_car", "rz_car_lift"),
                       ("rz_tgt", "rz_tgt_lift")):
        e = sum(f[f"e_{stat}_{p}"] * f[f"pos_{p}"] for p in POS)
        lg = sum(f[f"L_{stat}_{p}"] * f[f"pos_{p}"] for p in POS)
        f[name] = np.log(np.clip(e, EPS, None) / np.clip(lg, EPS, None))
    xtd_team = f["e_drives"] * f["e_td_drive"]
    f["td_ratio"] = np.log(np.clip(xtd_team, 0.05, None) / T.clip(lower=0.5))
    trips, ltrips = f["e_drives"] * f["e_rz_trip"], f["L_drives"] * f["L_rz_trip"]
    f["rz_trips_dev"] = np.log(np.clip(trips, 0.05, None) / np.clip(ltrips, 0.05, None))
    f["rz_conv_dev"] = _logit(f["e_rz_conv"]) - _logit(f["L_rz_conv"])
    rc, rt = f["ew_rz_car_share"], f["ew_rz_tgt_share"]
    f["xtd_rz"] = T * (sl * rc + (1 - sl) * rt)
    f["log_xtd_rz"] = np.log(f["xtd_rz"] + 0.01)
    se = f["e_td_rush"]
    f["xtd_rz_e"] = T * (se * rc + (1 - se) * rt)
    f["log_xtd_rz_e"] = np.log(f["xtd_rz_e"] + 0.01)
    return f


def load_inputs():
    games = nd.load_games()
    lines = td.load_lines()
    approach = na.load_approach()
    players = player_positions(na.load_players(), lines)
    return games, lines, approach, players


def build(games, lines, approach, players, params=CoachParams(), base=None):
    if base is None:
        base = td.build(games, lines)
    teams, plays = build_state(games, approach, players, lines, params)
    return features(base, split_fit(teams, games, approach), plays)


# ---------------------------------------------------------------- the model

def _fit(train, cols):
    est = make_pipeline(StandardScaler(), LogisticRegression(C=td.LOGISTIC_C, max_iter=3000))
    est.fit(td._x(train, cols), train["scored"].astype(int).to_numpy())
    return est


def walk_forward(frame, sets, start=DEV[0], end=None):
    """Out-of-sample probability per feature set: season S from a fit on
    nfl_td.TRAIN_FROM to S - 1."""
    end = end or int(frame["season"].max())
    out = pd.DataFrame(np.nan, index=frame.index, columns=list(sets), dtype=float)
    ok = td.labelled(frame)
    for season in range(start, end + 1):
        test = frame["season"] == season
        train = frame[ok & frame["season"].between(td.TRAIN_FROM, season - 1)]
        if not test.any() or not len(train):
            continue
        for name in sets:
            cols = FEATURE_SETS[name] if name in FEATURE_SETS else TRIED[name]
            out.loc[test, name] = _fit(train, cols).predict_proba(td._x(frame[test], cols))[:, 1]
    return out


# ---------------------------------------------------------------- measurement

def _ll(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(frame, p_new, p_base, mask):
    """Mean per-player-game log loss difference (new - base) with its SE,
    naive and clustered by game (players of one game share its score)."""
    y = frame.loc[mask, "scored"].to_numpy(float)
    d = _ll(p_new[mask], y) - _ll(p_base[mask], y)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    g = pd.Series(d - mean).groupby(frame.loc[mask, "game_id"].to_numpy()).sum()
    G = len(g)
    se_c = float(math.sqrt((g ** 2).sum() * G / (G - 1)) / n)
    return {"rows": n, "games": G, "base_log_loss": float(_ll(p_base[mask], y).mean()),
            "new_log_loss": float(_ll(p_new[mask], y).mean()), "diff": mean, "se": se,
            "se_game": se_c, "z_game": mean / se_c if se_c > 0 else None}


def dev_report(frame, preds, new, base="base"):
    """DEV 2014-2019 only. `preds` columns: feature-set names."""
    sel = (td.labelled(frame) & frame["season"].between(*DEV)
           & preds[[new, base]].notna().all(axis=1)).to_numpy()
    pn, pb = preds[new].to_numpy(float), preds[base].to_numpy(float)
    y_all = frame["scored"].to_numpy(float)
    out = {"period": list(DEV), "new": new, "base": base, "all": paired(frame, pn, pb, sel)}
    pos = frame["position"].to_numpy()
    out["by_position"] = {}
    for p in REPORT_POS:
        m = sel & (pos == p)
        if not m.any():
            continue
        r = paired(frame, pn, pb, m)
        y = y_all[m]
        r.update(scored=float(y.mean()), base_said=float(pb[m].mean()), new_said=float(pn[m].mean()),
                 base_calibration=td.calibration(pb[m], y), new_calibration=td.calibration(pn[m], y))
        out["by_position"][p] = r
    rows = frame[sel]
    out["top3"] = {"base": td.top_picks(rows, pb[sel], 3), "new": td.top_picks(rows, pn[sel], 3)}
    out["top1"] = {"base": td.top_picks(rows, pb[sel], 1), "new": td.top_picks(rows, pn[sel], 1)}
    out["top3_changed"] = _top3_changes(rows, pb[sel], pn[sel])
    out["by_season"] = {int(s): paired(frame, pn, pb, sel & (frame["season"] == s).to_numpy())["diff"]
                        for s in range(DEV[0], DEV[1] + 1)}
    return out


def _top3_changes(rows, pb, pn):
    """Team-games whose top three differ, and the scoring of the players
    swapped in against those swapped out."""
    r = rows[["game_id", "team", "gsis_id", "scored"]].assign(pb=pb, pn=pn)
    rb = r.sort_values(["game_id", "team", "pb"], ascending=[True, True, False], kind="stable")
    rn = r.sort_values(["game_id", "team", "pn"], ascending=[True, True, False], kind="stable")
    tb = set(map(tuple, rb.groupby(["game_id", "team"], sort=False).head(3)[["game_id", "team", "gsis_id"]].to_numpy()))
    tn = set(map(tuple, rn.groupby(["game_id", "team"], sort=False).head(3)[["game_id", "team", "gsis_id"]].to_numpy()))
    scored = dict(zip(map(tuple, r[["game_id", "team", "gsis_id"]].to_numpy()), r["scored"]))
    ins, outs = tn - tb, tb - tn
    return {"team_games_changed": len({(g, t) for g, t, _ in ins}), "swapped": len(ins),
            "in_scored": float(np.mean([scored[k] for k in ins])) if ins else None,
            "out_scored": float(np.mean([scored[k] for k in outs])) if outs else None}


def split_check(frame_teams, approach, mask_seasons=DEV):
    """Team-game level: does the expected rushing share of TDs predict the
    realised split? Binomial log loss per TD of rushing TDs out of offensive
    TDs, on DEV team-games with a TD: the matchup's s, the fitted splits
    (split_fit), and the league's s. `frame_teams` is split_fit()'s."""
    a = approach[["game_id", "team", "season", "rush_td", "pass_td"]]
    m = frame_teams.merge(a, on=["game_id", "team"])
    m = m[m["season"].between(*mask_seasons) & ((m["rush_td"] + m["pass_td"]) > 0)]
    r, n = m["rush_td"].to_numpy(float), (m["rush_td"] + m["pass_td"]).to_numpy(float)

    def bll(s):
        s = np.clip(s, 1e-6, 1 - 1e-6)
        return float(-(r * np.log(s) + (n - r) * np.log(1 - s)).sum() / n.sum())

    real = r / n
    return {"team_games": len(m), "tds": int(n.sum()),
            "log_loss_per_td": {"matchup_e": bll(m["e_td_rush"].to_numpy()),
                                "fitted_coach_split": bll(m["s_fit"].to_numpy()),
                                "fitted_market_only": bll(m["s_mkt"].to_numpy()),
                                "league": bll(m["L_td_rush"].to_numpy())},
            "corr_with_realised_share": {c: float(np.corrcoef(m[c], real)[0, 1])
                                         for c in ("e_td_rush", "s_fit", "s_mkt")}}


# ---------------------------------------------------------------- scramble

def hide_results_from(games, lines, approach, players, cutoff, seed=0):
    """Every result not final at `cutoff` hidden: scores removed, played
    False, and those games' box score, snap counts, approach rows and
    approach player rows scrambled (nfl_td's test helper, plus the two
    play-by-play files). Who took a snap is kept, as there."""
    g = games.copy()
    later = g["kickoff"] + nf.FINAL_AFTER > cutoff
    gone = set(g.loc[later, "game_id"])
    g.loc[later, ["home_score", "away_score", "result", "total"]] = np.nan
    g.loc[later, "played"] = False
    rng = np.random.default_rng(seed)
    lines = lines.copy()
    hit = lines["game_id"].isin(gone)
    for col in td.STATS[:-1]:
        lines.loc[hit, col] = rng.integers(0, 5, hit.sum()).astype(float)
    lines.loc[hit, "tds"] = lines.loc[hit, "rushing_tds"] + lines.loc[hit, "receiving_tds"]
    lines.loc[hit, "scored"] = (lines.loc[hit, "tds"] >= 1).astype(float)
    snapped = hit & lines["in_snaps"]
    lines.loc[snapped, "offense_pct"] = rng.uniform(0.01, 1.0, snapped.sum())
    a = approach.copy()
    ah = a["game_id"].isin(gone)
    for c in na.COUNTS:
        a.loc[ah, c] = rng.integers(0, 15, ah.sum())
    for c in set(na.STATS) - set(na.COUNTS):
        a.loc[ah, c] = rng.uniform(-0.5, 1.0, ah.sum())
    a.loc[ah, "coach"] = rng.permutation(a.loc[ah, "coach"].to_numpy())
    p = players.copy()
    ph = p["game_id"].isin(gone)
    for c in na.PLAYER_STATS:
        p.loc[ph, c] = rng.integers(0, 4, ph.sum())
    return g, lines, a, p


def scramble_test(games, lines, approach, players, params, season=2018, week=10):
    """One cut: the late Sunday window of `season` `week` (its 1 pm games are
    still being played, so the shared-state rule is exercised). Every row of
    a game kicking off at or before the cut reads identical features with
    the future hidden; rows two weeks later must differ (the control)."""
    wk = games[(games["season"] == season) & (games["week"] == week)]
    sunday = wk[wk["kickoff"].dt.dayofweek == 6]
    slots = sorted(sunday["kickoff"].unique())
    cut = pd.Timestamp(slots[1])
    full = build(games, lines, approach, players, params)
    hg, hl, ha, hp = hide_results_from(games, lines, approach, players, cut)
    hidden = build(hg, hl, ha, hp, params)
    cols = SCRAMBLE_COLUMNS
    key = ["game_id", "gsis_id"]
    at = full[(full["kickoff"] <= cut) & (full["kickoff"] > cut - pd.Timedelta(days=7))].set_index(key)
    b = hidden.set_index(key).loc[at.index]
    diff = (at[cols].astype(float) - b[cols].astype(float)).abs().to_numpy()
    same = bool(np.nanmax(diff) <= 1e-12) and bool((at[cols].isna().to_numpy() == b[cols].isna().to_numpy()).all())
    at_cut = int((at["kickoff"] == cut).sum())
    later = full[(full["season"] == season) & (full["week"] == week + 2)].set_index(key)
    lb = hidden.set_index(key).loc[later.index]
    moved = {c: bool(not np.allclose(later[c].astype(float), lb[c].astype(float)))
             for c in ["log_xtd_e_rush", "log_xtd_fit_rec", "lean_RB", "rush_lift", "rz_conv_dev",
                       "ew_rz_car_share", "td_ratio"]}
    earlier_same_day = int(((sunday["kickoff"] < cut)).sum())
    return {"cut": str(cut), "rows_checked": len(at), "rows_at_cut": at_cut,
            "games_hidden_before_cut_same_day": earlier_same_day,
            "identical": same, "max_abs_diff": float(np.nanmax(diff)), "control_moved": moved}


# ---------------------------------------------------------------- run

def grid(games, lines, approach, players, base, keys=("coach", "team")):
    """The DEV choice of the state's parameters, for each keying, by the
    'coach' feature set's DEV log loss (2014-2019)."""
    out = []
    for key in keys:
        for hl in (16.0, 48.0):
            for keep in (0.6, 0.9):
                for k in (4.0, 16.0, 64.0):
                    p = CoachParams(half_life=hl, offseason_keep=keep, prior_games=k, key=key)
                    f = build(games, lines, approach, players, p, base=base)
                    pr = walk_forward(f, ["coach"], end=DEV[1])
                    sel = (td.labelled(f) & f["season"].between(*DEV)).to_numpy()
                    y = f.loc[sel, "scored"].to_numpy(float)
                    out.append({"key": key, "half_life": hl, "offseason_keep": keep, "prior_games": k,
                                "coach": bt.log_loss(pr["coach"].to_numpy()[sel], y)})
                    print(out[-1], flush=True)
    return out


# Chosen by grid() on DEV (the 'coach' set's 2014-2019 log loss; 12 settings
# per keying, all within 0.40284-0.40301): coach keying best at half-life 16,
# keep 0.9, k 16 (0.40284); team keying best at half-life 48, keep 0.6, k 16
# (0.40285), and 0.40286 at the coach's setting.
CHOSEN = CoachParams(half_life=16.0, offseason_keep=0.9, prior_games=16.0)
TEAM_BEST = CoachParams(half_life=48.0, offseason_keep=0.6, prior_games=16.0, key="team")


def main(argv):
    t0 = time.time()
    games, lines, approach, players = load_inputs()
    base = td.build(games, lines)
    if "--grid" in argv:
        res = grid(games, lines, approach, players, base)
        print(json.dumps(res, indent=1))
        return
    if "--scramble" in argv:
        print(json.dumps(scramble_test(games, lines, approach, players, CHOSEN), indent=1))
        return
    out_csv = Path(argv[0]) if argv else None
    rep_path = Path(argv[1]) if len(argv) > 1 else None
    f = build(games, lines, approach, players, CHOSEN, base=base)
    preds = walk_forward(f, list(FEATURE_SETS))               # every season 2014 on (the file)
    tried = walk_forward(f, list(TRIED), end=DEV[1])
    team_f = build(games, lines, approach, players, replace(CHOSEN, key="team"), base=base)
    team_best_f = build(games, lines, approach, players, TEAM_BEST, base=base)
    team_preds = walk_forward(team_f, ["coach"], end=DEV[1])["coach"]
    team_best_preds = walk_forward(team_best_f, ["coach"], end=DEV[1])["coach"]
    teams = split_fit(build_state(games, approach, players, lines, CHOSEN)[0], games, approach)
    teams_t = split_fit(build_state(games, approach, players, lines, TEAM_BEST)[0], games, approach)

    sel = (td.labelled(f) & f["season"].between(*DEV)).to_numpy()
    y = f.loc[sel, "scored"].to_numpy(float)
    rep = {"period": list(DEV), "params": CHOSEN.__dict__, "team_params": TEAM_BEST.__dict__,
           "features": {k: [c for c in v if c not in td.FEATURES] for k, v in {**FEATURE_SETS, **TRIED}.items()}}
    rep["dev_log_loss"] = {k: bt.log_loss(preds[k].to_numpy()[sel], y) for k in FEATURE_SETS}
    rep["dev_log_loss"].update({f"tried_{k}": bt.log_loss(tried[k].to_numpy()[sel], y) for k in TRIED})
    rep["dev_log_loss"]["coach_keyed_by_team_same_params"] = bt.log_loss(team_preds.to_numpy()[sel], y)
    rep["dev_log_loss"]["coach_keyed_by_team_best_params"] = bt.log_loss(team_best_preds.to_numpy()[sel], y)
    rep["vs_base"] = dev_report(f, preds, "coach")
    cols = {c: preds[c].to_numpy(float) for c in preds}
    cols["team_same"], cols["team_best"] = team_preds.to_numpy(float), team_best_preds.to_numpy(float)
    rep["paired"] = {f"{a} - {b}": paired(f, cols[a], cols[b], sel) for a, b in (
        ("coach", "base"), ("split_e", "base"), ("split_fit", "base"), ("split_mkt", "base"),
        ("coach_fit", "base"), ("player_rz", "base"), ("both", "base"),
        ("split_e", "split_mkt"), ("coach", "split_mkt"), ("coach", "split_e"), ("both", "coach"),
        ("coach", "team_same"), ("coach", "team_best"), ("team_best", "base"))}
    rep["split_check"] = {"coach": split_check(teams, approach), "team": split_check(teams_t, approach)}
    rep["runtime_seconds"] = round(time.time() - t0, 1)
    if out_csv is not None:
        keep = ["game_id", "season", "week", "game_type", "kickoff", "team", "opponent", "gsis_id", "name",
                "position", "scored", "implied", "coach_key", "new_coach", "e_td_rush", "L_td_rush",
                "s_fit", "s_mkt", "e_rz_rush", "L_rz_rush", "ew_carry_share", "ew_target_share",
                "xtd", "xtd_e", "rush_lift", "rec_lift", "rz_car_lift", "rz_tgt_lift",
                "ew_rz_car_share", "ew_rz_tgt_share"]
        w = f[keep].copy()
        w["prob_base"] = preds["base"].to_numpy()
        w["prob_coach"] = preds["coach"].to_numpy()
        for k in FEATURE_SETS:
            if k not in ("base", "coach"):
                w[f"prob_{k}"] = preds[k].to_numpy()
        w = w[w["season"] >= DEV[0]].sort_values(["kickoff", "game_id", "team", "gsis_id"], kind="mergesort")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        w.to_csv(out_csv, index=False, float_format="%.6f", compression="gzip")
        rep["file"] = {"path": str(out_csv), "rows": len(w), "seasons": [int(w["season"].min()), int(w["season"].max())]}
    text = json.dumps(rep, indent=1, default=float)
    if rep_path is not None:
        rep_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
