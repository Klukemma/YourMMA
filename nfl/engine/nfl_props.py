"""Player yardage: a pre-game projection of a player's passing, rushing and
receiving yards in a game, for every QB, RB, FB, WR and TE expected to play
offense. The app shows each team's key players as "proj. 248 pass yds",
"proj. 64 rush yds", "proj. 71 rec yds".

A projection is a MEDIAN: the player should go over it in about half his
games. Yardage is skewed - most receivers' games are short, a few are long -
so a projection of the mean sits above what he usually gains and he goes
under it far more often than over (a ridge fitted to raw yards went over in
38-39% of DEV rushing and receiving games). Every learner below is fitted
and judged for the median: mean absolute error, and the share of games the
player went over.

THE ROWS. One per (game, player) who took an offensive snap (2013 on), the
rows of nfl_td: snap counts define who played, joined to the box score by
nfl_td.player_lines(), with attempts, completions and passing yards added
from the same box-score line. A player who took a snap and has no box-score
line gained nothing. Which yards a player is projected for follows his
position for the game:

  passing    QB
  rushing    QB, RB, FB, WR
  receiving  RB, FB, WR, TE

The labels are the game's passing_yards, rushing_yards and receiving_yards.
Playoff games are included and flagged.

THE STATE. One pass over the schedule in kickoff order, as nfl_td. Each game
first reads every player's state as it stood BEFORE kickoff, then - once the
game is played and its box score is in - folds it in. (A game played whose
box score is late is not folded: its zeros would read as a game with no
yards.) What a player carries forward, from every game he played for any
team back to 1999 (snap shares from 2013):

  recent      attempts, completions, carries, targets and receptions, and
              passing, rushing and receiving yards, exponentially weighted
              over his recent games (decay 0.25, nfl_td's), with his
              offensive snap share and his share of his team's carries and
              targets weighted the same way;
  steadier    the same at decay 0.1, about a season's memory, so one big
              game moves it less;
  efficiency  yards per attempt, per carry and per target over a long
              memory, shrunk toward his position's rate;
  career      games played, games this season, and whether his last game
              was for another team.

Between seasons the weighted history keeps only offseason_keep of its
weight: the means stand until he plays, then his first games move them fast.

What the game adds: the team's implied points from the closing lines
(nfl_td.implied_points: home = (total + spread) / 2, away = (total - spread)
/ 2, spread_line being the home side's expected margin; DEFAULT_IMPLIED when
either is missing) and the opponent's - a team expected to trail throws
more - and those points handed out by his recent carry and target shares,
home or away, playoff or not. The player's position is his roster
listing for the game, known before kickoff. (The yards a team's offence
gained and its opponent's defence allowed, weighted over their recent games,
were tried and added nothing on DEV once the implied points were in.)

POINT IN TIME. Nothing from a game reaches its own row: rows are read before
the game is folded in, and a game is folded only once the schedule calls it
played. tests/test_nfl_props.py hides every result from a kickoff on - the
score, and the box score and snap counts scrambled - rebuilds, and requires
identical features for the players in the games starting then.

THE MODELS. Walk-forward by season, as nfl_td: season S is projected by
models fitted on every labelled row from TRAIN_FROM (2013) to S - 1 (2013
trains and is never scored). For each stat three projections are compared:

  mean    the player's own recent weighted mean of those yards (baseline);
  ridge   a ridge regression fitted to the square root of the yards
          (signed: a run can lose yards) and squared back - on that scale
          yardage is close to symmetric, so the fitted mean sits near the
          median; the counts go in as square roots too (RIDGE_FEATURES);
  hgb     sklearn's HistGradientBoostingRegressor with absolute-error loss,
          which fits the median directly.

The learner is chosen per stat on DEV (2014-2019) mean absolute error alone
and MODEL names the choices; report() recomputes them every run
(dev_choice) so a change shows. CONFIRM (2020-2025) was not looked at while
choosing. Each stat is measured on the players the number is for:

  passing    the quarterback who took most of his team's snaps - in effect
             the starter, the one the app shows
  rushing    RBs, QBs and WRs with recent weighted carries of at least RECENT
  receiving  RBs, WRs and TEs with recent weighted targets of at least RECENT

Rushing and receiving are fitted on every row of their positions (a
receiver who never runs learns to be projected for no rushing yards).
Passing is fitted on the quarterbacks who took most of their team's snaps:
fitted on every quarterback who took a snap, the kneel-downs and mop-up
series pulled the starters' projections down (DEV MAE 68 ridge, 62
boosting, against 57). So a quarterback's passing projection is his IF HE
STARTS.

    python nfl/engine/nfl_props.py [report.json]   # walk-forward and report

MEASURED (walk-forward, build 5 s and walk-forward 60 s; MAE in yards;
"over" is the share of games the player gained more than the chosen
model's projection - a median's is 50%):

                    DEV 2014-2019                 CONFIRM 2020-2025
             rows   own mean  model  over     rows   own mean  model  over
  passing   3,204     66.2    56.7   50.0%   3,386     67.5    56.2   48.6%   ridge
  rushing  11,333     18.6    17.7   49.6%  12,500     18.5    17.6   50.6%   ridge
  receiving 24,304    20.9    19.5   50.2%  25,840     19.8    18.3   49.2%   hgb

  - The player's own mean is the wrong number to show: long games pull it
    up, and he went over it in only 40% of his rushing and receiving games
    (both periods); a starter's passing mean is pulled down by the games he
    left early, and he went over it in 54-56%.
  - The two learners are close. Passing: ridge by 1.0 yard in both periods.
    Rushing: ridge by 0.06 on DEV, boosting by 0.09 on CONFIRM. Receiving:
    boosting by 0.003 yards on DEV (19.506 against 19.509), a tie well
    inside the season-to-season spread of the difference (sd 0.06), which
    the rule gives to the boosting; it was 0.08 better on CONFIRM.
  - The boosting's receiving projections stop near 90 yards (its largest
    leaf), so the very best receivers read a little low: the 50 DEV games it
    projected at 80+ went over 62% of the time. On DEV each team's three
    most projected receivers - the ones the app shows - went over 52.7% of
    the time (the ridge's 52.4%), at an MAE of 27.1 (27.2).
  - Passing projections of 275+ went over 40-43% of the time on CONFIRM
    against 46-49% on DEV: starters passed for less in 2020-2025 (232 yards
    a game against 247) and a model fitted on earlier seasons expects more.
    A shrink of every
    projection toward the middle, fitted on DEV, gained under 0.1 yard of
    MAE for any stat, so none is applied.
  - Over-shares by projection bin are in report(); the lowest rushing and
    receiving bin sits at 44-49% because most of those players gain 0-2
    yards, which a projection of 2.4 is above.

LIVE. For games still to be played:

    live = fit_live(season)       # MODEL fitted on seasons before `season`,
                                  # every player's state after the last game played
    project_team(live, "KC", implied_for, implied_against, home=True,
                 exclude={gsis ids ruled out or on reserve})
    -> [{"gsis_id", "name", "position", "pass_yds", "rush_yds", "rec_yds"}, ...]
       most total yards first; None where the stat does not apply
    project_player(live, gsis_id, "KC", implied_for, implied_against, home=True)
    -> one such dict, or None when he is not one of the team's candidates

implied_for and implied_against are implied_points() of the game's total
and home spread for the team and for its opponent. Passing yards go to one
quarterback, the expected starter (expected_starter(): the one named as
`starter`, else the one who took most of the team's snaps in its last game,
else the candidate with most recent attempts); rush_yds is None for a TE,
rec_yds for a QB. Candidates are
nfl_td's: the players whose most recent game was for the team this season
(last season, if it has not played yet) with weighted carries + targets +
pass attempts of at least MIN_USAGE, plus the quarterback who took most of
its snaps in its last game; `roster` (gsis ids) replaces the
most-recent-game rule - a player signed from another team counts, one
released does not - and the usage rule still applies.

EVERY PROJECTION IS CONDITIONAL ON THE PLAYER PLAYING. The models were
fitted only on players who took an offensive snap - the passing model only
on quarterbacks who took most of their team's snaps - so a player who is
inactive is outside them: exclude him, and read a questionable player's
number as "if he plays" (a quarterback's passing yards as "if he starts").
Nor do they know who else is out - a back whose starter is ruled out is
projected on his own usage so far.
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits   # ships with scikit-learn, a required dependency

import nfl_data as nd
import nfl_td as nt

DEV, CONFIRM, TRAIN_FROM = nt.DEV, nt.CONFIRM, nt.TRAIN_FROM
POSITIONS = nt.POSITIONS
DEFAULT_IMPLIED = nt.DEFAULT_IMPLIED
implied_points = nt.implied_points
MIN_USAGE = nt.MIN_USAGE   # weighted carries + targets + attempts a live candidate needs
RECENT = 1.0               # weighted carries (targets) that put a player in the rushing (receiving) measure

# stat -> the box-score column it projects, the positions it is projected
# for, the positions it is measured on, and its key in the live projection.
YARDS = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}
APPLIES = {"pass": ("QB",), "rush": ("QB", "RB", "FB", "WR"), "rec": ("RB", "FB", "WR", "TE")}
MEASURED_ON = {"pass": ("QB",), "rush": ("RB", "QB", "WR"), "rec": ("RB", "WR", "TE")}
LIVE_KEY = {"pass": "pass_yds", "rush": "rush_yds", "rec": "rec_yds"}

# Chosen on DEV (report()["dev_choice"] recomputes it). Passing and rushing
# by MAE. Receiving was a tie on MAE (hgb 19.506, ridge 19.509, against a
# season-to-season spread of 0.06), so it was decided on DEV calibration
# where the app's numbers live - the top receivers: ridge's 60-100 yard
# projections went over 48-49% of the time, while hgb never projects much
# above 90 and its 80+ projections went over 62% - it undersells exactly
# the players the card shows.
MODEL = {"pass": "ridge", "rush": "ridge", "rec": "ridge"}
LEARNERS = ("ridge", "hgb")
BASELINES = ("mean",)
KINDS = BASELINES + LEARNERS
RIDGE_ALPHA = 10.0
# early_stopping off: a fixed 200 trees every season, rather than a count
# that moves with a random 10% holdout of each season's rows; on DEV the two
# scored the same.
HGB = dict(loss="absolute_error", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
           min_samples_leaf=100, l2_regularization=1.0, early_stopping=False, random_state=0)


@dataclass(frozen=True)
class PropsParams:
    decay: float = 0.25           # EWMA weight of a player's newest game (nfl_td's)
    steady_decay: float = 0.1     # the same, for the steadier means
    offseason_keep: float = 0.5   # share of its weight that history keeps into a new season
    eff_decay: float = 0.05       # per-game decay of the efficiency memory
    eff_prior_n: float = 40.0     # pseudo-attempts (carries, targets) behind an efficiency


# A box-score line as a player's weighted history takes it, in this order.
STATS = ("attempts", "completions", "passing_yards", "carries", "rushing_yards",
         "targets", "receptions", "receiving_yards")
_I = {s: i for i, s in enumerate(STATS)}
_EFF = [_I[s] for s in ("attempts", "passing_yards", "carries", "rushing_yards", "targets",
                        "receiving_yards")]
# Yards per attempt, carry and target by position (2013-2019 rows): the
# priors a player's own efficiency is shrunk toward.
YPA_PRIOR = {"QB": 7.2}
YPC_PRIOR = {"QB": 4.2, "RB": 4.2, "FB": 3.2, "WR": 6.1, "TE": 5.0}
YPT_PRIOR = {"QB": 3.8, "RB": 6.0, "FB": 5.9, "WR": 7.8, "TE": 7.3}
YPA_OTHER, YPC_OTHER, YPT_OTHER = 7.2, 4.2, 7.3

_MEANS = list(STATS) + ["snap_pct", "carry_share", "target_share"]
PLAYER_FEATURES = ([f"ew_{s}" for s in _MEANS] + [f"sw_{s}" for s in _MEANS]
                   + ["ypa", "ypc", "ypt", "games", "games_season", "new_team"])
GAME_FEATURES = ["implied", "opp_implied", "home", "playoff"]
RAW_FEATURES = [f"pos_{q}" for q in POSITIONS] + PLAYER_FEATURES + GAME_FEATURES
# The counts and yards, and the implied points handed out by carry and
# target share, go to the ridge as square roots only: its target is on that
# scale, so roots line up with it, and a raw count beside its root let the
# fit bend upward at the top - DEV's 60+ yard receiving projections went
# over 46% of the time against 48% with roots alone, its 275+ passing ones
# 47% against 49%, at the same MAE to within 0.1 yard.
ROOTED = [f"{w}_{s}" for w in ("ew", "sw") for s in STATS] + ["x_rush", "x_rec"]
FEATURES = RAW_FEATURES + ["x_rush", "x_rec"] + [f"rt_{c}" for c in ROOTED]   # every pre-game column
RIDGE_FEATURES = [c for c in FEATURES if c not in ROOTED]
HGB_FEATURES = RAW_FEATURES   # a tree is indifferent to roots and products of what it has
LABELS = list(YARDS.values())
META = ["game_id", "season", "week", "game_type", "kickoff", "team", "opponent", "gsis_id",
        "name", "position", "started"]


# ---------------------------------------------------------------- the rows

def prop_lines(player_games, snaps, player_ids):
    """nfl_td.player_lines() - every (game, player) who took an offensive
    snap or recorded a stat - with attempts, completions and passing_yards
    from the same box-score line (0 without one)."""
    lines = nt.player_lines(player_games, snaps, player_ids)
    passing = player_games[["game_id", "player_id", "attempts", "completions", "passing_yards"]].rename(
        columns={"player_id": "gsis_id"})
    lines = lines.merge(passing, on=["game_id", "gsis_id"], how="left")
    for c in ("attempts", "completions", "passing_yards"):
        lines[c] = lines[c].fillna(0).astype(float)
    return lines


def load_lines():
    """prop_lines() of the snapshot on file."""
    return prop_lines(nd.load_player_games(), nd.load_snaps(), nd.load_player_ids())


# ---------------------------------------------------------------- the state

class _Means:
    """A player's box-score lines, snap share and team shares, exponentially
    weighted at one decay. Kept as (value, weight), as nfl_td._Player, so
    cutting the weight at a new season leaves the value alone and only makes
    the next games count for more."""

    __slots__ = ("decay", "stats", "w", "snap", "snap_w", "share", "share_w")

    def __init__(self, decay):
        self.decay = decay
        self.stats = np.zeros(len(STATS))
        self.w = self.snap = self.snap_w = self.share_w = 0.0
        self.share = np.zeros(2)

    def offseason(self, keep):
        self.w *= keep
        self.snap_w *= keep
        self.share_w *= keep

    def add(self, stats, pct, share):
        d = self.decay
        self.w = (1 - d) * self.w + d
        self.stats += (d / self.w) * (stats - self.stats)
        if pct == pct:
            self.snap_w = (1 - d) * self.snap_w + d
            self.snap += (d / self.snap_w) * (pct - self.snap)
        if share is not None:
            self.share_w = (1 - d) * self.share_w + d
            self.share += (d / self.share_w) * (share - self.share)

    def read(self):
        return [*self.stats, self.snap, self.share[0], self.share[1]]


class _Player:
    """One player's history: recent and steadier means, a long efficiency
    memory, and where and how often he has played."""

    __slots__ = ("name", "position", "team", "season", "games", "season_games",
                 "recent", "steady", "eff")

    def __init__(self, p):
        self.name = self.position = self.team = self.season = None
        self.games = self.season_games = 0
        self.recent, self.steady = _Means(p.decay), _Means(p.steady_decay)
        self.eff = np.zeros(len(_EFF))   # attempts, passing yds, carries, rushing yds, targets, receiving yds

    def add(self, season, name, team, position, pct, stats, team_carries, team_targets, p):
        """Fold one game in. `pct` is NaN where there is no snap count."""
        if self.season != season:
            if self.season is not None:
                self.recent.offseason(p.offseason_keep)
                self.steady.offseason(p.offseason_keep)
            self.season, self.season_games = season, 0
        share = None
        if team_carries > 0 or team_targets > 0:
            c, t = stats[_I["carries"]], stats[_I["targets"]]
            share = np.array([c / team_carries if team_carries > 0 else 0.0,
                              t / team_targets if team_targets > 0 else 0.0])
        self.recent.add(stats, pct, share)
        self.steady.add(stats, pct, share)
        self.eff = (1 - p.eff_decay) * self.eff + stats[_EFF]
        self.games += 1
        self.season_games += 1
        self.name, self.team = name, team
        if isinstance(position, str):
            self.position = position

    def read(self, season, team, position, p):
        """PLAYER_FEATURES as he carries them into a game in `season` for
        `team`. Reading changes nothing."""
        n = p.eff_prior_n
        att, pyd, car, ryd, tgt, cyd = self.eff
        return [*self.recent.read(), *self.steady.read(),
                (pyd + YPA_PRIOR.get(position, YPA_OTHER) * n) / (att + n),
                (ryd + YPC_PRIOR.get(position, YPC_OTHER) * n) / (car + n),
                (cyd + YPT_PRIOR.get(position, YPT_OTHER) * n) / (tgt + n),
                math.log1p(self.games), self.season_games if self.season == season else 0,
                float(self.team != team)]

    def usage(self):
        """Recent weighted carries + targets + pass attempts."""
        s = self.recent.stats
        return s[_I["carries"]] + s[_I["targets"]] + s[_I["attempts"]]


def _debut(position):
    """PLAYER_FEATURES of a player with no history."""
    return ([0.0] * (2 * len(_MEANS))
            + [YPA_PRIOR.get(position, YPA_OTHER), YPC_PRIOR.get(position, YPC_OTHER),
               YPT_PRIOR.get(position, YPT_OTHER), 0.0, 0, 0.0])


# A team's last season played and the quarterback with most of its snaps in
# its last game: nfl_td's, kept the same way.
_Team = nt._Team


def _derive(frame):
    """Columns computed from the others: the team's implied points handed
    out by the player's carry and target shares, and square roots for the
    ridge."""
    frame["x_rush"] = frame["implied"] * frame["ew_carry_share"]
    frame["x_rec"] = frame["implied"] * frame["ew_target_share"]
    for c in ROOTED:
        frame[f"rt_{c}"] = np.sqrt(frame[c].clip(lower=0))
    return frame


def build(games, lines, params=PropsParams()):
    """One row per (game, player) who took an offensive snap: META, pre-game
    FEATURES, and the yardage LABELS (NaN until the game is played and its
    box score is in). `lines` is prop_lines()."""
    return build_with_state(games, lines, params)[0]


def build_with_state(games, lines, params=PropsParams()):
    """build(), plus every player's and team's state after the last game
    played - what the live projection reads."""
    with_stats = set(lines.loc[lines["has_stats"], "game_id"])
    stats = lines[list(STATS)].to_numpy(float)
    label_cols = [_I[c] for c in LABELS]
    by_game = {}
    cols = ["game_id", "gsis_id", "name", "team", "position", "offense_pct", "in_snaps"]
    for i, (gid, gsis, name, team, pos, pct, snapped) in enumerate(zip(
            *(lines[c].tolist() for c in cols))):
        by_game.setdefault(gid, []).append(
            (gsis, name, team, pos, float(pct) if snapped else np.nan, bool(snapped), i))

    players, teams = {}, {}
    rows, meta, labels = [], [], []
    no_label = [np.nan] * len(LABELS)
    for g in games.itertuples(index=False):
        game_lines = by_game.get(g.game_id)
        if not game_lines:
            continue
        side = {g.home: (1.0, g.away), g.away: (0.0, g.home)}
        implied = {t: implied_points(g.total_line, g.spread_line, side[t][0] == 1.0) for t in side}
        playoff = float(g.game_type != "REG")
        labelled = bool(g.played) and g.game_id in with_stats
        # The quarterback who took most of each side's snaps: whom the
        # passing measure is about. It reads this game's snap counts, so it
        # is a label, never a feature.
        starter = {}
        for gsis, _, team, pos, pct, snapped, _ in game_lines:
            if snapped and pos == "QB" and pct > starter.get(team, (None, -1.0))[1]:
                starter[team] = (gsis, pct)
        for gsis, name, team, pos, pct, snapped, i in game_lines:
            if not snapped or team not in side:
                continue
            home, opp = side[team]
            pl = players.get(gsis)
            f = pl.read(g.season, team, pos, params) if pl else _debut(pos)
            rows.append([float(pos == q) for q in POSITIONS] + f
                        + [implied[team], implied[opp], home, playoff])
            meta.append((g.game_id, g.season, g.week, g.game_type, g.kickoff, team, opp, gsis,
                         name, pos, float(starter.get(team, (None,))[0] == gsis)))
            labels.append(stats[i, label_cols] if labelled else no_label)
        if not labelled:
            continue
        # Fold the game in.
        totals = {}
        for _, _, team, _, _, _, i in game_lines:
            totals[team] = totals.get(team, 0.0) + stats[i]
        for gsis, name, team, pos, pct, snapped, i in game_lines:
            pl = players.get(gsis)
            if pl is None:
                pl = players[gsis] = _Player(params)
            t = totals[team]
            pl.add(g.season, name, team, pos, pct, stats[i], t[_I["carries"]], t[_I["targets"]], params)
        for team in side:
            t = teams.setdefault(team, _Team())
            t.season = g.season
            if team in starter:
                t.qb = starter[team][0]

    meta = pd.DataFrame(meta, columns=META)
    frame = pd.concat([meta, pd.DataFrame(rows, columns=RAW_FEATURES, dtype=float),
                       pd.DataFrame(np.array(labels, float).reshape(-1, len(LABELS)), columns=LABELS)],
                      axis=1)
    return _derive(frame), {"players": players, "teams": teams, "params": params}


# ---------------------------------------------------------------- the model

def _x(frame, cols):
    return frame[cols].astype(float).fillna(0.0).to_numpy()


class _OwnMean:
    """The baseline: the player's own recent weighted mean of the yards."""

    def fit(self, x, y):
        return self

    def predict(self, x):
        return x[:, 0]


class _RootRidge:
    """A ridge regression fitted to signed square-root yards and squared
    back, so that what it fits sits near the median, not the mean."""

    def __init__(self):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))

    def fit(self, x, y):
        self.pipe.fit(x, np.sign(y) * np.sqrt(np.abs(y)))
        return self

    def predict(self, x):
        r = self.pipe.predict(x)
        return np.sign(r) * r * r


@dataclass
class PropModel:
    """A fitted learner for one stat and the columns it reads."""
    stat: str
    kind: str
    estimator: object
    columns: list
    trained_on: int

    def predict(self, rows):
        """Projected yards, never below zero: a median projection of a loss
        is not one anyone would use, and only kneel-down quarterbacks and
        receivers who never run sit there."""
        with threadpool_limits(1, user_api="openmp"):
            return np.clip(self.estimator.predict(_x(rows, self.columns)), 0.0, None)


def applies(frame, stat):
    """Rows whose position the stat is projected for."""
    return frame["position"].isin(APPLIES[stat])


def measured(frame, stat):
    """Rows the stat is measured on: the starting quarterback for passing,
    the players with recent carries (targets) for rushing (receiving)."""
    pos = frame["position"].isin(MEASURED_ON[stat])
    if stat == "pass":
        return pos & (frame["started"] == 1)
    if stat == "rush":
        return pos & (frame["ew_carries"] >= RECENT)
    return pos & (frame["ew_targets"] >= RECENT)


def trains_on(frame, stat):
    """Rows the stat's learners are fitted on: the starting quarterback for
    passing, every row of the positions for rushing and receiving."""
    if stat == "pass":
        return applies(frame, stat) & (frame["started"] == 1)
    return applies(frame, stat)


def labelled(frame):
    return frame[LABELS].notna().all(axis=1)


def fit(train, stat, kind=None):
    """One learner ('mean', 'ridge' or 'hgb'; MODEL's by default) for one
    stat, on the labelled rows of `train` that stat trains on."""
    kind = kind or MODEL[stat]
    train = train[labelled(train) & trains_on(train, stat)]
    if kind == "mean":
        est, cols = _OwnMean(), [f"ew_{YARDS[stat]}"]
    elif kind == "ridge":
        est, cols = _RootRidge(), RIDGE_FEATURES
    elif kind == "hgb":
        est, cols = HistGradientBoostingRegressor(**HGB), HGB_FEATURES
    else:
        raise ValueError(f"unknown learner {kind!r}")
    # One OpenMP thread: the boosting's threads spin while they wait, and on
    # a machine with other work running, a fit that takes half a second on
    # one thread was measured taking over three minutes on four.
    with threadpool_limits(1, user_api="openmp"):
        est.fit(_x(train, cols), train[YARDS[stat]].to_numpy(float))
    return PropModel(stat, kind, est, list(cols), len(train))


def walk_forward(frame, start=DEV[0], end=None, kinds=KINDS, stats=tuple(YARDS)):
    """Out-of-sample projections, season by season: one column per
    "<stat>_<kind>", plus "<stat>", MODEL's. Season S's come from rows of
    TRAIN_FROM to S - 1; a row whose position a stat does not apply to stays
    NaN in that stat's columns."""
    end = end or int(frame["season"].max())
    out = pd.DataFrame(np.nan, index=frame.index, columns=[f"{s}_{k}" for s in stats for k in kinds],
                       dtype=float)
    for season in range(start, end + 1):
        in_season = frame["season"] == season
        train = frame[frame["season"].between(TRAIN_FROM, season - 1)]
        if not in_season.any() or not labelled(train).any():
            continue
        for stat in stats:
            test = in_season & applies(frame, stat)
            if test.any():
                for kind in kinds:
                    out.loc[test, f"{stat}_{kind}"] = fit(train, stat, kind).predict(frame[test])
    for stat in stats:
        if f"{stat}_{MODEL[stat]}" in out:
            out[stat] = out[f"{stat}_{MODEL[stat]}"]
    return out


# ---------------------------------------------------------------- measurement

BIN_EDGES = {
    "pass": (0, 175, 200, 225, 250, 275, 300, math.inf),
    "rush": (0, 5, 10, 20, 30, 45, 60, 80, math.inf),
    "rec": (0, 5, 10, 20, 30, 45, 60, 80, math.inf),
}


def over_by_bin(proj, actual, edges):
    """The share of games the player went over his projection, by how big
    the projection was."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (proj >= lo) & (proj < hi)
        if m.sum():
            rows.append({"lo": lo, "hi": None if hi == math.inf else hi, "rows": int(m.sum()),
                         "projected": float(proj[m].mean()), "actual": float(actual[m].mean()),
                         "median_actual": float(np.median(actual[m])),
                         "over": float((actual[m] > proj[m]).mean())})
    return rows


def report(frame, preds):
    """Everything measured about the yardage projections, JSON-able. `preds`
    is walk_forward(frame).

    Per period and stat: rows; the mean and median yards gained; for each
    learner its MAE, median absolute error, mean projection, and the share
    of games the player went over it (and hit it exactly - a projection of
    0 met by a game of 0); and for MODEL's the over-share by projection bin.
    """
    out = {"model": dict(MODEL), "features": {"ridge": RIDGE_FEATURES, "hgb": HGB_FEATURES},
           "recent": RECENT,
           "periods": {"dev": list(DEV), "confirm": list(CONFIRM), "train_from": TRAIN_FROM},
           "measured_on": {s: list(MEASURED_ON[s]) for s in YARDS}}
    for lo, hi in (DEV, CONFIRM):
        period = {}
        for stat, col in YARDS.items():
            kinds = [k for k in KINDS if f"{stat}_{k}" in preds]
            if not kinds:
                continue
            sel = (labelled(frame) & frame["season"].between(lo, hi) & measured(frame, stat)
                   & preds[[f"{stat}_{k}" for k in kinds]].notna().all(axis=1))
            if not sel.any():
                continue
            y = frame.loc[sel, col].to_numpy(float)
            p = {k: preds.loc[sel, f"{stat}_{k}"].to_numpy(float) for k in kinds}
            err = {k: np.abs(p[k] - y) for k in kinds}
            chosen = MODEL[stat] if MODEL[stat] in p else kinds[0]
            period[stat] = {
                "rows": int(sel.sum()),
                "mean_actual": float(y.mean()),
                "median_actual": float(np.median(y)),
                "mae": {k: float(err[k].mean()) for k in kinds},
                "median_ae": {k: float(np.median(err[k])) for k in kinds},
                "mean_projection": {k: float(p[k].mean()) for k in kinds},
                "over": {k: float((y > p[k]).mean()) for k in kinds},
                "push": {k: float((y == p[k]).mean()) for k in kinds},
                "over_by_bin": over_by_bin(p[chosen], y, BIN_EDGES[stat]),
            }
        out[f"{lo}-{hi}"] = period
    out["dev_choice"] = {}
    for stat, r in out[f"{DEV[0]}-{DEV[1]}"].items():
        learners = {k: r["mae"][k] for k in LEARNERS if k in r["mae"]}
        out["dev_choice"][stat] = dict(learners, prefers=min(learners, key=learners.get) if learners else None)
    return out


def summary(rep):
    """The report as the few lines worth reading in a terminal."""
    lines = ["Yardage projections - model: " + ", ".join(
        f"{s} {k} (DEV prefers {rep['dev_choice'].get(s, {}).get('prefers')})" for s, k in rep["model"].items())]
    for key in (f"{DEV[0]}-{DEV[1]}", f"{CONFIRM[0]}-{CONFIRM[1]}"):
        for stat, r in rep.get(key, {}).items():
            lines.append(f"\n  {key} {YARDS[stat]}: {r['rows']:,} player-games, mean {r['mean_actual']:.1f}, "
                         f"median {r['median_actual']:.1f}")
            lines.append(f"    {'':6s} {'MAE':>7s} {'MedAE':>7s} {'mean proj':>9s} {'over':>7s} {'push':>6s}")
            for k in r["mae"]:
                lines.append(f"    {k:6s} {r['mae'][k]:7.2f} {r['median_ae'][k]:7.2f} "
                             f"{r['mean_projection'][k]:9.1f} {r['over'][k]:7.1%} {r['push'][k]:6.1%}")
            lines.append(f"    over its projection, by projection ({rep['model'][stat]}):")
            for b in r["over_by_bin"]:
                span = f"{b['lo']}+" if b["hi"] is None else f"{b['lo']}-{b['hi']}"
                lines.append(f"      {span:8s} proj {b['projected']:6.1f}  actual mean {b['actual']:6.1f}"
                             f" median {b['median_actual']:6.1f}  over {b['over']:6.1%}  ({b['rows']:,})")
    return "\n".join(lines)


# ---------------------------------------------------------------- live

@dataclass
class PropsLive:
    """What the weekly run projects from: MODEL fitted for each stat on
    every season before `season`, and the state after every game played."""
    season: int
    models: dict
    state: dict


def fit_live(season=None, games=None, lines=None, built=None):
    """The live models for `season` (default: the season of the next game
    still to be played). `built` - build_with_state()'s result - saves a
    rebuild when the caller has one."""
    if games is None and (built is None or season is None):
        games = nd.load_games()
    if built is None:
        built = build_with_state(games, load_lines() if lines is None else lines)
    frame, state = built
    if season is None:
        future = games.loc[~games["played"], "season"]
        season = int(future.min()) if len(future) else int(games["season"].max())
    train = frame[frame["season"].between(TRAIN_FROM, season - 1)]
    return PropsLive(season, {s: fit(train, s) for s in YARDS}, state)


def candidates(live, team, roster=None, min_usage=MIN_USAGE):
    """[(gsis_id, _Player)] who may play for `team` in a game of
    live.season: nfl_td.candidates' rule, with pass attempts counted as
    usage."""
    players, season = live.state["players"], live.season
    t = live.state["teams"].get(team) or _Team()
    if roster is not None:
        pool = [(g, players[g]) for g in dict.fromkeys(roster) if g in players]
    else:
        recent = season if t.season == season else season - 1
        pool = [(g, pl) for g, pl in players.items() if pl.team == team and pl.season == recent]
    return [(g, pl) for g, pl in pool
            if pl.usage() >= min_usage or (g == t.qb and pl.position == "QB")]


def expected_starter(live, team, pool, starter=None):
    """The gsis id of the quarterback whose passing yards are projected:
    `starter` if he is in `pool`, else the one who took most of the team's
    snaps in its last game if he is, else the pool's quarterback with most
    recent weighted pass attempts (None without one)."""
    qbs = {g: pl for g, pl in pool if pl.position == "QB"}
    t = live.state["teams"].get(team) or _Team()
    for g in (starter, t.qb):
        if g in qbs:
            return g
    if not qbs:
        return None
    return max(qbs, key=lambda g: qbs[g].recent.stats[_I["attempts"]])


def _project(live, pool, team, implied_for, implied_against, home, playoff, starter):
    """The projection dicts of `pool`, in its order; passing yards for
    `starter` only."""
    if not pool:
        return []
    p = live.state["params"]
    implied_for, implied_against = (DEFAULT_IMPLIED if v is None or v != v else float(v)
                                    for v in (implied_for, implied_against))
    game = [implied_for, implied_against, float(bool(home)), float(bool(playoff))]
    rows = _derive(pd.DataFrame(
        [[float(pl.position == q) for q in POSITIONS] + pl.read(live.season, team, pl.position, p) + game
         for _, pl in pool], columns=RAW_FEATURES, dtype=float))
    proj = {s: live.models[s].predict(rows) for s in YARDS}
    out = []
    for j, (gsis, pl) in enumerate(pool):
        c = {"gsis_id": gsis, "name": pl.name, "position": pl.position}
        for s, key in LIVE_KEY.items():
            c[key] = float(proj[s][j]) if pl.position in APPLIES[s] else None
        if gsis != starter:
            c["pass_yds"] = None
        out.append(c)
    return out


class _Debut:
    """A named starter with no NFL history - a rookie's first start: read as
    build() reads a debut row, so the live projection is the backtest's."""

    def __init__(self, name, position="QB"):
        self.name, self.position = name, position

    def read(self, season, team, position, p):
        return _debut(position)


def project_team(live, team, implied_for, implied_against, home, exclude=(), roster=None,
                 playoff=False, starter=None, min_usage=MIN_USAGE, starter_name=None):
    """Every candidate's projected (median) passing, rushing and receiving
    yards in `team`'s next game, IF HE PLAYS. Most total yards first;
    unrounded (the app rounds for display).

    -> [{"gsis_id", "name", "position", "pass_yds", "rush_yds", "rec_yds"}]

    rush_yds is None for a TE, rec_yds for a QB. pass_yds is given to one
    quarterback, the expected starter (expected_starter(): `starter` if
    named, else the one who took most of the team's snaps in its last game,
    else the candidate with most recent pass attempts; a named starter with
    no history - a rookie's first start - is projected as a debut, never
    replaced), and is his yards IF HE STARTS - the passing model was fitted on starters only, so it has
    nothing to say about a backup's mop-up series. It is None for everyone
    else.

    implied_for, implied_against: the team's and its opponent's expected
    points (implied_points() of the game's total and home spread; None
    reads as DEFAULT_IMPLIED); home: bool; exclude: gsis ids ruled out, on
    reserve or otherwise not playing (an excluded starter hands pass_yds
    to the next quarterback); roster: gsis ids to draw candidates from
    instead of the most-recent-game rule.
    """
    exclude = set(exclude or ())
    pool = [(g, pl) for g, pl in candidates(live, team, roster, min_usage) if g not in exclude]
    # A NAMED STARTER STARTS. He is put in the pool whatever the roster and
    # usage rules say - with his history if he has one, as a debut if not -
    # so his yards are never handed to a backup the card does not name.
    if starter is not None and starter not in exclude:
        if starter not in {g for g, _ in pool}:
            pl = live.state["players"].get(starter) or _Debut(starter_name or starter)
            pool.append((starter, pl))
        qb = starter
    else:
        qb = expected_starter(live, team, pool, None)
    out = _project(live, pool, team, implied_for, implied_against, home, playoff, qb)
    out.sort(key=lambda c: -sum(c[k] or 0.0 for k in LIVE_KEY.values()))
    return out


def project_player(live, gsis_id, team, implied_for, implied_against, home, exclude=(), roster=None,
                   playoff=False, starter=None, min_usage=MIN_USAGE):
    """One player's projection exactly as project_team() gives it - IF HE
    PLAYS; pass_yds only if he is the expected starter (name him `starter`
    for a quarterback's yards if he starts) - or None when he is not one of
    `team`'s candidates or is excluded."""
    got = [c for c in project_team(live, team, implied_for, implied_against, home, exclude, roster,
                                   playoff, starter, min_usage) if c["gsis_id"] == gsis_id]
    return got[0] if got else None


if __name__ == "__main__":
    t0 = time.time()
    frame = build(nd.load_games(), load_lines())
    t1 = time.time()
    preds = walk_forward(frame)
    rep = report(frame, preds)
    rep["runtime_seconds"] = {"build": round(t1 - t0, 1), "walk_forward": round(time.time() - t1, 1)}
    print(summary(rep))
    print(f"\n  build {rep['runtime_seconds']['build']}s, walk-forward {rep['runtime_seconds']['walk_forward']}s")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(rep, indent=1))
