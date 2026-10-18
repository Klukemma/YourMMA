"""Anytime touchdown: the chance a player scores at least one rushing or
receiving touchdown in a game, for every QB, RB, FB, WR and TE expected to
play offense. The app shows each team's three most likely scorers.

THE ROWS. One per (game, player) who took an offensive snap (snaps.csv,
offense_snaps > 0, 2013 on). Snap counts are keyed by Pro Football
Reference ids and the box score by gsis ids: the two are joined through
player_ids, and where that map has no id - or names a player with no box
score line in the game - by exact display name, team and game. A player
who took snaps and has no box-score line touched nothing and scored
nothing. The label is rushing_tds + receiving_tds >= 1; return,
fumble-recovery and passing touchdowns do not count. Playoff games are
included and flagged.

THE STATE. One pass over the schedule in kickoff order, as in
nfl_features. Each game first reads every player's state as it stood
BEFORE kickoff, then - if the game has been played - folds its box score
and snap counts in. What a player carries forward (from every game he
played, for any team, back to 1999; snap shares from 2013):

  usage      carries, targets, receptions, rushing and receiving yards,
             exponentially weighted over his recent games, and his share of
             his team's carries and targets in those games;
  scoring    rushing and receiving TDs and the share of games he scored in,
             weighted the same way, plus a slow TD-per-touch shrunk toward
             his position's rate;
  snaps      offensive snap share, weighted the same way;
  last four  plain means of carries, targets, TDs and snap share;
  career     games played, games this season, and whether his last game
             was for another team.

Between seasons the weighted history keeps only offseason_keep of its
weight: the means are untouched until he plays, then his first games of
the new season move them quickly.

What the game adds: the team's implied points from the closing lines
(home = (total + spread) / 2, away = (total - spread) / 2, spread_line
being the home side's expected margin; DEFAULT_IMPLIED when either is
missing), home or away, playoff or not, and xtd - those points turned into
expected TDs and handed out by the player's carry and target shares. The
player's position is his roster listing for the game, known before kickoff.
(Team and opponent TD habits and days of rest were tried and added nothing
on DEV once the implied points were in.)

POINT IN TIME. Nothing from a game reaches its own row: rows are read
before the game is folded in, and a game is folded only once the schedule
calls it played. tests/test_nfl_td.py hides every result from a kickoff on
- the score, and the box score and snap counts scrambled - rebuilds, and
requires identical features for the players in the games starting then.
Rows of a game not yet played, or whose box score is not in, carry no label.

THE MODEL. Walk-forward by season, as nfl_model: season S is predicted by
a model fitted on every labelled row from TRAIN_FROM to S - 1 (2013 trains
and is never scored). Two learners - a regularised logistic regression on
FEATURES and sklearn's HistGradientBoostingClassifier - were compared on
DEV log loss alone and MODEL names the winner; report() recomputes the
comparison every run (dev_choice) so a change shows. Both are reported
beside two baselines: the position's base rate, and a logistic regression
on usage alone (weighted carries, targets and snap share).

    python nfl/engine/nfl_td.py [report.json]   # walk-forward and report

LIVE. For games still to be played:

    live = fit_live(season)       # MODEL fitted on seasons before `season`,
                                  # every player's state after the last game played
    predict_team(live, "KC", implied_points(total, spread, home=True), home=True,
                 exclude={gsis ids ruled out or on reserve})
    -> [{"gsis_id", "name", "position", "prob", "carries", "targets", "snap_pct",
         "tds_season", "games_season", "last_team"}, ...]   highest prob first

Candidates are the players whose most recent game was for that team this
season (last season, if the team has not played yet this season) with
weighted carries + targets of at least MIN_USAGE, plus the quarterback who
took most of the team's snaps in its last game. `roster` (gsis ids) replaces
the most-recent-game rule - a player signed from another team counts, one
released does not - and the usage rule still applies.

EVERY PROBABILITY IS CONDITIONAL ON THE PLAYER PLAYING. The model was
fitted only on players who took an offensive snap, so a player who is
inactive is outside it: exclude him, and read a questionable player's
number as "if he plays". Nor does it know who else is out - a back whose
starter is ruled out is priced on his own usage so far.
"""

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nfl_backtest as bt
import nfl_data as nd

DEV = (2014, 2019)
CONFIRM = (2020, 2025)
TRAIN_FROM = nd.FIRST_SNAP_SEASON   # 2013: trains, never scored
DEFAULT_IMPLIED = 22.0              # a team's points when the market posted no line
MIN_USAGE = 1.0                     # weighted carries + targets a live candidate needs
POSITIONS = nd.OFFENSE_POSITIONS

MODEL = "logistic"                  # chosen on DEV log loss; see report()["dev_choice"]
LEARNERS = ("logistic", "hgb")
BASELINES = ("usage", "base")
LOGISTIC_C = 0.05
HGB = dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=7, min_samples_leaf=200,
           l2_regularization=1.0, random_state=0)


@dataclass(frozen=True)
class TdParams:
    decay: float = 0.25               # EWMA weight of a player's newest game
    offseason_keep: float = 0.5       # share of its weight that history keeps into a new season
    slow_decay: float = 0.05          # per-game decay of the TD-per-touch memory
    td_rate_prior_n: float = 30.0     # pseudo-touches behind a player's TD-per-touch
    last_n: int = 4


# A box-score line as a player's weighted history takes it, in this order.
STATS = ("carries", "targets", "receptions", "rushing_tds", "receiving_tds",
         "rushing_yards", "receiving_yards", "scored")
# TDs per carry + target by position (2013-2019 rows): the prior a
# player's own rate is shrunk toward.
TD_RATE_PRIOR = {"QB": 0.039, "RB": 0.029, "FB": 0.046, "WR": 0.047, "TE": 0.057}
TD_RATE_OTHER = 0.04
# Share of these players' TDs scored rushing (2013-2019), and points per
# offensive TD: the market's points become rushing and receiving TDs.
RUSH_TD_SHARE = 0.34
POINTS_PER_TD = 7.0

PLAYER_FEATURES = (
    [f"ew_{s}" for s in STATS]
    + ["ew_snap_pct", "ew_carry_share", "ew_target_share",
       "l4_carries", "l4_targets", "l4_tds", "l4_snap_pct",
       "td_per_touch", "games", "games_season", "new_team"]
)
GAME_FEATURES = ["implied", "home", "playoff"]
RAW_FEATURES = [f"pos_{q}" for q in POSITIONS] + PLAYER_FEATURES + GAME_FEATURES
LOGGED = ["ew_carries", "ew_targets", "ew_receptions", "ew_rushing_yards", "ew_receiving_yards",
          "l4_carries", "l4_targets"]
FEATURES = RAW_FEATURES + ["xtd", "log_xtd"] + [f"log_{c}" for c in LOGGED]
USAGE_FEATURES = ["ew_carries", "ew_targets", "ew_snap_pct"]
META = ["game_id", "season", "week", "game_type", "kickoff", "team", "opponent", "gsis_id",
        "name", "position", "tds"]


def implied_points(total_line, spread_line, home):
    """A team's expected points from the game's total and the HOME side's
    expected margin; DEFAULT_IMPLIED when either is missing."""
    try:
        total, spread = float(total_line), float(spread_line)
    except (TypeError, ValueError):
        return DEFAULT_IMPLIED
    if not (math.isfinite(total) and math.isfinite(spread)):
        return DEFAULT_IMPLIED
    return (total + spread) / 2 if home else (total - spread) / 2


# ---------------------------------------------------------------- the rows

def player_lines(player_games, snaps, player_ids):
    """Every (game, player) who took an offensive snap or recorded a stat:
    the snap counts joined to the box score.

    Columns: game_id, gsis_id (or "pfr:<id>" when no gsis id can be
    found), name, team, position, offense_pct (NaN without a snap line),
    in_snaps (took an offensive snap), has_stats (has a box-score line),
    the STATS counts (0 without a box-score line), tds.
    """
    pg = player_games[["game_id", "player_id", "player_display_name", "team", "position",
                       "carries", "targets", "receptions", "rushing_tds", "receiving_tds",
                       "rushing_yards", "receiving_yards"]].rename(columns={
        "player_id": "gsis_id", "player_display_name": "pg_name", "team": "pg_team",
        "position": "pg_position"})
    sn = snaps[(snaps["offense_snaps"] > 0) & snaps["position"].isin(POSITIONS)][
        ["game_id", "player", "pfr_player_id", "position", "team", "offense_snaps",
         "offense_pct"]].copy()
    ids = player_ids.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id")
    sn["gsis_id"] = sn["pfr_player_id"].map(ids.set_index("pfr_id")["gsis_id"])

    # The id map can be silent (no gsis id) or wrong (a gsis id with no box
    # score line in this game); then an exact name on the same team in the
    # same game, when it is unique, is the same player.
    have = pd.MultiIndex.from_frame(pg[["game_id", "gsis_id"]])
    miss = ~pd.MultiIndex.from_frame(sn[["game_id", "gsis_id"]].fillna("")).isin(have)
    unique = pg.drop_duplicates(["game_id", "pg_team", "pg_name"], keep=False)[
        ["game_id", "pg_team", "pg_name", "gsis_id"]].rename(columns={"gsis_id": "by_name"})
    named = sn.loc[miss, ["game_id", "team", "player"]].reset_index().merge(
        unique, left_on=["game_id", "team", "player"], right_on=["game_id", "pg_team", "pg_name"])
    sn.loc[named["index"], "gsis_id"] = named["by_name"].to_numpy()
    sn["gsis_id"] = sn["gsis_id"].fillna("pfr:" + sn["pfr_player_id"])

    lines = sn.merge(pg, on=["game_id", "gsis_id"], how="outer", indicator=True)
    # A whole game's snap counts can carry the wrong team (Super Bowl XLIX
    # has the sides swapped): trust the box score's team where the two meet,
    # and move the rest of that side with it.
    both = lines[lines["_merge"] == "both"]
    side = (both.groupby(["game_id", "team", "pg_team"]).size().reset_index(name="n")
            .sort_values("n", ascending=False, kind="stable").drop_duplicates(["game_id", "team"])
            .rename(columns={"pg_team": "side"})[["game_id", "team", "side"]])
    lines = lines.merge(side, on=["game_id", "team"], how="left")
    lines["team"] = lines["pg_team"].fillna(lines["side"]).fillna(lines["team"])
    lines["name"] = lines["player"].fillna(lines["pg_name"])
    lines["position"] = lines["position"].fillna(lines["pg_position"])
    lines["in_snaps"] = lines["offense_snaps"].fillna(0) > 0
    lines["has_stats"] = lines["_merge"] != "left_only"
    for s in STATS[:-1]:
        lines[s] = lines[s].fillna(0).astype(float)
    lines["tds"] = lines["rushing_tds"] + lines["receiving_tds"]
    lines["scored"] = (lines["tds"] >= 1).astype(float)
    keep = ["game_id", "gsis_id", "name", "team", "position", "offense_pct", "in_snaps",
            "has_stats", *STATS, "tds"]
    return lines[keep].reset_index(drop=True)


def load_lines():
    """player_lines() of the snapshot on file."""
    return player_lines(nd.load_player_games(), nd.load_snaps(), nd.load_player_ids())


# ---------------------------------------------------------------- the state

class _Player:
    """One player's history. Weighted means are kept as (value, weight), as
    nfl_features._Ewma, so cutting the weight at a new season leaves the
    value alone and only makes the next games count for more."""

    __slots__ = ("name", "position", "team", "season", "games", "season_games",
                 "ew", "ew_w", "snap", "snap_w", "share", "share_w", "recent", "recent_snap",
                 "season_tds", "slow_tds", "slow_touches")

    def __init__(self, n_last):
        self.name = self.position = self.team = self.season = None
        self.games = self.season_games = 0
        self.ew = np.zeros(len(STATS))
        self.ew_w = self.snap = self.snap_w = self.share_w = 0.0
        self.share = np.zeros(2)
        self.recent = deque(maxlen=n_last)
        self.recent_snap = deque(maxlen=n_last)
        self.season_tds = self.slow_tds = self.slow_touches = 0.0

    def add(self, season, name, team, position, pct, stats, team_carries, team_targets, p):
        """Fold one game in. `pct` is NaN where there is no snap count."""
        if self.season != season:
            if self.season is not None:
                self.ew_w *= p.offseason_keep
                self.snap_w *= p.offseason_keep
                self.share_w *= p.offseason_keep
            self.season, self.season_games, self.season_tds = season, 0, 0.0
        d = p.decay
        self.ew_w = (1 - d) * self.ew_w + d
        self.ew += (d / self.ew_w) * (stats - self.ew)
        if pct == pct:
            self.snap_w = (1 - d) * self.snap_w + d
            self.snap += (d / self.snap_w) * (pct - self.snap)
            self.recent_snap.append(pct)
        if team_carries > 0 or team_targets > 0:
            share = np.array([stats[0] / team_carries if team_carries > 0 else 0.0,
                              stats[1] / team_targets if team_targets > 0 else 0.0])
            self.share_w = (1 - d) * self.share_w + d
            self.share += (d / self.share_w) * (share - self.share)
        tds = stats[3] + stats[4]
        self.recent.append((stats[0], stats[1], tds))
        keep = 1 - p.slow_decay
        self.slow_tds = keep * self.slow_tds + tds
        self.slow_touches = keep * self.slow_touches + stats[0] + stats[1]
        self.games += 1
        self.season_games += 1
        self.season_tds += tds
        self.name, self.team = name, team
        if isinstance(position, str):
            self.position = position

    def read(self, season, team, position, p):
        """PLAYER_FEATURES as he carries them into a game in `season` for
        `team`. Reading changes nothing."""
        prior = TD_RATE_PRIOR.get(position, TD_RATE_OTHER)
        n = len(self.recent)
        return [*self.ew, self.snap, self.share[0], self.share[1],
                sum(r[0] for r in self.recent) / n, sum(r[1] for r in self.recent) / n,
                sum(r[2] for r in self.recent) / n,
                sum(self.recent_snap) / len(self.recent_snap) if self.recent_snap else 0.0,
                (self.slow_tds + prior * p.td_rate_prior_n) / (self.slow_touches + p.td_rate_prior_n),
                math.log1p(self.games), self.season_games if self.season == season else 0,
                float(self.team != team)]


def _debut(position):
    """PLAYER_FEATURES of a player with no history."""
    return [0.0] * (len(STATS) + 7) + [TD_RATE_PRIOR.get(position, TD_RATE_OTHER), 0.0, 0, 0.0]


@dataclass
class _Team:
    season: int = None   # of its last game played
    qb: str = None       # the quarterback with most of its snaps in that game


def _derive(frame):
    """Columns computed from the others: xtd, the team's implied points as
    expected rushing and receiving TDs given out by the player's carry and
    target shares, and logs of the counts (the logistic regression's scale)."""
    tds = frame["implied"] / POINTS_PER_TD
    frame["xtd"] = tds * (RUSH_TD_SHARE * frame["ew_carry_share"]
                          + (1 - RUSH_TD_SHARE) * frame["ew_target_share"])
    frame["log_xtd"] = np.log(frame["xtd"] + 0.01)
    for c in LOGGED:
        frame[f"log_{c}"] = np.log1p(frame[c].clip(lower=0))
    return frame


def build(games, lines, params=TdParams()):
    """One row per (game, player) who took an offensive snap: META,
    pre-game FEATURES, and `scored` (NaN until the game is played and its
    box score is in). `lines` is player_lines()."""
    return build_with_state(games, lines, params)[0]


def build_with_state(games, lines, params=TdParams()):
    """build(), plus every player's and team's state after the last game
    played - what the live prediction reads."""
    with_stats = set(lines.loc[lines["has_stats"], "game_id"])
    stats = lines[list(STATS)].to_numpy(float)
    by_game = {}
    cols = ["game_id", "gsis_id", "name", "team", "position", "offense_pct", "in_snaps"]
    for i, (gid, gsis, name, team, pos, pct, snapped) in enumerate(zip(
            *(lines[c].tolist() for c in cols))):
        by_game.setdefault(gid, []).append(
            (gsis, name, team, pos, float(pct) if snapped else np.nan, bool(snapped), i))

    players, teams = {}, {}
    rows, meta = [], []
    for g in games.itertuples(index=False):
        game_lines = by_game.get(g.game_id)
        if not game_lines:
            continue
        side = {g.home: (1.0, g.away), g.away: (0.0, g.home)}
        implied = {t: implied_points(g.total_line, g.spread_line, side[t][0] == 1.0) for t in side}
        playoff = float(g.game_type != "REG")
        labelled = bool(g.played) and g.game_id in with_stats
        for gsis, name, team, pos, pct, snapped, i in game_lines:
            if not snapped or team not in side:
                continue
            home, opp = side[team]
            pl = players.get(gsis)
            f = pl.read(g.season, team, pos, params) if pl else _debut(pos)
            rows.append([float(pos == q) for q in POSITIONS] + f + [implied[team], home, playoff])
            meta.append((g.game_id, g.season, g.week, g.game_type, g.kickoff, team, opp, gsis,
                         name, pos, stats[i, 3] + stats[i, 4] if labelled else np.nan))
        if not g.played:
            continue
        # Fold the game in.
        totals, qbs = {}, {}
        for _, _, team, _, _, _, i in game_lines:
            totals[team] = totals.get(team, 0.0) + stats[i, :2]
        for gsis, name, team, pos, pct, snapped, i in game_lines:
            pl = players.get(gsis)
            if pl is None:
                pl = players[gsis] = _Player(params.last_n)
            pl.add(g.season, name, team, pos, pct, stats[i], *totals[team], params)
            if pos == "QB" and snapped and pct > qbs.get(team, (None, -1.0))[1]:
                qbs[team] = (gsis, pct)
        for team in side:
            t = teams.setdefault(team, _Team())
            t.season = g.season
            if team in qbs:
                t.qb = qbs[team][0]

    meta = pd.DataFrame(meta, columns=META)
    frame = pd.concat([meta, pd.DataFrame(rows, columns=RAW_FEATURES, dtype=float)], axis=1)
    frame["scored"] = np.where(frame["tds"].notna(), (frame["tds"] >= 1).astype(float), np.nan)
    return _derive(frame), {"players": players, "teams": teams, "params": params}


# ---------------------------------------------------------------- the model

def _x(frame, cols):
    return frame[cols].astype(float).fillna(0.0).to_numpy()


class _PositionRate:
    """The base rate of each position in the training rows (the pos_ one-hot
    columns in, one rate per column)."""

    def fit(self, x, y):
        overall = float(np.mean(y))
        self.rates = np.array([float(np.mean(y[x[:, j] == 1])) if (x[:, j] == 1).any() else overall
                               for j in range(x.shape[1])])
        self.overall = overall
        return self

    def predict_proba(self, x):
        p = np.where(x.any(axis=1), x @ self.rates, self.overall)
        return np.column_stack([1 - p, p])


@dataclass
class TdModel:
    """A fitted learner and the columns it reads."""
    kind: str
    estimator: object
    columns: list
    trained_on: int

    def predict(self, rows):
        return self.estimator.predict_proba(_x(rows, self.columns))[:, 1]


def fit(train, kind=MODEL):
    """One learner ('logistic', 'hgb', 'usage' or 'base') on labelled rows."""
    if kind == "logistic":
        est, cols = make_pipeline(StandardScaler(), LogisticRegression(C=LOGISTIC_C, max_iter=3000)), FEATURES
    elif kind == "hgb":
        est, cols = HistGradientBoostingClassifier(**HGB), FEATURES
    elif kind == "usage":
        est, cols = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), USAGE_FEATURES
    elif kind == "base":
        est, cols = _PositionRate(), [f"pos_{q}" for q in POSITIONS]
    else:
        raise ValueError(f"unknown learner {kind!r}")
    est.fit(_x(train, cols), train["scored"].astype(int).to_numpy())
    return TdModel(kind, est, list(cols), len(train))


def labelled(frame):
    return frame["scored"].notna()


def walk_forward(frame, start=DEV[0], end=None, kinds=LEARNERS + BASELINES):
    """Out-of-sample probabilities, season by season: one column per kind,
    plus `prob`, MODEL's. Season S's come from rows of TRAIN_FROM to S - 1."""
    end = end or int(frame["season"].max())
    out = pd.DataFrame(np.nan, index=frame.index, columns=list(kinds), dtype=float)
    ok = labelled(frame)
    for season in range(start, end + 1):
        test = frame["season"] == season
        train = frame[ok & frame["season"].between(TRAIN_FROM, season - 1)]
        if not test.any() or not len(train):
            continue
        for kind in kinds:
            out.loc[test, kind] = fit(train, kind).predict(frame[test])
    if MODEL in out:
        out["prob"] = out[MODEL]
    return out


# ---------------------------------------------------------------- measurement

CALIBRATION_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.0)


def calibration(p, y, edges=CALIBRATION_EDGES):
    """Predicted against observed scoring rate in probability bins."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            rows.append({"lo": lo, "hi": hi, "rows": int(m.sum()),
                         "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def top_picks(rows, p, n=1):
    """The `n` most likely scorers of every team-game: their hit rate
    against their mean probability, and how often at least one scored."""
    ranked = rows[["game_id", "team", "scored"]].assign(p=np.asarray(p, float))
    ranked = ranked.sort_values(["game_id", "team", "p"], ascending=[True, True, False], kind="stable")
    top = ranked.groupby(["game_id", "team"], sort=False).head(n)
    by_team = top.groupby(["game_id", "team"], sort=False)
    return {"team_games": int(by_team.ngroups), "hit_rate": float(top["scored"].mean()),
            "mean_prob": float(top["p"].mean()),
            "any_hit_rate": float(by_team["scored"].max().mean())}


def report(frame, preds):
    """Everything measured about the anytime-TD model, JSON-able. `preds` is
    walk_forward(frame)."""
    kinds = [k for k in LEARNERS + BASELINES if k in preds]
    out = {"model": MODEL, "features": FEATURES,
           "periods": {"dev": list(DEV), "confirm": list(CONFIRM), "train_from": TRAIN_FROM}}
    for lo, hi in (DEV, CONFIRM):
        sel = labelled(frame) & frame["season"].between(lo, hi) & preds[kinds].notna().all(axis=1)
        rows, y = frame[sel], frame.loc[sel, "scored"].to_numpy(float)
        p = {k: preds.loc[sel, k].to_numpy(float) for k in kinds}
        out[f"{lo}-{hi}"] = {
            "rows": int(sel.sum()),
            "scored_rate": float(y.mean()),
            "log_loss": {k: bt.log_loss(p[k], y) for k in kinds},
            "brier": {k: bt.brier(p[k], y) for k in kinds},
            "auc": {k: bt.auc(p[k], y) for k in kinds},
            "calibration": calibration(p[MODEL], y),
            "top_pick": {k: top_picks(rows, p[k], 1) for k in kinds if k != "base"},
            "top3": top_picks(rows, p[MODEL], 3),
        }
    dev = out[f"{DEV[0]}-{DEV[1]}"]["log_loss"]
    learners = {k: dev[k] for k in LEARNERS if k in dev}
    out["dev_choice"] = dict(learners, prefers=min(learners, key=learners.get) if learners else None)
    return out


def summary(rep):
    """The report as the few lines worth reading in a terminal."""
    lines = [f"Anytime TD - model: {rep['model']} (DEV prefers {rep['dev_choice']['prefers']})"]
    for key in (f"{DEV[0]}-{DEV[1]}", f"{CONFIRM[0]}-{CONFIRM[1]}"):
        r = rep[key]
        lines.append(f"\n  {key}: {r['rows']:,} player-games, {r['scored_rate']:.1%} scored")
        lines.append(f"    {'':10s} {'log loss':>9s} {'Brier':>8s} {'AUC':>7s}")
        for k in r["log_loss"]:
            lines.append(f"    {k:10s} {r['log_loss'][k]:9.4f} {r['brier'][k]:8.4f} {r['auc'][k]:7.3f}")
        lines.append("    calibration (predicted -> observed):")
        for c in r["calibration"]:
            lines.append(f"      {c['lo']:.2f}-{c['hi']:.2f}  {c['predicted']:6.3f} -> {c['observed']:6.3f}"
                         f"  ({c['rows']:,})")
        for k, t in r["top_pick"].items():
            lines.append(f"    top pick ({k}): scored {t['hit_rate']:.1%} of {t['team_games']:,} "
                         f"team-games, mean prob {t['mean_prob']:.1%}")
        t = r["top3"]
        lines.append(f"    top 3 ({rep['model']}): each scored {t['hit_rate']:.1%} (mean prob "
                     f"{t['mean_prob']:.1%}); one or more scored in {t['any_hit_rate']:.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------- live

@dataclass
class TdLive:
    """What the weekly run predicts from: MODEL fitted on every season
    before `season`, and the state after every game played."""
    season: int
    model: TdModel
    state: dict


def fit_live(season=None, games=None, lines=None, built=None):
    """The live model for `season` (default: the season of the next game
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
    train = frame[labelled(frame) & frame["season"].between(TRAIN_FROM, season - 1)]
    return TdLive(season, fit(train, MODEL), state)


def candidates(live, team, roster=None, min_usage=MIN_USAGE):
    """[(gsis_id, _Player)] who may play for `team` in a game of live.season."""
    players, season = live.state["players"], live.season
    t = live.state["teams"].get(team) or _Team()
    if roster is not None:
        pool = [(g, players[g]) for g in dict.fromkeys(roster) if g in players]
    else:
        recent = season if t.season == season else season - 1
        pool = [(g, pl) for g, pl in players.items() if pl.team == team and pl.season == recent]
    return [(g, pl) for g, pl in pool
            if pl.ew[0] + pl.ew[1] >= min_usage or (g == t.qb and pl.position == "QB")]


def predict_team(live, team, implied, home, exclude=(), roster=None, playoff=False,
                 min_usage=MIN_USAGE):
    """Every candidate's chance of a rushing or receiving TD in `team`'s
    next game, if he plays, highest first.

    implied: the team's expected points (implied_points() of the game's
    total and home spread; None reads as DEFAULT_IMPLIED); home:
    bool; exclude: gsis ids ruled out, on reserve or otherwise not playing;
    roster: gsis ids to draw candidates from instead of the
    most-recent-game rule.
    """
    exclude = set(exclude or ())
    pool = [(g, pl) for g, pl in candidates(live, team, roster, min_usage) if g not in exclude]
    if not pool:
        return []
    p = live.state["params"]
    implied = DEFAULT_IMPLIED if implied is None or implied != implied else float(implied)
    rows = [[float(pl.position == q) for q in POSITIONS] + pl.read(live.season, team, pl.position, p)
            + [implied, float(bool(home)), float(bool(playoff))] for _, pl in pool]
    probs = live.model.predict(_derive(pd.DataFrame(rows, columns=RAW_FEATURES, dtype=float)))
    out = []
    for (gsis, pl), prob in zip(pool, probs):
        this_season = pl.season == live.season
        out.append({
            "gsis_id": gsis, "name": pl.name, "position": pl.position, "prob": float(prob),
            "carries": float(pl.ew[0]), "targets": float(pl.ew[1]), "snap_pct": float(pl.snap),
            "tds_season": int(pl.season_tds) if this_season else 0,
            "games_season": int(pl.season_games) if this_season else 0,
            "last_team": pl.team,
        })
    out.sort(key=lambda c: -c["prob"])
    return out


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
