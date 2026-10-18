"""Pre-game features for every NFL game, built in kickoff order.

One pass over the schedule. Each game reads every team's and quarterback's
state as it stood BEFORE kickoff, then - if the game has been played - folds
that game's result and box score into the state. A game still to come reads
the latest state and changes nothing. That ordering is the whole of the
point-in-time guarantee, and tests/test_nfl_features.py checks it by
predicting a season with the future deleted and getting the same numbers.

What a team carries forward:
  elo            nfl_elo's rating (fitted parameters).
  net EPA/play   offense minus defense, exponentially weighted over recent
                 games and shrunk toward zero between seasons.
  points         scored and allowed, weighted the same way (totals model).
  last starter   the quarterback who started its previous game.
  last kickoff   for rest days, computed from kickoffs rather than trusted.

What a quarterback carries forward:
  EPA per dropback, weighted over his recent games and shrunk toward a
  replacement-level prior by a fixed number of pseudo-dropbacks - so a
  rookie making his first start reads as a backup, not as average.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import nfl_data as nd
import nfl_elo as ne


@dataclass(frozen=True)
class FeatureParams:
    decay: float = 0.10            # EWMA weight of the newest game
    offseason_keep: float = 0.6    # share of a team's EPA form carried into a new season
    qb_decay: float = 0.06         # per-game decay of a QB's history
    qb_prior: float = -0.12        # replacement-level EPA per dropback
    qb_prior_n: float = 120.0      # pseudo-dropbacks of that prior
    qb_offseason_keep: float = 0.8


FEATURES = [
    "elo_diff", "epa_diff", "off_epa_diff", "def_epa_diff", "qb_diff",
    "home_qb_change", "away_qb_change", "rest_diff", "home_bye", "away_bye",
    "home_short", "away_short", "tz_shift", "div_game", "neutral", "playoff",
]
TOTAL_FEATURES = [
    "pts_for_sum", "pts_against_sum", "off_epa_sum", "def_epa_sum",
    "league_total", "dome", "qb_sum",
]


class _Ewma:
    """Exponentially weighted mean that can be shrunk between seasons."""

    __slots__ = ("value", "weight")

    def __init__(self, value=0.0, weight=0.0):
        self.value, self.weight = value, weight

    def add(self, x, decay):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.weight = (1 - decay) * self.weight + decay
        self.value += (decay / self.weight) * (x - self.value)

    def shrink(self, keep, toward=0.0):
        self.value = toward + keep * (self.value - toward)


class _Qb:
    __slots__ = ("epa", "dropbacks", "starts", "season")

    def __init__(self):
        self.epa = self.dropbacks = 0.0
        self.starts = 0
        self.season = None

    def rating(self, p):
        return (self.epa + p.qb_prior * p.qb_prior_n) / (self.dropbacks + p.qb_prior_n)

    def add(self, epa, dropbacks, p):
        keep = 1 - p.qb_decay
        self.epa = keep * self.epa + epa
        self.dropbacks = keep * self.dropbacks + dropbacks


def _team_lines(team_games):
    """{(game_id, team): (plays, epa, points-free box)} for quick lookup."""
    out = {}
    for r in team_games.itertuples(index=False):
        out[(r.game_id, r.team)] = (r.plays, r.epa)
    return out


def _qb_lines(qb_games):
    by_game = {}
    for r in qb_games.itertuples(index=False):
        team = nd.franchise(r.team) if isinstance(r.team, str) else None
        by_game.setdefault(r.game_id, []).append((r.player_id, team, r.dropbacks, r.epa))
    return by_game


# A game's result is assumed known this long after kickoff: a regulation
# game takes about three and a quarter hours, overtime rarely four.
FINAL_AFTER = pd.Timedelta(hours=4)

ROOFED = ("dome", "retractable")


def roof_classes(games):
    """Each stadium's roof as a property of the building - dome, retractable
    or outdoors - known before any game is played there.

    The schedule's own roof column says 'open' or 'closed' for a retractable
    roof only once that game day's decision is made, and is blank for games
    still to come: read per game, the backtest and the live week would see
    the same stadium two different ways."""
    out = {}
    for stadium, roofs in games.groupby("stadium")["roof"]:
        seen = set(roofs.dropna())
        if seen & {"open", "closed"}:
            out[stadium] = "retractable"
        elif "dome" in seen:
            out[stadium] = "dome"
        elif seen:
            out[stadium] = "outdoors"
    return out


def roof_of(game, classes, home_roof):
    """The roof class for one game: its stadium's, else the home team's most
    recent home stadium's (a renamed or new building), else outdoors."""
    return classes.get(game.stadium) or home_roof.get(game.home) or "outdoors"


def build(games, team_games, qb_games, elo_params=ne.EloParams(), params=FeatureParams()):
    """One row per game: identifiers, pre-game features, outcomes.

    The frame keeps every column of `games` and adds the features named in
    FEATURES and TOTAL_FEATURES plus the Elo columns.
    """
    return build_with_state(games, team_games, qb_games, elo_params, params)[0]


def build_with_state(games, team_games, qb_games, elo_params=ne.EloParams(), params=FeatureParams()):
    """build(), plus every team's and quarterback's state after the last
    game played - what the app's team table shows.

    A game still to be played whose schedule names no quarterback (the
    projection is posted a week or so ahead) is given the team's last
    starter, and `home_qb_source` / `away_qb_source` say which it was:
    'started' (a game played), 'projected' (named by the schedule) or
    'last starter' (assumed).
    """
    elo, elo_now = ne.run(games, elo_params)
    lines = _team_lines(team_games)
    qblines = _qb_lines(qb_games)
    classes = roof_classes(games)
    home_roof = {}

    off, dfn, pf, pa = {}, {}, {}, {}
    season_of, last_kick, last_qb = {}, {}, {}
    qbs = {}
    team_qbs = {}                    # team -> [(season, kickoff, qb, dropbacks)]
    league = _Ewma(44.0, 1.0)        # league points per game, for totals
    league_season = None
    # SHARED STATE WAITS FOR THE FINAL WHISTLE. A team or quarterback plays
    # one game at a time, so its own state can be updated the moment its game
    # is folded in. The league average is shared by every game: updated
    # straight away, the second of two 1pm games would read the first one's
    # final score, and a 4:25pm game would read 1pm scores that were not yet
    # final. Its updates wait until FINAL_AFTER past their kickoff.
    pending_league = []

    def season_roll(q, season):
        """Discount a quarterback's history once, the first time he is
        seen in a new season - as a starter or in relief."""
        if q.season is not None and q.season != season:
            q.epa *= params.qb_offseason_keep
            q.dropbacks *= params.qb_offseason_keep
        q.season = season

    rows = []
    for g in games.itertuples(index=False):
        if league_season != g.season:
            league_season = g.season
        while pending_league and pending_league[0][0] + FINAL_AFTER <= g.kickoff:
            league.add(pending_league.pop(0)[1], 0.02)
        for team in (g.home, g.away):
            if team not in off:
                off[team], dfn[team] = _Ewma(), _Ewma()
                pf[team], pa[team] = _Ewma(21.0, 1.0), _Ewma(21.0, 1.0)
            elif season_of.get(team) != g.season:
                off[team].shrink(params.offseason_keep)
                dfn[team].shrink(params.offseason_keep)
                pf[team].shrink(params.offseason_keep, 21.0)
                pa[team].shrink(params.offseason_keep, 21.0)
            season_of[team] = g.season

        def qb_state(qb_id):
            if not isinstance(qb_id, str):
                return None
            q = qbs.get(qb_id)
            if q is None:
                q = qbs[qb_id] = _Qb()
            season_roll(q, g.season)
            return q

        home_qb_id, away_qb_id = g.home_qb_id, g.away_qb_id
        home_src = away_src = "started" if g.played else "projected"
        if not g.played and not isinstance(home_qb_id, str):
            home_qb_id, home_src = last_qb.get(g.home), "last starter"
        if not g.played and not isinstance(away_qb_id, str):
            away_qb_id, away_src = last_qb.get(g.away), "last starter"
        hq, aq = qb_state(home_qb_id), qb_state(away_qb_id)
        h_rating = hq.rating(params) if hq else params.qb_prior
        a_rating = aq.rating(params) if aq else params.qb_prior

        def rest(team):
            prev = last_kick.get(team)
            if prev is None:
                return np.nan
            return (g.kickoff - prev).total_seconds() / 86400.0

        h_rest, a_rest = rest(g.home), rest(g.away)
        # Rest across an offseason is not rest; it is a new season.
        h_rest = h_rest if h_rest == h_rest and h_rest < 30 else np.nan
        a_rest = a_rest if a_rest == a_rest and a_rest < 30 else np.nan

        roof = roof_of(g, classes, home_roof)
        if g.neutral:
            tz = 0.0
        else:
            tz = float(nd.home_offset(g.home_then) - nd.home_offset(g.away_then))

        row = {
            "game_id": g.game_id,
            "home_off_epa": off[g.home].value, "home_def_epa": dfn[g.home].value,
            "away_off_epa": off[g.away].value, "away_def_epa": dfn[g.away].value,
            "home_pf": pf[g.home].value, "home_pa": pa[g.home].value,
            "away_pf": pf[g.away].value, "away_pa": pa[g.away].value,
            "home_qb_rating": h_rating, "away_qb_rating": a_rating,
            "home_qb_starts": hq.starts if hq else 0, "away_qb_starts": aq.starts if aq else 0,
            "home_qb_change": int(isinstance(home_qb_id, str) and last_qb.get(g.home) not in (None, home_qb_id)),
            "away_qb_change": int(isinstance(away_qb_id, str) and last_qb.get(g.away) not in (None, away_qb_id)),
            "home_qb_used": home_qb_id, "away_qb_used": away_qb_id,
            "home_qb_source": home_src if isinstance(home_qb_id, str) else None,
            "away_qb_source": away_src if isinstance(away_qb_id, str) else None,
            "home_rest_days": h_rest, "away_rest_days": a_rest,
            "tz_shift": tz,
            "league_total": league.value,
            "roof_class": roof,
        }
        rows.append(row)

        # The schedule is public: rest is counted from the last SCHEDULED
        # game, so a result nflverse has not posted yet is not a bye week.
        last_kick[g.home] = last_kick[g.away] = g.kickoff
        if not g.neutral and isinstance(g.stadium, str) and g.stadium in classes:
            home_roof[g.home] = classes[g.stadium]
        if not g.played:
            continue
        # Fold the game in.
        for team, opp, scored, allowed in ((g.home, g.away, g.home_score, g.away_score),
                                           (g.away, g.home, g.away_score, g.home_score)):
            mine, theirs = lines.get((g.game_id, team)), lines.get((g.game_id, opp))
            if mine and mine[0] > 0:
                off[team].add(mine[1] / mine[0], params.decay)
            if theirs and theirs[0] > 0:
                dfn[team].add(theirs[1] / theirs[0], params.decay)
            pf[team].add(float(scored), params.decay)
            pa[team].add(float(allowed), params.decay)
        pending_league.append((g.kickoff, float(g.home_score + g.away_score)))
        for qb_id, qb_team, dropbacks, epa in qblines.get(g.game_id, []):
            if dropbacks and dropbacks > 0:
                q = qbs.get(qb_id)
                if q is None:
                    q = qbs[qb_id] = _Qb()
                season_roll(q, g.season)
                q.add(float(epa), float(dropbacks), params)
                if qb_team:
                    team_qbs.setdefault(qb_team, []).append((g.season, g.kickoff, qb_id, float(dropbacks)))
        for team, qb_id in ((g.home, g.home_qb_id), (g.away, g.away_qb_id)):
            if isinstance(qb_id, str):
                last_qb[team] = qb_id
                if qb_id in qbs:
                    qbs[qb_id].starts += 1

    f = pd.DataFrame(rows)
    out = games.merge(elo, on="game_id").merge(f, on="game_id")
    out["epa_diff"] = ((out.home_off_epa - out.home_def_epa) - (out.away_off_epa - out.away_def_epa))
    out["off_epa_diff"] = out.home_off_epa - out.away_off_epa
    out["def_epa_diff"] = out.home_def_epa - out.away_def_epa
    out["qb_diff"] = out.home_qb_rating - out.away_qb_rating
    out["rest_diff"] = (out.home_rest_days.fillna(7) - out.away_rest_days.fillna(7)).clip(-7, 7)
    out["home_bye"] = (out.home_rest_days >= 12).astype(int)
    out["away_bye"] = (out.away_rest_days >= 12).astype(int)
    out["home_short"] = (out.home_rest_days <= 5).astype(int)
    out["away_short"] = (out.away_rest_days <= 5).astype(int)
    out["playoff"] = (out.game_type != "REG").astype(int)
    out["pts_for_sum"] = out.home_pf + out.away_pf
    out["pts_against_sum"] = out.home_pa + out.away_pa
    out["off_epa_sum"] = out.home_off_epa + out.away_off_epa
    out["def_epa_sum"] = out.home_def_epa + out.away_def_epa
    out["qb_sum"] = out.home_qb_rating + out.away_qb_rating
    out["dome"] = out.roof_class.isin(ROOFED).astype(int)
    out["home_win"] = np.where(out.played, (out.result > 0).astype(float), np.nan)
    out.loc[out.played & (out.result == 0), "home_win"] = 0.5
    state = {
        "elo": dict(elo_now),
        "off_epa": {t: e.value for t, e in off.items()},
        "def_epa": {t: e.value for t, e in dfn.items()},
        "points_for": {t: e.value for t, e in pf.items()},
        "points_against": {t: e.value for t, e in pa.items()},
        "last_qb": dict(last_qb),
        "qb_rating": {q: s.rating(params) for q, s in qbs.items()},
        "qb_starts": {q: s.starts for q, s in qbs.items()},
        "team_qbs": {t: rows[-40:] for t, rows in team_qbs.items()},
    }
    return out, state


def with_quarterback(row, side, qb_id, state, source, params=FeatureParams()):
    """`row` (one game, a Series) with `side`'s starting quarterback replaced.

    For a game still to be played whose projected starter has been ruled
    out: the model was measured on the quarterbacks who actually started,
    so feeding it a starter the injury report says will not play is feeding
    it a different game. `state` is build_with_state's; a quarterback with
    no history (qb_id None or unknown) reads as replacement level, as a
    first start does everywhere else.
    """
    row = row.copy()
    known = isinstance(qb_id, str)
    row[f"{side}_qb_used"] = qb_id if known else None
    row[f"{side}_qb_source"] = source
    row[f"{side}_qb_rating"] = state["qb_rating"].get(qb_id, params.qb_prior) if known else params.qb_prior
    row[f"{side}_qb_starts"] = state["qb_starts"].get(qb_id, 0) if known else 0
    row[f"{side}_qb_change"] = int(known and state["last_qb"].get(row[side]) not in (None, qb_id)) if known else 1
    row["qb_diff"] = row["home_qb_rating"] - row["away_qb_rating"]
    row["qb_sum"] = row["home_qb_rating"] + row["away_qb_rating"]
    return row
