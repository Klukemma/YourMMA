"""Team Elo ratings for the NFL, game by game in kickoff order.

The FiveThirtyEight form: a rating moves after each game by K times the
surprise, scaled up for a blowout and damped when the favourite was the one
who blew the other team out (otherwise a strong team's margins inflate its
own rating). Between seasons every rating is pulled part of the way back to
the league mean, because rosters turn over.

POINT IN TIME. Each game's ratings are read BEFORE the game is applied, and
games are applied in kickoff order, so a prediction only ever sees results
that were final when it was made. A game not yet played reads the latest
ratings and changes nothing.

The parameters are fitted on 2002-2015 and judged on 2016 onwards (see
nfl_backtest.py); the defaults here are the fitted values, not 538's.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 48.0          # home-field advantage, Elo points
    revert: float = 1 / 3      # share of the gap to the mean closed each offseason
    mov: bool = True           # margin-of-victory multiplier
    playoff_k: float = 1.0     # K multiplier in the playoffs
    mean: float = 1505.0
    points_per_elo: float = 1 / 25   # Elo gap -> expected point margin


def expected(diff):
    """Chance the side with an Elo edge of `diff` wins."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def mov_multiplier(margin, winner_edge):
    """538's: grows with the log of the margin, shrinks when the winner was
    already the stronger side (`winner_edge` = winner's Elo edge, home field
    included)."""
    return math.log(abs(margin) + 1.0) * 2.2 / (winner_edge * 0.001 + 2.2)


def run(games, params=EloParams()):
    """Pre-game ratings for every game, and the final ratings.

    `games` is nfl_data.load_games() (sorted by kickoff). Returns
    (frame, ratings): frame has game_id, home_elo, away_elo, elo_diff
    (home minus away, home field included unless neutral), elo_prob and
    elo_spread (expected home margin); ratings maps team -> current Elo.
    """
    ratings, season_of = {}, {}
    rows = []
    for g in games.itertuples(index=False):
        for team in (g.home, g.away):
            if team not in ratings:
                ratings[team] = params.mean
            elif season_of[team] != g.season:
                ratings[team] += params.revert * (params.mean - ratings[team])
            season_of[team] = g.season
        home, away = ratings[g.home], ratings[g.away]
        diff = home - away + (0.0 if g.neutral else params.hfa)
        p = expected(diff)
        rows.append((g.game_id, home, away, diff, p, diff * params.points_per_elo))
        if not g.played:
            continue
        margin = g.result
        outcome = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        k = params.k * (params.playoff_k if g.game_type != "REG" else 1.0)
        if params.mov:
            if margin == 0:
                mult = 1.0
            else:
                winner_edge = diff if margin > 0 else -diff
                mult = mov_multiplier(margin, winner_edge)
        else:
            mult = 1.0
        shift = k * mult * (outcome - p)
        ratings[g.home] = home + shift
        ratings[g.away] = away - shift
    frame = pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo",
                                        "elo_diff", "elo_prob", "elo_spread"])
    return frame, ratings


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
