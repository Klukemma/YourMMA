"""Coaching approach: how each offence played each game, read from nflverse's
play-by-play. One row per team per game - the team on offence, its head
coach that day, the opponent and the opponent's head coach - with how often
it threw when the situation did not force it, how it called the red zone,
when it went for it on fourth down, how fast it played and how well it all
worked; and beside it one row per player per team-game for who got the ball
near the goal line. DATA ONLY: nothing here reaches a pick until it is
measured (THE MEASUREMENT PLAN, at the end of this docstring).

    python nfl/engine/nfl_approach.py            # this season
    python nfl/engine/nfl_approach.py --full     # every season since 1999
    python nfl/engine/nfl_approach.py 2023 2024  # those seasons

THE SOURCE. nflverse's play_by_play_{season}.csv.gz (nfl_data.RELEASE +
/pbp/), ~18 MB and ~48,000 rows a season, 1999 on. Only the columns below
are kept from its 372; a season is downloaded, aggregated and dropped before
the next is fetched, so memory holds one season at a time. xpass is
nflfastR's chance that a play is a dropback given down, distance, field
position, clock, score and win probability; the probe found it on ~76% of
the rows of each season it read (2006, 2016, 2025) - the scrimmage plays -
and on every scrimmage play of its two sample games. A season without it
gets empty xpass/PROE columns and keeps everything else.

THE ROW. Keyed (game_id, team). team and opponent are franchise codes as
nfl_data uses them (OAK -> LV, SD -> LAC, STL -> LA), so a row joins the
schedule and the team-game snapshot on game_id. coach and opp_coach are
home_coach / away_coach as nflverse spells them, by side: a coach's history
is his name's, and it follows him from team to team. One name is two
coaches: nflverse spells Jim Mora Sr. (IND 1999-2001) and his son (ATL
2004-06, SEA 2009) both "Jim Mora", and COACH_NAMES tells them apart by
season. No other name in the schedule's 177 (1999-2026) is two head coaches:
every other name with a gap between seasons is one coach returning (Gruden,
Payton, Jim Harbaugh, Vrabel...). home is 1 for the home side (the schedule's,
even at a neutral site). spread is the closing line from this offence's side
(positive: favoured by that many), known at kickoff. The defence that faced
this offence is the opponent's: what it allowed in the game is this very
row, play for play - allowed() re-keys the rows by the defence (team = the
defence, coach = its head coach, every statistic prefixed allowed_), so a
matchup reads as this offence's tendencies against that defence's profile
without a second copy of every number on file.

DEFINITIONS. Per team per game; a rate is NaN (empty on file) where its
count is 0, and a count is an integer.

  scrimmage play   a row with a posteam and a down where nflverse marks a
                   pass or a run (pass == 1 or rush == 1), less two-point
                   tries, kneels and spikes. Penalties: a snap wiped out by
                   a penalty still counts when nflverse can tell what was
                   called (no_play with pass or rush = 1: holding on a run,
                   interference on a throw); a pre-snap foul (false start,
                   delay of game) is no play at all. Overtime counts.
  plays            scrimmage plays. pass_plays: dropbacks (pass == 1 -
                   throws, sacks and scrambles, as nflverse counts them);
                   rush_plays: the rest (designed runs).
  pass_rate        pass_plays / plays.
  xpass, proe      the mean of nflverse's xpass, and pass rate over
                   expected = mean of (dropback - xpass), as a fraction
                   (nflverse's pass_oe is the same in percentage points),
                   over scrimmage plays with an xpass; xpass_plays counts
                   them. A play with no xpass counts everywhere else.
  early_*          downs 1 and 2, any situation: plays, pass rate, PROE.
  neutral_*        NEUTRAL SITUATION: an early down (1st or 2nd) in a
                   NEUTRAL GAME STATE - win probability (nflverse's wp,
                   before the snap) between 0.2 and 0.8, quarters 1 to 3,
                   and not the last two minutes of the first half
                   (half_seconds_remaining > 120). Plays, pass rate, mean
                   xpass, PROE, no-huddle rate.
  rz_*             RED ZONE: the snap at the opponent's 20 or closer
                   (yardline_100 <= 20). Plays, rush rate (1 - pass rate;
                   a scramble is a dropback), PROE, and rushing and passing
                   touchdowns scored on red-zone snaps.
  gtg_*            GOAL TO GO (nflverse's goal_to_go == 1): the same.
  rush_td, pass_td the offence's rushing and passing touchdowns on its
                   scrimmage plays (rush_touchdown / pass_touchdown, the
                   scoring team being the offence). A scramble that scores
                   is a rushing touchdown, as in the box score; a two-point
                   try is never a touchdown; a defensive return is not this
                   offence's.
  drives           drives (nflverse's fixed_drive) with a scrimmage play.
  rz_drives        RED-ZONE TRIPS: those drives with a scrimmage play or a
                   field goal or punt snapped at the opponent's 20 or
                   closer - a drive that reached the 18 on third down and
                   kicked is a trip; one whose only snaps inside the 20 were
                   kneels or spikes is not. rz_td_drives: the trips that
                   ended with an offensive touchdown. nflverse's own
                   drive_inside20 differs in two ways: it wants the ball
                   strictly inside the 20 (a drive that stalled AT the 20
                   is not its trip, and is ours), and it counts a drive
                   whatever the play (a kneel-down drive). On the two
                   sample games the two agree but for one New Orleans drive
                   that kicked from the 20 (ours 5, nflverse's 4).
  fourth_downs     FOURTH-DOWN DECISIONS: snaps on 4th down that were a
                   run, a pass, a punt or a field goal attempt - or a run or
                   pass wiped out by a penalty - one per series (nflverse's
                   series): when a penalty makes the offence do it again,
                   the last one counts (a field goal nullified and kicked
                   again is one decision). Kneels and spikes are not
                   decisions; a pre-snap foul is not a snap, and a punt or
                   kick wiped out by a penalty is not a decision either
                   (nflverse does not mark what it was).
  fourth_go        decisions that were a run or a pass (a fake punt or
                   field goal is going for it). fourth_conv: of those, the
                   ones that gained the first down, scored, or drew a
                   defensive penalty for a first down.
  gz_*             THE GO ZONE: 4th and 2 or shorter, from the opponent's
                   45 to its 1 (yardline_100 <= 45), win probability 0.2 to
                   0.8, regulation (quarters 1-4), not the last two minutes
                   of either half - where going for it is usually right and
                   the game does not force the choice. gz_downs decisions,
                   gz_go went for it, gz_conv converted, gz_go_rate =
                   gz_go / gz_downs - all on the LAST decision of a series,
                   the one that stood.
  gz_first_*       the same on the FIRST decision of each series: the call
                   the coach made before any penalty replayed it. A run on
                   4th and 1 at the 40 wiped out by holding, then a punt on
                   4th and 11, is a go in the go zone here (gz_first_downs
                   1, gz_first_go 1) and no go-zone decision at all above.
                   gz_first_go_rate = gz_first_go / gz_first_downs. The
                   measurement chooses between the two on DEV.
  pace_*           NEUTRAL PACE: the game-clock seconds between consecutive
                   offensive snaps of the same drive (game_seconds_remaining
                   at the earlier snap minus at the later one; kneels and
                   spikes are snaps here, special-teams plays are not),
                   counted when the later snap is a scrimmage play in a
                   neutral game state (any down). A gap of 0 or less, or of
                   over 60 seconds (more than a play clock and a play: a
                   missing row), is dropped. pace_snaps gaps, pace_sec their
                   mean. The game clock stops after incompletions and
                   out-of-bounds plays, so pace_sec is the offence's tempo
                   and its play mix together.
  shotgun_rate, no_huddle_rate   over scrimmage plays. A season in which
                   nflverse flags no scrimmage play at all as shotgun (or
                   no-huddle) did not record it: sync() leaves that
                   season's rates empty rather than writing a false 0.
  epa_play, success_rate         nflverse's epa and success (epa > 0) over
                   scrimmage plays; pass_* over dropbacks, rush_* over runs.

THE PLAYER ROWS. Keyed (game_id, team, player_id): player_id is nflverse's
gsis id (as in player_games), one row for every player of the offence with
a red-zone or goal-to-go carry, target or touchdown in the game; position
is read from player_games or the rosters when the rows are used, as nfl_td
does. On the same snaps as rz_* and gtg_* above:
  rz_carries, gtg_carries   scrimmage plays with him as rusher_player_id -
                   designed runs and scrambles (a scramble is a carry, as in
                   the box score, and a dropback in the team row);
  rz_targets, gtg_targets   dropbacks with him as receiver_player_id;
  rz_rush_td, rz_rec_td, gtg_rush_td, gtg_rec_td   the offence's rushing
                   and passing touchdowns with him as td_player_id (a pass
                   touchdown is the receiver's).
nflverse names no player on a play wiped out by a penalty, so the player
counts are the plays that stood, as the box score's are. A team-game's
players' touchdowns add up to its rz_rush_td / rz_pass_td (and gtg_*): on
the two sample games, exactly.

A GAME IS KEPT ONLY WHOLE. A season's file can hold a game still being
played. A game is aggregated when the schedule columns of the file carry
its final score, its plays reach the fourth quarter, and each side ran at
least MIN_TEAM_PLAYS scrimmage plays.

THE SNAPSHOTS. nfl/data/approach.csv.gz (the team rows) and
nfl/data/approach_players.csv.gz (the player rows), each sorted (season,
week, game_id, team[, player_id]), gzip with mtime 0 and fixed decimals (4;
pace 2; spread 1), so an unchanged season is byte-identical and git sees
nothing. A missing value is an empty field, never the string "nan". Each is
written to a temporary file beside it and renamed into place, so a run
killed mid-write leaves the old file whole. sync() keeps what is on file
for a season whose download fails, does not match its Content-Length, is
not a whole gzip, lacks any column in REQUIRED (every column whose absence
would change a number rather than empty it), holds another season, lacks
a game already on file or no longer has it whole, or - for a finished
season - has fewer than MIN_SEASON_GAMES whole games; and it rewrites only
the seasons whose rows changed: every other season's lines are carried
over as they were read. Each season's log line says how much of it nflverse
populated (xpass, shotgun and no-huddle shares of the scrimmage plays, and
the share of the runs that stood that name a rusher), so a warm-up season
with a column present but empty shows.

POINT IN TIME. Every number in a row is known once its game is over. The
nflverse model outputs in it - xpass, wp, epa and success - come from
models fitted across many seasons, later ones included: a mild look-ahead
the engine already accepts for team-game EPA (team_games), which is why the
plan below reads PROE against the league's walk-forward value, not against
a fixed zero. A feature for a game reads these rows by the engine's own
rule (nfl_features.build_with_state), joining kickoff from
nfl_data.load_games on game_id:
  own state     a coach's or a team's own rows (its offence's, or the
                allowed() rows of its defence) fold in as soon as its game
                is folded: it plays one game at a time, so its next game
                cannot kick off before this one is over.
  shared state  anything pooled across teams or coaches - league means,
                league slopes, any regression fitted across coaches - folds
                a game's rows only once its kickoff + nfl_features.FINAL_AFTER
                (4 hours) <= the reading game's kickoff. A 4:25 pm game does
                not read the 1 pm games still being played, which a live run
                could not see either.

THE MEASUREMENT PLAN. Fixed here, before any result is seen; the next step
builds and measures it, and nothing ships unmeasured.

  1. Coach state, walk-forward. One pass over the schedule in kickoff
     order, as nfl_features and nfl_td: each game reads both head coaches'
     state as it stood before kickoff, then folds in its two rows here once
     the game is played (own state, POINT IN TIME). The state is keyed by
     the coach's name as above (COACH_NAMES applied), not by the team: a
     coach who moves brings his history; a team that changes coach starts
     from the league. Per coach, exponentially weighted sums (rate x count,
     count) - half-life in games and an offseason keep, chosen on DEV - of:
     neutral PROE, early-down PROE, red-zone rush rate and PROE, goal-to-go
     rush rate, the rush share of offensive touchdowns (rush_td / (rush_td
     + pass_td)), the go-zone rate (last or first decision, chosen on DEV),
     neutral pace and neutral no-huddle rate, shotgun rate.
  2. Shrunk toward the league. value = (n x coach + k x league) / (n + k),
     n the coach's weighted count (plays; decisions for fourth down; gaps
     for pace), league the league's weighted value - SHARED state: only
     games whose kickoff + FINAL_AFTER is at or before this kickoff - and k
     in the same units, chosen on DEV. A first-time head coach (interim
     ones included) has n = 0 and reads the league, with a new_coach flag
     and his head-coach games so far (log) beside it.
  3. The approach against this opponent. The opponent defence's profile
     before kickoff, from allowed() rows of its earlier games (the
     defence's team history, own state - its players stay when a coach
     goes - shrunk the same way toward the shared league value): rush
     EPA/play allowed minus pass EPA/play allowed, the PROE it drew, the
     rush share of touchdowns it allowed; and the closing spread. The
     coach's expected PROE for the game = his shrunk base PROE + his own
     slopes on (opponent rush-minus-pass EPA allowed) and on the spread,
     each a regression on his earlier games (own state) shrunk with the
     same k rule to the league slope - a regression pooled across every
     coach, so SHARED state: fitted only on games final by this kickoff.
  4. How much the approach pays. Expected EPA/play for the game = p x (his
     pass EPA/play + the defence's pass EPA/play allowed - league) +
     (1 - p) x (the same for runs), p = the league pass rate in neutral
     situations + the expected PROE of 3, every league value shared state:
     the matchup value of the plan, beside the offence and defence EPA the
     model already reads.
  5. The game model. Home-minus-away differences of the expected PROE, the
     expected matchup EPA/play (4), the go-zone rate and neutral pace, the
     new-coach flags and head-coach games, into nfl_features.FEATURES (the
     L1 winner and the margin ridge); the sum of both sides' neutral pace
     and expected pass rates into TOTAL_FEATURES. The market blend refits
     exactly as today.
  6. The touchdown model. Today a team's implied points become expected
     TDs (xtd) handed out by the player's carry and target shares. With the
     coach: the team's expected TDs split into rushing and receiving by an
     expected rush share - the coach's shrunk rush share of touchdowns and
     red-zone and goal-to-go rush rates, against the opponent's shrunk rush
     share of touchdowns allowed - and handed out by carry share (the
     rushing part) and target share (the receiving part): xtd_rush and
     xtd_rec beside xtd. Near the goal line the hand-out reads the player
     rows (approach_players.csv.gz): the player's own weighted share of his
     team's red-zone and goal-to-go carries and targets, and the coach's
     red-zone split by position - the shares of his offences' red-zone
     carries, targets and touchdowns that went to RBs, QBs, WRs and TEs,
     position from player_games at read time - so a run-first goal line
     lifts the backs and the sneaking quarterback and lowers the receivers.
     The player's shares are his own state; the coach's split is the
     coach's own state, shrunk toward the shared league split.
  7. Periods and ship rules, pre-declared.
       game model  every choice (features, half-life, keep, k, slopes or
                   none) on DEV 2006-2015 by the pick's log loss (the market
                   blend); CONFIRM 2016-2025 computed once for that choice.
                   Ships only if the pick's log loss improves in BOTH.
       TD model    every choice on 2014-2019 by TD log loss; 2020-2025 once.
                   Ships only if TD log loss improves in BOTH, and the
                   calibration rule (nfl_td.shipped_groups) still passes.
       baseline    the same features keyed by TEAM instead of coach are
                   measured beside: coach keying is kept only if it beats
                   the team on DEV. A failure is written into the README's
                   "What was tried", not shipped.
     1999-2005 (and any season with no xpass) only warm the state up.
"""

import gzip
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import nfl_data as nd

APPROACH = nd.DATA / "approach.csv.gz"
PLAYERS = nd.DATA / "approach_players.csv.gz"
URL = nd.RELEASE + "/pbp/play_by_play_{season}.csv.gz"
FIRST_SEASON = nd.FIRST_SEASON

NEUTRAL_WP = (0.2, 0.8)
LATE_HALF = 120            # the last two minutes of a half, in seconds
GO_ZONE_TO_GO = 2          # 4th and this or shorter...
GO_ZONE_YARDLINE = 45      # ...from the opponent's 45 in
RED_ZONE = 20
MAX_GAP = 60               # seconds between snaps; more is a missing row
MIN_TEAM_PLAYS = 20        # a side with fewer scrimmage plays is a partial game
MIN_SEASON_GAMES = 250     # a finished season has 259 (1999-2001) to 285 games
DECISIONS = ("pass", "run", "punt", "field_goal")
KICKS = ("punt", "field_goal")

# Old codes the franchise map does not cover, should a season spell them.
ALIASES = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL",
           "CLV": "CLE", "HST": "HOU", "SL": "LA"}

# Two head coaches nflverse spells the same (THE ROW): (name, first season,
# last season, the name used here).
COACH_NAMES = [("Jim Mora", 1999, 2001, "Jim Mora Sr."),     # IND 1999-2001
               ("Jim Mora", 2004, 2009, "Jim Mora Jr.")]     # ATL 2004-06, SEA 2009

TEXT_IN = ["game_id", "season_type", "posteam", "defteam", "home_team", "away_team",
           "home_coach", "away_coach", "play_type", "td_team", "rusher_player_id",
           "receiver_player_id", "td_player_id"]
NUMBER_IN = ["season", "week", "play_id", "order_sequence", "fixed_drive", "drive", "series",
             "qtr", "down", "ydstogo", "yardline_100", "goal_to_go", "half_seconds_remaining",
             "game_seconds_remaining", "wp", "pass", "rush", "xpass", "epa", "success", "shotgun",
             "no_huddle", "qb_kneel", "qb_spike", "two_point_attempt", "rush_touchdown",
             "pass_touchdown", "fourth_down_converted", "first_down_penalty", "spread_line",
             "home_score", "away_score"]
# Every column whose absence would change a number rather than leave it
# empty (all are in 2006, 2016 and 2025). Not required: xpass (its absence
# empties xpass/PROE), spread_line (empties spread), order_sequence and
# drive (play_id and fixed_drive stand in), defteam, qb_kneel and qb_spike
# (play_type says the same).
REQUIRED = {"game_id", "season", "week", "play_id", "posteam", "home_team", "away_team",
            "home_coach", "away_coach", "home_score", "away_score", "play_type", "pass", "rush",
            "down", "ydstogo", "yardline_100", "goal_to_go", "qtr", "half_seconds_remaining",
            "game_seconds_remaining", "wp", "epa", "success", "shotgun", "no_huddle",
            "fixed_drive", "series", "rush_touchdown", "pass_touchdown", "td_team",
            "two_point_attempt", "fourth_down_converted", "first_down_penalty",
            "rusher_player_id", "receiver_player_id", "td_player_id"}
WANTED = set(TEXT_IN) | set(NUMBER_IN)

KEYS = ["season", "week", "season_type", "game_id", "team", "opponent", "home", "coach",
        "opp_coach", "spread"]
STATS = [
    "plays", "pass_plays", "rush_plays", "pass_rate", "xpass_plays", "xpass", "proe",
    "early_plays", "early_pass_rate", "early_proe",
    "neutral_plays", "neutral_pass_rate", "neutral_xpass", "neutral_proe",
    "rz_plays", "rz_rush_rate", "rz_proe", "rz_rush_td", "rz_pass_td",
    "gtg_plays", "gtg_rush_rate", "gtg_proe", "gtg_rush_td", "gtg_pass_td",
    "rush_td", "pass_td", "drives", "rz_drives", "rz_td_drives",
    "fourth_downs", "fourth_go", "fourth_conv", "gz_downs", "gz_go", "gz_conv", "gz_go_rate",
    "gz_first_downs", "gz_first_go", "gz_first_go_rate",
    "pace_snaps", "pace_sec",
    "shotgun_rate", "no_huddle_rate", "neutral_no_huddle_rate",
    "epa_play", "success_rate", "pass_epa_play", "pass_success_rate", "rush_epa_play",
    "rush_success_rate",
]
COLUMNS = KEYS + STATS
TEXT = {"season_type", "game_id", "team", "opponent", "coach", "opp_coach"}
COUNTS = ["plays", "pass_plays", "rush_plays", "xpass_plays", "early_plays", "neutral_plays",
          "rz_plays", "rz_rush_td", "rz_pass_td", "gtg_plays", "gtg_rush_td", "gtg_pass_td",
          "rush_td", "pass_td", "drives", "rz_drives", "rz_td_drives", "fourth_downs",
          "fourth_go", "fourth_conv", "gz_downs", "gz_go", "gz_conv", "gz_first_downs",
          "gz_first_go", "pace_snaps"]
INTS = {"season", "week", "home", *COUNTS}
DECIMALS = {"spread": 1, "pace_sec": 2}     # every other number: 4
SORT = ["season", "week", "game_id", "team"]

PLAYER_KEYS = ["season", "week", "season_type", "game_id", "team", "player_id"]
PLAYER_STATS = ["rz_carries", "rz_targets", "rz_rush_td", "rz_rec_td",
                "gtg_carries", "gtg_targets", "gtg_rush_td", "gtg_rec_td"]
PLAYER_COLUMNS = PLAYER_KEYS + PLAYER_STATS
PLAYER_TEXT = {"season_type", "game_id", "team", "player_id"}
PLAYER_INTS = {"season", "week", *PLAYER_STATS}
PLAYER_SORT = ["season", "week", "game_id", "team", "player_id"]


class Incomplete(Exception):
    """A season's download that must not replace what is on file."""


def team_code(code):
    """Today's franchise code for whatever the season's play-by-play called the team."""
    if not isinstance(code, str):
        return code
    return nd.franchise(ALIASES.get(code, code))


def coach_names(names, seasons):
    """nflverse's coach names, with COACH_NAMES applied by season."""
    names = pd.Series(names, dtype=object).reset_index(drop=True)
    seasons = pd.Series(seasons, dtype=float).reset_index(drop=True)
    out = names.copy()
    for name, first, last, new in COACH_NAMES:
        out[(names == name) & seasons.between(first, last)] = new
    return out.to_numpy(dtype=object)


# ---------------------------------------------------------------- the plays

def _number(raw, name):
    if name in raw:
        return pd.to_numeric(raw[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=raw.index, dtype=float)


def _text(raw, name):
    if name in raw:
        return raw[name].astype(object).where(raw[name].notna(), None)
    return pd.Series(None, index=raw.index, dtype=object)


def _plays(raw):
    """The columns read, typed, in play order within each game, and every
    mask the aggregation uses (see DEFINITIONS)."""
    cols = {c: _number(raw, c) for c in NUMBER_IN}
    cols.update({c: _text(raw, c) for c in TEXT_IN})
    f = pd.DataFrame(cols, index=raw.index)
    f["_order"] = f["order_sequence"].fillna(f["play_id"])
    f["_row"] = np.arange(len(f))
    f = f.sort_values(["game_id", "_order", "_row"], kind="mergesort").reset_index(drop=True)
    f["_drive"] = f["fixed_drive"].fillna(f["drive"])

    kind = f["play_type"].fillna("")
    offence = f["posteam"].notna()
    kneel = (f["qb_kneel"] == 1) | (kind == "qb_kneel")
    spike = (f["qb_spike"] == 1) | (kind == "qb_spike")
    two = f["two_point_attempt"] == 1
    called = (f["pass"] == 1) | (f["rush"] == 1)
    scrim = offence & f["down"].notna() & called & ~two & ~kneel & ~spike
    f["_scrim"] = scrim
    f["_pass"] = scrim & (f["pass"] == 1)
    f["_rush"] = scrim & ~f["_pass"]
    f["_x"] = scrim & f["xpass"].notna()
    f["_oe"] = np.where(f["_x"], f["_pass"].astype(float) - f["xpass"], 0.0)
    f["_xv"] = np.where(f["_x"], f["xpass"], 0.0)
    lo, hi = NEUTRAL_WP
    calm = f["wp"].between(lo, hi) & (f["half_seconds_remaining"] > LATE_HALF)
    f["_state"] = calm & (f["qtr"] <= 3)                       # neutral game state
    f["_early"] = scrim & f["down"].isin([1, 2])
    f["_neutral"] = f["_early"] & f["_state"]
    inside = f["yardline_100"] <= RED_ZONE
    f["_rz"] = scrim & inside
    f["_gtg"] = scrim & (f["goal_to_go"] == 1)
    f["_rz_trip"] = offence & inside & ~two & (scrim | kind.isin(KICKS))
    own = f["td_team"].isna() | (f["td_team"] == f["posteam"])
    f["_rush_td"] = scrim & (f["rush_touchdown"] == 1) & own
    f["_pass_td"] = scrim & (f["pass_touchdown"] == 1) & own
    f["_snap"] = offence & ~two & (called | kneel | spike)
    f["_decision"] = (offence & (f["down"] == 4) & ~two & ~kneel & ~spike
                      & (kind.isin(DECISIONS) | ((kind == "no_play") & called)))
    f["_go"] = f["_decision"] & ~kind.isin(KICKS)
    f["_go_zone"] = (calm & (f["qtr"] <= 4) & (f["ydstogo"] <= GO_ZONE_TO_GO)
                     & (f["yardline_100"] <= GO_ZONE_YARDLINE))
    return f


def _complete(f, scores=True):
    g = f.groupby("game_id")
    done = g["qtr"].max() >= 4
    if scores:
        done &= g["home_score"].max().notna() & g["away_score"].max().notna()
    plays = f[f["_scrim"]].groupby(["game_id", "posteam"]).size()
    sides = plays[plays >= MIN_TEAM_PLAYS].groupby(level=0).size()
    done &= sides.reindex(done.index).fillna(0) >= 2
    return set(done[done].index)


def complete_games(raw):
    """The game_ids of `raw` that are whole (A GAME IS KEPT ONLY WHOLE)."""
    if raw is None or not len(raw):
        return set()
    return _complete(_plays(raw), scores="home_score" in raw and "away_score" in raw)


def _ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _sides(f):
    """Each game's season, week, type, sides, coaches (COACH_NAMES applied) and line."""
    games = f.groupby("game_id").agg(
        season=("season", "max"), week=("week", "max"), season_type=("season_type", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_coach=("home_coach", "first"), away_coach=("away_coach", "first"),
        spread_line=("spread_line", "first"))
    games["home_coach"] = coach_names(games["home_coach"], games["season"])
    games["away_coach"] = coach_names(games["away_coach"], games["season"])
    return games


def aggregate(raw):
    """Play-by-play rows (whole games) -> one row per team per game, COLUMNS."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=COLUMNS)
    return _aggregate(_plays(raw))


def _aggregate(f):
    f = f[f["posteam"].notna()]
    if not len(f):
        return pd.DataFrame(columns=COLUMNS)
    key = ["game_id", "posteam"]

    def on(mask, values=None):
        return (np.where(mask, values, 0.0) if values is not None else mask.astype(float))

    epa_ok, succ_ok = f["epa"].notna(), f["success"].notna()
    parts = {
        "plays": on(f["_scrim"]), "pass_plays": on(f["_pass"]), "rush_plays": on(f["_rush"]),
        "xpass_plays": on(f["_x"]), "x_sum": f["_xv"], "oe_sum": f["_oe"],
        "early_plays": on(f["_early"]), "early_pass": on(f["_early"] & f["_pass"]),
        "early_x": on(f["_early"] & f["_x"]), "early_oe": on(f["_early"], f["_oe"]),
        "neutral_plays": on(f["_neutral"]), "neutral_pass": on(f["_neutral"] & f["_pass"]),
        "neutral_x": on(f["_neutral"] & f["_x"]), "neutral_xv": on(f["_neutral"], f["_xv"]),
        "neutral_oe": on(f["_neutral"], f["_oe"]),
        "neutral_nh": on(f["_neutral"] & (f["no_huddle"] == 1)),
        "rz_plays": on(f["_rz"]), "rz_rush": on(f["_rz"] & f["_rush"]), "rz_x": on(f["_rz"] & f["_x"]),
        "rz_oe": on(f["_rz"], f["_oe"]), "rz_rush_td": on(f["_rz"] & f["_rush_td"]),
        "rz_pass_td": on(f["_rz"] & f["_pass_td"]),
        "gtg_plays": on(f["_gtg"]), "gtg_rush": on(f["_gtg"] & f["_rush"]),
        "gtg_x": on(f["_gtg"] & f["_x"]), "gtg_oe": on(f["_gtg"], f["_oe"]),
        "gtg_rush_td": on(f["_gtg"] & f["_rush_td"]), "gtg_pass_td": on(f["_gtg"] & f["_pass_td"]),
        "rush_td": on(f["_rush_td"]), "pass_td": on(f["_pass_td"]),
        "shotgun": on(f["_scrim"] & (f["shotgun"] == 1)),
        "no_huddle": on(f["_scrim"] & (f["no_huddle"] == 1)),
        "epa_n": on(f["_scrim"] & epa_ok), "epa_sum": on(f["_scrim"] & epa_ok, f["epa"]),
        "succ_n": on(f["_scrim"] & succ_ok), "succ_sum": on(f["_scrim"] & succ_ok, f["success"]),
        "pepa_n": on(f["_pass"] & epa_ok), "pepa_sum": on(f["_pass"] & epa_ok, f["epa"]),
        "psucc_n": on(f["_pass"] & succ_ok), "psucc_sum": on(f["_pass"] & succ_ok, f["success"]),
        "repa_n": on(f["_rush"] & epa_ok), "repa_sum": on(f["_rush"] & epa_ok, f["epa"]),
        "rsucc_n": on(f["_rush"] & succ_ok), "rsucc_sum": on(f["_rush"] & succ_ok, f["success"]),
    }
    s = pd.DataFrame(parts, index=f.index).groupby([f["game_id"], f["posteam"]]).sum()
    s.index.names = key

    # Drives with a scrimmage play; red-zone trips; trips that ended in a touchdown.
    sc = f[(f["_scrim"] | f["_rz_trip"]) & f["_drive"].notna()]
    drive = sc.assign(_td=sc["_rush_td"] | sc["_pass_td"]).groupby(key + ["_drive"]).agg(
        scrim=("_scrim", "max"), rz=("_rz_trip", "max"), td=("_td", "max"))
    drive = drive[drive["scrim"]]
    dr = pd.DataFrame({"drives": 1.0, "rz_drives": drive["rz"].astype(float),
                       "rz_td_drives": (drive["rz"] & drive["td"]).astype(float)},
                      index=drive.index).groupby(level=[0, 1]).sum()

    # Fourth-down decisions: the last one of each series (the one that
    # stood), and the first (the call before any penalty replayed it).
    d4 = f[f["_decision"]].copy()
    d4["_series"] = d4["series"].fillna(-1.0 - d4.index.to_series().astype(float))
    series = d4.groupby(key + ["_series"], sort=False)
    first, d4 = series.head(1), series.tail(1)
    conv = d4["_go"] & ((d4["fourth_down_converted"] == 1) | (d4["first_down_penalty"] == 1)
                        | d4["_rush_td"] | d4["_pass_td"])
    zone = d4["_go_zone"]
    fd = pd.DataFrame({
        "fourth_downs": 1.0, "fourth_go": d4["_go"].astype(float), "fourth_conv": conv.astype(float),
        "gz_downs": zone.astype(float), "gz_go": (zone & d4["_go"]).astype(float),
        "gz_conv": (zone & conv).astype(float),
    }, index=d4.index).groupby([d4["game_id"], d4["posteam"]]).sum()
    fd.index.names = key
    called = pd.DataFrame({
        "gz_first_downs": first["_go_zone"].astype(float),
        "gz_first_go": (first["_go_zone"] & first["_go"]).astype(float),
    }, index=first.index).groupby([first["game_id"], first["posteam"]]).sum()
    called.index.names = key

    # Neutral pace: gaps between consecutive offensive snaps of a drive.
    sn = f[f["_snap"] & f["_drive"].notna()]
    gap = sn.groupby(key + ["_drive"])["game_seconds_remaining"].shift(1) - sn["game_seconds_remaining"]
    ok = sn["_scrim"] & sn["_state"] & (gap > 0) & (gap <= MAX_GAP)
    pace = pd.DataFrame({"pace_snaps": ok.astype(float), "pace_total": np.where(ok, gap, 0.0)},
                        index=sn.index).groupby([sn["game_id"], sn["posteam"]]).sum()
    pace.index.names = key

    t = (s.join(dr, how="left").join(fd, how="left").join(called, how="left")
         .join(pace, how="left").fillna(0.0))
    out = pd.DataFrame(index=t.index)
    for c in COUNTS:
        out[c] = t[c]
    out["pass_rate"] = _ratio(t["pass_plays"], t["plays"])
    out["xpass"] = _ratio(t["x_sum"], t["xpass_plays"])
    out["proe"] = _ratio(t["oe_sum"], t["xpass_plays"])
    out["early_pass_rate"] = _ratio(t["early_pass"], t["early_plays"])
    out["early_proe"] = _ratio(t["early_oe"], t["early_x"])
    out["neutral_pass_rate"] = _ratio(t["neutral_pass"], t["neutral_plays"])
    out["neutral_xpass"] = _ratio(t["neutral_xv"], t["neutral_x"])
    out["neutral_proe"] = _ratio(t["neutral_oe"], t["neutral_x"])
    out["rz_rush_rate"] = _ratio(t["rz_rush"], t["rz_plays"])
    out["rz_proe"] = _ratio(t["rz_oe"], t["rz_x"])
    out["gtg_rush_rate"] = _ratio(t["gtg_rush"], t["gtg_plays"])
    out["gtg_proe"] = _ratio(t["gtg_oe"], t["gtg_x"])
    out["gz_go_rate"] = _ratio(t["gz_go"], t["gz_downs"])
    out["gz_first_go_rate"] = _ratio(t["gz_first_go"], t["gz_first_downs"])
    out["pace_sec"] = _ratio(t["pace_total"], t["pace_snaps"])
    out["shotgun_rate"] = _ratio(t["shotgun"], t["plays"])
    out["no_huddle_rate"] = _ratio(t["no_huddle"], t["plays"])
    out["neutral_no_huddle_rate"] = _ratio(t["neutral_nh"], t["neutral_plays"])
    out["epa_play"] = _ratio(t["epa_sum"], t["epa_n"])
    out["success_rate"] = _ratio(t["succ_sum"], t["succ_n"])
    out["pass_epa_play"] = _ratio(t["pepa_sum"], t["pepa_n"])
    out["pass_success_rate"] = _ratio(t["psucc_sum"], t["psucc_n"])
    out["rush_epa_play"] = _ratio(t["repa_sum"], t["repa_n"])
    out["rush_success_rate"] = _ratio(t["rsucc_sum"], t["rsucc_n"])
    out = out[out["plays"] > 0].reset_index()

    # Who was on the field: the game's sides, coaches and line.
    out = out.join(_sides(f), on="game_id")
    home = out["posteam"] == out["home_team"]
    away = out["posteam"] == out["away_team"]
    out = out[home | away].copy()
    home = home[out.index]
    out["home"] = home.astype(int)
    out["team"] = out["posteam"].map(team_code)
    out["opponent"] = np.where(home, out["away_team"], out["home_team"])
    out["opponent"] = out["opponent"].map(team_code)
    out["coach"] = np.where(home, out["home_coach"], out["away_coach"])
    out["opp_coach"] = np.where(home, out["away_coach"], out["home_coach"])
    out["spread"] = np.where(home, out["spread_line"], -out["spread_line"])
    out = out[COLUMNS]
    for c in INTS:
        out[c] = out[c].astype(int)
    return out.sort_values(SORT).reset_index(drop=True)


def aggregate_players(raw):
    """Play-by-play rows (whole games) -> one row per player per team-game
    with a red-zone or goal-to-go carry, target or touchdown, PLAYER_COLUMNS."""
    if raw is None or not len(raw):
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    return _aggregate_players(_plays(raw))


def _aggregate_players(f):
    f = f[f["posteam"].notna()]
    carry = f["_scrim"] & f["rusher_player_id"].notna()
    target = f["_pass"] & f["receiver_player_id"].notna()
    scorer = f["td_player_id"].notna()
    events = []
    for zone, area in (("rz", f["_rz"]), ("gtg", f["_gtg"])):
        for stat, mask, who in (("carries", carry, "rusher_player_id"),
                                ("targets", target, "receiver_player_id"),
                                ("rush_td", f["_rush_td"] & scorer, "td_player_id"),
                                ("rec_td", f["_pass_td"] & scorer, "td_player_id")):
            m = area & mask
            events.append(pd.DataFrame({"game_id": f.loc[m, "game_id"], "posteam": f.loc[m, "posteam"],
                                        "player_id": f.loc[m, who], "stat": f"{zone}_{stat}"}))
    ev = pd.concat(events, ignore_index=True)
    if not len(ev):
        return pd.DataFrame(columns=PLAYER_COLUMNS)
    out = (ev.groupby(["game_id", "posteam", "player_id", "stat"]).size()
           .unstack("stat", fill_value=0).reindex(columns=PLAYER_STATS, fill_value=0).reset_index())
    out = out.join(_sides(f), on="game_id")
    out = out[(out["posteam"] == out["home_team"]) | (out["posteam"] == out["away_team"])].copy()
    out["team"] = out["posteam"].map(team_code)
    out = out[PLAYER_COLUMNS]
    for c in PLAYER_INTS:
        out[c] = out[c].astype(int)
    return out.sort_values(PLAYER_SORT).reset_index(drop=True)


def allowed(frame):
    """The same rows from the defence's side: team = the defence (this row's
    opponent), coach = its head coach, opponent = the offence it faced, home
    and spread from the defence's side, every statistic prefixed allowed_ -
    what that defence allowed in that game."""
    d = frame.rename(columns={s: f"allowed_{s}" for s in STATS})
    d = d.rename(columns={"team": "opponent", "opponent": "team", "coach": "opp_coach",
                          "opp_coach": "coach"})
    d["home"] = 1 - d["home"]
    d["spread"] = -d["spread"]
    return d[KEYS + [f"allowed_{s}" for s in STATS]].sort_values(SORT).reset_index(drop=True)


# ---------------------------------------------------------------- the files

class _Table:
    """One snapshot's layout: its columns, which are text and which whole
    numbers, the decimals of the rest, and its sort order."""

    def __init__(self, columns, text, ints, decimals, sort):
        self.columns, self.text, self.ints = columns, text, ints
        self.decimals, self.sort = decimals, sort


TEAM_TABLE = _Table(COLUMNS, TEXT, INTS, DECIMALS, SORT)
PLAYER_TABLE = _Table(PLAYER_COLUMNS, PLAYER_TEXT, PLAYER_INTS, {}, PLAYER_SORT)


def players_path(path):
    """The player rows' file beside a team rows' file (approach.csv.gz ->
    approach_players.csv.gz)."""
    path = Path(path)
    name = path.name[:-len(".csv.gz")] if path.name.endswith(".csv.gz") else path.stem
    return path.with_name(f"{name}_players.csv.gz")


def _cell(value, column, table=TEAM_TABLE):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if column in table.text:
        return "" if pd.isna(value) else str(value)
    if pd.isna(value):
        return ""
    if column in table.ints:
        return str(int(value))
    text = f"{float(value):.{table.decimals.get(column, 4)}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def as_text(frame, table=TEAM_TABLE):
    """Rows as the strings written to file (the table's columns, fixed
    decimals, '' for missing)."""
    return pd.DataFrame({c: [_cell(v, c, table) for v in frame[c]] if c in frame else [""] * len(frame)
                         for c in table.columns}, dtype=object)


def _sort_text(frame, table=TEAM_TABLE):
    if not len(frame):
        return frame.reset_index(drop=True)
    order = pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") if c in ("season", "week")
                          else frame[c] for c in table.sort})
    return frame.loc[order.sort_values(table.sort, kind="mergesort").index].reset_index(drop=True)


def _read_text(path, table=TEAM_TABLE):
    """The file as it is, every cell the string on disk ('' when empty)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=table.columns, dtype=object)
    t = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    return t.reindex(columns=table.columns, fill_value="").astype(object)


def _write_text(frame, path, table=TEAM_TABLE):
    """Write the rows, all or nothing: a temporary file beside the target,
    renamed onto it once whole, so a run killed mid-write leaves the old
    file as it was."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = frame[table.columns].to_csv(index=False, lineterminator="\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    try:
        # mtime 0 and no file name in the header: the same rows are the same bytes.
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(body)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write(frame, path=APPROACH):
    """Write aggregated team rows as the snapshot (every row, replacing the file)."""
    return _write_text(_sort_text(as_text(frame)), path)


def write_players(frame, path=PLAYERS):
    """Write aggregated player rows as their snapshot (every row, replacing the file)."""
    return _write_text(_sort_text(as_text(frame, PLAYER_TABLE), PLAYER_TABLE), path, PLAYER_TABLE)


def _load(path, table):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=object if c in table.text else float)
                             for c in table.columns})
    a = pd.read_csv(path, low_memory=False, dtype={c: str for c in table.text})
    for c in table.columns:
        if c not in a:
            a[c] = np.nan
    for c in table.text:
        a[c] = a[c].astype(object).where(a[c].notna(), None)
    return a[table.columns].sort_values(table.sort).reset_index(drop=True)


def load_approach(path=APPROACH):
    """The team rows, typed: one row per team per game, COLUMNS. Empty
    (with the columns) before the first sync."""
    return _load(path, TEAM_TABLE)


def load_players(path=PLAYERS):
    """The player rows, typed: one row per player per team-game with a
    red-zone or goal-to-go touch, PLAYER_COLUMNS. Empty before the first sync."""
    return _load(path, PLAYER_TABLE)


# ---------------------------------------------------------------- the download

def _fetch(url, timeout=300):
    """The body of one file; a short read raises (Content-Length checked)."""
    try:
        import requests
    except ImportError:
        from urllib.request import Request, urlopen
        with urlopen(Request(url, headers={"User-Agent": "YourNFL"}), timeout=timeout) as handle:
            body = handle.read()
            check_length(body, handle.headers.get("Content-Length"), handle.headers.get("Content-Encoding"))
            return body
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "YourNFL"})
    response.raise_for_status()
    body = response.content
    check_length(body, response.headers.get("Content-Length"), response.headers.get("Content-Encoding"))
    return body


def check_length(body, content_length, content_encoding=None):
    """A body shorter or longer than its Content-Length is a cut transfer.
    (With a Content-Encoding the header counts the encoded bytes: skipped.)"""
    if content_length is None or content_encoding:
        return
    if len(body) != int(content_length):
        raise Incomplete(f"{len(body):,} bytes of {int(content_length):,}")


def read_season(body):
    """One season's file -> the columns kept. A cut or corrupt gzip raises."""
    try:
        # low_memory (chunked parsing) holds ~120 MB for a season where one
        # pass over all 372 columns holds ~500 MB; text read as text, numbers
        # coerced in _plays, so a mixed-type chunk changes nothing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.DtypeWarning)
            raw = pd.read_csv(io.BytesIO(body), compression="gzip", low_memory=True,
                              usecols=lambda c: c in WANTED, dtype={c: str for c in TEXT_IN})
    except (EOFError, OSError, gzip.BadGzipFile, UnicodeDecodeError, pd.errors.ParserError) as err:
        raise Incomplete(f"not a whole gzip csv ({type(err).__name__}: {str(err)[:80]})") from err
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise Incomplete(f"columns missing: {sorted(missing)}")
    return raw


class Season:
    """One season read whole: its team rows, its player rows, and how much
    of it nflverse populated (shares of the scrimmage plays)."""

    def __init__(self, rows, players, coverage):
        self.rows, self.players, self.coverage = rows, players, coverage


def _coverage(f):
    sc = f[f["_scrim"]]
    stood = f["_rush"] & (f["play_type"] != "no_play")
    n = max(len(sc), 1)
    return {"xpass": float(sc["xpass"].notna().sum()) / n,
            "shotgun": float((sc["shotgun"] == 1).sum()) / n,
            "no_huddle": float((sc["no_huddle"] == 1).sum()) / n,
            "rusher_ids": float((stood & f["rusher_player_id"].notna()).sum()) / max(int(stood.sum()), 1)}


def season_rows(body, season, on_file=(), finished=False):
    """A season's body -> Season (its team and player rows), or Incomplete
    (keep the file). `on_file`: the game_ids on file for the season, every
    one of which must be whole in the download."""
    raw = read_season(body)
    seasons = set(pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int))
    if seasons and seasons != {season}:
        raise Incomplete(f"file holds seasons {sorted(seasons)}")
    f = _plays(raw)
    del raw
    whole = _complete(f)
    if not whole:
        raise Incomplete("no whole game")
    lost = sorted(set(on_file) - whole)
    if lost:
        raise Incomplete(f"{len(lost)} games on file not whole in the download ({', '.join(lost[:4])}"
                         + (", ..." if len(lost) > 4 else "") + ")")
    if finished and len(whole) < MIN_SEASON_GAMES:
        raise Incomplete(f"{len(whole)} whole games in a finished season")
    f = f[f["game_id"].isin(whole)].reset_index(drop=True)
    coverage = _coverage(f)
    rows = _aggregate(f)
    # A column nflverse did not record this season reads 0 on every play: empty, not 0.
    if coverage["shotgun"] == 0:
        rows["shotgun_rate"] = np.nan
    if coverage["no_huddle"] == 0:
        rows["no_huddle_rate"] = np.nan
        rows["neutral_no_huddle_rate"] = np.nan
    return Season(rows, _aggregate_players(f), coverage)


def _coverage_text(c):
    text = ", ".join(f"{name} {c[key]:.0%}" for name, key in (
        ("xpass", "xpass"), ("shotgun", "shotgun"), ("no huddle", "no_huddle"),
        ("rusher ids", "rusher_ids")))
    blank = [name for name, key in (("shotgun", "shotgun"), ("no huddle", "no_huddle")) if c[key] == 0]
    return text + (f"; {' and '.join(blank)} not recorded, left empty" if blank else "")


def sync(seasons=None, full=False, log=print, fetch=None, path=APPROACH, now=None, players=None):
    """Refresh both snapshots from nflverse's play-by-play, season by season.

    `seasons` defaults to the current one; `full` means every season since
    1999. `players` is the player rows' file (beside `path` by default). A
    season that cannot be read whole keeps its rows on file in both (see
    THE SNAPSHOTS); only seasons whose rows changed are rewritten."""
    fetch = fetch or _fetch
    now = now if now is not None else nd.current_season()
    players = Path(players) if players is not None else players_path(path)
    seasons = list(range(FIRST_SEASON, now + 1)) if full else list(seasons or [now])
    old = _read_text(path)
    old_players = _read_text(players, PLAYER_TABLE)
    old_season = pd.to_numeric(old["season"], errors="coerce")
    old_player_season = pd.to_numeric(old_players["season"], errors="coerce")
    merged, merged_players = old, old_players
    changed, unchanged, kept = [], [], {}
    team_changed = player_changed = False
    for season in seasons:
        on_file = old[old_season == season]
        on_file_players = old_players[old_player_season == season]
        try:   # the body lives only inside this call: one season in memory at a time
            got = season_rows(fetch(URL.format(season=season)), season,
                              on_file=set(on_file["game_id"]), finished=season < now)
        except Exception as err:   # a missing, cut or short file: keep the rows on file
            kept[season] = str(err)[:200]
            log(f"   pbp {season}: kept {len(on_file):,} rows on file ({kept[season]})")
            continue
        rows = got.rows
        unknown = sorted((set(rows["team"]) | set(rows["opponent"])) - set(nd.TEAMS))
        if unknown:
            log(f"   pbp {season}: team codes with no franchise entry {unknown}")
        fresh = _sort_text(as_text(rows))
        fresh_players = _sort_text(as_text(got.players, PLAYER_TABLE), PLAYER_TABLE)
        same = len(fresh) == len(on_file) and fresh.equals(_sort_text(on_file).astype(object))
        same_players = (len(fresh_players) == len(on_file_players) and fresh_players.equals(
            _sort_text(on_file_players, PLAYER_TABLE).astype(object)))
        games = rows["game_id"].nunique()
        if same and same_players:
            unchanged.append(season)
            log(f"   pbp {season}: {games} games, unchanged ({_coverage_text(got.coverage)})")
            continue
        if not same:
            merged = pd.concat([merged[pd.to_numeric(merged["season"], errors="coerce") != season],
                                fresh], ignore_index=True)
            team_changed = True
        if not same_players:
            merged_players = pd.concat(
                [merged_players[pd.to_numeric(merged_players["season"], errors="coerce") != season],
                 fresh_players], ignore_index=True)
            player_changed = True
        changed.append(season)
        log(f"   pbp {season}: {games} games, {len(rows):,} team rows, {len(fresh_players):,} player "
            f"rows ({_coverage_text(got.coverage)})")
    if team_changed:
        _write_text(_sort_text(merged), path)
    if player_changed:
        _write_text(_sort_text(merged_players, PLAYER_TABLE), players, PLAYER_TABLE)
    log(f"   approach: {len(changed)} seasons rewritten, {len(unchanged)} unchanged, "
        f"{len(kept)} kept on file" + (f" {sorted(kept)}" if kept else ""))
    return {"seasons": seasons, "changed": changed, "unchanged": unchanged, "kept": kept,
            "rows": len(merged), "player_rows": len(merged_players),
            "written": team_changed or player_changed}


if __name__ == "__main__":
    full = "--full" in sys.argv
    picked = [int(a) for a in sys.argv[1:] if a.isdigit()] or None
    print("Syncing nflverse play-by-play -> coaching approach" + (" (every season)" if full else ""))
    sync(seasons=picked, full=full)
