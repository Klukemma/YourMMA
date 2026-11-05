"""The coaching approach rows (nfl_approach): counted by hand on two real
games of nflverse's play-by-play, edge cases on made-up plays, the files,
the sync's keep-what-is-on-file rules, and the runner's wiring."""

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_approach as nap
import nfl_data as nd

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pbp" / "two_games_2025.csv.gz"
ARI_NO, BAL_BUF = "2025_01_ARI_NO", "2025_01_BAL_BUF"


@pytest.fixture(scope="module")
def raw():
    """Two whole real games (2025 week 1), reduced from the runner's probe
    (nfl/data/pbp_probe/pbp_sample_2025.csv.gz) to the columns read plus
    nflverse's own pass_oe, qb_scramble and drive_inside20, the player
    names and the play descriptions."""
    return pd.read_csv(FIXTURE, low_memory=False)


@pytest.fixture(scope="module")
def rows(raw):
    return nap.aggregate(raw).set_index("team")


def body_of(frame):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False).encode())
    return buf.getvalue()


# ------------------------------------------------------------ the real games, by hand

def test_one_row_per_team_with_its_coach_and_opponent(rows):
    assert sorted(rows.index) == ["ARI", "BAL", "BUF", "NO"]
    assert rows.loc["ARI", "coach"] == "Jonathan Gannon" and rows.loc["ARI", "opp_coach"] == "Kellen Moore"
    assert rows.loc["NO", "coach"] == "Kellen Moore" and rows.loc["NO", "home"] == 1
    assert rows.loc["BAL", "coach"] == "John Harbaugh" and rows.loc["BUF", "coach"] == "Sean McDermott"
    assert rows.loc["ARI", "opponent"] == "NO" and rows.loc["BUF", "opponent"] == "BAL"
    # The closing line from each offence's side: Arizona was a 6-point road favourite.
    assert rows.loc["ARI", "spread"] == 6.0 and rows.loc["NO", "spread"] == -6.0
    assert rows.loc["BUF", "spread"] == -1.5


def test_pass_rate_counted_by_hand(rows):
    """Arizona's 66 snaps, read off the play list drive by drive: 41
    dropbacks (four sacks, three scrambles, three throws wiped out by a
    penalty among them) and 25 runs (one wiped out). The false start and
    the delay of game are not plays; the kickoffs, punts and kicks are not."""
    ari = rows.loc["ARI"]
    assert (ari["plays"], ari["pass_plays"], ari["rush_plays"]) == (66, 41, 25)
    assert ari["pass_rate"] == pytest.approx(41 / 66)
    assert ari["xpass_plays"] == 66            # xpass on every scrimmage play
    assert rows.loc["BAL", "pass_rate"] == pytest.approx(23 / 51)


def test_proe_is_nflverses_own_pass_oe(raw, rows):
    """PROE is mean(dropback - xpass); nflverse's pass_oe is the same in
    percentage points, so the two must agree play set for play set."""
    sc = raw[raw["posteam"].notna() & raw["down"].notna() & ((raw["pass"] == 1) | (raw["rush"] == 1))
             & (raw["two_point_attempt"] != 1)]
    for team, plays in sc.groupby("posteam"):
        assert rows.loc[team, "proe"] == pytest.approx(plays["pass_oe"].mean() / 100, abs=1e-9)
        assert rows.loc[team, "xpass"] == pytest.approx(plays["xpass"].mean(), abs=1e-9)


def test_neutral_situations_by_hand(raw, rows):
    """Neutral: 1st/2nd down, wp 0.2-0.8, quarters 1-3, not the last two
    minutes of the half. Arizona: 28 such snaps, 16 of them dropbacks -
    its 1st-and-10 at 0.8006 in the third quarter is out, as is its 1st and
    10 at exactly 2:00 before the half."""
    ari = rows.loc["ARI"]
    assert (ari["neutral_plays"], round(ari["neutral_pass_rate"] * 28)) == (28, 16)
    oe = []
    for _, p in raw[(raw["game_id"] == ARI_NO) & (raw["posteam"] == "ARI")].iterrows():
        play = (p["pass"] == 1 or p["rush"] == 1) and p["two_point_attempt"] != 1
        if (play and p["down"] in (1, 2) and 0.2 <= p["wp"] <= 0.8 and p["qtr"] <= 3
                and p["half_seconds_remaining"] > 120):
            oe.append(p["pass"] - p["xpass"])
    assert len(oe) == 28 and ari["neutral_proe"] == pytest.approx(np.mean(oe), abs=1e-9)


def test_red_zone_and_goal_to_go_by_hand(rows):
    """Arizona inside the 20: 11 snaps (the 19 on its first scoring drive,
    six on the next, four before the half), 5 of them runs; touchdowns
    thrown from the 1 and the 4. Goal to go: 5 snaps, 1 run. Three trips,
    two touchdowns."""
    ari = rows.loc["ARI"]
    assert (ari["rz_plays"], ari["rz_rush_rate"]) == (11, pytest.approx(5 / 11))
    assert (ari["rz_rush_td"], ari["rz_pass_td"]) == (0, 2)
    assert (ari["gtg_plays"], ari["gtg_rush_rate"]) == (5, pytest.approx(0.2))
    assert (ari["gtg_pass_td"], ari["rz_drives"], ari["rz_td_drives"]) == (2, 3, 2)
    # Kamara's 18-yard run is a red-zone touchdown, not a goal-to-go one.
    assert (rows.loc["NO", "rz_rush_td"], rows.loc["NO", "gtg_rush_td"]) == (1, 0)
    assert pd.isna(rows.loc["BAL", "gtg_rush_rate"]) and rows.loc["BAL", "gtg_plays"] == 0


def test_red_zone_trips_against_nflverses_drive_inside20(raw, rows):
    """A trip is a drive with a scrimmage play or a kick snapped at the 20
    or closer. Buffalo's last drive reached the 9 on a completion from the
    34, knelt three times and kicked the winner from the 14: a trip (by
    nflverse too), though its only red-zone snaps were kneels and the kick.
    New Orleans's first field-goal drive reached the 18 on third down: a
    trip. Its third stalled at the 20 itself and kicked: a trip here, not
    in nflverse's drive_inside20, which wants the ball inside the 20."""
    assert rows["rz_drives"].to_dict() == {"ARI": 3, "NO": 5, "BAL": 3, "BUF": 7}
    theirs = raw[raw["posteam"].notna()].groupby("posteam")["fixed_drive"].apply(
        lambda d: raw.loc[d.index].groupby("fixed_drive")["drive_inside20"].max().sum())
    assert theirs.to_dict() == {"ARI": 3, "NO": 4, "BAL": 3, "BUF": 7}
    assert rows["rz_td_drives"].to_dict() == {"ARI": 2, "NO": 1, "BAL": 1, "BUF": 5}


def test_player_rows_by_hand(raw):
    """Who got the ball inside the 20, read off the play list. Arizona's 11
    red-zone snaps: Benson ran twice, Murray once (a designed run), Conner
    twice; Higgins, Harrison, McBride and Conner were thrown to; the two
    throws wiped out by a penalty name nobody. Buffalo's 18: Allen carried
    five times (three sneaks or runs, the scramble that scored, a run that
    did not), Cook four; nine throws to six receivers. Baltimore's only red-
    zone dropback was a sack: no target."""
    p = nap.aggregate_players(raw)
    names = {}
    for who in ("rusher", "receiver", "td"):
        ids = raw[f"{who}_player_id"].notna()
        names.update(zip(raw.loc[ids, f"{who}_player_id"], raw.loc[ids, f"{who}_player_name"]))
    p["name"] = p["player_id"].map(names)
    p = p.set_index(["team", "name"])
    stats = ["rz_carries", "rz_targets", "rz_rush_td", "rz_rec_td",
             "gtg_carries", "gtg_targets", "gtg_rush_td", "gtg_rec_td"]
    line = lambda team, name: tuple(int(v) for v in p.loc[(team, name), stats])  # noqa: E731
    assert line("ARI", "J.Conner") == (2, 1, 0, 1, 1, 1, 0, 1)
    assert line("ARI", "M.Harrison") == (0, 1, 0, 1, 0, 1, 0, 1)
    assert line("ARI", "K.Murray") == (1, 0, 0, 0, 0, 0, 0, 0)
    assert line("ARI", "T.Benson") == (2, 0, 0, 0, 0, 0, 0, 0)
    assert line("BUF", "J.Allen") == (5, 0, 2, 0, 5, 0, 2, 0)
    assert line("BUF", "J.Cook") == (4, 0, 1, 0, 2, 0, 1, 0)
    assert line("BUF", "K.Coleman") == (0, 2, 0, 1, 0, 0, 0, 0)
    assert line("NO", "A.Kamara") == (2, 0, 1, 0, 1, 0, 0, 0)
    assert sorted(p.loc["BAL"].index) == ["J.Hill", "L.Jackson", "R.Ali"]
    by_team = p.groupby(level=0)[stats].sum()
    assert by_team.loc["ARI", ["rz_carries", "rz_targets"]].tolist() == [5, 4]
    assert by_team.loc["BUF", ["rz_carries", "rz_targets"]].tolist() == [9, 9]
    assert by_team.loc["BAL", ["rz_carries", "rz_targets"]].tolist() == [3, 0]
    # The players' touchdowns add up to the team rows' exactly.
    team = nap.aggregate(raw).set_index("team")
    for a, b in (("rz_rush_td", "rz_rush_td"), ("rz_rec_td", "rz_pass_td"),
                 ("gtg_rush_td", "gtg_rush_td"), ("gtg_rec_td", "gtg_pass_td")):
        assert by_team[a].to_dict() == team[b].to_dict(), a
    assert (p.reset_index()["season"] == 2025).all() and p.reset_index()["player_id"].str.match(r"00-\d{7}$").all()


def test_touchdowns_match_the_box_score(rows):
    """Final scores 20-13 and 41-40: Murray threw two; Kamara ran one; Henry
    (30, 46) and Jackson (10) ran three for Baltimore and Jackson threw two;
    Buffalo ran three - Cook, Allen's scramble (a rushing touchdown in the
    box score) and Allen's sneak - and threw two. Three failed two-point
    tries are not touchdowns."""
    got = {t: (rows.loc[t, "rush_td"], rows.loc[t, "pass_td"]) for t in rows.index}
    assert got == {"ARI": (0, 2), "NO": (1, 0), "BAL": (3, 2), "BUF": (3, 2)}


def test_fourth_down_decisions_by_hand(rows):
    """Arizona: four punts and three field goals, never went. New Orleans:
    four punts, three field goals - the one nullified by its own penalty
    and kicked again is one decision - and two throws in the last half
    minute, one converted. Buffalo went twice: a 4th-and-5 throw wiped out
    by pass interference (a first down) and the 4th-and-2 touchdown. None
    of them in the go zone: every short fourth down came with the game
    decided or the half ending."""
    assert tuple(rows.loc["ARI", ["fourth_downs", "fourth_go", "fourth_conv"]]) == (7, 0, 0)
    assert tuple(rows.loc["NO", ["fourth_downs", "fourth_go", "fourth_conv"]]) == (9, 2, 1)
    assert tuple(rows.loc["BUF", ["fourth_downs", "fourth_go", "fourth_conv"]]) == (8, 2, 2)
    assert (rows["gz_downs"] == 0).all() and rows["gz_go_rate"].isna().all()


def test_neutral_pace_by_hand(rows):
    """Arizona's gaps between snaps of a drive, ending at a neutral snap:
    4 + 10 + 7 + 7 + 3 = 31 of them, 991 game-clock seconds in all (the
    third-quarter drive at wp above 0.8 and the fourth quarter are out)."""
    assert rows.loc["ARI", "pace_snaps"] == 31
    assert rows.loc["ARI", "pace_sec"] == pytest.approx(991 / 31)
    # New Orleans went no-huddle on a third of its snaps and played faster.
    assert rows.loc["NO", "pace_sec"] < rows.loc["ARI", "pace_sec"]
    assert rows.loc["NO", "no_huddle_rate"] == pytest.approx(24 / 73)


def test_epa_and_success_split(raw, rows):
    sc = raw[(raw["posteam"] == "BAL") & raw["down"].notna() & ((raw["pass"] == 1) | (raw["rush"] == 1))
             & (raw["two_point_attempt"] != 1)]
    bal = rows.loc["BAL"]
    assert bal["epa_play"] == pytest.approx(sc["epa"].mean())
    assert bal["pass_epa_play"] == pytest.approx(sc[sc["pass"] == 1]["epa"].mean())
    assert bal["rush_success_rate"] == pytest.approx(sc[sc["pass"] != 1]["success"].mean())


def test_allowed_is_the_same_game_from_the_defence(raw, rows):
    a = nap.allowed(nap.aggregate(raw))
    no_def = a[a["team"] == "NO"].iloc[0]
    assert no_def["opponent"] == "ARI" and no_def["coach"] == "Kellen Moore"
    assert no_def["home"] == 1 and no_def["spread"] == -6.0
    assert no_def["allowed_plays"] == 66 and no_def["allowed_proe"] == pytest.approx(rows.loc["ARI", "proe"])


# ------------------------------------------------------------ edge cases, made up

def game(*plays, home="KC", away="BUF", game_id="2030_01_BUF_KC", season=2030, scores=(24, 20)):
    """A whole made-up game: defaults for every column read, plays in order,
    30 game-clock seconds apart unless a play says otherwise."""
    base = dict(game_id=game_id, season=season, week=1, season_type="REG", posteam=home, defteam=away,
                home_team=home, away_team=away, home_coach="Andy Reid", away_coach="Sean McDermott",
                play_type="run", down=1, ydstogo=10, yardline_100=75, goal_to_go=0, qtr=1, wp=0.5,
                xpass=0.5, epa=0.1, success=1, shotgun=0, no_huddle=0, qb_kneel=0, qb_spike=0,
                two_point_attempt=0, rush_touchdown=0, pass_touchdown=0, fourth_down_converted=0,
                first_down_penalty=0, fixed_drive=1, series=1, spread_line=3.0,
                home_score=scores[0], away_score=scores[1], td_team=None, **{"pass": 0, "rush": 1})
    out, clock = [], 3000.0
    for i, p in enumerate(plays):
        row = dict(base)
        row.update(p)
        if "game_seconds_remaining" not in p:
            row["game_seconds_remaining"] = clock
        clock = row["game_seconds_remaining"] - 30
        row.setdefault("half_seconds_remaining", row["game_seconds_remaining"] - 1800
                       if row["game_seconds_remaining"] > 1800 else row["game_seconds_remaining"])
        row["play_id"] = row["order_sequence"] = 10 * (i + 1)
        out.append(row)
    return pd.DataFrame(out)


RUN = {"play_type": "run", "pass": 0, "rush": 1}
PASS = {"play_type": "pass", "pass": 1, "rush": 0}


def test_kneels_spikes_two_point_tries_and_presnap_fouls_are_not_plays():
    g = game(RUN,
             {"play_type": "qb_kneel", "qb_kneel": 1, "pass": 0, "rush": 1},       # some seasons flag rush
             {"play_type": "qb_spike", "qb_spike": 1, "pass": 1, "rush": 0},
             {**PASS, "down": None, "two_point_attempt": 1, "pass_touchdown": 1, "td_team": "KC"},
             {"play_type": "no_play", "pass": 0, "rush": 0},                       # false start
             {"play_type": "no_play", "pass": 0, "rush": 1},                       # holding on a run
             {"play_type": "no_play", "pass": 1, "rush": 0})                       # interference on a throw
    r = nap.aggregate(g).iloc[0]
    assert (r["plays"], r["pass_plays"], r["rush_plays"]) == (3, 1, 2)
    assert r["pass_td"] == 0                     # a two-point try is never a touchdown


def test_overtime_counts_but_is_never_neutral_nor_the_go_zone():
    ot = {"qtr": 5, "game_seconds_remaining": 500, "half_seconds_remaining": 500}
    g = game(RUN, {**PASS, **ot, "fixed_drive": 2, "series": 2},
             {**RUN, **ot, "game_seconds_remaining": 470, "half_seconds_remaining": 470, "down": 4,
              "ydstogo": 1, "yardline_100": 30, "fixed_drive": 2, "series": 2, "fourth_down_converted": 1})
    r = nap.aggregate(g).iloc[0]
    assert (r["plays"], r["early_plays"], r["neutral_plays"]) == (3, 2, 1)
    assert (r["fourth_downs"], r["fourth_go"], r["fourth_conv"], r["gz_downs"]) == (1, 1, 1, 0)


def test_a_play_with_no_xpass_counts_everywhere_but_proe():
    g = game({**PASS, "xpass": 0.6}, {**RUN, "xpass": None})
    r = nap.aggregate(g).iloc[0]
    assert (r["plays"], r["xpass_plays"]) == (2, 1)
    assert r["pass_rate"] == 0.5 and r["proe"] == pytest.approx(0.4) and r["xpass"] == pytest.approx(0.6)
    none = nap.aggregate(game({**PASS, "xpass": None}, {**RUN, "xpass": None})).iloc[0]
    assert none["xpass_plays"] == 0 and pd.isna(none["proe"]) and none["pass_rate"] == 0.5


def test_the_go_zone():
    """4th and 2 or less from the opponent's 45 in, wp 0.2-0.8, regulation,
    not the last two minutes of a half; one decision per series."""
    four = {"down": 4, "ydstogo": 1, "yardline_100": 30, "qtr": 2}
    g = game({**RUN, **four, "series": 1, "fourth_down_converted": 1},                   # in, went, converted
             {"play_type": "punt", "pass": 0, "rush": 0, **four, "yardline_100": 60, "series": 2},
             {"play_type": "field_goal", "pass": 0, "rush": 0, **four, "series": 4, "wp": 0.9},
             {"play_type": "no_play", "pass": 0, "rush": 0, **four, "series": 5},      # FG nullified...
             {"play_type": "field_goal", "pass": 0, "rush": 0, **four, "series": 5},   # ...and kicked again
             {"play_type": "no_play", "pass": 0, "rush": 1, **four, "series": 6,
              "first_down_penalty": 1},                                                 # run, flag: a first down
             {"play_type": "field_goal", "pass": 0, "rush": 0, **four, "series": 7,
              "game_seconds_remaining": 1900, "half_seconds_remaining": 100})          # the half ending
    r = nap.aggregate(g).iloc[0]
    assert (r["fourth_downs"], r["fourth_go"], r["fourth_conv"]) == (6, 2, 2)
    assert (r["gz_downs"], r["gz_go"], r["gz_conv"]) == (3, 2, 2)
    assert r["gz_go_rate"] == pytest.approx(2 / 3)
    # The nullified kick is no decision, so every series' first call is its last.
    assert (r["gz_first_downs"], r["gz_first_go"]) == (3, 2)


def test_the_first_call_of_a_series_is_kept_beside_the_one_that_stood():
    """A run on 4th and 1 at the 40 wiped out by holding, then a punt on 4th
    and 11: the decision that stood is a punt outside the go zone; the call
    the coach made was a go in it."""
    four = {"down": 4, "qtr": 2, "series": 3}
    g = game({"play_type": "no_play", "pass": 0, "rush": 1, **four, "ydstogo": 1, "yardline_100": 40},
             {"play_type": "punt", "pass": 0, "rush": 0, **four, "ydstogo": 11, "yardline_100": 50})
    r = nap.aggregate(g).iloc[0]
    assert (r["fourth_downs"], r["fourth_go"], r["gz_downs"]) == (1, 0, 0) and pd.isna(r["gz_go_rate"])
    assert (r["gz_first_downs"], r["gz_first_go"], r["gz_first_go_rate"]) == (1, 1, 1.0)


def test_pace_counts_gaps_between_snaps_of_one_drive():
    g = game({**RUN, "game_seconds_remaining": 3000},
             {**PASS, "game_seconds_remaining": 2970},                                  # 30
             {**RUN, "game_seconds_remaining": 2965},                                   # 5
             {"play_type": "qb_spike", "qb_spike": 1, "pass": 0, "rush": 0,
              "game_seconds_remaining": 2950},                                          # a snap, not a play
             {**RUN, "game_seconds_remaining": 2949},                                   # 1
             {**RUN, "game_seconds_remaining": 2880},                                   # 69: a missing row
             {**RUN, "game_seconds_remaining": 2860, "wp": 0.9},                        # not neutral
             {**PASS, "game_seconds_remaining": 2800, "fixed_drive": 2},               # a new drive
             {**PASS, "game_seconds_remaining": 2780, "fixed_drive": 2, "down": 3})    # 20, any down
    r = nap.aggregate(g).iloc[0]
    assert r["pace_snaps"] == 4 and r["pace_sec"] == pytest.approx((30 + 5 + 1 + 20) / 4)


def test_only_the_offences_touchdowns_count():
    g = game({**PASS, "yardline_100": 15, "pass_touchdown": 1, "td_team": "BUF"},     # not Kansas City's
             {**RUN, "yardline_100": 3, "goal_to_go": 1, "rush_touchdown": 1, "td_team": "KC",
              "fixed_drive": 2},
             {**PASS, "yardline_100": 40, "pass_touchdown": 1, "td_team": "KC", "fixed_drive": 3})
    r = nap.aggregate(g).iloc[0]
    assert (r["rush_td"], r["pass_td"], r["rz_rush_td"], r["gtg_rush_td"], r["rz_pass_td"]) == (1, 1, 1, 1, 0)
    assert (r["drives"], r["rz_drives"], r["rz_td_drives"]) == (3, 2, 1)


def test_old_codes_map_to_the_franchise_and_the_coach_keeps_his_name():
    g = pd.concat([game(RUN, home="OAK", away="SD", game_id="2015_01_SD_OAK", season=2015),
                   game({**RUN, "posteam": "SD", "defteam": "OAK"}, home="OAK", away="SD",
                        game_id="2015_01_SD_OAK", season=2015).assign(play_id=999, order_sequence=999)])
    r = nap.aggregate(g).set_index("team")
    assert sorted(r.index) == ["LAC", "LV"]
    assert r.loc["LV", "opponent"] == "LAC" and r.loc["LAC", "coach"] == "Sean McDermott"
    assert r.loc["LAC", "spread"] == -3.0 and r.loc["LAC", "home"] == 0


def test_jim_mora_father_and_son_are_two_coaches():
    """nflverse spells both "Jim Mora": the father coached Indianapolis to
    2001, the son Atlanta from 2004 and Seattle in 2009."""
    sr = game(RUN, home="IND", away="NE", game_id="2001_01_NE_IND", season=2001)
    jr = game(RUN, home="ATL", away="NE", game_id="2004_01_NE_ATL", season=2004)
    both = pd.concat([sr, jr]).assign(home_coach="Jim Mora", away_coach="Bill Belichick")
    away = both.assign(posteam="NE", play_id=both["play_id"] + 1000, order_sequence=both["play_id"] + 1000)
    r = nap.aggregate(pd.concat([both, away])).set_index(["game_id", "team"])
    assert r.loc[("2001_01_NE_IND", "IND"), "coach"] == "Jim Mora Sr."
    assert r.loc[("2004_01_NE_ATL", "ATL"), "coach"] == "Jim Mora Jr."
    assert r.loc[("2004_01_NE_ATL", "NE"), "opp_coach"] == "Jim Mora Jr."
    assert r.loc[("2001_01_NE_IND", "NE"), "coach"] == "Bill Belichick"
    assert list(nap.coach_names(["Jim Mora", "Andy Reid", None], [2009, 2009, 2009])) == [
        "Jim Mora Jr.", "Andy Reid", None]


def test_player_rows_count_the_plays_that_stood_near_the_goal_line():
    goal = {"yardline_100": 8, "goal_to_go": 1}
    g = game({**RUN, **goal, "rusher_player_id": "RB1"},
             {**PASS, **goal, "receiver_player_id": "WR1"},
             {**PASS, **goal, "play_type": "run", "rusher_player_id": "QB1"},          # a scramble
             {**PASS, **goal, "receiver_player_id": "TE1", "pass_touchdown": 1, "td_team": "KC",
              "td_player_id": "TE1"},
             {**PASS, "down": None, "two_point_attempt": 1, "yardline_100": 2,
              "receiver_player_id": "WR1"},                                              # a try: no target
             {"play_type": "qb_kneel", "qb_kneel": 1, "pass": 0, "rush": 1, **goal,
              "rusher_player_id": "QB1"},                                                # a kneel: no carry
             {**RUN, "yardline_100": 30, "rusher_player_id": "RB1"},                    # outside the 20
             {**RUN, "yardline_100": 15, "rusher_player_id": "RB2", "rush_touchdown": 1, "td_team": "BUF",
              "td_player_id": "LB1"})                                                    # returned the other way
    p = nap.aggregate_players(g).set_index("player_id")
    assert sorted(p.index) == ["QB1", "RB1", "RB2", "TE1", "WR1"]
    assert tuple(p.loc["RB1", ["rz_carries", "gtg_carries"]]) == (1, 1)
    assert tuple(p.loc["QB1", ["rz_carries", "gtg_carries", "rz_targets"]]) == (1, 1, 0)
    assert tuple(p.loc["WR1", ["rz_targets", "gtg_targets"]]) == (1, 1)
    assert tuple(p.loc["TE1", ["rz_targets", "rz_rec_td", "gtg_rec_td", "rz_rush_td"]]) == (1, 1, 1, 0)
    assert tuple(p.loc["RB2", ["rz_carries", "gtg_carries", "rz_rush_td"]]) == (1, 0, 0)
    assert (p["team"] == "KC").all() and nap.aggregate_players(game(RUN)).empty


def test_a_game_still_being_played_is_left_out(raw):
    cut = raw[~((raw["game_id"] == BAL_BUF) & (raw["qtr"] >= 3))]
    assert nap.complete_games(cut) == {ARI_NO}
    no_score = raw.assign(home_score=np.where(raw["game_id"] == ARI_NO, np.nan, raw["home_score"]))
    assert nap.complete_games(no_score) == {BAL_BUF}


# ------------------------------------------------------------ the file

def test_the_file_is_deterministic_with_no_nan_strings(raw, tmp_path):
    rows = nap.aggregate(raw)
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    nap.write(rows, a)
    nap.write(rows.sample(frac=1, random_state=1), b)          # any order in, one order out
    assert a.read_bytes() == b.read_bytes()
    text = gzip.decompress(a.read_bytes()).decode()
    assert "nan" not in text.lower() and "None" not in text
    header, first = text.splitlines()[:2]
    assert header.split(",") == nap.COLUMNS
    cells = dict(zip(header.split(","), first.split(",")))
    assert cells["plays"] == "66" and cells["gz_go_rate"] == "" and cells["pace_sec"] == "31.97"
    back = nap.load_approach(a)
    assert list(back.columns) == nap.COLUMNS and len(back) == 4
    assert back.loc[back["team"] == "ARI", "pass_rate"].item() == pytest.approx(41 / 66, abs=5e-5)
    assert back["gz_go_rate"].isna().all() and back["coach"].tolist()[0] == "Jonathan Gannon"


def test_the_player_file_is_deterministic_too(raw, tmp_path):
    p = nap.aggregate_players(raw)
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    nap.write_players(p, a)
    nap.write_players(p.sample(frac=1, random_state=2), b)
    assert a.read_bytes() == b.read_bytes()
    text = gzip.decompress(a.read_bytes()).decode()
    assert text.splitlines()[0].split(",") == nap.PLAYER_COLUMNS and "nan" not in text.lower()
    back = nap.load_players(a)
    assert len(back) == len(p) and back["player_id"].iloc[0] == "00-0033553"   # ids stay text
    assert int(back["rz_carries"].sum()) == int(p["rz_carries"].sum())


def test_load_before_the_first_sync(tmp_path):
    empty = nap.load_approach(tmp_path / "none.csv.gz")
    assert list(empty.columns) == nap.COLUMNS and empty.empty
    none = nap.load_players(tmp_path / "none_players.csv.gz")
    assert list(none.columns) == nap.PLAYER_COLUMNS and none.empty
    assert nap.players_path(nap.APPROACH) == nap.PLAYERS


def test_a_write_cut_short_leaves_the_old_file_whole(raw, tmp_path, monkeypatch):
    """The rows go to a temporary file renamed into place: if the run dies
    before the rename, the old file is untouched and the temporary one gone."""
    path = tmp_path / "approach.csv.gz"
    nap.write(nap.aggregate(raw), path)
    before = path.read_bytes()

    def killed(src, dst):
        raise KeyboardInterrupt
    monkeypatch.setattr(nap.os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        nap.write(nap.aggregate(raw).head(1), path)
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["approach.csv.gz"]


def test_a_short_transfer_is_caught():
    nap.check_length(b"x" * 10, "10")
    nap.check_length(b"x" * 10, None)
    nap.check_length(b"x" * 10, "99", "gzip")       # encoded length: not comparable
    with pytest.raises(nap.Incomplete):
        nap.check_length(b"x" * 10, "11")


# ------------------------------------------------------------ the sync

@pytest.fixture
def body():
    return FIXTURE.read_bytes()


def test_sync_writes_a_season_and_leaves_an_unchanged_one_alone(body, tmp_path):
    path, lines = tmp_path / "approach.csv.gz", []
    players = tmp_path / "approach_players.csv.gz"
    got = nap.sync(fetch=lambda url: body, path=path, now=2025, log=lines.append)
    assert got["changed"] == [2025] and got["rows"] == 4 and got["player_rows"] == 23
    assert path.exists() and players.exists()
    # Shotgun on 210 of the 268 scrimmage plays, no huddle on 47.
    assert "xpass 100%, shotgun 78%, no huddle 18%, rusher ids 100%" in lines[0]
    before, before_players = path.read_bytes(), players.read_bytes()
    again = nap.sync(fetch=lambda url: body, path=path, now=2025, log=lines.append)
    assert again["unchanged"] == [2025] and not again["written"]
    assert path.read_bytes() == before and players.read_bytes() == before_players


def test_a_season_with_no_shotgun_flags_did_not_record_them(raw, tmp_path):
    """Before nflverse recorded formations a season reads shotgun = 0 on
    every play: its rates are left empty, not written as 0."""
    lines = []
    bare = body_of(raw.assign(shotgun=0, no_huddle=0))
    nap.sync(fetch=lambda url: bare, path=tmp_path / "a.csv.gz", now=2025, log=lines.append)
    a = nap.load_approach(tmp_path / "a.csv.gz")
    assert a["shotgun_rate"].isna().all() and a["no_huddle_rate"].isna().all()
    assert a["neutral_no_huddle_rate"].isna().all() and a["pass_rate"].notna().all()
    assert "shotgun and no huddle not recorded, left empty" in lines[0]


@pytest.mark.parametrize("column", ["fixed_drive", "series", "goal_to_go", "shotgun", "no_huddle", "success",
                                    "first_down_penalty", "fourth_down_converted", "td_team", "home_score",
                                    "away_score", "rusher_player_id", "receiver_player_id", "td_player_id"])
def test_a_column_that_changes_a_number_is_required(raw, body, tmp_path, column):
    """Without it a season would be written with zeros where it has numbers
    (or, without the scores, with games still being played)."""
    path = tmp_path / "approach.csv.gz"
    nap.sync(fetch=lambda url: body, path=path, now=2025, log=lambda _: None)
    before = path.read_bytes()
    got = nap.sync(fetch=lambda url: body_of(raw.drop(columns=[column])), path=path, now=2025,
                   log=lambda _: None)
    assert column in got["kept"][2025] and path.read_bytes() == before


def test_a_download_that_lost_a_game_on_file_is_kept(raw, body, tmp_path):
    """As many whole games as on file is not enough: every game on file
    must still be whole in the download."""
    path = tmp_path / "approach.csv.gz"
    nap.sync(fetch=lambda url: body, path=path, now=2025, log=lambda _: None)
    before = path.read_bytes()
    bal_buf = raw[raw["game_id"] == BAL_BUF]
    new = bal_buf.assign(game_id="2025_02_BAL_CLE", week=2, home_team="CLE", away_team="BAL",
                         posteam=bal_buf["posteam"].map({"BUF": "CLE", "BAL": "BAL"}))
    got = nap.sync(fetch=lambda url: body_of(pd.concat([bal_buf, new])), path=path, now=2025,
                   log=lambda _: None)
    assert "1 games on file not whole in the download (2025_01_ARI_NO)" in got["kept"][2025]
    assert path.read_bytes() == before
    # With every game on file still whole, a new week is added.
    grown = nap.sync(fetch=lambda url: body_of(pd.concat([raw, new])), path=path, now=2025,
                     log=lambda _: None)
    assert grown["changed"] == [2025] and grown["rows"] == 6


def test_what_is_on_file_survives_a_failed_cut_or_short_download(raw, body, tmp_path):
    path, players = tmp_path / "approach.csv.gz", tmp_path / "approach_players.csv.gz"
    nap.sync(fetch=lambda url: body, path=path, now=2025, log=lambda _: None)
    before, before_players = path.read_bytes(), players.read_bytes()

    def fails(url):
        raise OSError("404")
    one_game = body_of(raw[raw["game_id"] == ARI_NO])          # Buffalo's game, on file, is gone
    no_coach = body_of(raw.drop(columns=["home_coach"]))
    other_season = body_of(raw.assign(season=2024))
    for fetch in (fails, lambda url: body[: len(body) // 2], lambda url: b"<html>rate limited</html>",
                  lambda url: one_game, lambda url: no_coach, lambda url: other_season):
        got = nap.sync(fetch=fetch, path=path, now=2025, log=lambda _: None)
        assert list(got["kept"]) == [2025] and not got["written"]
        assert path.read_bytes() == before and players.read_bytes() == before_players


def test_a_finished_season_needs_its_games(body, tmp_path):
    got = nap.sync(seasons=[2025], fetch=lambda url: body, path=tmp_path / "a.csv.gz", now=2026,
                   log=lambda _: None)
    assert "whole games in a finished season" in got["kept"][2025]


def test_only_the_seasons_that_changed_are_rewritten(raw, body, tmp_path):
    """Another season's lines are carried over exactly as they were on file,
    even in a formatting this version would not write; a season whose
    numbers moved is replaced."""
    path = tmp_path / "approach.csv.gz"
    nap.sync(fetch=lambda url: body, path=path, now=2025, log=lambda _: None)
    text = gzip.decompress(path.read_bytes()).decode()
    header = text.splitlines()[0]
    odd = dict.fromkeys(nap.COLUMNS, "")
    odd.update(season="2024", week="3", season_type="REG", game_id="2024_03_KC_ATL", team="KC",
               opponent="ATL", home="0", coach="Andy Reid", opp_coach="Raheem Morris", plays="64",
               pass_rate="0.50000")
    line = ",".join(odd[c] for c in header.split(","))
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write((header + "\n" + line + "\n" + "\n".join(text.splitlines()[1:]) + "\n").encode())
    moved = raw.copy()
    moved.loc[(moved["game_id"] == ARI_NO) & (moved["play_id"] == 63), "epa"] = 1.5
    got = nap.sync(fetch=lambda url: body_of(moved), path=path, now=2025, log=lambda _: None)
    assert got["changed"] == [2025]
    after = gzip.decompress(path.read_bytes()).decode().splitlines()
    assert after[1] == line                                   # 2024 sorts first, untouched
    assert len(after) == 6
    ari = nap.load_approach(path).set_index(["season", "team"]).loc[(2025, "ARI")]
    assert ari["rush_epa_play"] > nap.aggregate(raw).set_index("team").loc["ARI", "rush_epa_play"]


def test_a_change_in_the_player_rows_alone_rewrites_only_that_file(raw, body, tmp_path):
    path, players = tmp_path / "approach.csv.gz", tmp_path / "approach_players.csv.gz"
    nap.sync(fetch=lambda url: body, path=path, now=2025, log=lambda _: None)
    before = path.read_bytes()
    moved = raw.copy()
    cook = moved["rusher_player_name"] == "J.Cook"
    moved.loc[cook, "rusher_player_id"] = "00-0099999"          # nflverse re-keys a player
    got = nap.sync(fetch=lambda url: body_of(moved), path=path, now=2025, log=lambda _: None)
    assert got["changed"] == [2025] and path.read_bytes() == before
    assert "00-0099999" in set(nap.load_players(players)["player_id"])


def test_full_means_every_season_since_1999_one_at_a_time(tmp_path):
    asked = []

    def fetch(url):
        asked.append(url)
        raise OSError("offline")
    got = nap.sync(full=True, fetch=fetch, path=tmp_path / "a.csv.gz", now=2001, log=lambda _: None)
    assert asked == [nap.URL.format(season=s) for s in (1999, 2000, 2001)]
    assert asked[0] == ("https://github.com/nflverse/nflverse-data/releases/download/pbp/"
                        "play_by_play_1999.csv.gz")
    assert sorted(got["kept"]) == [1999, 2000, 2001] and not (tmp_path / "a.csv.gz").exists()
    assert nap.sync(fetch=fetch, path=tmp_path / "a.csv.gz", now=2026, log=lambda _: None)["seasons"] == [2026]


# ------------------------------------------------------------ the runner

def test_predict_refreshes_this_seasons_approach_and_survives_its_failure(monkeypatch):
    import nfl_predict as npd

    class Stop(Exception):
        pass
    calls = []

    def stop():
        raise Stop
    monkeypatch.setattr(npd.nd, "sync", lambda log=print: calls.append("data"))
    monkeypatch.setattr(npd.nd, "load_games", stop)
    monkeypatch.setattr(npd.nap, "sync", lambda log=print: calls.append("approach"))
    with pytest.raises(Stop):
        npd.run(sync=True, log=lambda *_: None)
    assert calls == ["data", "approach"]

    def broken(log=print):
        raise RuntimeError("nflverse down")
    monkeypatch.setattr(npd.nap, "sync", broken)
    lines = []
    with pytest.raises(Stop):                 # reached the picks all the same
        npd.run(sync=True, log=lines.append)
    assert any("[approach] failed" in line for line in lines)


# ------------------------------------------------------------ the snapshot's health

@pytest.mark.snapshot
def test_the_snapshot_covers_every_finished_game(games):
    if not nap.APPROACH.exists():
        pytest.skip("approach.csv.gz is written by the runner's first sync")
    a = nap.load_approach()
    per_game = a.groupby("game_id").size()
    assert (per_game == 2).all()
    assert set(a["team"]) | set(a["opponent"]) <= set(nd.TEAMS)
    assert a["coach"].notna().all() and (a["plays"] >= nap.MIN_TEAM_PLAYS).all()
    for c in ("pass_rate", "rz_rush_rate", "gz_go_rate", "shotgun_rate", "success_rate"):
        v = a[c].dropna()
        assert ((v >= 0) & (v <= 1)).all(), c
    cutoff = games["kickoff"][games["played"]].max() - pd.Timedelta(days=4)
    done = games[games["played"] & (games["kickoff"] < cutoff)
                 & games["season"].isin(a["season"].unique())]
    missing = sorted(set(done["game_id"]) - set(per_game.index))
    assert len(missing) <= 5, missing[:20]
    if nap.PLAYERS.exists():
        p = nap.load_players()
        assert set(zip(p["game_id"], p["team"])) <= set(zip(a["game_id"], a["team"]))
        mine = p.groupby(["game_id", "team"])[["rz_rush_td", "rz_rec_td"]].sum().add_prefix("players_")
        team = a.set_index(["game_id", "team"])[["rz_rush_td", "rz_pass_td"]]
        both = team.join(mine, how="left").fillna(0)
        agree = ((both["players_rz_rush_td"] == both["rz_rush_td"])
                 & (both["players_rz_rec_td"] == both["rz_pass_td"]))
        assert agree.mean() >= 0.99, both[~agree].head(20)
