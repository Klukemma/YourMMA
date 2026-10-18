"""The snapshot and its loaders: franchise codes, kickoffs, line convention."""

import pandas as pd
import pytest

import nfl_data as nd


def test_moved_franchises_keep_one_history():
    assert nd.franchise("OAK") == "LV" and nd.franchise("SD") == "LAC" and nd.franchise("STL") == "LA"
    assert nd.franchise("KC") == "KC"


def test_home_city_follows_the_code_the_season_used():
    # The Rams were in St. Louis (Central) before 2016 and are in Los Angeles now.
    assert nd.home_offset("STL") == -6 and nd.home_offset("LA") == -8


@pytest.mark.parametrize("day,clock,utc", [
    ("2026-10-18", "13:00", "2026-10-18 17:00"),   # daylight saving: UTC-4
    ("2026-11-08", "13:00", "2026-11-08 18:00"),   # standard time: UTC-5
    ("2026-10-18", "09:30", "2026-10-18 13:30"),   # a London morning kickoff
    ("2026-10-18", None, "2026-10-18 17:00"),      # no time posted yet
])
def test_kickoffs_are_utc(day, clock, utc):
    assert nd.kickoff_utc(day, clock) == pd.Timestamp(utc)


def test_games_load_in_kickoff_order_with_known_teams(games):
    assert games["kickoff"].is_monotonic_increasing
    assert set(games["home"]) | set(games["away"]) <= set(nd.TEAMS)
    assert len(nd.TEAMS) == 32
    assert games["season"].min() == nd.FIRST_SEASON


def test_spread_line_is_the_home_sides_expected_margin(games):
    played = games[games["played"] & games["spread_line"].notna()]
    # Favourites win far more often than not: the sign convention is right.
    agree = ((played["spread_line"] > 0) == (played["result"] > 0))[played["spread_line"] != 0]
    assert agree.mean() > 0.6


def test_played_means_both_scores(games):
    played = games[games["played"]]
    assert played["home_score"].notna().all() and played["away_score"].notna().all()
    assert (played["result"] == played["home_score"] - played["away_score"]).all()


def test_every_completed_game_has_team_stats(games, team_games):
    have = team_games.groupby("game_id").size()
    done = games[games["played"] & (games["season"] >= 2000)]
    missing = done[~done["game_id"].isin(have.index)]
    assert len(missing) <= 5, missing["game_id"].tolist()


def test_team_stat_rows_are_offense(team_games):
    sample = team_games.dropna(subset=["plays"])
    assert (sample["plays"] >= 0).all()
    assert sample["plays"].median() > 50
