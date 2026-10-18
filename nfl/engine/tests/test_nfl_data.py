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


@pytest.mark.snapshot
def test_every_completed_game_has_team_stats(games, team_games):
    """nflverse posts results within hours and statistics overnight, so the
    last few days' finals may have none yet; anything older must."""
    have = team_games.groupby("game_id").size()
    cutoff = games["kickoff"][games["played"]].max() - pd.Timedelta(days=3)
    done = games[games["played"] & (games["season"] >= 2000) & (games["kickoff"] < cutoff)]
    missing = done[~done["game_id"].isin(have.index)]
    assert len(missing) <= 5, missing["game_id"].tolist()


def test_team_stat_rows_are_offense(team_games):
    sample = team_games.dropna(subset=["plays"])
    assert (sample["plays"] >= 0).all()
    assert sample["plays"].median() > 50


def test_a_missing_statistics_file_keeps_the_rest_of_the_sync(monkeypatch, tmp_path):
    """Week 1: the schedule and the injury report exist, the season's
    statistics do not yet. The injuries must still arrive and the
    statistics on file must survive."""
    import io
    for name in ("GAMES", "TEAM_GAMES", "QB_GAMES", "INJURIES"):
        monkeypatch.setattr(nd, name, tmp_path / f"{name.lower()}.csv.gz")
    old_team = pd.DataFrame({"season": [2025], "week": [1], "game_id": ["g"], "team": ["KC"]})
    nd._write(old_team, nd.TEAM_GAMES)
    games_csv = ("game_id,season,week,gameday,gametime,home_team,away_team,home_score,away_score,spread_line\n"
                 "2026_01_KC_BUF,2026,1,2026-09-13,13:00,BUF,KC,,,3\n")
    injuries_csv = "season,game_type,team,week,gsis_id,position,full_name,report_primary_injury,report_status,practice_status\n2026,REG,BUF,1,x,QB,Q B,Ankle,Out,DNP\n"

    def fake_read(url, **kw):
        if url == nd.GAMES_URL:
            return pd.read_csv(io.StringIO(games_csv))
        if "injuries" in url:
            return pd.read_csv(io.StringIO(injuries_csv))
        raise OSError("404 Not Found")
    monkeypatch.setattr(nd, "_read", fake_read)
    lines = []
    result = nd.sync(seasons=[2026], log=lines.append)
    assert result["fetched"]["injuries"] == [2026] and result["fetched"]["team"] == []
    assert len(nd.load_injuries(nd.INJURIES)) == 1
    assert pd.read_csv(nd.TEAM_GAMES)["season"].tolist() == [2025]
    assert any("keeping what is on file" in l for l in lines)
