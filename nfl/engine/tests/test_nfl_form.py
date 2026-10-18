"""The record and last five beside each team's name."""

import pandas as pd

import nfl_form as nform


def _games():
    rows = [
        ("2025_20_KC_BUF", 2025, 20, "DIV", "2026-01-18 20:00", "BUF", "KC", 27, 24),
        ("2026_01_KC_LAC", 2026, 1, "REG", "2026-09-10 00:15", "LAC", "KC", 20, 23),
        ("2026_02_BUF_KC", 2026, 2, "REG", "2026-09-20 17:00", "KC", "BUF", 31, 31),
        ("2026_03_KC_DEN", 2026, 3, "REG", "2026-09-27 20:25", "DEN", "KC", 10, 30),
        ("2026_04_LV_KC", 2026, 4, "REG", "2026-10-04 17:00", "KC", "LV", None, None),
    ]
    g = pd.DataFrame(rows, columns=["game_id", "season", "week", "game_type", "kickoff",
                                    "home", "away", "home_score", "away_score"])
    g["kickoff"] = pd.to_datetime(g["kickoff"])
    g["played"] = g["home_score"].notna()
    g["neutral"] = 0
    return g


def test_record_counts_only_this_regular_season_before_the_game():
    f = nform.form(_games(), "KC", pd.Timestamp("2026-10-04 17:00"), 2026)
    assert f["record"] == {"wins": 2, "losses": 0, "ties": 1}
    assert "playoffs" not in f


def test_last_five_newest_first_with_margins_and_reaching_back_a_season():
    f = nform.form(_games(), "KC", pd.Timestamp("2026-10-04 17:00"), 2026)
    assert [b["result"] for b in f["last5"]] == ["W", "T", "W", "L"]
    assert [b["code"] for b in f["last5"]] == ["20", "T", "3", "3"]
    assert f["last5"][0]["score"] == "30-10" and f["last5"][0]["home"] is False


def test_a_game_never_appears_beside_itself():
    kick = pd.Timestamp("2026-09-27 20:25")
    f = nform.form(_games(), "KC", kick, 2026)
    assert all(b["date"] < "2026-09-27" for b in f["last5"])
