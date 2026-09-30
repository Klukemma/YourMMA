"""Camp power may only count what the camp had done before the fight."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from camp_strength import CampIndex

PROFILES = [
    {"fighter": "Carl Brown",
     "affiliations": [{"canonical": "Elite Gym", "start": 2015, "end": None,
                       "dated": True},
                      {"canonical": "Old Gym", "start": 2010, "end": 2014,
                       "dated": True},
                      {"canonical": "Undated Gym", "start": None, "end": None,
                       "dated": False}],
     "trainers": [{"canonical": "Champ Coach"}]},
    {"fighter": "Star Teammate",
     "affiliations": [{"canonical": "Elite Gym", "start": 2012, "end": None,
                       "dated": True}],
     "trainers": []},
]
ARCHIVE = pd.DataFrame([
    ("2016-03-01", "Star Teammate", "X1", "Star Teammate", 0),
    ("2016-09-01", "Star Teammate", "X2", "Star Teammate", 1),   # title win
    ("2018-01-01", "Star Teammate", "X3", "Star Teammate", 1),
    ("2013-05-01", "Champ Coach", "Y1", "Champ Coach", 1),
], columns=["date", "r_name", "b_name", "winner", "title_fight"])


def index():
    return CampIndex(PROFILES, ARCHIVE)


def test_the_gym_is_the_one_the_fighter_was_at_on_that_date():
    camp = index()
    assert camp.gyms("Carl Brown", "2013-01-01") == ["old gym"]
    assert camp.gyms("Carl Brown", "2017-01-01") == ["elite gym"]


def test_an_undated_listing_never_places_a_fighter_in_the_past():
    camp = index()
    assert "undated gym" not in camp.gyms("Carl Brown", "2017-01-01")


def test_strength_counts_only_results_before_the_fight():
    camp = index()
    before_title = camp.at("Carl Brown", "2016-06-01")
    assert (before_title["bouts"], before_title["titles"]) == (1, 0)
    after_title = camp.at("Carl Brown", "2017-01-01")
    assert (after_title["bouts"], after_title["titles"]) == (2, 1)
    # The 2018 title win is invisible to a 2017 fight.
    assert camp.at("Carl Brown", "2018-01-01")["titles"] == 1


def test_a_champion_coach_is_one_who_had_won_a_title_by_then():
    camp = index()
    assert not camp.at("Carl Brown", "2013-01-01")["champion_coach"]
    assert camp.at("Carl Brown", "2013-06-01")["champion_coach"]


def test_a_fighter_with_no_profile_is_a_neutral_camp():
    got = index().at("Nobody", "2020-01-01")
    assert got["gyms"] == [] and got["rate"] == 0.5 and got["titles"] == 0
