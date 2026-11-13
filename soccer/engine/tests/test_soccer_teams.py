"""A club's badge: its own code and colours where known, and where not, a
neutral chip whose code no other club of its league shows - the same codes
whatever order the clubs are asked in."""

import collections

import pandas as pd
import pytest

import soccer_teams as st


def _row(league, season, home, away):
    return {"league": league, "season": season, "home": home, "away": away}


def test_clubs_whose_first_letters_clash_get_codes_unique_in_their_league():
    clashes = {
        "SC0": ["Dundee", "Dundee United", "Dunfermline", "Hearts"],
        "B1": ["Genk", "Gent", "Gen Club"],
        "USA": ["New York City", "New York Red Bulls", "New England", "Newcastle Jets"],
        "CHN": ["Shanghai Port", "Shanghai Shenhua", "Shanghai SIPG", "Shanghai Shenxin"],
    }
    for league, clubs in clashes.items():
        got = st.codes_for(clubs)
        codes = [got[c]["code"] for c in clubs]
        assert len(set(codes)) == len(codes), (league, codes)
        for c in clubs:
            assert c not in st.TEAMS and (got[c]["bg"], got[c]["fg"]) == st.NEUTRAL and got[c]["name"] == c
            assert len(got[c]["code"]) == 3 and got[c]["code"] in st.code_candidates(c)
    # the first in name order keeps its first three letters, as every club showed before
    got = st.codes_for(clashes["SC0"])
    assert got["Dundee"]["code"] == "DUN" == st.display("Dundee")["code"]
    assert st.codes_for(["Genk", "Gent"])["Genk"]["code"] == "GEN"


def test_the_codes_are_the_same_whatever_order_the_clubs_come_in():
    clubs = ["New York City", "New York Red Bulls", "New England", "Arsenal", "Newcastle", "Nottingham"]
    ref = st.codes_for(clubs)
    for k in range(len(clubs)):
        turned = clubs[k:] + clubs[:k]
        assert st.codes_for(turned) == ref and st.codes_for(reversed(turned)) == ref
    assert st.codes_for(clubs + clubs) == ref                   # asked twice: once


def test_a_known_club_keeps_its_own_code_and_a_missing_one_steps_round_it():
    # Newcastle (NEW, from TEAMS) is in the league: the missing New England cannot take NEW
    got = st.codes_for(["Newcastle", "New England"])
    assert got["Newcastle"] == st.display("Newcastle") and got["Newcastle"]["code"] == "NEW"
    assert got["New England"]["code"] != "NEW"
    assert st.display("New England", taken={"NEW"})["code"] == got["New England"]["code"]
    # nothing free among the letters: two letters and a digit, still unique
    taken = set(st.code_candidates("Ab")[:-1])
    assert st.display("Ab", taken=taken)["code"] not in taken


def test_every_league_gets_its_codes_from_every_club_it_ever_had():
    frame = pd.DataFrame([_row("SC0", 2024, "Dundee", "Hearts"), _row("SC0", 2025, "Dundee United", "Hearts"),
                          _row("B1", 2025, "Genk", "Gent"), _row("SC0", 2026, "Dundee United", None)])
    got = st.by_league(frame)
    assert set(got) == {"SC0", "B1"} and set(got["SC0"]) == {"Dundee", "Dundee United", "Hearts"}
    # Dundee United's code is the same in a season Dundee is not in: the league's, not the season's
    assert got["SC0"]["Dundee United"]["code"] != got["SC0"]["Dundee"]["code"]
    assert got["SC0"] == st.codes_for(["Dundee", "Dundee United", "Hearts"])


@pytest.mark.snapshot
def test_no_two_clubs_of_one_league_share_a_code_on_the_snapshot():
    import soccer_data as sd
    frame = pd.concat([sd.load_matches(leagues="all"), sd.load_fixtures(leagues="all")], ignore_index=True)
    codes = st.by_league(frame)
    assert set(codes) == set(frame["league"])
    for league, clubs in codes.items():
        dup = {c: n for c, n in collections.Counter(d["code"] for d in clubs.values()).items() if n > 1}
        assert not dup, (league, dup)
