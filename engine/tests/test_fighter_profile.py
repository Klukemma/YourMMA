"""Camp, coaches and credentials from a fighter's infobox."""

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fighter_profile as fp

PAGE = (
    "{{Infobox martial artist\n"
    "| name = Carl Brown\n"
    "| image = Brown.jpg\n"
    "| birth_date = {{birth date and age|1990|1|2}}\n"
    "| height = {{convert|6|ft|0|in|cm|abbr=on}}\n"
    "| team = [[American Top Team]] (2015–present)<br />"
    "[[Team Alpha Male]] (2010–2015)<br>Local Gym (2008)<br>"
    "[[American Kickboxing Academy|AKA]]\n"
    "| trainer = [[Mike Brown (fighter)|Mike Brown]]<br>Dan Lambert\n"
    "| rank = [[Black belt]] in [[Brazilian jiu-jitsu]] under "
    "[[Ricardo Liborio]]<ref>x</ref><br>Black belt in Judo\n"
    "| fightingoutof = [[Coconut Creek, Florida]]\n"
    "}}\n"
    "'''Carl Brown''' is an American mixed martial artist.\n"
)


def test_infobox_fields_survive_nested_templates():
    box = fp.infobox(PAGE)
    assert box["name"] == "Carl Brown"
    assert box["birth_date"].startswith("{{birth date and age")
    assert "American Top Team" in box["team"]
    assert box["fightingoutof"] == "[[Coconut Creek, Florida]]"


def test_affiliations_are_dated_where_the_years_are_stated():
    got = {a["canonical"]: a for a in fp.profile(PAGE)["affiliations"]}
    assert got["American Top Team"]["start"] == 2015
    assert got["American Top Team"]["end"] is None          # present
    assert (got["Team Alpha Male"]["start"],
            got["Team Alpha Male"]["end"]) == (2010, 2015)
    assert (got["Local Gym"]["start"], got["Local Gym"]["end"]) == \
        (2008, 2008)


def test_an_undated_team_is_marked_so_it_cannot_inform_the_past():
    got = {a["canonical"]: a for a in fp.profile(PAGE)["affiliations"]}
    aka = got["American Kickboxing Academy"]
    assert aka["team"] == "AKA" and not aka["dated"]


def test_trainers_by_link_target():
    names = [(p["name"], p["canonical"]) for p in fp.profile(PAGE)["trainers"]]
    assert ("Mike Brown", "Mike Brown (fighter)") in names
    assert ("Dan Lambert", "Dan Lambert") in names


def test_ranks_keep_belt_art_and_lineage():
    ranks = fp.profile(PAGE)["ranks"]
    bjj = ranks[0]
    assert bjj["belt"] == "black"
    assert bjj["art"] == "Brazilian jiu-jitsu"
    assert bjj["under"] == "Ricardo Liborio"
    assert ranks[1]["art"] == "Judo"


def test_no_infobox_is_an_empty_profile():
    got = fp.profile("Just prose about a fighter.")
    assert got["affiliations"] == [] and got["trainers"] == []
