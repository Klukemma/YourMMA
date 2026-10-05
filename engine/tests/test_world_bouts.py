"""Bouts from every promotion, parsed from real Wikipedia pages.

The fixtures under tests/fixtures/wiki were saved from Wikipedia by
probe_world.py on the Actions runner; nothing here is hand-written markup
except where a test says so.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import world_bouts as wb

WIKI = ENGINE / "tests" / "fixtures" / "wiki"


def page(name):
    path = WIKI / f"{name}.wiki"
    if not path.exists():
        pytest.skip(f"fixture {name} not saved yet")
    return path.read_text()


def test_a_year_page_gives_every_professional_bout_with_its_event_date():
    bouts = wb.event_bouts(page("2024_in_Legacy_Fighting_Alliance"), "2024 LFA")
    assert len(bouts) > 250
    assert all(b["date"] and b["date"].startswith("2024") for b in bouts)
    first = bouts[0]
    assert (first["event"], first["date"]) == \
        ("LFA 174: Jones vs. Gennrich", "2024-01-12")
    assert (first["winner"], first["loser"]) == ("Kegan Gennrich",
                                                 "JaCobi Jones")
    assert first["method"].startswith("Submission (arm-triangle choke)")
    assert (first["round"], first["time"]) == ("5", "2:08")
    assert len({b["event"] for b in bouts}) >= 20


def test_amateur_bouts_are_not_professional_record():
    bouts = wb.event_bouts(page("2024_in_Legacy_Fighting_Alliance"), "2024 LFA")
    names = {b["winner"] for b in bouts} | {b["loser"] for b in bouts}
    # LFA 174's amateur card, verbatim from the page.
    assert "Ezayah Gomez Oropeza" not in names
    assert "Xavier Rosenbloom" not in names
    assert not any("amateur" in b["card"].lower() for b in bouts)


def test_a_draw_or_no_contest_has_no_winner():
    bouts = wb.event_bouts(page("2024_in_Cage_Warriors"), "2024 CW")
    results = Counter(b["result"] for b in bouts)
    assert results["win"] > 200 and results["nc"] >= 1
    for b in bouts:
        if b["result"] != "win":
            assert "def" not in b["method"].lower()


def test_an_event_page_takes_its_infobox_name_and_date():
    bouts = wb.event_bouts(page("Bellator_300"), "Bellator 300")
    assert bouts and {b["date"] for b in bouts} == {"2023-10-07"}
    assert {b["event"] for b in bouts} == \
        {"Bellator 300: Nurmagomedov vs. Primus"}
    assert {b["promotion"] for b in bouts} == {"Bellator MMA"}


def test_a_ufc_event_page_parses_too():
    bouts = wb.event_bouts(page("UFC_300"), "UFC 300")
    assert len(bouts) == 13 and {b["date"] for b in bouts} == {"2024-04-13"}


def test_a_fighter_record_is_read_from_the_subjects_side():
    bouts = wb.record_bouts(page("Gilbert_Burns"), "Gilbert Burns")
    assert len(bouts) >= 30 and all(b["date"] for b in bouts)
    latest = bouts[0]
    assert (latest["winner"], latest["loser"]) == ("Mike Malott",
                                                   "Gilbert Burns")   # a Loss
    assert latest["date"] == "2026-04-18"
    assert latest["round"] == "3" and latest["time"] == "2:08"
    wins = [b for b in bouts if b["winner"] == "Gilbert Burns"]
    assert len(wins) > 20


def test_record_dates_in_every_written_form():
    bouts = wb.record_bouts(page("Mateusz_Gamrot"), "Mateusz Gamrot")
    assert all(b["date"] for b in bouts)
    assert any(b["date"] == "2012-04-21" for b in bouts)   # {{dts|2012.04.21}}


@pytest.mark.parametrize("text,iso", [
    ("{{dts|2026|April|18|format=dmy}}", "2026-04-18"),
    ("{{dts|2012.04.21}}", "2012-04-21"),
    ("{{Start date|2023|10|7}}", "2023-10-07"),
    ("January 12, 2024", "2024-01-12"),
    ("12 January 2024", "2024-01-12"),
    ("Sept 22, 2026", "2026-09-22"),
])
def test_dates(text, iso):
    assert wb.parse_date(text) == iso


def test_split_params_keeps_links_whole():
    assert wb.split_params("a|[[B c|d]]|{{x|y}}|e") == \
        ["a", "[[B c|d]]", "{{x|y}}", "e"]


# --- harvest_world helpers ----------------------------------------------------

import harvest_world as hw


def test_a_plain_results_table_is_read_when_there_are_no_templates():
    text = ("==Week 7 – September 22, 2026==\n"
            "{| class=\"wikitable\"\n|-\n"
            "| Lightweight || Piero Guaylupo ||def.|| Callum Connor "
            "|| Decision (unanimous) || 3 || 5:00\n|}\n")
    bouts = hw.table_bouts(text, "DWCS season 10")
    assert len(bouts) == 1
    b = bouts[0]
    assert (b["winner"], b["loser"], b["date"]) == \
        ("Piero Guaylupo", "Callum Connor", "2026-09-22")
    assert b["method"] == "Decision (unanimous)"


def test_one_bout_told_by_an_event_page_and_a_fighter_record_is_one_row():
    event = {"date": "2024-01-12", "winner": "Kegan Gennrich",
             "loser": "JaCobi Jones", "source": "2024 in LFA"}
    record = {"date": "2024-01-12", "winner": "Kegan Gennrich",
              "loser": "Jacobi Jones", "source": "record:Kegan Gennrich"}
    assert hw.dedupe([record, event]) == [event]


def test_an_undated_bout_is_not_kept():
    assert hw.dedupe([{"date": None, "winner": "A", "loser": "B",
                       "source": "x"}]) == []
