"""Joining an extracted name to the fighter who was actually on the card.

This is where a label gets attached to the wrong person, and a label on the
wrong fighter is worse than a missing one: a missing label costs a little
statistical power, a wrong one is indistinguishable from evidence.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fetch_fight_changes as ffc

CARD = ["Raul Rosas Jr.", "Raoni Barcelos", "Norma Dumont", "Ailin Perez",
        "Montel Jackson", "Ricky Simon"]


def test_a_full_name_matches_itself():
    assert ffc.match_to_card("Norma Dumont", CARD) == "Norma Dumont"


def test_a_surname_matches_when_it_is_unique_on_the_card():
    """The case that makes this worth doing at all: Wikipedia names a fighter
    in full once and by surname every time after."""
    assert ffc.match_to_card("Barcelos", CARD) == "Raoni Barcelos"


def test_a_surname_shared_by_two_fighters_matches_nobody():
    card = CARD + ["Danny Barcelos"]
    assert ffc.match_to_card("Barcelos", card) is None


def test_a_fighter_not_on_this_card_matches_nobody():
    assert ffc.match_to_card("Conor McGregor", CARD) is None


def test_a_generational_suffix_does_not_break_the_surname():
    """short_name() keeps "Rosas Jr." together; if this regressed, every
    fighter with a suffix would silently stop matching."""
    assert ffc.match_to_card("Rosas Jr.", CARD) == "Raul Rosas Jr."


def test_case_and_spacing_do_not_matter():
    assert ffc.match_to_card("  norma  dumont ", CARD) == "Norma Dumont"


def test_an_empty_name_matches_nobody():
    assert ffc.match_to_card("", CARD) is None


def test_card_names_gathers_both_corners():
    archive = pd.DataFrame({
        "event_name": ["UFC 1", "UFC 1", "UFC 2"],
        "r_name": ["A One", "B Two", "C Three"],
        "b_name": ["D Four", "E Five", "F Six"],
    })
    assert ffc.card_names(archive, "UFC 1") == \
        ["A One", "B Two", "D Four", "E Five"]


# --- the fighter whose opponent was switched --------------------------------

BOUTS = [("Natalia Silva", "Wang Cong"), ("Deiveson Figueiredo",
                                          "Payton Talbott")]


def test_the_kept_fighter_is_the_replacements_opponent():
    """"Wang replaced Shevchenko" says nothing about Silva. She is found as
    Wang's opponent on the card - the one who prepared for someone else."""
    assert ffc.opponent_on_card("Wang Cong", BOUTS) == "Natalia Silva"
    assert ffc.opponent_on_card("Natalia Silva", BOUTS) == "Wang Cong"


def test_a_fighter_not_on_the_card_has_no_opponent():
    assert ffc.opponent_on_card("Valentina Shevchenko", BOUTS) is None


def test_card_bouts_reads_pairs_and_skips_blanks():
    archive = pd.DataFrame({
        "event_name": ["UFC 1", "UFC 1", "UFC 2"],
        "r_name": ["A One", None, "C Three"],
        "b_name": ["B Two", "E Five", "D Four"],
    })
    assert ffc.card_bouts(archive, "UFC 1") == [("A One", "B Two")]
