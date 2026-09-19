"""Tests for matching a fighter to a bookmaker's price.

This is the only place in the engine where being WRONG is worse than knowing
nothing. An unmatched fighter shows no price, which is visibly missing on the
card. A fighter matched to somebody else's price produces an edge, a value
rating and a bet recommendation that all look completely ordinary and are
computed against a number that has nothing to do with the fight.

The functions are lifted out of predict_card rather than imported, because
importing that module runs the whole twenty-minute pipeline. test_app_export
holds the guard that the two stay in step.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
SOURCE = (ENGINE / "predict_card.py").read_text()


def _load():
    """Execute just the matcher out of predict_card, with its helpers."""
    start = SOURCE.index("FIRST_NAME_MIN_RATIO = ")
    end = SOURCE.index("def calculate_value(")
    namespace = {"re": re, "SequenceMatcher": SequenceMatcher}
    exec(SOURCE[start:end], namespace)
    return namespace["match_fighter_to_odds"]


match = _load()

CARD = {
    "bruno silva": {"name": "Bruno Silva", "best_odds": -200},
    "joel alvarez": {"name": "Joel Alvarez", "best_odds": 150},
    "jonathan martinez": {"name": "Jonathan Martinez", "best_odds": -135},
}


def test_an_exact_name_matches():
    assert match("Bruno Silva", CARD)["name"] == "Bruno Silva"
    assert match("  joel alvarez ", CARD)["name"] == "Joel Alvarez"


@pytest.mark.parametrize("other", [
    "Anderson Silva", "Jean Silva", "Erick Silva", "Douglas Silva",
])
def test_a_different_fighter_with_the_same_surname_gets_no_price(other):
    """The bug this file exists for.

    The previous rule accepted any shared last name whose whole-string
    similarity cleared 0.5. Anderson Silva scored 0.64 against Bruno Silva and
    was handed his odds. The edge computed from that is fiction that reads as
    fact.
    """
    assert match(other, CARD) is None, f"{other} was priced as somebody else"


def test_a_shortened_first_name_still_matches():
    """Books and the dataset disagree about first names constantly."""
    assert match("Jon Martinez", CARD)["name"] == "Jonathan Martinez"


def test_an_initial_still_matches():
    assert match("J. Martinez", CARD)["name"] == "Jonathan Martinez"


def test_a_different_surname_gets_no_price():
    assert match("Bruno Souza", CARD) is None
    assert match("Joel Alvarado", CARD) is None


def test_an_accent_or_spelling_difference_still_matches():
    """Near-identity on the whole string is allowed; two people cannot reach
    it, but Álvarez and Alvarez can."""
    assert match("Joel Álvarez", CARD)["name"] == "Joel Alvarez"


def test_an_empty_or_unusable_name_gets_no_price():
    assert match("", CARD) is None
    assert match("Bruno Silva", {}) is None
    assert match("Bruno Silva", None) is None


def test_the_matcher_is_never_a_coin_toss_between_two_same_surname_entries():
    """With BOTH Silvas on the card, each must get his own price."""
    both = dict(CARD)
    both["jean silva"] = {"name": "Jean Silva", "best_odds": 320}
    assert match("Jean Silva", both)["best_odds"] == 320
    assert match("Bruno Silva", both)["best_odds"] == -200
    assert match("Anderson Silva", both) is None
