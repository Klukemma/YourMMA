"""Tests for fighter name resolution.

Runs against the real fighter list from the dataset, so these are regression
tests against actual data, not a toy fixture.

    python -m pytest engine/tests/ -q
"""

import csv
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from name_resolution import NameResolver, norm_name, load_aliases, format_failure

CSV_PATH = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


@pytest.fixture(scope="module")
def resolver():
    names, appearances = set(), {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("r_name", "b_name"):
                nm = row[col]
                if nm:
                    names.add(nm)
                    appearances[nm] = appearances.get(nm, 0) + 1
    return NameResolver(names=names, appearances=appearances)


# --- the regression that started this ------------------------------------

def test_unknown_fighter_is_refused_not_substituted(resolver):
    """'Leon Shahbazyan' used to silently resolve to 'Cameron Saaiman'."""
    res = resolver.resolve("Leon Shahbazyan")
    assert res["status"] == "NOT_FOUND"
    assert res["name"] is None


def test_suggestions_are_ranked_by_quality_not_alphabetically(resolver):
    """The old code sorted candidates A-Z, so 0.60 beat 0.84."""
    res = resolver.resolve("Leon Shahbazyan")
    names = [n for _, n in res["suggestions"]]
    assert "Edmen Shahbazyan" in names
    assert names[0] == "Edmen Shahbazyan", f"best match should lead, got {names}"
    if "Cameron Saaiman" in names:
        assert names.index("Edmen Shahbazyan") < names.index("Cameron Saaiman")


@pytest.mark.parametrize("typed", ["Levan Chokheli", "Shane Collins"])
def test_other_known_bad_cards_are_refused(resolver, typed):
    assert resolver.resolve(typed)["status"] == "NOT_FOUND"


# --- names that must still work -------------------------------------------

@pytest.mark.parametrize("typed", [
    "Ilia Topuria", "ilia topuria", "  Ilia   Topuria  ",
    "Alexandre Pantoja", "Brandon Moreno",
])
def test_real_fighters_resolve(resolver, typed):
    res = resolver.resolve(typed)
    assert res["status"] == "OK"
    assert norm_name(res["name"]) == norm_name(typed)


def test_accents_and_punctuation_are_ignored(resolver):
    assert resolver.resolve("Jose Aldo")["status"] == "OK"
    assert resolver.resolve("José Aldo")["status"] == "OK"
    assert resolver.resolve("Jose Aldo")["name"] == resolver.resolve("José Aldo")["name"]


# --- the guard rails ------------------------------------------------------

def test_ambiguous_surname_is_refused(resolver):
    """'Pereira' alone matches several fighters - refuse, don't guess."""
    res = resolver.resolve("Pereira")
    assert res["status"] == "AMBIGUOUS"
    assert res["name"] is None
    assert len(res["suggestions"]) > 1


def test_unique_surname_resolves(resolver):
    res = resolver.resolve("Khabib")
    assert res["status"] == "OK"
    assert "Khabib" in res["name"]


def test_typo_in_first_name_is_accepted(resolver):
    """Surname spelled right + very close overall -> treat as a typo."""
    r = NameResolver(names=["Alex Pereira", "Cameron Saaiman"], auto_accept_ratio=0.90)
    res = r.resolve("Alexx Pereira")
    assert res["status"] == "OK" and res["name"] == "Alex Pereira" and res["how"] == "typo"


def test_typo_in_surname_is_refused(resolver):
    """A close ratio is not enough - a misspelled surname gets a suggestion."""
    r = NameResolver(names=["Alex Pereira", "Cameron Saaiman"], auto_accept_ratio=0.90)
    res = r.resolve("Alex Perreira")
    assert res["status"] == "NOT_FOUND"
    assert [n for _, n in res["suggestions"]][:1] == ["Alex Pereira"]


def test_typo_never_reaches_an_unrelated_fighter(resolver):
    r = NameResolver(names=["Alex Pereira", "Cameron Saaiman"], auto_accept_ratio=0.90)
    assert r.resolve("Alex Pereir")["name"] != "Cameron Saaiman"


def test_empty_and_garbage_input(resolver):
    for bad in ["", "   ", None, "!!!", "12345"]:
        assert resolver.resolve(bad)["status"] != "OK"


def test_nonexistent_fighter_is_not_found(resolver):
    res = resolver.resolve("Zzzqqq Notarealfighter")
    assert res["status"] == "NOT_FOUND"
    assert res["name"] is None


# --- aliases --------------------------------------------------------------

def test_alias_overrides_resolution(resolver):
    r = NameResolver(names=["Jon Jones"], aliases={norm_name("Bones Jones"): "Jon Jones"})
    res = r.resolve("Bones Jones")
    assert res["status"] == "OK" and res["name"] == "Jon Jones" and res["how"] == "alias"


def test_alias_file_ignores_comment_keys(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"_comment": "ignore me", "Bones": "Jon Jones"}')
    aliases = load_aliases(f)
    assert aliases == {"bones": "Jon Jones"}


def test_missing_alias_file_is_not_an_error(tmp_path):
    assert load_aliases(tmp_path / "nope.json") == {}


# --- search (for the phone app autocomplete) ------------------------------

def test_search_returns_ranked_matches(resolver):
    out = resolver.search("topuria")
    assert out and any("Topuria" in n for n in out)


def test_search_on_empty_query(resolver):
    assert resolver.search("") == []


def test_failure_message_includes_suggestions(resolver):
    res = resolver.resolve("Leon Shahbazyan")
    msg = format_failure("Leon Shahbazyan", res)
    assert "NO DATA" in msg and "Did you mean" in msg
