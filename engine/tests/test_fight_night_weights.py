"""Measured fight-night weights, read from real Wikipedia event pages."""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fight_night_weights as fnw

WIKI = ENGINE / "tests" / "fixtures" / "wiki"


def test_bellator_300_verbatim():
    path = WIKI / "Bellator_300.wiki"
    if not path.exists():
        pytest.skip("fixture not saved")
    rows = fnw.weights(path.read_text(), "Bellator 300")
    assert len(rows) == 32
    first = rows[0]
    assert first["fighter"] == "Usman Nurmagomedov"
    assert (first["official_lbs"], first["fight_night_lbs"]) == (154.8, 173.2)
    assert first["date"] == "2023-10-07"
    assert abs(first["gain_pct"] - 11.89) < 0.01
    moicano = next(r for r in rows if r["fighter"] == "Renato Moicano")
    assert moicano["gain_lbs"] == 25.8
    # A fighter who LOST weight overnight is kept - it happens.
    assert any(r["fight_night_lbs"] < r["official_lbs"] for r in rows)


@pytest.mark.parametrize("line,expected", [
    ("*'''Carl Brown''': 154.8 to 173.2 pounds (12%)", ("Carl Brown", 154.8, 173.2)),
    ("* [[Carl Brown]] – 155 lb → 171 lb", ("Carl Brown", 155.0, 171.0)),
    ("*Carl Brown: 145.5 lbs / 160.0 lbs", ("Carl Brown", 145.5, 160.0)),
])
def test_the_written_forms(line, expected):
    text = "==Fight night weights==\n" + line + "\n==Aftermath==\n"
    rows = fnw.weights(text, "X")
    assert [(r["fighter"], r["official_lbs"], r["fight_night_lbs"])
            for r in rows] == [expected]


def test_numbers_that_are_not_a_weight_pair_are_refused():
    text = ("==Fight night weights==\n*Carl Brown: 155 to 400 pounds\n"
            "*Record: 12 to 3\n==Aftermath==\n")
    assert fnw.weights(text, "X") == []


def test_only_the_weights_section_is_read():
    text = ("==Results==\n*Carl Brown: 155 to 171 pounds\n"
            "==Fight night weights==\n*Dan Evans: 145 to 160 pounds\n"
            "==Aftermath==\n*Ed Green: 170 to 185 pounds\n")
    assert [r["fighter"] for r in fnw.weights(text, "X")] == ["Dan Evans"]
