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
    # The section cites MMA Fighting's October 9 write-up: that is when the
    # numbers became public, two days after the October 7 event.
    assert {r["published_at"] for r in rows} == {"2023-10-09"}


def test_publication_date_is_the_cited_article_else_a_week_after():
    assert fnw.published_at("2023-10-07", "") == "2023-10-14"
    cited = "<ref>{{Cite web |date=2023-10-09 |access-date=2024-01-01 |title=x}}</ref>"
    assert fnw.published_at("2023-10-07", cited) == "2023-10-09"
    # A citation dated before the event (a weigh-in report) cannot have
    # carried fight-night numbers; an access-date is not a publication date.
    early = "<ref>{{Cite web |date=2023-10-06 |access-date=2023-10-20 |title=weigh-ins}}</ref>"
    assert fnw.published_at("2023-10-07", early) == "2023-10-14"
    assert fnw.published_at(None, cited) is None


def test_a_results_article_cited_next_to_the_weights_article_does_not_date_the_numbers():
    results = "<ref>{{Cite web |date=2023-10-08 |title=Bellator 300 results: Nurmagomedov wins |url=r}}</ref>"
    weights = "<ref>{{Cite web |date=2023-10-09 |title=Bellator 300 fight night weights: Four gain 20 pounds |url=w}}</ref>"
    salaries = "<ref>{{Cite web |date=2023-10-12 |title=Bellator 300 salaries |url=s}}</ref>"
    # A fight-night-titled citation dates the numbers, whatever else is cited.
    assert fnw.published_at("2023-10-07", results + weights) == "2023-10-09"
    assert fnw.published_at("2023-10-07", weights + salaries) == "2023-10-09"
    # Without one, the LATEST on/after-event citation: never earlier than the truth.
    assert fnw.published_at("2023-10-07", results + salaries) == "2023-10-12"
    assert fnw.published_at("2023-10-07", results) == "2023-10-08"


def test_cite_dates_in_prose_parse_and_malformed_ones_fall_back_without_raising():
    prose = "<ref>{{Cite news |date=October 9, 2023 |title=x}}</ref>"
    assert fnw.published_at("2023-10-07", prose) == "2023-10-09"
    assert fnw.published_at("2023-10-07", "<ref>{{cite web |title=y |date=9 October 2023}}</ref>") == "2023-10-09"
    # A typo date (February 30) is dropped, not raised: the harvest goes on.
    bad = "<ref>{{Cite web |date=2023-02-30 |title=x}}</ref>"
    assert fnw.published_at("2023-10-07", bad) == "2023-10-14"
    assert fnw.citations(bad) == []
    # A results cite with a typo next to a good weights cite: the good one.
    good = "<ref>{{Cite web |date=2023-10-09 |title=fight night weights |url=w}}</ref>"
    assert fnw.published_at("2023-10-07", bad + good) == "2023-10-09"


def test_rows_from_a_section_without_a_dated_citation_lag_a_week():
    text = "==Fight night weights==\n*Carl Brown: 155 to 171 pounds\n==Aftermath==\n"
    text = "{{Infobox MMA event\n|name=X\n|date={{Start date|2024|03|09}}\n}}\n" + text
    rows = fnw.weights(text, "X")
    assert rows[0]["date"] == "2024-03-09" and rows[0]["published_at"] == "2024-03-16"


def test_with_published_at_fills_only_what_is_missing():
    import pandas as pd
    frame = pd.DataFrame({"date": ["2024-03-09", "2024-03-09"],
                          "published_at": ["2024-03-10", None]})
    got = fnw.with_published_at(frame)
    assert list(got["published_at"]) == ["2024-03-10", "2024-03-16"]
    no_column = fnw.with_published_at(pd.DataFrame({"date": ["2024-03-09"]}))
    assert list(no_column["published_at"]) == ["2024-03-16"]


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
