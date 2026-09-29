"""The replacement parser, tested against the sentence shapes Wikipedia uses.

These are real shapes, lightly anonymised. They matter because the probe this
belongs to reports a COVERAGE number, and a parser that quietly misses half
the variants reports "too thin" and kills a line of work that was actually
viable. A false negative here is more expensive than a false positive.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import check_fight_changes as cfc


def only(text):
    rows = cfc.changes_in(text)
    assert len(rows) == 1, f"expected one row, got {len(rows)}: {rows}"
    return rows[0]


# --- the shapes that must be caught ---------------------------------------

def test_withdrew_and_was_replaced_by():
    row = only("Alex Smith was expected to face Bob Jones. However, Jones "
               "withdrew due to injury and was replaced by Carl Brown.")
    assert row["stepped_in"] == "Carl Brown"
    # A SURNAME, because that is what the sentence says. Wikipedia introduces
    # a fighter in full and then uses the surname on every later mention, so
    # most extracted names come back partial. Reaching back into the previous
    # sentence to expand it would be guessing; the join resolves it instead,
    # against the twenty-odd people who were actually on that card, where a
    # surname is very nearly unique. The probe counts these so the joining
    # cost is visible before anything is built on top.
    assert row["replaced"] == "Jones"


def test_replaced_in_the_active_voice():
    row = only("Carl Brown replaced Bob Jones at the event.")
    assert row["stepped_in"] == "Carl Brown"
    assert row["replaced"] == "Bob Jones"


def test_stepped_in_for():
    row = only("Carl Brown stepped in for Bob Jones on short notice.")
    assert row["stepped_in"] == "Carl Brown"
    assert row["replaced"] == "Bob Jones"


def test_pulled_out_is_the_same_event_as_withdrew():
    row = only("Bob Jones pulled out of the bout with a knee injury and was "
               "replaced by Carl Brown.")
    assert row["replaced"] == "Bob Jones"
    assert row["stepped_in"] == "Carl Brown"


# --- the notice period ----------------------------------------------------

def test_a_notice_in_days_is_read_as_days():
    row = only("Carl Brown replaced Bob Jones on nine days' notice.")
    assert row["days_notice"] == 9


def test_a_notice_in_weeks_becomes_days():
    row = only("Carl Brown replaced Bob Jones on two weeks notice.")
    assert row["days_notice"] == 14


def test_a_numeral_works_as_well_as_a_word():
    assert only("Carl Brown replaced Bob Jones on 11 days notice."
                )["days_notice"] == 11


def test_short_notice_with_no_number_is_recorded_as_unquantified():
    """Zero, not None, and not a guess.

    "On short notice" is a real observation and must not be dropped; inventing
    a number for it would be worse than recording that none was given. The
    probe counts these separately for exactly that reason.
    """
    assert only("Carl Brown replaced Bob Jones on short notice."
                )["days_notice"] == 0


def test_no_notice_phrase_at_all_is_None():
    assert only("Carl Brown replaced Bob Jones.")["days_notice"] is None


# --- what must NOT be caught ----------------------------------------------

def test_a_results_table_is_not_a_replacement():
    """Names in bout tables are the whole page. Reading one as a replacement
    would inflate the coverage estimate with rows that mean nothing."""
    table = ("{{MMAevent|Lightweight|Alex Smith|def.|Bob Jones|Decision}}\n"
             "{{MMAevent|Welterweight|Carl Brown|def.|Dan White|KO}}")
    assert cfc.changes_in(table) == []


def test_prose_with_no_change_finds_nothing():
    assert cfc.changes_in("The event drew an announced attendance of 15,000 "
                          "and a live gate of $2 million.") == []


def test_a_reference_block_is_not_read_as_prose():
    text = ("The card was announced in June.<ref>Carl Brown replaced Bob "
            "Jones, according to the report.</ref>")
    assert cfc.changes_in(text) == []


def test_links_are_unwrapped_so_a_linked_name_still_matches():
    row = only("[[Bob Jones]] withdrew and was replaced by "
               "[[Carl Brown|Brown]].")
    assert row["replaced"] == "Bob Jones"


# --- the arithmetic the probe's verdict rests on --------------------------

def test_a_lowercase_word_is_not_taken_for_a_name():
    """"was replaced by a newcomer" names nobody, and a row with no name is
    a row that cannot be joined to a fight."""
    assert cfc.changes_in("Bob Jones withdrew and was replaced by a "
                          "promotional newcomer.") == []


# --- resolving an event to an article -------------------------------------
# Every failure this probe has had so far was here, not in the parser, and
# each one came out of CI looking like a fact about Wikipedia's coverage.

def test_a_numbered_event_offers_its_bare_number_as_a_title():
    """"UFC 182: Jones vs Cormier" lives at "UFC 182". Guessing that saves a
    search for roughly half of all events."""
    assert "UFC 182" in cfc.title_candidates("UFC 182: Jones vs Cormier")


def test_a_fight_night_offers_both_spellings_of_vs():
    got = cfc.title_candidates("UFC Fight Night: McGregor vs Siver")
    assert "UFC Fight Night: McGregor vs Siver" in got
    assert "UFC Fight Night: McGregor vs. Siver" in got


def test_candidates_are_unique_and_keep_the_best_guess_first():
    got = cfc.title_candidates("UFC 300")
    assert got[0] == "UFC 300"
    assert len(got) == len(set(got))


def test_a_plain_infobox_date_is_read():
    assert cfc.infobox_date("| date = April 13, 2024") == \
        pd.Timestamp("2024-04-13").date()


def test_a_start_date_template_is_read():
    assert cfc.infobox_date("| date = {{Start date|2024|4|13}}") == \
        pd.Timestamp("2024-04-13").date()


def test_no_date_field_returns_None():
    assert cfc.infobox_date("| venue = T-Mobile Arena") is None


def test_an_article_whose_date_disagrees_is_not_a_match():
    """The check that would have caught UFC 16 - a 1998 card - being resolved
    to a 2009 event. Without it the probe fetched the wrong articles, found
    nothing in them, and reported that Wikipedia does not carry this."""
    text = "| date = May 13, 1998"
    assert not cfc._dated(text, pd.Timestamp("2009-04-18").date())
    assert cfc._dated(text, pd.Timestamp("1998-05-13").date())


def test_a_day_either_side_is_allowed():
    """A card that starts late on the 13th local time is the 14th in UTC, and
    the archive and Wikipedia do not always agree which they mean."""
    assert cfc._dated("| date = April 13, 2024",
                      pd.Timestamp("2024-04-14").date())
    assert not cfc._dated("| date = April 13, 2024",
                          pd.Timestamp("2024-04-16").date())


# --- missed weight --------------------------------------------------------
# A kind that exists at scale in Wikipedia and is pre-fight by construction:
# the weigh-in is the day before. A parser that misses the common phrasing
# would report the kind as rare and the measurement would never run.

def miss(text):
    rows = cfc.weigh_ins_in(text)
    assert len(rows) == 1, rows
    return rows[0]


def test_the_standard_weigh_in_sentence():
    row = miss("At the weigh-ins, Mackenzie Dern weighed in at 117 pounds, "
               "one pound over the strawweight non-title fight limit.")
    assert row["fighter"] == "Mackenzie Dern"
    assert row["weighed_lbs"] == 117 and row["over_by_lbs"] == 1


def test_the_convert_template_wikipedia_actually_uses():
    """{{convert|159|lb|kg}} is how almost every weight is written. Stripping
    templates before reading it would delete the number."""
    row = miss("Kevin Lee weighed in at {{convert|158.5|lb|kg}}, "
               "{{convert|2.5|lb|kg}} over the lightweight limit.")
    assert row["weighed_lbs"] == 158.5 and row["over_by_lbs"] == 2.5


def test_a_metric_parenthetical_is_skipped():
    row = miss("Yoel Romero weighed in at 187.7 pounds (85.1 kg), "
               "1.7 pounds over the middleweight title fight limit.")
    assert row["over_by_lbs"] == 1.7


def test_half_a_pound():
    assert miss("Chris Barnett weighed in at 266.5 pounds, half a pound over "
                "the heavyweight limit.")["over_by_lbs"] == 0.5


def test_two_and_a_half_pounds():
    assert miss("Jones weighed in at 158.5 pounds, two and a half pounds "
                "over the limit.")["over_by_lbs"] == 2.5


def test_missed_weight_with_no_margin_is_still_a_miss():
    """Recorded with the margin blank, never with a guessed one."""
    row = miss("Paulo Costa missed weight for the bout.")
    assert row["fighter"] == "Paulo Costa" and row["over_by_lbs"] is None


def test_a_linked_name_is_read():
    assert miss("[[Mackenzie Dern]] weighed in at 117 pounds, one pound "
                "over the limit.")["fighter"] == "Mackenzie Dern"


def test_making_weight_is_not_a_miss():
    assert cfc.weigh_ins_in("Dern weighed in at 116 pounds for the bout.") == []


def test_a_pronoun_is_not_a_fighter():
    assert cfc.weigh_ins_in("She weighed in at 117 pounds, one pound over "
                            "the limit.") == []


def test_a_weigh_in_inside_a_reference_is_ignored():
    """A citation title is another page's sentence."""
    assert cfc.weigh_ins_in(
        "The event was held.<ref>Dern weighed in at 117 pounds, one pound "
        "over the limit</ref>") == []


def test_the_same_fighter_twice_on_a_page_is_counted_once():
    text = ("Dern weighed in at 117 pounds, one pound over the limit. "
            "Dern missed weight and was fined.")
    assert len(cfc.weigh_ins_in(text)) == 1


def test_a_pronoun_is_not_a_replacement_either():
    assert cfc.changes_in("He replaced Bob Jones on the card.") == []
