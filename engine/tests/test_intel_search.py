"""The scout's ingest, which is the only thing standing between a language
model's fluent paragraph and a file the app treats as fact.

The dangerous failures are not crashes. They are a plausible, undated,
unsourced claim landing in fight_intel.json and being indistinguishable from
research a week later - and, worse, a claim gathered AFTER the fight, which
reads as prescient and is hindsight.
"""

import json
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fight_intel as intel
import intel_search as scout


def empty():
    return {"schema": intel.SCHEMA, "observations": []}


def finding(**over):
    base = {
        "fighter": "Brady Hiestand",
        "event_date": "2026-09-26",
        "kind": "injury",
        "confidence": "reported",
        "note": "knee trouble reported in the final week of camp",
        "source": "https://www.sherdog.com/news/example",
        "gathered": "2026-09-24T10:00:00Z",
    }
    base.update(over)
    return base


# --- the brief -------------------------------------------------------------

def test_the_brief_covers_both_corners_of_every_fight():
    plan = scout.brief([("A One", "B Two"), ("C Three", "D Four")],
                       "2026-09-26")
    assert [p["fighter"] for p in plan] == ["A One", "B Two", "C Three",
                                            "D Four"]


def test_a_fighter_on_the_card_twice_is_briefed_once():
    plan = scout.brief([("A One", "B Two"), ("A One", "C Three")], "2026-09-26")
    assert [p["fighter"] for p in plan] == ["A One", "B Two", "C Three"]


def test_every_question_names_a_kind_the_store_accepts():
    """A query labelled with a kind fight_intel would refuse sends the
    researcher off to find something that cannot be filed."""
    plan = scout.brief([("A One", "B Two")], "2026-09-26")
    for question in plan[0]["queries"]:
        assert question["kind"] in intel.KINDS


def test_the_brief_puts_the_fighter_and_year_in_the_query():
    plan = scout.brief([("Raul Rosas Jr.", "B Two")], "2026-09-26")
    first = plan[0]["queries"][0]["query"]
    assert "Raul Rosas Jr." in first and "2026" in first


# --- the ingest, and what it refuses --------------------------------------

def test_a_good_finding_is_recorded():
    store = empty()
    added, rejected = scout.ingest([finding()], store=store)
    assert (added, rejected) == (1, [])
    assert store["observations"][0]["fighter"] == "Brady Hiestand"


def test_a_finding_gathered_after_the_fight_is_rejected_as_hindsight():
    """The one that matters most. A search run today about last Saturday's
    fight returns coverage written afterwards, phrased as though it were
    known beforehand."""
    store = empty()
    added, rejected = scout.ingest(
        [finding(gathered="2026-09-28T10:00:00Z")], store=store)
    assert added == 0
    assert "hindsight" in rejected[0]
    assert store["observations"] == []


def test_a_finding_gathered_on_the_day_is_allowed():
    """The card is predicted on the morning of the event, so same-day news is
    legitimately pre-fight."""
    added, _ = scout.ingest([finding(gathered="2026-09-26T08:00:00Z")],
                            store=empty())
    assert added == 1


def test_a_finding_with_no_source_is_rejected():
    added, rejected = scout.ingest([finding(source="")], store=empty())
    assert added == 0 and "source" in rejected[0].lower()


def test_a_source_that_is_not_a_url_is_rejected():
    added, rejected = scout.ingest([finding(source="MMA Junkie")],
                                   store=empty())
    assert added == 0 and "source" in rejected[0].lower()


def test_an_invented_kind_is_rejected():
    added, rejected = scout.ingest([finding(kind="vibes")], store=empty())
    assert added == 0 and "unknown kind" in rejected[0]


def test_a_finding_with_no_event_date_is_rejected():
    added, rejected = scout.ingest([finding(event_date=None)], store=empty())
    assert added == 0 and "event_date" in rejected[0]


def test_an_empty_note_is_rejected():
    added, rejected = scout.ingest([finding(note="hurt")], store=empty())
    assert added == 0


def test_one_bad_finding_does_not_lose_the_good_ones():
    """A researcher returning six findings and one dud should keep the six."""
    store = empty()
    added, rejected = scout.ingest(
        [finding(), finding(fighter="Other Guy", source="not a url"),
         finding(fighter="Third Guy")], store=store)
    assert added == 2 and len(rejected) == 1


def test_the_same_finding_twice_is_recorded_once():
    store = empty()
    scout.ingest([finding()], store=store)
    added, _ = scout.ingest([finding()], store=store)
    assert added == 0 and len(store["observations"]) == 1


def test_an_opponent_switch_needs_the_fighter_who_was_replaced():
    added, rejected = scout.ingest(
        [finding(kind="opponent_switch",
                 note="stepped in when the original opponent withdrew")],
        store=empty())
    assert added == 0 and "replaced_opponent" in rejected[0]


def test_an_opponent_switch_with_the_original_is_recorded():
    added, _ = scout.ingest(
        [finding(kind="opponent_switch", replaced_opponent="Mickey Gall",
                 note="stepped in when Gall withdrew with a medical issue")],
        store=empty())
    assert added == 1


def test_nothing_the_scout_records_can_move_a_probability():
    """The whole point. No weight is fitted for any kind, so a full store
    still yields an empty context."""
    store = empty()
    scout.ingest([finding(), finding(fighter="Rinya Nakamura")], store=store)
    records = intel.for_fight(store, "Brady Hiestand", "Rinya Nakamura",
                              "2026-09-26", known_by="2026-09-26")
    assert len(records) == 2
    assert intel.context_from(records, "Brady Hiestand", "Rinya Nakamura") == {}
