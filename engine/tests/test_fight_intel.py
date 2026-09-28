"""Soft intel must be sourced, point-in-time, and unable to move the number.

The appeal of an AI researcher filling this store is obvious and so is the
failure mode: a language model that cannot find a fact will write a plausible
one, and a note gathered the morning after a knockout will explain it
perfectly. Both produce a store that looks like evidence and is not.

So: every observation carries a source URL and a gathered timestamp, and
nothing moves a probability until a weight has been fitted against outcomes.
The context hook it would drive already exists in predict_card with
hand-typed constants nobody measured - feeding better facts into an unfitted
-4% does not make the prediction better, it makes the guess louder.
"""

import json
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fight_intel as fi


def good(**over):
    base = dict(fighter="Brady Hiestand", event_date="2026-09-26",
                kind="injury", confidence="reported",
                note="knee trouble reported in the final week of camp",
                source="https://example.com/report", gathered="2026-09-20")
    base.update(over)
    return fi.observation(**base)


# --- the door --------------------------------------------------------------

def test_an_observation_needs_a_source_url():
    """A claim nobody can check cannot be told apart from one invented."""
    with pytest.raises(fi.IntelError, match="source URL"):
        good(source="")
    with pytest.raises(fi.IntelError, match="source URL"):
        good(source="MMA Junkie")          # a name is not a source
    with pytest.raises(fi.IntelError, match="source URL"):
        good(source="ftp://example.com/x")


def test_an_observation_needs_a_readable_note():
    with pytest.raises(fi.IntelError):
        good(note="hurt")


def test_the_kind_is_a_closed_list():
    """Forty spellings of one idea can never be fitted against outcomes."""
    with pytest.raises(fi.IntelError, match="unknown kind"):
        good(kind="vibes")
    assert good(kind="short_notice")["kind"] == "short_notice"


def test_confidence_is_coarse_on_purpose():
    with pytest.raises(fi.IntelError):
        good(confidence=0.85)
    for value in fi.CONFIDENCE:
        assert good(confidence=value)["confidence"] == value


def test_a_fighter_is_required():
    with pytest.raises(fi.IntelError):
        good(fighter="  ")


# --- point in time ---------------------------------------------------------

def test_something_gathered_after_the_fight_is_not_evidence():
    before = good(gathered="2026-09-20")
    after = good(gathered="2026-09-27")
    assert fi.usable_at(before, "2026-09-26")
    assert not fi.usable_at(after, "2026-09-26")


def test_an_observation_with_no_timestamp_is_never_usable():
    record = good()
    record["gathered"] = ""
    assert not fi.usable_at(record, "2026-09-26")


def test_for_fight_hides_anything_gathered_too_late():
    store = {"schema": fi.SCHEMA, "observations": [
        good(gathered="2026-09-20"),
        good(kind="new_camp", gathered="2026-09-28",
             source="https://example.com/late"),
    ]}
    found = fi.for_fight(store, "Brady Hiestand", "Rinya Nakamura", "2026-09-26")
    assert len(found) == 1
    assert found[0]["kind"] == "injury"


def test_for_fight_matches_either_corner_and_ignores_other_events():
    store = {"schema": fi.SCHEMA, "observations": [
        good(fighter="Rinya Nakamura", gathered="2026-09-01"),
        good(event_date="2026-12-01", gathered="2026-09-01"),
    ]}
    found = fi.for_fight(store, "Brady Hiestand", "Rinya Nakamura", "2026-09-26")
    assert [o["fighter"] for o in found] == ["Rinya Nakamura"]


# --- it cannot move the number ---------------------------------------------

def test_nothing_gathered_today_moves_a_probability():
    """Every kind ships with applies=False because no weight is fitted."""
    store = {"schema": fi.SCHEMA, "observations": [
        good(), good(kind="short_notice", source="https://example.com/sn"),
        good(kind="hard_weight_cut", source="https://example.com/cut")]}
    found = fi.for_fight(store, "Brady Hiestand", "Rinya Nakamura", "2026-09-26")
    assert len(found) == 3
    assert fi.context_from(found, "Brady Hiestand", "Rinya Nakamura") == {}, (
        "an unfitted observation moved the prediction")


def test_a_kind_with_no_hook_cannot_claim_to_apply():
    with pytest.raises(fi.IntelError, match="no context hook"):
        good(kind="injury", applies=True)


def test_a_fitted_observation_drives_the_hook_it_names():
    """The path that exists so a measured weight has one place to land."""
    record = good(kind="short_notice", source="https://example.com/sn",
                  applies=True)
    assert fi.context_from([record], "Brady Hiestand", "Rinya Nakamura") == {
        "red_short_notice": True}
    # Same fact, other corner.
    assert fi.context_from([record], "Rinya Nakamura", "Brady Hiestand") == {
        "blue_short_notice": True}


def test_an_observation_about_neither_corner_is_dropped():
    """It is the wrong fight; guessing a corner would apply it to a stranger."""
    record = good(fighter="Somebody Else", kind="short_notice",
                  source="https://example.com/sn", applies=True)
    assert fi.context_from([record], "Brady Hiestand", "Rinya Nakamura") == {}


def test_every_kind_maps_to_a_real_context_key_or_to_none():
    """A hook that predict_card does not read would apply nothing, silently."""
    source = (ENGINE / "predict_card.py").read_text()
    for kind, suffix in fi.KINDS.items():
        if suffix is None:
            continue
        for corner in ("red", "blue"):
            key = f"{corner}_{suffix}"
            assert f"context.get('{key}'" in source, (
                f"{kind} claims to drive {key}, which predict_card never reads")


# --- the store -------------------------------------------------------------

def test_adding_is_append_only_and_refuses_duplicates(tmp_path):
    store = fi.load(tmp_path / "none.json")
    assert fi.add(store, [good(), good()]) == 1
    assert fi.add(store, [good(source="https://example.com/second")]) == 1
    assert len(store["observations"]) == 2


def test_a_correction_is_a_new_observation_not_an_edit(tmp_path):
    """Rewriting an observation destroys the only property that makes it
    usable: what was believed, and when."""
    store = fi.load(tmp_path / "none.json")
    fi.add(store, [good(gathered="2026-09-20", note="knee trouble reported")])
    fi.add(store, [good(gathered="2026-09-24", note="knee trouble denied by camp",
                        source="https://example.com/denial")])
    assert len(store["observations"]) == 2
    assert fi.for_fight(store, "Brady Hiestand", "x", "2026-09-26",
                        known_by="2026-09-22")[0]["note"].endswith("reported")


def test_a_round_trip_survives(tmp_path):
    path = tmp_path / "intel.json"
    store = fi.load(path)
    fi.add(store, [good()])
    fi.save(store, path)
    assert fi.load(path)["observations"] == store["observations"]


def test_a_store_from_a_future_schema_is_refused(tmp_path):
    path = tmp_path / "intel.json"
    path.write_text(json.dumps({"schema": 99, "observations": []}))
    with pytest.raises(fi.IntelError, match="schema"):
        fi.load(path)
