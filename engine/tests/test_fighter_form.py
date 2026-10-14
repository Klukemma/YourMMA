"""Tests for the pro record and last-five shown beside each fighter's name.

Display only, never a model input - but a wrong record next to a name is
worse than none, so the rules are pinned: only bouts before the card, the
codes the app prints, one bout never counted twice, and a debutant's
name-only match marked as such.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import fighter_form as ff


@pytest.mark.parametrize("method,rnd,code", [
    ("KO (Punch)", "1", "K1"), ("TKO (Punches)", "2", "K2"),
    ("TKO (Doctor Stoppage)", "3", "K3"), ("KO/TKO", 1, "K1"),
    ("Technical Knockout", "2", "K2"),
    ("Submission (Rear-Naked Choke)", "2", "S2"), ("SUB Guillotine Choke", 1, "S1"),
    ("Technical Submission (Arm-Triangle Choke)", "1", "S1"),
    ("Decision (Unanimous)", "3", "DC"), ("U-DEC", 3, "DC"), ("Decision - Split", 3, "DC"),
    ("Technical Decision (Majority)", "3", "DC"),
    ("DQ (Illegal Knee)", "2", "DQ"), ("Disqualification", "1", "DQ"),
    ("No Contest (Accidental Eye Poke)", "1", "NC"), ("Overturned", 2, "NC"),
    ("Could Not Continue", 1, "NC"), ("N/A", "1", "??"), (None, None, "??"),
    ("KO (Punch)", "0", "K?"), ("Submission", None, "S?"),
    ("TKO (Submission to Punches)", "1", "K1"),
    ("TKO (Submission to Leg Kicks and Punches)", "1", "K1"),
    ("No Decision (Overturned by NSAC)", "3", "NC"),
    ("ND (Decision Overturned by NSAC)", "3", "NC"),
    ("No Decision - Overturned by CSAC", "2", "NC"),
    ("Technical Draw (Majority)", "2", "DC"),
])
def test_how_a_bout_ended_is_two_characters(method, rnd, code):
    assert ff.method_code(method, rnd) == code


def _sherdog(record, verified=True, looked_up="2026-10-02"):
    return {"Sam Fighter": {"target": "Sam Fighter", "id": 1, "verified": verified,
                            "looked_up": looked_up, "record": record}}


def _bout(day, result, method="Decision (Unanimous)", rnd="3", opponent="Other Guy"):
    return {"date": day, "result": result, "method": method, "round": rnd,
            "opponent": opponent, "event": "Some Card"}


def test_the_record_and_last_five_newest_first():
    rec = [_bout("2026-01-01", "win", "KO (Punch)", "1", "A A"),
           _bout("2025-06-01", "loss", "Submission (Armbar)", "2", "B B"),
           _bout("2025-01-01", "draw", opponent="C C"),
           _bout("2024-06-01", "nc", "No Contest", "1", "D D"),
           _bout("2024-01-01", "win", opponent="E E"),
           _bout("2023-01-01", "win", opponent="F F")]
    out = ff.form("Sam Fighter", "2026-10-17", _sherdog(rec))
    assert out["record"] == {"wins": 3, "losses": 1, "draws": 1, "nc": 1}
    assert [(b["result"], b["code"]) for b in out["last5"]] == [
        ("W", "K1"), ("L", "S2"), ("D", "DC"), ("NC", "NC"), ("W", "DC")]
    assert out["source"] == "sherdog" and out["verified"] is True


def test_nothing_on_or_after_the_card_counts():
    """A record harvested after the fight must not show its result."""
    rec = [_bout("2026-10-17", "win", "KO (Punch)", "1"), _bout("2026-01-01", "loss")]
    out = ff.form("Sam Fighter", "2026-10-17", _sherdog(rec))
    assert out["record"] == {"wins": 0, "losses": 1, "draws": 0, "nc": 0}
    assert [b["result"] for b in out["last5"]] == ["L"]


def _archive(rows):
    return pd.DataFrame(rows, columns=["date", "r_name", "b_name", "winner",
                                       "method", "finish_round", "event_name"])


def test_only_ufc_bouts_after_the_sherdog_lookup_are_added_and_once():
    """The record was complete on the day it was read: a UFC bout before
    that is already in it under another spelling ('Sopai'/'Sopaj') and was
    counted twice."""
    rec = [_bout("2026-01-01", "win", opponent="Bernardo Sopai")]
    arch = _archive([
        ["2026-01-01", "Sam Fighter", "Benardo Sopaj", "Sam Fighter", "U-DEC", 3, "UFC X"],
        ["2026-05-09", "New Rival", "Sam Fighter", "New Rival", "KO/TKO", 2, "UFC Y"],
        ["2026-05-10", "Sam Fighter", "New Rival", "New Rival", "KO/TKO", 2, "UFC Y"]])  # same bout, a day off
    out = ff.form("Sam Fighter", "2026-10-17", _sherdog(rec, looked_up="2026-02-01"), arch)
    assert out["record"] == {"wins": 1, "losses": 1, "draws": 0, "nc": 0}
    assert [(b["result"], b["code"], b["opponent"]) for b in out["last5"]] == [
        ("L", "K2", "New Rival"), ("W", "DC", "Bernardo Sopai")]


def test_a_namesakes_ufc_bouts_never_enter_a_verified_record():
    """The archive's 'Bruno Silva' is two men; the Sherdog row is one."""
    rec = [_bout("2025-06-07", "win", opponent="Joshua Van")]
    arch = _archive([["2025-05-10", "Marc-Andre Barriault", "Sam Fighter",
                      "Marc-Andre Barriault", "KO/TKO", 1, "UFC Z"]])
    out = ff.form("Sam Fighter", "2026-10-17", _sherdog(rec), arch)
    assert out["record"] == {"wins": 1, "losses": 0, "draws": 0, "nc": 0}


def test_without_sherdog_the_ufc_bouts_alone_are_marked_ufc():
    arch = _archive([["2026-03-01", "Sam Fighter", "X Y", "X Y", "SUB Armbar", 1, "UFC Z"],
                     ["2025-03-01", "Z W", "Sam Fighter", None, "Overturned", 2, "UFC Q"]])
    out = ff.form("Sam Fighter", "2026-10-17", {}, arch)
    assert out["source"] == "ufc"
    assert [(b["result"], b["code"]) for b in out["last5"]] == [("L", "S1"), ("NC", "NC")]


def test_a_name_only_match_is_marked_unverified():
    out = ff.form("Sam Fighter", "2026-10-17",
                  _sherdog([_bout("2025-01-01", "win")], verified=False))
    assert out["verified"] is False


def test_an_unknown_fighter_has_no_record_rather_than_zero_and_zero():
    assert ff.form("Nobody Here", "2026-10-17", {}, _archive([])) is None


def test_the_real_harvest_parses_into_codes_the_app_can_print():
    sherdog = ff.load_sherdog()
    if not sherdog:
        pytest.skip("no harvest on file")
    allowed = {"DC", "DQ", "NC", "??"} | {f"{k}{r}" for k in "KS" for r in list("123456789?")}
    bad = [(n, b.get("method"), b.get("round")) for n, row in sherdog.items()
           for b in row.get("record") or []
           if ff.method_code(b.get("method"), b.get("round")) not in allowed]
    assert len(bad) <= 5, bad[:5]           # ten-round bouts from another era
