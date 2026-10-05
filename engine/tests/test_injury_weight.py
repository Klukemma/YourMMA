"""The injury experiment's populations may only use what was known before."""

import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.injury_weight import populations


def history(rows):
    columns = ["fighter", "matched", "vague", "kind", "disclosed", "surgery",
               "months_out", "anchor_date", "known_by"]
    return pd.DataFrame(rows, columns=columns)


def test_a_revelation_after_a_fight_is_not_in_that_fights_populations():
    h = history([("brown", True, False, "fought_hurt", "after", False, None,
                  "2017-01-28", "2017-01-28")])
    fights = [("brown", pd.Timestamp("2017-01-28"), pd.Timestamp("2016-03-05"))]
    assert all(not members for members in populations(h, fights).values())


def test_it_is_in_the_next_fights():
    h = history([("brown", True, False, "fought_hurt", "after", False, None,
                  "2017-01-28", "2017-01-28")])
    fights = [("brown", pd.Timestamp("2017-04-08"), pd.Timestamp("2017-01-28"))]
    assert ("brown", pd.Timestamp("2017-04-08").date()) in \
        populations(h, fights)["any_known_injury_1y"]


def test_a_withdrawal_between_fights_marks_the_return():
    h = history([("brown", True, False, "withdrawal", "before", False, None,
                  "2016-07-09", "2016-07-08")])
    fights = [("brown", pd.Timestamp("2017-01-28"), pd.Timestamp("2016-03-05"))]
    assert ("brown", pd.Timestamp("2017-01-28").date()) in \
        populations(h, fights)["withdrew_since_last_fight"]


def test_hurt_in_the_previous_bout_is_seen_at_the_next_one_only():
    h = history([("brown", True, False, "in_fight", "at", False, None,
                  "2017-01-28", "2017-01-28")])
    same = populations(h, [("brown", pd.Timestamp("2017-01-28"),
                            pd.Timestamp("2016-03-05"))])
    later = populations(h, [("brown", pd.Timestamp("2017-04-08"),
                             pd.Timestamp("2017-01-28"))])
    assert not same["hurt_in_last_fight"]
    assert later["hurt_in_last_fight"]


def test_a_pre_fight_injury_report_counts_for_that_fight():
    h = history([("brown", True, False, "fought_hurt", "before", False, None,
                  "2017-01-28", "2017-01-27")])
    got = populations(h, [("brown", pd.Timestamp("2017-01-28"),
                           pd.Timestamp("2016-03-05"))])
    assert got["known_hurt_going_in"]


def test_vague_and_unmatched_records_are_never_used():
    h = history([("brown", True, True, "withdrawal", "unclear", False, None,
                  "2016-07-09", "2016-07-09"),
                 ("brown", False, False, "withdrawal", "before", False, None,
                  "2016-07-09", "2016-07-08")])
    got = populations(h, [("brown", pd.Timestamp("2017-01-28"),
                           pd.Timestamp("2016-03-05"))])
    assert all(not members for members in got.values())
