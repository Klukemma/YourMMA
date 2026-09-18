"""Tests for the blank-window exposure split.

The experiment's conclusion rests on classifying each fight by how many of its
two fighters have a bout inside a date window, so that classification is worth
pinning down: an off-by-one on the window boundary, or a silent failure to match
names, would quietly turn the result into noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.stale_form import (
    affected_count,
    blank_window_bounds,
    fighters_active_in,
    score,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "r_name", "b_name"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


ROSTER = frame([
    ("2025-09-12", "Early", "Partner"),      # day before the window
    ("2025-09-13", "OnOpen", "Partner"),     # first day, inclusive
    ("2025-10-20", "Middle", "Partner"),
    ("2025-12-06", "OnClose", "Partner"),    # last day, inclusive
    ("2025-12-07", "Late", "Partner"),       # day after
])

START = pd.Timestamp("2025-09-13")
END = pd.Timestamp("2025-12-06")


def test_window_is_inclusive_at_both_ends():
    active = fighters_active_in(ROSTER, START, END)
    assert "OnOpen" in active, "a fight on the opening day is inside the window"
    assert "OnClose" in active, "a fight on the closing day is inside the window"


def test_window_excludes_the_days_either_side():
    active = fighters_active_in(ROSTER, START, END)
    assert "Early" not in active
    assert "Late" not in active


def test_both_corners_count():
    """A fighter is exposed whichever corner they fought from."""
    red_only = frame([("2025-10-01", "Fighter", "Someone")])
    blue_only = frame([("2025-10-01", "Someone", "Fighter")])
    assert "Fighter" in fighters_active_in(red_only, START, END)
    assert "Fighter" in fighters_active_in(blue_only, START, END)


def test_affected_count_is_zero_one_or_two():
    bouts = frame([
        ("2026-01-01", "Clean", "AlsoClean"),
        ("2026-01-02", "Exposed", "Clean"),
        ("2026-01-03", "Exposed", "AlsoExposed"),
    ])
    counts = affected_count(bouts, {"Exposed", "AlsoExposed"})
    assert list(counts) == [0, 1, 2]


def test_affected_count_ignores_missing_names():
    bouts = pd.DataFrame({"r_name": [None], "b_name": ["Exposed"]})
    assert list(affected_count(bouts, {"Exposed"})) == [1]


def test_score_refuses_a_single_class():
    """AUC is undefined when every fight went the same way; say so, don't guess."""
    p = np.array([0.6, 0.7, 0.8])
    assert score(p, np.array([1.0, 1.0, 1.0])) is None


def test_score_reports_a_perfect_ranking():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = score(p, y)
    assert s["n"] == 4
    assert s["auc"] == pytest.approx(1.0)
    assert s["accuracy"] == pytest.approx(1.0)


def test_score_reports_an_inverted_ranking():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert score(p, y)["auc"] == pytest.approx(0.0)


def test_blank_window_matches_the_constants_in_the_module():
    """If a future sync repairs or moves the hole, this test should fail loudly
    rather than let the experiment keep splitting on a window that no longer
    describes the data."""
    from experiments import stale_form

    start, end, n = blank_window_bounds()
    if start is None:
        pytest.skip("no blank statistics in the dataset any more")
    assert start == stale_form.BLANK_START
    assert end == stale_form.BLANK_END
    assert n > 0


def test_placebo_window_is_a_year_before_and_the_same_length():
    from experiments import stale_form

    real = stale_form.BLANK_END - stale_form.BLANK_START
    sham = stale_form.PLACEBO_END - stale_form.PLACEBO_START
    assert sham == real, "the placebo must cover the same span to be comparable"
    assert stale_form.PLACEBO_END < stale_form.BLANK_START, \
        "the placebo window must not overlap the real one"
