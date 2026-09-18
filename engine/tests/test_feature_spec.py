"""Tests for the feature primitives.

Every feature in the model is built from these four functions, so a defect
here is a defect in sixty features at once. The fill rule gets the most
attention because inverting it is what broke fifteen features in the
hand-written version.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from feature_spec import (
    Derived,
    Interaction,
    Paired,
    build_all,
    emitted_names,
    interaction,
    paired_diff,
    paired_known,
    paired_level,
    ratio,
    to_number,
)


# --- the rule that matters -------------------------------------------------

def test_a_missing_operand_makes_the_difference_unknown():
    """The old code gave 0 - 45 = -45: a debutant handed away a 45-point gap."""
    red = pd.Series([np.nan])
    blue = pd.Series([45.0])
    assert pd.isna(paired_diff(red, blue).iloc[0])


def test_a_missing_operand_is_never_read_as_zero():
    """0% striking accuracy is the worst fighter alive, not a neutral default."""
    out = paired_diff(pd.Series([np.nan, 50.0]), pd.Series([45.0, 45.0]))
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(5.0)


def test_both_missing_is_still_unknown_not_even():
    assert pd.isna(paired_diff(pd.Series([np.nan]), pd.Series([np.nan])).iloc[0])


def test_a_real_difference_is_plain_subtraction():
    out = paired_diff(pd.Series([60.0, 40.0]), pd.Series([45.0, 45.0]))
    assert list(out) == [15.0, -5.0]


def test_text_that_is_not_a_number_becomes_unknown():
    assert pd.isna(paired_diff(pd.Series(["n/a"]), pd.Series([45.0])).iloc[0])


def test_numeric_strings_still_work():
    assert paired_diff(pd.Series(["60"]), pd.Series(["45"])).iloc[0] == 15.0


# --- level -----------------------------------------------------------------

def test_level_is_the_mean_of_the_pair():
    assert paired_level(pd.Series([60.0]), pd.Series([40.0])).iloc[0] == 50.0


def test_level_separates_fights_a_difference_cannot():
    """Two elites and two novices can share a difference of zero."""
    elite = paired_level(pd.Series([80.0]), pd.Series([80.0])).iloc[0]
    novice = paired_level(pd.Series([20.0]), pd.Series([20.0])).iloc[0]
    assert paired_diff(pd.Series([80.0]), pd.Series([80.0])).iloc[0] == \
           paired_diff(pd.Series([20.0]), pd.Series([20.0])).iloc[0]
    assert elite != novice


def test_level_is_unknown_when_a_side_is_missing():
    assert pd.isna(paired_level(pd.Series([np.nan]), pd.Series([40.0])).iloc[0])


# --- known flag ------------------------------------------------------------

def test_known_is_one_only_when_both_sides_are_present():
    red = pd.Series([1.0, np.nan, 1.0, np.nan])
    blue = pd.Series([1.0, 1.0, np.nan, np.nan])
    assert list(paired_known(red, blue)) == [1.0, 0.0, 0.0, 0.0]


def test_known_distinguishes_an_even_matchup_from_no_data():
    """Both give a difference of 0 once the matrix fills NaN; only the flag
    tells them apart."""
    even = paired_known(pd.Series([45.0]), pd.Series([45.0])).iloc[0]
    unknown = paired_known(pd.Series([np.nan]), pd.Series([np.nan])).iloc[0]
    assert even == 1.0 and unknown == 0.0


# --- ratio -----------------------------------------------------------------

def test_ratio_divides():
    assert ratio(pd.Series([10.0]), pd.Series([4.0])).iloc[0] == 2.5


def test_a_zero_denominator_is_unknown_rather_than_infinite():
    assert pd.isna(ratio(pd.Series([10.0]), pd.Series([0.0])).iloc[0])


def test_a_tiny_denominator_does_not_explode():
    assert pd.isna(ratio(pd.Series([10.0]), pd.Series([1e-12])).iloc[0])


# --- interaction -----------------------------------------------------------

def test_an_interaction_is_zero_when_either_side_is_average():
    attack = pd.Series([1.0, 2.0, 3.0])
    defence = pd.Series([2.0, 2.0, 2.0])
    assert list(interaction(attack, defence)) == [0.0, 0.0, 0.0]


def test_high_offence_against_high_defence_is_positive():
    """Both above their means: the crossing is the point of the term."""
    attack = pd.Series([1.0, 5.0, 3.0])
    defence = pd.Series([1.0, 5.0, 3.0])
    assert interaction(attack, defence).iloc[1] > 0


def test_high_offence_against_low_defence_is_negative():
    attack = pd.Series([1.0, 5.0, 3.0])
    defence = pd.Series([5.0, 1.0, 3.0])
    assert interaction(attack, defence).iloc[1] < 0


# --- the declaration -------------------------------------------------------

FRAME = pd.DataFrame({
    "r_acc": [50.0, np.nan, 40.0],
    "b_acc": [45.0, 45.0, 40.0],
    "r_td": [2.0, 1.0, 0.0],
    "b_tdd": [60.0, 70.0, 80.0],
})


def test_a_paired_spec_emits_only_what_it_declares():
    spec = Paired("acc", "r_acc", "b_acc", why="striking accuracy")
    assert spec.emits == ["acc_diff"]
    assert list(spec.build(FRAME)) == ["acc_diff"]


def test_level_and_known_are_opt_in():
    spec = Paired("acc", "r_acc", "b_acc", why="x", level=True, known=True)
    assert spec.emits == ["acc_diff", "acc_level", "acc_known"]


def test_emitted_names_match_what_build_all_produces():
    """The bug this design exists to prevent: a name declared but not built."""
    specs = [
        Paired("acc", "r_acc", "b_acc", why="x", level=True, known=True),
        Interaction("td_vs_tdd", "r_td", "b_tdd", why="y"),
        Derived("const", lambda df: pd.Series(1.0, index=df.index), why="z"),
    ]
    built = build_all(specs, FRAME)
    assert list(built.columns) == emitted_names(specs)


def test_build_all_preserves_the_index():
    frame = FRAME.set_index(pd.Index([10, 11, 12]))
    specs = [Paired("acc", "r_acc", "b_acc", why="x")]
    assert list(build_all(specs, frame).index) == [10, 11, 12]


def test_a_single_row_builds_the_same_columns_as_many():
    """Training and a single fight must go through the same code."""
    specs = [Paired("acc", "r_acc", "b_acc", why="x", level=True, known=True)]
    many = build_all(specs, FRAME)
    one = build_all(specs, FRAME.iloc[[0]])
    assert list(many.columns) == list(one.columns)


def test_every_spec_carries_a_reason():
    specs = [Paired("acc", "r_acc", "b_acc", why="striking accuracy")]
    for spec in specs:
        assert spec.why and len(spec.why) > 3
