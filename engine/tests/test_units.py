"""Tests for the unit and format conversions on upstream profile columns.

The local dataset stores metric. Upstream publishes imperial and says so in its
column names, so mapping r_reach_inches straight onto r_reach wrote inches into
a centimetres column. Nothing was null and nothing raised: 358 synced rows just
carried a mean reach of 71.5 where every row before them had 181.9, and the
model read every 2026 fighter as freakishly short-armed.

These tests pin the conversions and the guard that should have caught it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from transform import (
    _height_to_cm,
    _inches_to_cm,
    _lbs_to_kg,
    _normalise_dob,
)
from sync_kaggle import distribution_shift


# --- reach -----------------------------------------------------------------

def test_reach_inches_become_centimetres():
    assert _inches_to_cm(71.5) == pytest.approx(181.61, abs=0.01)


def test_reach_lands_in_the_range_the_existing_rows_use():
    """Every row before the sync had a reach near 182cm, not near 72."""
    assert 160 < _inches_to_cm(72.0) < 210


def test_reach_handles_a_series_and_keeps_blanks_blank():
    out = _inches_to_cm(pd.Series([71.5, None, 76.0]))
    assert out.iloc[0] == pytest.approx(181.61, abs=0.01)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(193.04, abs=0.01)


# --- weight ----------------------------------------------------------------

def test_weight_pounds_become_kilograms():
    assert _lbs_to_kg(155) == pytest.approx(70.31, abs=0.01)


def test_a_lightweight_weighs_about_seventy_kilos():
    """155lb is the UFC lightweight limit; the local column stores ~70."""
    assert 65 < _lbs_to_kg(155) < 75


# --- height ----------------------------------------------------------------

def test_feet_and_inches_become_centimetres():
    assert _height_to_cm("6' 3\"") == pytest.approx(190.5, abs=0.01)
    assert _height_to_cm("5' 10\"") == pytest.approx(177.8, abs=0.01)


def test_whole_feet_with_no_inches():
    assert _height_to_cm("6'") == pytest.approx(182.88, abs=0.01)


def test_a_number_is_already_centimetres_and_is_not_converted_twice():
    """Re-running over converted data must be a no-op, not a second convert."""
    assert _height_to_cm("185.42") == pytest.approx(185.42, abs=0.01)
    assert _height_to_cm(185.42) == pytest.approx(185.42, abs=0.01)


def test_unparseable_height_is_blank_rather_than_wrong():
    assert pd.isna(_height_to_cm("six foot"))
    assert pd.isna(_height_to_cm(""))
    assert pd.isna(_height_to_cm(np.nan))


# --- date of birth ---------------------------------------------------------

def test_dob_separator_is_normalised():
    assert _normalise_dob("1987-08-03") == "1987/08/03"


def test_dob_already_in_local_format_is_unchanged():
    assert _normalise_dob("1992/07/17") == "1992/07/17"


def test_unparseable_dob_is_blank():
    assert pd.isna(_normalise_dob("not a date"))
    assert pd.isna(_normalise_dob(np.nan))


# --- the guard -------------------------------------------------------------

def frames(new_reach):
    existing = pd.DataFrame({"r_reach": np.full(200, 182.0) + np.arange(200) % 7})
    new = pd.DataFrame({"r_reach": np.full(60, new_reach) + np.arange(60) % 7})
    return existing, new


def test_the_reach_bug_is_caught():
    """Inches written into a centimetres column must stop the sync."""
    shifts = distribution_shift(*frames(71.5))
    assert [r["column"] for r in shifts] == ["r_reach"]
    assert shifts[0]["shift"] < -2


def test_ordinary_drift_passes():
    """A couple of centimetres year on year is not a unit error."""
    assert distribution_shift(*frames(184.0)) == []


def test_a_column_missing_from_the_new_rows_is_not_a_shift():
    existing, _ = frames(182.0)
    assert distribution_shift(existing, pd.DataFrame({"other": [1.0] * 60})) == []


def test_text_columns_are_ignored():
    existing = pd.DataFrame({"r_name": ["Alpha"] * 200})
    new = pd.DataFrame({"r_name": ["Beta"] * 60})
    assert distribution_shift(existing, new) == []


def test_too_few_rows_to_judge_is_not_a_shift():
    existing = pd.DataFrame({"r_reach": np.full(200, 182.0) + np.arange(200) % 7})
    new = pd.DataFrame({"r_reach": [71.5, 72.0]})
    assert distribution_shift(existing, new) == []


def test_a_constant_column_does_not_divide_by_zero():
    existing = pd.DataFrame({"flag": np.zeros(200)})
    new = pd.DataFrame({"flag": np.ones(60)})
    assert distribution_shift(existing, new) == []


def test_shifts_are_reported_worst_first():
    existing = pd.DataFrame({
        "small": np.arange(200) % 7 + 100.0,
        "huge": np.arange(200) % 7 + 100.0,
    })
    new = pd.DataFrame({
        "small": np.arange(60) % 7 + 106.0,
        "huge": np.arange(60) % 7 + 200.0,
    })
    shifts = distribution_shift(existing, new)
    assert [r["column"] for r in shifts] == ["huge", "small"]


# --- fixing rows already written -------------------------------------------

from transform import fix_units, rows_in_imperial_units  # noqa: E402


def mixed_frame():
    """Two rows as the original builder wrote them, two as the broken sync did."""
    return pd.DataFrame({
        "r_height": ["185.42", "175.26", "6' 3\"", "5' 10\""],
        "b_height": ["177.8", "167.64", "6' 0\"", "5' 9\""],
        "r_reach": [193.04, 182.88, 76.0, 70.0],
        "b_reach": [185.42, 177.8, 74.0, 69.0],
        "r_weight": [70.31, 83.91, 155.0, 185.0],
        "b_weight": [70.31, 83.91, 155.0, 185.0],
        "r_dob": ["1992/07/17", "1995/09/07", "1987-08-03", "1998-04-15"],
        "b_dob": ["1985/08/11", "1999/01/17", "1993-09-09", "1994-04-20"],
    })


def test_only_the_imperial_rows_are_flagged():
    assert list(rows_in_imperial_units(mixed_frame())) == [False, False, True, True]


def test_metric_rows_are_left_exactly_as_they_were():
    fixed, _ = fix_units(mixed_frame())
    before = mixed_frame()
    for col in before.columns:
        assert str(fixed.loc[0, col]) == str(before.loc[0, col]), col
        assert str(fixed.loc[1, col]) == str(before.loc[1, col]), col


def test_imperial_rows_are_converted():
    fixed, report = fix_units(mixed_frame())
    assert report["rows"] == 2
    assert fixed.loc[2, "r_reach"] == pytest.approx(193.04, abs=0.01)
    assert fixed.loc[2, "r_weight"] == pytest.approx(70.31, abs=0.01)
    assert fixed.loc[2, "r_height"] == pytest.approx(190.5, abs=0.01)
    assert fixed.loc[2, "r_dob"] == "1987/08/03"


def test_running_twice_changes_nothing_the_second_time():
    """The fix must be safe to re-run; a second convert would halve reach."""
    once, first = fix_units(mixed_frame())
    twice, second = fix_units(once)
    assert first["rows"] == 2
    assert second["rows"] == 0
    assert twice.equals(once)


def test_a_frame_already_metric_is_untouched():
    metric = mixed_frame().iloc[:2].reset_index(drop=True)
    fixed, report = fix_units(metric)
    assert report["rows"] == 0
    assert fixed.equals(metric)


def test_the_converted_reach_sits_with_the_metric_rows():
    """The whole point: after the fix, both halves are on one scale."""
    fixed, _ = fix_units(mixed_frame())
    assert fixed["r_reach"].min() > 150
    assert fixed["r_weight"].max() < 120


def test_fix_units_is_a_registered_command():
    import sync_kaggle

    assert hasattr(sync_kaggle, "cmd_fix_units")
