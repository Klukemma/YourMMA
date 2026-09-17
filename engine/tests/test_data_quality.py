"""Tests for detecting cells that hold a printed Series instead of a value."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from data_quality import (describe_corruption, find_corrupt_cells,
                          is_series_repr, parse_series_repr)

CORRUPT = ("name_norm\nbruno silva    14\nbruno silva    23\n"
           "Name: wins, dtype: int64")


def test_detects_a_series_repr():
    assert is_series_repr(CORRUPT)


@pytest.mark.parametrize("value", ["14", "Bruno Silva", "", None, 14, 14.5])
def test_ordinary_values_are_not_flagged(value):
    assert not is_series_repr(value)


def test_parses_the_values_back_out():
    assert parse_series_repr(CORRUPT) == ["14", "23"]


def test_parses_float_and_text_series():
    assert parse_series_repr(
        "name_norm\nx    162.56\nx    182.88\nName: height, dtype: float64"
    ) == ["162.56", "182.88"]
    assert parse_series_repr(
        "name_norm\nx    Orthodox\nx    Southpaw\nName: stance, dtype: object"
    ) == ["Orthodox", "Southpaw"]


def test_finds_corrupt_cells_in_a_frame():
    df = pd.DataFrame({"a": ["fine", CORRUPT], "b": [1, 2]})
    assert find_corrupt_cells(df) == [(1, "a")]


def test_works_on_pandas_3_string_dtype():
    """pandas 3 gives text columns dtype str, not object.

    An earlier version checked `dtype != object` and so found nothing at all
    on pandas 3, silently passing corrupt data straight through.
    """
    df = pd.DataFrame({"a": pd.Series(["fine", CORRUPT], dtype="str")})
    assert find_corrupt_cells(df) == [(1, "a")]


def test_shipped_dataset_is_clean():
    """Regression guard: row 8138 held two fighters named Bruno Silva."""
    df = pd.read_csv(ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv",
                     low_memory=False)
    assert describe_corruption(df) == {}


def test_repaired_row_has_the_flyweight_not_the_middleweight():
    """Located by content, not row number - a data sync shifts the index."""
    df = pd.read_csv(ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv",
                     low_memory=False)
    match = df[(df["date"] == "2025-10-18")
               & (df["r_name"] == "Bruno Silva")
               & (df["b_name"] == "HyunSung Park")]
    assert len(match) == 1, "the repaired bout should appear exactly once"
    row = match.iloc[0]
    assert row["division"] == "flyweight"
    assert float(row["r_height"]) == 162.56
    assert float(row["r_weight"]) == 56.7
    assert float(row["r_wins"]) == 14
