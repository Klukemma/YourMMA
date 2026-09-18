"""Tests for refilling bouts whose statistics never arrived.

The repair writes into the dataset every other part of the project reads, so
the guarantee that matters most is the negative one: it must not move a value
that is already there. Ratings are a continuous per-fighter series and the
win/loss records are point-in-time, both built from local history rather than
from any single upstream row, so overwriting them would silently invalidate
every rating downstream of the repaired fight.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import sync_kaggle
from sync_kaggle import PROTECTED_FROM_REPAIR, blank_stat_rows, repair_blank_stats


def local_frame():
    """Two bouts: the first has statistics, the second is blank."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-08-01", "2025-10-15"]),
        "r_name": ["Alpha", "Gamma"],
        "b_name": ["Beta", "Delta"],
        "r_total_str_landed": [55.0, None],
        "b_total_str_landed": [40.0, None],
        "r_td_atmpted": [3.0, None],
        "winner": ["Alpha", "Gamma"],
        "r_mmr_pre": [1500.0, 1610.0],
        "b_mmr_pre": [1490.0, 1480.0],
        "r_wins": [10.0, 4.0],
        "b_wins": [8.0, 6.0],
    })


def upstream_frame(**overrides):
    """The same two bouts as upstream now publishes them."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-08-01", "2025-10-15"]),
        "r_name": ["Alpha", "Gamma"],
        "b_name": ["Beta", "Delta"],
        "r_total_str_landed": [999.0, 77.0],
        "b_total_str_landed": [999.0, 62.0],
        "r_td_atmpted": [999.0, 5.0],
        "winner": ["Alpha", "Gamma"],
        "r_mmr_pre": [1.0, 1.0],
        "b_mmr_pre": [1.0, 1.0],
        "r_wins": [999.0, 999.0],
        "b_wins": [999.0, 999.0],
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def test_blank_rows_are_the_ones_without_statistics():
    mask = blank_stat_rows(local_frame())
    assert list(mask) == [False, True]


def test_the_blank_row_gets_filled():
    repaired, report = repair_blank_stats(local_frame(), upstream_frame())
    assert repaired.loc[1, "r_total_str_landed"] == 77.0
    assert repaired.loc[1, "b_total_str_landed"] == 62.0
    assert repaired.loc[1, "r_td_atmpted"] == 5.0
    assert report["rows_repaired"] == 1
    assert report["matched"] == 1


def test_an_existing_value_is_never_overwritten():
    """Upstream says 999 for the first bout. It already has 55, so it keeps 55."""
    repaired, _ = repair_blank_stats(local_frame(), upstream_frame())
    assert repaired.loc[0, "r_total_str_landed"] == 55.0
    assert repaired.loc[0, "r_td_atmpted"] == 3.0


def test_ratings_are_never_touched():
    repaired, _ = repair_blank_stats(local_frame(), upstream_frame())
    before = local_frame()
    for col in ["r_mmr_pre", "b_mmr_pre"]:
        assert repaired[col].equals(before[col]), f"{col} moved"


def test_carried_forward_records_are_never_touched():
    repaired, _ = repair_blank_stats(local_frame(), upstream_frame())
    before = local_frame()
    for col in ["r_wins", "b_wins"]:
        assert repaired[col].equals(before[col]), f"{col} moved"


def test_protected_columns_cover_ratings_and_records():
    import schema_map

    for col in schema_map.RATING_COLUMNS + schema_map.CARRIED_FORWARD:
        assert col in PROTECTED_FROM_REPAIR
    for col in ["date", "r_name", "b_name"]:
        assert col in PROTECTED_FROM_REPAIR, "identity must not be rewritten"


def test_a_bout_missing_upstream_is_reported_not_guessed():
    upstream = upstream_frame()
    upstream = upstream[upstream["r_name"] != "Gamma"]
    repaired, report = repair_blank_stats(local_frame(), upstream)
    assert report["matched"] == 0
    assert report["rows_repaired"] == 0
    assert len(report["unmatched"]) == 1
    assert pd.isna(repaired.loc[1, "r_total_str_landed"]), "no data means no data"


def test_upstream_still_blank_leaves_the_row_blank():
    """Upstream has the bout but has not backfilled it either."""
    upstream = upstream_frame(r_total_str_landed=[999.0, None],
                              b_total_str_landed=[999.0, None],
                              r_td_atmpted=[999.0, None])
    repaired, report = repair_blank_stats(local_frame(), upstream)
    assert report["matched"] == 1
    assert report["rows_repaired"] == 0
    assert report["cells_filled"] == 0
    assert pd.isna(repaired.loc[1, "r_total_str_landed"])


def test_nothing_to_repair_returns_the_frame_unchanged():
    clean = local_frame()
    clean.loc[1, "r_total_str_landed"] = 70.0
    repaired, report = repair_blank_stats(clean, upstream_frame())
    assert report["blank_rows"] == 0
    assert report["cells_filled"] == 0
    assert repaired.equals(clean)


def test_columns_absent_upstream_are_left_alone():
    """A local-only column must not be dropped or blanked by the repair."""
    local = local_frame()
    local["local_only"] = [1.0, None]
    repaired, _ = repair_blank_stats(local, upstream_frame())
    assert "local_only" in repaired.columns
    assert pd.isna(repaired.loc[1, "local_only"])


def test_duplicate_upstream_keys_do_not_raise():
    upstream = pd.concat([upstream_frame(), upstream_frame()], ignore_index=True)
    repaired, report = repair_blank_stats(local_frame(), upstream)
    assert report["rows_repaired"] == 1
    assert repaired.loc[1, "r_total_str_landed"] == 77.0


def test_repair_is_a_registered_command():
    assert hasattr(sync_kaggle, "cmd_repair")
