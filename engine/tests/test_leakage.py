"""Tests for the career-total leak detector.

This is the check that should have existed before the backtest reported
+16.2% over 5,943 bets. The existing audits verify shift(1) on rolling
windows, which is correct and catches nothing here, because the leaking
columns are not rolling - they are joined career totals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from leakage import PROFILE_COLUMNS, constant_across_career, scan


def career(name, values, column="r_splm"):
    return pd.DataFrame({"r_name": [name] * len(values), column: values})


def test_a_career_total_is_caught():
    """The same number on every bout: computed over the whole career."""
    df = career("Fighter", [4.2] * 8)
    assert constant_across_career(df, "r_splm")["constant_share"] == 1.0


def test_a_point_in_time_statistic_is_not_caught():
    """A running average moves as the career progresses."""
    df = career("Fighter", [3.0, 3.4, 3.2, 3.9, 4.1, 4.0, 4.3, 4.2])
    assert constant_across_career(df, "r_splm")["constant_share"] == 0.0


def test_fighters_with_too_few_bouts_are_ignored():
    """One or two fights say nothing about whether a value moves."""
    df = pd.concat([career("Rookie", [4.2, 4.2]),
                    career("Veteran", [3.0, 3.2, 3.4, 3.6, 3.8, 4.0])])
    out = constant_across_career(df, "r_splm")
    assert out["fighters"] == 1
    assert out["constant_share"] == 0.0


def test_a_mixed_roster_reports_the_share():
    df = pd.concat([career("Leaky", [4.2] * 6),
                    career("Honest", [3.0, 3.2, 3.4, 3.6, 3.8, 4.0])])
    assert constant_across_career(df, "r_splm")["constant_share"] == pytest.approx(0.5)


def test_a_missing_column_makes_no_claim():
    assert constant_across_career(career("F", [1.0] * 6), "not_a_column") is None


def test_an_empty_roster_makes_no_claim():
    assert constant_across_career(pd.DataFrame({"r_name": [], "r_splm": []}),
                                  "r_splm") is None


def test_missing_values_do_not_count_as_a_change():
    """NaN is absence, not a second distinct value."""
    df = career("Fighter", [4.2, np.nan, 4.2, 4.2, np.nan, 4.2, 4.2])
    assert constant_across_career(df, "r_splm")["constant_share"] == 1.0


def test_scan_returns_only_the_suspicious_columns():
    df = pd.concat([
        career("A", [4.2] * 6).assign(r_str_acc=[50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
        career("B", [3.9] * 6).assign(r_str_acc=[40.0, 41.0, 42.0, 43.0, 44.0, 45.0]),
    ])
    found = [r["column"] for r in scan(df, ["r_splm", "r_str_acc"])]
    assert found == ["r_splm"]


def test_the_real_dataset_still_shows_the_leak():
    """Recorded so the fix can be verified: when these columns become
    point-in-time this test fails and should be updated, not deleted."""
    csv = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not csv.exists():
        pytest.skip("dataset not present")
    df = pd.read_csv(csv, low_memory=False)
    findings = {r["column"]: r["constant_share"] for r in scan(df, PROFILE_COLUMNS)}
    assert "r_splm" in findings, "the known leak has vanished; verify the fix"
    assert findings["r_splm"] > 0.5
