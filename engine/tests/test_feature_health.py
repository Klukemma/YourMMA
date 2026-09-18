"""Tests for the feature-health checks.

These decide which variables get attention, so a check that quietly misses a
dead feature is worse than no check. Each test is written as the defect it is
meant to find.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.feature_health import (
    duplicate_pairs,
    degenerate_recently,
    mostly_imputed,
    near_constant,
    univariate_auc,
)

RNG = np.random.default_rng(0)


def test_a_truly_constant_feature_is_caught():
    X = pd.DataFrame({"flat": np.ones(200), "real": RNG.normal(size=200)})
    assert [r["feature"] for r in near_constant(X)] == ["flat"]


def test_a_nearly_constant_feature_is_caught():
    """base_prob: valid, varying, and carrying almost nothing."""
    values = np.full(200, 0.5)
    values[:5] = 0.51
    X = pd.DataFrame({"base_prob": values, "real": RNG.normal(size=200)})
    assert [r["feature"] for r in near_constant(X)] == ["base_prob"]


def test_a_normal_feature_is_not_flagged():
    X = pd.DataFrame({"real": RNG.normal(size=500)})
    assert near_constant(X) == []


def test_a_balanced_binary_flag_is_not_flagged():
    """A 50/50 flag varies perfectly well despite having two values."""
    X = pd.DataFrame({"flag": np.tile([0.0, 1.0], 100)})
    assert near_constant(X) == []


def test_a_rare_flag_is_flagged():
    """One value on 99% of rows leaves almost nothing to split on."""
    flag = np.zeros(500)
    flag[:3] = 1.0
    assert [r["feature"] for r in near_constant(pd.DataFrame({"rare": flag}))] == ["rare"]


def test_mostly_imputed_counts_the_fill_value():
    values = np.zeros(100)
    values[:10] = RNG.normal(size=10)
    out = mostly_imputed(pd.DataFrame({"sparse": values}))
    assert out[0]["feature"] == "sparse"
    assert out[0]["share"] == pytest.approx(0.90, abs=0.01)


def test_a_dense_feature_is_not_called_imputed():
    X = pd.DataFrame({"dense": RNG.normal(size=500)})
    assert mostly_imputed(X) == []


def test_duplicate_features_are_paired():
    base = RNG.normal(size=300)
    X = pd.DataFrame({"a": base, "b": base * 2.0 + 1.0, "c": RNG.normal(size=300)})
    pairs = duplicate_pairs(X)
    assert len(pairs) == 1
    assert {pairs[0]["a"], pairs[0]["b"]} == {"a", "b"}


def test_a_negative_duplicate_is_caught():
    """r = -1 is just as redundant as r = +1."""
    base = RNG.normal(size=300)
    pairs = duplicate_pairs(pd.DataFrame({"a": base, "b": -base}))
    assert len(pairs) == 1


def test_merely_correlated_features_are_left_alone():
    base = RNG.normal(size=2000)
    X = pd.DataFrame({"a": base, "b": base + RNG.normal(size=2000)})
    assert duplicate_pairs(X) == []


def test_a_feature_that_stops_varying_is_caught():
    dates = pd.Series(pd.to_datetime(
        ["2024-01-01"] * 100 + ["2026-01-01"] * 100))
    values = np.concatenate([RNG.normal(size=100), np.zeros(100)])
    X = pd.DataFrame({"stopped": values})
    assert [r["feature"] for r in degenerate_recently(X, dates)] == ["stopped"]


def test_a_feature_still_varying_recently_is_not_flagged():
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 100 + ["2026-01-01"] * 100))
    X = pd.DataFrame({"fine": RNG.normal(size=200)})
    assert degenerate_recently(X, dates) == []


def test_too_few_recent_rows_makes_no_claim():
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 100 + ["2026-01-01"] * 10))
    X = pd.DataFrame({"x": np.concatenate([RNG.normal(size=100), np.zeros(10)])})
    assert degenerate_recently(X, dates) == []


def test_univariate_auc_ranks_a_perfect_feature_first():
    y = np.repeat([0.0, 1.0], 100)
    X = pd.DataFrame({"perfect": y * 10 + RNG.normal(scale=0.01, size=200),
                      "noise": RNG.normal(size=200)})
    ranked = univariate_auc(X, y)
    assert ranked[-1]["feature"] == "perfect"
    assert ranked[-1]["auc"] > 0.99


def test_a_constant_feature_scores_exactly_a_coin_flip():
    y = np.repeat([0.0, 1.0], 100)
    out = univariate_auc(pd.DataFrame({"flat": np.ones(200)}), y)
    assert out[0]["auc"] == 0.5


def test_an_inverted_feature_is_recognised_as_informative():
    """AUC 0.02 is as useful as 0.98; the ranking must not treat it as noise."""
    y = np.repeat([0.0, 1.0], 100)
    X = pd.DataFrame({"inverted": -(y * 10) + RNG.normal(scale=0.01, size=200)})
    assert univariate_auc(X, y)[0]["auc"] < 0.02
