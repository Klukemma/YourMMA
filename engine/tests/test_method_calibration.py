"""Calibration must not reorder, and the threshold must not be a knob.

The method model ranks fights well and prices them badly: it said 79% where
64% happened. Isotonic regression fixes the price without touching the order,
which is the property that makes it safe here - anything that reordered would
be a second model wearing the name of a correction.

The threshold is the dangerous half. Moving it down makes the app say
"finish" more often, which feels right and is exactly how a model stops
tracking reality. So it is chosen against balanced accuracy - which no
constant call can game - on data the model did not train on, and it is
allowed to come back 0.5.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from method_calibration import (MIN_CALIBRATION_ROWS, THRESHOLD_RANGE,
                                apply_calibrator, balanced_accuracy,
                                choose_threshold, fit_calibrator,
                                rebuild_three_way)


def overconfident(n=1500, seed=0):
    """Scores that rank well and are stretched outward, like the real thing."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.2, 0.8, n)
    finished = (rng.random(n) < truth).astype(float)
    # Push away from the middle, the way class weighting does.
    stretched = np.clip(0.5 + (truth - 0.5) * 1.9, 0.01, 0.99)
    return stretched, finished, truth


# --- calibration -----------------------------------------------------------

def test_calibration_pulls_the_ends_back_toward_what_happens():
    p, finished, _ = overconfident()
    cal = fit_calibrator(p[:750], finished[:750])
    out = apply_calibrator(cal, p[750:])
    held = finished[750:]

    top = p[750:] > 0.75
    assert p[750:][top].mean() > held[top].mean() + 0.05, "no distortion to fix"
    assert abs(out[top].mean() - held[top].mean()) < \
        abs(p[750:][top].mean() - held[top].mean())


def test_calibration_never_changes_which_fight_is_likelier():
    """Isotonic is monotone by construction. A correction that reordered
    would be a second model, not a correction."""
    p, finished, _ = overconfident()
    cal = fit_calibrator(p, finished)
    grid = np.linspace(0.02, 0.98, 200)
    out = apply_calibrator(cal, grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_calibration_refuses_rather_than_memorising():
    p, finished, _ = overconfident(n=MIN_CALIBRATION_ROWS - 1)
    assert fit_calibrator(p, finished) is None
    # And with nothing to learn from: every fight ended the same way.
    p, _, _ = overconfident(n=500)
    assert fit_calibrator(p, np.ones(500)) is None


def test_no_calibrator_means_the_raw_value_passes_through():
    p = np.array([0.1, 0.5, 0.9])
    assert list(apply_calibrator(None, p)) == list(p)


# --- the threshold ---------------------------------------------------------

def test_balanced_accuracy_cannot_be_gamed_by_a_constant_call():
    """Plain accuracy rewards always-say-the-common-class. This does not."""
    finished = np.array([1.0] * 30 + [0.0] * 70)
    always_decision = np.zeros(100)
    always_finish = np.ones(100)
    assert balanced_accuracy(always_decision, finished, 0.5) == pytest.approx(0.5)
    assert balanced_accuracy(always_finish, finished, 0.5) == pytest.approx(0.5)


def test_a_threshold_is_only_searched_inside_a_band():
    """Outside it the call stops being a call."""
    p = np.linspace(0.01, 0.99, 400)
    finished = (p > 0.9).astype(float)     # would reward a very high cutoff
    t = choose_threshold(p, finished)
    assert THRESHOLD_RANGE[0] <= t <= THRESHOLD_RANGE[1]


def test_a_flat_objective_leaves_the_default_alone():
    """Ties go to the value nearest a half, so rounding noise cannot drift
    the threshold to the edge of the band."""
    rng = np.random.default_rng(1)
    p = rng.random(800)
    finished = rng.integers(0, 2, 800).astype(float)
    t = choose_threshold(p, finished)
    assert abs(t - 0.5) <= 0.06


def test_the_threshold_moves_when_the_data_really_asks_it_to():
    """A guard that never moves is not a guard, it is a constant."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 2000)
    # Finishes happen well below a half, so the honest cutoff is lower.
    finished = (rng.random(2000) < np.clip(p + 0.25, 0, 1)).astype(float)
    assert choose_threshold(p, finished) < 0.5


# --- putting it back together ----------------------------------------------

def test_the_three_way_split_still_sums_to_one():
    proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.5, 0.4]])
    out = rebuild_three_way(proba, np.array([0.3, 0.9]))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[:, 0], [0.7, 0.1])


def test_the_ko_versus_submission_split_is_left_exactly_alone():
    """Calibration was measured on the binary. It says nothing about how a
    finish arrives, so moving that split would invent a correction."""
    proba = np.array([[0.4, 0.42, 0.18]])
    before = proba[0, 1] / (proba[0, 1] + proba[0, 2])
    out = rebuild_three_way(proba, np.array([0.8]))
    after = out[0, 1] / (out[0, 1] + out[0, 2])
    assert after == pytest.approx(before)


def test_a_fight_with_no_finish_mass_splits_evenly_rather_than_dividing_by_zero():
    out = rebuild_three_way(np.array([[1.0, 0.0, 0.0]]), np.array([0.6]))
    assert np.allclose(out, [[0.4, 0.3, 0.3]])
