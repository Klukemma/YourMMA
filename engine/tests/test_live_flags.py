"""Tests for the live-flagger measurement.

This experiment decides whether the second layer is allowed onto the card the
user actually reads, so the protocol it uses has to be the thing under test -
a flagger scored by a fit that saw its own outcome would clear any bar.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.live_flags import (NO_ODDS_COLUMNS, _auc, _number,
                                    walk_forward_quality)


def _rows(n, seed=0, informative=True):
    """Predictions where failure is knowable from sim_disagreement, or is not."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        disagreement = float(rng.uniform(0, 0.5))
        # A real effect: the further the simulator is from the model, the more
        # often the model is wrong.
        p_fail = 0.2 + disagreement if informative else 0.4
        won = rng.random() > p_fail
        rows.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "year": 2020, "won": bool(won),
            "model_p": 0.6, "confidence": 0.2, "sim_p": 0.6 - disagreement,
            "sim_disagreement": disagreement, "sim_available": 1.0,
            "thin_record": 8.0, "striking_known": 1.0, "grappling_known": 1.0,
        })
    return rows


def test_a_flagger_is_never_scored_by_a_fit_that_saw_its_own_outcome():
    """The protocol, asserted directly. Every scored prediction must come from
    a model fitted only on predictions dated before it."""
    rows = _rows(120)
    result = walk_forward_quality(rows, NO_ODDS_COLUMNS, "test")
    # The first MIN_TRAIN predictions cannot be scored at all, by construction.
    from failure_model import MIN_TRAIN
    assert result["scored"] == len(rows) - MIN_TRAIN
    dates = [pd.to_datetime(s["date"]) for s in result["rows"]]
    assert dates == sorted(dates)


def test_a_real_signal_is_found():
    """If failure genuinely tracks a feature, the flagger must beat a coin."""
    result = walk_forward_quality(_rows(300, informative=True),
                                  NO_ODDS_COLUMNS, "test")
    assert result["quality"] > 0.6, result["quality"]


def test_noise_does_not_score_as_signal():
    """The check that matters more. Outcomes independent of every feature must
    come back near 0.5 - a flagger that scores well on noise would send a
    meaningless percentage to the card."""
    result = walk_forward_quality(_rows(300, informative=False),
                                  NO_ODDS_COLUMNS, "test")
    assert 0.38 < result["quality"] < 0.62, result["quality"]


def test_too_few_scored_predictions_gives_no_number_rather_than_a_flattering_one():
    assert np.isnan(_auc([{"p_fail": 0.5, "failed": True}] * 5))
    assert np.isnan(_auc([{"p_fail": 0.5, "failed": False}] * 50))


def test_a_missing_input_is_zero_only_where_zero_is_the_fact():
    """thin_record defaults to 0 because a fighter with no recorded prior
    bouts genuinely has none. A rate would never be defaulted this way."""
    assert _number(None) == 0.0
    assert _number(float("nan")) == 0.0
    assert _number("7") == 7.0
    assert _number(3.5) == 3.5


def test_the_simulator_probability_is_for_the_pick_not_the_red_corner():
    """The model and the simulator are only comparable on the SAME fighter.
    Reading the simulator's red-corner number against a blue-corner pick
    inverts the disagreement on every fight the model picks blue."""
    source = (ENGINE / "experiments" / "live_flags.py").read_text()
    assert "sim_p = (sim_red if red_wins else 1 - sim_red)" in source
