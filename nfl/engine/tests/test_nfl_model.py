"""The walk-forward: a season's predictions come only from seasons before it."""

import numpy as np
import pandas as pd
import pytest

import nfl_backtest as bt
import nfl_model as nm


@pytest.fixture(scope="module")
def frame(built):
    f = built[0].copy()
    f["market_prob"] = bt.market_prob(f)
    return f


def test_a_season_is_predicted_the_same_without_the_seasons_after_it(frame):
    full = nm.walk_forward(frame, start=2012, end=2014)
    cut = nm.walk_forward(frame[frame["season"] <= 2013], start=2012, end=2013)
    rows = frame.index[frame["season"] == 2013]
    pd.testing.assert_frame_equal(full.loc[rows].astype(float), cut.loc[rows].astype(float))


def test_the_periods_do_not_overlap_and_train_before_both():
    assert bt.DEV[1] < bt.CONFIRM[0]
    assert bt.TRAIN_FROM < bt.DEV[0]


def test_probabilities_are_probabilities(frame):
    preds = nm.walk_forward(frame, start=2020, end=2021)
    sel = frame["season"].between(2020, 2021)
    for col in ("model_prob", "blend_prob"):
        p = preds.loc[sel, col].astype(float)
        assert p.notna().all() and ((p > 0) & (p < 1)).all()


def test_the_blend_falls_back_to_the_model_without_a_price():
    model_p = np.array([0.7, 0.3])
    assert np.allclose(nm.apply_blend(None, model_p, np.array([0.6, np.nan])), model_p)


def test_tiers_cover_every_probability():
    for p in (0.5, 0.549, 0.55, 0.649, 0.65, 0.749, 0.75, 0.99):
        assert bt.tier_of(p) in {name for name, _ in bt.TIERS}
    assert bt.tier_of(0.76) == "STRONG" and bt.tier_of(0.52) == "TOSS-UP"


def test_market_probability_removes_the_margin():
    f = pd.DataFrame({"home_ml": [-110.0], "away_ml": [-110.0], "spread_line": [0.0]})
    assert bt.market_prob(f).iloc[0] == pytest.approx(0.5)


def test_a_value_label_needs_profit_in_both_periods(frame):
    preds = nm.walk_forward(frame, start=bt.DEV[0])
    gate = bt.value_gate(frame, preds["model_prob"].astype(float))
    if gate["enabled_at"] is not None:
        per = gate["edges"][str(gate["enabled_at"])]
        assert all(r["roi"] > 0 and r["bets"] >= bt.MIN_GATE_BETS for r in per.values())
