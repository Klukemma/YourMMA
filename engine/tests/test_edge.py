"""Tests for the model-vs-market comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from edge import brier, build_rows, devig, log_loss
from roi import american_to_implied, load_odds


# --- de-vigging -----------------------------------------------------------

def test_devig_removes_the_margin():
    """-150/+130 implies 60% + 43.5% = 103.5%. The pair must sum to 1."""
    a = american_to_implied(-150)
    b = american_to_implied(+130)
    assert a + b > 1.0, "raw implied probabilities include the margin"
    assert devig(a, b) + devig(b, a) == pytest.approx(1.0)


def test_devig_leaves_a_fair_pair_alone():
    assert devig(0.5, 0.5) == pytest.approx(0.5)


def test_devig_keeps_the_favourite_favoured():
    a, b = american_to_implied(-300), american_to_implied(+250)
    assert devig(a, b) > 0.5 > devig(b, a)


def test_devig_on_a_degenerate_pair_is_nan():
    assert np.isnan(devig(0.0, 0.0))


# --- scoring --------------------------------------------------------------

def test_brier_rewards_being_right_and_confident():
    assert brier([0.9], [1.0]) < brier([0.6], [1.0]) < brier([0.1], [1.0])


def test_brier_of_a_coin_flip():
    assert brier([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.25)


def test_log_loss_is_finite_at_the_extremes():
    """A confident miss must not produce inf and poison the average."""
    assert np.isfinite(log_loss([1.0], [0.0]))
    assert np.isfinite(log_loss([0.0], [1.0]))


# --- pairing bets with the market ----------------------------------------

@pytest.fixture
def odds_index(tmp_path):
    path = tmp_path / "odds.csv"
    pd.DataFrame({
        'date': ['2026-02-01'],
        'fighter_a': ['Fav One'], 'fighter_b': ['Dog Two'],
        'odds_a': [-300], 'odds_b': [+250],
    }).to_csv(path, index=False)
    return load_odds(path)


def _bet(pick, opponent, prob, won):
    return {'date': pd.Timestamp('2026-02-01'), 'event': 'E', 'pick': pick,
            'opponent': opponent, 'win_probability': prob, 'won': won,
            'confidence': abs(2 * prob - 1)}


def test_pairs_a_bet_with_the_market_estimate(odds_index):
    df = build_rows([_bet('Fav One', 'Dog Two', 0.8, True)], odds_index)
    assert len(df) == 1
    row = df.iloc[0]
    assert row['model'] == 0.8
    assert 0.5 < row['market'] < 1.0
    assert row['is_favourite']
    assert row['disagreement'] == pytest.approx(0.8 - row['market'])


def test_underdog_side_is_priced_from_its_own_line(odds_index):
    df = build_rows([_bet('Dog Two', 'Fav One', 0.6, False)], odds_index)
    row = df.iloc[0]
    assert row['odds'] == 250
    assert row['market'] < 0.5
    assert not row['is_favourite']
    assert row['profit'] == -1.0


def test_a_fight_priced_on_only_one_side_is_dropped(tmp_path):
    """Comparing against the market needs both prices, not one."""
    path = tmp_path / "odds.csv"
    pd.DataFrame({'date': ['2026-02-01'], 'fighter_a': ['Solo'],
                  'fighter_b': ['Ghost'], 'odds_a': [-200],
                  'odds_b': [np.nan]}).to_csv(path, index=False)
    idx = load_odds(path)
    assert build_rows([_bet('Solo', 'Ghost', 0.7, True)], idx).empty


def test_winning_underdog_profit_matches_the_line(odds_index):
    df = build_rows([_bet('Dog Two', 'Fav One', 0.6, True)], odds_index)
    assert df.iloc[0]['profit'] == pytest.approx(2.5)
