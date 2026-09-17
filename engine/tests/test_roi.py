"""Tests for flat-stake ROI and break-even analysis."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from roi import (american_to_implied, american_to_profit, build_bets, find_odds,
                 load_odds, probability_to_american, settle)


# --- odds arithmetic ------------------------------------------------------

@pytest.mark.parametrize("odds,profit", [
    (100, 1.0), (-100, 1.0), (+200, 2.0), (-200, 0.5),
    (+130, 1.30), (-150, 2/3),
])
def test_profit_on_a_winning_unit(odds, profit):
    assert american_to_profit(odds) == pytest.approx(profit, abs=1e-6)


@pytest.mark.parametrize("odds,implied", [
    (+100, 0.5), (-100, 0.5), (-200, 2/3), (+200, 1/3), (-400, 0.8),
])
def test_implied_probability(odds, implied):
    assert american_to_implied(odds) == pytest.approx(implied, abs=1e-6)


def test_zero_odds_rejected():
    with pytest.raises(ValueError):
        american_to_profit(0)


@pytest.mark.parametrize("p,line", [(0.5, -100), (0.8, -400), (0.2, 400), (0.75, -300)])
def test_fair_line_for_a_probability(p, line):
    assert probability_to_american(p) == line


def test_break_even_line_round_trips():
    """The fair line for p must imply p back."""
    for p in (0.55, 0.6, 0.75, 0.9):
        assert american_to_implied(probability_to_american(p)) == pytest.approx(p, abs=1e-3)


# --- bet construction -----------------------------------------------------

def _graded(date, red, blue, pick, prob, correct):
    return {'actual_date': pd.Timestamp(date), 'event_name': 'E',
            'red_corner': red, 'blue_corner': blue, 'predicted_winner': pick,
            'win_probability': prob, 'correct': correct}


def test_only_counts_from_the_start_date():
    rows = [_graded('2025-11-01', 'A', 'B', 'A', 0.7, True),
            _graded('2026-02-01', 'C', 'D', 'C', 0.7, True)]
    bets = build_bets(rows, pd.Timestamp('2026-01-01'))
    assert len(bets) == 1 and bets[0]['pick'] == 'C'


def test_opponent_is_the_other_corner_either_way():
    rows = [_graded('2026-02-01', 'A', 'B', 'A', 0.7, True),
            _graded('2026-02-02', 'C', 'D', 'D', 0.7, True)]
    bets = build_bets(rows, pd.Timestamp('2026-01-01'))
    assert bets[0]['opponent'] == 'B'
    assert bets[1]['opponent'] == 'C'


def test_confidence_filter():
    rows = [_graded('2026-02-01', 'A', 'B', 'A', 0.55, True),   # confidence 0.10
            _graded('2026-02-02', 'C', 'D', 'C', 0.80, True)]   # confidence 0.60
    assert len(build_bets(rows, pd.Timestamp('2026-01-01'), min_confidence=0.4)) == 1


def test_bets_are_ordered_by_date():
    rows = [_graded('2026-05-01', 'A', 'B', 'A', 0.7, True),
            _graded('2026-02-01', 'C', 'D', 'C', 0.7, True)]
    bets = build_bets(rows, pd.Timestamp('2026-01-01'))
    assert [b['pick'] for b in bets] == ['C', 'A']


# --- settlement -----------------------------------------------------------

@pytest.fixture
def odds_file(tmp_path):
    path = tmp_path / "odds.csv"
    pd.DataFrame({
        'date': ['2026-02-01', '2026-02-02'],
        'fighter_a': ['A One', 'C Three'], 'fighter_b': ['B Two', 'D Four'],
        'odds_a': [-200, +150], 'odds_b': [+170, -180],
    }).to_csv(path, index=False)
    return load_odds(path)


def test_winning_favourite_pays_less_than_the_stake(odds_file):
    bets = build_bets([_graded('2026-02-01', 'A One', 'B Two', 'A One', 0.7, True)],
                      pd.Timestamp('2026-01-01'))
    priced, unpriced = settle(bets, odds_file)
    assert unpriced == []
    assert priced[0]['profit'] == pytest.approx(0.5)      # -200


def test_a_loss_costs_exactly_one_unit(odds_file):
    bets = build_bets([_graded('2026-02-01', 'A One', 'B Two', 'A One', 0.7, False)],
                      pd.Timestamp('2026-01-01'))
    priced, _ = settle(bets, odds_file)
    assert priced[0]['profit'] == -1.0


def test_odds_match_regardless_of_corner_order(odds_file):
    """The pick may be listed as fighter_b in the odds file."""
    bets = build_bets([_graded('2026-02-01', 'B Two', 'A One', 'B Two', 0.6, True)],
                      pd.Timestamp('2026-01-01'))
    priced, _ = settle(bets, odds_file)
    assert priced[0]['odds'] == 170                        # B Two's price


def test_a_pick_with_no_price_is_excluded_not_assumed(odds_file):
    bets = build_bets([_graded('2026-02-01', 'X', 'Y', 'X', 0.7, True)],
                      pd.Timestamp('2026-01-01'))
    priced, unpriced = settle(bets, odds_file)
    assert priced == [] and len(unpriced) == 1


def test_no_odds_index_prices_nothing():
    bets = build_bets([_graded('2026-02-01', 'A', 'B', 'A', 0.7, True)],
                      pd.Timestamp('2026-01-01'))
    priced, unpriced = settle(bets, None)
    assert priced == [] and len(unpriced) == 1


def test_a_break_even_strategy_returns_zero():
    """Win at +100 as often as you lose and the ROI must be exactly 0."""
    from roi import american_to_profit
    results = [True, False, True, False]
    profit = sum(american_to_profit(100) if w else -1.0 for w in results)
    assert profit == pytest.approx(0.0)
