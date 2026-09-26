"""Tests for the parlay backtest.

A parlay's arithmetic is where optimism hides: multiplying decimal odds is
easy to get subtly wrong, and a single mistake turns a losing product into a
winning one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from parlay import (break_even_hit_rate, build_parlays, decimal_odds,
                    settle_parlays, summarise_parlays)


def leg(won=True, odds=-200.0, confidence=0.3, event="UFC 1",
        date="2026-01-24", pick="Alpha"):
    return {"date": pd.Timestamp(date), "event": event, "pick": pick,
            "opponent": "Beta", "confidence": confidence, "won": won,
            "odds": odds, "profit": 0.0}


# --- odds arithmetic -------------------------------------------------------

def test_a_favourite_converts_to_a_decimal_below_two():
    assert decimal_odds(-200.0) == pytest.approx(1.5)


def test_an_underdog_converts_to_a_decimal_above_two():
    assert decimal_odds(170.0) == pytest.approx(2.7)


def test_an_even_money_price_is_two():
    assert decimal_odds(100.0) == pytest.approx(2.0)


# --- building --------------------------------------------------------------

def test_a_parlay_takes_the_most_confident_legs():
    legs = [leg(confidence=c, pick=f"F{c}") for c in (0.1, 0.9, 0.5)]
    built = build_parlays(legs, legs=2)
    assert [l["confidence"] for l in built[0]["legs"]] == [0.9, 0.5]


def test_parlays_do_not_cross_events():
    legs = [leg(event="UFC 1"), leg(event="UFC 2")]
    assert build_parlays(legs, legs=2) == []


def test_a_card_too_small_for_the_parlay_is_skipped():
    assert build_parlays([leg()], legs=3) == []


def test_each_card_contributes_one_parlay_by_default():
    legs = [leg(confidence=c, pick=f"F{c}") for c in (0.1, 0.2, 0.3, 0.4)]
    assert len(build_parlays(legs, legs=2)) == 1


# --- settlement ------------------------------------------------------------

def test_every_leg_must_win():
    parlay = {"event": "UFC 1", "date": "2026-01-24", "n_legs": 2,
              "legs": [leg(won=True), leg(won=False)]}
    settled = settle_parlays([parlay])[0]
    assert settled["won"] is False
    assert settled["profit"] == pytest.approx(-1.0)


def test_a_winning_parlay_multiplies_the_legs():
    """1.5 x 2.7 = 4.05, so a unit returns 3.05 profit."""
    parlay = {"event": "UFC 1", "date": "2026-01-24", "n_legs": 2,
              "legs": [leg(won=True, odds=-200.0), leg(won=True, odds=170.0)]}
    settled = settle_parlays([parlay])[0]
    assert settled["multiplier"] == pytest.approx(4.05)
    assert settled["profit"] == pytest.approx(3.05)


def test_a_losing_parlay_costs_one_unit_however_many_legs():
    for n in (2, 3, 4):
        parlay = {"event": "UFC 1", "date": "2026-01-24", "n_legs": n,
                  "legs": [leg(won=False)] * n}
        assert settle_parlays([parlay])[0]["profit"] == pytest.approx(-1.0)


# --- reporting -------------------------------------------------------------

def test_the_break_even_rate_is_the_inverse_of_the_payout():
    """A parlay paying 4.05 needs to land about a quarter of the time."""
    settled = settle_parlays([{
        "event": "UFC 1", "date": "2026-01-24", "n_legs": 2,
        "legs": [leg(won=True, odds=-200.0), leg(won=True, odds=170.0)]}])
    assert break_even_hit_rate(settled) == pytest.approx(1 / 4.05)


def test_the_summary_arithmetic():
    settled = settle_parlays([
        {"event": "A", "date": "2026-01-24", "n_legs": 2,
         "legs": [leg(won=True, odds=100.0), leg(won=True, odds=100.0)]},
        {"event": "B", "date": "2026-01-25", "n_legs": 2,
         "legs": [leg(won=False), leg(won=True)]},
    ])
    s = summarise_parlays(settled, "2 legs")
    assert s["parlays"] == 2 and s["wins"] == 1
    assert s["profit"] == pytest.approx(3.0 - 1.0)
    assert s["roi"] == pytest.approx(1.0)


def test_no_parlays_reports_nothing_rather_than_dividing_by_zero():
    s = summarise_parlays([], "2 legs")
    assert s["parlays"] == 0 and np.isnan(s["roi"])
