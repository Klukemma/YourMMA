"""Tests for the three betting strategies.

These decide whether a strategy looks profitable, so the arithmetic has to be
exactly right and a fade has to be a genuine opposite rather than a relabelled
copy of the same bet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from strategies import (bootstrap_roi, fade, key_of, running_return,
                        settle_all, strategy_combined, strategy_fade,
                        strategy_model, summarise)


def bet(pick="Alpha", opponent="Beta", won=True, confidence=0.3,
        date="2026-01-24", event="UFC 1"):
    return {"date": pd.Timestamp(date), "event": event, "pick": pick,
            "opponent": opponent, "win_probability": 0.5 + confidence / 2,
            "confidence": confidence, "won": won}


ODDS = {
    ("alpha", "beta"): [(pd.Timestamp("2026-01-24"), -200.0, 170.0)],
    ("beta", "alpha"): [(pd.Timestamp("2026-01-24"), 170.0, -200.0)],
}


# --- fading ---------------------------------------------------------------

def test_fading_swaps_the_fighters_and_the_result():
    original = bet(won=True)
    faded = fade(original)
    assert faded["pick"] == "Beta"
    assert faded["opponent"] == "Alpha"
    assert faded["won"] is False


def test_fading_a_loser_produces_a_winner():
    assert fade(bet(won=False))["won"] is True


def test_fading_is_marked_so_a_report_can_tell_them_apart():
    assert fade(bet())["faded"] is True
    assert strategy_model([bet()])[0]["faded"] is False


def test_fading_twice_returns_to_the_original():
    once = fade(bet(won=True))
    twice = fade(once)
    assert twice["pick"] == "Alpha" and twice["won"] is True


def test_a_fight_keeps_one_identity_whichever_side_is_backed():
    """Otherwise a faded bet would not match its own flag."""
    assert key_of(bet()) == key_of(fade(bet()))


# --- selection ------------------------------------------------------------

def test_the_confidence_floor_excludes_weak_picks():
    bets = [bet(confidence=0.1), bet(confidence=0.5)]
    assert len(strategy_model(bets, min_confidence=0.3)) == 1


def test_fade_backs_only_the_flagged_fights():
    flagged_bet = bet(pick="Alpha")
    other = bet(pick="Gamma", opponent="Delta")
    out = strategy_fade([flagged_bet, other], {key_of(flagged_bet)})
    assert len(out) == 1
    assert out[0]["pick"] == "Beta"


def test_combined_fades_a_flagged_pick_rather_than_skipping_it():
    """A flag says the confidence is misplaced, which is a reason to back the
    other fighter, not a reason to sit out."""
    flagged = bet(confidence=0.9)
    out = strategy_combined([flagged], {key_of(flagged)}, min_confidence=0.2)
    assert len(out) == 1 and out[0]["faded"] is True


def test_combined_backs_an_unflagged_confident_pick():
    out = strategy_combined([bet(confidence=0.9)], set(), min_confidence=0.2)
    assert out[0]["faded"] is False


def test_combined_skips_an_unflagged_weak_pick():
    assert strategy_combined([bet(confidence=0.05)], set(), 0.2) == []


# --- settlement -----------------------------------------------------------

def test_a_winning_favourite_pays_the_right_profit():
    """-200 risks 1 to win 0.5."""
    priced, _ = settle_all([bet(won=True)], ODDS)
    assert priced[0]["profit"] == pytest.approx(0.5)


def test_a_losing_bet_costs_exactly_the_stake():
    priced, _ = settle_all([bet(won=False)], ODDS)
    assert priced[0]["profit"] == pytest.approx(-1.0)


def test_a_faded_bet_is_priced_on_the_other_fighter():
    """Backing Beta at +170 pays 1.7, not the favourite's 0.5."""
    priced, _ = settle_all([fade(bet(won=False))], ODDS)
    assert priced[0]["profit"] == pytest.approx(1.7)


def test_a_bet_with_no_price_is_reported_not_dropped():
    """A strategy that looks good because its losers were unpriced is not a
    strategy."""
    priced, unpriced = settle_all([bet(pick="Nobody", opponent="Nemo")], ODDS)
    assert priced == [] and len(unpriced) == 1


# --- reporting ------------------------------------------------------------

def test_the_summary_arithmetic():
    priced, _ = settle_all([bet(won=True), bet(won=False)], ODDS)
    s = summarise(priced, "x")
    assert s["bets"] == 2 and s["wins"] == 1
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["profit"] == pytest.approx(-0.5)
    assert s["roi"] == pytest.approx(-0.25)


def test_an_empty_strategy_reports_nothing_rather_than_dividing_by_zero():
    s = summarise([], "x")
    assert s["bets"] == 0 and np.isnan(s["roi"])


def test_the_running_return_accumulates_in_date_order():
    late = bet(won=True, date="2026-02-01")
    early = bet(won=False, date="2026-01-24")
    priced, _ = settle_all([late, early], {
        **ODDS,
        ("alpha", "beta"): [(pd.Timestamp("2026-01-24"), -200.0, 170.0),
                            (pd.Timestamp("2026-02-01"), -200.0, 170.0)],
    })
    curve = running_return(priced)
    assert curve[0]["cumulative"] == pytest.approx(-1.0)
    assert curve[-1]["cumulative"] == pytest.approx(-0.5)


def test_the_interval_brackets_the_return():
    priced, _ = settle_all([bet(won=w) for w in [True, False] * 20], ODDS)
    low, high = bootstrap_roi(priced)
    assert low < summarise(priced, "x")["roi"] < high


def test_a_short_run_produces_a_wide_interval():
    """Four bets cannot establish an edge, and the interval should say so."""
    priced, _ = settle_all([bet(won=w) for w in [True, True, True, False]], ODDS)
    low, high = bootstrap_roi(priced)
    assert high - low > 0.3
