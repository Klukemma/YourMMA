"""Tests for the rating-derived variables.

These constants were wrong for a long time because they lived inside a file
that takes ten minutes to import, so nothing could assert on them. The point of
these tests is that each one now states the scale it belongs to, and fails if a
value from a different scale is put back.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import skill_features as sf


# --- scale sanity: the guard against another Elo-era constant --------------

def test_constants_sit_on_the_trueskill_scale():
    """Observed ratings: mu 13.7-48.5, sigma 2.5-8.3, mmr -25..+31."""
    assert 0 < sf.OPP_MU_ABOVE_AVG < sf.OPP_MU_STRONG < sf.OPP_MU_ELITE < 50
    assert -30 < sf.TRUESKILL_DEFAULT_MMR < 30
    assert 1 < sf.MMR_SCALE < 30, "a scale near 120 belongs to an Elo rating"


def test_base_prob_covers_a_usable_range_on_real_spread():
    """At MMR_SCALE=120 this spanned 0.448-0.565, std 0.0153 - near constant."""
    diffs = np.random.default_rng(0).normal(0, 7.3, 5000)
    p = sf.base_probability(diffs)
    assert p.std() > 0.15, f"base_prob is still nearly constant (std {p.std():.4f})"
    assert p.min() < 0.15 and p.max() > 0.85


def test_base_probability_is_centred_and_monotone():
    assert sf.base_probability(0.0) == pytest.approx(0.5)
    assert sf.base_probability(10.0) > sf.base_probability(1.0) > 0.5
    assert sf.base_probability(-10.0) < 0.5


# --- post-fight rating -----------------------------------------------------

def test_a_win_raises_mu_and_a_loss_lowers_it():
    up_mu, _, _ = sf.trueskill_post_fight(25, 8.33, 25, 8.33, True)
    down_mu, _, _ = sf.trueskill_post_fight(25, 8.33, 25, 8.33, False)
    assert up_mu > 25 > down_mu


def test_any_result_reduces_uncertainty():
    for won in (True, False):
        _, sigma, _ = sf.trueskill_post_fight(25, 8.33, 25, 8.33, won)
        assert sigma < 8.33, "observing a fight must not increase uncertainty"


def test_an_unknown_result_leaves_the_rating_alone():
    """A draw or no-contest must not invent a direction."""
    mu, sigma, mmr = sf.trueskill_post_fight(30.0, 4.0, 25.0, 8.0, None)
    assert (mu, sigma) == (30.0, 4.0)
    assert mmr == pytest.approx(30.0 - 3 * 4.0)


def test_the_move_is_proportionate_to_the_scale():
    """The Elo K-factor added 10-50 to a rating whose whole range is 56."""
    before = 25 - 3 * 8.33
    _, _, after = sf.trueskill_post_fight(25, 8.33, 25, 8.33, True)
    assert abs(after - before) < 15, "one fight must not cross most of the scale"


def test_beating_a_stronger_opponent_gains_more():
    weak, _, _ = sf.trueskill_post_fight(25, 5.0, 15.0, 5.0, True)
    strong, _, _ = sf.trueskill_post_fight(25, 5.0, 40.0, 5.0, True)
    assert strong > weak


def test_mmr_is_mu_minus_three_sigma():
    mu, sigma, mmr = sf.trueskill_post_fight(25, 8.33, 25, 8.33, True)
    assert mmr == pytest.approx(mu - 3 * sigma)


# --- loss penalty ----------------------------------------------------------

def test_losing_to_an_elite_hurts_least():
    assert sf.loss_penalty_scale(40.0) < sf.loss_penalty_scale(33.0)
    assert sf.loss_penalty_scale(33.0) < sf.loss_penalty_scale(29.0)
    assert sf.loss_penalty_scale(29.0) < sf.loss_penalty_scale(20.0)


def test_the_penalty_actually_varies_over_real_ratings():
    """With Elo thresholds every real mu returned 1.0 and this never fired."""
    real_mus = [14.0, 20.0, 28.6, 30.0, 33.0, 36.5, 48.0]
    assert len({sf.loss_penalty_scale(m) for m in real_mus}) > 1


def test_an_unknown_opponent_gets_the_full_penalty():
    assert sf.loss_penalty_scale(np.nan) == 1.0


# --- opponent quality ------------------------------------------------------

def test_no_opponents_is_not_a_number():
    """It used to be 1500, on a scale where real values run about -3 to 31."""
    assert np.isnan(sf.weighted_opponent_quality([]))
    assert np.isnan(sf.weighted_opponent_quality(None))


def test_quality_is_the_mean_when_all_bouts_weigh_the_same():
    assert sf.weighted_opponent_quality([20.0, 30.0, 40.0]) == pytest.approx(30.0)


def test_recent_bouts_count_double():
    """Five bouts: the last four weigh 2, the oldest weighs 1."""
    mus = [10.0, 30.0, 30.0, 30.0, 30.0]
    expected = (1 * 10.0 + 2 * 30.0 * 4) / (1 + 2 * 4)
    assert sf.weighted_opponent_quality(mus) == pytest.approx(expected)


def test_quality_stays_on_the_mu_scale():
    """Averaging real opponent mus must land inside the real mu range."""
    q = sf.weighted_opponent_quality([14.0, 28.6, 48.5])
    assert 10 < q < 50


# --- standardised gap ------------------------------------------------------

def test_the_standardised_gap_is_zero_for_equals():
    assert sf.standardised_skill_gap(25, 5, 25, 5, 5.0) == pytest.approx(0.0)


def test_uncertainty_shrinks_the_gap_rather_than_the_skill():
    """Sigma belongs in the denominator: it pulls toward even, not toward worse."""
    certain = sf.standardised_skill_gap(35, 2.5, 25, 2.5, 5.0)
    unsure = sf.standardised_skill_gap(35, 8.3, 25, 8.3, 5.0)
    assert certain > unsure > 0, "more doubt must move the gap toward zero"


def test_the_gap_keeps_its_sign():
    assert sf.standardised_skill_gap(20, 5, 30, 5, 5.0) < 0


def test_it_matches_the_trueskill_probability():
    """bayesian_prob is the normal CDF of exactly this quantity."""
    from scipy.stats import norm

    z = sf.standardised_skill_gap(33.0, 4.0, 27.0, 6.0, 5.0)
    expected = norm.cdf((33.0 - 27.0) / np.sqrt(4.0**2 + 6.0**2 + 2 * 5.0**2))
    assert norm.cdf(z) == pytest.approx(expected)


def test_it_vectorises_over_series():
    z = sf.standardised_skill_gap(
        pd.Series([30.0, 20.0]), pd.Series([5.0, 5.0]),
        pd.Series([25.0, 25.0]), pd.Series([5.0, 5.0]), 5.0)
    assert z[0] > 0 > z[1]
