"""Rating-derived variables, defined once and testable on their own.

These used to live inside predict_card.py, which takes about ten minutes to
import because importing it runs the whole pipeline. That made every one of
them unverifiable, and it is how a set of constants from an era when the rating
was Elo-like survived the switch to TrueSkill:

    MMR_SCALE = 120.0       squashing a variable whose std is 7.3
    fillna(1500)            on a rating spanning -25..+31
    opp_quality = 1500      for 24.5% of fights, against real values of -3..31
    loss_scale > 1450       a threshold no TrueSkill rating can ever reach, so
                            the adjustment it guarded never once fired
    mmr_gain 10..50         an Elo K-factor applied to a 56-point scale

Every number here is on the TrueSkill scale the ratings actually use:
mu about 13.7-48.5 (mean 28.9, std 5.5), sigma 2.5-8.3, mmr = mu - 3*sigma.
"""

import numpy as np
import pandas as pd

from ratings import DEFAULT as TRUESKILL_CFG, conservative_rating, rate_1v1

TRUESKILL_DEFAULT_MU = 25.0
TRUESKILL_DEFAULT_SIGMA = 25.0 / 3.0

# mu - 3*sigma for a fighter with no rating at all.
TRUESKILL_DEFAULT_MMR = TRUESKILL_DEFAULT_MU - 3.0 * TRUESKILL_DEFAULT_SIGMA

# Logistic scale for base_prob. Matches the observed spread of mmr_diff (std
# 7.3) so the feature covers a usable range; at the old 120.0 it spanned
# 0.448-0.565 with a std of 0.0153, which is a constant with extra steps.
# Choosing a scale from a feature's own spread uses no outcome information.
MMR_SCALE = 7.3

# Observed mu quantiles, for judging how good an opponent was:
# p50 28.60, p75 32.68, p90 36.34.
OPP_MU_ELITE = 36.3
OPP_MU_STRONG = 32.7
OPP_MU_ABOVE_AVG = 28.6

# Recency weighting for strength of schedule: the last four bouts count double.
RECENT_BOUTS = 4
RECENT_WEIGHT = 2.0


def trueskill_post_fight(mu, sigma, opp_mu, opp_sigma, won):
    """Rating after one bout, as (mu, sigma, mmr).

    The previous version applied an Elo K-factor, adding or subtracting 10-50
    points to a rating whose entire range is 56 wide, so one fight could move a
    fighter across most of the scale. ratings.rate_1v1 is the update that
    produced every rating stored in the dataset, so it is what a post-fight
    estimate should use.

    `won` of None covers a draw, no-contest or unknown result and leaves the
    rating where it was rather than inventing a direction for it.
    """
    if won is True:
        new_mu, new_sigma, _, _ = rate_1v1(mu, sigma, opp_mu, opp_sigma,
                                           TRUESKILL_CFG)
    elif won is False:
        _, _, new_mu, new_sigma = rate_1v1(opp_mu, opp_sigma, mu, sigma,
                                           TRUESKILL_CFG)
    else:
        new_mu, new_sigma = mu, sigma
    return new_mu, new_sigma, conservative_rating(new_mu, new_sigma)


def loss_penalty_scale(opp_mu):
    """How much a loss counts against momentum, given who it was against.

    Losing to an elite fighter should sting less than losing to a journeyman.
    The thresholds were 1450/1550/1600 on an Elo scale while the rating is
    TrueSkill mu, so every comparison was false and this returned 1.0 for every
    fight ever scored.
    """
    if pd.isna(opp_mu):
        return 1.0
    if opp_mu > OPP_MU_ELITE:
        return 0.3
    if opp_mu > OPP_MU_STRONG:
        return 0.5
    if opp_mu > OPP_MU_ABOVE_AVG:
        return 0.7
    return 1.0


def weighted_opponent_quality(opponent_mus):
    """Recency-weighted mean skill of opponents faced, oldest first.

    Returns NaN with nothing to average. A fighter with no recorded opponents
    previously got the constant 1500, which made opp_quality_diff a debut flag
    multiplied by about 1486: its std was 628.8 overall against 6.30 among
    fights where both fighters had history.
    """
    if opponent_mus is None or len(opponent_mus) == 0:
        return np.nan
    weights, values = [], []
    for i, opp_mu in enumerate(reversed(list(opponent_mus))):
        weights.append(RECENT_WEIGHT if i < RECENT_BOUTS else 1.0)
        values.append(opp_mu)
    total = sum(weights)
    if total == 0:
        return np.nan
    return sum(w * v for w, v in zip(weights, values)) / total


def standardised_skill_gap(mu_r, sigma_r, mu_b, sigma_b, beta):
    """The skill gap in units of its own uncertainty.

    The TrueSkill win probability is the normal CDF of exactly this. A linear
    model cannot invert that CDF, so the raw value is worth exposing next to it.
    """
    denominator = np.sqrt(np.asarray(sigma_r) ** 2 + np.asarray(sigma_b) ** 2
                          + 2.0 * beta ** 2)
    return (np.asarray(mu_r) - np.asarray(mu_b)) / denominator


def base_probability(mmr_diff, scale=MMR_SCALE):
    """Logistic of the rating difference, on a scale that matches its spread."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(mmr_diff, dtype=float) / scale)))
