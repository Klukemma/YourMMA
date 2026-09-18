"""Tests for the matchup variables.

Each test is named as the behaviour or the defect it pins, and asserts one
thing. Three families:

  * CONTRACT tests, which fail when a missing value turns into a confident
    number. This is the defect feature_spec.py was rewritten to remove - a
    missing striking accuracy filled with 0 asserts the fighter lands 0% of
    his strikes, which is the worst fighter alive - and it must not come back
    through a new module.

  * SCALE tests, which fail when a unit convention drifts. matchup.py consumes
    stats career_stats.py has not been asked to emit yet, so the contract is
    the largest risk in the design: per-15-minute rates instead of per-minute
    would mis-scale every advantage and NOTHING ELSE WOULD FAIL.

  * CALIBRATION tests, which fail when the model stops agreeing with the
    dataset it was measured on. These read the real CSV and skip without it.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import matchup as mu

UFC_CSV = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


# --- fixtures --------------------------------------------------------------

def league_form(**overrides):
    """A fighter who is exactly league average at everything.

    The reference point for every scale test: feed him to both sides of an
    interaction and each expected_* function must return its own league
    constant, because both ratio-to-league terms are 1 and log5 of the league
    rate against itself is the league rate.
    """
    base = dict(
        sig_att_per_min=mu.LEAGUE_SIG_ATT_PER_MIN,
        sig_att_faced_per_min=mu.LEAGUE_SIG_ATT_PER_MIN,
        sig_accuracy=mu.LEAGUE_SIG_ACC,
        sig_accuracy_conceded=mu.LEAGUE_SIG_ACC,
        td_att_per_min=mu.LEAGUE_TD_ATT_PER_MIN,
        td_att_faced_per_min=mu.LEAGUE_TD_ATT_PER_MIN,
        td_accuracy=mu.LEAGUE_TD_ACC,
        td_accuracy_conceded=mu.LEAGUE_TD_ACC,
        ctrl_sec_per_min=mu.LEAGUE_CTRL_SEC_PER_MIN,
        ctrl_sec_conceded_per_min=mu.LEAGUE_CTRL_SEC_PER_MIN,
        sub_att_per_min=mu.LEAGUE_SUB_ATT_PER_MIN,
        sub_att_conceded_per_min=mu.LEAGUE_SUB_ATT_PER_MIN,
    )
    base.update(overrides)
    return mu.Form(**base)


def league_durability(**overrides):
    base = dict(
        ko_for_per_min=mu.LEAGUE_KO_FOR_PER_MIN,
        ko_against_per_min=mu.LEAGUE_KO_AGAINST_PER_MIN,
        sub_for_per_min=mu.LEAGUE_SUB_FOR_PER_MIN,
        sub_against_per_min=mu.LEAGUE_SUB_AGAINST_PER_MIN,
    )
    base.update(overrides)
    return mu.Durability(**base)


def physique(height=180.0, reach=185.0, weight=70.0):
    return mu.Physique(height_cm=height, reach_cm=reach, weight_kg=weight)


@pytest.fixture(scope="module")
def ufc():
    if not UFC_CSV.exists():
        pytest.skip(f"dataset not present at {UFC_CSV}")
    return pd.read_csv(UFC_CSV, low_memory=False)


# --- the contract: a missing input is never a confident number -------------

def test_a_missing_reach_gives_a_nan_size_advantage_not_a_zero_one():
    """Blue-corner reach is genuinely absent on 7.74% of fights."""
    value = mu.size_advantage(physique(), physique(reach=None))
    assert math.isnan(value)


def test_a_missing_reach_advantage_is_not_reported_as_an_even_matchup():
    """0.0 would assert two identically framed fighters, which is a claim."""
    assert mu.size_advantage(physique(), physique(reach=None)) != 0.0


def test_an_even_matchup_is_zero_and_an_unknown_one_is_nan():
    """The defect feature_spec.paired_known was introduced to fix."""
    even = mu.striking_advantage(league_form(), league_form())
    unknown = mu.striking_advantage(league_form(), mu.Form())
    assert even == 0.0 and math.isnan(unknown)


def test_an_even_matchup_is_known_and_an_unknown_one_is_not():
    even = mu.striking_advantage(league_form(), league_form())
    unknown = mu.striking_advantage(league_form(), mu.Form())
    assert (mu.advantage_known(even), mu.advantage_known(unknown)) == (1.0, 0.0)


def test_size_advantage_known_distinguishes_even_from_unrecorded():
    known = mu.size_advantage_known(physique(), physique())
    unknown = mu.size_advantage_known(physique(), physique(reach=None))
    assert (known, unknown) == (1.0, 0.0)


def test_a_missing_stat_never_becomes_a_maximal_disadvantage():
    """The fillna(0) defect, restated as the property that forbids it.

    A fighter with no recorded striking accuracy must not be described as the
    worst striker who ever competed. Every advantage against him is unknown.
    """
    blank = league_form(sig_accuracy=math.nan)
    assert math.isnan(mu.striking_advantage(league_form(), blank))


def test_a_nan_takedown_accuracy_propagates_rather_than_being_treated_as_zero_percent():
    """663 red corners in this file have no career takedown attempt.

    career_stats.py owes matchup.py a NaN there, not a 0.0. matchup.py does not
    repair it; log5 propagates it, which is the correct answer.
    """
    no_attempts = league_form(td_accuracy=math.nan)
    assert math.isnan(mu.expected_takedowns_per_min(no_attempts, league_form()))


def test_a_missing_field_in_the_mapping_raises_rather_than_defaulting():
    """A silently defaulted stat is indistinguishable from a measured one."""
    with pytest.raises(KeyError, match="sub_att_conceded_per_min"):
        mu.Form.from_mapping({"sig_att_per_min": 7.0})


def test_from_mapping_reads_a_none_as_unknown():
    p = mu.Physique.from_mapping(
        {"height_cm": 180.0, "reach_cm": None, "weight_kg": 70.0})
    assert math.isnan(p.reach_cm)


# --- plausibility gates ----------------------------------------------------

def test_an_impossible_height_becomes_nan_rather_than_the_tallest_fighter_alive():
    """17 fights carry 233.68-381.00 cm, all from the 2025 scrape window.

    Clipping 381 to 215 would assert the fighter is the tallest in UFC history
    (Stefan Struve, 210.82 cm), which is the same class of error as filling a
    missing accuracy with 0.
    """
    assert math.isnan(mu.plausible_height_cm(381.0))


def test_the_tallest_real_fighter_survives_the_height_gate():
    """Stefan Struve at 210.82 cm is a data point, not a defect."""
    assert mu.plausible_height_cm(210.82) == 210.82


def test_the_reach_gate_passes_every_reach_the_dataset_actually_holds():
    """Observed reach range is 147.32-213.36 cm; the gate is a tripwire."""
    assert mu.plausible_reach_cm(147.32) == 147.32
    assert mu.plausible_reach_cm(213.36) == 213.36


def test_heights_and_reaches_are_read_on_the_centimetre_scale():
    """feature_spec filled these with 70, meaning inches, against cm columns."""
    assert math.isnan(mu.plausible_height_cm(70.0))


def test_ape_index_is_reach_minus_height():
    assert mu.ape_index_cm(180.0, 185.0) == pytest.approx(5.0)


def test_an_impossible_height_makes_the_ape_index_unknown():
    assert math.isnan(mu.ape_index_cm(381.0, 185.0))


# --- sign convention -------------------------------------------------------

def test_swapping_the_corners_negates_every_advantage():
    """The property that makes the sign convention checkable, not documented."""
    red_p, blue_p = physique(180.0, 185.0, 70.0), physique(175.0, 178.0, 66.0)
    red_f = league_form(sig_att_per_min=9.0, ctrl_sec_per_min=20.0,
                        td_att_per_min=0.5, sub_att_per_min=0.06)
    blue_f = league_form(sig_accuracy=0.52, ctrl_sec_conceded_per_min=18.0,
                         td_accuracy=0.45, sub_att_conceded_per_min=0.05)
    pairs = [
        (mu.size_advantage(red_p, blue_p), mu.size_advantage(blue_p, red_p)),
        (mu.mass_advantage(red_p, blue_p), mu.mass_advantage(blue_p, red_p)),
        (mu.striking_advantage(red_f, blue_f), mu.striking_advantage(blue_f, red_f)),
        (mu.takedown_advantage(red_f, blue_f), mu.takedown_advantage(blue_f, red_f)),
        (mu.control_advantage(red_f, blue_f), mu.control_advantage(blue_f, red_f)),
        (mu.submission_advantage(red_f, blue_f), mu.submission_advantage(blue_f, red_f)),
        (mu.grappling_advantage(red_f, blue_f), mu.grappling_advantage(blue_f, red_f)),
    ]
    for forward, reversed_ in pairs:
        assert forward == pytest.approx(-reversed_, abs=1e-12)


def test_a_longer_fighter_carries_a_positive_size_advantage():
    assert mu.size_advantage(physique(185.0, 193.0), physique(175.0, 178.0)) > 0


def test_mass_is_not_folded_into_the_size_advantage():
    """r_weight is a division label: constant across 97.5% of careers."""
    light = physique(180.0, 185.0, weight=61.0)
    heavy = physique(180.0, 185.0, weight=93.0)
    assert mu.size_advantage(heavy, light) == 0.0
    assert mu.mass_advantage(heavy, light) > 0


def test_ape_index_is_reported_but_carries_no_weight_in_the_composite():
    """Measured: +0.0092 log-odds per cm alone, -0.0042 alongside reach."""
    assert mu.SIZE_W_APE == 0.0


# --- interaction primitives ------------------------------------------------

def test_two_league_average_fighters_return_the_league_rate_from_log5():
    assert mu.log5(0.4483, 0.4483, 0.4483) == pytest.approx(0.4483, abs=1e-12)


def test_a_league_rate_outside_zero_to_one_raises():
    """A broken constant is a code defect and must fail loudly.

    skill_features.py documents the alternative: an Elo-era threshold survived
    a switch to TrueSkill, so the adjustment it guarded never once fired.
    """
    with pytest.raises(ValueError, match="league_rate"):
        mu.log5(0.5, 0.5, 44.83)


def test_a_perfect_accuracy_does_not_produce_an_infinite_advantage():
    """PROBABILITY_EPS keeps the odds finite on a degenerate input."""
    assert math.isfinite(mu.log5(1.0, 1.0, mu.LEAGUE_SIG_ACC))


def test_a_zero_accuracy_does_not_produce_an_infinite_advantage():
    assert math.isfinite(mu.log5(0.0, 0.0, mu.LEAGUE_SIG_ACC))


def test_a_fighter_with_no_submission_attempts_still_has_a_nonzero_threat():
    """Unclipped, a rate of 0 asserts the fight CANNOT end that way at all."""
    never = league_form(sub_att_per_min=0.0)
    assert mu.expected_sub_attempts_per_min(never, league_form()) > 0.0


def test_the_rate_ratio_clip_never_binds_on_a_well_observed_fighter():
    """Observed shrunk ratios run 0.147 to 4.205 across the eight rate stats."""
    assert mu.RATE_RATIO_FLOOR < 0.147 and mu.RATE_RATIO_CAP > 4.205


def test_an_unknown_rate_is_not_silently_a_league_average_one():
    assert math.isnan(mu.rate_ratio(math.nan, mu.LEAGUE_SIG_ATT_PER_MIN))


# --- scale: the guard against a unit convention drifting -------------------

def test_a_league_average_pair_returns_every_league_constant():
    """The test that fires the moment career_stats emits the wrong unit.

    Per-15-minute rates instead of per-minute would mis-scale every advantage
    in the module and nothing else would fail.
    """
    avg = league_form()
    assert mu.expected_sig_attempts_per_min(avg, avg) == pytest.approx(
        mu.LEAGUE_SIG_ATT_PER_MIN, abs=1e-9)
    assert mu.expected_sig_accuracy(avg, avg) == pytest.approx(
        mu.LEAGUE_SIG_ACC, abs=1e-9)
    assert mu.expected_sig_landed_per_min(avg, avg) == pytest.approx(
        mu.LEAGUE_SIG_LANDED_PER_MIN, abs=1e-9)
    assert mu.expected_control_per_min(avg, avg) == pytest.approx(
        mu.LEAGUE_CTRL_SEC_PER_MIN, abs=1e-9)
    assert mu.expected_sub_attempts_per_min(avg, avg) == pytest.approx(
        mu.LEAGUE_SUB_ATT_PER_MIN, abs=1e-9)
    assert mu.expected_takedowns_per_min(avg, avg) == pytest.approx(
        mu.LEAGUE_TD_ATT_PER_MIN * mu.LEAGUE_TD_ACC, abs=1e-9)


def test_the_league_striking_constants_cannot_drift_apart():
    """LEAGUE_SIG_ATT_PER_MIN is derived, so the three stay consistent."""
    assert mu.LEAGUE_SIG_ATT_PER_MIN * mu.LEAGUE_SIG_ACC == pytest.approx(
        mu.LEAGUE_SIG_LANDED_PER_MIN, abs=1e-12)


def test_accuracies_are_fractions_and_not_percentages():
    """career_stats reads 0.465 where the leaked r_str_acc reads 46.5."""
    assert 0.0 < mu.LEAGUE_SIG_ACC < 1.0 and 0.0 < mu.LEAGUE_TD_ACC < 1.0


# --- the interaction is a matchup, not a difference of main effects --------

def test_the_same_striker_lands_more_against_a_porous_defender():
    """The claim that justifies the whole module.

    A high-volume striker against a porous defender and the same striker
    against an elusive one are different fights, and only an interaction can
    say so: a difference of main effects gives the same answer to both.
    """
    striker = league_form(sig_att_per_min=10.0, sig_accuracy=0.50)
    porous = league_form(sig_accuracy_conceded=0.55)
    elusive = league_form(sig_accuracy_conceded=0.35)
    assert (mu.expected_sig_landed_per_min(striker, porous)
            > mu.expected_sig_landed_per_min(striker, elusive))


def test_a_smothering_defender_cuts_volume_while_an_elusive_one_cuts_accuracy():
    """Pace and accuracy are separated so the caller can tell which it is."""
    striker = league_form(sig_att_per_min=10.0)
    smotherer = league_form(sig_att_faced_per_min=5.0)
    elusive = league_form(sig_accuracy_conceded=0.35)
    assert (mu.expected_sig_attempts_per_min(striker, smotherer)
            < mu.expected_sig_attempts_per_min(striker, elusive))
    assert (mu.expected_sig_accuracy(striker, smotherer)
            > mu.expected_sig_accuracy(striker, elusive))


def test_a_takedown_artist_lands_more_against_poor_takedown_defence():
    wrestler = league_form(td_att_per_min=0.8, td_accuracy=0.50)
    leaky = league_form(td_accuracy_conceded=0.55)
    stout = league_form(td_accuracy_conceded=0.20)
    assert (mu.expected_takedowns_per_min(wrestler, leaky)
            > mu.expected_takedowns_per_min(wrestler, stout))


def test_the_striking_exchange_reports_both_directions():
    red = league_form(sig_att_per_min=10.0)
    exchange = mu.striking_exchange(red, league_form())
    assert exchange.red_landed_per_min > exchange.blue_landed_per_min


def test_the_grappling_exchange_reports_all_six_directional_quantities():
    grappler = league_form(ctrl_sec_per_min=25.0, sub_att_per_min=0.08)
    exchange = mu.grappling_exchange(grappler, league_form())
    assert exchange.red_control_sec_per_min > exchange.blue_control_sec_per_min
    assert exchange.red_sub_attempts_per_min > exchange.blue_sub_attempts_per_min


# --- weighting: arbitrary where it is arbitrary ----------------------------

def test_control_outweighs_submission_which_outweighs_takedowns():
    """Fit gave control +0.233 and submission +0.095 per SD; takedowns -0.009."""
    assert mu.GRAP_W_CTRL > mu.GRAP_W_SUB > mu.GRAP_W_TD > 0.0


def test_the_grappling_weights_sum_to_one():
    assert mu.GRAP_W_CTRL + mu.GRAP_W_SUB + mu.GRAP_W_TD == pytest.approx(1.0)


def test_the_control_to_submission_split_matches_the_measured_ratio():
    """0.233 : 0.095 is 71:29, and 0.60 : 0.25 of the non-takedown weight is 71:29."""
    measured = 0.2325 / (0.2325 + 0.0949)
    chosen = mu.GRAP_W_CTRL / (mu.GRAP_W_CTRL + mu.GRAP_W_SUB)
    assert chosen == pytest.approx(measured, abs=0.01)


def test_the_size_weights_sum_to_one_with_ape_excluded():
    assert mu.SIZE_W_HEIGHT + mu.SIZE_W_REACH + mu.SIZE_W_APE == pytest.approx(1.0)


def test_reach_is_weighted_above_height_as_the_joint_fit_found():
    """Joint logit: reach +0.0120 log-odds per cm against height's +0.0042."""
    assert mu.SIZE_W_REACH > mu.SIZE_W_HEIGHT


def test_the_chosen_height_weight_lies_inside_the_bootstrap_interval():
    """Bootstrap mean 0.207, 90% interval [-0.223, +0.598]. 0.20 is a choice."""
    assert -0.223 <= mu.SIZE_W_HEIGHT <= 0.598


# --- duration repair -------------------------------------------------------

def test_a_round_one_finish_reads_the_same_under_both_time_conventions():
    """Why 'cumulative iff > 300' is exact rather than a heuristic."""
    assert mu.fight_duration_seconds(180.0, 1) == 180.0


def test_a_within_round_time_is_lifted_by_the_rounds_already_fought():
    """A round-3 finish at 2:10 is 12:10 of fight, not 2:10."""
    assert mu.fight_duration_seconds(130.0, 3) == 730.0


def test_a_cumulative_time_from_the_2025_scrape_is_not_inflated_again():
    """101 rows dated 2025-09-13 to 2025-12-06 already hold elapsed time."""
    assert mu.fight_duration_seconds(1500.0, 5) == 1500.0


def test_a_decision_runs_the_full_scheduled_distance():
    assert mu.fight_duration_seconds(300.0, 3) == 900.0


def test_an_unknown_finish_round_gives_an_unknown_duration():
    assert math.isnan(mu.fight_duration_seconds(180.0, math.nan))


# --- schedule validation ---------------------------------------------------

def test_an_uncalibrated_schedule_length_raises_rather_than_guessing():
    """Reusing the 3-round shape for a 2-round bout would invent a constant."""
    with pytest.raises(ValueError, match="calibrated"):
        mu.directional_finish_hazards(league_durability(), league_durability(), 2)


def test_a_float_total_rounds_from_the_csv_is_accepted():
    """total_rounds arrives from the CSV as float64."""
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3.0)
    assert hazards.ko_red_over_blue > 0


def test_a_fractional_number_of_rounds_raises():
    with pytest.raises(ValueError, match="whole number"):
        mu.directional_finish_hazards(league_durability(), league_durability(), 3.5)


# --- the closed form -------------------------------------------------------

def test_every_outcome_of_a_fight_is_accounted_for():
    """p_decision plus the four directional finishes sum to exactly 1."""
    outcome = mu.round_outcome(
        mu.directional_finish_hazards(
            league_durability(ko_for_per_min=0.03),
            league_durability(sub_for_per_min=0.02), 5), 5)
    total = (outcome.p_decision + outcome.p_red_ko + outcome.p_blue_ko
             + outcome.p_red_sub + outcome.p_blue_sub)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_a_fight_with_no_finishing_threat_goes_the_full_distance():
    """The explicit zero-hazard limit, where the closed forms divide by zero."""
    dead = mu.FinishHazards(0.0, 0.0, 0.0, 0.0)
    outcome = mu.round_outcome(dead, 3)
    assert outcome.p_decision == 1.0
    assert outcome.expected_seconds == pytest.approx(900.0)


def test_a_more_dangerous_pair_reaches_the_judges_less_often():
    tame = mu.probability_of_decision(league_durability(),
                                      league_durability(), 3)
    lethal = mu.probability_of_decision(
        league_durability(ko_for_per_min=0.05),
        league_durability(ko_for_per_min=0.05), 3)
    assert lethal < tame


def test_a_more_dangerous_pair_fights_for_less_time():
    tame = mu.expected_fight_seconds(league_durability(), league_durability(), 3)
    lethal = mu.expected_fight_seconds(
        league_durability(ko_for_per_min=0.05),
        league_durability(ko_for_per_min=0.05), 3)
    assert lethal < tame


def test_the_fight_is_the_same_length_whichever_corner_you_stand_in():
    """Duration and the decision probability are symmetric; the causes are not."""
    red = league_durability(ko_for_per_min=0.04, sub_against_per_min=0.02)
    blue = league_durability(sub_for_per_min=0.03)
    assert (mu.expected_fight_seconds(red, blue, 3)
            == pytest.approx(mu.expected_fight_seconds(blue, red, 3), abs=1e-12))


def test_swapping_the_corners_swaps_the_directional_finish_probabilities():
    red = league_durability(ko_for_per_min=0.04)
    blue = league_durability(sub_for_per_min=0.03)
    forward = mu.round_outcome(mu.directional_finish_hazards(red, blue, 3), 3)
    reversed_ = mu.round_outcome(mu.directional_finish_hazards(blue, red, 3), 3)
    assert forward.p_red_ko == pytest.approx(reversed_.p_blue_ko, abs=1e-12)


def test_the_knockout_hazard_falls_faster_than_the_submission_hazard():
    """3-round shapes: KO 1.000/0.746/0.436, submission 1.000/0.908/0.547."""
    assert mu.KO_ROUND_SHAPE[3][2] < mu.SUB_ROUND_SHAPE[3][2]


def test_five_rounders_are_not_harder_to_knock_out_only_harder_to_submit():
    """Round-1 KO hazard 0.038803 vs 0.038808; submission 0.020834 vs 0.010282."""
    assert mu.LAM_KO_R1_PER_MIN[3] == pytest.approx(mu.LAM_KO_R1_PER_MIN[5], abs=1e-5)
    assert mu.LAM_SUB_R1_PER_MIN[5] < 0.6 * mu.LAM_SUB_R1_PER_MIN[3]


def test_a_missing_durability_makes_every_outcome_unknown():
    outcome = mu.round_outcome(
        mu.directional_finish_hazards(league_durability(),
                                      mu.Durability(), 3), 3)
    assert math.isnan(outcome.p_decision)


# --- simulation ------------------------------------------------------------

def test_two_simulations_with_the_same_seed_agree_exactly():
    """An explicit Generator, never global numpy state."""
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3)
    first = mu.simulate_fights(hazards, 3, 20_000, seed=11)
    second = mu.simulate_fights(hazards, 3, 20_000, seed=11)
    assert first == second


def test_two_simulations_with_different_seeds_do_not_agree_exactly():
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3)
    first = mu.simulate_fights(hazards, 3, 20_000, seed=11)
    second = mu.simulate_fights(hazards, 3, 20_000, seed=12)
    assert first.decision_share != second.decision_share


def test_a_single_sampled_fight_is_deterministic_given_its_generator():
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3)
    a = mu.sample_fight_outcome(np.random.default_rng(3), hazards, 3)
    b = mu.sample_fight_outcome(np.random.default_rng(3), hazards, 3)
    assert a == b


def test_a_sampled_decision_has_no_winner_and_runs_the_full_schedule():
    dead = mu.FinishHazards(0.0, 0.0, 0.0, 0.0)
    drawn = mu.sample_fight_outcome(np.random.default_rng(0), dead, 5)
    assert (drawn.winner, drawn.method, drawn.elapsed_seconds) == (None, "DEC", 1500.0)


def test_the_simulator_reproduces_the_closed_form_decision_probability():
    """The test that catches a sampler silently disagreeing with the model."""
    hazards = mu.directional_finish_hazards(
        league_durability(ko_for_per_min=0.025),
        league_durability(sub_for_per_min=0.015), 3)
    exact = mu.round_outcome(hazards, 3).p_decision
    drawn = mu.simulate_fights(hazards, 3, 200_000, seed=5)
    assert abs(drawn.decision_share - exact) < 4 * drawn.decision_standard_error


def test_the_simulator_reproduces_the_closed_form_round_distribution():
    """Aggregate agreement hides a sampler using the wrong round's shape."""
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 5)
    exact = mu.round_outcome(hazards, 5).finish_probability_by_round
    drawn = mu.simulate_fights(hazards, 5, 200_000, seed=6)
    for modelled, simulated in zip(exact, drawn.finish_share_by_round):
        assert abs(simulated - modelled) < 4 * mu.monte_carlo_standard_error(
            modelled, drawn.draws)


def test_the_simulator_reproduces_the_closed_form_expected_duration():
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3)
    exact = mu.round_outcome(hazards, 3).expected_seconds
    drawn = mu.simulate_fights(hazards, 3, 200_000, seed=8)
    assert drawn.mean_seconds == pytest.approx(exact, rel=0.01)


def test_the_simulator_reproduces_the_closed_form_win_split():
    hazards = mu.directional_finish_hazards(
        league_durability(ko_for_per_min=0.04),
        league_durability(), 3)
    outcome = mu.round_outcome(hazards, 3)
    exact = outcome.p_red_ko + outcome.p_red_sub
    drawn = mu.simulate_fights(hazards, 3, 200_000, seed=9)
    assert abs(drawn.red_win_share - exact) < 4 * drawn.red_win_standard_error


def test_simulating_an_unknown_fighter_does_not_return_a_confident_coin_flip():
    """A debutant must not come back as a 50/50 with a standard error."""
    hazards = mu.directional_finish_hazards(
        league_durability(), mu.Durability(), 3)
    drawn = mu.simulate_fights(hazards, 3, 10_000, seed=0)
    assert drawn.draws == 0 and math.isnan(drawn.red_win_share)


def test_a_simulation_of_zero_fights_raises_rather_than_dividing_by_zero():
    hazards = mu.directional_finish_hazards(
        league_durability(), league_durability(), 3)
    with pytest.raises(ValueError, match="draws"):
        mu.simulate_fights(hazards, 3, 0)


def test_a_million_draws_resolve_far_less_than_the_models_own_error():
    """The docstring's claim that more than a million draws buy nothing."""
    assert mu.monte_carlo_standard_error(0.5, 1_000_000) < 0.001


# --- shrinkage helper ------------------------------------------------------

def test_a_fighter_with_no_exposure_is_pulled_all_the_way_to_the_league():
    assert mu.shrunk_rate(0.0, 0.0, mu.LEAGUE_SIG_ACC, mu.K_SIG_ACC_ATT) == \
        pytest.approx(mu.LEAGUE_SIG_ACC)


def test_a_fighter_with_exactly_the_prior_weight_sits_halfway():
    value = mu.shrunk_rate(1.0 * 100.0, 100.0, 0.0, 100.0)
    assert value == pytest.approx(0.5)


def test_takedown_accuracy_is_shrunk_far_harder_than_takedown_defence():
    """Split-half r = +0.071 against +0.237; the K constants must reflect it."""
    assert mu.K_TD_ACC_ATT > 4 * mu.K_TD_DEF_ATT


# --- calibration against the dataset ---------------------------------------

def test_the_time_convention_split_is_exactly_where_the_module_says(ufc):
    """101 cumulative rows, all inside the 2025-09-13..2025-12-06 window."""
    match_time = pd.to_numeric(ufc["match_time_sec"], errors="coerce")
    cumulative = ufc.loc[match_time > mu.ROUND_SECONDS, "date"]
    assert len(cumulative) == 101
    assert cumulative.min() >= "2025-09-13" and cumulative.max() <= "2025-12-06"


def test_no_within_round_time_can_exceed_a_round(ufc):
    """The data-quality assertion the > 300 rule depends on."""
    match_time = pd.to_numeric(ufc["match_time_sec"], errors="coerce")
    within = match_time[match_time <= mu.ROUND_SECONDS]
    assert within.max() == mu.ROUND_SECONDS


def test_the_repaired_duration_never_exceeds_the_scheduled_distance(ufc):
    """2,015 rows violated this before the repair; 0 do now."""
    rows = ufc.head(2000)
    for match_time, finish_round, total in zip(
            rows["match_time_sec"], rows["finish_round"], rows["total_rounds"]):
        elapsed = mu.fight_duration_seconds(match_time, finish_round)
        if math.isfinite(elapsed) and math.isfinite(total):
            assert 0 < elapsed <= total * mu.ROUND_SECONDS


def test_exactly_seventeen_fights_carry_an_impossible_height(ufc):
    """All dated 2025-09-13 to 2025-11-22, all exact inch multiples."""
    red = pd.to_numeric(ufc["r_height"], errors="coerce")
    blue = pd.to_numeric(ufc["b_height"], errors="coerce")
    corrupt = (red > mu.HEIGHT_MAX_CM) | (blue > mu.HEIGHT_MAX_CM)
    assert corrupt.sum() == 17


def test_the_height_gate_rejects_no_fight_outside_that_scrape_window(ufc):
    """The gate must catch the defect and nothing else."""
    red = pd.to_numeric(ufc["r_height"], errors="coerce")
    blue = pd.to_numeric(ufc["b_height"], errors="coerce")
    corrupt = (red > mu.HEIGHT_MAX_CM) | (blue > mu.HEIGHT_MAX_CM)
    assert ufc.loc[corrupt, "date"].min() >= "2025-09-13"


def test_no_reach_in_the_dataset_trips_the_reach_gate(ufc):
    """No corruption is present in reach; the gate is a tripwire only."""
    reach = pd.concat([pd.to_numeric(ufc["r_reach"], errors="coerce"),
                       pd.to_numeric(ufc["b_reach"], errors="coerce")]).dropna()
    assert reach.min() >= mu.REACH_MIN_CM and reach.max() <= mu.REACH_MAX_CM


def test_the_physique_spreads_match_the_dataset(ufc):
    """A re-pull that moves these by 5% makes every standardisation wrong."""
    def gated(column, low, high):
        values = pd.to_numeric(ufc[column], errors="coerce")
        return values.where((values >= low) & (values <= high))
    height = gated("r_height", mu.HEIGHT_MIN_CM, mu.HEIGHT_MAX_CM) - \
        gated("b_height", mu.HEIGHT_MIN_CM, mu.HEIGHT_MAX_CM)
    reach = gated("r_reach", mu.REACH_MIN_CM, mu.REACH_MAX_CM) - \
        gated("b_reach", mu.REACH_MIN_CM, mu.REACH_MAX_CM)
    assert height.std() == pytest.approx(mu.HEIGHT_DIFF_SD_CM, rel=0.05)
    assert reach.std() == pytest.approx(mu.REACH_DIFF_SD_CM, rel=0.05)


def test_the_weight_difference_is_a_mismatch_flag_not_a_size_variable(ufc):
    """Exactly 0 on 66.0% of fights, which is why it is not in size_advantage."""
    diff = (pd.to_numeric(ufc["r_weight"], errors="coerce")
            - pd.to_numeric(ufc["b_weight"], errors="coerce"))
    assert (diff == 0).mean() > 0.6


def test_the_league_striking_rates_match_the_dataset(ufc):
    """Pooled over 17,174 fighter-bouts and 183,687 fighter-minutes."""
    landed = sum(pd.to_numeric(ufc[f"{c}_sig_str_landed"], errors="coerce").sum()
                 for c in ("r", "b"))
    attempted = sum(pd.to_numeric(ufc[f"{c}_sig_str_atmpted"], errors="coerce").sum()
                    for c in ("r", "b"))
    assert landed / attempted == pytest.approx(mu.LEAGUE_SIG_ACC, abs=0.002)


def test_the_league_control_rate_matches_the_dataset(ufc):
    minutes = 2.0 * sum(
        mu.fight_duration_seconds(t, r) for t, r in
        zip(ufc["match_time_sec"], ufc["finish_round"])) / mu.SECONDS_PER_MINUTE
    control = sum(pd.to_numeric(ufc[f"{c}_ctrl"], errors="coerce").sum()
                  for c in ("r", "b"))
    assert control / minutes == pytest.approx(mu.LEAGUE_CTRL_SEC_PER_MIN, rel=0.02)


def test_the_pooled_and_modern_league_accuracies_still_agree(ufc):
    """The eight rate baselines are pooled 2000-2026; the hazards are 2011+.

    The sport changed - the 3-round decision share went 28.8% to 49.9% - so a
    pooled baseline biases the ratio-to-league for old fights. That is
    tolerable only while the pooled and modern accuracies are close. This test
    exists so a future re-pull that moves them apart is noticed.
    """
    modern = ufc[pd.to_datetime(ufc["date"]) >= "2011-01-01"]
    landed = sum(pd.to_numeric(modern[f"{c}_sig_str_landed"], errors="coerce").sum()
                 for c in ("r", "b"))
    attempted = sum(pd.to_numeric(modern[f"{c}_sig_str_atmpted"], errors="coerce").sum()
                    for c in ("r", "b"))
    assert landed / attempted == pytest.approx(mu.LEAGUE_SIG_ACC, abs=0.02)


def test_a_league_average_pair_reproduces_the_observed_decision_share(ufc):
    """Modelled 0.5071 against an observed 0.5071 on 6,539 3-round bouts."""
    modern = ufc[pd.to_datetime(ufc["date"]) >= "2011-01-01"]
    three = modern[pd.to_numeric(modern["total_rounds"], errors="coerce") == 3]
    method = three["method"].astype(str)
    decided = method.str.startswith("Decision") | method.str.endswith("DEC")
    finished = (method.str.startswith("KO/TKO") | method.str.startswith("TKO")
                | method.str.startswith("SUB") | method.eq("Submission"))
    observed = decided.sum() / (decided.sum() + finished.sum())
    modelled = mu.probability_of_decision(league_durability(),
                                          league_durability(), 3)
    assert modelled == pytest.approx(observed, abs=0.005)


def test_the_expected_duration_lands_within_two_percent_of_the_observed_mean(ufc):
    """An INDEPENDENT check: the shapes were fit to the decision share.

    The model under-predicts by 1.1-1.6% because the within-round hazard is not
    flat - finishes cluster late in a round and this model spreads them evenly.
    """
    modern = ufc[pd.to_datetime(ufc["date"]) >= "2011-01-01"]
    three = modern[pd.to_numeric(modern["total_rounds"], errors="coerce") == 3]
    observed = np.mean([mu.fight_duration_seconds(t, r) for t, r in
                        zip(three["match_time_sec"], three["finish_round"])])
    modelled = mu.expected_fight_seconds(league_durability(),
                                         league_durability(), 3)
    assert modelled == pytest.approx(observed, rel=0.02)


def test_the_module_reads_none_of_the_eight_leaking_profile_columns():
    """The reason this module exists, asserted against its own source."""
    source = (ENGINE / "matchup.py").read_text()
    leaked = ("r_splm", "r_str_acc", "r_sapm", "r_str_def", "r_td_avg",
              "r_td_def", "r_td_avg_acc", "r_sub_avg")
    used = [c for c in leaked if f'"{c}"' in source or f"'{c}'" in source]
    assert used == []
