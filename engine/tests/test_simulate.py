"""Tests for the Monte Carlo fight simulator.

These pin three separate classes of defect, in descending order of how quietly
they would do damage.

1. THE MISSING-DATA INVERSION. A rate that nobody measured must never become a
   number that means something else. A zero SLpM is a fighter who never throws,
   a zero striking defence is a fighter hit by everything, and a zero knockdown
   rate is a fighter who cannot hurt anyone - all three are the worst fighter
   alive rather than an unknown one. This is the defect engine/feature_spec.py
   was rewritten to remove, and it is the easiest one to reintroduce here,
   because five of the thirteen rates CANNOT come from career_stats.py at all.

2. DOUBLE-COUNTING THE AVERAGE OPPONENT. Every fighter statistic was already
   measured against average opposition, so composing two of them naively
   charges the average opponent's defence twice. The ratio form is pinned by
   asserting that two league-average fighters reproduce the league value
   exactly - a composition error would not raise, it would just move the
   second decimal place, which is the range that decides a bet.

3. SILENT ASYMMETRY AND SILENT CONFLATION. A fixed A-then-B tie-break would
   hand the red corner a systematic edge, in a repo already fighting a 61.5%
   red-corner rate. A decision counted as a final-round finish would misprice a
   different market entirely.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import simulate as sim
from simulate import (
    FighterRates,
    calibrate_judge_noise,
    judge_verdict,
    knockdown_probability,
    league_rates,
    rates_from_career_stats,
    resolve_rates,
    round_score_margin,
    simulate_card,
    simulate_fight,
    strike_land_rate,
    submission_success_probability,
    takedown_success_probability,
    validate_rates,
)

# Small enough to run fast, large enough that the assertions below are not
# measuring Monte Carlo noise. At 20,000 the standard error on a win
# probability is 0.35 percentage points.
FAST_SIMS = 20_000


def rates(name="X", **overrides):
    """A league-average fighter with named fields overridden."""
    base = league_rates(name)
    return type(base)(**{**base.__dict__, **overrides})


# --- the missing-data policy ----------------------------------------------

def test_a_missing_rate_never_becomes_zero():
    """The inversion this whole repo exists to prevent, in its own words."""
    unknown = rates("Debutant", slpm=float("nan"))
    filled, _ = resolve_rates(unknown, defaults=league_rates())
    assert filled.slpm == pytest.approx(sim.LEAGUE_SLPM)


def test_a_missing_rate_with_no_default_raises_rather_than_guessing():
    unknown = rates("Debutant", str_def=float("nan"))
    with pytest.raises(ValueError, match="str_def"):
        resolve_rates(unknown, defaults=None)


def test_the_imputed_field_is_named_so_the_caller_can_see_it():
    unknown = rates("Debutant", td_def=float("nan"))
    _, imputed = resolve_rates(unknown, defaults=league_rates())
    assert imputed == frozenset({"td_def"})


def test_a_fully_measured_fighter_imputes_nothing():
    _, imputed = resolve_rates(league_rates(), defaults=league_rates())
    assert imputed == frozenset()


def test_the_distribution_reports_which_rates_were_imputed():
    """A caller must be able to tell an assumed durability from a measured one."""
    guessy = rates("A", ko_loss_per_kd_absorbed=float("nan"))
    out = simulate_fight(guessy, rates("B"), n_sims=FAST_SIMS, rng=0,
                         defaults=league_rates())
    assert out.imputed_a == frozenset({"ko_loss_per_kd_absorbed"})


def test_career_stats_cannot_supply_the_outcome_derived_rates():
    """career_stats reads no outcome column, so five rates must come back NaN.

    If this ever passes with fewer than five, career_stats has started reading
    `method` or `winner` and the leak audit needs re-running.
    """
    row = {"r_cd_slpm": 4.0, "r_cd_str_acc": 0.5, "r_cd_sapm": 3.0,
           "r_cd_str_def": 0.6, "r_cd_td_per15": 2.0, "r_cd_td_acc": 0.4,
           "r_cd_td_def": 0.7, "r_cd_sub_per15": 0.8, "r_cd_kd_per15": 0.3,
           "r_cd_bouts": 9, "r_name": "Someone"}
    got = rates_from_career_stats(row, "r")
    missing = {f for f in sim._REQUIRED_RATE_FIELDS
               if getattr(got, f) != getattr(got, f)}
    assert missing == {
        "sub_success_per_att", "kd_abs_per_str_absorbed",
        "ko_loss_per_kd_absorbed", "tko_loss_per_head_absorbed",
        "sub_loss_per_sub_faced",
    }


def test_takedown_attempts_are_reconstructed_from_landed_and_accuracy():
    """career_stats emits takedowns LANDED per 15; the simulator shoots attempts.

    2.0 landed per 15 at 40% accuracy is 5.0 attempts per 15. Reading
    cd_td_per15 as an attempt rate would understate every wrestler's volume by
    the reciprocal of his accuracy.
    """
    row = {"r_cd_td_per15": 2.0, "r_cd_td_acc": 0.4, "r_cd_slpm": 4.0}
    assert rates_from_career_stats(row, "r").td_att_per15 == pytest.approx(5.0)


def test_a_zero_takedown_accuracy_leaves_the_attempt_rate_unknown():
    """0 landed at 0% accuracy is 0/0, which is unknown, not zero attempts."""
    row = {"r_cd_td_per15": 0.0, "r_cd_td_acc": 0.0, "r_cd_slpm": 4.0}
    got = rates_from_career_stats(row, "r")
    assert got.td_att_per15 != got.td_att_per15


def test_knockdown_rate_per_strike_is_reconstructed_from_the_per_15_rate():
    """cd_kd_per15 / (cd_slpm * 15): 0.6 knockdowns per 15 at 4.0 SLpM is
    0.6/60 = 0.01 per landed strike."""
    row = {"r_cd_kd_per15": 0.6, "r_cd_slpm": 4.0}
    assert rates_from_career_stats(row, "r").kd_per_str_landed == pytest.approx(0.01)


# --- input validation -------------------------------------------------------

def test_a_defence_of_exactly_one_is_refused():
    """A one-bout artefact that would simulate an untouchable fighter."""
    with pytest.raises(ValueError, match="invulnerable"):
        validate_rates(rates("Perfect", td_def=1.0))


def test_a_share_above_one_is_refused():
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_rates(rates("Impossible", str_acc=1.4))


def test_a_negative_rate_is_refused():
    with pytest.raises(ValueError, match="negative"):
        validate_rates(rates("Impossible", slpm=-1.0))


def test_a_merely_extreme_fighter_is_allowed():
    """0.85 takedown defence and 6.5 SLpM are real fighters, not errors."""
    validate_rates(rates("Elite", td_def=0.85, slpm=6.5))


# --- composition: the league identity ---------------------------------------

def test_two_league_fighters_strike_at_the_league_rate_in_the_time_weighted_mean():
    """The normaliser's whole job: the three positions must average to SLpM.

    Without it the simulator loses 2.3% of all striking output, because the
    positional regression's three means do not reproduce the pooled total.
    """
    lg = league_rates()
    mean = (
        sim.NEUTRAL_TIME_SHARE
        * strike_land_rate(lg.slpm, lg.str_def, sim.POSITION_STRIKE_MULT["neutral"])
        + sim.OWN_CONTROL_TIME_SHARE
        * strike_land_rate(lg.slpm, lg.str_def, sim.POSITION_STRIKE_MULT["top"])
        + sim.OWN_CONTROL_TIME_SHARE
        * strike_land_rate(lg.slpm, lg.str_def, sim.POSITION_STRIKE_MULT["bottom"])
    )
    assert float(mean) == pytest.approx(sim.LEAGUE_SLPM, rel=1e-9)


def test_a_better_defender_is_hit_less_than_an_average_one():
    lg = league_rates()
    average = strike_land_rate(lg.slpm, sim.STR_DEF_LEAGUE, 1.0)
    good = strike_land_rate(lg.slpm, 0.65, 1.0)
    assert float(good) < float(average)


def test_the_defence_ratio_does_not_double_count_the_average_opponent():
    """An average defender must leave the attacker's own SLpM untouched.

    The naive composition slpm * (1 - opp_str_def) would return 1.58 here
    rather than 3.53, because the attacker's SLpM was already measured against
    opponents defending at 55.17%.
    """
    lg = league_rates()
    got = strike_land_rate(lg.slpm, sim.STR_DEF_LEAGUE, sim.POSITION_MULT_NORMALISER)
    assert float(got) == pytest.approx(sim.LEAGUE_SLPM, rel=1e-9)


def test_two_league_fighters_knock_down_at_the_league_rate():
    assert float(knockdown_probability(sim.KD_PER_LANDED_LEAGUE,
                                       sim.KD_PER_LANDED_LEAGUE)) == pytest.approx(
        sim.KD_PER_LANDED_LEAGUE)


def test_two_league_fighters_take_each_other_down_at_the_league_rate():
    assert float(takedown_success_probability(sim.TD_ACC_LEAGUE,
                                              sim.TD_DEF_LEAGUE)) == pytest.approx(
        sim.TD_ACC_LEAGUE)


def test_two_league_fighters_submit_at_the_league_rate():
    assert float(submission_success_probability(sim.SUB_SUCCESS_LEAGUE,
                                                sim.SUB_SUCCESS_LEAGUE)) == pytest.approx(
        sim.SUB_SUCCESS_LEAGUE)


def test_a_composed_probability_is_clipped_to_a_probability():
    """An elite wrestler against a fighter with no takedown defence at all."""
    assert float(takedown_success_probability(0.9, 0.0)) <= 1.0


# --- scoring ----------------------------------------------------------------

def test_an_identical_round_scores_level():
    assert float(round_score_margin(20, 20, 1, 1, 60, 60, 0, 0)) == 0.0


def test_a_knockdown_outweighs_ten_significant_strikes():
    """The fitted weight is 10.12 strikes per knockdown, so it must."""
    assert float(round_score_margin(0, 10, 0, 0, 0, 0, 1, 0)) > 0


def test_the_scoring_weights_are_the_fitted_ones():
    """Pins the four weights so a re-fit cannot land silently.

    One takedown, one control minute and one knockdown, each alone, against
    the logistic fit on 4,000 real decisions.
    """
    assert float(round_score_margin(0, 0, 1, 0, 0, 0, 0, 0)) == pytest.approx(1.9192)
    assert float(round_score_margin(0, 0, 0, 0, 60, 0, 0, 0)) == pytest.approx(2.8804)
    assert float(round_score_margin(0, 0, 0, 0, 0, 0, 1, 0)) == pytest.approx(10.1239)


def test_a_clear_sweep_is_unanimous_with_no_noise():
    margins = np.full((100, 3), 50.0)
    verdict, kind = judge_verdict(margins, np.random.default_rng(0), 0.0)
    assert set(np.unique(verdict)) == {1} and set(np.unique(kind)) == {sim.VERDICT_UNANIMOUS}


def test_a_completely_blank_fight_is_a_draw_and_is_never_assigned():
    """Two fighters who do nothing at all produce level cards, not a coin flip.

    Every round margin is exactly 0, every round is 10-10, and the result must
    be reported as a draw rather than broken toward either corner.
    """
    margins = np.zeros((500, 3))
    verdict, kind = judge_verdict(margins, np.random.default_rng(0), 0.0)
    assert set(np.unique(verdict)) == {0} and set(np.unique(kind)) == {sim.VERDICT_DRAW}


def test_a_majority_decision_still_has_a_winner():
    """2-1 with the third judge level is a majority, and somebody won it.

    A majority and a split are different labels on the real data (2.49%
    against 20.30%) and the simulator must not merge them, nor let a majority
    leak into the draw bucket.
    """
    margins = np.random.default_rng(0).normal(0.0, 9.0, size=(20_000, 3))
    verdict, kind = judge_verdict(margins, np.random.default_rng(1), 7.4)
    majority = kind == sim.VERDICT_MAJORITY
    assert majority.any() and (verdict[majority] != 0).all()


def test_a_draw_is_labelled_a_draw_and_not_a_split():
    margins = np.random.default_rng(0).normal(0.0, 9.0, size=(20_000, 3))
    verdict, kind = judge_verdict(margins, np.random.default_rng(1), 7.4)
    assert set(np.unique(kind[verdict == 0])) == {sim.VERDICT_DRAW}


def test_a_ten_eight_round_is_what_makes_a_level_card_possible():
    """Without 10-8 rounds, majority decisions and draws cannot occur at all.

    With 10-9 rounds only and an odd number of rounds a judge's card is a sum
    of three odd numbers and can never be zero, so the observed 2.49% majority
    and 1.45% draw shares would both come out as exactly 0.00%.
    """
    margins = np.random.default_rng(0).normal(0.0, 9.0, size=(20_000, 3))
    rng = np.random.default_rng(1)
    with_ten_eight = judge_verdict(margins, rng, 7.4)[1]
    without = judge_verdict(margins, np.random.default_rng(1), 7.4,
                            dominant_margin=np.inf)[1]
    assert (without == sim.VERDICT_MAJORITY).sum() == 0
    assert (with_ten_eight == sim.VERDICT_MAJORITY).sum() > 0


def test_more_judge_noise_produces_more_split_decisions():
    """The monotonicity calibrate_judge_noise bisects on."""
    margins = np.random.default_rng(0).normal(0.0, 10.0, size=(5_000, 3))
    quiet = judge_verdict(margins, np.random.default_rng(1), 1.0)[1]
    loud = judge_verdict(margins, np.random.default_rng(1), 12.0)[1]
    assert (loud == sim.VERDICT_SPLIT).mean() > (quiet == sim.VERDICT_SPLIT).mean()


def test_calibrating_judge_noise_hits_the_target_split_share():
    margins = np.random.default_rng(0).normal(0.0, 12.0, size=(4_000, 3))
    sd = calibrate_judge_noise(margins, target_split_share=0.203, rng=0)
    kind = judge_verdict(margins, np.random.default_rng(99), sd)[1]
    assert (kind == sim.VERDICT_SPLIT).mean() == pytest.approx(0.203, abs=0.02)


# --- determinism and independence -------------------------------------------

def test_the_same_seed_gives_the_same_answer():
    a, b = rates("A"), rates("B", slpm=4.4)
    first = simulate_fight(a, b, n_sims=FAST_SIMS, rng=7)
    second = simulate_fight(a, b, n_sims=FAST_SIMS, rng=7)
    assert first.win_prob_a == second.win_prob_a


def test_a_different_seed_gives_a_different_answer():
    a, b = rates("A"), rates("B", slpm=4.4)
    first = simulate_fight(a, b, n_sims=FAST_SIMS, rng=7)
    second = simulate_fight(a, b, n_sims=FAST_SIMS, rng=8)
    assert first.win_prob_a != second.win_prob_a


def test_an_unseeded_run_reports_no_seed():
    """A caller must always be able to tell a reproducible result from one that
    is not."""
    out = simulate_fight(rates("A"), rates("B"), n_sims=FAST_SIMS, rng=None)
    assert out.seed is None


def test_an_int_seed_is_recorded():
    out = simulate_fight(rates("A"), rates("B"), n_sims=FAST_SIMS, rng=3)
    assert out.seed == 3


def test_the_global_random_state_is_never_used():
    """Seeding numpy's legacy global state must not change the answer."""
    a, b = rates("A"), rates("B", slpm=4.4)
    np.random.seed(1)
    first = simulate_fight(a, b, n_sims=FAST_SIMS, rng=5)
    np.random.seed(99999)
    second = simulate_fight(a, b, n_sims=FAST_SIMS, rng=5)
    assert first.win_prob_a == second.win_prob_a


def test_removing_a_bout_from_a_card_does_not_move_the_others():
    """Each bout draws from its own child stream.

    With a shared stream, scratching one fight would silently change every
    prediction below it on the card.
    """
    a, b, c = rates("A"), rates("B", slpm=4.2), rates("C", td_acc=0.5)
    pairs = [(a, b, 3), (b, c, 3), (a, c, 5)]
    full = simulate_card(pairs, n_sims=FAST_SIMS, rng=11)
    trimmed = simulate_card(pairs[:2], n_sims=FAST_SIMS, rng=11)
    assert [d.win_prob_a for d in trimmed] == [d.win_prob_a for d in full[:2]]


# --- symmetry ---------------------------------------------------------------

def test_two_identical_fighters_are_a_coin_flip():
    """A fixed A-then-B tie-break in the same tick would show up here.

    This repo is already fighting a red-corner bias - the red corner wins 61.5%
    of the decisions the judging weights were fitted on - so a simulator that
    quietly favoured corner A would be very hard to spot downstream.
    """
    out = simulate_fight(rates("A"), rates("A2"), n_sims=100_000, rng=4)
    assert out.win_prob_a == pytest.approx(out.win_prob_b, abs=0.006)


def test_swapping_the_corners_mirrors_the_method_probabilities():
    a, b = rates("A", slpm=5.0, kd_per_str_landed=0.009), rates("B", td_acc=0.55)
    forward = simulate_fight(a, b, n_sims=60_000, rng=21)
    reverse = simulate_fight(b, a, n_sims=60_000, rng=21)
    assert forward.ko_prob_a == pytest.approx(reverse.ko_prob_b, abs=0.01)


# --- the outcome distribution -----------------------------------------------

def test_every_outcome_probability_sums_to_one():
    out = simulate_fight(rates("A"), rates("B", slpm=4.1), n_sims=FAST_SIMS, rng=2)
    total = (out.ko_prob_a + out.ko_prob_b + out.sub_prob_a + out.sub_prob_b
             + out.dec_prob_a + out.dec_prob_b + out.draw_prob)
    assert total == pytest.approx(1.0)


def test_win_probabilities_and_the_draw_sum_to_one():
    out = simulate_fight(rates("A"), rates("B", slpm=4.1), n_sims=FAST_SIMS, rng=2)
    assert out.win_prob_a + out.win_prob_b + out.draw_prob == pytest.approx(1.0)


def test_a_decision_is_never_counted_as_a_final_round_finish():
    """finish_round_prob[0] is the decision bucket, deliberately.

    A decision recorded as a round-3 finish would misprice a completely
    different market, and nothing else in the return value would look wrong.
    """
    out = simulate_fight(rates("A"), rates("B"), n_sims=FAST_SIMS, rng=2)
    decisions = out.dec_prob_a + out.dec_prob_b + out.draw_prob
    assert out.finish_round_prob[0] == pytest.approx(decisions)


def test_the_finish_round_distribution_covers_every_round_and_the_decision():
    out = simulate_fight(rates("A"), rates("B"), rounds=5, n_sims=FAST_SIMS, rng=2)
    assert len(out.finish_round_prob) == 6
    assert out.finish_round_prob.sum() == pytest.approx(1.0)


def test_a_five_round_fight_lasts_longer_than_a_three_round_one():
    a, b = rates("A"), rates("B")
    three = simulate_fight(a, b, rounds=3, n_sims=FAST_SIMS, rng=6)
    five = simulate_fight(a, b, rounds=5, n_sims=FAST_SIMS, rng=6)
    assert five.mean_duration_sec > three.mean_duration_sec


def test_a_fight_never_lasts_longer_than_its_rounds_allow():
    out = simulate_fight(rates("A"), rates("B"), rounds=3, n_sims=FAST_SIMS, rng=6)
    assert out.mean_duration_sec <= 3 * sim.ROUND_SECONDS


def test_the_standard_error_shrinks_with_more_simulations():
    a, b = rates("A"), rates("B", slpm=4.1)
    small = simulate_fight(a, b, n_sims=2_000, rng=1)
    large = simulate_fight(a, b, n_sims=50_000, rng=1)
    assert large.win_prob_se_a < small.win_prob_se_a


# --- the rates actually drive the fight -------------------------------------

def test_a_harder_puncher_wins_more_by_knockout():
    weak = rates("Weak", kd_per_str_landed=0.002)
    strong = rates("Strong", kd_per_str_landed=0.012)
    out = simulate_fight(strong, weak, n_sims=60_000, rng=12)
    assert out.ko_prob_a > out.ko_prob_b


def test_a_better_striker_wins_more_often():
    out = simulate_fight(rates("Sharp", slpm=5.2, str_def=0.62),
                         rates("Blunt", slpm=2.6, str_def=0.48),
                         n_sims=60_000, rng=13)
    assert out.win_prob_a > out.win_prob_b


def test_a_better_wrestler_spends_more_time_in_control():
    out = simulate_fight(rates("Wrestler", td_att_per15=8.0, td_acc=0.55, td_def=0.85),
                         rates("Striker", td_att_per15=0.5, td_acc=0.25, td_def=0.45),
                         n_sims=40_000, rng=14)
    assert out.mean_round_tallies["ctrl_sec_a"].sum() > out.mean_round_tallies["ctrl_sec_b"].sum()


def test_a_grappler_who_cannot_get_on_top_cannot_submit_from_the_top():
    """Submission threat must flow through position, not straight from the rate.

    A dangerous submission fighter who is never taken down and never takes
    anyone down should submit far less than the same fighter who lands
    takedowns, even though his sub_att_per15 is identical.
    """
    threat = dict(sub_att_per15=3.0, sub_success_per_att=0.5)
    grounded = simulate_fight(rates("Grappler", td_att_per15=9.0, td_acc=0.6, **threat),
                              rates("B", td_def=0.3), n_sims=40_000, rng=15)
    stranded = simulate_fight(rates("Grappler", td_att_per15=0.1, td_acc=0.1, **threat),
                              rates("B", td_def=0.95, td_att_per15=0.1),
                              n_sims=40_000, rng=15)
    assert grounded.sub_prob_a > stranded.sub_prob_a


def test_a_fighter_with_a_better_chin_is_knocked_out_less():
    fragile = rates("Glass", ko_loss_per_kd_absorbed=0.9,
                    tko_loss_per_head_absorbed=0.006)
    durable = rates("Granite", ko_loss_per_kd_absorbed=0.15,
                    tko_loss_per_head_absorbed=0.0005)
    against_fragile = simulate_fight(rates("Puncher"), fragile, n_sims=40_000, rng=16)
    against_durable = simulate_fight(rates("Puncher"), durable, n_sims=40_000, rng=16)
    assert against_fragile.ko_prob_a > against_durable.ko_prob_a


def test_accumulated_damage_carries_across_rounds():
    """The accumulation TKO is the only path to 30.1% of real KO/TKO wins.

    A fighter whose only knockout mechanism is accumulation must still finish
    fights, and must finish more of them in a five-round fight than a
    three-round one, because damage does not reset between rounds.
    """
    grinder = rates("Grinder", kd_per_str_landed=0.0)
    fragile = rates("Fragile", tko_loss_per_head_absorbed=0.004,
                    kd_per_str_landed=0.0)
    three = simulate_fight(grinder, fragile, rounds=3, n_sims=40_000, rng=17)
    five = simulate_fight(grinder, fragile, rounds=5, n_sims=40_000, rng=17)
    assert 0 < three.ko_prob_a < five.ko_prob_a


def test_a_fighter_who_scores_no_knockdowns_can_still_win_by_ko():
    """838 of 2,787 real KO/TKO losses had zero recorded knockdowns."""
    out = simulate_fight(rates("A", kd_per_str_landed=0.0),
                         rates("B", kd_per_str_landed=0.0),
                         n_sims=40_000, rng=18)
    assert out.ko_prob_a > 0


# --- tick invariance --------------------------------------------------------

def test_the_win_probability_does_not_depend_on_the_tick_size():
    """Counts are Poisson(lambda*dt) and binary events 1-exp(-lambda*dt).

    Using lambda*dt as a probability would lose 18% of events at the 5s default
    and the answer would move with the tick.
    """
    a, b = rates("A", slpm=4.8), rates("B", slpm=3.1, str_def=0.51)
    coarse = simulate_fight(a, b, n_sims=60_000, rng=19, tick_seconds=5.0)
    fine = simulate_fight(a, b, n_sims=60_000, rng=20, tick_seconds=1.0)
    assert coarse.win_prob_a == pytest.approx(fine.win_prob_a, abs=0.012)


def test_a_tick_that_does_not_divide_the_round_is_refused():
    """A short final tick would apply a whole tick's hazard to a partial
    interval and quietly inflate late-round finishes."""
    with pytest.raises(ValueError, match="does not divide"):
        simulate_fight(rates("A"), rates("B"), n_sims=FAST_SIMS, rng=0,
                       tick_seconds=7.0)


# --- guards -----------------------------------------------------------------

def test_zero_simulations_is_refused():
    with pytest.raises(ValueError, match="n_sims"):
        simulate_fight(rates("A"), rates("B"), n_sims=0, rng=0)


def test_too_few_simulations_warns_about_false_precision():
    with pytest.warns(UserWarning, match="standard error"):
        simulate_fight(rates("A"), rates("B"), n_sims=200, rng=0)


def test_an_uncalibrated_round_count_warns():
    """Two-round fights exist 5 times in 8,587 bouts; nothing was fitted on them."""
    with pytest.warns(UserWarning, match="uncalibrated"):
        simulate_fight(rates("A"), rates("B"), rounds=2, n_sims=FAST_SIMS, rng=0)


def test_an_impossible_round_count_is_refused():
    with pytest.raises(ValueError, match="rounds"):
        simulate_fight(rates("A"), rates("B"), rounds=4, n_sims=FAST_SIMS, rng=0)


def test_a_nonsense_rng_is_refused():
    with pytest.raises(TypeError, match="Generator"):
        simulate_fight(rates("A"), rates("B"), n_sims=FAST_SIMS, rng="seed")


def test_an_extreme_multiplier_is_reported_rather_than_refused():
    """Small-sample rates are flagged for the caller to discount, not clipped.

    Shrinking noisy per-fighter rates is career_stats' job, where the evidence
    counts live, so this module invents no threshold of its own.
    """
    freak = rates("Freak", kd_per_str_landed=0.05)
    out = simulate_fight(freak, rates("B", kd_abs_per_str_absorbed=0.05),
                         n_sims=FAST_SIMS, rng=0)
    assert any(name == "a_knockdown" for name, _ in out.extreme_multipliers)


def test_an_ordinary_matchup_reports_no_extreme_multipliers():
    out = simulate_fight(rates("A"), rates("B", slpm=4.2), n_sims=FAST_SIMS, rng=0)
    assert out.extreme_multipliers == ()


# --- reach ------------------------------------------------------------------

def test_the_reach_effect_is_off_by_default():
    """SLpM and striking defence already contain each fighter's own reach as
    measured against real opposition, so applying it again double-counts."""
    long_armed = rates("Long", reach_cm=200.0)
    short = rates("Short", reach_cm=170.0)
    off = simulate_fight(long_armed, short, n_sims=60_000, rng=22)
    on = simulate_fight(long_armed, short, n_sims=60_000, rng=22, reach_effect=True)
    assert on.win_prob_a > off.win_prob_a


def test_a_missing_reach_disables_the_term_rather_than_assuming_parity():
    """A fighter nobody measured is not a fighter of average reach."""
    out = simulate_fight(rates("Unknown", reach_cm=float("nan")),
                         rates("B", reach_cm=180.0),
                         n_sims=FAST_SIMS, rng=0, reach_effect=True)
    assert "reach_cm" in out.imputed_a


# --- calibration against the real dataset -----------------------------------

def test_the_simulated_control_share_matches_the_observed_one():
    """41.65% of fight time is control time, measured over 16,484 fighter-bouts.

    This pins CLINCH_CONTROL_ENTRY_PER_NEUTRAL_MIN, which is the constant that
    constant was calibrated to reproduce. The analytic derivation that ignores
    round-end truncation gives 0.1485 and a 32.8% control share, so this test
    is what separates the calibrated value from the derived one.
    """
    lg = league_rates()
    out = simulate_fight(lg, lg, rounds=3, n_sims=60_000, rng=23)
    control = (out.mean_round_tallies["ctrl_sec_a"].sum()
               + out.mean_round_tallies["ctrl_sec_b"].sum())
    assert control / out.mean_duration_sec == pytest.approx(0.4165, abs=0.015)


def test_two_league_fighters_land_at_about_the_league_rate():
    """3.5292 significant strikes per minute, measured over 8,587 bouts."""
    lg = league_rates()
    out = simulate_fight(lg, lg, rounds=3, n_sims=40_000, rng=24)
    landed = out.mean_round_tallies["sig_a"].sum()
    per_min = landed / (out.mean_duration_sec / 60.0)
    assert per_min == pytest.approx(sim.LEAGUE_SLPM, rel=0.10)


def test_the_league_method_split_is_near_the_observed_one():
    """Observed over 7,698 three-round bouts: 48.78% decision, 31.74% KO/TKO,
    19.49% submission.

    A league-average self-match is not the population - the population mixes
    uneven matchups, which finish faster - so this is a loose sanity band, not
    a calibration. It exists to catch a method split that has gone structurally
    wrong, such as submissions collapsing to zero when position stops working.
    """
    lg = league_rates()
    out = simulate_fight(lg, lg, rounds=3, n_sims=60_000, rng=25)
    assert out.finish_round_prob[0] == pytest.approx(0.4878, abs=0.06)
    assert out.ko_prob_a + out.ko_prob_b == pytest.approx(0.3174, abs=0.06)
    assert out.sub_prob_a + out.sub_prob_b == pytest.approx(0.1949, abs=0.06)


def test_the_default_judge_noise_reproduces_the_observed_split_share():
    """20.30% of 4,059 real decisions were split.

    Re-derives DEFAULT_JUDGE_NOISE_SD from the same margin distribution it was
    fitted on, so a change to the scoring weights that invalidates it fails
    here rather than silently shifting every decision market.
    """
    # The real per-round margin spread: whole-fight margins over the 3,703
    # three-round decisions have a median absolute value of 8.93 per round.
    margins = np.random.default_rng(0).laplace(0.0, 13.0, size=(8_000, 3))
    sd = calibrate_judge_noise(margins, rng=0)
    assert sd == pytest.approx(sim.DEFAULT_JUDGE_NOISE_SD, rel=0.45)


def test_the_head_share_uses_the_breakdown_denominator():
    """401,810 head of 637,709 head+body+leg = 0.6301, not 0.6198.

    Dividing by the 648,264 total significant strikes instead would understate
    the head share by 1.6% and silently under-produce accumulation TKOs,
    because 145 of 8,587 bouts carry no target breakdown at all.
    """
    assert sim.HEAD_SHARE_OF_SIG == pytest.approx(401810 / (401810 + 131957 + 103942),
                                                  abs=1e-4)


def test_the_method_split_is_dominated_by_the_imputed_durability_rates():
    """The headline caveat, pinned so it cannot be quietly forgotten.

    All five outcome-derived rates come from one league default, because
    career_stats.py reads no outcome column. Halving league durability moves
    the three-round decision rate by roughly twenty points, which is far larger
    than any difference the measured rates produce between two real fighters.
    A caller who ignores imputed_a/imputed_b is reading a number set mostly by
    an assumption.
    """
    lg = league_rates()
    tough = rates("Tough",
                  ko_loss_per_kd_absorbed=sim.KD_TO_FINISH_LEAGUE * 0.5,
                  tko_loss_per_head_absorbed=sim.TKO_ACCUM_PER_HEAD_STRIKE_LEAGUE * 0.5,
                  sub_loss_per_sub_faced=sim.SUB_SUCCESS_LEAGUE * 0.5)
    league = simulate_fight(lg, lg, rounds=3, n_sims=40_000, rng=31)
    durable = simulate_fight(tough, tough, rounds=3, n_sims=40_000, rng=31)
    assert durable.finish_round_prob[0] - league.finish_round_prob[0] > 0.15


def test_the_accumulation_hazard_is_constant_per_head_strike():
    """Documents a known miss rather than asserting the model is right.

    The real TKO hazard per head strike absorbed falls 7.4-fold as damage
    accumulates (0.00464 under 10 strikes, 0.00062 above 120). This model
    applies one constant hazard, which is why it over-produces late finishes -
    measured at 23.0% of three-round finishes in round 3 against an actual
    13.4%. The decline is mostly selection between fighters, not a within-fight
    mechanism, so it is left uncorrected here and belongs in career_stats.

    This test exists so that anyone who adds a damage-dependent hazard has to
    come here and say so.
    """
    fragile = rates("Fragile", kd_per_str_landed=0.0,
                    tko_loss_per_head_absorbed=0.004)
    out = simulate_fight(rates("Grinder", kd_per_str_landed=0.0), fragile,
                         rounds=5, n_sims=40_000, rng=32)
    finishes = out.finish_round_prob[1:]
    late = finishes[3:].sum() / finishes.sum()
    # A constant hazard keeps finishing at a steady clip; the real distribution
    # front-loads far harder than this.
    assert late > 0.10


# ---------------------------------------------------------------------------
# The five outcome-derived rates. These were imputed from the league for every
# fighter until career_stats began reading winner_id and method, which meant
# the simulator gave two fighters the same chin regardless of who they were.
# ---------------------------------------------------------------------------

import math

import pandas as pd


@pytest.fixture(scope="module")
def ufc():
    path = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not path.exists():
        pytest.skip(f"dataset not present at {path}")
    return pd.read_csv(path, low_memory=False)


def _career_row(**values):
    """A red-corner career_stats row carrying only what a test names."""
    row = {f"r_{k}": v for k, v in values.items()}
    row.setdefault("r_name", "Fighter")
    return row


def test_a_fighter_who_has_never_been_knocked_down_has_no_knockout_rate():
    """0/0 is unknown, not zero. Zero would assert a fighter who cannot be
    stopped however often he is hurt."""
    rates = sim.rates_from_career_stats(
        _career_row(cd_opp_kd_per15=0.0, cd_ko_against_per15=0.0), "r")
    assert math.isnan(rates.ko_loss_per_kd_absorbed)


def test_a_fighter_knocked_down_and_never_stopped_has_a_rate_of_zero():
    """A real 0 is different from a missing one and must survive as 0."""
    rates = sim.rates_from_career_stats(
        _career_row(cd_opp_kd_per15=2.0, cd_ko_against_per15=0.0), "r")
    assert rates.ko_loss_per_kd_absorbed == 0.0


def test_the_submission_rates_divide_by_the_matching_exposure():
    """Wins per attempt made, losses per attempt faced - not per bout, and not
    crossed over, which would read a fighter's offence as his defence."""
    rates = sim.rates_from_career_stats(
        _career_row(cd_sub_per15=4.0, cd_sub_for_per15=1.0,
                    cd_opp_sub_per15=2.0, cd_sub_against_per15=1.0), "r")
    assert rates.sub_success_per_att == pytest.approx(0.25)
    assert rates.sub_loss_per_sub_faced == pytest.approx(0.5)


def test_knockdowns_absorbed_divide_by_strikes_absorbed_not_landed():
    """cd_sapm is per MINUTE and cd_opp_kd_per15 is per fifteen, so this is
    the conversion most likely to be wrong by a factor of fifteen."""
    rates = sim.rates_from_career_stats(
        _career_row(cd_sapm=4.0, cd_opp_kd_per15=3.0), "r")
    assert rates.kd_abs_per_str_absorbed == pytest.approx(3.0 / 60.0)


def test_the_five_rates_are_measured_for_most_fighters_on_real_data(ufc):
    """The point of the change. Every one of these was imputed from the league
    for 100% of fighters beforehand."""
    import career_stats as cs
    df = ufc.sort_values("date").reset_index(drop=True)
    careers = cs.career_stats(df)
    merged = pd.concat([df.reset_index(drop=True),
                        careers.reset_index(drop=True)], axis=1)
    experienced = merged[careers["r_cd_bouts"].to_numpy() >= 3].head(1500)

    measured = {f: 0 for f in ("sub_success_per_att", "kd_abs_per_str_absorbed",
                               "ko_loss_per_kd_absorbed",
                               "tko_loss_per_head_absorbed",
                               "sub_loss_per_sub_faced")}
    for _, row in experienced.iterrows():
        rates = sim.rates_from_career_stats(row.to_dict(), "r")
        for field in measured:
            if not math.isnan(getattr(rates, field)):
                measured[field] += 1
    for field, count in measured.items():
        assert count / len(experienced) > 0.4, (field, count)
