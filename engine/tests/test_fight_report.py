"""Tests for the simulation layer.

The module's job is to add what the model cannot say - method, round and
duration - without ever overstating what it knows. Most of these check the
second half of that.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import career_stats as cs
import fight_report as fr

UFC_CSV = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


@pytest.fixture(scope="module")
def fighters():
    if not UFC_CSV.exists():
        pytest.skip(f"dataset not present at {UFC_CSV}")
    df = pd.read_csv(UFC_CSV, low_memory=False).sort_values("date")
    final = cs.final_stats(df.reset_index(drop=True))
    # Two fighters with plenty of history, so nothing is imputed.
    busy = final[final["cd_bouts"] >= 10]
    return dict(busy.iloc[0]), dict(busy.iloc[1])


def test_an_unknown_fighter_gives_no_simulation_rather_than_an_even_fight():
    """A silent 50/50 states a conclusion nobody computed, and one debutant
    must not take down a whole card."""
    assert fr.simulate_matchup({}, {}) is None
    # And a fighter with a record but no measured rates: the league defaults
    # would otherwise simulate an average fighter under his name.
    assert fr.simulate_matchup({"cd_bouts": 4}, {"cd_bouts": 4}) is None
    assert fr.summarise(None, "a", "b") is None
    assert "no simulation" in fr.format_line(None)


def test_the_probabilities_of_every_outcome_sum_to_one(fighters):
    red, blue = fighters
    s = fr.summarise(fr.simulate_matchup(red, blue), "red", "blue")
    assert s["red_win"] + s["blue_win"] + s["draw"] == pytest.approx(1.0, abs=1e-6)
    # The three methods share the probability space with a DRAW, which is
    # none of them, so they sum to 1 minus the draw rather than to 1.
    assert (s["ko"] + s["sub"] + s["decision"] + s["draw"]
            == pytest.approx(1.0, abs=1e-3))


def test_the_same_seed_gives_the_same_answer_twice(fighters):
    """A prediction that moves when nothing changed cannot be checked against
    what happened."""
    red, blue = fighters
    first = fr.summarise(fr.simulate_matchup(red, blue, seed=7), "a", "b")
    second = fr.summarise(fr.simulate_matchup(red, blue, seed=7), "a", "b")
    assert first["red_win"] == second["red_win"]
    assert first["ko"] == second["ko"]


def test_a_fighter_simulated_against_himself_is_an_even_fight(fighters):
    """The strongest check available without knowing the right answer: any
    asymmetry here is a bug in the simulator, not a feature of the fighters."""
    red, _ = fighters
    s = fr.summarise(fr.simulate_matchup(red, dict(red)), "a", "b")
    assert s["red_win"] == pytest.approx(s["blue_win"], abs=0.03)


def test_a_five_round_fight_lasts_longer_and_ends_by_decision_less_often(fighters):
    red, blue = fighters
    three = fr.summarise(fr.simulate_matchup(red, blue, rounds=3), "a", "b")
    five = fr.summarise(fr.simulate_matchup(red, blue, rounds=5), "a", "b")
    assert five["mean_seconds"] > three["mean_seconds"]
    assert five["decision"] < three["decision"]


def test_the_summary_says_which_rates_were_assumed(fighters):
    """A confident distribution built on league averages is the one thing a
    reader has to be able to see."""
    red, blue = fighters
    s = fr.summarise(fr.simulate_matchup(red, blue), "a", "b")
    assert "imputed_red" in s and "imputed_blue" in s
    thin = fr.summarise(fr.simulate_matchup({"cd_bouts": 1}, blue), "a", "b")
    if thin is not None:
        assert thin["imputed_red"], "a fighter with no history imputes nothing?"


def test_the_shrinkage_is_what_lets_a_thin_record_simulate_at_all():
    """simulate.py refuses a takedown defence of exactly 1.0 - a two-fight
    fighter would be invulnerable for the whole fight - and career_stats emits
    unshrunk ratios by design. Without shrunk_career_row this returns None.
    """
    import matchup_inputs as mi
    raw = {"r_cd_td_def": 1.0, "r_cd_minutes": 10.0, "r_cd_opp_td_atmpted": 2.0}
    assert mi.shrunk_career_row(raw, "r")["r_cd_td_def"] < 1.0


def test_the_round_distribution_is_indexed_from_round_one(fighters):
    """finish_round_prob[0] means "went to decision", not "round 1".

    Reading it as round 1 printed a 70% opening round on a fight the same
    line called a 66% decision. The check that catches it is that the two
    routes to the decision probability must agree - one from the method
    counts, one from the round histogram.
    """
    red, blue = fighters
    s = fr.summarise(fr.simulate_matchup(red, blue, rounds=3), "a", "b")
    assert len(s["finish_by_round"]) == 3
    # They agree up to the DRAW, which reaches the final bell like any other
    # decision but is a decision win for neither fighter. That gap is the
    # identity rather than an error in it.
    assert s["decision_share"] == pytest.approx(s["decision"] + s["draw"],
                                                abs=1e-9)
    assert (sum(s["finish_by_round"]) + s["decision_share"]
            == pytest.approx(1.0, abs=1e-6))


def test_a_five_round_fight_reports_five_rounds(fighters):
    red, blue = fighters
    s = fr.summarise(fr.simulate_matchup(red, blue, rounds=5), "a", "b")
    assert len(s["finish_by_round"]) == 5


def test_finishes_are_front_loaded_rather_than_rising(fighters):
    """Knockout power fades with fatigue and the survivors are the durable
    ones, so a later round cannot be the most likely finishing round."""
    red, blue = fighters
    s = fr.summarise(fr.simulate_matchup(red, blue, rounds=3), "a", "b")
    rounds = s["finish_by_round"]
    assert rounds[0] == max(rounds)
