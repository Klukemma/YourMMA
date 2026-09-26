"""Tests for point-in-time career statistics.

These pin the fix for the defect in engine/leakage.py: eight profile columns
carry a fighter's whole-career averages as of the data pull, joined onto every
bout they ever had, which is how a walk-forward backtest returned 68.6% and
+16.2% over 5,943 priced bets.

Two kinds of test here. The synthetic ones build three- and four-bout careers
by hand, because that is the only way to state exactly what a value should be
and therefore the only way to catch an off-by-one in the history window. The
real-data ones run the two verifiers over the 8,587-bout file, and include the
check that the verifiers can actually fail - a proof that cannot fail proves
nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import career_stats as cs
from leakage import constant_across_career

UFC_CSV = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


# --- fixtures --------------------------------------------------------------

def bout(date, r_id, b_id, **counts):
    """One bout with everything zeroed, so a test states only what it means.

    Zero is the right default here because these are COUNTS of things that
    happened in a bout, not rates: a fighter really can land no takedowns. The
    rule this module exists to enforce - never fill a missing rate with a
    number that means something else - is about the emitted statistics, and is
    tested separately with explicit NaNs.
    """
    # winner_id defaults to the red corner. A test that cares about the record
    # passes it explicitly; one that does not still needs a decided bout,
    # because None would silently make every fixture a no-contest.
    row = {"date": date, "match_time_sec": 300.0, "finish_round": 1,
           "total_rounds": 3, "r_id": r_id, "b_id": b_id,
           "r_name": r_id, "b_name": b_id, "winner_id": r_id,
           "method": "Decision - Unanimous"}
    for corner in cs.CORNERS:
        for column in cs._COUNT_COLUMNS:
            row[f"{corner}_{column}"] = 0.0
    row.update(counts)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


@pytest.fixture(scope="module")
def ufc():
    if not UFC_CSV.exists():
        pytest.skip(f"dataset not present at {UFC_CSV}")
    return pd.read_csv(UFC_CSV, low_memory=False)


@pytest.fixture(scope="module")
def ufc_careers(ufc):
    return cs.career_stats(ufc)


# --- fight duration: the shared divisor, so it needs its own evidence ------

def test_a_first_round_finish_is_read_as_the_round_clock():
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=120.0,
                    finish_round=1))
    assert cs.fight_elapsed_seconds(df).iloc[0] == 120.0


def test_a_later_round_finish_adds_the_rounds_already_completed():
    """match_time_sec is the clock WITHIN the final round on 8,486/8,587 rows."""
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=60.0,
                    finish_round=3))
    assert cs.fight_elapsed_seconds(df).iloc[0] == 660.0


def test_a_cumulative_match_time_is_not_counted_twice():
    """The other 101 rows already hold total elapsed; adding rounds doubles it."""
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=900.0,
                    finish_round=3, total_rounds=3))
    assert cs.fight_elapsed_seconds(df).iloc[0] == 900.0


def test_control_time_never_exceeds_the_derived_fight_duration(ufc):
    """2,015 rows violated this before the repair; 0 after it."""
    elapsed = cs.fight_elapsed_seconds(ufc)
    control = ufc["r_ctrl"] + ufc["b_ctrl"]
    assert int((control > elapsed).sum()) == 0


def test_every_decision_lands_inside_its_scheduled_distance(ufc):
    """A three-round decision cannot run past 900s, a five-round past 1500s."""
    elapsed = cs.fight_elapsed_seconds(ufc)
    decision = ufc["method"].astype(str).str.contains("Decision", case=False)
    limit = ufc["total_rounds"] * cs.ROUND_SECONDS
    assert int((elapsed[decision] > limit[decision]).sum()) == 0


def test_no_bout_in_the_file_has_an_impossible_duration(ufc):
    """The > 300 rule assumes five-minute rounds; this is what would catch it."""
    assert int(cs.duration_violations(ufc).sum()) == 0


# --- the history window ----------------------------------------------------

def test_a_debut_has_no_rates_at_all():
    careers = cs.career_stats(frame(bout("2020-01-01", "a", "b")))
    rates = [c for c in cs.career_columns("r")
             # Counts, not rates. Zero prior wins is a true statement about a
             # debutant; zero strikes per minute would not be.
             if c not in ("r_cd_bouts", "r_cd_minutes", "r_cd_sig_atmpted",
                          "r_cd_opp_sig_atmpted", "r_cd_td_atmpted",
                          "r_cd_opp_td_atmpted", "r_cd_breakdown_bouts",
                          "r_cd_wins", "r_cd_losses")]
    assert careers.loc[0, rates].isna().all()


def test_a_debutant_never_reads_as_a_fighter_who_lands_nothing():
    """0.0 SLpM is the worst fighter alive; no history is a different fact."""
    careers = cs.career_stats(frame(bout("2020-01-01", "a", "b")))
    assert pd.isna(careers.loc[0, "r_cd_slpm"])


def test_a_debut_reports_a_zero_bout_count_to_say_so():
    careers = cs.career_stats(frame(bout("2020-01-01", "a", "b")))
    assert careers.loc[0, "r_cd_bouts"] == 0


def test_a_fighters_own_bout_never_counts_toward_its_own_statistics():
    """The whole defect in one assertion."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=100.0),
               bout("2020-06-01", "a", "c", r_sig_str_landed=0.0))
    careers = cs.career_stats(df)
    # 100 strikes over 5 minutes in the first bout, and only that bout.
    assert careers.loc[1, "r_cd_slpm"] == pytest.approx(20.0)


def test_history_accumulates_across_both_corners():
    """A fighter is red in some rows and blue in others; it is one career."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0),
               bout("2020-06-01", "c", "a", b_sig_str_landed=60.0),
               bout("2021-01-01", "a", "d"))
    careers = cs.career_stats(df)
    assert careers.loc[2, "r_cd_bouts"] == 2
    assert careers.loc[2, "r_cd_slpm"] == pytest.approx(90.0 / 10.0)


def test_bouts_on_the_same_date_are_mutually_blind():
    """No intra-day timestamp exists, so a same-card bout cannot be ordered."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=50.0),
               bout("2020-01-01", "a", "c", r_sig_str_landed=50.0))
    careers = cs.career_stats(df)
    assert careers["r_cd_bouts"].tolist() == [0, 0]


def test_the_next_date_sees_every_bout_from_the_shared_one():
    """Blindness within a date must not become amnesia after it."""
    df = frame(bout("2020-01-01", "a", "b"),
               bout("2020-01-01", "a", "c"),
               bout("2020-06-01", "a", "d"))
    assert cs.career_stats(df).loc[2, "r_cd_bouts"] == 2


def test_two_fighters_sharing_a_name_keep_separate_careers():
    """28 names in the file map to two distinct ids - two Bruno Silvas."""
    df = frame(bout("2020-01-01", "id1", "x", r_sig_str_landed=100.0),
               bout("2020-06-01", "id2", "y"))
    careers = cs.career_stats(df.assign(r_name="Bruno Silva"))
    assert careers.loc[1, "r_cd_bouts"] == 0


def test_one_fighter_under_two_spellings_keeps_a_single_career():
    """5 ids in the file carry two name spellings."""
    df = frame(bout("2020-01-01", "id1", "x"),
               bout("2020-06-01", "id1", "y"))
    df.loc[1, "r_name"] = "Different Spelling"
    assert cs.career_stats(df).loc[1, "r_cd_bouts"] == 1


# --- the statistics themselves ---------------------------------------------

def test_strikes_absorbed_come_from_the_opponents_landed_count():
    df = frame(bout("2020-01-01", "a", "b", b_sig_str_landed=45.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_sapm"] == pytest.approx(9.0)


def test_striking_defence_is_one_minus_what_opponents_landed():
    df = frame(bout("2020-01-01", "a", "b", b_sig_str_landed=20.0,
                    b_sig_str_atmpted=100.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_str_def"] == pytest.approx(0.8)


def test_takedown_defence_counts_the_opponents_attempts():
    df = frame(bout("2020-01-01", "a", "b", b_td_landed=1.0, b_td_atmpted=5.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_td_def"] == pytest.approx(0.8)


def test_a_fighter_who_never_tried_a_takedown_has_no_takedown_accuracy():
    """663 rows with history. 0.0 would assert someone who tries and fails."""
    df = frame(bout("2020-01-01", "a", "b"), bout("2020-06-01", "a", "c"))
    assert pd.isna(cs.career_stats(df).loc[1, "r_cd_td_acc"])


def test_untested_takedown_defence_is_not_reported_as_perfect():
    """512 rows. Both 0.0 and 1.0 are lies when nobody has shot on you."""
    df = frame(bout("2020-01-01", "a", "b"), bout("2020-06-01", "a", "c"))
    assert pd.isna(cs.career_stats(df).loc[1, "r_cd_td_def"])


def test_control_share_is_seconds_over_the_full_fight():
    df = frame(bout("2020-01-01", "a", "b", r_ctrl=150.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_ctrl_share"] == pytest.approx(0.5)


def test_rates_are_pooled_career_totals_not_a_mean_of_per_fight_rates():
    """A 15-second bout must not weigh as much as a 15-minute one."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=10.0,
                    match_time_sec=60.0),
               bout("2020-06-01", "a", "c", r_sig_str_landed=10.0,
                    match_time_sec=540.0),
               bout("2021-01-01", "a", "d"))
    # Pooled: 20 strikes over 10 minutes. Mean-of-rates would give 6.0.
    assert cs.career_stats(df).loc[2, "r_cd_slpm"] == pytest.approx(2.0)


def test_per_minute_rates_divide_by_the_repaired_duration():
    """Taking match_time_sec at face value would report 30.0 here."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    match_time_sec=60.0, finish_round=2),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_slpm"] == pytest.approx(5.0)


def test_takedowns_are_normalised_per_fifteen_minutes():
    df = frame(bout("2020-01-01", "a", "b", r_td_landed=1.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_td_per15"] == pytest.approx(3.0)


def test_the_head_body_leg_shares_sum_to_one():
    df = frame(bout("2020-01-01", "a", "b", r_head_landed=6.0,
                    r_body_landed=3.0, r_leg_landed=1.0),
               bout("2020-06-01", "a", "c"))
    careers = cs.career_stats(df)
    total = sum(careers.loc[1, f"r_cd_{p}_share"]
                for p in ("head", "body", "leg"))
    assert total == pytest.approx(1.0)


def test_a_position_share_divides_by_the_same_landed_total():
    """distance+clinch+ground partitions the same strikes as head+body+leg."""
    df = frame(bout("2020-01-01", "a", "b", r_head_landed=8.0,
                    r_body_landed=2.0, r_dist_landed=7.0, r_clinch_landed=3.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_dist_share"] == pytest.approx(0.7)


# --- missing operands: each ratio skips only the bouts it has to -----------

def test_a_bout_missing_its_breakdown_still_counts_toward_slpm():
    """145 bouts in 2025 lost the whole breakdown for both corners at once."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    r_head_landed=np.nan, r_body_landed=np.nan,
                    r_leg_landed=np.nan, r_dist_landed=np.nan,
                    r_clinch_landed=np.nan, r_ground_landed=np.nan),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_slpm"] == pytest.approx(6.0)


def test_a_bout_missing_its_breakdown_is_left_out_of_the_shares():
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    r_head_landed=np.nan, r_body_landed=np.nan,
                    r_leg_landed=np.nan, r_dist_landed=np.nan,
                    r_clinch_landed=np.nan, r_ground_landed=np.nan),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_breakdown_bouts"] == 0


def test_a_partial_breakdown_is_never_mixed_with_a_complete_one():
    """Six parts present or none; a partial share is a ratio of two things."""
    df = frame(bout("2020-01-01", "a", "b", r_head_landed=10.0,
                    r_dist_landed=np.nan),
               bout("2020-06-01", "a", "c", r_head_landed=5.0,
                    r_body_landed=5.0, r_dist_landed=10.0),
               bout("2021-01-01", "a", "d"))
    assert cs.career_stats(df).loc[2, "r_cd_head_share"] == pytest.approx(0.5)


def test_an_accuracy_never_mixes_a_present_numerator_with_an_absent_denominator():
    """2 bouts in the file have sig_str_landed but no sig_str_atmpted."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    r_sig_str_atmpted=np.nan),
               bout("2020-06-01", "a", "c", r_sig_str_landed=10.0,
                    r_sig_str_atmpted=20.0),
               bout("2021-01-01", "a", "d"))
    careers = cs.career_stats(df)
    assert careers.loc[2, "r_cd_str_acc"] == pytest.approx(0.5)


def test_that_same_bout_still_counts_toward_strikes_per_minute():
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    r_sig_str_atmpted=np.nan),
               bout("2020-06-01", "a", "c", r_sig_str_landed=10.0,
                    r_sig_str_atmpted=20.0),
               bout("2021-01-01", "a", "d"))
    assert cs.career_stats(df).loc[2, "r_cd_slpm"] == pytest.approx(4.0)


def test_a_fighter_with_no_attempts_at_all_has_no_accuracy():
    df = frame(bout("2020-01-01", "a", "b"), bout("2020-06-01", "a", "c"))
    assert pd.isna(cs.career_stats(df).loc[1, "r_cd_str_acc"])


def test_the_evidence_behind_each_ratio_is_published_alongside_it():
    """So a caller can weight 20 attempts differently from 2,000."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=10.0,
                    r_sig_str_atmpted=25.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_sig_atmpted"] == 25.0


# --- shape, purity and the contract ----------------------------------------

def test_the_output_carries_exactly_the_declared_columns():
    careers = cs.career_stats(frame(bout("2020-01-01", "a", "b")))
    assert list(careers.columns) == cs.career_columns()


def test_career_columns_names_both_corners():
    assert len(cs.career_columns()) == 2 * len(cs.CAREER_COLUMNS) == 68


def test_career_columns_rejects_a_corner_that_does_not_exist():
    with pytest.raises(ValueError):
        cs.career_columns("red")


def test_the_output_is_aligned_to_a_non_default_input_index():
    df = frame(bout("2020-01-01", "a", "b"), bout("2020-06-01", "a", "c"))
    df.index = ["first", "second"]
    assert list(cs.career_stats(df).index) == ["first", "second"]


def test_a_shuffled_input_gives_identical_values():
    """The file happens to be date-sorted; nothing here may rely on that."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=50.0),
               bout("2020-06-01", "a", "c", r_sig_str_landed=10.0),
               bout("2021-01-01", "a", "d"))
    shuffled = df.iloc[[2, 0, 1]]
    pd.testing.assert_frame_equal(cs.career_stats(shuffled),
                                  cs.career_stats(df).loc[shuffled.index])


def test_the_input_frame_is_not_mutated():
    df = frame(bout("2020-01-01", "a", "b"), bout("2020-06-01", "a", "c"))
    before = df.copy()
    cs.career_stats(df)
    pd.testing.assert_frame_equal(df, before)


def test_a_missing_required_column_raises_instead_of_emitting_nan():
    """An all-NaN column reads downstream as 'nobody has any history'."""
    df = frame(bout("2020-01-01", "a", "b")).drop(columns=["r_ctrl"])
    with pytest.raises(KeyError, match="r_ctrl"):
        cs.career_stats(df)


def test_an_unorderable_date_raises_because_it_cannot_be_made_safe():
    df = frame(bout("not a date", "a", "b"))
    with pytest.raises(ValueError, match="date"):
        cs.career_stats(df)


def test_a_duration_past_the_scheduled_distance_raises_under_strict():
    """A non-five-minute round format would otherwise be silently misread."""
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=1200.0,
                    finish_round=5, total_rounds=3))
    with pytest.raises(ValueError, match="duration"):
        cs.career_stats(df)


def test_an_impossible_duration_becomes_unknown_rather_than_wrong():
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=1200.0,
                    finish_round=5, total_rounds=3, r_sig_str_landed=30.0),
               bout("2020-06-01", "a", "c"))
    careers = cs.career_stats(df, strict=False)
    assert pd.isna(careers.loc[1, "r_cd_slpm"])


def test_a_bout_with_an_unusable_duration_still_counts_as_a_bout():
    df = frame(bout("2020-01-01", "a", "b", match_time_sec=1200.0,
                    finish_round=5, total_rounds=3),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df, strict=False).loc[1, "r_cd_bouts"] == 1


# --- reliability -----------------------------------------------------------

def test_reliability_requires_both_corners_by_default():
    careers = pd.DataFrame({"r_cd_bouts": [5.0], "b_cd_bouts": [1.0]})
    assert not cs.reliability_mask(careers).iloc[0]


def test_reliability_can_be_asked_for_either_corner_instead():
    careers = pd.DataFrame({"r_cd_bouts": [5.0], "b_cd_bouts": [1.0]})
    assert cs.reliability_mask(careers, both_corners=False).iloc[0]


def test_career_stats_does_not_apply_the_reliability_threshold_itself():
    """A one-bout measurement is thin, not absent; discarding it is a new error."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=25.0),
               bout("2020-06-01", "a", "c"))
    assert cs.career_stats(df).loc[1, "r_cd_slpm"] == pytest.approx(5.0)


# --- the proofs, and proof that the proofs can fail ------------------------

def test_the_lookahead_verifier_passes_on_a_hand_built_career():
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0,
                    r_td_landed=1.0, r_td_atmpted=3.0),
               bout("2020-06-01", "c", "a", b_sig_str_landed=10.0),
               bout("2021-01-01", "a", "d"))
    assert cs.verify_no_lookahead(df, sample=None)["mismatches"] == 0


def test_the_lookahead_verifier_catches_a_planted_future_value():
    """A proof that cannot fail proves nothing."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0),
               bout("2020-06-01", "a", "c"))
    careers = cs.career_stats(df)
    careers.loc[0, "r_cd_slpm"] = 6.0  # what the first bout itself would give
    assert cs.verify_no_lookahead(df, careers, sample=None)["mismatches"] > 0


def test_the_truncation_verifier_catches_a_joined_career_total():
    """This is exactly what r_splm does, and why it must fail here."""
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0),
               bout("2022-01-01", "a", "c", r_sig_str_landed=60.0))
    careers = cs.career_stats(df)
    careers["r_cd_slpm"] = 9.0  # one whole-career number on every bout
    out = cs.verify_truncation_invariance(df, "2021-01-01", careers)
    assert out["mismatches"] > 0


def test_the_verifier_reports_which_column_disagreed():
    df = frame(bout("2020-01-01", "a", "b", r_sig_str_landed=30.0),
               bout("2020-06-01", "a", "c"))
    careers = cs.career_stats(df)
    careers.loc[1, "r_cd_sapm"] = 99.0
    failures = cs.verify_no_lookahead(df, careers, sample=None)["failures"]
    assert failures[0]["column"] == "cd_sapm"


# --- against the real 8,587-bout dataset -----------------------------------

def test_no_lookahead_on_a_sample_of_the_real_dataset(ufc, ufc_careers):
    """Exhaustive run over all 17,174 appearances also returns 0; this is the
    sampled version, kept at 400 so it stays inside a normal pytest run."""
    out = cs.verify_no_lookahead(ufc, ufc_careers)
    assert out["mismatches"] == 0, out["failures"]


def test_truncating_the_dataset_changes_nothing_before_the_cutoff(ufc, ufc_careers):
    out = cs.verify_truncation_invariance(ufc, "2020-01-01", ufc_careers)
    assert out["mismatches"] == 0, out["failures"]


def test_the_truncation_proof_covers_most_of_the_file(ufc, ufc_careers):
    """A proof over three rows would be worth little."""
    out = cs.verify_truncation_invariance(ufc, "2020-01-01", ufc_careers)
    assert out["rows"] > 5000


def test_a_career_to_date_rate_moves_as_the_career_progresses(ufc, ufc_careers):
    """The leak detector's own signature, run against the replacement."""
    joined = ufc.join(ufc_careers)
    leaked = constant_across_career(joined, "r_splm")
    honest = constant_across_career(joined, "r_cd_slpm")
    assert leaked["constant_share"] > 0.5 > honest["constant_share"]


def test_accuracies_are_fractions_not_percentages(ufc_careers):
    """cd_str_acc is 0.465 where the leaked r_str_acc is 46.5, by design."""
    for column in ("r_cd_str_acc", "r_cd_str_def", "r_cd_td_acc",
                   "r_cd_td_def", "r_cd_ctrl_share", "r_cd_head_share"):
        assert ufc_careers[column].max() <= 1.0, column


def test_no_rate_is_negative(ufc_careers):
    for column in ("r_cd_slpm", "r_cd_sapm", "r_cd_td_per15", "r_cd_kd_per15",
                   "r_cd_str_def", "r_cd_td_def"):
        assert ufc_careers[column].min() >= 0.0, column


def test_every_fighter_with_history_has_a_positive_career_minutes(ufc_careers):
    """A zero denominator would silently NaN every per-minute rate."""
    with_history = ufc_careers["r_cd_bouts"] > 0
    assert (ufc_careers.loc[with_history, "r_cd_minutes"] > 0).all()


def test_debut_rows_are_the_only_ones_without_a_strikes_per_minute(ufc_careers):
    """cd_bouts is what separates 'no history' from 'genuinely zero'."""
    missing = ufc_careers["r_cd_slpm"].isna()
    assert (ufc_careers.loc[missing, "r_cd_bouts"] == 0).all()


def test_about_a_quarter_of_rows_have_a_debutant_in_one_corner(ufc_careers):
    """24.7% measured - the share of rows a careless NaN fill would corrupt."""
    debutant = ((ufc_careers["r_cd_bouts"] == 0)
                | (ufc_careers["b_cd_bouts"] == 0))
    assert 0.20 < debutant.mean() < 0.30


def test_career_to_date_slpm_reproduces_the_published_career_average(ufc):
    """Jon Jones' whole career: 4.3834 here against a published 4.38.

    Read one bout past his last by dating a copy of it into the future, so the
    accumulator has folded in every bout he actually fought. That the published
    number falls out of minute-denominated totals is also what confirms the
    duration repair from outside this module.
    """
    jones = ufc[(ufc["r_name"] == "Jon Jones") | (ufc["b_name"] == "Jon Jones")]
    future = jones.tail(1).copy()
    future["date"] = "2030-01-01"
    extended = pd.concat([ufc, future], ignore_index=True)
    careers = cs.career_stats(extended)
    corner = "r" if extended["r_name"].iloc[-1] == "Jon Jones" else "b"
    assert careers[f"{corner}_cd_slpm"].iloc[-1] == pytest.approx(4.38, abs=0.01)


def test_the_pass_is_fast_enough_to_sit_in_the_pipeline(ufc):
    """8,587 rows x 2 corners; 0.05s measured, against a 10-minute pipeline."""
    import time
    started = time.perf_counter()
    cs.career_stats(ufc)
    assert time.perf_counter() - started < 5.0


# ---------------------------------------------------------------------------
# Gaps found by mutating the module and watching the suite stay green.
# Each mutation below broke the module's central promise and passed anyway.
# ---------------------------------------------------------------------------

def _three_bout_career(**overrides):
    """One fighter, three dated bouts, every required column present."""
    n = 3
    frame = {
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-01"]),
        "match_time_sec": [300.0] * n,
        "finish_round": [1.0] * n,
        "total_rounds": [3.0] * n,
        "winner_id": ["F1"] * n,
        "method": ["Decision - Unanimous"] * n,
    }
    for corner in ("r", "b"):
        frame[f"{corner}_id"] = (["F1"] * n if corner == "r"
                                 else ["A", "B", "C"])
        for col in cs._COUNT_COLUMNS:
            frame[f"{corner}_{col}"] = [0.0] * n
    frame.update(overrides)
    return pd.DataFrame(frame)


def test_a_bout_with_an_unknown_control_time_is_left_out_of_the_denominator():
    """Mutation that passed: zero-filling a NaN numerator before accumulating.

    A career of {bout 1: control unrecorded, bout 2: 300s of a 300s fight}
    must read as "controlled the whole of the fight we can see", not "half".
    Reading the unrecorded bout as 0 seconds asserts the fighter controlled
    nobody, which is a fact no one wrote down.
    """
    df = _three_bout_career(r_ctrl=[np.nan, 300.0, 0.0])
    out = cs.career_stats(df)
    assert out.loc[2, "r_cd_ctrl_share"] == pytest.approx(1.0)


def test_a_bout_with_an_unknown_strike_count_does_not_shrink_strikes_per_minute():
    """The same mutation on the striking side: an unrecorded bout must not be
    read as a bout in which the fighter landed nothing."""
    df = _three_bout_career(r_sig_str_landed=[np.nan, 50.0, 0.0])
    out = cs.career_stats(df)
    # 50 strikes over the single 5-minute bout we can see is 10.0 per minute,
    # not the 5.0 it would be if the unknown bout contributed 0 over 5 minutes.
    assert out.loc[2, "r_cd_slpm"] == pytest.approx(10.0)


def test_the_lookahead_proof_actually_checks_a_meaningful_sample():
    """Mutation that passed: dropping the sample size to 1.

    verify_truncation_invariance only catches information crossing its cutoff
    date, so it is structurally blind to a within-era off-by-one. The sampled
    lookahead proof is the only defence against that, and a proof that checks
    one row is not a proof.
    """
    csv = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
    if not csv.exists():
        pytest.skip("dataset not present")
    out = cs.verify_no_lookahead(pd.read_csv(csv, low_memory=False))
    assert out["mismatches"] == 0
    assert out["checked"] >= 200, (
        f"only {out['checked']} fighter-appearances checked; too small a "
        f"sample for this to prove anything")


def test_a_missing_required_column_is_refused_loudly():
    """An all-NaN frame reads downstream as "nobody has any history", which is
    indistinguishable from a roster of debutants and fails nothing."""
    df = _three_bout_career().drop(columns=["r_ctrl"])
    with pytest.raises(KeyError, match="r_ctrl"):
        cs.career_stats(df)


# ---------------------------------------------------------------------------
# The prior record. r_wins/r_losses were the largest leak in the dataset and
# these are their honest replacement, so the guarantee is worth stating twice.
# ---------------------------------------------------------------------------

def test_the_record_counts_only_bouts_already_fought():
    """Three bouts, all won. The record read at each is 0-0, 1-0, 2-0."""
    df = frame(bout("2020-01-01", "F1", "A"),
               bout("2020-06-01", "F1", "B"),
               bout("2020-12-01", "F1", "C"))
    out = cs.career_stats(df)
    assert list(out["r_cd_wins"]) == [0.0, 1.0, 2.0]
    assert list(out["r_cd_losses"]) == [0.0, 0.0, 0.0]


def test_a_loss_is_counted_against_the_fighter_who_lost_it():
    df = frame(bout("2020-01-01", "F1", "A", winner_id="A"),
               bout("2020-06-01", "F1", "B", winner_id="F1"),
               bout("2020-12-01", "F1", "C"))
    out = cs.career_stats(df)
    assert list(out["r_cd_wins"]) == [0.0, 0.0, 1.0]
    assert list(out["r_cd_losses"]) == [0.0, 1.0, 1.0]
    assert out.loc[2, "r_cd_win_rate"] == pytest.approx(0.5)


def test_the_record_follows_a_fighter_into_the_other_corner():
    """A fighter's history is their bouts, not their red-corner bouts."""
    df = frame(bout("2020-01-01", "A", "F1", winner_id="F1"),
               bout("2020-06-01", "F1", "B"))
    out = cs.career_stats(df)
    assert out.loc[1, "r_cd_wins"] == 1.0
    assert out.loc[1, "r_cd_losses"] == 0.0


def test_a_bout_nobody_won_counts_as_neither_a_win_nor_a_loss():
    """150 bouts name no winner - draws, no-contests and overturned results.

    They are real bouts and real cage time, so they must reach cd_bouts, and
    they decided nothing, so they must not reach the win rate.
    """
    df = frame(bout("2020-01-01", "F1", "A", winner_id=None),
               bout("2020-06-01", "F1", "B"))
    out = cs.career_stats(df)
    assert out.loc[1, "r_cd_bouts"] == 1.0
    assert out.loc[1, "r_cd_wins"] == 0.0
    assert out.loc[1, "r_cd_losses"] == 0.0
    assert pd.isna(out.loc[1, "r_cd_win_rate"])


def test_a_fighter_with_no_decided_bout_has_no_win_rate():
    """NaN, never 0.5. A coin-flip fighter is a claim; "unknown" is the fact."""
    out = cs.career_stats(frame(bout("2020-01-01", "F1", "A")))
    assert pd.isna(out.loc[0, "r_cd_win_rate"])
    assert out.loc[0, "r_cd_wins"] == 0.0


def test_the_win_rate_never_sees_the_bout_it_describes(ufc):
    """The property the leak violated, asserted on the real dataset.

    If cd_win_rate could see the current bout, the red corner's rate would be
    lifted on every row the red corner won. Regressing the winner on the rate
    is allowed to find skill; what it must not find is the perfect separation
    that a career total produces.
    """
    careers = cs.career_stats(ufc)
    won = ufc["winner_id"] == ufc["r_id"]
    rate = careers["r_cd_win_rate"]
    seen = rate.notna()
    # A leaked rate scores far above this; the honest one sits near 0.60.
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(won[seen], rate[seen])
    assert 0.50 < auc < 0.70, auc


def test_the_conceded_grappling_columns_mirror_the_offensive_ones(ufc):
    """Control conceded is the opponent's control, so the two pool equally.

    Every second one fighter controls, another is controlled, so across all
    fighter-appearances the two shares must have the same total exposure.
    """
    careers = cs.career_stats(ufc)
    for corner in cs.CORNERS:
        offensive = careers[f"{corner}_cd_ctrl_share"]
        conceded = careers[f"{corner}_cd_opp_ctrl_share"]
        assert (offensive.notna() == conceded.notna()).all()
        assert conceded.dropna().between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# final_stats: the other end of the same accumulation. Training reads what was
# known BEFORE a bout; predicting a fighter's next bout reads what is known
# after their last one. If these two ever disagree the model is served
# different numbers than it was trained on and nothing would say so.
# ---------------------------------------------------------------------------

def test_the_final_stats_equal_what_a_hypothetical_next_bout_would_read():
    """The defining property, checked by actually adding that next bout.

    A fighter's final_stats row must be exactly what career_stats emits for a
    bout dated after everything else - because that is what it claims to be.
    """
    history = frame(bout("2020-01-01", "F1", "A", r_sig_str_landed=40.0,
                         r_sig_str_atmpted=100.0),
                    bout("2020-06-01", "F1", "B", r_sig_str_landed=20.0,
                         r_sig_str_atmpted=100.0, winner_id="B"))
    final = cs.final_stats(history)

    with_next = frame(*history.to_dict("records"),
                      bout("2021-01-01", "F1", "C"))
    next_bout = cs.career_stats(with_next).iloc[-1]

    for name in cs.CAREER_COLUMNS:
        expected, actual = next_bout[f"r_{name}"], final.loc["F1", name]
        assert _equal_or_both_nan(expected, actual), name


def _equal_or_both_nan(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(left, right, rtol=1e-9))


def test_final_stats_agrees_with_career_stats_across_the_real_dataset(ufc):
    """Same check on every fighter in the file, via their last appearance.

    A fighter's last bout carries their stats BEFORE it, so adding that bout's
    own contribution must reproduce the final row. Checked here on the one
    column where the arithmetic is a plain count and so cannot be fudged.
    """
    df = ufc.sort_values("date").reset_index(drop=True)
    careers = cs.career_stats(df)
    final = cs.final_stats(df)

    last = {}
    for pos in range(len(df)):
        for corner in cs.CORNERS:
            last[df[f"{corner}_id"].iloc[pos]] = (pos, corner)

    for fighter, (pos, corner) in list(last.items())[:400]:
        before = careers[f"{corner}_cd_bouts"].iloc[pos]
        assert final.loc[fighter, "cd_bouts"] == before + 1, fighter


def test_every_ufc_win_is_somebody_elses_loss(ufc):
    """Pooled over all fighters the two totals must be identical.

    Not a tautology: it fails the moment a bout credits a win without debiting
    a loss, which is exactly what an id mismatch between the corners produces.
    """
    final = cs.final_stats(ufc.sort_values("date").reset_index(drop=True))
    assert final["cd_wins"].sum() == final["cd_losses"].sum()


def test_final_stats_covers_every_fighter_in_the_file(ufc):
    df = ufc.sort_values("date").reset_index(drop=True)
    everyone = set(df["r_id"]) | set(df["b_id"])
    assert set(cs.final_stats(df).index) == everyone


# ---------------------------------------------------------------------------
# Finish rates. These are what the simulator and matchup.Durability both need
# and what career_stats could not produce while it read no outcome column.
# ---------------------------------------------------------------------------

def test_a_knockout_win_counts_for_the_winner_and_against_the_loser():
    df = frame(bout("2020-01-01", "F1", "A", method="KO/TKO"),
               bout("2020-06-01", "F1", "B"))
    out = cs.career_stats(df)
    assert out.loc[1, "r_cd_ko_for_per15"] > 0
    assert out.loc[1, "r_cd_ko_against_per15"] == 0.0
    # And the opposite corner of that same bout carries the reverse.
    reverse = cs.career_stats(frame(bout("2020-01-01", "A", "F1", winner_id="A",
                                         method="KO/TKO"),
                                    bout("2020-06-01", "F1", "B")))
    assert reverse.loc[1, "r_cd_ko_against_per15"] > 0
    assert reverse.loc[1, "r_cd_ko_for_per15"] == 0.0


def test_a_decision_is_neither_a_knockout_nor_a_submission():
    df = frame(bout("2020-01-01", "F1", "A", method="Decision - Split"),
               bout("2020-06-01", "F1", "B"))
    out = cs.career_stats(df).loc[1]
    for column in ("r_cd_ko_for_per15", "r_cd_ko_against_per15",
                   "r_cd_sub_for_per15", "r_cd_sub_against_per15"):
        assert out[column] == 0.0, column


def test_an_overturned_result_counts_as_no_finish_at_all():
    """113 bouts are a DQ, a no-contest or overturned. Counting one as a
    knockout would credit a finish nobody scored."""
    df = frame(bout("2020-01-01", "F1", "A", method="Overturned",
                    winner_id=None),
               bout("2020-06-01", "F1", "B"))
    out = cs.career_stats(df).loc[1]
    assert out["r_cd_ko_for_per15"] == 0.0
    assert out["r_cd_sub_for_per15"] == 0.0
    assert out["r_cd_wins"] == 0.0


def test_the_pooled_finish_rates_reproduce_the_league_constants(ufc):
    """matchup.py measured these from the raw dataset by a different route.

    Two independent derivations landing on the same number is the evidence
    that the method classification and the accumulation are both right; a
    mis-parsed method string could not agree to four decimal places.
    """
    import matchup as mu
    final = cs.final_stats(ufc.sort_values("date").reset_index(drop=True))
    minutes = final["cd_minutes"]
    for column, league in (("cd_ko_for_per15", mu.LEAGUE_KO_FOR_PER_MIN),
                           ("cd_sub_for_per15", mu.LEAGUE_SUB_FOR_PER_MIN)):
        pooled = (final[column] * minutes / 15.0).sum() / minutes.sum()
        assert pooled == pytest.approx(league, rel=0.01), column


def test_a_knockout_win_is_somebody_elses_knockout_loss(ufc):
    final = cs.final_stats(ufc.sort_values("date").reset_index(drop=True))
    minutes = final["cd_minutes"]
    for won, lost in (("cd_ko_for_per15", "cd_ko_against_per15"),
                      ("cd_sub_for_per15", "cd_sub_against_per15")):
        a = (final[won] * minutes).sum()
        b = (final[lost] * minutes).sum()
        assert a == pytest.approx(b, rel=1e-9)
