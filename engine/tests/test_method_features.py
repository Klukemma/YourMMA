"""The method model now has features of its own.

It never did. It was trained on the matrix built and audited for "who wins",
where almost every column is a difference between the corners. For who wins
that is the right shape; for how a fight ends it is close to the wrong one,
because a difference cannot tell two finishers from two point-fighters:

    ko_rate_diff   0.0   two men who knock everyone out
    ko_rate_diff   0.0   two men who have never knocked anyone out

On 2026-09-26 the model called Hiestand vs Nakamura a decision at 55%. The
two of them attempt 2.4 submissions per fifteen minutes between them and
neither had ever been finished. It ended by submission in the second.

Two blocks fix the shape: the level of the five finish-family columns, and
the point-in-time per-fifteen-minute finish rates the simulator has always
read and the method model never saw. Measured walk-forward on the confirm
period (2020 onward, 3,310 fights), the pair beat the old model on log loss,
macro-F1, finish Brier, finish recall and finish precision at once.

These tests hold the shape in place. The gain was measured on the method
model alone, so a block leaking into the winner model would move the AUC, the
ROI and the flag quality all at once with none of it measured - which is why
the separation, not just the presence, is asserted here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import feature_inventory as fi
from feature_spec import build_all, paired_diff, paired_level


def test_the_finish_family_emits_a_level():
    """Their differences were always there. The levels are the change."""
    emitted = {n for spec in fi.all_specs() for n in spec.emits}
    for name in ("ko_rate", "sub_rate", "ko_vulnerability", "been_finished",
                 "sub_def"):
        assert f"{name}_diff" in emitted, f"{name}_diff went missing"
        assert f"{name}_level" in emitted, (
            f"{name}_level is the whole point of this change")


def test_the_rate_block_is_declared_and_emits_levels():
    names = fi.method_rate_names()
    assert names, "no finish-rate features declared"
    for spec in fi.METHOD_RATES:
        assert spec.level, f"{spec.name} without a level carries nothing new"
        assert f"{spec.name}_level" in names


def test_the_rate_block_reads_point_in_time_career_columns():
    """cd_* comes from career_stats, the module built to be leak-free. Any
    other source for these would be a fight reading its own result."""
    for spec in fi.METHOD_RATES:
        assert spec.red.startswith("r_cd_"), spec.red
        assert spec.blue.startswith("b_cd_"), spec.blue


def test_nothing_is_emitted_twice():
    """The first draft of the experiment restated the column list by hand and
    duplicated cd_kd_per15, which is declared under RECORD. That silently gave
    the tested variant two copies of one feature, and fixing it changed which
    variant won."""
    emitted = [n for spec in fi.all_specs() for n in spec.emits]
    duplicates = sorted({n for n in emitted if emitted.count(n) > 1})
    assert not duplicates, f"emitted more than once: {duplicates}"


def test_cd_kd_per15_is_not_restated_in_the_rate_block():
    """It is declared under RECORD with its level already."""
    assert "cd_kd_per15" not in {s.name for s in fi.METHOD_RATES}


def test_finish_level_names_are_checked_against_what_is_emitted():
    """A hand-written list that stops matching the specs is how the winner
    model quietly gains a feature nobody measured."""
    assert set(fi.finish_level_names()) <= {
        n for spec in fi.all_specs() for n in spec.emits}


# --- the shape claim, on real arithmetic -----------------------------------

def test_a_difference_cannot_tell_two_finishers_from_two_point_fighters():
    """The claim this whole change rests on, asserted rather than asserted."""
    finishers = pd.Series([0.8]), pd.Series([0.8])
    point_fighters = pd.Series([0.0]), pd.Series([0.0])

    assert paired_diff(*finishers).iloc[0] == paired_diff(*point_fighters).iloc[0]
    assert paired_level(*finishers).iloc[0] != paired_level(*point_fighters).iloc[0]


def test_the_level_separates_the_fight_that_prompted_this():
    """Hiestand vs Nakamura against Castaneda vs Alatengheili, on the real
    submission-attempt rates from the card of 2026-09-26.

    Their differences are the same size. Their levels differ by a factor of
    eighteen. The first ended by submission in round two; the second went to
    a split decision."""
    hiestand, nakamura = 1.067299, 1.288014
    castaneda, alatengheili = 0.129496, 0.0

    d_finishers = abs(paired_diff(pd.Series([hiestand]),
                                  pd.Series([nakamura])).iloc[0])
    d_others = abs(paired_diff(pd.Series([castaneda]),
                               pd.Series([alatengheili])).iloc[0])
    assert d_finishers == pytest.approx(0.221, abs=0.01)
    assert d_others == pytest.approx(0.129, abs=0.01)
    # Same order of magnitude: indistinguishable to a model given only these.
    assert 0.3 < d_others / d_finishers < 3

    l_finishers = paired_level(pd.Series([hiestand]), pd.Series([nakamura])).iloc[0]
    l_others = paired_level(pd.Series([castaneda]), pd.Series([alatengheili])).iloc[0]
    assert l_finishers / max(l_others, 1e-9) > 15


def test_a_missing_rate_stays_missing_through_the_level():
    """An unmeasured fighter must not read as one who never finishes anyone."""
    built = build_all(
        [s for s in fi.METHOD_RATES if s.name == "cd_ko_for_per15"],
        pd.DataFrame({"r_cd_ko_for_per15": [np.nan],
                      "b_cd_ko_for_per15": [0.5]}))
    assert np.isnan(built["cd_ko_for_per15_level"].iloc[0])
    assert np.isnan(built["cd_ko_for_per15_diff"].iloc[0])
    assert built["cd_ko_for_per15_known"].iloc[0] == 0


# --- the separation --------------------------------------------------------
# predict_card is not imported here: it trains four models on eight thousand
# fights at import time. The lists it builds from are read out of the source
# instead, which is enough to catch the failure that matters - a block
# crossing into the winner model.

def _source():
    return (ENGINE / "predict_card.py").read_text()


def test_the_new_blocks_are_held_out_of_the_winner_model():
    src = _source()
    assert "METHOD_RATE_FEATURES = method_rate_names() + finish_level_names()" in src
    assert "METHOD_ONLY_FEATURES = CAGE_CONTROL_FEATURES + METHOD_RATE_FEATURES" in src
    # SPEC_FEATURES is what the winner model is built from, and it must be
    # filtered by the method-only list, not by cage control alone.
    assert ("SPEC_FEATURES = [n for n in emitted_names(SPECS)\n"
            "                 if n not in METHOD_ONLY_FEATURES]") in src


def test_the_method_models_are_given_everything():
    src = _source()
    assert "feature_cols = SPEC_FEATURES + bespoke_features + METHOD_ONLY_FEATURES" in src
    assert "feature_cols_winner = SPEC_FEATURES + bespoke_features" in src


def test_the_differences_stay_in_the_winner_model():
    """Only the levels are new. Taking the differences away would change the
    winner model, and the winner model's numbers are what the ROI, the
    calibration and the failure model are all measured against."""
    method_only = set(fi.method_rate_names()) | set(fi.finish_level_names())
    for name in ("ko_rate_diff", "sub_rate_diff", "ko_vulnerability_diff",
                 "been_finished_diff", "sub_def_diff"):
        assert name not in method_only, (
            f"{name} has always been in the winner model and must stay")


# --- the UNAVAILABLE list -------------------------------------------------
# prediction_row.UNAVAILABLE forces a suffix to NaN at prediction time. Its
# career-column entries were written before career_stats.final_stats()
# existed, and for as long as they stayed there the winner model trained on
# twelve features it could never read.
#
# They are off the list now. Restoring them was measured and is NOT a gain:
# paired bootstrap over 3,318 confirm-period fights gives 95% [-1.5%, +0.1%]
# and AUC moves by a thousandth. It is a correctness fix - a model should not
# train on features it cannot read - and these tests keep it from drifting
# back rather than claiming it bought anything.

def test_no_career_column_is_forced_to_nan():
    from prediction_row import UNAVAILABLE

    from career_stats import CAREER_COLUMNS

    listed = sorted(set(CAREER_COLUMNS) & set(UNAVAILABLE))
    assert not listed, (
        f"final_stats supplies {listed} and predict_card merges them into the "
        f"stats dict, so listing them here throws away values that are present")


def test_what_remains_unavailable_really_is():
    """A snapshot cannot rebuild a window over a fighter's last three bouts.
    Everything still on the list must be of that kind."""
    from prediction_row import UNAVAILABLE

    from career_stats import CAREER_COLUMNS

    supplied = set(CAREER_COLUMNS)
    for suffix in UNAVAILABLE:
        assert suffix not in supplied, suffix


def test_a_listed_suffix_is_still_nan_even_when_a_value_is_present():
    """The mechanism itself, on a suffix that is genuinely unavailable."""
    import numpy as np

    from prediction_row import UNAVAILABLE, _value

    listed = next(iter(UNAVAILABLE))
    assert np.isnan(_value({listed: 0.97}, listed, None))
    # And a career column now comes through rather than being discarded.
    assert _value({"cd_kd_per15": 0.97}, "cd_kd_per15", None) == 0.97


def test_the_method_rate_block_is_readable_at_prediction_time():
    """A feature the model trains on and never sees live is worse than no
    feature: the training rows carry signal the prediction rows cannot."""
    from prediction_row import UNAVAILABLE

    for spec in fi.METHOD_RATES:
        suffix = spec.red[2:]          # strip the r_ prefix
        assert suffix not in UNAVAILABLE, (
            f"{suffix} feeds the method model but is forced to NaN live")
