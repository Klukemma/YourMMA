"""Tests for the training/prediction adapter.

A live prediction and a training row must produce the same features from the
same declaration. They did not before: the matrix came from dataframe columns
while a prediction was assembled by hand, and the two drifted until a run died
with a KeyError.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from feature_inventory import all_specs
from feature_spec import build_all, emitted_names
from prediction_row import (
    ALIASES,
    UNAVAILABLE,
    build_prediction_frame,
    required_suffixes,
)

SPECS = all_specs()
SUFFIXES = required_suffixes(SPECS)

# What a rebuilt fighter actually carries, as the stats builder produces it.
STATS_KEYS = set(
    "mmr_pre mu sigma wins losses draws dob stance splm str_acc sapm str_def "
    "td_avg td_def td_acc sub_avg kd momentum layoff streak won_last_fight "
    "last_fight_date height reach ko_rate sub_rate ko_losses been_finished "
    "absorption_eff footwork_proxy archetype cage_ctrl_fights cardio "
    "clinch_activity_ewm ctrl_rate_ewm grind_rate grind_score_ewm mom_quality "
    "opp_quality sub_def_score wc_move".split())

# What the prediction function computes rather than stores.
EXTRA_KEYS = {"age", "exp", "winrate", "southpaw", "consistency",
              "skill_conservative", "ape_index", "ring_rust", "prime_wc"}


def test_every_required_attribute_has_a_source():
    """Each one comes from the stats dict, the extras, or is documented as
    unavailable. Anything else would silently become a neutral value."""
    unexplained = [
        s for s in SUFFIXES
        if ALIASES.get(s, s) not in STATS_KEYS
        and s not in EXTRA_KEYS
        and s not in UNAVAILABLE
    ]
    assert unexplained == [], f"no source for: {unexplained}"


def test_every_unavailable_entry_says_why():
    for name, reason in UNAVAILABLE.items():
        assert len(reason) > 20, f"{name} is written off without a reason"


def test_unavailable_attributes_are_not_silently_invented():
    frame = build_prediction_frame({}, {}, ["won_L3"])
    assert pd.isna(frame.loc[0, "r_won_L3"])


def test_an_alias_reaches_the_right_key():
    """The specs read td_avg_acc; the stats dict calls it td_acc."""
    frame = build_prediction_frame({"td_acc": 42.0}, {"td_acc": 10.0},
                                   ["td_avg_acc"])
    assert frame.loc[0, "r_td_avg_acc"] == 42.0
    assert frame.loc[0, "b_td_avg_acc"] == 10.0


def test_extras_take_precedence_over_the_stats_dict():
    frame = build_prediction_frame({"age": 99.0}, {"age": 99.0}, ["age"],
                                   red_extra={"age": 30.0},
                                   blue_extra={"age": 28.0})
    assert frame.loc[0, "r_age"] == 30.0
    assert frame.loc[0, "b_age"] == 28.0


def test_a_missing_attribute_becomes_unknown_not_zero():
    """The defect this whole rebuild is about."""
    frame = build_prediction_frame({}, {"str_acc": 45.0}, ["str_acc"])
    assert pd.isna(frame.loc[0, "r_str_acc"])
    assert frame.loc[0, "b_str_acc"] == 45.0


def test_an_explicit_none_is_unknown_rather_than_zero():
    frame = build_prediction_frame({"str_acc": None}, {"str_acc": 45.0},
                                   ["str_acc"])
    assert pd.isna(frame.loc[0, "r_str_acc"])


def test_the_frame_has_one_row_with_both_corners():
    frame = build_prediction_frame({"splm": 4.0}, {"splm": 3.0}, ["splm"])
    assert len(frame) == 1
    assert set(frame.columns) == {"r_splm", "b_splm"}


# --- the guarantee that matters -------------------------------------------

def test_a_prediction_emits_exactly_the_training_features():
    """The same declaration, so a feature cannot exist on one side only."""
    stats = {k: 1.0 for k in STATS_KEYS}
    extra = {k: 1.0 for k in EXTRA_KEYS}
    frame = build_prediction_frame(stats, stats, SUFFIXES,
                                   red_extra=extra, blue_extra=extra)
    built = build_all(SPECS, frame)
    assert list(built.columns) == emitted_names(SPECS)


def test_a_prediction_with_nothing_known_still_builds_every_column():
    """An unknown fighter must produce a full, neutral row rather than raise."""
    frame = build_prediction_frame({}, {}, SUFFIXES)
    built = build_all(SPECS, frame)
    assert list(built.columns) == emitted_names(SPECS)
    known = [c for c in built.columns if c.endswith("_known")]
    assert known and all(built.iloc[0][c] == 0.0 for c in known)
