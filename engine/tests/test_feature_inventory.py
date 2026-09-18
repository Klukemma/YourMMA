"""Tests for the declared feature inventory.

The inventory is meant to make a whole class of defect impossible rather than
merely absent: a feature that names a column nobody produces, a feature
declared but never built, a duplicate wearing two names, or a feature with no
recorded purpose. Each of those happened in the hand-written version.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from feature_inventory import all_specs
from feature_spec import Derived, Interaction, Paired, build_all, emitted_names

SPECS = all_specs()
NAMES = emitted_names(SPECS)


def source_columns():
    needed = set()
    for spec in SPECS:
        if isinstance(spec, Paired):
            needed |= {spec.red, spec.blue}
        elif isinstance(spec, Interaction):
            needed |= {spec.red_attack, spec.blue_defence}
    return needed


# --- the inventory holds together -----------------------------------------

def test_no_feature_name_is_emitted_twice():
    """Two specs sharing a name means one silently overwrites the other."""
    duplicates = [n for n in set(NAMES) if NAMES.count(n) > 1]
    assert duplicates == []


def test_every_spec_records_what_it_is_for():
    """striking_trajectory_diff computed a constant for its whole life and
    nobody noticed, because nothing said what it was supposed to mean."""
    for spec in SPECS:
        assert spec.why and len(spec.why) > 15, f"{spec} has no usable reason"


def test_the_inventory_is_not_trivially_small():
    assert len(NAMES) > 60, "the rebuild should not quietly lose features"


# --- every source column is real ------------------------------------------

def test_every_source_column_is_produced_somewhere():
    """A feature naming a column nobody builds fails only at runtime."""
    dataset = set(pd.read_csv(
        ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv",
        nrows=5, low_memory=False).columns)
    source = (ENGINE / "predict_card.py").read_text()

    missing = []
    for col in sorted(source_columns()):
        if col in dataset:
            continue
        assigned = re.search(r"ufc\[.%s.\]\s*=" % re.escape(col), source)
        renamed = re.search(r"'\s*:\s*'%s'" % re.escape(col), source)
        if not (assigned or renamed):
            missing.append(col)
    assert missing == [], f"no column produces: {missing}"


def test_both_corners_are_referenced_for_every_paired_feature():
    """A spec pointing at the same column twice would emit a constant zero."""
    for spec in SPECS:
        if isinstance(spec, Paired):
            assert spec.red != spec.blue, f"{spec.name} compares a column to itself"


def test_paired_specs_use_matching_corner_prefixes():
    for spec in SPECS:
        if isinstance(spec, Paired):
            assert spec.red.startswith("r_"), spec.name
            assert spec.blue.startswith("b_"), spec.name
            assert spec.red[2:] == spec.blue[2:], (
                f"{spec.name} pairs {spec.red} with {spec.blue}")


# --- interactions ----------------------------------------------------------

def test_every_interaction_has_its_mirror():
    """The corners are not symmetric, so a one-sided crossing biases a corner."""
    def swap_prefix(part):
        if part.startswith("r_"):
            return "b_" + part[2:]
        if part.startswith("b_"):
            return "r_" + part[2:]
        return part

    names = {s.name for s in SPECS if isinstance(s, Interaction)}
    for name in names:
        # Swap only the corner prefix of each half; a blanket replace turns
        # "sub_" into "sur_" because it contains "b_".
        left, _, right = name.partition("_vs_")
        mirror = f"{swap_prefix(left)}_vs_{swap_prefix(right)}"
        assert mirror in names, f"{name} has no mirror ({mirror})"


def test_interactions_cross_the_corners():
    """An interaction within one corner is a main effect wearing a disguise."""
    for spec in SPECS:
        if isinstance(spec, Interaction):
            assert spec.red_attack.startswith("r_") != spec.blue_defence.startswith("r_"), \
                f"{spec.name} does not cross corners"


# --- it builds -------------------------------------------------------------

def frame(rows=4):
    rng = np.random.default_rng(0)
    data = {}
    for col in source_columns():
        data[col] = rng.normal(50, 10, rows)
    for col in ("r_splm_L3", "b_splm_L3", "r_str_acc_L3", "b_str_acc_L3",
                "r_splm", "b_splm", "r_str_acc", "b_str_acc",
                "r_striking_trajectory", "b_striking_trajectory",
                "r_accuracy_trajectory", "b_accuracy_trajectory"):
        data.setdefault(col, rng.normal(50, 10, rows))
    return pd.DataFrame(data)


def test_building_emits_exactly_the_declared_names():
    built = build_all(SPECS, frame())
    assert list(built.columns) == NAMES


def test_a_single_row_produces_the_same_columns():
    """Training rows and one fight go through the same code by construction."""
    many = build_all(SPECS, frame(8))
    one = build_all(SPECS, frame(8).iloc[[0]])
    assert list(many.columns) == list(one.columns)


def test_a_missing_stat_never_becomes_a_maximal_disadvantage():
    """The defect that hit fifteen features: 0 - 45 = -45 for a debutant."""
    df = frame()
    df.loc[0, "r_str_acc"] = np.nan
    built = build_all(SPECS, df)
    assert pd.isna(built.loc[0, "acc_diff"])
    assert built.loc[0, "acc_known"] == 0.0
    assert built.loc[1, "acc_known"] == 1.0
