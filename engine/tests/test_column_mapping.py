"""Tests for matching upstream Kaggle column names to our local schema.

The two schemas describe the same numbers with different house style:

    local     r_head_landed
    upstream  r_total_sig_str_landed_head

These pin the normalisation so a future upstream rename fails loudly here
rather than silently dropping a feature during a sync.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

_spec = importlib.util.spec_from_file_location("sync_kaggle", ENGINE / "sync_kaggle.py")
sync_kaggle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_kaggle)
_canon = sync_kaggle._canon


@pytest.mark.parametrize("local,upstream", [
    ("r_name", "r_fighter_name"),
    ("b_name", "b_fighter_name"),
    ("r_nick_name", "r_fighter_nick_name"),
    ("r_reach", "r_reach_inches"),
    ("r_weight", "r_weight_lbs"),
    ("r_splm", "r_slpm"),
    ("r_ctrl", "r_total_ctrl_seconds"),
    ("r_td_landed", "r_total_td_success"),
    ("r_td_atmpted", "r_total_td_atmp"),
    ("r_sub_att", "r_total_sub_att"),
    ("r_sig_str_landed", "r_total_sig_landed"),
    ("r_total_str_landed", "r_total_str_landed"),
    # the positional breakdown, where upstream keeps a "sig" token we drop
    ("r_head_landed", "r_total_sig_str_landed_head"),
    ("r_head_atmpted", "r_total_sig_str_atmp_head"),
    ("b_body_landed", "b_total_sig_str_landed_body"),
    ("r_leg_atmpted", "r_total_sig_str_atmp_leg"),
    ("r_dist_landed", "r_total_sig_str_landed_distance"),
    ("b_clinch_atmpted", "b_total_sig_str_atmp_clinch"),
    ("r_ground_landed", "r_total_sig_str_landed_ground"),
])
def test_equivalent_columns_match(local, upstream):
    assert _canon(local) == _canon(upstream), f"{local} should match {upstream}"


@pytest.mark.parametrize("a,b", [
    # significant strikes are not total strikes
    ("r_sig_str_landed", "r_total_str_landed"),
    # corners must never collapse into each other
    ("r_head_landed", "b_head_landed"),
    # landed is not attempted
    ("r_head_landed", "r_head_atmpted"),
    # different target areas stay distinct
    ("r_head_landed", "r_body_landed"),
    ("r_dist_landed", "r_ground_landed"),
])
def test_distinct_columns_do_not_collide(a, b):
    assert _canon(a) != _canon(b), f"{a} must not match {b}"


def test_canon_cannot_separate_career_from_per_fight_accuracy():
    """Documents why the real mapping is hand-written.

    Locally r_str_acc is the fighter's career striking accuracy and
    r_total_str_acc is their accuracy in this bout. They differ by one "total"
    token, which normalisation discards, so a token matcher cannot tell them
    apart. This is the specific ambiguity that makes an automatic map unsafe.
    """
    assert _canon("r_str_acc") == _canon("r_total_str_acc")


def test_the_shipped_mapping_keeps_them_apart():
    """The explicit map resolves what the matcher cannot."""
    import schema_map as sm
    assert "r_str_acc" in sm.profile_columns("r")   # career, from fighter.csv
    assert "r_total_str_acc" in sm.DERIVED          # per bout, computed
    assert sm.profile_columns("r")["r_str_acc"] == "str_acc"


def test_canon_is_only_used_for_suggestions():
    """propose-map may use _canon; the sync path must not."""
    source = (ENGINE / "sync_kaggle.py").read_text()
    sync_body = source[source.index("def cmd_sync("):]
    assert "_canon" not in sync_body, "the sync path must use schema_map, not the matcher"
