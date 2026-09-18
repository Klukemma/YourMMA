"""Tests for the constant-scale audit.

The audit's job is to catch a class of bug that reads perfectly well and never
raises: a literal left behind after the quantity it compares against changed
scale. The tests that matter are the ones proving it catches the five real
cases, so they are written against the shapes those actually took.
"""

import ast
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from audit_features import audit, column_of, column_ranges, default_of, is_benign

# The real ranges these columns take, so the tests describe the real bug.
RANGES = {"r_mmr_pre": (-4.36, 37.64), "b_mmr_pre": (-4.12, 37.35),
          "r_mu_pre": (13.66, 48.54), "match_time_sec": (5.0, 1500.0)}


def check(source, tmp_path, include_benign=False):
    path = tmp_path / "sample.py"
    path.write_text(source)
    return audit(path, RANGES, include_benign=include_benign)


# --- the bug this exists for ----------------------------------------------

def test_the_loss_scale_threshold_is_caught(tmp_path):
    """The real line: every branch false, so it returned 1.0 for every fight."""
    src = ("opp = row.get('r_mmr_pre', 1500)\n"
           "loss_scale = 0.3 if opp > 1600 else (0.5 if opp > 1550 else "
           "(0.7 if opp > 1450 else 1.0))\n")
    dead, _ = check(src, tmp_path)
    assert {f["threshold"] for f in dead} == {1600.0, 1550.0, 1450.0}
    assert all(f["column"] == "r_mmr_pre" for f in dead)


def test_the_fillna_default_is_caught(tmp_path):
    src = "x = ufc['r_mmr_pre'].fillna(1500) - ufc['b_mmr_pre'].fillna(1500)\n"
    _, defaults = check(src, tmp_path)
    assert len(defaults) == 2
    assert {f["default"] for f in defaults} == {1500.0}


def test_a_get_default_is_caught(tmp_path):
    src = "v = row.get('r_mmr_pre', 1500)\n"
    _, defaults = check(src, tmp_path)
    assert defaults[0]["default"] == 1500.0


# --- it must not cry wolf --------------------------------------------------

def test_a_threshold_inside_the_range_is_not_flagged(tmp_path):
    """36.3 is the mu p90 - a real threshold on the real scale."""
    src = "v = row.get('r_mu_pre', 25.0)\nscale = 0.3 if v > 36.3 else 1.0\n"
    dead, defaults = check(src, tmp_path)
    assert dead == [] and defaults == []


def test_a_default_inside_the_range_is_not_flagged(tmp_path):
    src = "x = ufc['r_mmr_pre'].fillna(0.0)\n"
    _, defaults = check(src, tmp_path)
    assert defaults == []


def test_an_unknown_column_is_ignored(tmp_path):
    """No range to check against means no claim, rather than a false alarm."""
    src = "v = row.get('not_a_real_column', 1500)\nx = 1 if v > 99999 else 0\n"
    dead, defaults = check(src, tmp_path)
    assert dead == [] and defaults == []


def test_a_threshold_below_the_range_is_caught(tmp_path):
    src = "v = ufc['r_mu_pre']\nx = 1 if v < 0 else 0\n"
    dead, _ = check(src, tmp_path)
    assert dead[0]["threshold"] == 0.0
    assert dead[0]["why"] == "never below"


def test_negative_literals_are_read_correctly(tmp_path):
    src = "v = ufc['r_mu_pre']\nx = 1 if v < -50 else 0\n"
    dead, _ = check(src, tmp_path)
    assert dead[0]["threshold"] == -50.0


def test_a_boolean_is_not_treated_as_a_number(tmp_path):
    src = "v = ufc['r_mu_pre']\nx = 1 if v > True else 0\n"
    dead, _ = check(src, tmp_path)
    assert dead == []


# --- helpers ---------------------------------------------------------------

def test_column_of_reads_the_access_shapes_used_here():
    for src, expected in [
        ("df['r_mu_pre']", "r_mu_pre"),
        ("row.get('r_mu_pre')", "r_mu_pre"),
        ("row.get('r_mu_pre', 25)", "r_mu_pre"),
        ("ufc['r_mu_pre'].fillna(0)", "r_mu_pre"),
    ]:
        assert column_of(ast.parse(src, mode="eval").body) == expected, src


def test_default_of_reads_both_forms():
    assert default_of(ast.parse("x.fillna(1500)", mode="eval").body) == 1500.0
    assert default_of(ast.parse("r.get('c', 1500)", mode="eval").body) == 1500.0
    assert default_of(ast.parse("r.get('c')", mode="eval").body) is None


# --- the allowlist ---------------------------------------------------------

def test_an_accepted_finding_is_suppressed_but_still_detectable(tmp_path):
    src = "t = row.get('match_time_sec', 300)\nx = 1 if t <= 0 else 0\n"
    assert check(src, tmp_path)[0] == []
    assert check(src, tmp_path, include_benign=True)[0] != []


def test_every_allowlisted_entry_states_a_reason():
    from audit_features import KNOWN_BENIGN

    for key, reason in KNOWN_BENIGN.items():
        assert isinstance(reason, str) and len(reason) > 20, (
            f"{key} is suppressed without explaining why")


# --- the regression guard --------------------------------------------------

def test_the_pipeline_has_no_unexplained_scale_constants():
    """Fails if another Elo-era constant is introduced."""
    dead, defaults = audit(ENGINE / "predict_card.py", column_ranges())
    assert dead == [], f"dead thresholds: {dead}"
    assert defaults == [], f"defaults outside the column range: {defaults}"
