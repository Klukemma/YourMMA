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


# ---------------------------------------------------------------------------
# training / prediction feature agreement
# ---------------------------------------------------------------------------

from audit_features import (  # noqa: E402
    declared_features,
    feature_group_lists,
    missing_prediction_features,
)

SAMPLE = '''
bayesian_features = ['mu_diff', 'mu_sum']
base_features = ['f00', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07',
                 'f08', 'f09', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
                 'f16', 'f17', 'f18', 'f19']
feature_cols = bayesian_features + base_features

def predict(r, b):
    feat = {
        'mu_diff': 1.0,
        'f00': 1.0, 'f01': 1.0, 'f02': 1.0, 'f03': 1.0, 'f04': 1.0,
        'f05': 1.0, 'f06': 1.0, 'f07': 1.0, 'f08': 1.0, 'f09': 1.0,
        'f10': 1.0, 'f11': 1.0, 'f12': 1.0, 'f13': 1.0, 'f14': 1.0,
        'f15': 1.0, 'f16': 1.0, 'f17': 1.0, 'f18': 1.0, 'f19': 1.0,
    }
    return feat
'''


def test_feature_groups_are_collected():
    groups = feature_group_lists(ast.parse(SAMPLE))
    assert groups["bayesian_features"] == ["mu_diff", "mu_sum"]


def test_declared_features_follows_the_concatenation():
    declared = declared_features(ast.parse(SAMPLE))
    assert {"mu_diff", "mu_sum", "f00", "f19"} <= declared
    assert len(declared) == 22


def test_a_feature_the_prediction_never_sets_is_caught(tmp_path):
    """The exact break: mu_sum added to training, absent from the dict."""
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE)
    findings = missing_prediction_features(path)
    assert len(findings) == 1
    assert findings[0]["missing"] == ["mu_sum"]


def test_a_matching_dict_is_not_flagged(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE.replace("'mu_diff': 1.0,\n", "'mu_diff': 1.0, 'mu_sum': 3.0,\n", 1))
    assert missing_prediction_features(path) == []


def test_an_unrelated_dict_is_ignored(tmp_path):
    """A big dict that is not a feature row must not be mistaken for one."""
    path = tmp_path / "sample.py"
    path.write_text(SAMPLE + "\nconfig = {'a%d' % i: i for i in range(30)}\n")
    assert len(missing_prediction_features(path)) == 1


def test_a_file_with_no_feature_list_reports_nothing(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("x = 1\n")
    assert missing_prediction_features(path) == []


def test_every_trained_feature_is_supplied_at_prediction_time():
    """Guard: a feature added to the model but not to the per-fight dict is a
    KeyError that only shows up ten minutes into a real run."""
    findings = missing_prediction_features(ENGINE / "predict_card.py")
    assert findings == [], f"prediction cannot supply: {findings}"
