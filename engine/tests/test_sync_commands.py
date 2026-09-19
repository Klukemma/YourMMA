"""Smoke tests for the sync_kaggle subcommands.

These exist because inspect-odds shipped with a missing `import json` and
failed in CI on its first line. py_compile does not catch a NameError and
nothing exercised the command, so the only feedback was a five-minute CI
round trip. Each command is now run locally with the network stubbed, which
catches that class of mistake in under a second.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

_spec = importlib.util.spec_from_file_location("sync_kaggle", ENGINE / "sync_kaggle.py")
sync_kaggle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_kaggle)


class _Failed:
    """A subprocess result standing in for an unreachable network."""
    returncode = 1
    stdout = ""
    stderr = "stubbed: no network in tests"


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(sync_kaggle.subprocess, "run", lambda *a, **k: _Failed())
    monkeypatch.setattr(sync_kaggle, "_have_token", lambda: True)
    return sync_kaggle


def test_inspect_odds_runs_without_network(offline, monkeypatch, capsys):
    """Regression: this raised NameError on its first statement."""
    monkeypatch.setattr(offline, "ODDS_CANDIDATES", ["someone/some-dataset"])
    offline.cmd_inspect_odds(None)
    out = capsys.readouterr().out
    assert "predictions to price" in out
    assert "download failed" in out


def test_search_odds_runs_without_network(offline, capsys):
    offline.cmd_search_odds(None)
    assert "candidate datasets" in capsys.readouterr().out


def test_every_subcommand_is_wired_to_a_real_function():
    """The dispatch table must not name a function that does not exist."""
    for name in ("cmd_inspect", "cmd_propose_map", "cmd_search_odds",
                 "cmd_inspect_odds", "cmd_sync"):
        assert callable(getattr(sync_kaggle, name)), f"{name} missing"


# Modes the workflow handles in shell rather than by calling sync_kaggle.
# Listed explicitly so a genuinely missing command still fails the test.
# "predict" runs predict_card.py, which is the only mode that produces a fresh
# card for the app and the only one that needs the odds key.
MODES_NOT_HANDLED_BY_SYNC_KAGGLE = {"experiment", "backtest", "predict"}


WORKFLOW = ENGINE.parent / ".github" / "workflows" / "update-dataset.yml"


def _workflow_inputs():
    """Parse the workflow properly.

    An earlier version split the raw text and picked up the cron schedule as
    if it were an experiment name.
    """
    import yaml
    doc = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the `on:` key as the boolean True.
    triggers = doc.get("on") or doc.get(True)
    return triggers["workflow_dispatch"]["inputs"]


def _workflow_modes():
    return list(_workflow_inputs()["mode"]["options"])


def test_parser_accepts_each_documented_mode(monkeypatch):
    """Every mode the workflow routes to sync_kaggle must parse and dispatch.

    Commands are stubbed by discovery rather than by name, so adding a new
    cmd_* does not silently leave this test calling the real one.
    """
    modes = [m for m in _workflow_modes() if m not in MODES_NOT_HANDLED_BY_SYNC_KAGGLE]
    assert modes, "no modes found in the workflow"

    commands = [n for n in dir(sync_kaggle) if n.startswith("cmd_")]
    for mode in modes:
        monkeypatch.setattr(sys, "argv", ["sync_kaggle.py", mode])
        called = {}
        for name in commands:
            monkeypatch.setattr(sync_kaggle, name,
                                lambda a, _n=name: called.setdefault("ran", _n))
        sync_kaggle.main()
        assert called, f"workflow offers mode {mode!r} but nothing ran"


def test_every_workflow_mode_has_a_command():
    """A mode in the dropdown with no matching cmd_ function fails at runtime."""
    for mode in _workflow_modes():
        if mode in MODES_NOT_HANDLED_BY_SYNC_KAGGLE:
            continue
        expected = "cmd_" + mode.replace("-", "_")
        assert hasattr(sync_kaggle, expected), f"mode {mode!r} needs {expected}()"


def test_each_named_experiment_exists():
    """The experiment dropdown must not name a script that is not there."""
    names = _workflow_inputs()["experiment"]["options"]
    assert names, "no experiments listed"
    for name in names:
        script = ENGINE / "experiments" / f"{name}.py"
        assert script.exists(), f"workflow offers {name!r} but {script} is missing"


# ---------------------------------------------------------------------------
# historical odds: the format check that stops a catastrophic misread
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from sync_kaggle import american_to_decimal, verify_american  # noqa: E402


def test_american_converts_to_decimal():
    assert american_to_decimal(pd.Series([-150.0]))[0] == pytest.approx(1.6667, abs=1e-3)
    assert american_to_decimal(pd.Series([130.0]))[0] == pytest.approx(2.30, abs=1e-3)
    assert american_to_decimal(pd.Series([100.0]))[0] == pytest.approx(2.00, abs=1e-3)


def test_a_genuine_american_column_verifies():
    american = pd.Series([-150.0, 130.0, -200.0, 250.0])
    decimal = pd.Series([1.6667, 2.30, 1.50, 3.50])
    agreement, n = verify_american(american, decimal)
    assert agreement == pytest.approx(1.0)
    assert n == 4


def test_decimal_odds_mislabelled_as_american_are_rejected():
    """Reading decimal prices as American would invert every favourite."""
    decimal = pd.Series([1.6667, 2.30, 1.50, 3.50])
    agreement, _ = verify_american(decimal, decimal)
    assert agreement < 0.5


def test_a_column_of_zeros_makes_no_claim():
    agreement, n = verify_american(pd.Series([0.0, 0.0]), pd.Series([1.5, 2.0]))
    assert n == 0 and agreement == 0.0


def test_missing_values_are_skipped_not_counted():
    american = pd.Series([-150.0, np.nan])
    decimal = pd.Series([1.6667, 2.30])
    agreement, n = verify_american(american, decimal)
    assert n == 1 and agreement == pytest.approx(1.0)


def test_fetch_odds_history_is_a_registered_command():
    assert hasattr(sync_kaggle, "cmd_fetch_odds_history")


def test_real_moneylines_are_detected_as_american():
    """The actual RedOdds values from valihameed/ufc-stats."""
    real = pd.Series([-250.0, -210.0, -380.0, -950.0, -130.0, 775.0, -150.0])
    assert sync_kaggle.detect_odds_format(real) == "american"


def test_real_decimal_prices_are_detected_as_decimal():
    assert sync_kaggle.detect_odds_format(
        pd.Series([1.40, 1.48, 2.30, 3.50, 1.67])) == "decimal"


def test_no_american_price_sits_between_minus_100_and_100():
    """The property the fetch guard leans on: an American price of -50 or +80
    cannot exist, while a decimal price is almost always in that band."""
    american = pd.Series([-250.0, 775.0, -130.0, 100.0, -100.0])
    inside = ((american > -100) & (american < 100) & (american != 0)).mean()
    assert inside == 0.0

    decimal = pd.Series([1.40, 2.30, 3.50])
    inside_dec = ((decimal > -100) & (decimal < 100) & (decimal != 0)).mean()
    assert inside_dec == 1.0


def test_the_prop_columns_are_named_and_excluded():
    """RedDecOdds is the DECISION prop, not a decimal conversion - it carries
    negative values. Comparing a moneyline against it gave 0% agreement and
    correctly refused the download."""
    assert "RedDecOdds" in sync_kaggle.PROP_ODDS_COLUMNS
    assert "RKOOdds" in sync_kaggle.PROP_ODDS_COLUMNS
    assert "RedOdds" not in sync_kaggle.PROP_ODDS_COLUMNS


# ---------------------------------------------------------------------------
# The odds key. It existed, the code to use it existed, and the workflow never
# passed it - so odds were empty on every run regardless of whether a secret
# was set. Nothing reported that, because an absent key is a supported state.
# ---------------------------------------------------------------------------

def test_the_workflow_passes_the_odds_key_to_the_runner():
    source = WORKFLOW.read_text()
    assert "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in source, (
        "the odds key is not passed through; the engine will fetch nothing")


def test_a_mode_exists_that_produces_a_card():
    """Every other mode syncs data or runs an experiment. Without this one the
    app's card only refreshes as a side effect of an experiment importing
    predict_card, which is not something to rely on."""
    assert "predict" in _workflow_modes()
    source = WORKFLOW.read_text()
    assert "python engine/predict_card.py" in source


def test_a_missing_odds_key_warns_rather_than_failing_the_run():
    """A card with no odds is still a card, and the model still predicts."""
    source = WORKFLOW.read_text()
    block = source[source.index('= "predict"'):]
    assert "::warning::" in block[:600]
    assert "exit 1" not in block[:600]
