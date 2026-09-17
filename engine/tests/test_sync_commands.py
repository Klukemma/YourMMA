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


def test_parser_accepts_each_documented_mode(monkeypatch):
    """Every mode the workflow offers must parse and reach a command.

    Commands are stubbed by discovery rather than by name, so adding a new
    cmd_* does not silently leave this test calling the real one.
    """
    workflow = (ENGINE.parent / ".github" / "workflows" / "update-dataset.yml").read_text()
    line = next(l for l in workflow.splitlines() if "options:" in l)
    modes = [m.strip() for m in line.split("[", 1)[1].rstrip("]").split(",")]
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
    workflow = (ENGINE.parent / ".github" / "workflows" / "update-dataset.yml").read_text()
    line = next(l for l in workflow.splitlines() if "options:" in l)
    modes = [m.strip() for m in line.split("[", 1)[1].rstrip("]").split(",")]
    for mode in modes:
        expected = "cmd_" + mode.replace("-", "_")
        assert hasattr(sync_kaggle, expected), f"mode {mode!r} needs {expected}()"
