"""Shared fixtures for the NFL engine's tests: the snapshot, loaded once."""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import nfl_data as nd  # noqa: E402
import nfl_features as nf  # noqa: E402
import nfl_model as nm  # noqa: E402


@pytest.fixture(scope="session")
def games():
    return nd.load_games()


@pytest.fixture(scope="session")
def team_games():
    return nd.load_team_games()


@pytest.fixture(scope="session")
def qb_games():
    return nd.load_qb_games()


@pytest.fixture(scope="session")
def built(games, team_games, qb_games):
    frame, state = nf.build_with_state(games, team_games, qb_games, nm.ELO)
    return frame, state
