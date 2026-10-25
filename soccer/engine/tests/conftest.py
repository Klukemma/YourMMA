"""Shared setup for the soccer engine's tests."""

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "snapshot: checks the health of the committed football-data.co.uk snapshot, "
                   "which depends on the site's publishing clock - reported by the runner, never a gate")
