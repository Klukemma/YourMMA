"""Undefined names in the network-only scripts fail here, not on the runner.

check_fight_changes.py's main() cannot run in tests - it needs Wikipedia - so
a constant renamed by a find-and-replace that matched nothing sailed through
every test and crashed the probe on the runner with a NameError, one CI round
later. Static analysis catches that class of mistake without any network.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
NETWORK_ONLY = ["check_fight_changes.py", "fetch_fight_changes.py",
                "harvest_injuries.py",
                "intel_backfill.py", "verify_intel_dates.py",
                "intel_search.py", "fight_intel.py",
                "experiments/short_notice_weight.py"]

pyflakes = pytest.importorskip("pyflakes")


@pytest.mark.parametrize("script", NETWORK_ONLY)
def test_no_undefined_names(script):
    result = subprocess.run([sys.executable, "-m", "pyflakes",
                             str(ENGINE / script)],
                            capture_output=True, text=True)
    problems = [line for line in result.stdout.splitlines()
                if "undefined name" in line]
    assert not problems, "\n".join(problems)
