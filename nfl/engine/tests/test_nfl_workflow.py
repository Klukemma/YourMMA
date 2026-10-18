"""The runner: the NFL workflow's wiring, read as text."""

from pathlib import Path

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def test_it_queues_with_the_mma_runner():
    nfl = (WF / "nfl.yml").read_text()
    mma = (WF / "update-dataset.yml").read_text()
    group = "group: update-dataset-${{ github.ref }}"
    assert group in nfl and group in mma


def test_the_key_comes_from_the_secret():
    assert "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in (WF / "nfl.yml").read_text()


def test_picks_prices_and_the_app_are_committed_after_the_tests():
    nfl = (WF / "nfl.yml").read_text()
    assert 'PATHS="nfl/data/ nfl/app/data/ nfl/app/yournfl.html"' in nfl
    # Tests gate the code before the run, never the commit of what it bought.
    assert nfl.index("python -m pytest nfl/engine/tests") < nfl.index("- name: Run")
    assert "set -o pipefail" in nfl


def test_the_refresh_schedules_are_the_ones_named_in_the_run_step():
    nfl = (WF / "nfl.yml").read_text()
    for cron in ('"17 16 * 9-12,1-2 THU"', '"47 12 * 9-12,1-2 SUN"'):
        assert f"cron: {cron}" in nfl and f"github.event.schedule }}}}\" = {cron}" in nfl
