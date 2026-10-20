"""The runner: the NFL workflow's wiring, read as text."""

from pathlib import Path

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def test_it_has_its_own_queue():
    """A shared concurrency group lets a new run cancel the other workflow's
    pending one; the two touch different files and push with a retry."""
    nfl = (WF / "nfl.yml").read_text()
    mma = (WF / "update-dataset.yml").read_text()
    assert "group: nfl-${{ github.ref }}" in nfl and "cancel-in-progress: false" in nfl
    assert "group: nfl-" not in mma
    assert "git pull --rebase --autostash" in nfl


def test_snapshot_health_never_gates_a_run():
    nfl = (WF / "nfl.yml").read_text()
    assert 'pytest nfl/engine/tests -q -m "not snapshot"' in nfl
    snap = nfl.split("- name: Snapshot checks")[1].split("- name:")[0]
    assert "continue-on-error: true" in snap


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
    for cron in ('"17 12 * 9-12,1-2 THU"', '"17 6 * 9-12,1-2 SUN"'):
        assert f"cron: {cron}" in nfl and f"github.event.schedule }}}}\" = {cron}" in nfl


def test_every_game_day_has_a_run_before_it():
    nfl = (WF / "nfl.yml").read_text()
    for day in ("TUE", "THU", "SAT", "SUN", "MON"):
        assert f"9-12,1-2 {day}" in nfl, day
