"""The runner: the soccer workflow's wiring, read as text."""

from pathlib import Path

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def test_it_has_its_own_queue_and_retries_its_push():
    soccer = (WF / "soccer.yml").read_text()
    assert "group: soccer-${{ github.ref }}" in soccer and "cancel-in-progress: false" in soccer
    for other in ("nfl.yml", "update-dataset.yml"):
        assert "group: soccer-" not in (WF / other).read_text()
    assert "git pull --rebase --autostash" in soccer


def test_tests_gate_the_code_and_snapshot_health_never_gates_a_run():
    soccer = (WF / "soccer.yml").read_text()
    assert 'pytest soccer/engine/tests -q -m "not snapshot"' in soccer
    assert soccer.index('-m "not snapshot"') < soccer.index("- name: Run")
    snap = soccer.split("- name: Snapshot checks")[1].split("- name:")[0]
    assert "continue-on-error: true" in snap
    assert "set -o pipefail" in soccer


def test_the_picks_and_the_app_are_committed():
    soccer = (WF / "soccer.yml").read_text()
    assert 'PATHS="soccer/data/ soccer/app/data/ soccer/app/yoursoccer.html"' in soccer
    assert "python soccer/engine/soccer_predict.py" in soccer


def test_runs_follow_the_fixtures_and_the_results():
    """Fixtures with prices arrive Friday and Tuesday afternoons; results
    after each round."""
    soccer = (WF / "soccer.yml").read_text()
    for day in ("FRI", "SAT", "SUN", "MON", "TUE", "THU"):
        assert f"8-12,1-5 {day}" in soccer, day
