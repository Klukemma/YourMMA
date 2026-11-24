"""The runner: the NFL workflow's wiring, read as text."""

import re
from pathlib import Path

import pandas as pd
import pytest

import nfl_odds as no

WF = Path(__file__).resolve().parents[3] / ".github" / "workflows"
DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
# GitHub starts scheduled runs late under load: the first Tuesday run began
# five hours after its slot (the workflow's own note), so a game's TD prices
# need a slot at least this far ahead of kickoff.
LATE_HOURS = 5


def _months(spec):
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def _slots(start, end):
    """Every scheduled run's start, naive UTC as the schedule's kickoffs are,
    from the workflow's crons ("M H * months DAY") between two days."""
    crons = re.findall(r'- cron: "([^"]+)"', (WF / "nfl.yml").read_text())
    out = []
    for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()):
        for cron in crons:
            minute, hour, dom, months, dow = cron.split()
            assert dom == "*", cron
            if day.month in _months(months) and DAYS[day.day_of_week] == dow:
                out.append(day + pd.Timedelta(hours=int(hour), minutes=int(minute)))
    return sorted(out)


def _without_a_td_run(kickoffs):
    """The kickoffs no slot buys anytime-TD prices for in time: none starts
    within nfl_odds.TD_WINDOW_HOURS of kickoff and LATE_HOURS or more ahead
    of it (nfl_odds.td_due's window, read at a run's start)."""
    kickoffs = [pd.Timestamp(k) for k in kickoffs]
    slots = _slots(min(kickoffs) - pd.Timedelta(days=3), max(kickoffs))
    window, late = pd.Timedelta(hours=no.TD_WINDOW_HOURS), pd.Timedelta(hours=LATE_HOURS)
    return [k for k in kickoffs if not any(k - window <= t <= k - late for t in slots)]


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
    for day in ("TUE", "WED", "THU", "SAT", "SUN", "MON"):
        assert f"9-12,1-2 {day}" in nfl, day


# One kickoff of each day and time the 2026 schedule and the 2025 playoffs
# use (UTC): the opener and the Thanksgiving eve game on a Wednesday night,
# Thanksgiving, Black Friday, Christmas, London, Saturday, Sunday, Monday.
KICKOFF_KINDS = ("2026-09-10 00:20", "2026-09-11 00:35", "2026-09-13 17:00", "2026-09-13 20:25",
                 "2026-09-14 00:20", "2026-09-15 00:15", "2026-09-18 00:15", "2026-09-20 20:05",
                 "2026-10-04 13:30", "2026-11-01 18:00", "2026-11-01 21:05", "2026-11-01 21:25",
                 "2026-11-02 01:20", "2026-11-03 01:15", "2026-11-06 01:15", "2026-11-08 14:30",
                 "2026-11-26 01:00", "2026-11-26 18:00", "2026-11-26 21:30", "2026-11-27 01:20",
                 "2026-11-27 20:00", "2026-12-19 22:00", "2026-12-20 01:20", "2026-12-25 18:00",
                 "2026-12-25 21:30", "2026-12-26 01:15", "2026-01-03 21:30", "2026-01-10 21:30",
                 "2026-01-11 01:15", "2026-01-12 01:15")


def test_every_kind_of_kickoff_gets_a_run_inside_the_td_window():
    """A game with no run within TD_WINDOW_HOURS of kickoff has its tickets
    logged with every TD leg unpriced, and one unpriced graded ticket blanks
    its kind's live return. GB @ LA on Thanksgiving eve (01:00 UTC
    Thursday) had none: Tuesday's run is 38.7 hours ahead, Thursday's after it."""
    assert _without_a_td_run(KICKOFF_KINDS) == []


@pytest.mark.snapshot
def test_every_game_of_the_season_on_file_gets_a_run_inside_the_td_window(games):
    """The same over the newest season's schedule as nflverse has it now -
    a snapshot check, so a game moved to a new slot warns in the snapshot
    step and never gates a run."""
    season = games[games["season"] == games["season"].max()]
    assert _without_a_td_run(season["kickoff"]) == []


def test_a_failed_run_keeps_the_prices_it_paid_for():
    """An exception after the purchase used to lose the paid-for odds, and the
    next run bought them again."""
    text = (WF / "nfl.yml").read_text()
    step = text.split("- name: Keep the prices a failed run paid for")[1].split("- name:")[0]
    assert "if: failure() && (inputs.mode || 'predict') == 'predict'" in step
    assert "nfl/data/odds_cache.json" in step and "git push" in step
    assert text.index("- name: Run") < text.index("- name: Keep the prices a failed run paid for")


def test_ncaa_sync_is_a_mode_with_the_full_flag_and_the_token():
    """College football's data step: its own mode, the same full flag, the
    token only for GitHub's API, and its files under nfl/data/ - which the
    Commit step already takes."""
    text = (WF / "nfl.yml").read_text()
    assert "options: [predict, sync, backtest, ncaa-sync, probe-ncaa, probe-pbp]" in text
    run = text.split("- name: Run")[1].split("- name:")[0]
    step = run.split('"$MODE" = "ncaa-sync"')[1].split("elif")[0]
    assert "python nfl/engine/ncaa_data.py ${{ inputs.full && '--full' || '' }}" in step
    assert 'GITHUB_TOKEN="${{ github.token }}"' in step and "secrets." not in step
    assert 'PATHS="nfl/data/ ' in text
    commit = text.split("- name: Commit")[1]
    assert '"${{ inputs.mode }}" = "ncaa-sync"' in commit


def test_sync_mode_refreshes_the_coaching_approach_with_the_full_flag():
    """The play-by-play rows (nfl_approach.py) ride the sync mode, every
    season with full=true, after the nflverse snapshot - and a failure there
    warns instead of throwing that snapshot away (the Commit step would not
    run). The predict mode refreshes this season inside nfl_predict.run."""
    text = (WF / "nfl.yml").read_text()
    run = text.split("- name: Run")[1].split("- name:")[0]
    step = run.split('if [ "$MODE" = "sync" ]; then')[1].split("elif")[0]
    data = step.index("python nfl/engine/nfl_data.py ${{ inputs.full && '--full' || '' }}")
    approach = step.index("python nfl/engine/nfl_approach.py ${{ inputs.full && '--full' || '' }}")
    assert data < approach
    assert '|| echo "::warning::' in step[approach:]
    assert "tee -a nfl.log" in step[approach:]
    predict = (Path(__file__).resolve().parents[1] / "nfl_predict.py").read_text()
    assert "nap.sync(log=log)" in predict.split("def run(")[1].split("games = nd.load_games()")[0]


def test_the_ticket_history_is_replayed_when_stale_before_each_predict_run():
    """The same-game tickets' backtest (~30 s) runs in the predict mode only
    when a final game's box score is not in it yet, before nfl_predict.py
    reads it - and a failure warns rather than costing the week's picks."""
    run = (WF / "nfl.yml").read_text().split("- name: Run")[1].split("- name:")[0]
    predict = run.split("export ODDS_REFRESH=1")[1]
    replay = predict.index("python nfl/engine/experiments/nfl_sgp_backtest.py --if-stale")
    picks = predict.index("python nfl/engine/nfl_predict.py")
    assert replay < picks
    assert '|| echo "::warning::' in predict[replay:picks]
    assert "tee -a nfl.log" in predict[picks:]
