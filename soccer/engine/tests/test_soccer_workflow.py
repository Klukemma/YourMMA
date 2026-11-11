"""The runner: the soccer workflow's wiring, read as text."""

import re
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
    assert 'PATHS="soccer/data/ soccer/app/data/ soccer/app/yoursoccer.html soccer/engine/tests/fixtures/feeds/"' in soccer
    assert "python soccer/engine/soccer_predict.py" in soccer


def test_runs_follow_the_fixtures_and_the_results():
    """Fixtures with prices arrive Friday and Tuesday afternoons; results
    after each round - in every month: the calendar-year leagues play in
    June and July."""
    soccer = (WF / "soccer.yml").read_text()
    for day in ("FRI", "SAT", "SUN", "MON", "TUE", "WED", "THU"):
        assert f"* * {day}" in soccer, day


def _crons(text):
    return re.findall(r'- cron: "([^"]+)"', text)


def test_the_feeds_bring_results_the_evening_they_happen_and_the_next_round_early():
    """The free feeds have each score that evening and the next rounds all
    week: a run on Saturday and Sunday evenings, and one on Wednesday
    morning after Tuesday's games - and every run there was before is still
    there, now in every month. The evening runs start at 22:17 UTC: a 21:00
    CET kickoff (20:00 UTC all winter) ends about 21:55."""
    crons = _crons((WF / "soccer.yml").read_text())
    for cron in ("17 22 * * SAT", "17 22 * * SUN", "17 7 * * WED"):
        assert cron in crons, cron
    for cron in crons:
        minute, hour, *_ = cron.split()
        assert not (int(hour) == 21 or (int(hour) == 22 and int(minute) < 10)), cron   # never before full time
    before = ["47 13 * * FRI", "47 16 * * FRI", "17 7 * * SAT", "17 7 * * SUN", "17 9 * * MON", "47 13 * * TUE",
              "47 16 * * TUE", "17 9 * * THU", "17 8 20-31 12 *", "17 8 1-3 1 *"]
    assert set(before) <= set(crons) and len(crons) == len(set(crons))


def test_every_month_is_run_for_the_calendar_year_leagues():
    """ARG BRA CHN FIN IRL NOR SWE USA play through June and July: no weekly
    slot is limited to Europe's season (only the Christmas ones name a
    month)."""
    import soccer_data as sd
    calendar = {c for c, m in sd.ALL_LEAGUES.items() if m.calendar == "calendar-year"}
    assert calendar == {"ARG", "BRA", "CHN", "FIN", "IRL", "NOR", "SWE", "USA"}
    for cron in _crons((WF / "soccer.yml").read_text()):
        minute, hour, dom, month, dow = cron.split()
        assert month == "*" or (dom, month) in (("20-31", "12"), ("1-3", "1")), cron
    assert "8-12,1-5" not in (WF / "soccer.yml").read_text()


def _utc_hour(cron):
    minute, hour, *_ = cron.split()
    return int(hour) + int(minute) / 60


# GitHub started every scheduled run from Oct 26 to Nov 5 between 3 and 10 h
# late (soccer.yml's notes): a slot is for a kickoff only if it lands before
# it even 10 h late.
LATE_HOURS = 10


def test_the_americas_evenings_and_asias_mornings_have_a_run_before_kickoff_every_day():
    """MLS, Liga MX, Brazil and Argentina kick off about 22:00-04:00 UTC, the
    J1 League and the Chinese Super League about 04:00-12:00 UTC the next
    day, on most days of the week: a daily slot each that lands before those
    kickoffs, however late GitHub starts it."""
    daily = [c for c in _crons((WF / "soccer.yml").read_text()) if c.split()[2:] == ["*", "*", "*"]]
    americas = [c for c in daily if _utc_hour(c) + LATE_HOURS <= 22]
    asia = [c for c in daily if 12 <= _utc_hour(c) and _utc_hour(c) + LATE_HOURS - 24 <= 4]
    assert americas == ["37 11 * * *"] and asia == ["37 17 * * *"]
    text = (WF / "soccer.yml").read_text()
    assert "Americas' evenings" in text and "Asia's mornings" in text


def test_the_predict_run_syncs_the_feeds_and_commits_their_snapshot():
    """soccer_predict's sync is football-data.co.uk and then the feeds
    (soccer_feed.sync, which writes soccer/data/feed.json); the commit step
    adds soccer/data/, so the feeds' snapshot goes in with the picks. A feed
    that fails is caught inside the run, never the step's exit code."""
    import inspect

    import soccer_feed
    import soccer_predict

    src = inspect.getsource(soccer_predict.sync_all)
    assert "sd.sync(" in src and "sfeed.sync(" in src and "except Exception" in src
    assert soccer_feed.FEED.parent.name == "data" and soccer_feed.FEED.parent.parent.name == "soccer"
    soccer = (WF / "soccer.yml").read_text()
    paths = re.search(r'PATHS="([^"]+)"', soccer).group(1).split()
    assert "soccer/data/" in paths
    run = soccer.split("- name: Run")[1].split("- name:")[0]
    assert "python soccer/engine/soccer_predict.py 2>&1" in run and "--no-sync" not in run


def test_a_friday_serie_a_kickoff_at_half_past_four_has_a_run_before_it():
    soccer = (WF / "soccer.yml").read_text()
    for day in ("FRI", "TUE"):
        assert f'"47 13 * * {day}"' in soccer, day
    assert '"17 8 20-31 12 *"' in soccer
