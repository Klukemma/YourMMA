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


# GitHub's start delays (hours) of the soccer runner's scheduled runs, Oct 26 -
# Nov 11 2026 (gh run list --workflow soccer.yml --event schedule: the 26 runs,
# each against its slot, in order) - every one late. By slot ("minute hour").
OBSERVED_LATE = {"17 7": [6.7, 6.3, 7.0, 8.1, 6.7, 7.1, 7.9], "17 9": [6.9, 7.0, 10.2, 8.2, 10.6],
                 "47 13": [5.2, 5.1, 6.5, 6.3, 6.8], "47 16": [4.2, 4.4, 5.3, 5.0, 5.2], "17 22": [3.0, 3.2, 3.2, 3.4]}
WORST = max(max(v) for v in OBSERVED_LATE.values())
# When a pick closes (soccer_schedule CLOSE): at kickoff where the time is
# confirmed (both feeds agree, or football-data.co.uk's own fixture), else at
# 00:00 UK on match day - every match of a league only ESPN covers is listed
# ten days ahead unconfirmed and picked by the model alone, so it closes then.
AMERICAS_KICKOFF = 22.0         # UTC: the earliest of MLS's, Liga MX's, Brazil's and Argentina's evening kickoffs
UK_MIDNIGHT = {"winter": 24.0, "summer": 23.0}   # 00:00 UK on match day, in UTC hours of the day before


def _neighbours(cron):
    """The observed delays of the slots either side of a daily one."""
    t = _utc_hour(cron)
    seen = sorted(OBSERVED_LATE, key=_utc_hour)
    before = [k for k in seen if _utc_hour(k + " * * *") < t][-1:]
    after = [k for k in seen if _utc_hour(k + " * * *") > t][:1]
    return [d for k in before + after for d in OBSERVED_LATE[k]]


def test_the_daily_slots_add_margin_and_the_comments_say_no_slot_can_guarantee_a_deadline():
    """The real deadlines, against GitHub's measured delays: the Americas'
    slot starts before a 22:00 UTC kickoff at its neighbours' usual delays,
    Asia's before 00:00 UK in winter - but neither under the worst delay
    seen (10.6 h, Mon Nov 9's 09:17 slot), so the workflow must not claim
    "however late it starts", and says so."""
    text = (WF / "soccer.yml").read_text()
    daily = [c for c in _crons(text) if c.split()[2:] == ["*", "*", "*"]]
    assert daily == ["37 11 * * *", "37 17 * * *"]
    assert WORST == 10.6 and min(min(v) for v in OBSERVED_LATE.values()) == 3.0
    americas, asia = (_utc_hour(c) for c in daily)
    # margin: at the median delay of the slots either side, each is in time
    assert americas + sorted(_neighbours(daily[0]))[len(_neighbours(daily[0])) // 2] < AMERICAS_KICKOFF
    assert asia + sorted(_neighbours(daily[1]))[len(_neighbours(daily[1])) // 2] < UK_MIDNIGHT["winter"]
    # no guarantee: at the worst delay seen, neither is
    assert americas + WORST > AMERICAS_KICKOFF and asia + WORST > UK_MIDNIGHT["winter"]
    # ... and the workflow says so, with the measured numbers, never "however late"
    assert "however late" not in text and "no slot can guarantee" in text.lower()
    for said in ("3.0 to 10.6 hours", "Mon Nov 9's started at", "00:00 UK time on", "18 leagues only ESPN covers",
                 "Americas' evenings", "Asia's mornings"):
        assert said in " ".join(line.lstrip("# ").strip() for line in text.splitlines()), said


def test_a_run_where_only_the_stamps_changed_commits_nothing():
    """only_stamps.py puts back the files whose only change is when the run
    happened, before the commit step looks at git status - and over the
    same paths the step commits."""
    soccer = (WF / "soccer.yml").read_text()
    commit = soccer.split("- name: Commit")[1]
    assert "python soccer/engine/only_stamps.py $PATHS" in commit
    assert commit.index("only_stamps.py") < commit.index("git status --porcelain")
    import only_stamps
    assert only_stamps.PAGE == "soccer/app/yoursoccer.html" and only_stamps.ROOT == WF.parents[1]
    for p in only_stamps.STAMPED:
        assert any(p.startswith(d) for d in re.search(r'PATHS="([^"]+)"', soccer).group(1).split()), p


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


def test_the_friday_and_tuesday_slots_are_set_for_a_half_past_four_kickoff():
    """Set for a 16:30 UTC Friday kickoff in Serie A - which GitHub's delays
    (13:47 started 5.1-6.8 h late) have made it miss so far, as the
    workflow says."""
    soccer = (WF / "soccer.yml").read_text()
    assert min(OBSERVED_LATE["47 13"]) + 13 + 47 / 60 > 16.5 and "made it miss so far" in soccer
    for day in ("FRI", "TUE"):
        assert f'"47 13 * * {day}"' in soccer, day
    assert '"17 8 20-31 12 *"' in soccer
