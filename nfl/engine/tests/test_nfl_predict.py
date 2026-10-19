"""The weekly run: which week, picks logged before kickoff only, grading,
the ruled-out quarterback, and the files the app reads."""

import json
import math

import pandas as pd
import pytest

import nfl_predict as npd
import nfl_standalone


def _schedule(played_through):
    """A two-week season and the start of the playoffs: Thursday, Sunday,
    Monday games, played up to `played_through` (inclusive)."""
    rows = [("w1_thu", 2025, 1, "REG", "2025-09-11 00:15"), ("w1_sun", 2025, 1, "REG", "2025-09-14 17:00"),
            ("w1_mon", 2025, 1, "REG", "2025-09-16 00:15"), ("w2_thu", 2025, 2, "REG", "2025-09-19 00:15"),
            ("w2_sun", 2025, 2, "REG", "2025-09-21 17:00"), ("wc", 2025, 19, "WC", "2026-01-10 21:30")]
    g = pd.DataFrame(rows, columns=["game_id", "season", "week", "game_type", "kickoff"])
    g["kickoff"] = pd.to_datetime(g["kickoff"])
    order = list(g["game_id"])
    g["played"] = [order.index(x) <= order.index(played_through) if played_through else False for x in g["game_id"]]
    return g


@pytest.mark.parametrize("now,played_through,expected", [
    ("2025-09-12 12:00", "w1_thu", (2025, 1)),     # between Thursday and Sunday
    ("2025-09-15 12:00", "w1_sun", (2025, 1)),     # Monday night still to come
    ("2025-09-16 12:00", "w1_mon", (2025, 2)),     # week 1 done: the next one
    ("2025-12-30 12:00", "w2_sun", (2025, 19)),    # regular season over: the playoffs
    ("2026-02-20 12:00", "wc", (2025, 19)),        # season over: the last week played
])
def test_the_week_is_the_one_with_the_next_kickoff(now, played_through, expected):
    season, week, _ = npd.current_week(_schedule(played_through), pd.Timestamp(now))
    assert (season, week) == expected


def test_a_quarterback_off_the_active_roster_cannot_start_or_stand_in():
    rosters = pd.DataFrame([
        {"season": 2026, "week": 4, "team": "NYG", "gsis_id": "dart", "status": "RES"},
        {"season": 2026, "week": 6, "team": "NYG", "gsis_id": "winston", "status": "ACT"},
        {"season": 2026, "week": 5, "team": "NYG", "gsis_id": "backup", "status": "INA"},
    ])
    assert npd.off_roster(rosters, 2026, 6) == {"dart"}      # last week's inactive is not this week's
    state = {"team_qbs": {"NYG": [(2026, 1, "dart", 30.0), (2026, 5, "winston", 30.0)]}}
    assert npd.stand_in(state, "NYG", 2026, {("winston", "Jameis Winston")}, set(), {}) == "dart"
    assert npd.stand_in(state, "NYG", 2026, {("winston", "Jameis Winston"), ("dart", None)}, set(), {}) is None


def _game(kickoff="2026-10-25T17:00:00Z", **kw):
    g = {"game_id": "2026_07_LV_KC", "season": 2026, "week": 7, "game_type": "REG",
         "kickoff": kickoff, "home": "KC", "away": "LV", "pick": "KC", "pick_prob": 0.8,
         "home_prob": 0.8, "model_prob": 0.78, "market_prob": 0.81, "market_source": "nflverse",
         "tier": "STRONG", "model_spread": 8.0, "market_spread": 9.5,
         "spread_pick": {"team": "LV", "line": 9.5, "edge": 1.5}, "model_total": 46.0,
         "market_total": 44.5, "total_pick": {"side": "OVER", "line": 44.5, "edge": 1.5}}
    g.update(kw)
    return g


def test_a_pick_is_logged_only_before_kickoff_and_keeps_its_first_time():
    log = {"schema": 1, "picks": {}}
    early, later, after = (pd.Timestamp(t) for t in ("2026-10-20 12:00", "2026-10-25 12:00", "2026-10-25 17:00"))
    assert npd.log_pick(log, _game(), early)
    assert npd.log_pick(log, _game(pick_prob=0.81), later)
    assert not npd.log_pick(log, _game(pick_prob=0.99), after)
    entry = log["picks"]["2026_07_LV_KC"]
    assert entry["pick_prob"] == 0.81
    assert entry["first_logged"] == "2026-10-20T12:00:00Z" and entry["last_logged"] == "2026-10-25T12:00:00Z"


def _final(home, away):
    return pd.DataFrame([{"game_id": "2026_07_LV_KC", "played": True, "home_score": home, "away_score": away}])


@pytest.mark.parametrize("home,away,correct,ats,total", [
    (31, 10, True, False, False),     # KC by 21: covers 9.5; 41 under 44.5
    (20, 17, True, True, False),      # KC by 3: LV +9.5 covers
    (17, 20, False, True, False),
    (27, 27, None, True, True),       # a tie voids the pick; LV +9.5 covers; 54 over
])
def test_grading(home, away, correct, ats, total):
    log = {"schema": 1, "picks": {"2026_07_LV_KC": dict(_game())}}
    assert npd.grade(log, _final(home, away)) == 1
    p = log["picks"]["2026_07_LV_KC"]
    assert p["correct"] == correct and p["spread_correct"] == ats and p["total_correct"] == total
    assert npd.grade(log, _final(home, away)) == 0   # graded once


def test_a_ruled_out_starter_is_replaced_by_the_last_one(built, games):
    frame, state = built
    row = frame[frame["played"]].iloc[-1].copy()
    names = npd._qb_names(games)
    starter = row["home_qb_used"]
    injuries = pd.DataFrame([{"season": row["season"], "week": row["week"], "team": row["home"],
                              "gsis_id": starter, "position": "QB", "full_name": names.get(starter, "X"),
                              "report_status": "Out", "report_primary_injury": "Ankle"}])
    team_qbs = {row["home"]: [(row["season"], row["kickoff"], "00-BACKUP", 20.0),
                              (row["season"], row["kickoff"], starter, 30.0)]}
    state = dict(state, team_qbs=team_qbs)
    new, replaced = npd.replace_ruled_out(row, state, injuries, names, row["season"], row["week"])
    assert replaced == {"home": names.get(starter, starter)}
    assert new["home_qb_used"] == "00-BACKUP" and new["home_qb_source"] == "replaced"


def test_a_stand_in_must_be_this_seasons_and_not_someone_elses_starter():
    names = {"A": "Starter", "B": "Old Backup", "C": "Traded", "D": "Relief"}
    state = {"team_qbs": {"ATL": [(2025, 1, "B", 30.0), (2026, 2, "D", 12.0), (2026, 3, "C", 25.0)]}}
    out = {("A", "Starter")}
    assert npd.stand_in(state, "ATL", 2026, out, set(), names) == "C"
    assert npd.stand_in(state, "ATL", 2026, out, {"C"}, names) == "D"     # C starts elsewhere
    assert npd.stand_in(state, "ATL", 2026, out | {("D", "Relief")}, {"C"}, names) is None  # never 2025's B


@pytest.fixture(scope="module")
def run_output(tmp_path_factory, games):
    out = tmp_path_factory.mktemp("app")
    log = out / "log.json"
    now = games[games["played"]]["kickoff"].iloc[-1] + pd.Timedelta(hours=6)
    produced = npd.run(sync=False, now=now, out_dir=out, log_path=log, log=lambda *_: None,
                       cache_path=out / "odds_cache.json", api_key="")
    return out, log, produced


def test_the_run_writes_the_three_files_the_app_reads(run_output):
    out, _, produced = run_output
    for name in ("week", "teams", "performance"):
        text = (out / f"{name}.json").read_text()
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["schema"] == 1
    week = json.loads((out / "week.json").read_text())
    assert week["games"] and {"label", "season", "caveats", "tiers"} <= set(week)
    for g in week["games"]:
        assert g["status"] in ("final", "in progress", "upcoming")
        assert {"home_form", "away_form", "home_injuries", "away_injuries"} <= set(g)
        if g["status"] == "upcoming":
            assert 0.5 <= g["pick_prob"] < 1 and g["pick"] in (g["home"], g["away"])
            assert g["tier"] in ("STRONG", "SOLID", "LEAN", "TOSS-UP")
        elif not g.get("logged_at"):
            assert g["pick"] is None    # no pick invented after kickoff
    teams = json.loads((out / "teams.json").read_text())
    assert len(teams["teams"]) == 32


def test_every_upcoming_pick_is_logged(run_output):
    _, log, produced = run_output
    picks = json.loads(log.read_text())["picks"]
    upcoming = [g for g in produced if g["status"] == "upcoming"]
    assert {g["game_id"] for g in upcoming} <= set(picks)


def test_the_standalone_page_carries_the_data(run_output, tmp_path):
    out, _, _ = run_output
    page = nfl_standalone.build(tmp_path / "x.html", out)
    html = page.read_text()
    assert "window.__YOURNFL_DATA__" in html and 'type="module"' not in html
    assert 'rel="manifest"' not in html


def test_upcoming_games_carry_the_likeliest_scorers(run_output):
    out, _, produced = run_output
    week = json.loads((out / "week.json").read_text())
    for g in week["games"]:
        if g["status"] != "upcoming":
            continue
        for side in ("home_td", "away_td"):
            picks = g[side]
            assert 0 < len(picks) <= npd.TD_SHOWN
            probs = [p["prob"] for p in picks]
            assert probs == sorted(probs, reverse=True) and all(0 < p < 1 for p in probs)
            assert all(p["status"] != "out" for p in picks)
    perf = json.loads((out / "performance.json").read_text())
    assert perf["td"]["model"] and "td" in week["caveats"]


def test_touchdown_picks_are_graded_from_the_box_score():
    log = {"schema": 1, "picks": {"g": {"final": {"home": 24, "away": 17},
                                        "home_td": [{"gsis_id": "a"}, {"gsis_id": "b"}],
                                        "away_td": [{"gsis_id": "c"}]}}}
    box = pd.DataFrame({"game_id": ["g", "g"], "player_id": ["a", "c"],
                        "rushing_tds": [1, 0], "receiving_tds": [0, 0]})
    assert npd.grade_touchdowns(log, pd.DataFrame(columns=box.columns)) == 0   # box score not in yet
    assert npd.grade_touchdowns(log, box) == 1
    p = log["picks"]["g"]
    assert [t["scored"] for t in p["home_td"]] == [True, False] and p["away_td"][0]["scored"] is False
    assert npd.grade_touchdowns(log, box) == 0


def test_byes_are_the_teams_without_a_regular_season_game():
    slate = pd.DataFrame({"home": ["KC", "BUF"], "away": ["LV", "MIA"]})
    assert len(npd.byes(slate, "REG")) == 28 and "KC" not in npd.byes(slate, "REG")
    assert npd.byes(slate, "WC") == []


def test_upcoming_games_carry_projected_leaders(run_output):
    out, _, _ = run_output
    week = json.loads((out / "week.json").read_text())
    upcoming = [g for g in week["games"] if g["status"] == "upcoming"]
    for g in upcoming:
        for side in ("home_leaders", "away_leaders"):
            leaders = g[side]
            assert set(leaders) <= {"passing", "rushing", "receiving"} and "passing" in leaders
            assert all(l["yards"] >= 0 and l["status"] != "out" for l in leaders.values())
    perf = json.loads((out / "performance.json").read_text())
    assert perf["props"]["model"] and "props" in week["caveats"]


def test_yardage_leaders_are_graded_from_the_box_score():
    log = {"schema": 1, "picks": {"g": {"final": {"home": 24, "away": 17},
                                        "home_leaders": {"passing": {"gsis_id": "a", "yards": 250.0},
                                                         "receiving": {"gsis_id": "b", "yards": 70.0}},
                                        "away_leaders": {"rushing": {"gsis_id": "c", "yards": 60.0}}}}}
    box = pd.DataFrame({"game_id": ["g", "g"], "player_id": ["a", "b"], "passing_yards": [301.0, 0.0],
                        "rushing_yards": [5.0, 0.0], "receiving_yards": [0.0, 44.0]})
    assert npd.grade_yards(log, box) == 1
    p = log["picks"]["g"]
    assert p["home_leaders"]["passing"]["over"] is True and p["home_leaders"]["receiving"]["over"] is False
    assert p["away_leaders"]["rushing"]["actual"] == 0.0     # no box-score line: he gained nothing
