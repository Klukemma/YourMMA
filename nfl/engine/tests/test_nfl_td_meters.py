"""The touchdown record meters: TOP 3 -> SCORED (of the times a player was in
his team's logged top three and played, how often he scored) and NOT TOP 3
-> NO TD (of the times he played outside it, how often he did not), for
every player, team and week - from the live log and the box scores."""

import json
import math

import pandas as pd
import pytest

import nfl_data as nd
import nfl_export as nx
import nfl_predict as npd

GAME, LATER = "2026_07_LV_KC", "2026_09_KC_DEN"


def _pick(gsis, name, position, **kw):
    return dict({"gsis_id": gsis, "name": name, "position": position, "prob": 0.3}, **kw)


def _log():
    """Week 7: only the top three logged (graded), the rest rebuilt from
    snaps. Week 9: the whole list logged and graded. Week 9's LV side has no
    picks at all, and an ungraded game counts for nothing."""
    return {"schema": 1, "picks": {
        GAME: {"game_id": GAME, "season": 2026, "week": 7, "kickoff": "2026-10-25T17:00:00Z",
               "home": "KC", "away": "LV", "final": {"home": 27, "away": 20}, "td_graded": True,
               "home_td": [_pick("a", "Al Alpha", "RB", scored=True), _pick("b", "Bo Beta", "WR", scored=False),
                           _pick("c", "Cy Gamma", "TE", played=False, scored=None)],
               "away_td": [_pick("v", "Vic Visitor", "WR", scored=False)]},
        LATER: {"game_id": LATER, "season": 2026, "week": 9, "kickoff": "2026-11-08T18:00:00Z",
                "home": "DEN", "away": "KC", "final": {"home": 10, "away": 31}, "td_graded": True,
                "td_all_graded": True,
                "away_td": [_pick("d", "Dee Delta", "RB", scored=True), _pick("e", "Ed Echo", "WR", scored=False),
                            _pick("b", "Bo Beta", "WR", scored=True)],
                "away_td_all": [_pick("d", "Dee Delta", "RB", rank=1, scored=True),
                                _pick("e", "Ed Echo", "WR", rank=2, scored=False),
                                _pick("b", "Bo Beta", "WR", rank=3, scored=True),
                                _pick("a", "Al Alpha", "RB", rank=4, scored=False),
                                _pick("x", "Xi Bench", "WR", rank=5, prob=None, played=False, scored=None),
                                _pick("q", "Quin Quarter", "QB", rank=6, scored=True)]},
        "2026_09_NE_MIA": {"game_id": "2026_09_NE_MIA", "season": 2026, "week": 9, "home": "MIA", "away": "NE",
                           "kickoff": "2026-11-08T18:00:00Z",
                           "home_td": [_pick("m", "Mo Miami", "WR")], "away_td": [_pick("n", "Ned NE", "WR")]},
    }}


def _box(rows):
    cols = ["game_id", "player_id", "player_display_name", "team", "position", "carries", "targets",
            "receptions", "rushing_tds", "receiving_tds", "rushing_yards", "receiving_yards"]
    return pd.DataFrame(rows, columns=cols)


def _snaps(rows):
    out = pd.DataFrame(rows, columns=["game_id", "player", "pfr_player_id", "position", "team", "offense_snaps"])
    out["offense_pct"] = out["offense_snaps"] / 65.0
    out["defense_snaps"] = 0.0
    return out


@pytest.fixture()
def week7_data():
    """Week 7's box score and snap counts. KC: the top three's A and B
    played (C did not); D ran one in; E took snaps and has no box line; Q
    the quarterback did not score; H is a back the snap counts call HB;
    T is a tackle. LV: V and his teammate W."""
    box = _box([
        (GAME, "a", "Al Alpha", "KC", "RB", 15, 2, 2, 1, 0, 70, 12),
        (GAME, "b", "Bo Beta", "KC", "WR", 0, 8, 5, 0, 0, 0, 60),
        (GAME, "d", "Dee Delta", "KC", "RB", 6, 1, 1, 1, 0, 20, 4),
        (GAME, "q", "Quin Quarter", "KC", "QB", 3, 0, 0, 0, 0, 9, 0),
        (GAME, "h", "Hal Halfback", "KC", "RB", 2, 0, 0, 0, 0, 5, 0),
        (GAME, "v", "Vic Visitor", "LV", "WR", 0, 6, 4, 0, 0, 0, 51),
        (GAME, "w", "Wes Wide", "LV", "WR", 0, 4, 3, 0, 1, 0, 30),
    ])
    snaps = _snaps([
        (GAME, "Al Alpha", "AlphAl00", "RB", "KC", 40), (GAME, "Bo Beta", "BetaBo00", "WR", "KC", 50),
        (GAME, "Dee Delta", "DeltDe00", "RB", "KC", 20), (GAME, "Ed Echo", "EchoEd00", "WR", "KC", 6),
        (GAME, "Quin Quarter", "QuarQu00", "QB", "KC", 65), (GAME, "Hal Halfback", "HalfHa00", "HB", "KC", 9),
        (GAME, "Tom Tackle", "TackTo00", "T", "KC", 65),
        (GAME, "Vic Visitor", "VisiVi00", "WR", "LV", 55), (GAME, "Wes Wide", "WideWe00", "WR", "LV", 40),
    ])
    ids = pd.DataFrame({"gsis_id": ["a", "b", "d", "e", "q", "h", "t", "v", "w"],
                        "pfr_id": ["AlphAl00", "BetaBo00", "DeltDe00", "EchoEd00", "QuarQu00", "HalfHa00",
                                   "TackTo00", "VisiVi00", "WideWe00"]})
    return box, snaps, ids


@pytest.fixture()
def meters(week7_data):
    log = _log()
    lines = npd.td_record_lines(log, *week7_data)
    return nx.td_meters(log, lines)


def _player(m, gsis):
    return next(p for p in m["players"] if p["gsis_id"] == gsis)


def test_the_two_groups_of_a_rebuilt_game(week7_data):
    log = _log()
    del log["picks"][LATER]
    m = nx.td_meters(log, npd.td_record_lines(log, *week7_data))
    kc = next(t for t in m["teams"] if t["team"] == "KC")
    # Top 3: A scored, B did not; C did not play and is left out.
    assert kc["top3"] == {"n": 2, "scored": 1, "rate": 0.5}
    # Not top 3: D (scored), E (snaps, no box line: no TD), Q, H (an "HB" is a
    # back). The tackle is not a QB, RB, FB, WR or TE; A and B are top 3.
    assert kc["rest"]["n"] == 4 and kc["rest"]["no_td"] == 3
    assert {p["gsis_id"] for p in m["players"] if p["rest"]["n"]} == {"d", "e", "q", "h", "w"}
    assert _player(m, "e")["rest"] == {"n": 1, "no_td": 1, "rate": 1.0}
    assert _player(m, "d")["rest"] == {"n": 1, "no_td": 0, "rate": 0.0}
    assert "t" not in {p["gsis_id"] for p in m["players"]}
    assert "c" not in {p["gsis_id"] for p in m["players"]}      # never played: no meter at all
    lv = next(t for t in m["teams"] if t["team"] == "LV")
    assert lv["top3"] == {"n": 1, "scored": 0, "rate": 0.0} and lv["rest"] == {"n": 1, "no_td": 0, "rate": 0.0}
    assert m["by_week"][0]["rest_from"] == ["snaps"]


def test_a_logged_list_counts_the_rest_of_the_list_who_played(meters):
    week9 = next(w for w in meters["by_week"] if w["week"] == 9)
    # D, E, B in the top 3 (D and B scored); A, Q the rest who played - A
    # did not score, Q did; X did not play.
    assert week9["top3"] == {"n": 3, "scored": 2, "rate": pytest.approx(2 / 3)}
    assert week9["rest"] == {"n": 2, "no_td": 1, "rate": 0.5}
    assert week9["rest_from"] == ["list"] and tuple(meters["list_from"]) == (2026, 9)
    assert "x" not in {p["gsis_id"] for p in meters["players"]}
    # The ungraded game and the side with no picks count for nothing.
    assert {w["week"] for w in meters["by_week"]} == {7, 9}
    assert not {"MIA", "NE", "DEN"} & {t["team"] for t in meters["teams"]}


def test_players_and_teams_add_up(meters):
    # Bo Beta: top 3 twice (no TD, then a TD). Al Alpha: top 3 once (TD),
    # then outside it once (no TD).
    assert _player(meters, "b")["top3"] == {"n": 2, "scored": 1, "rate": 0.5}
    assert _player(meters, "b")["rest"] == {"n": 0, "no_td": 0, "rate": None}
    a = _player(meters, "a")
    assert a["top3"] == {"n": 1, "scored": 1, "rate": 1.0} and a["rest"] == {"n": 1, "no_td": 1, "rate": 1.0}
    assert a["team"] == "KC" and a["name"] == "Al Alpha"
    kc = next(t for t in meters["teams"] if t["team"] == "KC")
    assert kc["top3"]["n"] == 5 and kc["top3"]["scored"] == 3
    assert kc["rest"]["n"] == 6 and kc["rest"]["no_td"] == 4
    for kind, key in (("top3", "scored"), ("rest", "no_td")):
        o = meters["overall"][kind]
        assert o["n"] == sum(t[kind]["n"] for t in meters["teams"]) == sum(p[kind]["n"] for p in meters["players"])
        assert o[key] == sum(w[kind][key] for w in meters["by_week"])
    # Ordered by sample: KC, with the most games, first.
    assert meters["teams"][0]["team"] == "KC"


def test_a_side_without_snap_counts_waits(week7_data):
    log = _log()
    box, snaps, ids = week7_data
    lines = npd.td_record_lines(log, box, snaps[snaps["team"] != "LV"], ids)
    m = nx.td_meters(log, lines)
    lv = next(t for t in m["teams"] if t["team"] == "LV")
    assert lv["top3"]["n"] == 1 and lv["rest"]["n"] == 0 and m["pending"] == 1
    log["picks"][LATER]["td_all_graded"] = False
    assert nx.td_meters(log, lines)["pending"] == 2


def test_the_whole_list_is_graded_like_the_top_three():
    log = {"schema": 1, "picks": {"g": {
        "final": {"home": 24, "away": 17}, "td_graded": True,
        "home_td_all": [{"gsis_id": "a", "name": "A"}, {"gsis_id": "b", "name": "B"},
                        {"gsis_id": "r", "name": "Rookie Name"}, {"gsis_id": "z", "name": "Z"}]}}}
    box = pd.DataFrame({"game_id": ["g"], "player_id": ["a"], "rushing_tds": [1], "receiving_tds": [0]})
    snaps = pd.DataFrame({"game_id": ["g"] * 3, "player": ["A", "B", "Rookie N. Name"],
                          "pfr_player_id": ["Aa00", "Bb00", "Rr00"],
                          "offense_snaps": [50, 30, 4], "defense_snaps": [0, 0, 0]})
    ids = pd.DataFrame({"gsis_id": ["a", "b", "r"], "pfr_id": ["Aa00", "Bb00", "Rr00"]})
    assert npd.grade_touchdown_lists(log, box) == 0                 # b, r, z: no box line, no snaps yet
    assert npd.grade_touchdown_lists(log, box, snaps, ids) == 1
    entries = log["picks"]["g"]["home_td_all"]
    assert [e["scored"] for e in entries] == [True, False, False, None]
    assert entries[3]["played"] is False                             # Z took no snap: not a miss
    assert log["picks"]["g"]["td_all_graded"] is True
    assert npd.grade_touchdown_lists(log, box, snaps, ids) == 0      # graded once


def test_meters_on_the_week_payload_are_json_safe(meters, tmp_path):
    logged = _log()["picks"][LATER]
    game = {"home": "DEN", "away": "KC", "away_td": logged["away_td"], "away_td_all": logged["away_td_all"],
            "home_td": [_pick("zz", "New Guy", "WR", prob=float("nan"))]}
    nx.attach_td_meters(game, meters)
    assert "meters" not in logged["away_td"][0]                      # the logged list is left as logged
    assert game["away_td"][0]["meters"] == {"top3": _player(meters, "d")["top3"], "rest": _player(meters, "d")["rest"]}
    assert game["home_td"][0]["meters"]["top3"] == {"n": 0, "scored": 0, "rate": None}
    assert game["td_meters"]["home"]["rest"]["rate"] is None         # DEN: no picks yet - missing, not 0%
    path = nx.write("week", {"games": [game], "td_meters": meters}, tmp_path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["games"][0]["home_td"][0]["prob"] is None


def test_the_committed_log_gives_sensible_totals():
    """The real log and box scores. Top 3 -> scored counts exactly the
    graded top-three picks the track record counts; the rest are mostly
    players who did not score."""
    log = npd.load_log()
    lines = npd.td_record_lines(log, nd.load_player_games(), nd.load_snaps(), nd.load_player_ids())
    m = nx.td_meters(log, lines)
    track = nx.track_record(log)["td"]
    o = m["overall"]
    assert o["top3"]["n"] == track["picks"] and o["top3"]["scored"] == track["scored"]
    if not o["top3"]["n"]:
        pytest.skip("no graded touchdown picks in the log")
    assert o["rest"]["n"] > 2 * o["top3"]["n"]
    assert 0.2 < o["top3"]["rate"] < 0.7 and 0.75 < o["rest"]["rate"] < 0.98
    assert o["rest"]["rate"] > o["top3"]["rate"]
    assert all(isinstance(p["name"], str) and p["team"] in nd.TEAMS for p in m["players"])
    # One player-game per player per side: never more rest games than graded sides.
    sides = sum(1 for p in log["picks"].values() if p.get("td_graded")
                for s in ("home_td", "away_td") if p.get(s))
    assert max(p["top3"]["n"] + p["rest"]["n"] for p in m["players"]) <= sides
    for t in m["teams"]:
        for kind in ("top3", "rest"):
            r = t[kind]["rate"]
            assert r is None or (0 <= r <= 1 and not math.isnan(r))


def _sheet(m, team):
    """What a team's sheet adds up: each player's games for that team."""
    total = {"top3": [0, 0], "rest": [0, 0]}
    for p in m["players"]:
        mine = (p.get("by_team") or {}).get(team) if p.get("by_team") else (p if p["team"] == team else None)
        for kind, key in (("top3", "scored"), ("rest", "no_td")):
            if mine:
                total[kind][0] += mine[kind]["n"]
                total[kind][1] += mine[kind][key]
    return total


def test_a_player_who_moved_teams_is_split_by_team(week7_data):
    """Al Alpha, KC's back in week 7 and 9, is traded to LV and in its top
    three in week 10: his own meters cover all three games, his team is LV,
    and by_team keeps each team's share so both sheets add up."""
    log = _log()
    log["picks"]["2026_10_LV_SEA"] = {
        "game_id": "2026_10_LV_SEA", "season": 2026, "week": 10, "kickoff": "2026-11-15T18:00:00Z",
        "home": "LV", "away": "SEA", "final": {"home": 20, "away": 13}, "td_graded": True, "td_all_graded": True,
        "home_td": [_pick("a", "Al Alpha", "RB", scored=True)],
        "home_td_all": [_pick("a", "Al Alpha", "RB", rank=1, scored=True),
                        _pick("w", "Wes Wide", "WR", rank=2, scored=False)]}
    m = nx.td_meters(log, npd.td_record_lines(log, *week7_data))
    a = _player(m, "a")
    assert a["team"] == "LV" and a["top3"] == {"n": 2, "scored": 2, "rate": 1.0}
    assert a["by_team"]["KC"]["top3"]["n"] == 1 and a["by_team"]["KC"]["rest"]["n"] == 1
    assert a["by_team"]["LV"] == {"top3": {"n": 1, "scored": 1, "rate": 1.0},
                                  "rest": {"n": 0, "no_td": 0, "rate": None}}
    assert "by_team" not in _player(m, "b")                          # one team: nothing extra
    for t in m["teams"]:
        total = _sheet(m, t["team"])
        assert total["top3"] == [t["top3"]["n"], t["top3"]["scored"]]
        assert total["rest"] == [t["rest"]["n"], t["rest"]["no_td"]]


def test_every_team_sheet_adds_up_on_the_committed_log():
    log = npd.load_log()
    m = nx.td_meters(log, npd.td_record_lines(log, nd.load_player_games(), nd.load_snaps(),
                                              nd.load_player_ids()))
    for t in m["teams"]:
        total = _sheet(m, t["team"])
        assert total["top3"] == [t["top3"]["n"], t["top3"]["scored"]], t["team"]
        assert total["rest"] == [t["rest"]["n"], t["rest"]["no_td"]], t["team"]
