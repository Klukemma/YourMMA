"""Key players and whether they are playing: the alerts the card leads with."""

import pandas as pd
import pytest

import nfl_players as npl

KICK = pd.Timestamp("2026-10-18 17:00")


def _schedule():
    rows = [("g1", 2026, pd.Timestamp("2026-09-13 17:00"), True),
            ("g2", 2026, pd.Timestamp("2026-09-20 17:00"), True),
            ("g3", 2026, pd.Timestamp("2026-09-27 17:00"), True),
            ("g4", 2026, pd.Timestamp("2026-10-04 17:00"), True),
            ("g5", 2026, KICK, False),
            ("old", 2025, pd.Timestamp("2025-12-28 17:00"), True)]
    return pd.DataFrame(rows, columns=["game_id", "season", "kickoff", "played"])


def _offense():
    def r(pid, name, pos, game, season, epa, **kw):
        base = {"player_id": pid, "player_display_name": name, "position": pos, "game_id": game,
                "season": season, "week": int(game[-1]) if game[-1].isdigit() else 17, "team": "KC",
                "epa": epa, "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
                "rushing_yards": 0, "rushing_tds": 0, "receiving_yards": 0, "receiving_tds": 0, "receptions": 0}
        base.update(kw)
        return base
    rows = [r("qb", "Pat Q", "QB", g, 2026, 10.0, passing_yards=250, passing_tds=2) for g in ("g1", "g2", "g3", "g4")]
    rows += [r("wr", "Wide R", "WR", g, 2026, 4.0, receptions=5, receiving_yards=70) for g in ("g1", "g2", "g3", "g4")]
    rows += [r("te", "Tight E", "TE", g, 2026, 1.0, receptions=3, receiving_yards=30) for g in ("g1", "g2")]
    rows += [r("rb", "Run B", "RB", g, 2026, 0.5, rushing_yards=60) for g in ("g1", "g2", "g3", "g4")]
    return pd.DataFrame(rows)


def _data(injuries=None, rosters=None, snaps=None):
    d = npl.PlayerData(player_games=_offense(), defense_games=pd.DataFrame(),
                       rosters=rosters if rosters is not None else pd.DataFrame(),
                       injuries=injuries if injuries is not None else pd.DataFrame(),
                       snaps=snaps if snaps is not None else pd.DataFrame())
    d.set_schedule(_schedule())
    return d


def test_ranked_by_epa_this_season_with_box_score_lines():
    kp = npl.key_players(_data(), "KC", 2026, KICK)
    assert [p["name"] for p in kp["offense"]] == ["Pat Q", "Wide R", "Tight E"]
    assert kp["offense"][0]["value"] == pytest.approx(40.0) and kp["offense"][0]["rank"] == 1
    assert kp["offense"][0]["line"].startswith("1,000 pass yds, 8 TD")


def test_only_games_before_kickoff_count():
    kp = npl.key_players(_data(), "KC", 2026, pd.Timestamp("2026-09-21 00:00"))
    assert kp["offense"][0]["value"] == pytest.approx(20.0)


def _injury(status, name="Pat Q", pid="qb", week=6):
    return pd.DataFrame([{"season": 2026, "week": week, "team": "KC", "gsis_id": pid, "full_name": name,
                          "report_status": status, "report_primary_injury": "Ankle"}])


@pytest.mark.parametrize("status,expected", [("Out", "out"), ("Doubtful", "doubtful"),
                                             ("Questionable", "questionable")])
def test_the_injury_report_decides(status, expected):
    d = _data(injuries=_injury(status))
    a = npl.availability(d, {"gsis_id": "qb", "name": "Pat Q"}, "KC", 2026, 6)
    assert a["status"] == expected and "ankle" in a["reason"]


def test_a_reserve_list_is_out_but_last_weeks_inactive_is_not():
    rosters = pd.DataFrame([{"season": 2026, "week": 5, "team": "KC", "gsis_id": "qb", "status": "INA"},
                            {"season": 2026, "week": 5, "team": "KC", "gsis_id": "wr", "status": "RES"}])
    d = _data(rosters=rosters)
    assert npl.availability(d, {"gsis_id": "qb", "name": "Pat Q"}, "KC", 2026, 6)["status"] == "active"
    assert npl.availability(d, {"gsis_id": "wr", "name": "Wide R"}, "KC", 2026, 6)["status"] == "out"


def test_a_finished_game_reads_snap_counts_once_they_exist():
    snaps = pd.DataFrame([{"game_id": "g4", "team": "KC", "player": "Wide R", "offense_snaps": 40, "defense_snaps": 0},
                          {"game_id": "g4", "team": "KC", "player": "Tight E", "offense_snaps": 0, "defense_snaps": 0}])
    d = _data(snaps=snaps, injuries=_injury("Questionable", "Tight E", "te", week=4))
    assert npl.availability(d, {"gsis_id": "wr", "name": "Wide R"}, "KC", 2026, 4, "g4")["status"] == "active"
    assert npl.availability(d, {"gsis_id": "te", "name": "Tight E"}, "KC", 2026, 4, "g4")["reason"] == "did not play"
    # No snap counts yet for g3: the pre-game report stands.
    assert npl.availability(d, {"gsis_id": "te", "name": "Tight E"}, "KC", 2026, 3, "g3")["status"] == "active"


def test_the_most_valuable_player_out_leads_the_alerts():
    d = _data(injuries=pd.concat([_injury("Out"), _injury("Questionable", "Tight E", "te")]))
    row = pd.Series({"game_id": "g5", "played": False, "home": "KC", "away": "LV"})
    out = npl.for_game(d, row, 2026, 6, KICK)
    assert out["alerts"][0]["name"] == "Pat Q" and out["alerts"][0]["role"] == "most valuable player"
    # A questionable #3 is not an alert; a questionable MVP would be.
    assert [a["name"] for a in out["alerts"]] == ["Pat Q"]
