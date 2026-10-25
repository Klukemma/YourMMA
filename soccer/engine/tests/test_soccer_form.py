"""Recent form and the league table, from results before the day asked."""

import pandas as pd

import soccer_form as sf


def _m(date, home, away, hg, ag, league="E0", season=2026):
    return {"match_id": f"{date}{home}{away}", "league": league, "season": season, "date": date,
            "kickoff": pd.NaT, "home": home, "away": away, "hg": hg, "ag": ag,
            "result": "H" if hg > ag else "A" if hg < ag else "D"}


MATCHES = pd.DataFrame([
    _m("2026-08-15", "Arsenal", "Chelsea", 2, 0),
    _m("2026-08-22", "Chelsea", "Spurs", 1, 1),
    _m("2026-08-29", "Spurs", "Arsenal", 3, 1),
    _m("2026-09-05", "Arsenal", "Spurs", 0, 0),
    _m("2026-09-05", "Chelsea", "Hull", 4, 1),
])


def test_form_is_newest_first_from_the_teams_side_and_ignores_the_day_itself():
    f = sf.form(MATCHES, "Arsenal", "2026-09-05")
    assert [r["result"] for r in f] == ["L", "W"]
    assert f[0] == {"date": "2026-08-29", "opponent": "Spurs", "home": False, "gf": 1, "ga": 3,
                    "league": "E0", "result": "L"}


def test_the_table_orders_by_points_then_goal_difference():
    t = sf.table(MATCHES, "E0", 2026)
    # Spurs 5 points; Chelsea and Arsenal 4 each, Chelsea ahead on goal difference.
    assert [r["team"] for r in t] == ["Spurs", "Chelsea", "Arsenal", "Hull"]
    chelsea = t[1]
    assert (chelsea["position"], chelsea["played"], chelsea["points"], chelsea["gd"]) == (2, 3, 4, 1)
    before = sf.table(MATCHES, "E0", 2026, before="2026-09-05")
    assert next(r for r in before if r["team"] == "Hull")["played"] == 0
