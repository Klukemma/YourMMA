"""A team's recent results, as the app shows them beside every match.

Shown, never a model input: the model reads every match through its own
ratings. "Form" is the last five league matches played on days BEFORE the
one asked about - in either tier, so a promoted team's run-in to promotion
shows in August - newest first.
"""

import pandas as pd

LAST = 5


def form(matches, team, before, n=LAST):
    """[{date, opponent, home, gf, ga, result}] newest first, result W/D/L
    from `team`'s side."""
    day = str(pd.Timestamp(before).date()) if not isinstance(before, str) else before[:10]
    mine = matches[((matches["home"] == team) | (matches["away"] == team))
                   & (matches["date"] < day) & matches["result"].isin(("H", "D", "A"))]
    out = []
    for _, m in mine.sort_values(["date", "kickoff"], na_position="first").tail(n).iloc[::-1].iterrows():
        home = m["home"] == team
        gf, ga = (int(m["hg"]), int(m["ag"])) if home else (int(m["ag"]), int(m["hg"]))
        out.append({"date": m["date"], "opponent": m["away"] if home else m["home"], "home": bool(home),
                    "gf": gf, "ga": ga, "league": m["league"],
                    "result": "W" if gf > ga else "L" if gf < ga else "D"})
    return out


def table(matches, league, season, before=None):
    """The league table from results on days before `before` (all of the
    season's results when None): played, won, drawn, lost, goals, points,
    ordered by points, goal difference, goals scored, then name."""
    day = None if before is None else str(pd.Timestamp(before).date())
    m = matches[(matches["league"] == league) & (matches["season"] == season)
                & matches["result"].isin(("H", "D", "A"))]
    if day is not None:
        m = m[m["date"] < day]
    teams = sorted(set(matches[(matches["league"] == league) & (matches["season"] == season)]["home"])
                   | set(matches[(matches["league"] == league) & (matches["season"] == season)]["away"]))
    rows = {t: {"team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0}
            for t in teams}
    for h, a, hg, ag in zip(m["home"], m["away"], m["hg"], m["ag"]):
        hg, ag = int(hg), int(ag)
        for team, f, g in ((h, hg, ag), (a, ag, hg)):
            r = rows[team]
            r["played"] += 1
            r["gf"] += f
            r["ga"] += g
            if f > g:
                r["won"] += 1
                r["points"] += 3
            elif f == g:
                r["drawn"] += 1
                r["points"] += 1
            else:
                r["lost"] += 1
    out = sorted(rows.values(), key=lambda r: (-r["points"], -(r["gf"] - r["ga"]), -r["gf"], r["team"]))
    for i, r in enumerate(out, 1):
        r["position"] = i
        r["gd"] = r["gf"] - r["ga"]
    return out
