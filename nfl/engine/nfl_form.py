"""A team's record and its last five results, for the app.

Shown beside each team's name - the NFL twin of the fighter's pro record
and last five - and never a model input.

  record   this season's wins, losses and ties before the game, playoffs
           counted separately (a 13-4 team that won a wild-card game is
           13-4, 1-0 in the playoffs, as every standings page prints it)
  last5    the five most recent games before this one, newest first, across
           the offseason if the season is young: a result (W, L, T) and the
           margin, which is the number printed inside the circle

POINT IN TIME. Only games that kicked off strictly before `before` count, so
a game's own result can never appear beside it.
"""

import pandas as pd


def team_games(games, team):
    """Every game the franchise has played, newest first, from its side."""
    mine = games[games["played"] & ((games["home"] == team) | (games["away"] == team))]
    rows = []
    for g in mine.itertuples(index=False):
        home = g.home == team
        scored = g.home_score if home else g.away_score
        allowed = g.away_score if home else g.home_score
        rows.append({
            "kickoff": g.kickoff, "season": int(g.season), "week": int(g.week),
            "game_type": g.game_type, "home": home, "neutral": bool(g.neutral),
            "opponent": g.away if home else g.home,
            "scored": int(scored), "allowed": int(allowed),
        })
    rows.sort(key=lambda r: r["kickoff"], reverse=True)
    return rows


def _result(r):
    return "W" if r["scored"] > r["allowed"] else "L" if r["scored"] < r["allowed"] else "T"


def form(games, team, before, season, n=5):
    """{record, playoffs, last5} for `team` as of `before` (a UTC kickoff)."""
    before = pd.Timestamp(before)
    past = [r for r in team_games(games, team) if r["kickoff"] < before]
    regular = [r for r in past if r["season"] == season and r["game_type"] == "REG"]
    post = [r for r in past if r["season"] == season and r["game_type"] != "REG"]

    def tally(rows):
        res = [_result(r) for r in rows]
        return {"wins": res.count("W"), "losses": res.count("L"), "ties": res.count("T")}

    last = []
    for r in past[:n]:
        result = _result(r)
        last.append({
            "result": result,
            "code": str(abs(r["scored"] - r["allowed"])) if result != "T" else "T",
            "score": f"{r['scored']}-{r['allowed']}",
            "opponent": r["opponent"],
            "home": r["home"],
            "neutral": r["neutral"],
            "season": r["season"],
            "week": r["week"],
            "game_type": r["game_type"],
            "date": r["kickoff"].strftime("%Y-%m-%d"),
        })
    out = {"record": tally(regular), "last5": last}
    if post:
        out["playoffs"] = tally(post)
    return out
