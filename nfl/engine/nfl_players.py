"""Each team's most valuable players, and whether they are playing.

Shown beside every game, never a model input - with one exception the model
already makes: the starting quarterback (nfl_predict replaces a ruled-out
starter). Everything here is SHOWN, NOT APPLIED, until a weight for it has
been measured.

MOST VALUABLE ON OFFENCE is ranked by expected points added (EPA) this
season: the points a player's plays added above what an average play in the
same down, distance and field position adds, as nflverse computes it -
passing and rushing EPA for a quarterback, rushing and receiving for
everyone else. Early in a season, when a team has played fewer than four
games, last season's per-game rate fills the missing games at half weight,
so a star does not drop off the list in week 1 because he has had one game.

TOP DEFENDERS have no EPA of their own in the data, so they are ranked by
the plays a defender makes himself: sacks, takeaways (interceptions and
forced fumbles), passes defended, tackles for loss, defensive touchdowns.

AVAILABILITY is read from what is published before kickoff: the weekly
roster (reserve lists - injured reserve and the like - and game-day
inactives) and the official injury report (Out, Doubtful, Questionable).
A listed player who is not playing is called out at the top of the game.
For a game already played, availability is what happened: a player with no
stats in it did not play.
"""

import pandas as pd

import nfl_data as nd

TOP_OFFENSE = 3
TOP_DEFENSE = 2
EARLY_GAMES = 4          # below this many team games, last season fills in
LAST_SEASON_WEIGHT = 0.5

ROSTER_OUT = {"RES": "on a reserve list", "INA": "inactive", "CUT": "released",
              "RET": "retired", "TRD": "traded", "TRT": "traded", "EXE": "exempt list",
              "DEV": "on the practice squad", "SUS": "suspended"}
REPORT_RANK = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable"}


class PlayerData:
    """Everything the player layer reads, loaded once per run."""

    def __init__(self, player_games=None, defense_games=None, rosters=None, injuries=None, snaps=None):
        self.offense = nd.load_player_games() if player_games is None else player_games
        self.defense = nd.load_defense_games() if defense_games is None else defense_games
        self.rosters = nd.load_rosters() if rosters is None else rosters
        self.injuries = nd.load_injuries() if injuries is None else injuries
        self.snaps = nd.load_snaps() if snaps is None else snaps

    def games_of(self, frame, before, season):
        return frame[(frame["season"] == season) & (frame["game_id"].isin(self._played_before(before, season)))]

    def set_schedule(self, games):
        self._schedule = games[["game_id", "season", "kickoff", "played"]]

    def _played_before(self, before, season):
        s = self._schedule
        return set(s[(s["season"] == season) & s["played"] & (s["kickoff"] < before)]["game_id"])


# ------------------------------------------------------------- stat lines

def _fmt(n):
    n = float(n)
    return f"{n:,.0f}" if abs(n) >= 10 or n == int(n) else f"{n:.1f}"


def offense_line(rows, position):
    """This season's line in the words a box score uses."""
    t = rows.sum(numeric_only=True)
    if position == "QB":
        parts = [f"{_fmt(t.get('passing_yards', 0))} pass yds", f"{_fmt(t.get('passing_tds', 0))} TD",
                 f"{_fmt(t.get('passing_interceptions', 0))} INT"]
        if t.get("rushing_yards", 0) >= 100:
            parts.append(f"{_fmt(t['rushing_yards'])} rush yds")
        return ", ".join(parts)
    if position in ("RB", "FB"):
        parts = [f"{_fmt(t.get('rushing_yards', 0))} rush yds"]
        if t.get("receiving_yards", 0) >= 50:
            parts.append(f"{_fmt(t['receiving_yards'])} rec yds")
        parts.append(f"{_fmt(t.get('rushing_tds', 0) + t.get('receiving_tds', 0))} TD")
        return ", ".join(parts)
    return (f"{_fmt(t.get('receptions', 0))} rec, {_fmt(t.get('receiving_yards', 0))} yds, "
            f"{_fmt(t.get('receiving_tds', 0) + t.get('rushing_tds', 0))} TD")


def defense_line(t):
    parts = []
    for key, word in (("def_sacks", "sacks"), ("def_interceptions", "INT"), ("def_fumbles_forced", "FF"),
                      ("def_pass_defended", "PD"), ("def_tackles_for_loss", "TFL"), ("def_tds", "TD")):
        v = float(t.get(key, 0) or 0)
        if v:
            parts.append(f"{_fmt(v)} {word if not (word == 'sacks' and v == 1) else 'sack'}")
    return ", ".join(parts[:3]) or "—"


def defense_score(t):
    g = lambda k: float(t.get(k, 0) or 0)
    return (g("def_sacks") + g("def_interceptions") + g("def_fumbles_forced")
            + 0.5 * g("def_pass_defended") + 0.25 * g("def_tackles_for_loss") + 2 * g("def_tds"))


# ------------------------------------------------------------- ranking

def _current_team(rows):
    """The team of a player's most recent game in `rows`."""
    return rows.sort_values(["season", "week"])["team"].iloc[-1]


def key_players(data, team, season, before):
    """{offense: [...], defense: [...]} for `team`, from games played before
    `before` (UTC kickoff). Each player: gsis_id, name, position, value (EPA
    or defensive plays), line (this season), games."""
    before = pd.Timestamp(before)
    off_now = data.games_of(data.offense, before, season)
    team_games = len(set(off_now[off_now["team"] == team]["game_id"]))
    off_last = data.offense[data.offense["season"] == season - 1]

    # A player belongs to the team he last played for this season - or, if
    # he has not played yet this season, the team whose roster lists him.
    roster_team = {}
    if len(data.rosters):
        r = data.rosters[data.rosters["season"] == season].sort_values("week")
        roster_team = dict(zip(r["gsis_id"], r["team"]))
    candidates = set(off_now[off_now["team"] == team]["player_id"])
    candidates |= {p for p, t in roster_team.items() if t == team} & set(off_last["player_id"])

    offense = []
    for pid in candidates:
        mine = off_now[off_now["player_id"] == pid]
        if len(mine) and _current_team(mine) != team:
            continue
        if not len(mine) and roster_team.get(pid) != team:
            continue
        before_rows = off_last[off_last["player_id"] == pid]
        value = float(mine["epa"].sum())
        if team_games < EARLY_GAMES and len(before_rows):
            per_game = float(before_rows["epa"].sum()) / len(before_rows)
            value += (EARLY_GAMES - team_games) * per_game * LAST_SEASON_WEIGHT
        src = mine if len(mine) else before_rows
        last = src.sort_values(["season", "week"]).iloc[-1]
        offense.append({
            "gsis_id": pid, "name": last["player_display_name"], "position": last["position"],
            "value": value, "games": int(len(mine)),
            "line": offense_line(mine, last["position"]) if len(mine) else
                    f"last season: {offense_line(before_rows, last['position'])}",
        })
    offense.sort(key=lambda p: (-p["value"], p["name"]))   # a tie never depends on set order

    dfn_now = data.games_of(data.defense, before, season) if len(data.defense) else data.defense
    defense = []
    if len(dfn_now):
        for pid, rows in dfn_now.groupby("player_id"):
            if _current_team(rows) != team:
                continue
            totals = rows.sum(numeric_only=True)
            last = rows.sort_values(["season", "week"]).iloc[-1]
            defense.append({"gsis_id": pid, "name": last["player_display_name"],
                            "position": last["position"], "value": defense_score(totals),
                            "games": int(len(rows)), "line": defense_line(totals)})
        defense.sort(key=lambda p: (-p["value"], p["name"]))
    for rank, p in enumerate(offense[:TOP_OFFENSE], 1):
        p["rank"] = rank
    for rank, p in enumerate(defense[:TOP_DEFENSE], 1):
        p["rank"] = rank
    return {"offense": offense[:TOP_OFFENSE], "defense": [p for p in defense[:TOP_DEFENSE] if p["value"] > 0]}


# ------------------------------------------------------------- availability

def availability(data, player, team, season, week, played_game=None):
    """{status, reason} for one player in one game.

    status is 'out', 'doubtful', 'questionable' or 'active'. For a game
    already played (`played_game` = its game_id) it is whether he recorded
    a stat in it."""
    pid, name = player["gsis_id"], player["name"]
    # A finished game says who played once its snap counts are in - every
    # snap, not just the plays that make a box score (a receiver who was
    # never thrown to still played). nflverse posts them hours after the
    # result; until then, the pre-game report stands.
    if played_game is not None and len(data.snaps):
        snaps = data.snaps[(data.snaps["game_id"] == played_game) & (data.snaps["team"] == team)]
        if len(snaps):
            mine = snaps[snaps["player"] == name]
            played = bool(len(mine)) and float(mine[["offense_snaps", "defense_snaps"]].fillna(0).sum(axis=1).max()) > 0
            return {"status": "active" if played else "out", "reason": None if played else "did not play"}

    inj = data.injuries
    if len(inj):
        rows = inj[(inj["season"] == season) & (inj["week"] == week) & (inj["team"].map(nd.franchise) == team)
                   & ((inj["gsis_id"] == pid) | (inj["full_name"] == name))]
        rows = rows[rows["report_status"].isin(list(REPORT_RANK))]
        if len(rows):
            r = rows.iloc[0]
            injury = r["report_primary_injury"] if isinstance(r["report_primary_injury"], str) else None
            return {"status": REPORT_RANK[r["report_status"]],
                    "reason": f"{r['report_status']} on the injury report" + (f" ({injury.lower()})" if injury else "")}
    ro = data.rosters
    if len(ro):
        rows = ro[(ro["season"] == season) & (ro["gsis_id"] == pid) & (ro["week"] <= week)]
        if len(rows):
            latest = rows.sort_values("week").iloc[-1]
            status = latest["status"]
            # A game-day inactive only means this week if it IS this week.
            if status in ROSTER_OUT and not (status == "INA" and latest["week"] != week):
                return {"status": "out", "reason": ROSTER_OUT[status]}
    return {"status": "active", "reason": None}


def for_game(data, row, season, week, before):
    """Both teams' key players for one game, each with availability, and
    the alerts the card leads with: a listed player out or doubtful, or the
    most valuable player questionable."""
    played = row["game_id"] if bool(row["played"]) else None
    out = {"alerts": []}
    for side in ("home", "away"):
        team = row[side]
        kp = key_players(data, team, season, before)
        for group in ("offense", "defense"):
            for p in kp[group]:
                p.update(availability(data, p, team, season, week, played))
                most = group == "offense" and p["rank"] == 1
                if p["status"] in ("out", "doubtful") or (most and p["status"] == "questionable"):
                    out["alerts"].append({
                        "team": team, "side": side, "name": p["name"], "position": p["position"],
                        "status": p["status"], "reason": p["reason"],
                        "role": ("most valuable player" if most else
                                 f"#{p['rank']} on offence" if group == "offense" else f"#{p['rank']} defender"),
                    })
        out[side] = kp
    order = {"out": 0, "doubtful": 1, "questionable": 2}
    out["alerts"].sort(key=lambda a: (order.get(a["status"], 3), a["role"] != "most valuable player"))
    return out
