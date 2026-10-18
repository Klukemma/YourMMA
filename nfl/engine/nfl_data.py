"""NFL data: schedules and closing lines, team-game stats, quarterbacks and
injury reports, all from nflverse's public releases (github.com/nflverse).

Snapshotted into nfl/data/ so history never depends on the network: a sync
refreshes the current season and keeps what is already on file. Everything
else in the NFL engine reads the snapshot through the loaders below.

FRANCHISE CODES. The schedule names a team as it was called that season
(OAK, SD, STL); nflverse's team and player stats use today's code for every
season (LV, LAC, LA). A rating that followed the code would reset the
Raiders when they moved to Las Vegas and give the Rams two histories, so the
schedule is mapped onto the franchise as it is called now, and the code a
season actually used is kept beside it for display.

POINT IN TIME. A kickoff is a moment, not a date: Thursday's game is over
before Sunday's starts, and a Sunday morning London game is over before the
1pm slate. The schedule's local Eastern kickoff is converted to UTC once,
here, and every "known before" question downstream is asked of that.

THE LINES ARE CLOSING LINES. spread_line, total_line and the moneylines are
the market at kickoff for a game already played, and the latest posted line
for one still to come. spread_line is from the HOME side's point of view in
the schedule's own convention: positive means the home team is favoured by
that many points (it predicts result = home score - away score).
"""

import gzip
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NFL = Path(__file__).resolve().parents[1]
DATA = NFL / "data"
GAMES = DATA / "games.csv.gz"
TEAM_GAMES = DATA / "team_games.csv.gz"
QB_GAMES = DATA / "qb_games.csv.gz"
INJURIES = DATA / "injuries.csv.gz"

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_URL = RELEASE + "/stats_team/stats_team_week_{season}.csv"
PLAYER_URL = RELEASE + "/stats_player/stats_player_week_{season}.csv"
INJURY_URL = RELEASE + "/injuries/injuries_{season}.csv"

FIRST_SEASON = 1999
EASTERN = ZoneInfo("America/New_York")

# Old code -> the franchise's code today.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# code: (name, conference, division, home UTC offset in winter hours)
TEAMS = {
    "ARI": ("Arizona Cardinals", "NFC", "West", -7),
    "ATL": ("Atlanta Falcons", "NFC", "South", -5),
    "BAL": ("Baltimore Ravens", "AFC", "North", -5),
    "BUF": ("Buffalo Bills", "AFC", "East", -5),
    "CAR": ("Carolina Panthers", "NFC", "South", -5),
    "CHI": ("Chicago Bears", "NFC", "North", -6),
    "CIN": ("Cincinnati Bengals", "AFC", "North", -5),
    "CLE": ("Cleveland Browns", "AFC", "North", -5),
    "DAL": ("Dallas Cowboys", "NFC", "East", -6),
    "DEN": ("Denver Broncos", "AFC", "West", -7),
    "DET": ("Detroit Lions", "NFC", "North", -5),
    "GB": ("Green Bay Packers", "NFC", "North", -6),
    "HOU": ("Houston Texans", "AFC", "South", -6),
    "IND": ("Indianapolis Colts", "AFC", "South", -5),
    "JAX": ("Jacksonville Jaguars", "AFC", "South", -5),
    "KC": ("Kansas City Chiefs", "AFC", "West", -6),
    "LA": ("Los Angeles Rams", "NFC", "West", -8),
    "LAC": ("Los Angeles Chargers", "AFC", "West", -8),
    "LV": ("Las Vegas Raiders", "AFC", "West", -8),
    "MIA": ("Miami Dolphins", "AFC", "East", -5),
    "MIN": ("Minnesota Vikings", "NFC", "North", -6),
    "NE": ("New England Patriots", "AFC", "East", -5),
    "NO": ("New Orleans Saints", "NFC", "South", -6),
    "NYG": ("New York Giants", "NFC", "East", -5),
    "NYJ": ("New York Jets", "AFC", "East", -5),
    "PHI": ("Philadelphia Eagles", "NFC", "East", -5),
    "PIT": ("Pittsburgh Steelers", "AFC", "North", -5),
    "SEA": ("Seattle Seahawks", "NFC", "West", -8),
    "SF": ("San Francisco 49ers", "NFC", "West", -8),
    "TB": ("Tampa Bay Buccaneers", "NFC", "South", -5),
    "TEN": ("Tennessee Titans", "AFC", "South", -6),
    "WAS": ("Washington Commanders", "NFC", "East", -5),
}

# Where a franchise played before it moved, by the code it used then.
OLD_HOME_OFFSET = {"STL": -6, "SD": -8, "OAK": -8}

TEAM_COLUMNS = [
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    "attempts", "sacks_suffered", "carries", "passing_epa", "rushing_epa",
    "passing_yards", "rushing_yards", "passing_interceptions",
    "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
    "penalty_yards",
]
QB_COLUMNS = [
    "player_id", "player_display_name", "season", "week", "season_type",
    "game_id", "team", "opponent_team", "attempts", "completions",
    "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered",
    "passing_epa", "carries", "rushing_yards", "rushing_tds", "rushing_epa",
]
INJURY_COLUMNS = [
    "season", "game_type", "team", "week", "gsis_id", "position", "full_name",
    "report_primary_injury", "report_status", "practice_status",
]


def franchise(code):
    """Today's code for whatever a season called the team."""
    return FRANCHISE.get(code, code)


def home_offset(code_then):
    """UTC offset of the team's home city, as the code it used that season."""
    if code_then in OLD_HOME_OFFSET:
        return OLD_HOME_OFFSET[code_then]
    return TEAMS[franchise(code_then)][3]


def kickoff_utc(gameday, gametime):
    """Eastern local kickoff from the schedule -> naive UTC timestamp.

    A missing time is read as 13:00 Eastern, the standard Sunday slot, so a
    game still to be flexed sorts with its week rather than at midnight.
    """
    day = str(gameday)[:10]
    clock = str(gametime) if isinstance(gametime, str) and ":" in gametime else "13:00"
    local = datetime.fromisoformat(f"{day}T{clock[:5]}").replace(tzinfo=EASTERN)
    return pd.Timestamp(local.astimezone(timezone.utc).replace(tzinfo=None))


def _fetch(url):
    """The body of one nflverse file. requests if installed, else urllib."""
    try:
        import requests
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content
    except ImportError:
        from urllib.request import urlopen
        with urlopen(url, timeout=120) as handle:
            return handle.read()


def _read(url, **kwargs):
    return pd.read_csv(io.BytesIO(_fetch(url)), low_memory=False, **kwargs)


def _write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so an unchanged snapshot is byte-identical and git sees no change.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False).encode())
    return path


def _merge_seasons(old_path, fresh, seasons, key):
    """The snapshot with `seasons` replaced by `fresh`."""
    if old_path.exists():
        old = pd.read_csv(old_path, low_memory=False)
        old = old[~old["season"].isin(seasons)]
        fresh = pd.concat([old, fresh], ignore_index=True)
    return fresh.sort_values(key).reset_index(drop=True)


def current_season(today=None):
    """The season in progress: September to February belongs to the year it started."""
    today = today or datetime.now(timezone.utc).date()
    return today.year if today.month >= 3 else today.year - 1


def sync(seasons=None, full=False, log=print):
    """Refresh the snapshot from nflverse.

    The schedule is always fetched whole (it is one 2 MB file and carries
    the lines for the games still to come). Team, quarterback and injury
    files are fetched for `seasons` - by default the current one - unless
    `full`, which rebuilds every season since 1999.
    """
    now = current_season()
    seasons = list(range(FIRST_SEASON, now + 1)) if full else (seasons or [now])

    games = _read(GAMES_URL)
    missing = {"game_id", "season", "week", "gameday", "gametime", "home_team",
               "away_team", "home_score", "away_score", "spread_line"} - set(games.columns)
    if missing:
        raise RuntimeError(f"nflverse schedule lost columns {sorted(missing)}")
    _write(games, GAMES)
    log(f"   schedule: {len(games):,} games, seasons {games.season.min()}-{games.season.max()}")

    team, qbs, injuries = [], [], []
    for season in seasons:
        t = _read(TEAM_URL.format(season=season))
        t = t[t["team"].notna()]
        team.append(t[[c for c in TEAM_COLUMNS if c in t.columns]])
        p = _read(PLAYER_URL.format(season=season))
        qbs.append(p[p["position"] == "QB"][[c for c in QB_COLUMNS if c in p.columns]])
        if season >= 2009:   # nflverse's injury reports start in 2009
            try:
                i = _read(INJURY_URL.format(season=season))
                injuries.append(i[[c for c in INJURY_COLUMNS if c in i.columns]])
            except Exception as err:   # an off-season with no file yet
                log(f"   injuries {season}: none ({err})")
        log(f"   {season}: {len(t):,} team games, {len(qbs[-1]):,} QB games")

    _write(_merge_seasons(TEAM_GAMES, pd.concat(team), seasons, ["season", "week", "game_id", "team"]), TEAM_GAMES)
    _write(_merge_seasons(QB_GAMES, pd.concat(qbs), seasons, ["season", "week", "game_id", "player_id"]), QB_GAMES)
    if injuries:
        # Only recent reports are worth keeping: they are shown beside this
        # week's games and never used as a model input.
        fresh = pd.concat(injuries)
        kept = _merge_seasons(INJURIES, fresh, seasons, ["season", "week", "team", "full_name"])
        _write(kept[kept["season"] >= now - 1], INJURIES)
    return {"games": len(games), "seasons": seasons}


def load_games(path=GAMES):
    """The schedule, one row per game, in franchise codes, with UTC kickoffs.

    Columns: game_id, season, week, game_type, kickoff, home, away,
    home_then, away_then (the codes used that season), home_score,
    away_score, result (home margin), total, played, spread_line,
    total_line, home_ml, away_ml, div_game, neutral, roof, surface,
    home_rest, away_rest, home_qb_id, away_qb_id, home_qb, away_qb,
    home_coach, away_coach, stadium.
    """
    g = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "game_id": g["game_id"],
        "season": g["season"].astype(int),
        "week": g["week"].astype(int),
        "game_type": g["game_type"],
        "kickoff": [kickoff_utc(d, t) for d, t in zip(g["gameday"], g["gametime"])],
        "home_then": g["home_team"],
        "away_then": g["away_team"],
        "home": g["home_team"].map(franchise),
        "away": g["away_team"].map(franchise),
        "home_score": g["home_score"],
        "away_score": g["away_score"],
        "spread_line": g["spread_line"],
        "total_line": g["total_line"],
        "home_ml": g.get("home_moneyline"),
        "away_ml": g.get("away_moneyline"),
        "div_game": g.get("div_game", 0).fillna(0).astype(int),
        "neutral": (g.get("location") == "Neutral").astype(int),
        "roof": g.get("roof"),
        "surface": g.get("surface"),
        "home_rest": g.get("home_rest"),
        "away_rest": g.get("away_rest"),
        "home_qb_id": g.get("home_qb_id"),
        "away_qb_id": g.get("away_qb_id"),
        "home_qb": g.get("home_qb_name"),
        "away_qb": g.get("away_qb_name"),
        "home_coach": g.get("home_coach"),
        "away_coach": g.get("away_coach"),
        "stadium": g.get("stadium"),
    })
    out["played"] = out["home_score"].notna() & out["away_score"].notna()
    out["result"] = out["home_score"] - out["away_score"]
    out["total"] = out["home_score"] + out["away_score"]
    unknown = set(out["home"]) | set(out["away"])
    unknown -= set(TEAMS)
    if unknown:
        raise ValueError(f"teams with no franchise entry: {sorted(unknown)}")
    return out.sort_values(["kickoff", "game_id"]).reset_index(drop=True)


def load_team_games(path=TEAM_GAMES):
    """One row per team per game: plays, EPA, yards and giveaways on offense."""
    t = pd.read_csv(path, low_memory=False)
    t = t[t["team"].notna()].copy()
    t["dropbacks"] = t["attempts"].fillna(0) + t["sacks_suffered"].fillna(0)
    t["plays"] = t["dropbacks"] + t["carries"].fillna(0)
    t["epa"] = t["passing_epa"].fillna(0) + t["rushing_epa"].fillna(0)
    lost = [c for c in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost") if c in t]
    t["giveaways"] = t["passing_interceptions"].fillna(0) + t[lost].fillna(0).sum(axis=1)
    return t.rename(columns={"opponent_team": "opponent"}).reset_index(drop=True)


def load_qb_games(path=QB_GAMES):
    """One row per quarterback per game he recorded a stat in."""
    q = pd.read_csv(path, low_memory=False)
    q["dropbacks"] = q["attempts"].fillna(0) + q["sacks_suffered"].fillna(0)
    q["epa"] = q["passing_epa"].fillna(0) + q["rushing_epa"].fillna(0)
    return q


def load_injuries(path=INJURIES):
    if not Path(path).exists():
        return pd.DataFrame(columns=INJURY_COLUMNS)
    return pd.read_csv(path, low_memory=False)


if __name__ == "__main__":
    full = "--full" in sys.argv
    seasons = [int(a) for a in sys.argv[1:] if a.isdigit()] or None
    print("Syncing nflverse" + (" (every season)" if full else ""))
    sync(seasons=seasons, full=full)
