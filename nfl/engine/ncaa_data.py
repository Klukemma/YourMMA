"""College football (NCAA FBS) data: schedules, scores and betting lines from
sportsdataverse's public releases (github.com/sportsdataverse/
sportsdataverse-data, built from ESPN), and the live week from ESPN's FBS
scoreboard.

Snapshotted into nfl/data/ncaa/ so history never depends on the network, the
way nfl_data.py keeps nflverse:

    schedules/{season}.csv.gz   the games with at least one FBS team, as the
                                cfb_schedules release publishes them (text kept)
    lines/{season}.csv.gz       the espn_cfb_betting release's rows, as published
    teams.csv.gz                display info: names, abbreviations, colours, logo
    week.json                   the live board: games, kickoffs, status, scores and
                                ESPN's current line, with the moment it was seen

One file per season, gzip with mtime 0, written only when its CSV text
changed, so a sync rewrites only what changed and git sees nothing else. A
failed, truncated or unreadable download keeps what is on file (see sync()).
Everything downstream reads the snapshot through load_games(), load_week()
and load_teams().

WHAT THE PROBE SHOWED (probe_ncaa.py on GitHub's machines, 2026-10-30
20:43 UTC; its raw answers were reduced to tests/fixtures/ncaa/ and removed).

  sportsdataverse releases, free, no key. The ones used here:
    cfb_schedules     cfb_schedules_{2001..2026}.csv.gz (also parquet/rds):
                      every division, one row per ESPN game id, 43 columns
                      (season, week, season_type, start_date, neutral_site,
                      conference_game, home/away id, team, abbreviation,
                      division, conference, points, rank, playoff_*, notes...)
    espn_cfb_betting  betting_{2004..2026}.csv (fresh: rewritten 2026-10-02..30)
                      and betting_{2004..2025}.csv.gz (stale copies of July
                      2026; same byte length as the .csv where compared), one
                      row per game: game_spread, over_under, home_favorite,
                      home_team_spread, game_spread_available, odds_source.
                      No moneyline, no provider, no timestamp.
  Kept for later, not synced yet: espn_cfb_teams (cfb_teams_{season}.csv,
  ESPN branding as of each season), cfb_team_info, espn_cfb_team_box,
  cfb_fpi_weekly, espn_cfb_power_index, cfb_ratings(_weekly), cfb_team_talent,
  cfb_returning_production, cfb_recruits, espn_cfb_injuries (2026 only),
  ncaa_mfb_pbp_cfbfastr (2013+). collegefootballdata.com answers 401 without
  a key: not used.

  Per season, games with an FBS team (FBS v FBS + FBS v FCS), postseason,
  and games with a real spread (FBS v FBS of those):
    2010    808 (718 + 90), 35 post,    0 lines - all 810 betting rows are
            "default" placeholders (2.5 / 55.5, game_spread_available false)
    2016    873 (760 + 113), 41 post, 108 lines (3) - of its 879 betting rows,
            108 have numbers (core odds API; 105 of them FBS v FCS), 762
            "summary_pickcenter" rows carry who was favoured and no spread
            (759 no total either; 3 a total of exactly 55.0 - Troy-Austin
            Peay, Rutgers-Howard and the postponed ECU-Navy - filler like the
            placeholders' 55.5, so not loaded), 9 are placeholders (3 of
            them all-star games)
    2024    920 (799 + 121), 46 post, 910 lines (794)
    2025    934 (808 + 126), 46 post, 924 lines (808)
    2026    888 (761 + 127), regular season so far: 572 played, 572 lines (450)
  The other seasons were not probed. The release's .csv.gz copies compress
  like the probed ones: 2004-2011 like 2010 (ratio 0.072-0.074: placeholders
  only), 2012-2022 like 2016 (0.082-0.096: favourite flags, few numbers),
  2023 like 2024 (0.142). So, an inference until the runner's full sync logs
  each season's count: the release has a market line for nearly every FBS
  game from 2023 on, and almost none before. ESPN's core odds API itself does
  hold full lines for 2016 games (10-12 providers each, an "Opening" one
  included) where the release has none, and nothing at all for 2010 games:
  a per-game backfill from it is the way to older lines, not built here.

DECISIONS.

  SCOPE. A game is kept if at least one team is FBS (ESPN group 80, per
  season, from the schedule's home/away_division). FBS v FBS games have
  fbs_game = 1. FBS v FCS games are kept with fcs_game = 1: they count in an
  FBS team's record and carry a line in 105 (2016), 116 (2024, 2025) and
  122 (2026 so far) of them - none in 2010 - but the FCS side is mostly
  unrated. In every probed season the non-FBS side is FCS (one 2010 game
  has the FCS team at home); the division columns say so per game. FCS v FCS, Division II/III and the all-star games ("offseason",
  no division) are left out. Season types: regular (conference
  championships are regular season, weeks 14-16) and postseason (bowls and
  the College Football Playoff, all filed as postseason week 1).

  NEUTRAL SITES AND HOME FIELD. `neutral` is ESPN's flag as published:
  63-80 games a season, most bowls (41 of 46 in 2024; the four CFP
  first-round games are on campus) and the kickoff classics. On a neutral
  site "home" is only the designated side. The flag is wrong in a few games,
  so `home_field` says who really is at home, from the stadium: +1 the
  designated home team, -1 the designated AWAY team, 0 neither. A team's
  stadium is the venue of most of its non-neutral home games that season
  (ties all count). Only a game at one team's own stadium overrides the
  flag; any other game (a "home" game in Little Rock, a venue missing)
  keeps it (+1 not neutral, 0 neutral). In the probed seasons 13 of 4,423
  games differ:
    - the designated away team at its own stadium, -1: the 2024 Myrtle
      Beach Bowl (UTSA "home", not neutral, at Coastal Carolina's Brooks
      Stadium), the 2025-season CFP title game (Indiana "home", at Miami's
      Hard Rock Stadium), the 2010 and 2016 Hawai'i Bowls (at Aloha
      Stadium) and the 2016 New Mexico Bowl (at New Mexico's own stadium);
    - FBS v FCS games filed neutral at the FBS team's own stadium, +1: 2010
      Rutgers-Norfolk State, Tulane-SE Louisiana, San Diego State-Nicholls,
      Baylor-Sam Houston, Houston-Texas State, Hawai'i-Charleston Southern;
      2016 UNLV-Jackson State, San Diego State-New Hampshire.
  None in 2024-2026's regular seasons; the runner's full sync logs the
  count per season. The model reads home_field, not neutral.

  THE HISTORICAL LINE IS MOST LIKELY THE CLOSE: ESPN's main provider's last
  number before kickoff. The release calls itself "closing betting lines and
  odds" but does not say how, so it was checked:
    - 2026: betting_2026.csv (written 2026-10-30 08:34) holds exactly the
      572 completed FBS games, Thursday night's included, and no game still
      to come: rows are written after the game.
    - Every probed game where ESPN's core odds API keeps an open and a close
      that differ matches the CLOSE: 2024 Alabama-WKU (open -31, close and
      release -37.5), 2025 Florida State-Alabama (total open 50.5, close and
      release 46.5), 2026 Alabama-East Carolina (open -25.5/54.5, close and
      release -28.5/52.5). Not the in-play "ESPN Bet - Live Odds" provider
      (FSU -17.5 at its last update).
    - 2016 Baylor-Northwestern State: release -50/81 = 5Dimes' (and
      BETNOW's) number, not the "Opening" provider (-48.5/78.5) or the
      consensus (-48.5).
  The provider is not recorded per row; in the probe it was DraftKings
  (2026), ESPN BET (2024-2025) and 5Dimes (2016). A row with odds_source
  "default" or game_spread_available false is a placeholder, and the
  pickcenter rows (2016; by the inference above, 2012-2022) have no spread:
  both load as MISSING lines (has_line False), never as 2.5 / 55.5 - the
  rule is the source flag, not the numbers (Delaware at WKU, 2026, really
  closed -2.5 / 55.5). A placeholder's home_favorite is meaningless and is
  dropped too. The pickcenter favourite flag is real
  information (2016: favourites won 73.1% against 58.2% for home teams) and
  is kept as home_favorite with no spread. A total loads only alongside a
  spread: the only totals without one in the probe are the three 55.0
  fillers above; has_line means a spread, and the sync logs how many
  totals were dropped this way per season.

  RANKS (home_rank, away_rank) are ESPN's Top 25 as of each game: they
  change week to week (Alabama 2024: 5, 4, 4, 4, 1, 7, 7, 15...), and a
  game not yet played carries the current ranking as of the sync. NaN is
  unranked OR MISSING: 2016 has gaps, whole games with neither rank -
  Alabama, No. 1 all season, has none in weeks 4, 5 and 8, Ohio State -
  Michigan (both top 3) none either. 11 games (13 teams) have a team ranked
  in the top 15 in its previous and next games and no rank (0 in 2010 and
  2024-2026). summary()'s rank_gaps counts those per season, and the sync
  logs it; a gap of consecutive games (Alabama's weeks 4-5) is missed.

  LIVE AGAINST HISTORY: the same ESPN odds feed. The scoreboard's
  competitions[].odds names provider 100 "Draft Kings" (priority 1), the
  provider of the 2026 historical rows. Before kickoff the board's number is
  the CURRENT line at fetch time (the board labels it "close" even then), so
  a pick made from it is made from an earlier, softer line than the close
  the history measures against. The board drops a game's odds once it is
  final; week.json keeps the last line seen before kickoff and when it was
  seen (line_seen_at), and never takes a line from a game already under way.
  The board also carries moneylines and the opening line; history has
  neither.

  KICKOFFS ARE UTC. The schedule's start_date ("2026-09-05T16:00:00.000Z",
  some older rows "2011-01-22T21:00Z") and the board's date
  ("2026-10-29T23:30Z") are UTC with a Z; all 56 games of the probed week
  agree to the minute. They load as naive UTC timestamps, as nfl_data's do.
  A kickoff time not yet set (start_time_tbd, the board's timeValid false)
  is filed as midnight Eastern on the day (04:00Z/05:00Z) and loads with
  time_tbd True: a date, not a moment.

  SPREAD CONVENTION as nfl_data's spread_line: from the HOME side, positive
  = home favoured by that many points (it predicts home score - away
  score). The release's home_team_spread and ESPN's "spread" are the home
  handicap (negative = home favoured), so spread_line = -home_team_spread.

  TEAMS are ESPN team ids everywhere (schedule, lines, board). Display info
  comes from ESPN's teams list (limit=1000: the probe's limit=300 returned
  the first 300 teams of every division in the string order of their ids -
  2000, 2001, ... 272, 402, 43 - and groups=80 was ignored: 96 of 138 FBS
  teams), topped up by every team the board shows (each competitor
  carries name, abbreviation, colours and logo). A team seen in neither has
  the schedule's name and abbreviation and no colours: missing, not guessed.

Run on GitHub's machines (the sandbox has no internet):

    python nfl/engine/ncaa_data.py            # this season (and last, in March), teams, live board
    python nfl/engine/ncaa_data.py --full     # every season since 2004

ncaa-sync runs only when dispatched by hand for now: the picks step will
schedule it (or call sync() from its own run) before each slate.
"""

import gzip
import io
import json
import os
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

NFL = Path(__file__).resolve().parents[1]
DATA = NFL / "data" / "ncaa"

REPO = "sportsdataverse/sportsdataverse-data"
RELEASE_API = "https://api.github.com/repos/" + REPO + "/releases/tags/{tag}"
DOWNLOAD = "https://github.com/" + REPO + "/releases/download/{tag}/{name}"
SCHEDULE_TAG = "cfb_schedules"
BETTING_TAG = "espn_cfb_betting"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
BOARD_URL = ESPN + "/scoreboard?groups=80&limit=300"
TEAMS_URL = ESPN + "/teams?limit=1000"
UA = {"User-Agent": "YourNFL college football sync (github.com/Klukemma/YourMMA)"}

FIRST_SEASON = 2004          # the betting release starts here
SHRINK_TOLERANCE = 0.95      # a season returning fewer rows than this share of the file is kept

SCHEDULE_COLUMNS = [
    "game_id", "season", "week", "season_type", "start_date", "start_time_tbd",
    "completed", "status", "neutral_site", "conference_game", "venue_id", "venue",
    "home_id", "home_team", "home_abbreviation", "home_division", "home_conference",
    "home_points", "away_id", "away_team", "away_abbreviation", "away_division",
    "away_conference", "away_points", "fbs_game", "notes", "playoff_competition",
    "playoff_round_name", "playoff_bowl_name", "playoff_home_seed", "playoff_away_seed",
    "home_rank", "away_rank",
]
SCHEDULE_REQUIRED = {"game_id", "season", "week", "season_type", "start_date",
                     "home_id", "home_team", "home_division", "home_points",
                     "away_id", "away_team", "away_division", "away_points",
                     "neutral_site", "completed"}
LINE_REQUIRED = {"game_id", "home_team_spread", "over_under"}
TEAM_COLUMNS = ["team_id", "name", "short_name", "abbreviation", "location", "nickname",
                "color", "alt_color", "logo"]
GAME_COLUMNS = [
    "game_id", "season", "week", "season_type", "postseason", "kickoff", "time_tbd",
    "home_id", "away_id", "home", "away", "home_abbr", "away_abbr",
    "home_division", "away_division", "home_conference", "away_conference",
    "fbs_game", "fcs_game", "neutral", "home_field", "conference_game", "home_rank", "away_rank",
    "home_score", "away_score", "played", "result", "total", "status", "venue_id", "venue", "notes",
    "playoff_round", "spread_line", "total_line", "home_favorite", "has_line", "line_source",
]
SEASON_TYPES = {1: "preseason", 2: "regular", 3: "postseason", 4: "offseason"}
LINE_FIELDS = ["provider", "provider_id", "line_details", "spread_line", "home_spread_odds",
               "away_spread_odds", "total_line", "over_odds", "under_odds", "home_ml", "away_ml",
               "spread_open", "total_open", "home_ml_open", "away_ml_open", "line_seen_at"]


# ------------------------------------------------------------------ basics

def current_season(today=None):
    """The season in progress: March to February belongs to the year it
    starts in, so January's bowls and playoff are last year's season."""
    today = today or datetime.now(timezone.utc).date()
    return today.year if today.month >= 3 else today.year - 1


def _path(root, kind, season):
    return Path(root) / kind / f"{int(season)}.csv.gz"


def seasons_on_file(root=DATA, kind="schedules"):
    folder = Path(root) / kind
    if not folder.is_dir():
        return []
    return sorted(int(p.name.split(".")[0]) for p in folder.glob("*.csv.gz")
                  if p.name.split(".")[0].isdigit())


def _csv(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode()


def _write(body, path):
    """gzip with mtime 0 and no file name in it: the same CSV text is the
    same bytes, so git sees no change."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, 9, mtime=0))
    return path


def _on_file(path):
    try:
        return gzip.decompress(Path(path).read_bytes())
    except (OSError, EOFError, zlib.error):
        return None


def _text_frame(body):
    """CSV bytes -> every cell as the text it was published as."""
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False, na_filter=False)


def _by_game_id(frame):
    key = pd.to_numeric(frame["game_id"], errors="coerce")
    return frame.assign(_key=key).sort_values(["_key"], kind="mergesort").drop(columns="_key") \
        .reset_index(drop=True)


def store(frame, path, what, log=print, problems=None):
    """One season's rows -> its file, written only when the CSV text changed
    (compared uncompressed). A season that comes back with fewer than 95% of
    the rows on file is kept as it is - a cut-short download must not cost
    history - and reported; to take a shorter season on purpose, delete its
    file and sync. -> "written" | "unchanged" | "kept"."""
    body, old = _csv(frame), _on_file(path)
    if old == body:
        return "unchanged"
    if old is not None:
        on_file = len(_text_frame(old))
        if len(frame) < SHRINK_TOLERANCE * on_file:
            if problems is not None:
                problems.append(what)
            log(f"   {what}: {len(frame)} rows, FEWER than the {on_file} on file; kept what is on file")
            return "kept"
    _write(body, path)
    return "written"


# ------------------------------------------------------------------ network

def whole(body, headers):
    """The body, or IOError when it is shorter or longer than the response's
    own Content-Length (checked only for a body sent as is, not one the
    HTTP layer decoded from gzip or deflate) - the size check for when
    GitHub's API does not answer."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    encoding = (headers.get("content-encoding") or "identity").strip().lower()
    length = headers.get("content-length")
    if encoding == "identity" and length and str(length).strip().isdigit() \
            and len(body) != int(length):
        raise IOError(f"cut short: {len(body):,} of {int(length):,} bytes")
    return body


def _fetch(url, headers=None):
    """The body of one URL; None for a 404 (not published). Raises on
    anything else - a body shorter than its Content-Length included - which
    the caller logs and survives."""
    headers = {**UA, **(headers or {})}
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        response = requests.get(url, timeout=120, headers=headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return whole(response.content, response.headers)
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    try:
        with urlopen(Request(url, headers=headers), timeout=120) as handle:
            return whole(handle.read(), dict(handle.headers.items()))
    except HTTPError as err:
        if err.code == 404:
            return None
        raise


def release_assets(tag, fetch=_fetch, log=print):
    """{asset name: (download url, size in bytes)} for one release, from
    GitHub's API (with GITHUB_TOKEN when set, for the rate limit; the token
    is never logged). None when the API does not answer: the caller falls
    back to the fixed download URL and cannot check sizes."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        body = fetch(RELEASE_API.format(tag=tag), headers=headers)
        if not body:
            raise ValueError("no release")
        assets = json.loads(body)["assets"]
        return {a["name"]: (a["browser_download_url"], int(a["size"])) for a in assets}
    except Exception as err:                       # noqa: BLE001 - logged, fixed URLs instead
        log(f"   release {tag}: GitHub's API did not answer ({err.__class__.__name__}); "
            f"using the fixed download URLs, without size checks")
        return None


def cut_off(text):
    """Whether CSV text stops part-way through its last row: more than one
    line and no line break at the end. Every file the probe saw ends with
    one, so a body without it was cut - in its last field as much as
    between fields. (Were the release ever to drop the final line break,
    each sync would keep the file on file and report it: loud, not lost.)"""
    if not text or text.endswith(b"\n"):
        return False
    return b"\n" in text


def download(tag, candidates, assets, required, what, fetch=_fetch, log=print, problems=None):
    """One season's file from a release -> its rows as text, or None.

    The first candidate name the release lists (or, without the API, the
    first fixed URL that answers). A body whose length differs from the
    listed size, a gzip that does not decompress, CSV text cut off mid-row
    or a file missing its columns is a problem: logged, listed, and what is
    on file is kept. A 404 or a name the release does not list is "not
    published" (a season not started)."""
    problems = [] if problems is None else problems
    if assets is not None:
        names = [n for n in candidates if n in assets]
        if not names:
            log(f"   {what}: not published")
            return None
        tries = [(names[0], *assets[names[0]])]
    else:
        tries = [(n, DOWNLOAD.format(tag=tag, name=n), None) for n in candidates]
    for name, url, size in tries:
        try:
            body = fetch(url)
        except Exception as err:                   # noqa: BLE001 - logged, kept, run continues
            problems.append(what)
            log(f"   {what}: download failed ({err.__class__.__name__}: {str(err)[:120]}); "
                f"kept what is on file")
            return None
        if body is None:
            if assets is not None:                 # listed, yet not there
                problems.append(what)
                log(f"   {what}: the release lists {name} but it answered 404; kept what is on file")
                return None
            continue
        if size is not None and len(body) != size:
            problems.append(what)
            log(f"   {what}: {len(body):,} bytes of {size:,} - CUT SHORT; kept what is on file")
            return None
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError, zlib.error):
                problems.append(what)
                log(f"   {what}: the gzip does not decompress; kept what is on file")
                return None
        if cut_off(body):
            problems.append(what)
            log(f"   {what}: CUT OFF PART-WAY THROUGH ITS LAST ROW; kept what is on file")
            return None
        try:
            rows = _text_frame(body)
        except Exception as err:                   # noqa: BLE001
            problems.append(what)
            log(f"   {what}: not a CSV ({err.__class__.__name__}); kept what is on file")
            return None
        missing = required - set(rows.columns)
        if missing:
            problems.append(what)
            log(f"   {what}: lost columns {sorted(missing)}; kept what is on file")
            return None
        return rows
    log(f"   {what}: not published")
    return None


# ------------------------------------------------------------------ the schedule and lines

def in_scope(rows):
    """The games with at least one FBS team."""
    fbs = (rows["home_division"] == "fbs") | (rows["away_division"] == "fbs")
    return rows[fbs]


def schedule_rows(rows):
    """A published schedule (text) -> the rows and columns kept on file."""
    rows = in_scope(rows)
    return _by_game_id(rows[[c for c in SCHEDULE_COLUMNS if c in rows.columns]])


def sync_season(season, assets, fetch=_fetch, log=print, root=DATA, problems=None):
    """One season's schedule and lines. -> {"schedule": status, "lines": status}"""
    out = {}
    rows = download(SCHEDULE_TAG, [f"cfb_schedules_{season}.csv.gz"], assets.get(SCHEDULE_TAG),
                    SCHEDULE_REQUIRED, f"schedule {season}", fetch, log, problems)
    out["schedule"] = "not fetched" if rows is None else \
        store(schedule_rows(rows), _path(root, "schedules", season), f"schedule {season}", log, problems)
    # The plain .csv first: it is the one the release keeps current (the
    # .csv.gz copies stopped in July 2026, and 2026 has none).
    rows = download(BETTING_TAG, [f"betting_{season}.csv", f"betting_{season}.csv.gz"],
                    assets.get(BETTING_TAG), LINE_REQUIRED, f"lines {season}", fetch, log, problems)
    out["lines"] = "not fetched" if rows is None else \
        store(_by_game_id(rows), _path(root, "lines", season), f"lines {season}", log, problems)
    return out


def _flag(series):
    """'true'/'false' text (or bools) -> True/False, anything else NaN."""
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    return series.astype(str).str.strip().str.lower().map(m)


def _true(series):
    """'true' text -> True; false, blank or anything else -> False."""
    return _flag(series).eq(True)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _txt(series):
    return series.replace("", np.nan)


def _read_seasons(root, kind, seasons):
    parts = []
    for s in seasons:
        p = _path(root, kind, s)
        if p.exists():
            parts.append(pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def lines_frame(raw):
    """Betting rows (text) -> game_id, spread_line, total_line,
    home_favorite, has_line, line_source. Placeholders are missing, and a
    total loads only alongside a spread (the 2016 pickcenter rows' lone
    55.0 totals are filler)."""
    b = raw.copy()
    b["game_id"] = _num(b["game_id"]).astype("Int64")
    b = b.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
    source = _txt(b["odds_source"]) if "odds_source" in b else pd.Series(np.nan, index=b.index)
    avail = _flag(b["game_spread_available"]) if "game_spread_available" in b \
        else pd.Series(True, index=b.index)
    placeholder = (source == "default") | avail.eq(False)
    hts = _num(b["home_team_spread"])
    ou = _num(b["over_under"])
    fav = _flag(b["home_favorite"]) if "home_favorite" in b else pd.Series(np.nan, index=b.index)
    out = pd.DataFrame({
        "game_id": b["game_id"],
        "spread_line": (-hts).where(~placeholder),
        "total_line": ou.where(~placeholder & hts.notna()),
        "home_favorite": fav.where(~placeholder).map({True: 1.0, False: 0.0}),
        "line_source": source,
        "total_alone": ~placeholder & ou.notna() & hts.isna(),   # not loaded; the sync counts them
    })
    out["has_line"] = out["spread_line"].notna()
    # 0.0, not -0.0, for a pick'em
    out["spread_line"] = out["spread_line"] + 0.0
    return out


def kickoffs(text):
    """ESPN's UTC times - '2026-09-05T16:00:00.000Z', '2011-01-22T21:00Z',
    '2026-10-29T23:30Z' - as naive UTC timestamps."""
    return pd.to_datetime(pd.Series(text), utc=True, format="ISO8601").dt.tz_convert(None)


def _ids(values):
    """ESPN ids as text, numbers or NaN -> a list of int or None."""
    return [None if pd.isna(v) else int(v) for v in pd.to_numeric(pd.Series(list(values)), errors="coerce")]


def home_venues(season, home_id, venue_id, neutral):
    """{(season, team id): {venue ids}} - each team's own stadium that
    season: the venue of most of its non-neutral home games (ties all
    count). A team never at home on a non-neutral site has none."""
    f = pd.DataFrame({"season": _ids(season), "team": _ids(home_id), "venue": _ids(venue_id),
                      "neutral": [bool(n) for n in neutral]})
    f = f[~f["neutral"]].dropna(subset=["season", "team", "venue"])
    n = f.groupby(["season", "team", "venue"]).size().rename("n").reset_index()
    n = n[n["n"] == n.groupby(["season", "team"])["n"].transform("max")]
    out = {}
    for s, t, v in zip(n["season"], n["team"], n["venue"]):
        out.setdefault((int(s), int(t)), set()).add(int(v))
    return out


def home_field(season, home_id, away_id, venue_id, neutral, venues):
    """+1 the designated home team is at its own stadium (or, the venue
    telling nothing, ESPN does not flag the game neutral), -1 the designated
    away team is at its own stadium, 0 neither (or a stadium both call
    home). Only a game at one side's own stadium overrides ESPN's neutral
    flag (see the module notes). -> a list of ints"""
    out = []
    for s, h, a, v, nt in zip(_ids(season), _ids(home_id), _ids(away_id), _ids(venue_id), neutral):
        at_home = v is not None and v in venues.get((s, h), ())
        at_away = v is not None and v in venues.get((s, a), ())
        if at_home != at_away:
            out.append(1 if at_home else -1)
        elif at_home:
            out.append(0)
        else:
            out.append(0 if bool(nt) else 1)
    return out


def load_games(seasons=None, root=DATA):
    """Every game with an FBS team, one row per game, in kickoff order.

    Columns: game_id (ESPN event id), season, week, season_type ('regular'
    / 'postseason'), postseason (0/1), kickoff (naive UTC), time_tbd (the
    kickoff is a date only), home_id, away_id (ESPN team ids), home, away
    (names), home_abbr, away_abbr, home/away_division ('fbs', 'fcs', ...),
    home/away_conference (as of that season), fbs_game (both FBS, 0/1),
    fcs_game (one side not FBS, 0/1), neutral (ESPN's flag, 0/1), home_field
    (+1 home team at its own stadium, -1 the away team at its own, 0 neither:
    what a model should read - see the module notes), conference_game (0/1),
    home_rank, away_rank (ESPN's Top 25 as of the game; NaN is unranked OR
    missing - 2016 has gaps, see rank_gaps in summary()), home_score,
    away_score, played (completed with both scores), result (home - away),
    total, status, venue_id, venue, notes (bowl or classic name),
    playoff_round, spread_line (positive = home favoured, the close - see the
    module notes), total_line (only with a spread), home_favorite (1.0/0.0,
    NaN unknown), has_line (a spread), line_source (the release's
    odds_source; NaN when it has no row for the game).
    No moneylines: the history has none.
    """
    root = Path(root)
    seasons = seasons_on_file(root) if seasons is None else sorted(int(s) for s in seasons)
    g = _read_seasons(root, "schedules", seasons)
    if g is None:
        return pd.DataFrame(columns=GAME_COLUMNS)
    g = g.copy()
    for c in SCHEDULE_COLUMNS:
        if c not in g:
            g[c] = ""
    g["game_id"] = _num(g["game_id"]).astype("Int64")
    g = g.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
    kickoff = kickoffs(g["start_date"])
    home_fbs = g["home_division"] == "fbs"
    away_fbs = g["away_division"] == "fbs"
    status = _txt(g["status"])
    completed = _true(g["completed"])
    out = pd.DataFrame({
        "game_id": g["game_id"].astype("int64"),
        "season": _num(g["season"]).astype(int),
        "week": _num(g["week"]).astype(int),
        "season_type": g["season_type"],
        "postseason": (g["season_type"] == "postseason").astype(int),
        "kickoff": kickoff,
        "time_tbd": _true(g["start_time_tbd"]),
        "home_id": _num(g["home_id"]).astype("Int64"),
        "away_id": _num(g["away_id"]).astype("Int64"),
        "home": g["home_team"],
        "away": g["away_team"],
        "home_abbr": _txt(g["home_abbreviation"]),
        "away_abbr": _txt(g["away_abbreviation"]),
        "home_division": _txt(g["home_division"]),
        "away_division": _txt(g["away_division"]),
        "home_conference": _txt(g["home_conference"]),
        "away_conference": _txt(g["away_conference"]),
        "fbs_game": (home_fbs & away_fbs).astype(int),
        "fcs_game": (home_fbs ^ away_fbs).astype(int),
        "neutral": _true(g["neutral_site"]).astype(int),
        "venue_id": _num(g["venue_id"]).astype("Int64"),
        "conference_game": _true(g["conference_game"]).astype(int),
        "home_rank": _num(g["home_rank"]),
        "away_rank": _num(g["away_rank"]),
        "home_score": _num(g["home_points"]),
        "away_score": _num(g["away_points"]),
        "status": status,
        "venue": _txt(g["venue"]),
        "notes": _txt(g["notes"]),
        "playoff_round": _txt(g["playoff_round_name"]),
    })
    # A cancelled game can carry 0-0 and is not completed (2024 App State -
    # Liberty): played needs both.
    venues = home_venues(out["season"], out["home_id"], out["venue_id"], out["neutral"])
    out["home_field"] = home_field(out["season"], out["home_id"], out["away_id"], out["venue_id"],
                                   out["neutral"], venues)
    out["played"] = completed & out["home_score"].notna() & out["away_score"].notna()
    out.loc[~out["played"], ["home_score", "away_score"]] = np.nan
    out["result"] = out["home_score"] - out["away_score"]
    out["total"] = out["home_score"] + out["away_score"]

    raw = _read_seasons(root, "lines", seasons)
    if raw is None:
        lines = pd.DataFrame(columns=["game_id", "spread_line", "total_line", "home_favorite",
                                      "line_source", "has_line"])
    else:
        lines = lines_frame(raw)
        lines["game_id"] = lines["game_id"].astype("int64")
    out = out.merge(lines, on="game_id", how="left")
    out["has_line"] = out["has_line"].eq(True)
    for c in ("spread_line", "total_line", "home_favorite"):
        out[c] = out[c].astype(float)
    out = out[GAME_COLUMNS]
    return out.sort_values(["kickoff", "game_id"], kind="mergesort").reset_index(drop=True)


def rank_gaps(games):
    """Games where an FBS team has no rank but was ranked in the top 15 in
    both its previous and its next game that season: a rank missing, not an
    unranked team. -> a boolean Series per game (either side)."""
    sides = []
    for side in ("home", "away"):
        sides.append(pd.DataFrame({
            "row": games.index, "season": games["season"], "kickoff": games["kickoff"],
            "game_id": games["game_id"], "team": games[f"{side}_id"], "rank": games[f"{side}_rank"],
            "fbs": games[f"{side}_division"] == "fbs"}))
    t = pd.concat(sides, ignore_index=True)
    t = t[t["fbs"]].sort_values(["season", "team", "kickoff", "game_id"], kind="mergesort")
    by = t.groupby(["season", "team"])["rank"]
    gap = t["rank"].isna() & (by.shift(1) <= 15) & (by.shift(-1) <= 15)
    return pd.Series(games.index.isin(t.loc[gap, "row"]), index=games.index)


def summary(games):
    """Per season: games, FBS v FBS, v FCS, postseason, neutral (ESPN's
    flag), home_field_fixed (games where home_field overrides that flag),
    played, with a line (and FBS v FBS with a line), rank_gaps."""
    if games.empty:
        return pd.DataFrame()
    g = games.assign(fbs_line=games["has_line"] & (games["fbs_game"] == 1),
                     fixed=games["home_field"] != 1 - games["neutral"],
                     gap=rank_gaps(games))
    return g.groupby("season").agg(
        games=("game_id", "size"), fbs_v_fbs=("fbs_game", "sum"), v_fcs=("fcs_game", "sum"),
        postseason=("postseason", "sum"), neutral=("neutral", "sum"),
        home_field_fixed=("fixed", "sum"), played=("played", "sum"),
        lines=("has_line", "sum"), fbs_lines=("fbs_line", "sum"), rank_gaps=("gap", "sum"))


# ------------------------------------------------------------------ teams

def _team_record(t):
    logos = t.get("logos") or []
    logo = t.get("logo") or (logos[0].get("href") if logos and isinstance(logos[0], dict) else None)
    return {
        "team_id": int(t["id"]),
        "name": t.get("displayName"),
        "short_name": t.get("shortDisplayName"),
        "abbreviation": t.get("abbreviation"),
        "location": t.get("location"),
        "nickname": t.get("name"),
        "color": t.get("color") or None,
        "alt_color": t.get("alternateColor") or None,
        "logo": logo,
    }


def parse_teams(body):
    """ESPN's teams list -> one record per team."""
    data = json.loads(body)
    out = []
    for league in (data.get("sports") or [{}])[0].get("leagues") or []:
        for entry in league.get("teams") or []:
            t = entry.get("team") or {}
            if t.get("id"):
                out.append(_team_record(t))
    return out


def merge_teams(*sources):
    """Team records, later sources over earlier ones field by field; a
    missing field never erases one on file."""
    merged = {}
    for records in sources:
        for r in records:
            have = merged.setdefault(int(r["team_id"]), {c: None for c in TEAM_COLUMNS})
            for c in TEAM_COLUMNS:
                v = r.get(c)
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and v != "":
                    have[c] = v
    frame = pd.DataFrame(list(merged.values()), columns=TEAM_COLUMNS)
    frame["team_id"] = frame["team_id"].astype(int)
    return frame.sort_values("team_id").reset_index(drop=True)


def _teams_on_file(path):
    if not Path(path).exists():
        return []
    t = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    return [{c: (r.get(c) or None) for c in TEAM_COLUMNS} for r in t.to_dict("records")]


def load_teams(root=DATA):
    """Display info for every team on file or in the schedule: team_id,
    name, short_name, abbreviation, location, nickname, color and alt_color
    (hex without '#', as ESPN gives it; NaN when never seen), logo, and the
    division, conference and season of the latest season it appears in
    (fbs True when that division is FBS)."""
    root = Path(root)
    teams = merge_teams(_teams_on_file(root / "teams.csv.gz"))
    seasons = seasons_on_file(root)
    g = _read_seasons(root, "schedules", seasons)
    if g is not None:
        sides = []
        for side in ("home", "away"):
            sides.append(pd.DataFrame({
                "team_id": _num(g[f"{side}_id"]), "season": _num(g["season"]),
                "sched_name": g[f"{side}_team"], "sched_abbr": g[f"{side}_abbreviation"],
                "division": _txt(g[f"{side}_division"]), "conference": _txt(g[f"{side}_conference"]),
                "game": _num(g["game_id"])}))
        s = pd.concat(sides).dropna(subset=["team_id"])
        s["team_id"] = s["team_id"].astype(int)
        latest = s.sort_values(["season", "game"], kind="mergesort").groupby("team_id").tail(1)
        teams = teams.merge(latest[["team_id", "season", "sched_name", "sched_abbr", "division",
                                    "conference"]], on="team_id", how="outer")
        teams["name"] = teams["name"].fillna(teams["sched_name"])
        teams["abbreviation"] = teams["abbreviation"].fillna(teams["sched_abbr"])
        teams = teams.drop(columns=["sched_name", "sched_abbr"])
    else:
        teams["season"], teams["division"], teams["conference"] = np.nan, np.nan, np.nan
    teams["fbs"] = teams["division"] == "fbs"
    teams = teams.replace("", np.nan)
    return teams.sort_values("team_id").reset_index(drop=True)


# ------------------------------------------------------------------ the live board

def _american(text):
    """'+2000' -> 2000, '-110' -> -110, 'EVEN' -> 100; 'OFF' or nothing -> None."""
    if text is None:
        return None
    s = str(text).strip().upper()
    if s in ("EVEN", "EV"):
        return 100.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _points(text):
    """'+30.5' -> 30.5, 'o56.5' -> 56.5, 'PK'/'EVEN' -> 0.0; else None."""
    if text is None:
        return None
    s = str(text).strip().lower().lstrip("ou")
    if s in ("pk", "even", "ev"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _num_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _path_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_line(odds):
    """A competition's odds list -> the main provider's line (lowest
    priority number, then first), or None. spread_line is from the home
    side, positive = home favoured."""
    entries = [o for o in (odds or []) if isinstance(o, dict)]
    if not entries:
        return None
    def priority(e):
        p = _num_or_none(_path_get(e, "provider", "priority"))
        return 1e9 if p is None else p              # 0 is the first, not missing
    o = sorted(entries, key=priority)[0]
    home_handicap = _num_or_none(o.get("spread"))
    if home_handicap is None:
        home_handicap = _points(_path_get(o, "pointSpread", "home", "close", "line"))
    total = _num_or_none(o.get("overUnder"))
    if total is None:
        total = _points(_path_get(o, "total", "over", "close", "line"))
    open_handicap = _points(_path_get(o, "pointSpread", "home", "open", "line"))
    line = {
        "provider": _path_get(o, "provider", "name"),
        "provider_id": _path_get(o, "provider", "id"),
        "line_details": o.get("details"),
        "spread_line": None if home_handicap is None else -home_handicap + 0.0,
        "home_spread_odds": _american(_path_get(o, "pointSpread", "home", "close", "odds")),
        "away_spread_odds": _american(_path_get(o, "pointSpread", "away", "close", "odds")),
        "total_line": total,
        "over_odds": _american(_path_get(o, "total", "over", "close", "odds")),
        "under_odds": _american(_path_get(o, "total", "under", "close", "odds")),
        "home_ml": _american(_path_get(o, "moneyline", "home", "close", "odds")),
        "away_ml": _american(_path_get(o, "moneyline", "away", "close", "odds")),
        "spread_open": None if open_handicap is None else -open_handicap + 0.0,
        "total_open": _points(_path_get(o, "total", "over", "open", "line")),
        "home_ml_open": _american(_path_get(o, "moneyline", "home", "open", "odds")),
        "away_ml_open": _american(_path_get(o, "moneyline", "away", "open", "odds")),
    }
    if line["spread_line"] is None and line["total_line"] is None and line["home_ml"] is None:
        return None
    return line


def _utc_text(t):
    """A moment -> '2026-10-31T16:00:00Z' (naive input is read as UTC)."""
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_board(body):
    """One ESPN scoreboard -> (games, teams). Each game: ids, kickoff (UTC
    text), status, scores, ranks, flags and the main provider's line (or
    None); each team: display info as the board shows it."""
    data = json.loads(body)
    games, teams = [], []
    for e in data.get("events") or []:
        comp = (e.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        home, away = sides.get("home"), sides.get("away")
        if not home or not away:
            continue
        st = _path_get(comp, "status", "type") or _path_get(e, "status", "type") or {}
        state = st.get("state")
        g = {
            "game_id": int(e["id"]),
            "season": _path_get(e, "season", "year"),
            "season_type": SEASON_TYPES.get(_path_get(e, "season", "type")),
            "week": _path_get(e, "week", "number"),
            "kickoff": _utc_text(pd.Timestamp(comp.get("date") or e.get("date"))),
            "time_tbd": not comp.get("timeValid", True),
            "status": st.get("name"),
            "state": state,
            "completed": bool(st.get("completed")),
            "neutral": bool(comp.get("neutralSite")),
            "conference_game": bool(comp.get("conferenceCompetition")),
            "venue": _path_get(comp, "venue", "fullName"),
            "venue_id": _path_get(comp, "venue", "id"),
            "notes": "; ".join(n.get("headline", "") for n in comp.get("notes") or [] if n.get("headline")) or None,
        }
        for side, c in (("home", home), ("away", away)):
            t = c.get("team") or {}
            rank = _num_or_none(_path_get(c, "curatedRank", "current"))
            g[f"{side}_id"] = int(c.get("id") or t.get("id"))
            g[side] = t.get("displayName") or t.get("location")
            g[f"{side}_abbr"] = t.get("abbreviation")
            g[f"{side}_score"] = _num_or_none(c.get("score")) if state in ("in", "post") else None
            g[f"{side}_rank"] = rank if rank is not None and 1 <= rank <= 25 else None
            if t.get("id"):
                teams.append(_team_record(t))
        g["line"] = parse_line(comp.get("odds"))
        games.append(g)
    return games, teams


def merge_week(fresh, old, fetched_at):
    """This fetch's games with each one's line as known before kickoff.

    A line is taken from this fetch only while the game has not started
    (state 'pre', and the fetch is before a set kickoff); its line_seen_at is
    the fetch time. Otherwise - the game under way or final, when the board
    drops or may change its odds - the line on file is kept with the moment
    it was seen, if that was before kickoff; else the game has no line."""
    seen = _utc_text(fetched_at)
    before = {g["game_id"]: g for g in old}
    out = []
    for g in fresh:
        g = dict(g)
        line = g.pop("line", None)
        pre = g["state"] == "pre" and (g["time_tbd"] or seen < g["kickoff"])
        if line is not None and pre:
            line = {**line, "line_seen_at": seen}
        else:
            prior = before.get(g["game_id"]) or {}
            kept = prior.get("line_seen_at")
            ok = kept and (g["time_tbd"] or kept < g["kickoff"])
            line = {k: prior.get(k) for k in LINE_FIELDS} if ok else None
        g.update(line or {k: None for k in LINE_FIELDS})
        out.append(g)
    return sorted(out, key=lambda g: (g["kickoff"], g["game_id"]))


def sync_week(fetch=_fetch, log=print, root=DATA, now=None, days=9, problems=None):
    """The live board: ESPN's current FBS week, plus each of the next `days`
    days (so a Friday run already holds next Saturday). Written only when
    the current-week board answered. -> (games written, teams seen)"""
    problems = [] if problems is None else problems
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    path = Path(root) / "week.json"
    try:
        body = fetch(BOARD_URL)
        data = json.loads(body)
        if "events" not in data:
            raise ValueError("no events")
        games, teams = parse_board(body)
    except Exception as err:                        # noqa: BLE001
        problems.append("live board")
        log(f"   live board: not fetched ({err.__class__.__name__}); kept what is on file")
        return None, []
    have = {g["game_id"] for g in games}
    for k in range(days):
        day = (now + timedelta(days=k)).strftime("%Y%m%d")
        try:
            more, more_teams = parse_board(fetch(f"{BOARD_URL}&dates={day}"))
        except Exception as err:                    # noqa: BLE001 - an extra day, not the week
            log(f"   board {day}: not fetched ({err.__class__.__name__})")
            continue
        games += [g for g in more if g["game_id"] not in have]
        have |= {g["game_id"] for g in more}
        teams += more_teams
    old = []
    if path.exists():
        try:
            old = json.loads(path.read_text()).get("games") or []
        except ValueError:
            old = []
    week = merge_week(games, old, now)
    out = {"source": BOARD_URL, "fetched_at": _utc_text(now),
           "season": _path_get((data.get("leagues") or [{}])[0], "season", "year"),
           "week": _path_get(data, "week", "number"), "games": week}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(week_json(out))
    with_line = sum(1 for g in week if g.get("spread_line") is not None)
    log(f"   live board: {len(week)} games (week {out['week']}), {with_line} with a line")
    return len(week), teams


def week_json(out):
    """The board as JSON text, one game per line: small, and a diff shows
    which games changed."""
    head = json.dumps({k: v for k, v in out.items() if k != "games"}, sort_keys=True)
    games = ",\n".join(json.dumps(g, sort_keys=True) for g in out["games"])
    return head[:-1] + (', "games": [\n' + games + "\n]}\n" if games else ', "games": []}\n')


def load_week(root=DATA):
    """The live board on file, one row per game, in kickoff order: the
    schedule's identifiers and flags (fbs_game / fcs_game from the latest
    season's divisions on file; NaN when the team is not on file; neutral
    as ESPN flags it and home_field as load_games reads it, from each team's
    stadium in that season's schedule on file), status, scores, and the line
    with the moment it was seen (line_seen_at, naive UTC) and the board's
    fetch time (fetched_at). Empty when none."""
    root = Path(root)
    path = root / "week.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    w = pd.DataFrame(data.get("games") or [])
    if w.empty:
        return w
    w["kickoff"] = kickoffs(w["kickoff"])
    w["line_seen_at"] = kickoffs(w["line_seen_at"])
    w["fetched_at"] = pd.Timestamp(data["fetched_at"]).tz_convert(None)
    teams = load_teams(root)
    division = dict(zip(teams["team_id"], teams["division"])) if len(teams) else {}
    for side in ("home", "away"):
        w[f"{side}_division"] = w[f"{side}_id"].map(division)
    known = w["home_division"].notna() & w["away_division"].notna()
    hf, af = w["home_division"] == "fbs", w["away_division"] == "fbs"
    w["fbs_game"] = (hf & af).astype(float).where(known)
    w["fcs_game"] = (hf ^ af).astype(float).where(known)
    for c in ("neutral", "conference_game"):
        w[c] = w[c].astype(bool).astype(int)
    if "venue_id" not in w:
        w["venue_id"] = None
    w["venue_id"] = pd.array(_ids(w["venue_id"]), dtype="Int64")
    sched = _read_seasons(root, "schedules", sorted({int(s) for s in _ids(w["season"]) if s}))
    venues = {} if sched is None else home_venues(
        sched["season"], sched["home_id"], sched["venue_id"], _true(sched["neutral_site"]))
    w["home_field"] = home_field(w["season"], w["home_id"], w["away_id"], w["venue_id"],
                                 w["neutral"], venues)
    w["postseason"] = (w["season_type"] == "postseason").astype(int)
    w["played"] = w["completed"].astype(bool) & w["home_score"].notna() & w["away_score"].notna()
    return w.sort_values(["kickoff", "game_id"], kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------------ sync

def sync(full=False, seasons=None, today=None, now=None, fetch=_fetch, log=print, root=DATA,
         board=True):
    """Refresh the college football snapshot.

    Schedules and lines for `seasons` (default: the current one, and in
    March the one before too; every season since 2004 with `full`), resolved from GitHub's releases API
    (sizes checked) or the fixed download URL; then ESPN's teams list and
    the live board. Each file stands alone: one that fails, comes back cut
    short or unreadable keeps what is on file and the rest still refresh.
    """
    root = Path(root)
    problems = []
    today = today or datetime.now(timezone.utc).date()
    now_season = current_season(today)
    # Through March last season is refreshed too: its bowls' and playoff's
    # lines land after the games, and a fix to them should not need --full.
    recent = [now_season - 1, now_season] if today.month == 3 else [now_season]
    seasons = list(range(FIRST_SEASON, now_season + 1)) if full else \
        [int(s) for s in (seasons or recent)]
    assets = {tag: release_assets(tag, fetch, log) for tag in (SCHEDULE_TAG, BETTING_TAG)}
    result = {"seasons": seasons, "files": {}}
    for season in seasons:
        result["files"][season] = sync_season(season, assets, fetch, log, root, problems)
    for season in seasons:
        if _path(root, "schedules", season).exists():
            g = load_games([season], root)
            if len(g):
                s = summary(g).iloc[0]
                sources = g["line_source"].fillna("none").value_counts().to_dict()
                raw = _read_seasons(root, "lines", [season])
                alone = 0 if raw is None else int(lines_frame(raw)["total_alone"].sum())
                log(f"   {season}: {s.games} games with an FBS team ({s.fbs_v_fbs} FBS v FBS, "
                    f"{s.v_fcs} v FCS, {s.postseason} postseason), {s.played} played, "
                    f"{s.lines} with a line ({s.fbs_lines} FBS v FBS); line rows by source {sources}; "
                    f"{alone} totals without a spread not loaded; home field differs from ESPN's "
                    f"neutral flag in {s.home_field_fixed}; {s.rank_gaps} rank gaps")

    seen_teams = []
    try:
        body = fetch(TEAMS_URL)
        seen_teams = parse_teams(body) if body else []
        log(f"   teams: {len(seen_teams)} from ESPN's list")
    except Exception as err:                        # noqa: BLE001
        problems.append("teams")
        log(f"   teams: not fetched ({err.__class__.__name__}); kept what is on file")
    if board:
        result["week"], board_teams = sync_week(fetch, log, root, now, problems=problems)
        seen_teams += board_teams
    teams_path = root / "teams.csv.gz"
    if seen_teams:
        teams = merge_teams(_teams_on_file(teams_path), seen_teams)
        result["teams"] = store(teams, teams_path, "teams", log, problems)
    if problems:
        log(f"   KEPT ON FILE after a failed or unreadable download: {', '.join(problems)}")
    result["problems"] = problems
    return result


if __name__ == "__main__":
    full = "--full" in sys.argv
    seasons = [int(a) for a in sys.argv[1:] if a.isdigit()] or None
    print("Syncing college football (sportsdataverse + ESPN)" + (" - every season" if full else ""))
    sync(full=full, seasons=seasons)
