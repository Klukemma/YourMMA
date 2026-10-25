"""Soccer data: every match in Europe's top five leagues and their second
tiers since 2005-06 - result, match statistics and the betting market -
from football-data.co.uk, plus the fixtures still to be played.

Snapshotted into soccer/data/ so history never depends on the network: a
sync refreshes the current season and keeps what is already on file.
Everything else in the soccer engine reads the snapshot through the
loaders below.

WHY THE SECOND TIERS. Three teams come up into each top league every
summer. A rating that met them for the first time in August would have to
guess; one that watched them win the Championship has seen them play. The
second tiers are rated, never predicted.

THE MARKET, TWICE. football-data.co.uk records two prices for every match:
  pre-match  collected on Friday afternoon for weekend games and Tuesday
             afternoon for midweek ones - the same moment its fixtures file,
             which this engine predicts from, is published with odds
  closing    the last price before kickoff (from 2012-13 for Pinnacle, from
             2019-20 for the market average)
The model is measured and blended against the PRE-MATCH price, because that
is the price it will actually have when it picks. Closing is kept to show
how much the market still moved after the pick was made.

POINT IN TIME. A kickoff is a moment: football-data.co.uk gives the date
and the UK clock time (Europe/London) for every league. It is converted to
UTC once, here. Seasons before 2019-20 carry no kickoff time; their matches
sort by date, which is all a walk-forward over finished seasons needs.
"""

import csv
import gzip
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SOCCER = Path(__file__).resolve().parents[1]
DATA = SOCCER / "data"
MATCHES = DATA / "matches.csv.gz"
FIXTURES = DATA / "fixtures.csv"

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
UK = ZoneInfo("Europe/London")
FIRST_SEASON = 2005          # 2005-06: the first season with market averages

# code: (name, country, tier, The Odds API sport key)
LEAGUES = {
    "E0": ("Premier League", "England", 1, "soccer_epl"),
    "SP1": ("La Liga", "Spain", 1, "soccer_spain_la_liga"),
    "I1": ("Serie A", "Italy", 1, "soccer_italy_serie_a"),
    "D1": ("Bundesliga", "Germany", 1, "soccer_germany_bundesliga"),
    "F1": ("Ligue 1", "France", 1, "soccer_france_ligue_one"),
    "E1": ("Championship", "England", 2, None),
    "SP2": ("Segunda División", "Spain", 2, None),
    "I2": ("Serie B", "Italy", 2, None),
    "D2": ("2. Bundesliga", "Germany", 2, None),
    "F2": ("Ligue 2", "France", 2, None),
}
TOP = [code for code, meta in LEAGUES.items() if meta[2] == 1]
COUNTRY_TOP = {meta[1]: code for code, meta in LEAGUES.items() if meta[2] == 1}

# The snapshot's columns -> the football-data.co.uk columns that fill them,
# newest name first. The bookmaker averages were "BbAv..." until 2019-20 and
# "Avg..." since; Pinnacle's pre-match price was "PS..." and briefly "P...".
SOURCES = {
    "hg": ["FTHG", "HG"], "ag": ["FTAG", "AG"], "result": ["FTR", "Res"],
    "hthg": ["HTHG"], "htag": ["HTAG"],
    "hs": ["HS"], "as": ["AS"], "hst": ["HST"], "ast": ["AST"],
    "hc": ["HC"], "ac": ["AC"], "hy": ["HY"], "ay": ["AY"], "hr": ["HR"], "ar": ["AR"],
    # pre-match (Friday / Tuesday) prices
    "avg_h": ["AvgH", "BbAvH"], "avg_d": ["AvgD", "BbAvD"], "avg_a": ["AvgA", "BbAvA"],
    "max_h": ["MaxH", "BbMxH"], "max_d": ["MaxD", "BbMxD"], "max_a": ["MaxA", "BbMxA"],
    "ps_h": ["PSH", "PH"], "ps_d": ["PSD", "PD"], "ps_a": ["PSA", "PA"],
    "b365_h": ["B365H"], "b365_d": ["B365D"], "b365_a": ["B365A"],
    "avg_o25": ["Avg>2.5", "BbAv>2.5"], "avg_u25": ["Avg<2.5", "BbAv<2.5"],
    "max_o25": ["Max>2.5", "BbMx>2.5"], "max_u25": ["Max<2.5", "BbMx<2.5"],
    # closing prices
    "avgc_h": ["AvgCH"], "avgc_d": ["AvgCD"], "avgc_a": ["AvgCA"],
    "psc_h": ["PSCH"], "psc_d": ["PSCD"], "psc_a": ["PSCA"],
    "avgc_o25": ["AvgC>2.5"], "avgc_u25": ["AvgC<2.5"],
}
COLUMNS = (["match_id", "league", "season", "date", "time", "kickoff", "home", "away"]
           + list(SOURCES))
INTEGER = ["season", "hg", "ag", "hthg", "htag", "hs", "as", "hst", "ast",
           "hc", "ac", "hy", "ay", "hr", "ar"]
FIXTURE_COLUMNS = COLUMNS + ["fetched_at"]


def season_code(season):
    """2026 -> '2627', the season that starts in August 2026."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_of(day):
    """The season a date falls in: July onwards is the new one."""
    day = pd.Timestamp(day)
    return day.year if day.month >= 7 else day.year - 1


def current_season(today=None):
    return season_of(today or datetime.now(timezone.utc).date())


def parse_date(text):
    """football-data.co.uk writes dd/mm/yy in old files and dd/mm/yyyy since."""
    text = str(text).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def kickoff_utc(day, clock):
    """The UK-clock kickoff -> naive UTC timestamp, or NaT with no time."""
    clock = str(clock or "").strip()
    if day is None or ":" not in clock:
        return pd.NaT
    hour, minute = clock.split(":")[:2]
    local = datetime(day.year, day.month, day.day, int(hour), int(minute), tzinfo=UK)
    return pd.Timestamp(local.astimezone(timezone.utc).replace(tzinfo=None))


def match_id(league, day, home, away):
    slug = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    return f"{league}-{day:%Y%m%d}-{slug(home)}-{slug(away)}"


def _number(text):
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse(body, season=None):
    """One football-data.co.uk CSV (a season file or fixtures.csv) -> rows in
    the snapshot's columns.

    Read with the csv module, not pandas: old files carry rows longer than
    their header and trailing blank columns, and a parser that skips "bad
    lines" would silently drop real matches."""
    if isinstance(body, bytes):
        try:
            text = body.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = body.decode("latin-1")
    else:
        text = body.lstrip("﻿")
    reader = csv.reader(io.StringIO(text))
    header = [h.strip().lstrip("﻿") for h in next(reader, [])]
    out = []
    for raw in reader:
        row = dict(zip(header, (c.strip() for c in raw)))
        league, home, away = row.get("Div", ""), row.get("HomeTeam", ""), row.get("AwayTeam", "")
        day = parse_date(row.get("Date", ""))
        if league not in LEAGUES or not home or not away or day is None:
            continue
        rec = {
            "match_id": match_id(league, day, home, away),
            "league": league,
            "season": season if season is not None else season_of(day),
            "date": day.isoformat(),
            "time": row.get("Time", "") or None,
            "kickoff": kickoff_utc(day, row.get("Time")),
            "home": home, "away": away,
        }
        for column, names in SOURCES.items():
            value = None
            for name in names:
                if row.get(name, "") != "":
                    value = row[name]
                    break
            if column == "result":
                rec[column] = value if value in ("H", "D", "A") else None
            else:
                rec[column] = _number(value) if value is not None else None
        out.append(rec)
    return out


def frame(rows, columns=COLUMNS):
    df = pd.DataFrame(rows, columns=columns)
    for c in INTEGER:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    return df


# ------------------------------------------------------------- network

def _fetch(url):
    """The body of one file. requests if installed, else urllib."""
    headers = {"User-Agent": "YourSoccer data sync (github.com/Klukemma/YourMMA)"}
    try:
        import requests
        response = requests.get(url, timeout=120, headers=headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content
    except ImportError:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as handle:
                return handle.read()
        except HTTPError as err:
            if err.code == 404:
                return None
            raise


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so an unchanged snapshot is byte-identical and git sees no change.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(df.to_csv(index=False).encode())
    return path


def order(df):
    return df.sort_values(["date", "kickoff", "league", "home"], na_position="first",
                          kind="mergesort").reset_index(drop=True)


def sync(seasons=None, full=False, today=None, fetch=_fetch, log=print,
         matches_path=MATCHES, fixtures_path=FIXTURES):
    """Refresh the snapshot: the given seasons (default this one, plus last
    season in its first weeks so late corrections land), or every season
    since FIRST_SEASON with full=True. Seasons not fetched are kept as they
    are on file; a league-season the site does not have is skipped, never
    blanked."""
    now = current_season(today)
    if full:
        seasons = list(range(FIRST_SEASON, now + 1))
    elif seasons is None:
        month = pd.Timestamp(today or datetime.now(timezone.utc)).month
        seasons = [now - 1, now] if month in (7, 8) else [now]
    fetched = {}
    for season in seasons:
        for league in LEAGUES:
            body = fetch(SEASON_URL.format(code=season_code(season), league=league))
            if not body:
                log(f"   {league} {season}: not published")
                continue
            rows = parse(body, season)
            if rows:
                fetched[(league, season)] = rows
                log(f"   {league} {season}: {len(rows)} matches")
    old = load_matches(matches_path) if Path(matches_path).exists() else frame([])
    if fetched:
        keep = ~old.set_index(["league", "season"]).index.isin(list(fetched))
        new = frame([r for rows in fetched.values() for r in rows])
        merged = pd.concat([old[keep], new], ignore_index=True)
    else:
        merged = old
    merged = order(merged.drop_duplicates("match_id", keep="last"))
    _write(merged, Path(matches_path))
    log(f"   matches: {len(merged)} on file, {merged['result'].notna().sum()} with a result")

    body = fetch(FIXTURES_URL)
    if body:
        stamp = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("s")
        fx = frame(parse(body))
        fx["fetched_at"] = stamp
        fx = order(fx[fx["league"].isin(LEAGUES)])
        Path(fixtures_path).parent.mkdir(parents=True, exist_ok=True)
        fx.to_csv(fixtures_path, index=False)
        log(f"   fixtures: {len(fx)} still to play, prices as of {stamp:%Y-%m-%d %H:%M} UTC")
    return merged


# ------------------------------------------------------------- loaders

def load_matches(path=MATCHES):
    df = pd.read_csv(path, low_memory=False)
    for c in INTEGER:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["date"] = df["date"].astype(str)
    return df


def load_fixtures(path=FIXTURES):
    if not Path(path).exists():
        return frame([], FIXTURE_COLUMNS)
    df = pd.read_csv(path, low_memory=False)
    for c in INTEGER:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["date"] = df["date"].astype(str)
    return df


if __name__ == "__main__":
    full = "--full" in sys.argv
    print(f"Soccer sync ({'every season' if full else 'this season'}) from football-data.co.uk")
    sync(full=full)
