"""Soccer data: every league football-data.co.uk carries with results AND
prices over past seasons - its 22 European divisions and its 16 "extra"
leagues - and the fixtures still to be played.

Snapshotted into soccer/data/ so history never depends on the network: a
sync refreshes the current season and keeps what is already on file.
Everything else in the soccer engine reads the snapshot through the
loaders below.

THE TEN AND ALL. LEAGUES, TOP and COUNTRY_TOP are the ten divisions the
model and the app run on today (the top five leagues and their second
tiers), exactly as before. ALL_LEAGUES is every league this module knows
(38). sync() fetches all 38, but load_matches() and load_fixtures() return
the ten unless asked for leagues="all" (or a list of codes): the running
pipeline sees what it saw before until a later step switches it over on
purpose, league by league, once each is measured.

WHY THE SECOND TIERS. Three teams come up into each top league every
summer. A rating that met them for the first time in August would have to
guess; one that watched them win the Championship has seen them play. The
second tiers are rated, never predicted.

THE MARKET, TWICE. football-data.co.uk records two prices for every match
in its 22 European divisions:
  pre-match  collected on Friday afternoon for weekend games and Tuesday
             afternoon for midweek ones - the same moment its fixtures file,
             which this engine predicts from, is published with odds
  closing    the last price before kickoff (from 2012-13 for Pinnacle, from
             2019-20 for the market average)
The model is measured and blended against the PRE-MATCH price, because that
is the price it will actually have when it picks. Closing is kept to show
how much the market still moved after the pick was made.
football-data.co.uk (data.php, 2026-10-29): since 23/07/2025 Pinnacle's feed
has been unreliable and is left out of its averages and maxima - ps_* and
psc_* are patchy in 2025-26 and empty in 2026-27, in every league.

THE REGISTRY. ALL_LEAGUES[code] = League(name, country, tier, odds_key,
kind, fd_code, fd_leagues, calendar, espn); the first four fields are
LEAGUES' own, so code written against LEAGUES reads ALL_LEAGUES unchanged.
  kind        "main": one CSV per season, mmz4281/{yyzz}/{fd_code}.csv
              "extra": one CSV holding every season, new/{fd_code}.csv
  fd_leagues  an extra file's League values that belong to the league
  calendar    "aug-may" (July onwards is the new season) or "calendar-year";
              the CURRENT calendar - it dates a fixture, which carries no
              season of its own (history keeps football-data's label)
  espn        ESPN's league code, where ESPN knows the league (probe of
              2026-10-29: its events Oct 26-Nov 8 in brackets; "2025
              clubs" - ESPN still lists last season's clubs and no games)

  code  league              country      tier calendar  espn
  E0    Premier League      England       1   aug-may   eng.1
  E1    Championship        England       2   aug-may   eng.2 (36)
  E2    League One          England       3   aug-may   eng.3 (20)
  E3    League Two          England       4   aug-may   eng.4 (15)
  EC    National League     England       5   aug-may   eng.5 (25)
  SC0   Premiership         Scotland      1   aug-may   sco.1 (16)
  SC1   Championship        Scotland      2   aug-may   sco.2 (11)
  SC2   League One          Scotland      3   aug-may   sco.3 (0: 2025 clubs)
  SC3   League Two          Scotland      4   aug-may   sco.4 (0: 2025 clubs)
  SP1   La Liga             Spain         1   aug-may   esp.1
  SP2   Segunda Division    Spain         2   aug-may   esp.2 (22)
  I1    Serie A             Italy         1   aug-may   ita.1
  I2    Serie B             Italy         2   aug-may   ita.2 (30)
  D1    Bundesliga          Germany       1   aug-may   ger.1
  D2    2. Bundesliga       Germany       2   aug-may   ger.2 (18)
  F1    Ligue 1             France        1   aug-may   fra.1
  F2    Ligue 2             France        2   aug-may   fra.2 (20)
  N1    Eredivisie          Netherlands   1   aug-may   ned.1 (18)
  B1    Pro League          Belgium       1   aug-may   bel.1 (19)
  P1    Primeira Liga       Portugal      1   aug-may   por.1 (19)
  T1    Super Lig           Turkey        1   aug-may   tur.1 (21)
  G1    Super League        Greece        1   aug-may   gre.1 (14)
  ARG   Liga Profesional    Argentina     1   calendar  arg.1 (33)
  AUT   Bundesliga          Austria       1   aug-may   aut.1 (12)
  BRA   Serie A             Brazil        1   calendar  bra.1 (22)
  CHN   Super League        China         1   calendar  chn.1 (8)
  DNK   Superliga           Denmark       1   aug-may   den.1 (13)
  FIN   Veikkausliiga       Finland       1   calendar  fin.1 (0: no clubs)
  IRL   Premier Division    Ireland       1   calendar  irl.1 (0: no clubs)
  JPN   J1 League           Japan         1   aug-may   jpn.1 (20)
  MEX   Liga MX             Mexico        1   aug-may   mex.1 (19)
  NOR   Eliteserien         Norway        1   calendar  nor.1 (16)
  POL   Ekstraklasa         Poland        1   aug-may   - (pol.1 is a 404)
  ROU   SuperLiga           Romania       1   aug-may   rou.1 (0: 2025 clubs)
  RUS   Premier League      Russia        1   aug-may   rus.1 (16)
  SWE   Allsvenskan         Sweden        1   calendar  swe.1 (24)
  SWZ   Super League        Switzerland   1   aug-may   sui.1 (0: 2025 clubs)
  USA   MLS                 USA           1   calendar  usa.1 (44)
(odds_key, The Odds API's sport key, is set for the five top leagues only.)

TWO SHAPES OF FILE.
  main   results, half-time score, shots, shots on target, corners, cards
         (all 22 divisions since 2017-18; EC has no statistics in 2025-26),
         pre-match AND closing prices, over/under 2.5. Read by parse().
  extra  Country, League, Season, Date, Time, Home, Away, HG, AG, Res, then
         CLOSING prices only: PSC*, MaxC*, AvgC*, BFEC*, B365C* (RUS has no
         BFEC/B365C). UTF-8 with a BOM, padded names ("Liga Profesional ",
         " J1 League", "Denmark "), every season since 2012 (CHN since 2014).
         No statistics, no over/under, no pre-match price. Read by
         parse_extra(), into the same columns: AvgC* -> avgc_*, PSC* ->
         psc_*; every pre-match column stays empty. MaxC*, BFEC* and B365C*
         are not kept, as for the main divisions (no column takes them).
  Their fixtures, new_league_fixtures.csv, carry PRE-MATCH prices (PS*,
  Max*, Avg*, BFE*, B365*, collected at the main fixtures' Friday/Tuesday
  moments), read into avg_*/max_*/ps_*/b365_*. So an extra league's
  history prices only the close while a live pick would have the
  pre-match price: the two are different prices, never one column.

SEASONS. The engine names a season by the year it starts. An extra file's
Season reads "2012/2013" (-> 2012) or "2012" (-> 2012); that label is
football-data's and history keeps it (season_label). A fixture has no
label and gets its league's calendar (season_for). What the labels hold:
  JPN  calendar years to 2025, then "2026/2027" from August 2026 (the J1
       League moved to an autumn-spring season; the half-season tournament
       of Feb-Jun 2026 is not in the file). Its calendar is aug-may now.
  ARG  "2012/2013", "2013/2014", then "2014" (Aug-Dec), "2015", "2016"
       (Feb-May) AND "2016/2017" - both start in 2016, so season 2016 holds
       both (Feb 2016 - Jun 2017, 692 matches) - "2017/2018" to
       "2019/2020", and calendar years from "2020" (the Oct 2020 - Jan 2021
       cup). Since 2021 every ARG season is two competitions.
  Out of season order: one match. ARG 2013/2014's Boca Juniors v Velez
  Sarsfield was played on 29/01/2015, after season 2014's matches. A
  walk-forward must cut its training on dates, not on season numbers.
THE TWO OTHER LEAGUE VALUES.
  ARG "Copa De La Liga Profesional" (2020-2024) is part of ARG's season,
       not a league of its own: the same clubs as that year's Liga
       Profesional (2021 26 of 26, 2022 28 of 29, 2023 and 2024 28 of 28;
       2020's cup had the 24 clubs of 2019/2020), priced the same way, and
       how the top flight ran those years - a cup-format half and a league
       half of one season (from 2025 football-data calls both halves "Liga
       Profesional"). Its knockout rounds are in it (2022: 14-17 matches a
       club); a semi-final or final may be at a neutral ground, which the
       file does not mark.
  SWZ "Challenge League" is not a second tier: two rows, Thun v Sion
       27/05/2021 and Sion v Thun 30/05/2021, the 2020-21 promotion /
       relegation play-off. football-data carries no Challenge League
       season. Left out, as the main divisions' files leave out play-offs;
       parse_extra counts what it skips and sync logs it.
  Play-offs that ARE in a league's file: MLS Cup, Liga MX's liguilla,
  ARG's knockout rounds, and the split-season phases of B1, G1 and SC0.

POINT IN TIME, THE CLOCK. A kickoff is a moment: football-data.co.uk gives
the date and the UK clock time (Europe/London) for EVERY league, the extra
ones included - not the league's local clock, not UTC. It is converted to
UTC once, here (kickoff_utc). Main seasons before 2019-20 carry no time;
those matches sort by date. What the league probe of 2026-10-29 shows
(tests/test_soccer_leagues.py; every match it holds of a league in the
other source's span, paired by club names, score and day):
  ESPN's UTC kickoffs, Oct 26 - Nov 8 (this season's results and the
  fixtures; 69 matches): ARG 3/3, BRA 12/12, DNK 1/1, MEX 1/1, SWE 8/8,
    USA 14/14, E2 5/5, E3 2/2, EC 12/12, SC0 6/6, SC1 1/1, P1 1/1, T1 3/3
    to the minute on the UK clock; the local clock gets none of them right
    outside the UK's zone (Portugal shares it).
  fixturedownload.com's UTC kickoffs, the finished matches of 2025 and
  2026, most of them in British Summer Time: USA 890/906, N1 347/350, P1
    303/304, T1 380/385, SC0 60/60 to the minute on the UK clock. Read as
    UTC, 2 of USA's 756 summer kickoffs agree and 2 of T1's 223 (the two
    T1 matches of 14/09/2025 football-data has an hour early); read as the
    local clock, none outside the UK's zone. The rest are minutes - late
    starts - and the repeated hour below.
  each extra league's whole history: on the UK clock 98.5-100% of
    kickoffs fall at 11:00-22:59 local time (read as the local clock: MEX
    9%, USA 19%, JPN 35%, ARG and BRA 57%, CHN 78%, RUS 93%), and the local
    kickoff times are more alike between summer and winter on the UK clock
    than read as UTC, in all 16 (overlap UK/UTC: POL .94/.10, IRL .94/.07,
    MEX .93/.23, AUT .92/.12 ... JPN .54/.43, BRA .64/.58, ARG .58/.54).
  No UTC source overlapped AUT, CHN, FIN, IRL, JPN, NOR, POL, ROU, RUS,
  SWZ, B1, G1, SC2 or SC3 in the probe. They are read on the same clock -
  one pipeline, one format - and the history checks hold for each extra
  league (for CHN, JPN and RUS they rule out the local clock too).
  THE REPEATED HOUR. 01:00-01:59 on the last Sunday of October happens
  twice on the UK clock and football-data writes both as 01:xx: 52 extra
  league matches (ARG 4, BRA 9, MEX 27, USA 12), none in the main
  divisions' probe. They are read as the first pass, the earlier moment,
  so a kickoff is never placed after the real one: of the five MLS
  matches at "01:30" on 25/10/2026, fixturedownload has three at 00:30
  UTC and two at 01:30. The skipped hour of a March night (one match, USA
  2023-03-26 01:30) reads as GMT.

MATCH IDS. match_id(league, date, home, away), with football-data's own
date - the UK date of the kickoff, in every league, as its clock is the
UK's - and its own club names: the existing ten keep their ids, a fixture
and its result share one, and soccer_feed's uk_date ids agree (its
uk_date is the kickoff's date in Europe/London). Ids are unique within
every league in the probe. Two clubs are spelled two ways in one league:
ARG "Colon Santa FE" / "Colon Santa Fe" and NOR "Ham-Kam" / "HamKam". The
slug is the same (so are the ids); the names are kept as written - the
club tables are where they are joined.

THE SNAPSHOT. soccer/data/matches/{league}/{season}.csv.gz: one file per
league and season, gzip with mtime 0, written only when its CSV text
changes. A sync rewrites the seasons that changed and nothing else, so git
history grows by those, not by every match since 2005 on every run.
fixtures.csv stays one file, all leagues in it. Until 2026-10-29 the ten
lived in one matches.csv.gz rewritten on every sync; migrate() moved it,
and load_matches() returned the same frame from both (checked on the
day). The files of E2 E3 EC SC0-SC3 N1 B1 P1 T1 G1 for 2025 and 2026, and
of the extra leagues for every season, were seeded from the league probe's
downloads (soccer/engine/migrate_snapshot.py) until the runner's own sync
replaces them; the new divisions' seasons before 2025 arrive with the
runner's next full sync.

SYNC. Every league in ALL_LEAGUES unless told otherwise:
  main   this season (and last season in July and August, so late
         corrections land), or every season since 2005-06 with full=True.
         A 404 or an empty body is "not published": skipped, never blanked.
  extra  the league's one file, every run: the seasons from the latest one
         on file onwards (every season when none is on file, or full=True).
  fixtures.csv and new_league_fixtures.csv: each replaces its own leagues'
         rows; one that fails keeps its leagues' rows as they are on file.
A failed or unreadable download, a file with no rows, or a season that
comes back with fewer matches than are on file keeps what is on file and
is listed under KEPT ON FILE; it never costs the run the other files.

STALE OR THIN, AT THE PROBE. football-data's Russia file stops at
2026-08-02 (round 2 of 2026-27) while ESPN lists rus.1 matches through
November, and its 2025-26 has closing averages for 67% of matches. ESPN
has no current schedule for FIN, IRL, ROU, SWZ, SC2 or SC3 and no POL at
all.
"""

import csv
import gzip
import io
import re
import sys
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SOCCER = Path(__file__).resolve().parents[1]
DATA = SOCCER / "data"
MATCHES = DATA / "matches"                  # {league}/{season}.csv.gz
LEGACY_MATCHES = DATA / "matches.csv.gz"    # the single file of before (migrate)
FIXTURES = DATA / "fixtures.csv"

SITE = "https://www.football-data.co.uk"
SEASON_URL = SITE + "/mmz4281/{code}/{league}.csv"
FIXTURES_URL = SITE + "/fixtures.csv"
EXTRA_URL = SITE + "/new/{code}.csv"
EXTRA_FIXTURES_URL = SITE + "/new_league_fixtures.csv"
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

# Every league (the module's notes). The first four fields are LEAGUES' own.
League = namedtuple("League", "name country tier odds_key kind fd_code fd_leagues calendar espn")


def _main(code, name, country, tier, espn, odds_key=None):
    return code, League(name, country, tier, odds_key, "main", code, (), "aug-may", espn)


def _extra(code, name, country, fd_leagues, calendar, espn):
    return code, League(name, country, 1, None, "extra", code, tuple(fd_leagues), calendar, espn)


_TEN_ESPN = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1",
             "E1": "eng.2", "SP2": "esp.2", "I2": "ita.2", "D2": "ger.2", "F2": "fra.2"}
ALL_LEAGUES = dict([
    # the ten the pipeline runs on (LEAGUES), as they are
    *(_main(code, name, country, tier, _TEN_ESPN[code], key) for code, (name, country, tier, key) in LEAGUES.items()),
    # football-data's other European divisions
    _main("E2", "League One", "England", 3, "eng.3"),
    _main("E3", "League Two", "England", 4, "eng.4"),
    _main("EC", "National League", "England", 5, "eng.5"),
    _main("SC0", "Premiership", "Scotland", 1, "sco.1"),
    _main("SC1", "Championship", "Scotland", 2, "sco.2"),
    _main("SC2", "League One", "Scotland", 3, "sco.3"),
    _main("SC3", "League Two", "Scotland", 4, "sco.4"),
    _main("N1", "Eredivisie", "Netherlands", 1, "ned.1"),
    _main("B1", "Pro League", "Belgium", 1, "bel.1"),
    _main("P1", "Primeira Liga", "Portugal", 1, "por.1"),
    _main("T1", "Süper Lig", "Turkey", 1, "tur.1"),
    _main("G1", "Super League", "Greece", 1, "gre.1"),
    # its extra leagues: country as the file writes it, League values that belong
    _extra("ARG", "Liga Profesional", "Argentina", ["Liga Profesional", "Copa De La Liga Profesional"],
           "calendar-year", "arg.1"),
    _extra("AUT", "Bundesliga", "Austria", ["Bundesliga"], "aug-may", "aut.1"),
    _extra("BRA", "Série A", "Brazil", ["Serie A"], "calendar-year", "bra.1"),
    _extra("CHN", "Super League", "China", ["Super League"], "calendar-year", "chn.1"),
    _extra("DNK", "Superliga", "Denmark", ["Superliga"], "aug-may", "den.1"),
    _extra("FIN", "Veikkausliiga", "Finland", ["Veikkausliiga"], "calendar-year", "fin.1"),
    _extra("IRL", "Premier Division", "Ireland", ["Premier Division"], "calendar-year", "irl.1"),
    _extra("JPN", "J1 League", "Japan", ["J1 League"], "aug-may", "jpn.1"),
    _extra("MEX", "Liga MX", "Mexico", ["Liga MX"], "aug-may", "mex.1"),
    _extra("NOR", "Eliteserien", "Norway", ["Eliteserien"], "calendar-year", "nor.1"),
    _extra("POL", "Ekstraklasa", "Poland", ["Ekstraklasa"], "aug-may", None),
    _extra("ROU", "SuperLiga", "Romania", ["Superliga"], "aug-may", "rou.1"),
    _extra("RUS", "Premier League", "Russia", ["Premier League"], "aug-may", "rus.1"),
    _extra("SWE", "Allsvenskan", "Sweden", ["Allsvenskan"], "calendar-year", "swe.1"),
    _extra("SWZ", "Super League", "Switzerland", ["Super League"], "aug-may", "sui.1"),
    _extra("USA", "MLS", "USA", ["MLS"], "calendar-year", "usa.1"),
])
MAIN = [code for code, meta in ALL_LEAGUES.items() if meta.kind == "main"]
EXTRA = [code for code, meta in ALL_LEAGUES.items() if meta.kind == "extra"]
EXTRA_BY_NAME = {(meta.country, name): code for code, meta in ALL_LEAGUES.items()
                 if meta.kind == "extra" for name in meta.fd_leagues}


def league_codes(leagues=None):
    """None -> the ten the pipeline runs on (LEAGUES); "all" -> every league;
    a code or a list of codes -> those."""
    if leagues is None:
        return list(LEAGUES)
    if isinstance(leagues, str):
        return list(ALL_LEAGUES) if leagues == "all" else [leagues]
    return list(leagues)


# The snapshot's columns -> the football-data.co.uk columns that fill them,
# newest name first. The bookmaker averages were "BbAv..." until 2019-20 and
# "Avg..." since; Pinnacle's pre-match price was "PS..." and briefly "P...".
# The extra leagues' files use HG/AG/Res and only the closing columns.
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
TEXT = ["match_id", "league", "date", "time", "home", "away", "result"]
FIXTURE_COLUMNS = COLUMNS + ["fetched_at"]


def season_code(season):
    """2026 -> '2627', the season that starts in August 2026."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_of(day):
    """The season a date falls in: July onwards is the new one."""
    day = pd.Timestamp(day)
    return day.year if day.month >= 7 else day.year - 1


def season_for(league, day):
    """The season a date falls in for one league, by its calendar (a fixture
    carries no season label of its own)."""
    meta = ALL_LEAGUES.get(league)
    if meta is not None and meta.calendar == "calendar-year":
        return pd.Timestamp(day).year
    return season_of(day)


SEASON_LABEL = re.compile(r"^(\d{4})(?:/(\d{4}))?$")


def season_label(text):
    """An extra file's Season -> the year it starts: '2012/2013' -> 2012,
    '2012' -> 2012; anything else -> None."""
    m = SEASON_LABEL.match(str(text).strip())
    if not m or (m.group(2) and int(m.group(2)) != int(m.group(1)) + 1):
        return None
    return int(m.group(1))


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
    """The UK-clock kickoff -> naive UTC timestamp, or NaT with no time. The
    hour the clock repeats in October reads as its first pass (the earlier
    moment); the hour it skips in March as GMT (module notes)."""
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


def _text(body):
    if isinstance(body, bytes):
        try:
            return body.decode("utf-8-sig")
        except UnicodeDecodeError:
            return body.decode("latin-1")
    return body.lstrip("\ufeff")


def _sources(header):
    """SOURCES against one file's header: each column's names the file has,
    newest first (the same answer per row, worked out once per file)."""
    have = set(header)
    return [(column, [n for n in names if n in have]) for column, names in SOURCES.items()]


def _record(league, season, day, clock, home, away, row, sources=None):
    """One match in the snapshot's columns, from one row of a CSV."""
    rec = {
        "match_id": match_id(league, day, home, away),
        "league": league,
        "season": season,
        "date": day.isoformat(),
        "time": clock or None,
        "kickoff": kickoff_utc(day, clock),
        "home": home, "away": away,
    }
    for column, names in (SOURCES.items() if sources is None else sources):
        value = None
        for name in names:
            if row.get(name, "") != "":
                value = row[name]
                break
        if column == "result":
            rec[column] = value if value in ("H", "D", "A") else None
        else:
            rec[column] = _number(value) if value is not None else None
    # The result is read from the goals: a file whose FTR is missing, or a
    # row with a result but a goal count missing, would otherwise grade a
    # pick against nothing or crash the table.
    hg, ag = rec["hg"], rec["ag"]
    rec["result"] = None if hg is None or ag is None else "H" if hg > ag else "A" if hg < ag else "D"
    return rec


def _table(text):
    """A CSV -> its header and its rows as dicts, every cell stripped."""
    reader = csv.reader(io.StringIO(text))
    header = [h.strip().lstrip("\ufeff") for h in next(reader, [])]
    return header, (dict(zip(header, (c.strip() for c in raw))) for raw in reader)


def _rows(text):
    return _table(text)[1]


def parse(body, season=None, leagues=None):
    """One football-data.co.uk CSV of the main divisions (a season file or
    fixtures.csv) -> rows in the snapshot's columns, for the given leagues
    (league_codes; default the ten).

    Read with the csv module, not pandas: old files carry rows longer than
    their header and trailing blank columns, and a parser that skips "bad
    lines" would silently drop real matches."""
    accept = set(league_codes(leagues))
    header, rows = _table(_text(body))
    sources = _sources(header)
    out = []
    for row in rows:
        league, home, away = row.get("Div", ""), row.get("HomeTeam", ""), row.get("AwayTeam", "")
        if league not in accept or not home or not away:
            continue
        day = parse_date(row.get("Date", ""))
        if day is None:
            continue
        out.append(_record(league, season if season is not None else season_for(league, day), day,
                           row.get("Time", ""), home, away, row, sources))
    return out


def parse_extra(body, league=None, skipped=None, seasons=None, since=None):
    """One of football-data.co.uk's extra-league files - new/{CODE}.csv
    (every season, closing prices) or new_league_fixtures.csv (no season,
    pre-match prices) -> rows in the snapshot's columns, for one league or
    every one the file holds; only the given `seasons`, or the seasons from
    `since` on, when asked (a sync skips the rest before reading them). A
    row whose Country/League is not a registered league (SWZ's "Challenge
    League"), or whose date or season cannot be read, is counted in
    `skipped` (a Counter) and left out."""
    skipped = Counter() if skipped is None else skipped
    wanted = None if seasons is None else set(seasons)
    header, rows = _table(_text(body))
    sources, labelled = _sources(header), "Season" in header
    out = []
    for row in rows:
        country, name = row.get("Country", ""), " ".join(row.get("League", "").split())
        home, away = row.get("Home", ""), row.get("Away", "")
        if not (country or name or home or away):
            continue                                   # a blank line
        code = EXTRA_BY_NAME.get((country, name))
        if code is None:
            skipped[f"{country} {name}"] += 1
            continue
        if league is not None and code != league:
            continue
        if labelled:
            season = season_label(row.get("Season", ""))
            if season is not None and ((wanted is not None and season not in wanted)
                                       or (since is not None and season < since)):
                continue
            day = parse_date(row.get("Date", ""))
        else:
            day = parse_date(row.get("Date", ""))
            season = None if day is None else season_for(code, day)
        if not home or not away or day is None or season is None:
            skipped[f"{country} {name} (unreadable row)"] += 1
            continue
        out.append(_record(code, season, day, row.get("Time", ""), home, away, row, sources))
    return out


REQUIRED = ("Div", "Date", "HomeTeam", "AwayTeam")
EXTRA_REQUIRED = ("Country", "League", "Season", "Date", "Home", "Away", "HG", "AG")
EXTRA_FIXTURE_REQUIRED = ("Country", "League", "Date", "Home", "Away")


def looks_right(body, required=REQUIRED):
    """Whether a downloaded body is a football-data.co.uk CSV at all: an
    HTML error page served with a 200 must not replace a good file."""
    head = _text(body).split("\n", 1)[0]
    cells = [c.strip() for c in head.split(",")]
    return all(col in cells for col in required)


def frame(rows, columns=COLUMNS):
    df = pd.DataFrame(rows, columns=columns)
    for c in INTEGER:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    return df


def order(df):
    return df.sort_values(["date", "kickoff", "league", "home"], na_position="first",
                          kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------- the snapshot

def season_path(root, league, season):
    return Path(root) / league / f"{int(season)}.csv.gz"


def seasons_on_file(root, league):
    folder = Path(root) / league
    if not folder.is_dir():
        return []
    return sorted(int(p.name.split(".")[0]) for p in folder.glob("*.csv.gz") if p.name.split(".")[0].isdigit())


def _csv(df):
    return df.to_csv(index=False).encode()


def _write(df, path):
    """gzip with mtime 0 and no file name in it: the same frame is the same
    bytes, so git sees no change."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(_csv(df) if isinstance(df, pd.DataFrame) else df, 9, mtime=0))
    return path


def _on_file(path):
    """The CSV text of one season file, or None."""
    try:
        return gzip.decompress(Path(path).read_bytes())
    except (OSError, EOFError):
        return None


def store_season(root, league, season, rows, log=print, problems=None):
    """One league-season's matches (rows or a frame) -> its file, written
    only when its CSV text changed (compared uncompressed, so another zlib
    never rewrites an unchanged season). A season that comes back with
    fewer matches than are on file is kept as it is (a truncated download
    must not cost history) and reported; to take a shorter season on
    purpose, delete its file and sync. -> "written" | "unchanged" | "kept"."""
    new = rows if isinstance(rows, pd.DataFrame) else frame(rows)
    new = order(new.drop_duplicates("match_id", keep="last"))
    path = season_path(root, league, season)
    body, old = _csv(new), _on_file(path)
    if old == body:
        return "unchanged"
    if old is not None:
        on_file = len(pd.read_csv(io.BytesIO(old), usecols=["match_id"]))
        if len(new) < on_file:
            if problems is not None:
                problems.append(f"{league} {season}")
            log(f"   {league} {season}: {len(new)} matches, FEWER than the {on_file} on file; kept what is on file")
            return "kept"
    _write(body, path)
    return "written"


def migrate(legacy=LEGACY_MATCHES, root=MATCHES, log=print):
    """The single matches.csv.gz of before -> one file per league and season.
    Returns the frame it read."""
    old = load_matches(legacy)
    for (league, season), g in old.groupby(["league", "season"], sort=True):
        store_season(root, league, int(season), g, log=log)
    log(f"   migrated {len(old)} matches into {old.groupby(['league', 'season']).ngroups} season files under {root}")
    return old


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


def _download(fetch, url, what, required, log, problems):
    """One file, or None: a failure (a 503, a timeout, a page that is not
    the CSV) is logged and listed and keeps what is on file; a 404 or an
    empty body is "not published"."""
    try:
        body = fetch(url)
    except Exception as err:                  # noqa: BLE001 - logged, kept, run continues
        problems.append(what)
        log(f"   {what}: download failed ({err.__class__.__name__}); kept what is on file")
        return None
    if not body or not _text(body).strip():
        log(f"   {what}: not published")
        return None
    if not looks_right(body, required):
        problems.append(what)
        log(f"   {what}: NOT A football-data.co.uk CSV; kept what is on file")
        return None
    return body


def sync_matches(leagues="all", seasons=None, full=False, today=None, fetch=_fetch, log=print,
                 root=MATCHES, problems=None):
    """The results half of sync(). -> how many season files were written."""
    problems = [] if problems is None else problems
    codes = league_codes(leagues)
    unknown = [c for c in codes if c not in ALL_LEAGUES]
    if unknown:
        raise ValueError(f"unknown league codes: {unknown}")
    now = current_season(today)
    if full:
        main_seasons = list(range(FIRST_SEASON, now + 1))
    elif seasons is None:
        month = pd.Timestamp(today or datetime.now(timezone.utc)).month
        main_seasons = [now - 1, now] if month in (7, 8) else [now]
    else:
        main_seasons = list(seasons)
    written = 0
    for code in codes:
        meta = ALL_LEAGUES[code]
        if meta.kind == "main":
            for season in main_seasons:
                what = f"{code} {season}"
                body = _download(fetch, SEASON_URL.format(code=season_code(season), league=meta.fd_code),
                                 what, REQUIRED, log, problems)
                if body is None:
                    continue
                rows = parse(body, season, leagues=[code])
                if not rows:
                    problems.append(what)
                    log(f"   {what}: NO ROWS PARSED (format change?); kept what is on file")
                    continue
                done = store_season(root, code, season, rows, log, problems)
                written += done == "written"
                if done != "kept":
                    log(f"   {what}: {len(rows)} matches" + (" (unchanged)" if done == "unchanged" else ""))
            continue
        body = _download(fetch, EXTRA_URL.format(code=meta.fd_code), code, EXTRA_REQUIRED, log, problems)
        if body is None:
            continue
        # Which seasons: every one with full=True or with none on file, the
        # ones asked for, or else from the latest on file onwards.
        on_file = seasons_on_file(root, code)
        since = on_file[-1] if not full and seasons is None and on_file else None
        skipped = Counter()
        rows = parse_extra(body, league=code, skipped=skipped, since=since,
                           seasons=None if full or seasons is None else seasons)
        for what, n in sorted(skipped.items()):
            log(f"   {code}: {n} rows of {what} left out (not this league)")
        if not rows and seasons is None:
            problems.append(code)
            log(f"   {code}: NO ROWS PARSED (format change?); kept what is on file")
            continue
        by_season = defaultdict(list)
        for r in rows:
            by_season[r["season"]].append(r)
        for season in sorted(by_season):
            done = store_season(root, code, season, by_season[season], log, problems)
            written += done == "written"
            if done != "kept":
                log(f"   {code} {season}: {len(by_season[season])} matches"
                    + (" (unchanged)" if done == "unchanged" else ""))
    return written


def sync_fixtures(leagues="all", fetch=_fetch, log=print, path=FIXTURES, problems=None):
    """The fixtures half of sync(): fixtures.csv (main divisions) and
    new_league_fixtures.csv (extra leagues) into one file. Each source that
    was fetched replaces its own leagues' rows; one that was not (failed,
    unreadable, not published, not asked for) keeps its rows on file.
    -> the fixtures written, or None when nothing was fetched."""
    problems = [] if problems is None else problems
    codes = league_codes(leagues)
    old = load_fixtures(path, leagues="all")
    fresh, refreshed = [], set()
    for kind, url, what, required in (("main", FIXTURES_URL, "fixtures", REQUIRED),
                                      ("extra", EXTRA_FIXTURES_URL, "extra-league fixtures",
                                       EXTRA_FIXTURE_REQUIRED)):
        own = [c for c in codes if ALL_LEAGUES[c].kind == kind]
        if not own:
            continue
        body = _download(fetch, url, what, required, log, problems)
        if body is None:
            continue
        rows = parse(body, leagues=own) if kind == "main" else \
            [r for r in parse_extra(body) if r["league"] in own]
        fresh.append(frame(rows))
        refreshed.add(kind)
    if not refreshed:
        return None
    stamp = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("s")
    kind_of = old["league"].map(lambda c: ALL_LEAGUES[c].kind if c in ALL_LEAGUES else None)
    kept = old[~kind_of.isin(refreshed)].drop(columns="fetched_at")
    parts = [p for p in [kept] + fresh if len(p)]
    fx = order(pd.concat(parts, ignore_index=True)) if parts else frame([])
    fx["fetched_at"] = first_seen(fx, old, stamp)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fx.to_csv(path, index=False)
    log(f"   fixtures: {len(fx)} still to play in {fx['league'].nunique()} leagues, fetched {stamp:%Y-%m-%d %H:%M} UTC")
    return fx


def sync(seasons=None, full=False, today=None, fetch=_fetch, log=print,
         matches_path=MATCHES, fixtures_path=FIXTURES, leagues="all"):
    """Refresh the snapshot (the module's notes, SYNC): the given seasons of
    the main divisions (default this one, plus last season in its first
    weeks so late corrections land), or every season since FIRST_SEASON
    with full=True; the extra leagues' files; both fixtures files. Seasons
    not fetched are kept as they are on file; a league-season the site does
    not have is skipped, never blanked. -> the leagues synced, as loaded."""
    problems = []
    sync_matches(leagues, seasons=seasons, full=full, today=today, fetch=fetch, log=log,
                 root=matches_path, problems=problems)
    merged = load_matches(matches_path, leagues=leagues)
    log(f"   matches: {len(merged)} on file in {merged['league'].nunique()} leagues, "
        f"{merged['result'].notna().sum()} with a result")
    sync_fixtures(leagues, fetch=fetch, log=log, path=fixtures_path, problems=problems)
    if problems:
        log(f"   KEPT ON FILE after a failed or unreadable download: {', '.join(problems)}")
    return merged


PRICE_COLUMNS = ("avg_h", "avg_d", "avg_a", "avg_o25", "avg_u25")


def first_seen(fx, old, stamp):
    """When each fixture's prices were first seen: the earlier fetch's time
    where the match was on file with the same prices, this fetch's
    otherwise. The fixtures file is re-fetched every run, and a Friday price
    must not be labelled as Sunday's."""
    out = pd.Series(stamp, index=fx.index)
    if old is None or not len(old) or "fetched_at" not in old:
        return out
    prev = old.drop_duplicates("match_id", keep="last").set_index("match_id")
    for i, row in fx.iterrows():
        if row["match_id"] not in prev.index:
            continue
        p = prev.loc[row["match_id"]]
        same = all((pd.isna(row[c]) and pd.isna(p[c])) or (not pd.isna(row[c]) and not pd.isna(p[c])
                   and abs(float(row[c]) - float(p[c])) < 1e-9) for c in PRICE_COLUMNS)
        if same and not pd.isna(p["fetched_at"]):
            out[i] = p["fetched_at"]
    return out


# ------------------------------------------------------------- loaders

def _typed(df):
    for c in INTEGER:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["date"] = df["date"].astype(str)
    return df


def load_matches(path=MATCHES, leagues=None):
    """Every match on file for the given leagues (league_codes: default the
    ten the pipeline runs on; "all" for every league), oldest first. `path`
    is the snapshot's folder; a single .csv.gz (the layout of before) is
    read as it always was."""
    path = Path(path)
    codes = league_codes(leagues)
    if path.is_file():
        df = _typed(pd.read_csv(path, low_memory=False))
        return df[df["league"].isin(codes)].reset_index(drop=True)
    # One read for every file: the season files share a header, and pandas
    # sees the same text the single file held, in another order.
    groups = defaultdict(list)
    for code in codes:
        for season in seasons_on_file(path, code):
            head, _, body = gzip.decompress(season_path(path, code, season).read_bytes()).decode().partition("\n")
            groups[head].append(body)
    if not groups:
        return _typed(frame([]))
    parts = []
    for head, bodies in groups.items():
        names = head.split(",")
        parts.append(pd.read_csv(io.StringIO(head + "\n" + "".join(bodies)), low_memory=False,
                                 dtype={c: "str" for c in TEXT if c in names}))
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    return order(_typed(df))


def load_fixtures(path=FIXTURES, leagues=None):
    """The fixtures on file for the given leagues (league_codes: default the
    ten the pipeline runs on; "all" for every league)."""
    if not Path(path).exists():
        return frame([], FIXTURE_COLUMNS)
    df = pd.read_csv(path, low_memory=False)
    for c in INTEGER:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["kickoff"] = pd.to_datetime(df["kickoff"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["date"] = df["date"].astype(str)
    return df[df["league"].isin(league_codes(leagues))].reset_index(drop=True)


if __name__ == "__main__":
    full = "--full" in sys.argv
    print(f"Soccer sync ({'every season' if full else 'this season'}) from football-data.co.uk, "
          f"{len(ALL_LEAGUES)} leagues")
    sync(full=full)
