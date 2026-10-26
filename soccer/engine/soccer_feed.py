"""Fixtures, kickoff times and provisional results between football-data.co.uk's
postings, from two free feeds: ESPN's public scoreboard and fixturedownload.com.

WHY. football-data.co.uk (soccer_data.py) is the engine's source of record,
but it posts the weekend's fixtures only on Friday afternoon, the midweek's
on Tuesday afternoon, and results days late: on Monday 2026-10-26 its newest
top-league results were from Oct 17-21 while the Oct 23-26 round had been
played. Both feeds here already carried that round's scores and the next
two rounds' kickoffs. This module reads them; it never replaces
football-data.co.uk - a match it has is taken from there (the wiring is
soccer_predict's job, not this module's).

THE TWO FEEDS (as probed on GitHub's machines, 2026-10-26 13:13 UTC, saved
under tests/fixtures/feeds/ by probe_feeds.py):

  ESPN scoreboard   one request per league per day (?dates=YYYYMMDD; a date
                    RANGE answers HTTP 400). Each event: an id, the kickoff
                    in UTC, a timeValid flag, a status (STATUS_SCHEDULED,
                    STATUS_FULL_TIME, ... - see ESPN_STATUS), the score.
                    It also carries one US book's moneylines; they are
                    measured against nothing and are not read.
  fixturedownload   the whole season per league in one request: kickoff in
                    UTC, the score once played (null before), the round.
                    No status and no flag for a time not yet set: an
                    unscheduled round carries a PLACEHOLDER clock -
                    00:00 UTC in La Liga, Serie A and the Bundesliga, local
                    midnight (22:00/23:00 UTC) in Ligue 1 - on a NOMINAL day
                    of the round (La Liga's Sunday, Serie A's Saturday, the
                    Bundesliga's Friday), so a placeholder's date is not the
                    match's date either. In the 2026 samples: 250 La Liga,
                    190 Serie A, 198 Bundesliga and 144 Ligue 1 rows, every
                    one in a round with no real time yet, none in the
                    complete 2025 seasons. The Premier League's placeholder
                    is Saturday 15:00 UK, which is also a real kickoff time:
                    it cannot be told apart, so a fixturedownload time is
                    never "confirmed" on its own (see reconcile).

ROWS. parse_espn and parse_fixturedownload return one dict per match, every
key always present (ROW_FIELDS); None means "the source does not say":

  source         "espn" | "fixturedownload"
  source_id      ESPN event id | "<season>-<MatchNumber>"
  request        the request that returned it ("espn:E0:20261025",
                 "fixturedownload:E0:2026"): a fetch that fails keeps
                 every row of its request as it is on file
  league         football-data.co.uk code (E0, SP1, I1, D1, F1)
  season         football-data.co.uk convention: 2026 for 2026-27
  home, away     football-data.co.uk club names (CLUB NAMES below)
  home_name, away_name   the feed's own spelling, for the audit trail
  kickoff_utc    "2026-10-25T14:00:00Z", or None when the source has no
                 real time (ESPN timeValid false; a placeholder)
  time_confirmed ESPN's timeValid; always False for fixturedownload
  uk_date        the Europe/London date of the kickoff - football-data.co.uk
                 dates a match on the UK clock, so match_id(league, uk_date,
                 home, away) is its id for the same match; None without a
                 real time
  date_hint      uk_date, or the day the source files a timeless match
                 under (a placeholder's nominal day, in the league's own
                 time zone)
  match_id       soccer_data.match_id(...) when uk_date is known, else None
  status         scheduled | live | final | postponed | cancelled |
                 abandoned | unknown (an ESPN status this module does not
                 know: warned, never final)
  status_detail  ESPN's status name; None for fixturedownload
  hg, ag         goals, only when status is final
  round          fixturedownload's round number; None for ESPN
  seen_at        when this row, as it now reads, was first fetched - the
                 point in time a result or a kickoff became known

CLUB NAMES. Explicit tables, per league, from each feed's names to
football-data.co.uk's: ESPN_CLUBS (by ESPN team id, with ESPN's display name
beside it) and FD_CLUBS (by fixturedownload's exact name). Nothing is ever
fuzzy-matched at run time: a club missing from a table is skipped with a
warning, and its matches with it. ESPN is keyed by its team id because the
id is ESPN's identity for a club and its names are not unique handles (its
abbreviation "MUN" is Bayern Munich, "MAN" Manchester United); the id and
display name agreed on every event in the samples. tests/test_soccer_feed.py
checks every club in the samples maps, one to one, to a club football-data.co.uk
has in that league and season.

RECONCILING (reconcile). One row per match: league, season, home, away - a
pairing is played once per league season.

  kickoff     each source's time is kept (kickoff_espn, kickoff_fd); the
              match's kickoff_utc is the EARLIER of the two, so a pick is
              frozen no later than either source's kickoff. time_confirmed
              only when ESPN says timeValid AND fixturedownload has the same
              time. None for a postponed or cancelled match.
  status      ESPN's, except that "final" needs the sources not to
              contradict: ESPN final with fixturedownload's same score, or
              with no score yet, is "final" with hg, ag and result - a
              PROVISIONAL result, until football-data.co.uk posts its own.
              ESPN final against a different fixturedownload score is
              "disputed" (no result, warned). A fixturedownload score that
              ESPN does not confirm (no ESPN row, or ESPN not final) is
              "unconfirmed" (no result). Postponed, cancelled or abandoned
              per ESPN wins whatever fixturedownload says.

MEASURED (tests/test_soccer_feed.py, on the probe of 2026-10-26):
  ESPN v football-data.co.uk   its own fixtures for Oct 23-26, 48 top-league
      matches either side of the clock change (BST ended on the 25th): every
      match_id and kickoff equal.
  fixturedownload v football-data.co.uk   2026-27 to the probe, 346 played
      matches: every match_id, kickoff and score equal. 2025-26, all 1752:
      every score equal (1751; one Ligue 1 score never posted), the match_id
      on 1751 (Mallorca v Valencia dated a day apart), the kickoff on 1743 -
      nine differ, by 10 minutes to 5 hours, either way. Hence a
      fixturedownload time is never confirmed alone, and the earlier time
      freezes a pick.
  ESPN v fixturedownload   the 130 matches ESPN listed for Oct 22 - Nov 7:
      every kickoff equal (all confirmed); 46 finals, every score equal.
  One fixturedownload error: 2026-27 Ligue 1 lists Rennes v Paris SG twice
      (round 1, played, and round 23) and Paris SG v Rennes never. Warned;
      reconcile uses the played one.

THE SNAPSHOT. sync() fetches and writes soccer/data/feed.json: the source
rows (normalised, sorted, one per line, no NaN) and when each request last
succeeded. A request that fails, or returns something that is not the feed,
keeps its rows as they are on file; a re-fetch that drops a final result
keeps it too (a played match does not leave the scoreboard). Rows more than
KEEP_BEHIND days old or KEEP_AHEAD days ahead are pruned. load() reads it,
so a run without the network - and every test - works from the file.

    python soccer/engine/soccer_feed.py            # fetch, write, summarise
    python soccer/engine/soccer_feed.py --offline  # summarise the snapshot
"""

import json
import math
import sys
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import soccer_data as sd

FEED = sd.DATA / "feed.json"
SCHEMA = 1
# Rows kept on file: three weeks back (soccer_log voids a pick whose match
# has no result 21 days after kickoff; football-data.co.uk has every result
# well before that) and four weeks ahead (beyond the fetch window, for a
# run whose fetch fails).
KEEP_BEHIND = 21
KEEP_AHEAD = 28

ESPN_SLUGS = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1"}
ESPN_BOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={day}"
FD_SLUGS = {"E0": "epl", "SP1": "la-liga", "I1": "serie-a", "D1": "bundesliga", "F1": "ligue-1"}
FD_URL = "https://fixturedownload.com/feed/json/{slug}-{season}"
LOCAL = {"E0": ZoneInfo("Europe/London"), "SP1": ZoneInfo("Europe/Madrid"), "I1": ZoneInfo("Europe/Rome"),
         "D1": ZoneInfo("Europe/Berlin"), "F1": ZoneInfo("Europe/Paris")}
UA = {"User-Agent": "YourSoccer fixtures sync (github.com/Klukemma/YourMMA)"}

ROW_FIELDS = ("source", "source_id", "request", "league", "season", "home", "away", "home_name", "away_name",
              "kickoff_utc", "time_confirmed", "uk_date", "date_hint", "match_id", "status", "status_detail",
              "hg", "ag", "round", "seen_at")
# What a row says: a change in any of these is new information (seen_at moves).
CONTENT = ("league", "season", "home", "away", "kickoff_utc", "time_confirmed", "uk_date", "date_hint",
           "status", "hg", "ag")
STATUSES = ("scheduled", "live", "final", "postponed", "cancelled", "abandoned", "unknown")
MATCH_STATUSES = STATUSES + ("disputed", "unconfirmed")

# ESPN status names. The probe's day boards held only STATUS_SCHEDULED (84
# events) and STATUS_FULL_TIME (46); the rest are ESPN's other soccer and
# generic names.
# "final" also needs ESPN's completed flag and the "post" state. A name not
# here is "unknown": warned, never final. STATUS_FINAL_AET / _PEN are left
# out on purpose: a league match has no extra time, and a score after it is
# not a league score.
BY_STATE = "by state"
ESPN_STATUS = {
    "STATUS_SCHEDULED": "scheduled", "STATUS_TBD": "scheduled",
    "STATUS_FIRST_HALF": "live", "STATUS_HALFTIME": "live", "STATUS_SECOND_HALF": "live",
    "STATUS_IN_PROGRESS": "live", "STATUS_END_PERIOD": "live", "STATUS_END_OF_REGULATION": "live",
    "STATUS_FULL_TIME": "final", "STATUS_FINAL": "final",
    "STATUS_POSTPONED": "postponed",
    "STATUS_CANCELED": "cancelled", "STATUS_CANCELLED": "cancelled",
    "STATUS_ABANDONED": "abandoned",
    "STATUS_FORFEIT": "abandoned", "STATUS_FORFEIT_HOME_TEAM": "abandoned", "STATUS_FORFEIT_AWAY_TEAM": "abandoned",
    # before kickoff, during the match, or the end of it: ESPN's state says which
    "STATUS_DELAYED": BY_STATE, "STATUS_RAIN_DELAY": BY_STATE, "STATUS_SUSPENDED": BY_STATE,
}
STATE_STATUS = {"pre": "scheduled", "in": "live", "post": "abandoned"}

# ------------------------------------------------------------- club names

# league: {ESPN team id: (ESPN displayName, football-data.co.uk name)}
ESPN_CLUBS = {
    "E0": {
        "349": ("AFC Bournemouth", "Bournemouth"), "359": ("Arsenal", "Arsenal"),
        "362": ("Aston Villa", "Aston Villa"), "337": ("Brentford", "Brentford"),
        "331": ("Brighton & Hove Albion", "Brighton"), "363": ("Chelsea", "Chelsea"),
        "388": ("Coventry City", "Coventry"), "384": ("Crystal Palace", "Crystal Palace"),
        "368": ("Everton", "Everton"), "370": ("Fulham", "Fulham"), "306": ("Hull City", "Hull"),
        "373": ("Ipswich Town", "Ipswich"), "357": ("Leeds United", "Leeds"), "364": ("Liverpool", "Liverpool"),
        "382": ("Manchester City", "Man City"), "360": ("Manchester United", "Man United"),
        "361": ("Newcastle United", "Newcastle"), "393": ("Nottingham Forest", "Nott'm Forest"),
        "366": ("Sunderland", "Sunderland"), "367": ("Tottenham Hotspur", "Tottenham"),
    },
    "SP1": {
        "96": ("Alavés", "Alaves"), "93": ("Athletic Club", "Ath Bilbao"), "1068": ("Atlético Madrid", "Ath Madrid"),
        "83": ("Barcelona", "Barcelona"), "85": ("Celta Vigo", "Celta"), "90": ("Deportivo", "La Coruna"),
        "3751": ("Elche", "Elche"), "88": ("Espanyol", "Espanol"), "2922": ("Getafe", "Getafe"),
        "1538": ("Levante", "Levante"), "99": ("Málaga", "Malaga"), "97": ("Osasuna", "Osasuna"),
        "87": ("Racing Santander", "Santander"), "101": ("Rayo Vallecano", "Vallecano"),
        "244": ("Real Betis", "Betis"), "86": ("Real Madrid", "Real Madrid"), "89": ("Real Sociedad", "Sociedad"),
        "243": ("Sevilla", "Sevilla"), "94": ("Valencia", "Valencia"), "102": ("Villarreal", "Villarreal"),
    },
    "I1": {
        "103": ("AC Milan", "Milan"), "104": ("AS Roma", "Roma"), "105": ("Atalanta", "Atalanta"),
        "107": ("Bologna", "Bologna"), "2925": ("Cagliari", "Cagliari"), "2572": ("Como", "Como"),
        "109": ("Fiorentina", "Fiorentina"), "4057": ("Frosinone", "Frosinone"), "3263": ("Genoa", "Genoa"),
        "110": ("Internazionale", "Inter"), "111": ("Juventus", "Juventus"), "112": ("Lazio", "Lazio"),
        "113": ("Lecce", "Lecce"), "4007": ("Monza", "Monza"), "114": ("Napoli", "Napoli"),
        "115": ("Parma", "Parma"), "3997": ("Sassuolo", "Sassuolo"), "239": ("Torino", "Torino"),
        "118": ("Udinese", "Udinese"), "17530": ("Venezia", "Venezia"),
    },
    "D1": {
        "598": ("1. FC Union Berlin", "Union Berlin"), "131": ("Bayer Leverkusen", "Leverkusen"),
        "132": ("Bayern Munich", "Bayern Munich"), "124": ("Borussia Dortmund", "Dortmund"),
        "268": ("Borussia Mönchengladbach", "M'gladbach"), "125": ("Eintracht Frankfurt", "Ein Frankfurt"),
        "3841": ("FC Augsburg", "Augsburg"), "122": ("FC Cologne", "FC Koln"), "127": ("Hamburg SV", "Hamburg"),
        "2950": ("Mainz", "Mainz"), "11420": ("RB Leipzig", "RB Leipzig"), "126": ("SC Freiburg", "Freiburg"),
        "3307": ("SC Paderborn 07", "Paderborn"), "10388": ("SV Elversberg", "Elversberg"),
        "133": ("Schalke 04", "Schalke 04"), "7911": ("TSG Hoffenheim", "Hoffenheim"),
        "134": ("VfB Stuttgart", "Stuttgart"), "137": ("Werder Bremen", "Werder Bremen"),
    },
    "F1": {
        "172": ("AJ Auxerre", "Auxerre"), "174": ("AS Monaco", "Monaco"), "7868": ("Angers", "Angers"),
        "6997": ("Brest", "Brest"), "3236": ("Le Havre AC", "Le Havre"), "2697": ("Le Mans", "Le Mans"),
        "175": ("Lens", "Lens"), "166": ("Lille", "Lille"), "273": ("Lorient", "Lorient"), "167": ("Lyon", "Lyon"),
        "176": ("Marseille", "Marseille"), "2502": ("Nice", "Nice"), "6851": ("Paris FC", "Paris FC"),
        "160": ("Paris Saint-Germain", "Paris SG"), "169": ("Stade Rennais", "Rennes"),
        "180": ("Strasbourg", "Strasbourg"), "179": ("Toulouse", "Toulouse"), "170": ("Troyes", "Troyes"),
    },
}

# league: {fixturedownload name: football-data.co.uk name}. The 2025-26 and
# 2026-27 clubs; fixturedownload renamed one between them (Serie A's "Inter"
# is "Internazionale" from 2026), so two names may point at one club - within
# a season the table is one to one.
FD_CLUBS = {
    "E0": {
        "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
        "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley", "Chelsea": "Chelsea",
        "Coventry": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton", "Fulham": "Fulham",
        "Hull": "Hull", "Ipswich": "Ipswich", "Leeds": "Leeds", "Liverpool": "Liverpool", "Man City": "Man City",
        "Man Utd": "Man United", "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest", "Spurs": "Tottenham",
        "Sunderland": "Sunderland", "West Ham": "West Ham", "Wolves": "Wolves",
    },
    "SP1": {
        "Athletic Club": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid", "CA Osasuna": "Osasuna",
        "Celta": "Celta", "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "FC Barcelona": "Barcelona",
        "Getafe CF": "Getafe", "Girona FC": "Girona", "Levante UD": "Levante", "Málaga CF": "Malaga",
        "R. Racing Club": "Santander", "RC Deportivo": "La Coruna", "RCD Espanyol de Barcelona": "Espanol",
        "RCD Mallorca": "Mallorca", "Rayo Vallecano": "Vallecano", "Real Betis": "Betis",
        "Real Madrid": "Real Madrid", "Real Oviedo": "Oviedo", "Real Sociedad": "Sociedad",
        "Sevilla FC": "Sevilla", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
    },
    "I1": {
        "Atalanta": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari", "Como": "Como",
        "Cremonese": "Cremonese", "Fiorentina": "Fiorentina", "Frosinone": "Frosinone", "Genoa": "Genoa",
        "Hellas Verona": "Verona", "Inter": "Inter", "Internazionale": "Inter", "Juventus": "Juventus",
        "Lazio": "Lazio", "Lecce": "Lecce", "Milan": "Milan", "Monza": "Monza", "Napoli": "Napoli",
        "Parma": "Parma", "Pisa": "Pisa", "Roma": "Roma", "Sassuolo": "Sassuolo", "Torino": "Torino",
        "Udinese": "Udinese", "Venezia": "Venezia",
    },
    "D1": {
        "1. FC Heidenheim 1846": "Heidenheim", "1. FC Köln": "FC Koln", "1. FC Union Berlin": "Union Berlin",
        "1. FSV Mainz 05": "Mainz", "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
        "Borussia Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich", "FC Schalke 04": "Schalke 04",
        "FC St. Pauli": "St Pauli", "Hamburger SV": "Hamburg", "RB Leipzig": "RB Leipzig",
        "SC Paderborn 07": "Paderborn", "SV Elversberg": "Elversberg", "SV Werder Bremen": "Werder Bremen",
        "Sport-Club Freiburg": "Freiburg", "TSG Hoffenheim": "Hoffenheim", "VfB Stuttgart": "Stuttgart",
        "VfL Wolfsburg": "Wolfsburg",
    },
    "F1": {
        "AJ Auxerre": "Auxerre", "AS Monaco": "Monaco", "Angers SCO": "Angers", "Estac Troyes": "Troyes",
        "FC Lorient": "Lorient", "FC Metz": "Metz", "FC Nantes": "Nantes", "Havre Athletic Club": "Le Havre",
        "LOSC Lille": "Lille", "Le Mans FC": "Le Mans", "OGC Nice": "Nice", "Olympique Lyonnais": "Lyon",
        "Olympique de Marseille": "Marseille", "Paris FC": "Paris FC", "Paris Saint-Germain": "Paris SG",
        "RC Lens": "Lens", "RC Strasbourg Alsace": "Strasbourg", "Stade Brestois 29": "Brest",
        "Stade Rennais FC": "Rennes", "Toulouse FC": "Toulouse",
    },
}


class FeedError(ValueError):
    """A response that is not the feed it should be (an error page, a format change)."""


def _nfc(text):
    return unicodedata.normalize("NFC", str(text or "")).strip()


def espn_club(league, team, log=print):
    """ESPN's team object -> football-data.co.uk's name, or None (warned)."""
    team = team or {}
    tid, name = str(team.get("id") or ""), _nfc(team.get("displayName"))
    known = ESPN_CLUBS.get(league, {}).get(tid)
    if known is None:
        log(f"   espn {league}: no club mapping for ESPN team {tid or '?'} '{name}'; its match is skipped")
        return None
    if name and _nfc(known[0]) != name:
        log(f"   espn {league}: ESPN team {tid} is now called '{name}' (table: '{known[0]}'); mapped by its id")
    return known[1]


def fd_club(league, name, log=print):
    """fixturedownload's name -> football-data.co.uk's, or None (warned)."""
    key = _nfc(name)
    club = FD_CLUBS.get(league, {}).get(key)
    if club is None:
        log(f"   fixturedownload {league}: no club mapping for '{key}'; its match is skipped")
    return club


# ------------------------------------------------------------- time

def parse_utc(text):
    """'2026-10-25T14:00Z' (ESPN) or '2026-10-25 14:00:00Z' (fixturedownload) -> aware UTC datetime."""
    t = str(text).strip().replace(" ", "T")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    moment = datetime.fromisoformat(t)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def stamp(moment):
    """An aware or naive-UTC datetime/Timestamp -> '2026-10-25T14:00:00Z'."""
    if getattr(moment, "tzinfo", None) is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def uk_date(moment):
    """The Europe/London calendar date of a UTC moment: the date
    football-data.co.uk files the match under."""
    return moment.astimezone(sd.UK).date()


def _aware(now):
    """now as an aware UTC datetime: None (the clock), a datetime or pandas
    Timestamp (naive means UTC), or text."""
    now = now if now is not None else datetime.now(timezone.utc)
    if isinstance(now, str):
        return parse_utc(now)
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)


def _goals(score):
    """ESPN's score ("2", or {"value": 2.0, ...}) -> int, or None."""
    if isinstance(score, dict):
        score = score.get("value", score.get("displayValue"))
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0 or value != int(value):
        return None
    return int(value)


def _row(**kw):
    row = dict.fromkeys(ROW_FIELDS)
    row.update(kw)
    if row["uk_date"] and row["match_id"] is None:
        row["match_id"] = sd.match_id(row["league"], date.fromisoformat(row["uk_date"]), row["home"], row["away"])
    return row


# ------------------------------------------------------------- ESPN

def espn_status(kind, log=print, where=""):
    """ESPN's status.type -> (status, ESPN's name). Final only with ESPN's
    completed flag and post state as well; an unknown name is 'unknown'."""
    kind = kind or {}
    name, state, completed = kind.get("name"), kind.get("state"), kind.get("completed")
    status = ESPN_STATUS.get(name)
    if status == BY_STATE:
        status = STATE_STATUS.get(state)
    if status is None:
        log(f"   espn{where}: unknown status {name!r} (state {state!r}); not treated as final")
        return "unknown", name
    if status == "final" and not (completed is True and state == "post"):
        log(f"   espn{where}: {name} but completed={completed!r}, state={state!r}; not treated as final")
        return "unknown", name
    return status, name


def parse_espn(payload, league, request=None, seen_at=None, log=print):
    """One ESPN scoreboard response -> rows. Raises FeedError for a body that
    is not a scoreboard; an event it cannot read is skipped and warned."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise FeedError("not an ESPN scoreboard (no events list)")
    rows = []
    for event in payload["events"]:
        try:
            row = _espn_event(event, league, request, seen_at, log)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as err:
            eid = event.get("id") if isinstance(event, dict) else None
            log(f"   espn {league}: event {eid or '?'} unreadable ({err.__class__.__name__}: {err}); skipped")
            continue
        if row is not None:
            rows.append(row)
    return rows


def _espn_event(event, league, request, seen_at, log):
    comp = (event.get("competitions") or [{}])[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    if "home" not in sides or "away" not in sides:
        log(f"   espn {league}: event {event.get('id')} has no home and away side; skipped")
        return None
    home = espn_club(league, sides["home"].get("team"), log)
    away = espn_club(league, sides["away"].get("team"), log)
    if home is None or away is None:
        return None
    when = parse_utc(comp.get("date") or event["date"])
    valid = comp.get("timeValid") is True
    status, detail = espn_status((comp.get("status") or event.get("status") or {}).get("type"), log,
                                 f" {league} {home} v {away}")
    hg = ag = None
    if status == "final":
        hg, ag = _goals(sides["home"].get("score")), _goals(sides["away"].get("score"))
        if hg is None or ag is None:
            log(f"   espn {league} {home} v {away}: final without a readable score; not treated as final")
            status = "unknown"
            hg = ag = None
    day = uk_date(when)
    season = sd.season_of(day.isoformat())
    espn_season = (event.get("season") or {}).get("year")
    if espn_season is not None and int(espn_season) != season:
        log(f"   espn {league} {home} v {away}: ESPN files it in season {espn_season}, its date in {season}")
    return _row(source="espn", source_id=str(event["id"]), request=request, league=league, season=season,
                home=home, away=away, home_name=_nfc(sides["home"]["team"].get("displayName")),
                away_name=_nfc(sides["away"]["team"].get("displayName")),
                kickoff_utc=stamp(when) if valid else None, time_confirmed=valid,
                uk_date=day.isoformat() if valid else None, date_hint=day.isoformat(),
                status=status, status_detail=detail, hg=hg, ag=ag, seen_at=seen_at)


# ------------------------------------------------------------- fixturedownload

def is_placeholder(moment, league):
    """fixturedownload's clock for a kickoff not yet set: midnight UTC (La
    Liga, Serie A, the Bundesliga) or midnight in the league's own time
    zone (Ligue 1). No top-five league kicks off at either."""
    local = moment.astimezone(LOCAL[league])
    return (moment.hour, moment.minute) == (0, 0) or (local.hour, local.minute) == (0, 0)


def parse_fixturedownload(rows, league, season, request=None, seen_at=None, log=print):
    """One fixturedownload season feed -> rows. Raises FeedError for a body
    that is not one; a row it cannot read is skipped and warned."""
    if not isinstance(rows, list):
        raise FeedError("not a fixturedownload feed (not a list)")
    out, pairs = [], defaultdict(list)
    for raw in rows:
        try:
            row = _fd_row(raw, league, season, request, seen_at, log)
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            log(f"   fixturedownload {league}: row unreadable ({err.__class__.__name__}: {err}); skipped")
            continue
        if row is not None:
            out.append(row)
            pairs[(row["home"], row["away"])].append(row["source_id"])
    for (home, away), ids in pairs.items():
        if len(ids) > 1:
            log(f"   fixturedownload {league} {season}: {home} v {away} listed {len(ids)} times ({', '.join(ids)}); "
                f"a pairing is played once a season - the feed has home and away swapped somewhere")
    return out


def _fd_row(raw, league, season, request, seen_at, log):
    home, away = fd_club(league, raw["HomeTeam"], log), fd_club(league, raw["AwayTeam"], log)
    if home is None or away is None:
        return None
    when = parse_utc(raw["DateUtc"])
    placeholder = is_placeholder(when, league)
    hs, as_ = raw.get("HomeTeamScore"), raw.get("AwayTeamScore")
    hg, ag = _goals(hs), _goals(as_)
    if hs is None and as_ is None:
        status = "scheduled"
    elif hg is not None and ag is not None:
        status = "final"
    else:
        log(f"   fixturedownload {league} {home} v {away}: score {hs!r}-{as_!r} unreadable; not treated as final")
        status, hg, ag = "scheduled", None, None
    day = uk_date(when)
    hint = when.astimezone(LOCAL[league]).date() if placeholder else day
    if sd.season_of(hint.isoformat()) != int(season):
        log(f"   fixturedownload {league} {home} v {away}: dated {hint}, outside season {season}")
    return _row(source="fixturedownload", source_id=f"{int(season)}-{raw.get('MatchNumber')}", request=request,
                league=league, season=int(season), home=home, away=away,
                home_name=_nfc(raw["HomeTeam"]), away_name=_nfc(raw["AwayTeam"]),
                kickoff_utc=None if placeholder else stamp(when), time_confirmed=False,
                uk_date=None if placeholder else day.isoformat(), date_hint=hint.isoformat(),
                status=status, hg=hg if status == "final" else None, ag=ag if status == "final" else None,
                round=raw.get("RoundNumber"), seen_at=seen_at)


# ------------------------------------------------------------- network

def _get(url, timeout=30):
    """The body of one request; raises on anything but a 200."""
    try:
        import requests
        response = requests.get(url, timeout=timeout, headers=UA)
        response.raise_for_status()
        return response.content
    except ImportError:
        from urllib.request import Request, urlopen
        with urlopen(Request(url, headers=UA), timeout=timeout) as handle:
            return handle.read()


def _decode(body):
    if body is None:
        raise FeedError("nothing returned")
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8-sig")
    if isinstance(body, str):
        return json.loads(body)
    return body                                           # already parsed (a test's getter)


def espn_days(now, days_back=4, days_ahead=10):
    today = _aware(now).date()
    return [today + timedelta(days=k) for k in range(-days_back, days_ahead + 1)]


def requests_for(now, days_back=4, days_ahead=10, leagues=tuple(ESPN_SLUGS)):
    """[(request key, url, league, kind, season)] for one fetch."""
    season = sd.current_season(_aware(now).date())
    out = []
    for league in leagues:
        for day in espn_days(now, days_back, days_ahead):
            out.append((f"espn:{league}:{day:%Y%m%d}",
                        ESPN_BOARD.format(slug=ESPN_SLUGS[league], day=f"{day:%Y%m%d}"), league, "espn", season))
        out.append((f"fixturedownload:{league}:{season}", FD_URL.format(slug=FD_SLUGS[league], season=season),
                    league, "fixturedownload", season))
    return out


def fetch(now=None, days_back=4, days_ahead=10, get=_get, leagues=tuple(ESPN_SLUGS), log=print):
    """ESPN's scoreboard one league and UTC day at a time from days_back
    before now to days_ahead after (the scoreboard's calendar starts its days
    at 07:00 or 08:00 UTC - midnight US Pacific - before any European
    kickoff, so a European match is on the same day either way), plus
    fixturedownload's current season per league.

    Returns {"at", "rows", "ok": {request: rows}, "failed": {request: why}}.
    A request that fails is only reported: merge() keeps its rows on file."""
    at = stamp(_aware(now))
    rows, ok, failed = [], {}, {}
    for key, url, league, kind, season in requests_for(now, days_back, days_ahead, leagues):
        try:
            payload = _decode(get(url))
            if kind == "espn":
                got = parse_espn(payload, league, request=key, seen_at=at, log=log)
            else:
                got = parse_fixturedownload(payload, league, season, request=key, seen_at=at, log=log)
                if not got:
                    raise FeedError("no rows parsed (format change?)")
        except Exception as err:                       # noqa: BLE001 - reported, kept on file, run continues
            failed[key] = f"{err.__class__.__name__}: {str(err)[:120]}"
            log(f"   {key}: FAILED ({failed[key]}); kept what is on file")
            continue
        ok[key] = len(got)
        rows.extend(got)
    log(f"   feeds: {len(ok)} requests answered, {len(failed)} failed, {len(rows)} rows")
    return {"at": at, "rows": rows, "ok": ok, "failed": failed}


# ------------------------------------------------------------- the snapshot

def row_key(row):
    return (row["source"], row["league"], str(row["source_id"]))


def sort_key(row):
    return (row.get("date_hint") or "", row.get("kickoff_utc") or "", row["league"], row["home"], row["away"],
            row["source"], str(row["source_id"]))


def merge(old_rows, fetched, now=None, log=print):
    """The snapshot's rows after one fetch.

    Every row a successful request returned replaces the row on file with
    the same source id (wherever it was filed: a moved match moves). A row
    on file whose request was answered without it is dropped (moved or
    withdrawn) - unless it is a final result, which is kept (a played match
    does not leave the scoreboard). A failed request touches nothing. Each
    row keeps the seen_at of the first fetch that read it as it now reads."""
    now = _aware(now if now is not None else fetched.get("at"))
    answered = set(fetched.get("ok", {}))
    old_by_key = {row_key(r): r for r in old_rows}
    fresh = {}
    for r in fetched.get("rows", []):
        k = row_key(r)
        if k in fresh:
            continue                                   # listed twice in one fetch: the first
        before = old_by_key.get(k)
        r = dict(r)
        if before is not None and all(before.get(c) == r.get(c) for c in CONTENT) and before.get("seen_at"):
            r["seen_at"] = before["seen_at"]
        elif before is not None and before.get("status") == "final" and r.get("status") != "final":
            log(f"   {r['source']} {r['league']} {r['home']} v {r['away']}: was final "
                f"{before.get('hg')}-{before.get('ag')}, now {r.get('status')}")
        fresh[k] = r
    kept = []
    for r in old_rows:
        k = row_key(r)
        if k in fresh:
            continue
        if r.get("request") in answered:
            if r.get("status") == "final":
                log(f"   {r['request']}: no longer lists {r['home']} v {r['away']} "
                    f"({r['hg']}-{r['ag']}); the result is kept")
                kept.append(r)
            continue
        kept.append(r)
    lo = (now - timedelta(days=KEEP_BEHIND)).date().isoformat()
    hi = (now + timedelta(days=KEEP_AHEAD)).date().isoformat()
    rows = [r for r in kept + list(fresh.values()) if lo <= (r.get("date_hint") or "") <= hi]
    return sorted(rows, key=sort_key)


def empty():
    return {"schema": SCHEMA, "requests": {}, "last_run": None, "rows": []}


def load(path=FEED):
    path = Path(path)
    if not path.exists():
        return empty()
    snap = json.loads(path.read_text(encoding="utf-8"))
    if snap.get("schema") != SCHEMA:
        raise ValueError(f"feed snapshot is schema {snap.get('schema')}, expected {SCHEMA}")
    return snap


def _clean(x):
    """JSON-safe: NaN -> None; numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def dumps(snap):
    """Stable text: header keys sorted, one row per line in sort order, so
    an unchanged snapshot is byte-identical and a diff shows the rows that
    changed."""
    def one(x):
        return json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    snap = _clean(snap)
    head = {k: v for k, v in snap.items() if k != "rows"}
    lines = [f' "{k}": {one(head[k])},' for k in sorted(head)]
    rows = [one({f: r.get(f) for f in ROW_FIELDS}) for r in sorted(snap.get("rows", []), key=sort_key)]
    body = ",\n  ".join(rows)
    return "{\n" + "\n".join(lines) + '\n "rows": [' + (f"\n  {body}\n ]" if rows else "]") + "\n}\n"


def save(snap, path=FEED):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(snap), encoding="utf-8")
    return path


def sync(now=None, path=FEED, get=_get, days_back=4, days_ahead=10, leagues=tuple(ESPN_SLUGS), log=print):
    """Fetch both feeds, merge into the snapshot on file, write it. Returns
    the snapshot; its last_run says what answered and what failed."""
    now = _aware(now)
    snap = load(path)
    got = fetch(now, days_back, days_ahead, get=get, leagues=leagues, log=log)
    rows = merge(snap.get("rows", []), got, now, log=log)
    requests = dict(snap.get("requests") or {})
    requests.update({k: got["at"] for k in got["ok"]})
    live = {r["request"] for r in rows}
    oldest = (now - timedelta(days=KEEP_BEHIND)).strftime("%Y%m%d")
    requests = {k: v for k, v in requests.items()
                if k in live or k in got["ok"] or (k.startswith("espn:") and k.rsplit(":", 1)[1] >= oldest)}
    snap = {"schema": SCHEMA, "requests": requests, "rows": rows,
            "last_run": {"at": got["at"], "answered": len(got["ok"]), "failed": dict(sorted(got["failed"].items()))}}
    save(snap, path)
    log(f"   feed snapshot: {len(rows)} rows ({sum(r['source'] == 'espn' for r in rows)} ESPN, "
        f"{sum(r['source'] == 'fixturedownload' for r in rows)} fixturedownload) -> {path}")
    return snap


# ------------------------------------------------------------- reconcile

def match_key(row):
    return (row["league"], int(row["season"]), row["home"], row["away"])


def _score(row):
    return (row["hg"], row["ag"]) if row is not None and row.get("status") == "final" else None


def _one_espn(rows, notes):
    if len(rows) <= 1:
        return rows[0] if rows else None
    on = [r for r in rows if r["status"] not in ("postponed", "cancelled")]
    if len(on) == 1:
        notes.append(f"ESPN lists this pairing {len(rows)} times; event {on[0]['source_id']} is the one not "
                     f"postponed or cancelled")
        return on[0]
    pick = max(on or rows, key=lambda r: (r["kickoff_utc"] or r["date_hint"] or "", str(r["source_id"])))
    notes.append(f"ESPN lists this pairing {len(rows)} times "
                 f"({', '.join(sorted(str(r['source_id']) for r in rows))}); using the latest, {pick['source_id']}")
    return pick


def _one_fd(rows, notes):
    """fixturedownload listing a pairing twice is an error in its feed (its
    2026-27 Ligue 1 has Rennes v Paris SG in round 1, played 2-2, and again
    in round 23, where Paris SG v Rennes is missing: home and away swapped).
    The one with a score is the match played; otherwise the earliest."""
    if len(rows) <= 1:
        return rows[0] if rows else None
    played = [r for r in rows if r["status"] == "final"]
    pick = (played or sorted(rows, key=lambda r: (r["date_hint"] or "", str(r["source_id"]))))[0]
    notes.append(f"fixturedownload lists this pairing {len(rows)} times "
                 f"({', '.join(sorted(str(r['source_id']) for r in rows))}); using {pick['source_id']}"
                 + (", the one played" if played else ", the earliest"))
    return pick


def reconcile(espn_rows, fd_rows, log=print):
    """One row per match (league, season, home, away) from both feeds' rows.

    Each row: match_id, league, season, home, away, uk_date, date_hint,
    kickoff_utc (the EARLIER of the sources' times: the freeze time; None
    without one, or for a postponed or cancelled match), kickoff_espn,
    kickoff_fd, time_confirmed, status (MATCH_STATUSES), status_espn,
    status_fd, hg, ag, result (H/D/A, only for an accepted - provisional -
    final), provisional, result_sources, result_seen_at, score_espn,
    score_fd, espn_id, fd_id, round, sources, warnings."""
    groups = defaultdict(lambda: ([], []))
    for r in espn_rows:
        groups[match_key(r)][0].append(r)
    for r in fd_rows:
        groups[match_key(r)][1].append(r)
    out = []
    for key in sorted(groups):
        row = _reconcile_one(key, *groups[key])
        for note in row["warnings"]:
            log(f"   {row['league']} {row['home']} v {row['away']}: {note}")
        out.append(row)
    return sorted(out, key=lambda m: (m["date_hint"] or "", m["kickoff_utc"] or "", m["league"], m["home"]))


def _reconcile_one(key, espn_rows, fd_rows):
    league, season, home, away = key
    notes = []
    e, f = _one_espn(espn_rows, notes), _one_fd(fd_rows, notes)
    ke, kf = (e or {}).get("kickoff_utc"), (f or {}).get("kickoff_utc")
    times = sorted(t for t in (ke, kf) if t)
    when = times[0] if times else None
    if ke and kf and ke != kf:
        notes.append(f"kickoff differs: ESPN {ke}, fixturedownload {kf}; the earlier is kept")
    confirmed = bool(e and e["time_confirmed"] and ke and kf and ke == kf)

    se, sfd = (e or {}).get("status"), (f or {}).get("status")
    es, fs = _score(e), _score(f)
    result = None
    if se in ("postponed", "cancelled", "abandoned"):
        status = se
        if fs:
            notes.append(f"fixturedownload has a score {fs[0]}-{fs[1]} but ESPN says {se}; ESPN's {se} stands")
    elif se == "final":
        if fs is None or fs == es:
            status, result = "final", es
        else:
            status = "disputed"
            notes.append(f"scores contradict: ESPN {es[0]}-{es[1]}, fixturedownload {fs[0]}-{fs[1]}; "
                         f"no result accepted")
    elif fs:
        # One source's score is not a result. With no ESPN row at all (a
        # match outside the window ESPN was asked about) that is no
        # contradiction, so not warned.
        status = "unconfirmed"
        if e:
            notes.append(f"fixturedownload has {fs[0]}-{fs[1]} but ESPN says {se}; not accepted")
    else:
        status = se or sfd or "scheduled"

    day = uk_date(parse_utc(when)) if when else None
    hint = day.isoformat() if day else ((e or {}).get("date_hint") or (f or {}).get("date_hint"))
    hg, ag = result if result else (None, None)
    return {
        "match_id": sd.match_id(league, day, home, away) if day else None,
        "league": league, "season": season, "home": home, "away": away,
        "uk_date": day.isoformat() if day else None, "date_hint": hint,
        "kickoff_utc": None if status in ("postponed", "cancelled") else when,
        "kickoff_espn": ke, "kickoff_fd": kf, "time_confirmed": confirmed,
        "status": status, "status_espn": se, "status_fd": sfd,
        "hg": hg, "ag": ag,
        "result": None if result is None else "H" if hg > ag else "A" if hg < ag else "D",
        "provisional": result is not None,
        "result_sources": (["espn", "fixturedownload"] if fs else ["espn"]) if result else None,
        "result_seen_at": e["seen_at"] if result else None,
        "score_espn": list(es) if es else None, "score_fd": list(fs) if fs else None,
        "espn_id": (e or {}).get("source_id"), "fd_id": (f or {}).get("source_id"), "round": (f or {}).get("round"),
        "sources": [s for s, r in (("espn", e), ("fixturedownload", f)) if r is not None],
        "warnings": notes,
    }


def reconciled(snap=None, path=FEED, log=print):
    """The snapshot on file (or the one given), reconciled."""
    snap = snap if snap is not None else load(path)
    rows = snap.get("rows", [])
    return reconcile([r for r in rows if r["source"] == "espn"],
                     [r for r in rows if r["source"] == "fixturedownload"], log=log)


def summary(matches, log=print):
    by = defaultdict(lambda: defaultdict(int))
    for m in matches:
        by[m["league"]][m["status"]] += 1
    for league in sorted(by):
        log(f"   {league}: " + ", ".join(f"{n} {s}" for s, n in sorted(by[league].items())))


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if not offline:
        print("Fixture feeds (ESPN, fixturedownload)")
        sync()
    summary(reconciled(log=lambda *_: None))
