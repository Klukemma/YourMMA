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

EVERY LEAGUE (the league probe, GitHub's machines 2026-10-29 12:43 UTC, saved
under tests/fixtures/feeds/leagues/ by probe_leagues.py; checked in
tests/test_soccer_feed_leagues.py). Of soccer_data's 38 leagues, 31 have a
feed - FEED_LEAGUES - and 7 have none (football-data.co.uk alone):

  ESPN and fixturedownload   E0 SP1 I1 D1 F1, E1 (championship), F2
                             (ligue-2), SC0 (scottish-premiership), N1
                             (eredivisie), P1 (primeira-liga), T1 (super-lig),
                             USA (mls)
  ESPN only                  SP2 I2 D2, E2 E3 EC, SC1, B1, G1, ARG AUT BRA
                             CHN DNK JPN MEX NOR RUS SWE
  none                       SC2 SC3 ROU SWZ (ESPN listed 2025-26's clubs and
                             no events Oct 26 - Nov 8, while football-data.co.uk
                             had SC2 and ROU matches those days), FIN IRL
                             (ESPN listed no clubs; FIN had 3 matches those
                             days), POL (no ESPN league: pol.1 is a 404) -
                             ESPN_NOT_ASKED. fixturedownload answered for no
                             other league football-data.co.uk carries (of 28
                             slugs tried; its three European cups are not
                             leagues here).

A sync asks about LIVE_LEAGUES unless told otherwise - the five the app
predicts today (soccer_schedule joins only those): 80 requests. Every league
is leagues=FEED_LEAGUES: 31 x 15 ESPN days and 12 season files, 477
requests, about 80 seconds on GitHub's runner (the league probe made 997 in
161 s). RUS: ESPN lists its 2026-27 rounds, football-data.co.uk's file
stopped at 2026-08-02 (soccer_data). fixturedownload's placeholders, in its
2026-27 files: midnight UTC in the Eredivisie (100 rows) and the Primeira
Liga, local midnight in Ligue 2 (171), the Primeira Liga's summer time (198
with its UTC ones) and the Super Lig (162), each a whole unscheduled round
but one Eredivisie match moved with no time (Twente v Sparta, filed at
00:00 UTC on 2027-02-06); the Championship's and the Scottish Premiership's
read Saturday 15:00 UK like the Premier League's. MLS has none, and kicks
off at 00:00 UTC for real (LOCAL: no placeholder rule there).

ROWS. parse_espn and parse_fixturedownload return one dict per match, every
key always present (ROW_FIELDS); None means "the source does not say":

  source         "espn" | "fixturedownload"
  source_id      ESPN event id | "<season>-<MatchNumber>"
  request        the request that returned it ("espn:E0:20261025",
                 "fixturedownload:E0:2026"): a fetch that fails keeps
                 every row of its request as it is on file
  league         football-data.co.uk code (soccer_data.ALL_LEAGUES: E0, SP1,
                 ..., USA)
  season         football-data.co.uk convention for the league
                 (soccer_data.season_for): the year the season starts, 2026
                 for 2026-27 - a calendar year where the league plays one
                 (ARG BRA CHN FIN IRL NOR SWE USA; JPN until 2025, August-May
                 since). A request is for the league's own current season:
                 in January MLS's is the new year's, the Championship's not.
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
warning, and its matches with it (counted in the snapshot's
last_run.unmapped). The tables hold 2026-27's clubs (and fixturedownload's
2025-26 ones): they need the promoted clubs added EVERY SUMMER - a snapshot
check (tests, never a gate) reports any club football-data.co.uk has this
season that either table does not map. ESPN is keyed by its team id because the
id is ESPN's identity for a club and its names are not unique handles (its
abbreviation "MUN" is Bayern Munich, "MAN" Manchester United); the id and
display name agreed on every event in the samples. tests/test_soccer_feed.py
checks every club in the samples maps, one to one, to a club football-data.co.uk
has in that league and season.
  Every other league's tables were built from the league probe the same
way: normalised names only PROPOSED pairs, each was checked by hand, and the
tests hold each table to exactly the clubs ESPN listed (its teams page and
every event, by id, under the same display name) and football-data.co.uk
has that season - one to one. A calendar-year league's tables hold 2026's
clubs: its promoted ones arrive in the winter. fixturedownload renamed MLS's
clubs between 2025 ("LA", "New York", "D.C.") and 2026 ("LA Galaxy", "Red
Bull New York", "D.C. United"), so both sets are in its table.

RECONCILING (reconcile). One row per match: league, season, home, away - a
pairing is played once per league season - EXCEPT in the 15 leagues whose
format plays a pairing more than once a season (REPEATS: a split, a third or
fourth round robin, play-offs, two tournaments a season - every league where
football-data.co.uk has a pairing twice in a 2021-2025 season, often within
the feeds' seven-week window - within 14 days 13 times in MLS, 3 in Romania,
2 in Liga MX and Austria). There every ESPN event is a match, and a
fixturedownload row is the ESPN event's nearest in date within PAIR_DAYS (3:
half MLS's shortest gap, 7 days) or a match of its own (legs). So there a
match ESPN moves by more than three days while fixturedownload does not is
two rows, neither confirmed, until fixturedownload catches up; and ESPN
listing one match as two events (a postponed one beside its new date - not
seen: the probe's one postponed event, in Scotland, stood alone) would be
two rows, one postponed. NOTE for soccer_schedule and soccer_log: they key a
match by its pairing, which holds for the five they join today and not for
REPEATS.

  kickoff     each source's time is kept (kickoff_espn, kickoff_fd); the
              match's kickoff_utc - its date, its id, the time shown - is
              ESPN's where ESPN has a valid one, fixturedownload's
              otherwise: ESPN is the feed that tracks a move (MEASURED
              below), and the one that reports status and score. The
              earlier of the two (kickoff_first) still closes a pick
              (soccer_schedule). time_confirmed only when ESPN says
              timeValid AND fixturedownload has the same time. kickoff_utc
              is None for a postponed or cancelled match; its id and date
              are still the date it was postponed from.
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
      fixturedownload time is never confirmed alone, and an earlier time
      from either feed closes a pick.
  ESPN v fixturedownload   the 130 matches ESPN listed for Oct 22 - Nov 7:
      every kickoff equal (all confirmed); 46 finals, every score equal.
  A day later (the runner's snapshot of 2026-10-27 12:48): of the 58 of
      those still to play with a time both feeds agreed on, ESPN had moved
      three - all Ligue 1, round 10: Nice v Rennes 22h15m EARLIER (Fri
      18:00), Lyon v Angers 20h30m later, Lorient v Brest two days later -
      and fixturedownload none (its later Ligue 1 rounds look like a
      template: rounds 11-14 and 16-18 each read Fri 20:45 once, Sat 17:15
      once, Sat 20:45 four times, ...). So two feeds agreeing ("confirmed")
      is not a guarantee a week out, and a time can move EARLIER after it:
      every run re-reads the feeds before it logs, ESPN's "live" stops a
      pick, and what guarantees the record is that a pick logged at or
      after the real kickoff is void (soccer_log). Where the two disagree,
      ESPN's time is the match's.
  One fixturedownload error: 2026-27 Ligue 1 lists Rennes v Paris SG twice
      (round 1, played, and round 23) and Paris SG v Rennes never. Warned;
      reconcile uses the played one.

MEASURED, EVERY OTHER LEAGUE (tests/test_soccer_feed_leagues.py, on the league
probe of 2026-10-29, against football-data.co.uk as the same probe
downloaded it, and the second tiers' snapshot frozen that day):
  ESPN   all 531 events of the 26 leagues (Oct 26 - Nov 8) read and map, none
      warned; 15 of Argentina's have no time yet. Every football-data.co.uk
      match of those leagues in those days - 82: the 69 soccer_data paired by
      name likeness, and 13 of the second tiers' - is a match reconcile makes,
      under the same match_id at the same kickoff; the 48 with a result have
      ESPN's score. No play-off or cup tie was on a league's board.
  fixturedownload v football-data.co.uk   every played match pairs, under
      the same match_id but one (MLS 2025, Austin v Seattle, two hours
      apart across UK midnight). Kickoffs: 2025 E1 550/552, N1 301/305, P1
      305/306, T1 300/304, USA 496/510; 2026 every one but T1 80/81 and USA
      463/466 - minutes (late starts), T1's two of 14/09/2025 football-data
      has an hour early, MLS's three in the hour the UK clock repeats
      (25/10/2026; soccer_data reads it as its first pass). Scores: every
      one, but three of MLS 2026 - Galaxy v St. Louis and LAFC v Salt Lake
      of 23/07/2026 (3-1 and 1-3 here, 1-3 and 3-1 there: swapped between
      the two in one source) and Orlando v Chicago of 20/08/2026 (1-0 here,
      1-2 there). Which source is right is not settled here; a
      fixturedownload score is never a result alone.
  ESPN v fixturedownload   the 174 matches both list (seven leagues): every
      kickoff equal, but two Ligue 2 matches ESPN had moved and
      fixturedownload still had on Saturday at 13:00 UTC (Reims v Sochaux to
      Monday, Montpellier v Metz to Friday) - Ligue 1's pattern of
      2026-10-27 again; no score contradicts.
  KNOCK-OUT PHASES   ESPN names a phase in the event's season slug; every
      play-off or cup-tie event the probe saved (usa.usl.1's 8 play-off
      quarter-finals, 115 cup ties) matches KNOCKOUT, no league event does.
      In a main division - football-data.co.uk leaves its play-offs out - an
      event in such a phase is skipped (the EFL play-offs, a relegation
      play-off): never in a sample, so how ESPN will label them is not
      measured. A split phase it names a "round" or a "play-off" (Belgium's
      play-offs, which football-data.co.uk keeps) would be skipped with
      them: football-data.co.uk alone there. The extra leagues' files keep
      their play-offs (MLS Cup, the liguilla, Argentina's knock-outs), so
      their events are read; a lower-league club in a relegation play-off
      has no mapping and is skipped, counted in last_run.unmapped.

THE SNAPSHOT. sync() fetches and writes soccer/data/feed.json: the source
rows (normalised, sorted, one per line, no NaN) and when each request last
succeeded. A request that fails, or returns something that is not the feed,
keeps its rows as they are on file (and an outage is cut short: a league's
other requests to a feed are skipped after STRAIGHT_FAILS failures in a
row, a feed's after FEED_DOWN leagues in a row failed, and none starts
after FETCH_BUDGET seconds); so does an ESPN day that
answers with no events while matches still to play are on file for it (one
glitchy empty answer must not unconfirm a whole day; a second in a row is
believed); a re-fetch that drops
a final result keeps it too (a played match does not leave the scoreboard).
Rows more than KEEP_BEHIND days old or KEEP_AHEAD days ahead are pruned.
Each row's seen_at is the moment its own request was answered. A snapshot
on file that cannot be read is rebuilt from what this fetch brings (said
in the log; every committed version is in git), or left as it is when no
request answered. last_run also lists the
clubs a feed named that the tables below do not map. load() reads it, so a
run without the network - and every test - works from the file.

    python soccer/engine/soccer_feed.py            # fetch, write, summarise
    python soccer/engine/soccer_feed.py --all      # ...every league (FEED_LEAGUES)
    python soccer/engine/soccer_feed.py --offline  # summarise the snapshot
"""

import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
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
# An outage must not hold the run up: after STRAIGHT_FAILS failures in a row
# from one feed for one league, that league's other requests to it are
# skipped (each up to a 30-second timeout - 80 of them would be 40 minutes);
# after FEED_DOWN leagues in a row cut short that way (or, for
# fixturedownload's one request a league, failed) with no answer from the
# feed in between, the feed is taken for down and the rest of its requests
# are skipped; and no request starts once the fetch has run FETCH_BUDGET
# seconds. Skipped requests are failed requests: what is on file for them is
# kept. With every league (FEED_LEAGUES: 465 ESPN days and 12 season files)
# a run is about 80 seconds - the league probe's 997 requests took 161 s on
# GitHub's runner on 2026-10-29 - and an outage of both feeds, timing out,
# still stops at 5 x 3 + 5 = 20 requests: the ten minutes the five leagues
# alone could take.
STRAIGHT_FAILS = 3
FEED_DOWN = 5
FETCH_BUDGET = 600

ESPN_BOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={day}"
FD_URL = "https://fixturedownload.com/feed/json/{slug}-{season}"
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
    # ---- every other league ESPN covers (league probe of 2026-10-29)
    "E1": {
        "392": ("Birmingham City", "Birmingham"), "365": ("Blackburn Rovers", "Blackburn"),
        "358": ("Bolton Wanderers", "Bolton"), "333": ("Bristol City", "Bristol City"), "379": ("Burnley", "Burnley"),
        "347": ("Cardiff City", "Cardiff"), "372": ("Charlton Athletic", "Charlton"),
        "374": ("Derby County", "Derby"), "314": ("Lincoln City", "Lincoln"),
        "369": ("Middlesbrough", "Middlesbrough"), "391": ("Millwall", "Millwall"),
        "381": ("Norwich City", "Norwich"), "385": ("Portsmouth", "Portsmouth"),
        "394": ("Preston North End", "Preston"), "334": ("Queens Park Rangers", "QPR"),
        "398": ("Sheffield United", "Sheffield United"), "376": ("Southampton", "Southampton"),
        "336": ("Stoke City", "Stoke"), "318": ("Swansea City", "Swansea"), "395": ("Watford", "Watford"),
        "383": ("West Bromwich Albion", "West Brom"), "371": ("West Ham United", "West Ham"),
        "380": ("Wolverhampton Wanderers", "Wolves"), "352": ("Wrexham", "Wrexham"),
    },
    "SP2": {
        "2737": ("Albacete", "Albacete"), "6832": ("Almería", "Almeria"), "12597": ("Burgos", "Burgos"),
        "4438": ("Castellón", "Castellon"), "11487": ("CD Sabadell", "Sabadell"), "5404": ("Ceuta", "Ceuta"),
        "3842": ("Cádiz", "Cadiz"), "8447": ("Córdoba", "Cordoba"), "3752": ("Eibar", "Eibar"),
        "7320": ("Eldense", "Eldense"), "20179": ("FC Andorra", "Andorra"), "9812": ("Girona", "Girona"),
        "3747": ("Granada", "Granada"), "98": ("Las Palmas", "Las Palmas"), "17534": ("Leganés", "Leganes"),
        "84": ("Mallorca", "Mallorca"), "131858": ("RC Celta Fortuna", "Celta B"), "92": ("Real Oviedo", "Oviedo"),
        "20983": ("Real Sociedad II", "Sociedad B"), "95": ("Real Valladolid", "Valladolid"),
        "3788": ("Sporting Gijón", "Sp Gijon"), "245": ("Tenerife", "Tenerife"),
    },
    "I2": {
        "3345": ("Arezzo", "Arezzo"), "3346": ("Ascoli", "Ascoli"), "4059": ("Benevento", "Benevento"),
        "3988": ("Carrarese", "Carrarese"), "3257": ("Catanzaro", "Catanzaro"), "3337": ("Cesena", "Cesena"),
        "4050": ("Cremonese", "Cremonese"), "2574": ("Empoli", "Empoli"), "119": ("Hellas Verona", "Verona"),
        "3975": ("Juve Stabia", "Juve Stabia"), "3991": ("Mantova", "Mantova"), "2573": ("Modena", "Modena"),
        "3952": ("Padova", "Padova"), "2923": ("Palermo", "Palermo"), "3956": ("Pisa", "Pisa"),
        "2734": ("Sampdoria", "Sampdoria"), "11139": ("Sudtirol", "Sudtirol"), "4055": ("US Avellino", "Avellino"),
        "120": ("Vicenza", "Vicenza"), "11137": ("Virtus Entella", "Virtus Entella"),
    },
    "D2": {
        "6418": ("1. FC Heidenheim 1846", "Heidenheim"), "10382": ("1. FC Magdeburg", "Magdeburg"),
        "269": ("1. FC Nürnberg", "Nurnberg"), "2506": ("Arminia Bielefeld", "Bielefeld"),
        "7017": ("Dynamo Dresden", "Dresden"), "123": ("Energie Cottbus", "Cottbus"),
        "2428": ("Hannover 96", "Hannover"), "129": ("Hertha Berlin", "Hertha"),
        "7884": ("Holstein Kiel", "Holstein Kiel"), "130": ("Kaiserslautern", "Kaiserslautern"),
        "4471": ("Karlsruher SC", "Karlsruhe"), "3070": ("SpVgg Greuther Fürth", "Greuther Furth"),
        "270": ("St. Pauli", "St Pauli"), "3812": ("SV Darmstadt 98", "Darmstadt"),
        "3067": ("TSV Eintracht Braunschweig", "Braunschweig"), "121": ("VfL Bochum", "Bochum"),
        "7013": ("VfL Osnabruck", "Osnabruck"), "138": ("VfL Wolfsburg", "Wolfsburg"),
    },
    "F2": {
        "18066": ("Annecy", "Annecy"), "3267": ("AS Nancy Lorraine", "Nancy"), "7869": ("Boulogne", "Boulogne"),
        "3171": ("Clermont Foot", "Clermont"), "3170": ("Dijon FCO", "Dijon"), "7732": ("Dunkerque", "Dunkerque"),
        "6994": ("Grenoble", "Grenoble"), "171": ("Guingamp", "Guingamp"), "177": ("Metz", "Metz"),
        "274": ("Montpellier", "Montpellier"), "165": ("Nantes", "Nantes"), "10678": ("Pau", "Pau FC"),
        "11884": ("Red Star FC 93", "Red Star"), "7719": ("Rodez Aveyron", "Rodez"),
        "178": ("Saint-Étienne", "St Etienne"), "272": ("Sochaux", "Sochaux"), "3243": ("Stade de Reims", "Reims"),
        "3266": ("Stade Laval", "Laval"),
    },
    "E2": {
        "3802": ("AFC Wimbledon", "AFC Wimbledon"), "397": ("Barnsley", "Barnsley"),
        "346": ("Blackpool", "Blackpool"), "387": ("Bradford City", "Bradford"), "3129": ("Bromley", "Bromley"),
        "2567": ("Burton Albion", "Burton"), "351": ("Cambridge United", "Cambridge"),
        "279": ("Doncaster Rovers", "Doncaster"), "335": ("Huddersfield Town", "Huddersfield"),
        "375": ("Leicester City", "Leicester"), "309": ("Leyton Orient", "Leyton Orient"),
        "301": ("Luton Town", "Luton"), "304": ("Mansfield Town", "Mansfield"),
        "390": ("Milton Keynes Dons", "Milton Keynes Dons"), "340": ("Notts County", "Notts County"),
        "311": ("Oxford United", "Oxford"), "342": ("Peterborough United", "Peterboro"),
        "307": ("Plymouth Argyle", "Plymouth"), "338": ("Reading", "Reading"),
        "399": ("Sheffield Wednesday", "Sheffield Weds"), "285": ("Stevenage", "Stevenage"),
        "400": ("Stockport County", "Stockport"), "350": ("Wigan Athletic", "Wigan"),
        "344": ("Wycombe Wanderers", "Wycombe"),
    },
    "E3": {
        "2731": ("Accrington Stanley", "Accrington"), "280": ("Barnet", "Barnet"),
        "308": ("Bristol Rovers", "Bristol Rvs"), "320": ("Cheltenham Town", "Cheltenham"),
        "343": ("Chesterfield", "Chesterfield"), "339": ("Colchester United", "Colchester"),
        "2594": ("Crawley Town", "Crawley Town"), "396": ("Crewe Alexandra", "Crewe"),
        "324": ("Exeter City", "Exeter"), "3891": ("Fleetwood Town", "Fleetwood Town"),
        "389": ("Gillingham", "Gillingham"), "386": ("Grimsby Town", "Grimsby"),
        "635": ("Newport County", "Newport County"), "353": ("Northampton Town", "Northampton"),
        "332": ("Oldham Athletic", "Oldham"), "348": ("Port Vale", "Port Vale"), "303": ("Rochdale", "Rochdale"),
        "402": ("Rotherham United", "Rotherham"), "4703": ("Salford City", "Salford"),
        "302": ("Shrewsbury Town", "Shrewsbury"), "341": ("Swindon Town", "Swindon"),
        "345": ("Tranmere Rovers", "Tranmere"), "401": ("Walsall", "Walsall"), "315": ("York City", "York"),
    },
    "EC": {
        "13884": ("AFC Fylde", "Fylde"), "10167": ("AFC Hornchurch", "Hornchurch"),
        "632": ("Aldershot Town", "Aldershot"), "633": ("Altrincham", "Altrincham"), "642": ("Barrow", "Barrow"),
        "2595": ("Boreham Wood", "Boreham Wood"), "3256": ("Boston United", "Boston Utd"),
        "322": ("Carlisle United", "Carlisle"), "3897": ("Eastleigh", "Eastleigh"),
        "312": ("FC Halifax Town", "Halifax"), "282": ("Forest Green Rovers", "Forest Green"),
        "3138": ("Gateshead", "Gateshead"), "19262": ("Harrogate Town", "Harrogate"),
        "323": ("Hartlepool United", "Hartlepool"), "319": ("Kidderminster Harriers", "Kidderminster"),
        "313": ("Scunthorpe United", "Scunthorpe"), "13061": ("Solihull Moors", "Solihull"),
        "310": ("Southend United", "Southend"), "3231": ("Sutton United", "Sutton"), "645": ("Tamworth", "Tamworth"),
        "3887": ("Wealdstone", "Wealdstone"), "290": ("Woking", "Woking"), "128663": ("Worthing", "Worthing"),
        "284": ("Yeovil Town", "Yeovil"),
    },
    "SC0": {
        "263": ("Aberdeen", "Aberdeen"), "256": ("Celtic", "Celtic"), "261": ("Dundee", "Dundee"),
        "264": ("Dundee United", "Dundee United"), "254": ("Falkirk", "Falkirk"),
        "262": ("Heart of Midlothian", "Hearts"), "258": ("Hibernian", "Hibernian"),
        "260": ("Kilmarnock", "Kilmarnock"), "266": ("Motherwell", "Motherwell"), "257": ("Rangers", "Rangers"),
        "267": ("St Johnstone", "St Johnstone"), "250": ("St Mirren", "St Mirren"),
    },
    "SC1": {
        "252": ("Arbroath", "Arbroath"), "248": ("Ayr United", "Ayr"), "265": ("Dunfermline Athletic", "Dunfermline"),
        "411": ("Greenock Morton", "Morton"), "253": ("Inverness Caledonian Thistle", "Inverness C"),
        "259": ("Livingston", "Livingston"), "247": ("Partick Thistle", "Partick"),
        "422": ("Queen's Park", "Queens Park"), "249": ("Raith Rovers", "Raith Rvs"),
        "405": ("Stenhousemuir", "Stenhousemuir"),
    },
    "N1": {
        "2726": ("ADO Den Haag", "Den Haag"), "139": ("Ajax Amsterdam", "Ajax"), "140": ("AZ Alkmaar", "AZ Alkmaar"),
        "2566": ("Excelsior", "Excelsior"), "145": ("FC Groningen", "Groningen"), "152": ("FC Twente", "Twente"),
        "153": ("FC Utrecht", "Utrecht"), "142": ("Feyenoord Rotterdam", "Feyenoord"),
        "143": ("Fortuna Sittard", "For Sittard"), "3706": ("Go Ahead Eagles", "Go Ahead Eagles"),
        "146": ("Heerenveen", "Heerenveen"), "147": ("NEC Nijmegen", "Nijmegen"), "2565": ("PEC Zwolle", "Zwolle"),
        "148": ("PSV Eindhoven", "PSV Eindhoven"), "3736": ("SC Cambuur", "Cambuur"),
        "151": ("Sparta Rotterdam", "Sparta Rotterdam"), "3735": ("Telstar", "Telstar"),
        "156": ("Willem II", "Willem II"),
    },
    "B1": {
        "441": ("Anderlecht", "Anderlecht"), "17544": ("Antwerp", "Antwerp"),
        "3610": ("Cercle Brugge KSV", "Cercle Brugge"), "570": ("Club Brugge", "Club Brugge"),
        "3611": ("KAA Gent", "Gent"), "5786": ("KV Kortrijk", "Kortrijk"), "7879": ("KV Mechelen", "Mechelen"),
        "606": ("KVC Westerlo", "Westerlo"), "22269": ("Lommel SK", "Lommel SK"),
        "5579": ("OH Leuven", "Oud-Heverlee Leuven"), "131235": ("RAAL La Louvière", "RAAL La Louviere"),
        "938": ("Racing Genk", "Genk"), "3616": ("Royal Charleroi SC", "Charleroi"),
        "936": ("Sint-Truidense", "St Truiden"), "559": ("Standard Liege", "Standard"),
        "5807": ("Union St.-Gilloise", "St. Gilloise"), "13450": ("Waasland-Beveren", "Beveren"),
        "4691": ("Zulte-Waregem", "Waregem"),
    },
    "P1": {
        "21607": ("Académico de Viseu", "Academico Viseu"), "21613": ("Alverca", "Alverca"),
        "15784": ("Arouca", "Arouca"), "1929": ("Benfica", "Benfica"), "2994": ("Braga", "Sp Braga"),
        "3472": ("C.D. Nacional", "Nacional"), "21581": ("Casa Pia", "Casa Pia"), "12216": ("Estoril", "Estoril"),
        "21610": ("Estrela", "Estrela"), "12698": ("FC Famalicao", "Famalicao"), "437": ("FC Porto", "Porto"),
        "3699": ("Gil Vicente", "Gil Vicente"), "552": ("Maritimo", "Maritimo"), "3696": ("Moreirense", "Moreirense"),
        "3822": ("Rio Ave", "Rio Ave"), "12215": ("Santa Clara", "Santa Clara"), "2250": ("Sporting CP", "Sp Lisbon"),
        "5309": ("Vitória de Guimaraes", "Guimaraes"),
    },
    "T1": {
        "9078": ("Alanyaspor", "Alanyaspor"), "132335": ("Amed SFK", "Amedspor"), "1895": ("Besiktas", "Besiktas"),
        "7656": ("Caykur Rizespor", "Rizespor"), "19267": ("Erzurum BB", "Erzurumspor"),
        "20729": ("Eyupspor", "Eyupspor"), "436": ("Fenerbahce", "Fenerbahce"), "432": ("Galatasaray", "Galatasaray"),
        "20070": ("Gaziantep FK", "Gaziantep"), "996": ("Genclerbirligi", "Genclerbirligi"),
        "789": ("Goztepe", "Goztep"), "7914": ("Istanbul Basaksehir", "Buyuksehyr"),
        "6870": ("Kasimpasa", "Kasimpasa"), "995": ("Kocaelispor", "Kocaelispor"), "7648": ("Konyaspor", "Konyaspor"),
        "11429": ("Samsunspor", "Samsunspor"), "997": ("Trabzonspor", "Trabzonspor"), "132334": ("Çorum FK", "Corum"),
    },
    "G1": {
        "887": ("AEK Athens", "AEK"), "11553": ("Aris", "Aris"), "8354": ("Asteras Tripoli", "Asteras Tripolis"),
        "6790": ("Atromitos", "Atromitos"), "2575": ("Iraklis", "Iraklis"), "132429": ("Kalamata", "Kalamata"),
        "21969": ("Kifisia", "Kifisia"), "5276": ("Levadiakos", "Levadeiakos"), "1010": ("OFI Crete", "OFI Crete"),
        "435": ("Olympiacos", "Olympiakos"), "443": ("Panathinaikos", "Panathinaikos"),
        "11431": ("Panetolikos", "Panetolikos"), "605": ("PAOK", "PAOK"), "20043": ("Volos NFC", "Volos NFC"),
    },
    "ARG": {
        "9739": ("Aldosivi", "Aldosivi"), "3": ("Argentinos Juniors", "Argentinos Jrs"),
        "9785": ("Atlético Tucumán", "Atl. Tucuman"), "235": ("Banfield", "Banfield"),
        "10060": ("Barracas Central", "Barracas Central"), "4": ("Belgrano (Córdoba)", "Belgrano"),
        "5": ("Boca Juniors", "Boca Juniors"), "11989": ("Central Córdoba (Santiago del Estero)", "Central Cordoba"),
        "8950": ("Defensa y Justicia", "Defensa y Justicia"), "17702": ("Deportivo Riestra", "Dep. Riestra"),
        "8": ("Estudiantes de La Plata", "Estudiantes L.P."),
        "19685": ("Estudiantes de Río Cuarto", "Estudiantes Rio Cuarto"),
        "11972": ("Gimnasia (Mendoza)", "Gimnasia Mendoza"), "9": ("Gimnasia La Plata", "Gimnasia L.P."),
        "10": ("Huracán", "Huracan"), "11": ("Independiente", "Independiente"),
        "9744": ("Independiente Rivadavia", "Ind. Rivadavia"), "2975": ("Instituto (Córdoba)", "Instituto"),
        "12": ("Lanús", "Lanus"), "14": ("Newell's Old Boys", "Newells Old Boys"), "7764": ("Platense", "Platense"),
        "15": ("Racing Club", "Racing Club"), "16": ("River Plate", "River Plate"),
        "17": ("Rosario Central", "Rosario Central"), "18": ("San Lorenzo", "San Lorenzo"),
        "10158": ("Sarmiento (Junín)", "Sarmiento Junin"), "19": ("Talleres (Córdoba)", "Talleres Cordoba"),
        "7767": ("Tigre", "Tigre"), "20": ("Unión (Santa Fe)", "Union de Santa Fe"),
        "21": ("Vélez Sarsfield", "Velez Sarsfield"),
    },
    "AUT": {
        "21540": ("Austria Lustenau", "A. Lustenau"), "1382": ("Austria Vienna", "Austria Vienna"),
        "21846": ("Grazer AK", "Grazer AK"), "4411": ("LASK Linz", "LASK"), "519": ("Rapid Vienna", "SK Rapid"),
        "2790": ("RB Salzburg", "Salzburg"), "4405": ("SC Rheindorf Altach", "Altach"),
        "3746": ("SK Sturm Graz", "Sturm Graz"), "3759": ("SV Josko Ried", "Ried"),
        "6907": ("TSV Hartberg", "Hartberg"), "13294": ("Wolfsberger", "Wolfsberger AC"),
        "18794": ("WSG Swarovski Tirol", "Tirol"),
    },
    "BRA": {
        "3458": ("Athletico-PR", "Athletico-PR"), "7632": ("Atlético-MG", "Atletico-MG"), "9967": ("Bahia", "Bahia"),
        "6086": ("Botafogo", "Botafogo RJ"), "9318": ("Chapecoense", "Chapecoense-SC"),
        "874": ("Corinthians", "Corinthians"), "3456": ("Coritiba", "Coritiba"), "2022": ("Cruzeiro", "Cruzeiro"),
        "819": ("Flamengo", "Flamengo RJ"), "3445": ("Fluminense", "Fluminense"), "6273": ("Grêmio", "Gremio"),
        "1936": ("Internacional", "Internacional"), "9169": ("Mirassol", "Mirassol"),
        "2029": ("Palmeiras", "Palmeiras"), "6079": ("Red Bull Bragantino", "Bragantino"), "4936": ("Remo", "Remo"),
        "2674": ("Santos", "Santos"), "2026": ("São Paulo", "Sao Paulo"), "3454": ("Vasco da Gama", "Vasco"),
        "3457": ("Vitória", "Vitoria"),
    },
    "CHN": {
        "2052": ("Beijing Guoan", "Beijing Guoan"), "21355": ("Chengdu Rongcheng", "Chengdu Rongcheng"),
        "131704": ("Chongqing Tonglianglong", "Chongqing Tonglianglong"), "22537": ("Dalian Yingbo", "Dalian Yingbo"),
        "8240": ("Henan", "Henan Songshan Longmen"), "131705": ("Liaoning Tieren", "Liaoning Tieren"),
        "21910": ("Qingdao Hainiu", "Qingdao Hainiu"), "22198": ("Qingdao West Coast", "Qingdao West Coast"),
        "7521": ("Shandong Taishan", "Shandong Taishan"), "15515": ("Shanghai Port FC", "Shanghai Port"),
        "977": ("Shanghai Shenhua FC", "Shanghai Shenhua"),
        "22199": ("Shenzhen Xinpengcheng", "Shenzhen Xinpengcheng"),
        "8239": ("Tianjin Jinmen Tiger", "Tianjin Jinmen Tiger"), "21506": ("Wuhan Three Towns", "Wuhan Three Towns"),
        "22536": ("Yunnan Yukun", "Yunnan Yukun"), "18203": ("Zhejiang Professional FC", "Zhejiang Professional"),
    },
    "DNK": {
        "3585": ("AC Horsens", "Horsens"), "7853": ("AGF", "Aarhus"), "575": ("Brøndby IF", "Brondby"),
        "909": ("F.C. København", "FC Copenhagen"), "572": ("FC Midtjylland", "Midtjylland"),
        "3101": ("FC Nordsjælland", "Nordsjaelland"), "7860": ("Lyngby Boldklub", "Lyngby"),
        "11550": ("Odense Boldklub", "Odense"), "3132": ("Randers FC", "Randers FC"),
        "607": ("Silkeborg IF", "Silkeborg"), "8118": ("Sønderjyske Fodbold", "Sonderjyske"),
        "3153": ("Viborg FF", "Viborg"),
    },
    "JPN": {
        "7107": ("Avispa Fukuoka", "Avispa Fukuoka"), "7109": ("Cerezo Osaka", "Cerezo Osaka"),
        "22522": ("Fagiano Okayama", "Okayama"), "3384": ("FC Tokyo", "FC Tokyo"),
        "7102": ("Gamba Osaka", "Gamba Osaka"), "7111": ("JEF United Ichihara-Chiba", "Chiba"),
        "7115": ("Kashima Antlers", "Kashima Antlers"), "7476": ("Kashiwa Reysol", "Kashiwa Reysol"),
        "7112": ("Kawasaki Frontale", "Kawasaki Frontale"), "21361": ("Kyoto Sanga", "Kyoto"),
        "22167": ("Machida Zelvia", "Machida"), "131701": ("Mito Hollyhock", "Mito"),
        "7108": ("Nagoya Grampus", "Nagoya Grampus"), "7114": ("Sanfrecce Hiroshima", "Sanfrecce Hiroshima"),
        "7104": ("Shimizu S-Pulse", "Shimizu S-Pulse"), "3393": ("Tokyo Verdy 1969", "Verdy"),
        "3385": ("Urawa Red Diamonds", "Urawa Reds"), "19001": ("V-Varen Nagasaki", "V-Varen Nagasaki"),
        "7477": ("Vissel Kobe", "Vissel Kobe"), "7116": ("Yokohama F. Marinos", "Yokohama F. Marinos"),
    },
    "MEX": {
        "227": ("América", "Club America"), "226": ("Atlante", "Atlante"), "216": ("Atlas", "Atlas"),
        "15720": ("Atlético de San Luis", "Atl. San Luis"), "218": ("Cruz Azul", "Cruz Azul"),
        "17851": ("FC Juárez", "Juarez"), "219": ("Guadalajara", "Guadalajara Chivas"), "228": ("León", "Club Leon"),
        "220": ("Monterrey", "Monterrey"), "229": ("Necaxa", "Necaxa"), "234": ("Pachuca", "Pachuca"),
        "231": ("Puebla", "Puebla"), "233": ("Pumas UNAM", "UNAM Pumas"), "222": ("Querétaro", "Queretaro"),
        "225": ("Santos", "Santos Laguna"), "232": ("Tigres UANL", "Tigres UANL"),
        "10125": ("Tijuana", "Club Tijuana"), "223": ("Toluca", "Toluca"),
    },
    "NOR": {
        "3278": ("Aalesund", "Aalesund"), "2980": ("Bodo/Glimt", "Bodo/Glimt"),
        "3039": ("Fredrikstad", "Fredrikstad"), "21380": ("Hamarkameratene", "HamKam"), "6750": ("IK Start", "Start"),
        "22165": ("KFUM Oslo", "KFUM Oslo"), "6672": ("Kristiansund BK", "Kristiansund"),
        "987": ("Lillestrom", "Lillestrom"), "2715": ("Molde", "Molde"), "438": ("Rosenborg", "Rosenborg"),
        "3279": ("Sandefjord", "Sandefjord"), "5002": ("Sarpsborg FK", "Sarpsborg 08"), "620": ("SK Brann", "Brann"),
        "5270": ("Tromso", "Tromso"), "510": ("Viking FK", "Viking"), "2791": ("Vålerenga", "Valerenga"),
    },
    "RUS": {
        "2991": ("Akhmat Grozny", "Akhmat Grozny"), "22271": ("Akron Tolyatti", "Akron Togliatti"),
        "1963": ("CSKA Moscow", "CSKA Moscow"), "596": ("Dinamo Moscow", "Dynamo Moscow"),
        "22300": ("Dynamo Makhachkala", "Dynamo Makhachkala"), "21539": ("Fakel Voronezh", "Fakel Voronezh"),
        "21949": ("FC Baltika Kaliningrad", "Baltika"), "18285": ("Gazovik Orenburg", "Orenburg"),
        "11336": ("Krasnodar", "Krasnodar"), "3850": ("Krylia Sovetov", "Krylya Sovetov"),
        "442": ("Lokomotiv Moscow", "Lokomotiv Moscow"), "21927": ("Rodina Moscow", "Rodina Moscow"),
        "3852": ("Rostov", "FK Rostov"), "3851": ("Rubin Kazan", "Rubin Kazan"),
        "1941": ("Spartak Moscow", "Spartak Moscow"), "2533": ("Zenit St Petersburg", "Zenit"),
    },
    "SWE": {
        "994": ("AIK", "AIK"), "7834": ("BK Häcken", "Hacken"), "20856": ("Degerfors IF", "Degerfors"),
        "2339": ("Djurgården", "Djurgarden"), "8222": ("GAIS", "GAIS"), "3017": ("Halmstads BK", "Halmstad"),
        "2495": ("Hammarby IF", "Hammarby"), "8221": ("IF Brommapojkarna", "Brommapojkarna"),
        "529": ("IF Elfsborg", "Elfsborg"), "2556": ("IFK Göteborg", "Goteborg"), "8547": ("IK Sirius", "Sirius"),
        "3052": ("Kalmar FF", "Kalmar"), "2720": ("Malmö FF", "Malmo FF"), "20301": ("Mjällby AIF", "Mjallby"),
        "22163": ("Västerås SK", "Vasteras SK"), "131552": ("Örgryte IS", "Orgryte"),
    },
    "USA": {
        "18418": ("Atlanta United FC", "Atlanta Utd"), "20906": ("Austin FC", "Austin FC"),
        "9720": ("CF Montréal", "CF Montreal"), "21300": ("Charlotte FC", "Charlotte"),
        "182": ("Chicago Fire FC", "Chicago Fire"), "184": ("Colorado Rapids", "Colorado Rapids"),
        "183": ("Columbus Crew", "Columbus Crew"), "193": ("D.C. United", "DC United"),
        "18267": ("FC Cincinnati", "FC Cincinnati"), "185": ("FC Dallas", "FC Dallas"),
        "6077": ("Houston Dynamo FC", "Houston Dynamo"), "20232": ("Inter Miami CF", "Inter Miami"),
        "187": ("LA Galaxy", "Los Angeles Galaxy"), "18966": ("LAFC", "Los Angeles FC"),
        "17362": ("Minnesota United FC", "Minnesota United"), "18986": ("Nashville SC", "Nashville SC"),
        "189": ("New England Revolution", "New England Revolution"), "17606": ("New York City FC", "New York City"),
        "12011": ("Orlando City SC", "Orlando City"), "10739": ("Philadelphia Union", "Philadelphia Union"),
        "9723": ("Portland Timbers", "Portland Timbers"), "4771": ("Real Salt Lake", "Real Salt Lake"),
        "190": ("Red Bull New York", "New York Red Bulls"), "22529": ("San Diego FC", "San Diego FC"),
        "191": ("San Jose Earthquakes", "San Jose Earthquakes"), "9726": ("Seattle Sounders FC", "Seattle Sounders"),
        "186": ("Sporting Kansas City", "Sporting Kansas City"), "21812": ("St. Louis CITY SC", "St. Louis City"),
        "7318": ("Toronto FC", "Toronto FC"), "9727": ("Vancouver Whitecaps", "Vancouver Whitecaps"),
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
    # ---- every other league fixturedownload covers (league probe of 2026-10-29)
    "E1": {
        "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
        "Bristol City": "Bristol City", "Burnley": "Burnley", "Cardiff City": "Cardiff",
        "Charlton Athletic": "Charlton", "Coventry City": "Coventry", "Derby County": "Derby", "Hull City": "Hull",
        "Ipswich Town": "Ipswich", "Leicester City": "Leicester", "Lincoln City": "Lincoln",
        "Middlesbrough": "Middlesbrough", "Millwall": "Millwall", "Norwich City": "Norwich",
        "Oxford United": "Oxford", "Portsmouth": "Portsmouth", "Preston North End": "Preston",
        "Queens Park Rangers": "QPR", "Sheffield United": "Sheffield United", "Sheffield Wednesday": "Sheffield Weds",
        "Southampton": "Southampton", "Stoke City": "Stoke", "Swansea City": "Swansea", "Watford": "Watford",
        "West Bromwich Albion": "West Brom", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
        "Wrexham": "Wrexham",
    },
    "F2": {
        "AS Nancy Lorraine": "Nancy", "AS Saint-Étienne": "St Etienne", "Clermont Foot 63": "Clermont",
        "Dijon FCO": "Dijon", "EN Avant Guingamp": "Guingamp", "FC Annecy": "Annecy", "FC Metz": "Metz",
        "FC Nantes": "Nantes", "FC Sochaux-Montbéliard": "Sochaux", "Grenoble Foot 38": "Grenoble",
        "Montpellier Hérault SC": "Montpellier", "Pau FC": "Pau FC", "Red Star FC": "Red Star",
        "Rodez Aveyron Football": "Rodez", "Stade DE Reims": "Reims", "Stade Lavallois MFC": "Laval",
        "US Boulogne CO": "Boulogne", "USL Dunkerque": "Dunkerque",
    },
    "SC0": {
        "Aberdeen": "Aberdeen", "Celtic": "Celtic", "Dundee": "Dundee", "Dundee United": "Dundee United",
        "Falkirk": "Falkirk", "Heart of Midlothian": "Hearts", "Hibernian": "Hibernian", "Kilmarnock": "Kilmarnock",
        "Motherwell": "Motherwell", "Rangers": "Rangers", "St. Johnstone": "St Johnstone", "St. Mirren": "St Mirren",
    },
    "N1": {
        "ADO Den Haag": "Den Haag", "Ajax": "Ajax", "AZ": "AZ Alkmaar", "Excelsior Rotterdam": "Excelsior",
        "FC Groningen": "Groningen", "FC Twente": "Twente", "FC Utrecht": "Utrecht", "FC Volendam": "Volendam",
        "Feyenoord": "Feyenoord", "Fortuna Sittard": "For Sittard", "Go Ahead Eagles": "Go Ahead Eagles",
        "Heracles Almelo": "Heracles", "N.E.C. Nijmegen": "Nijmegen", "NAC Breda": "NAC Breda",
        "PEC Zwolle": "Zwolle", "PSV": "PSV Eindhoven", "SC Cambuur": "Cambuur", "sc Heerenveen": "Heerenveen",
        "Sparta Rotterdam": "Sparta Rotterdam", "Telstar": "Telstar", "Willem II": "Willem II",
    },
    "P1": {
        "Académico": "Academico Viseu", "AFS": "AVS", "Casa Pia AC": "Casa Pia", "CD Nacional": "Nacional",
        "CD Tondela": "Tondela", "Estoril Praia": "Estoril", "Estrela Amadora": "Estrela", "FC Alverca": "Alverca",
        "FC Arouca": "Arouca", "FC Famalicão": "Famalicao", "FC Porto": "Porto", "Gil Vicente FC": "Gil Vicente",
        "Marítimo M.": "Maritimo", "Moreirense FC": "Moreirense", "Rio Ave FC": "Rio Ave",
        "Santa Clara": "Santa Clara", "SC Braga": "Sp Braga", "SL Benfica": "Benfica", "Sporting CP": "Sp Lisbon",
        "Vitória SC": "Guimaraes",
    },
    "T1": {
        "Alanyaspor": "Alanyaspor", "Amedspor": "Amedspor", "Antalyaspor": "Antalyaspor", "Besiktas": "Besiktas",
        "Erzurumspor": "Erzurumspor", "Eyüpspor": "Eyupspor", "Fatih Karagümrük": "Karagumruk",
        "Fenerbahçe": "Fenerbahce", "Galatasaray": "Galatasaray", "Gaziantep": "Gaziantep",
        "Gençlerbirligi": "Genclerbirligi", "Göztepe": "Goztep", "Istanbul Basaksehir": "Buyuksehyr",
        "Kasimpasa": "Kasimpasa", "Kayserispor": "Kayserispor", "Kocaelispor": "Kocaelispor",
        "Konyaspor": "Konyaspor", "Samsunspor": "Samsunspor", "Trabzonspor": "Trabzonspor",
        "Çaykur Rizespor": "Rizespor", "Çorum": "Corum",
    },
    "USA": {
        "Atlanta": "Atlanta Utd", "Atlanta United": "Atlanta Utd", "Austin": "Austin FC", "Austin FC": "Austin FC",
        "CF Montréal": "CF Montreal", "Charlotte": "Charlotte", "Charlotte FC": "Charlotte",
        "Chicago": "Chicago Fire", "Chicago Fire FC": "Chicago Fire", "Cincinnati": "FC Cincinnati",
        "Colorado": "Colorado Rapids", "Colorado Rapids": "Colorado Rapids", "Columbus": "Columbus Crew",
        "Columbus Crew": "Columbus Crew", "D.C.": "DC United", "D.C. United": "DC United", "Dallas": "FC Dallas",
        "FC Cincinnati": "FC Cincinnati", "FC Dallas": "FC Dallas", "Houston": "Houston Dynamo",
        "Houston Dynamo FC": "Houston Dynamo", "Inter Miami CF": "Inter Miami", "Kansas City": "Sporting Kansas City",
        "LA": "Los Angeles Galaxy", "LA Galaxy": "Los Angeles Galaxy", "LAFC": "Los Angeles FC",
        "Los Angeles Football Club": "Los Angeles FC", "Miami": "Inter Miami", "Minnesota": "Minnesota United",
        "Minnesota United FC": "Minnesota United", "Montréal": "CF Montreal", "Nashville": "Nashville SC",
        "Nashville SC": "Nashville SC", "New England": "New England Revolution",
        "New England Revolution": "New England Revolution", "New York": "New York Red Bulls",
        "New York City": "New York City", "New York City Football Club": "New York City", "Orlando": "Orlando City",
        "Orlando City": "Orlando City", "Philadelphia": "Philadelphia Union",
        "Philadelphia Union": "Philadelphia Union", "Portland": "Portland Timbers",
        "Portland Timbers": "Portland Timbers", "Real Salt Lake": "Real Salt Lake",
        "Red Bull New York": "New York Red Bulls", "Salt Lake": "Real Salt Lake", "San Diego": "San Diego FC",
        "San Diego FC": "San Diego FC", "San Jose": "San Jose Earthquakes",
        "San Jose Earthquakes": "San Jose Earthquakes", "Seattle": "Seattle Sounders",
        "Seattle Sounders FC": "Seattle Sounders", "Sporting Kansas City": "Sporting Kansas City",
        "St. Louis": "St. Louis City", "St. Louis CITY SC": "St. Louis City", "Toronto": "Toronto FC",
        "Toronto FC": "Toronto FC", "Vancouver": "Vancouver Whitecaps",
        "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    },
}

# ------------------------------------------------------------- the leagues asked about

# ESPN's league code for each league with a club table: soccer_data's
# registry (ALL_LEAGUES[code].espn).
ESPN_SLUGS = {code: sd.ALL_LEAGUES[code].espn for code in ESPN_CLUBS}
# Leagues the registry has an ESPN code for that are not asked about: on the
# probe of 2026-10-29 ESPN had no current season for them (football-data.co.uk
# alone). POL has no ESPN code at all (pol.1 answered 404).
ESPN_NOT_ASKED = {
    "SC2": "sco.3 listed 2025-26's clubs and no events Oct 26 - Nov 8",
    "SC3": "sco.4 listed 2025-26's clubs and no events Oct 26 - Nov 8",
    "ROU": "rou.1 listed 2025-26's clubs and no events Oct 26 - Nov 8",
    "SWZ": "sui.1 listed 2025-26's clubs and no events Oct 26 - Nov 8",
    "FIN": "fin.1 listed no clubs and no events",
    "IRL": "irl.1 listed no clubs and no events",
}
# fixturedownload.com's feed name per league: the slugs that answered on the
# probe (2026-10-29; of 28 tried, the rest 404). Its "season" is the year the
# season starts, a calendar year for MLS - football-data.co.uk's convention.
FD_SLUGS = {"E0": "epl", "SP1": "la-liga", "I1": "serie-a", "D1": "bundesliga", "F1": "ligue-1",
            "E1": "championship", "F2": "ligue-2", "SC0": "scottish-premiership", "N1": "eredivisie",
            "P1": "primeira-liga", "T1": "super-lig", "USA": "mls"}
# The zone of fixturedownload's placeholder clock per league (is_placeholder):
# midnight UTC or midnight here. None: no placeholder rule - MLS kicks off at
# 00:00 UTC for real (20:00 Eastern summer time: 4 played matches in 2025,
# 5 in 2026 and 8 still to play), and its feed had no unscheduled round to
# learn one from.
LOCAL = {"E0": ZoneInfo("Europe/London"), "SP1": ZoneInfo("Europe/Madrid"), "I1": ZoneInfo("Europe/Rome"),
         "D1": ZoneInfo("Europe/Berlin"), "F1": ZoneInfo("Europe/Paris"), "E1": ZoneInfo("Europe/London"),
         "F2": ZoneInfo("Europe/Paris"), "SC0": ZoneInfo("Europe/London"), "N1": ZoneInfo("Europe/Amsterdam"),
         "P1": ZoneInfo("Europe/Lisbon"), "T1": ZoneInfo("Europe/Istanbul"), "USA": None}
# Every league either feed covers, in the registry's order (the five first,
# so a fetch cut short by its budget loses the others first) - and the five
# a sync asks about unless told otherwise: the leagues the app predicts today
# (soccer_schedule joins only those). Switching the app to more leagues is
# soccer_predict's sync call with leagues=FEED_LEAGUES.
FEED_LEAGUES = tuple(code for code in sd.ALL_LEAGUES if code in ESPN_SLUGS or code in FD_SLUGS)
LIVE_LEAGUES = tuple(sd.TOP)

# Leagues whose format plays one pairing (home v away) more than once a
# season - a split into groups, a third or fourth round robin, play-offs or
# two tournaments in one season. Every league where football-data.co.uk has a
# pairing twice in a season of 2021-2025 (none of the others has one): there
# reconcile pairs ESPN's and fixturedownload's rows of a pairing by date, not
# by the pairing alone (RECONCILING in the module's notes).
REPEATS = frozenset({"SC0", "SC1", "SC2", "SC3", "B1", "G1", "ARG", "AUT", "DNK", "FIN", "IRL", "MEX", "ROU",
                     "SWZ", "USA"})
# In those leagues an ESPN event and a fixturedownload row of one pairing are
# one match when their dates are at most this far apart (nearest first): half
# the shortest gap between a pairing's two meetings there, USA's 7 days.
PAIR_DAYS = 3
# ESPN's season slug for a knock-out phase: in the probe, every play-off and
# cup-tie event (usa.usl.1's 8 "...-conference-playoffs---quarterfinals", the
# 115 of eng.fa, eng.league_cup, esp.copa_del_rey and ger.dfb_pokal:
# "first-round", "second-round", "fourth-round") matched it and no league
# event (all 661 on the 31 leagues' boards) did. football-data.co.uk leaves
# play-offs out of its main divisions, so there an event in such a phase is
# skipped; the extra leagues' files keep theirs (MLS Cup, the liguilla, ...),
# and their events are read.
KNOCKOUT = re.compile(r"play-?offs?|(^|-)round(-|$)")


class FeedError(ValueError):
    """A response that is not the feed it should be (an error page, a format change)."""


def _nfc(text):
    return unicodedata.normalize("NFC", str(text or "")).strip()


def espn_club(league, team, log=print, missing=None):
    """ESPN's team object -> football-data.co.uk's name, or None (warned, and
    added to `missing` where one is given)."""
    team = team or {}
    tid, name = str(team.get("id") or ""), _nfc(team.get("displayName"))
    known = ESPN_CLUBS.get(league, {}).get(tid)
    if known is None:
        log(f"   espn {league}: no club mapping for ESPN team {tid or '?'} '{name}'; its match is skipped")
        if missing is not None:
            missing.add(f"espn:{league}:{tid or '?'} {name}")
        return None
    if name and _nfc(known[0]) != name:
        log(f"   espn {league}: ESPN team {tid} is now called '{name}' (table: '{known[0]}'); mapped by its id")
    return known[1]


def fd_club(league, name, log=print, missing=None):
    """fixturedownload's name -> football-data.co.uk's, or None (warned, and
    added to `missing` where one is given)."""
    key = _nfc(name)
    club = FD_CLUBS.get(league, {}).get(key)
    if club is None:
        log(f"   fixturedownload {league}: no club mapping for '{key}'; its match is skipped")
        if missing is not None:
            missing.add(f"fixturedownload:{league}:{key}")
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


def parse_espn(payload, league, request=None, seen_at=None, log=print, missing=None):
    """One ESPN scoreboard response -> rows. Raises FeedError for a body that
    is not a scoreboard; an event it cannot read is skipped and warned (a
    club with no mapping is also added to `missing`)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise FeedError("not an ESPN scoreboard (no events list)")
    rows = []
    for event in payload["events"]:
        try:
            row = _espn_event(event, league, request, seen_at, log, missing)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as err:
            eid = event.get("id") if isinstance(event, dict) else None
            log(f"   espn {league}: event {eid or '?'} unreadable ({err.__class__.__name__}: {err}); skipped")
            continue
        if row is not None:
            rows.append(row)
    return rows


def knockout_phase(event, league):
    """ESPN's season slug for the event where it names a knock-out phase (a
    play-off, a cup round: KNOCKOUT) in a division football-data.co.uk files
    without its play-offs (its main divisions); else None."""
    meta = sd.ALL_LEAGUES.get(league)
    phase = str((event.get("season") or {}).get("slug") or "")
    return phase if meta is not None and meta.kind == "main" and KNOCKOUT.search(phase) else None


def _espn_event(event, league, request, seen_at, log, missing=None):
    phase = knockout_phase(event, league)
    if phase:
        log(f"   espn {league}: event {event.get('id')} is in ESPN's '{phase}' phase, a play-off or cup tie - "
            f"football-data.co.uk leaves those out of this division; skipped")
        return None
    comp = (event.get("competitions") or [{}])[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
    if "home" not in sides or "away" not in sides:
        log(f"   espn {league}: event {event.get('id')} has no home and away side; skipped")
        return None
    home = espn_club(league, sides["home"].get("team"), log, missing)
    away = espn_club(league, sides["away"].get("team"), log, missing)
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
    season = sd.season_for(league, day.isoformat())         # football-data.co.uk's convention for the league
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
    Liga, Serie A, the Bundesliga, the Eredivisie, the Primeira Liga) or
    midnight in the league's own time zone (Ligue 1, Ligue 2, the Primeira
    Liga in summer time, the Super Lig). No league with a zone in LOCAL
    kicks off at either; a league with none (MLS) has no placeholder rule."""
    zone = LOCAL.get(league)
    if zone is None:
        return False
    local = moment.astimezone(zone)
    return (moment.hour, moment.minute) == (0, 0) or (local.hour, local.minute) == (0, 0)


def parse_fixturedownload(rows, league, season, request=None, seen_at=None, log=print, missing=None):
    """One fixturedownload season feed -> rows. Raises FeedError for a body
    that is not one; a row it cannot read is skipped and warned (a club with
    no mapping is also added to `missing`)."""
    if not isinstance(rows, list):
        raise FeedError("not a fixturedownload feed (not a list)")
    out, pairs = [], defaultdict(list)
    for raw in rows:
        try:
            row = _fd_row(raw, league, season, request, seen_at, log, missing)
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            log(f"   fixturedownload {league}: row unreadable ({err.__class__.__name__}: {err}); skipped")
            continue
        if row is not None:
            out.append(row)
            pairs[(row["home"], row["away"])].append(row["source_id"])
    for (home, away), ids in pairs.items():
        if len(ids) > 1 and league not in REPEATS:          # where pairings repeat, two rows are two matches
            log(f"   fixturedownload {league} {season}: {home} v {away} listed {len(ids)} times ({', '.join(ids)}); "
                f"a pairing is played once a season - the feed has home and away swapped somewhere")
    return out


def _fd_row(raw, league, season, request, seen_at, log, missing=None):
    home, away = fd_club(league, raw["HomeTeam"], log, missing), fd_club(league, raw["AwayTeam"], log, missing)
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
    if sd.season_for(league, hint.isoformat()) != int(season):
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


def requests_for(now, days_back=4, days_ahead=10, leagues=LIVE_LEAGUES):
    """[(request key, url, league, kind, season)] for one fetch: ESPN's days
    for a league in ESPN_SLUGS, fixturedownload's season file for one in
    FD_SLUGS. The season is the league's own (soccer_data.season_for: a
    calendar year for MLS, Brazil, ...; the year it starts otherwise)."""
    today = _aware(now).date()
    out = []
    for league in leagues:
        season = sd.season_for(league, today.isoformat())
        if league in ESPN_SLUGS:
            for day in espn_days(now, days_back, days_ahead):
                out.append((f"espn:{league}:{day:%Y%m%d}",
                            ESPN_BOARD.format(slug=ESPN_SLUGS[league], day=f"{day:%Y%m%d}"), league, "espn", season))
        if league in FD_SLUGS:
            out.append((f"fixturedownload:{league}:{season}", FD_URL.format(slug=FD_SLUGS[league], season=season),
                        league, "fixturedownload", season))
    return out


def fetch(now=None, days_back=4, days_ahead=10, get=_get, leagues=LIVE_LEAGUES, log=print,
          clock=time.monotonic, budget=FETCH_BUDGET, wall=None):
    """ESPN's scoreboard one league and UTC day at a time from days_back
    before now to days_ahead after, plus fixturedownload's current season
    per league. ESPN files an event under its US Eastern date: on the league
    probe, which of 47 leagues' boards answered empty fits that on all 658
    days (the Pacific date on 653, the UTC date on 642) - so a European
    match is on its UTC day, and an MLS evening or an Asian morning kickoff
    can be on the day before; the window has days to spare either side.

    Returns {"at", "rows", "ok": {request: rows}, "failed": {request: why},
    "unmapped": [clubs a feed named that the tables do not map]}. A request
    that fails is only reported: merge() keeps its rows on file.
    STRAIGHT_FAILS, FEED_DOWN and `budget` (seconds) bound how long an
    outage can take; the requests they skip are reported as failed.

    seen_at: with `wall` (a clock: a real run, sync) each row is stamped
    with the moment its own request was answered, never earlier; without it
    (a replay, a test) with `now`."""
    at = stamp(_aware(now))
    rows, ok, failed, missing = [], {}, {}, set()
    start, streak, skipping = clock(), defaultdict(int), set()
    asked = requests_for(now, days_back, days_ahead, leagues)
    per_league = Counter((kind, league) for _, _, league, kind, _ in asked)
    down = defaultdict(int)                  # feed -> leagues in a row it failed for, with no answer between
    for key, url, league, kind, season in asked:
        if streak[(kind, league)] >= STRAIGHT_FAILS:
            failed[key] = f"skipped: {STRAIGHT_FAILS} {kind} requests for {league} failed in a row"
            if (kind, league) not in skipping:
                skipping.add((kind, league))
                log(f"   {kind} {league}: {STRAIGHT_FAILS} failures in a row; its other requests skipped, "
                    f"kept what is on file")
            continue
        if down[kind] >= FEED_DOWN:
            failed[key] = f"skipped: {kind} failed for {FEED_DOWN} leagues in a row"
            if kind not in skipping:
                skipping.add(kind)
                log(f"   {kind}: failed for {FEED_DOWN} leagues in a row, taken for down; its other requests "
                    f"skipped, kept what is on file")
            continue
        if clock() - start > budget:
            failed[key] = f"skipped: the fetch ran past {budget} s"
            if "budget" not in skipping:
                skipping.add("budget")
                log(f"   feeds: past {budget} s; the requests left are skipped, kept what is on file")
            continue
        try:
            payload = _decode(get(url))
            seen = stamp(_aware(wall())) if wall is not None else at
            if kind == "espn":
                got = parse_espn(payload, league, request=key, seen_at=seen, log=log, missing=missing)
            else:
                got = parse_fixturedownload(payload, league, season, request=key, seen_at=seen, log=log,
                                            missing=missing)
                if not got:
                    raise FeedError("no rows parsed (format change?)")
        except Exception as err:                       # noqa: BLE001 - reported, kept on file, run continues
            failed[key] = f"{err.__class__.__name__}: {str(err)[:120]}"
            log(f"   {key}: FAILED ({failed[key]}); kept what is on file")
            streak[(kind, league)] += 1
            if streak[(kind, league)] == min(STRAIGHT_FAILS, per_league[(kind, league)]):
                down[kind] += 1                  # this league's requests to the feed stop here
            continue
        streak[(kind, league)] = down[kind] = 0
        ok[key] = len(got)
        rows.extend(got)
    log(f"   feeds: {len(ok)} requests answered, {len(failed)} failed, {len(rows)} rows"
        + (f"; {len(missing)} clubs with no mapping" if missing else ""))
    return {"at": at, "rows": rows, "ok": ok, "failed": failed, "unmapped": sorted(missing)}


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


EMPTY_BOARD = "empty board"


def distrust_empty(old_rows, fetched, log=print, previous=None):
    """An ESPN day answered with no events while matches still to play are on
    file for it, and no answered request lists them elsewhere (a moved match
    moves with its id): one glitchy answer must not unconfirm a whole day, so
    it is treated as a failed request and merge() keeps those rows - unless
    the run before had the same empty answer for it (`previous`: that run's
    last_run["failed"]), when the day really is empty and the rows go.
    Returns `fetched`, changed in place."""
    listed = {row_key(r) for r in fetched.get("rows", [])}
    previous = previous or {}
    for key, n in list(fetched.get("ok", {}).items()):
        if n or not key.startswith("espn:"):
            continue
        held = [r for r in old_rows if r.get("request") == key and r.get("status") not in ("final",)
                and row_key(r) not in listed]
        if not held:
            continue
        if str(previous.get(key, "")).startswith(EMPTY_BOARD):
            log(f"   {key}: answered with no events twice in a row; its {len(held)} matches still to play "
                f"on file are dropped")
            continue
        del fetched["ok"][key]
        fetched.setdefault("failed", {})[key] = (f"{EMPTY_BOARD} with {len(held)} matches on file still to "
                                                 f"play; kept until a second empty answer")
        log(f"   {key}: answered with no events, but {len(held)} matches on file for it are still to play; "
            f"treated as failed, kept what is on file")
    return fetched


def sync(now=None, path=FEED, get=_get, days_back=4, days_ahead=10, leagues=LIVE_LEAGUES, log=print,
         wall=None):
    """Fetch both feeds, merge into the snapshot on file, write it. Returns
    the snapshot; its last_run says what answered and what failed, and which
    clubs a feed named that the tables do not map. `now` None is a real run:
    each row is stamped when its own request was answered (fetch's `wall`).

    A snapshot on file that cannot be read is rebuilt from what this fetch
    brings, said so in the log (every committed version is in git) - and,
    where nothing answered, left as it is: there is nothing to rebuild it
    from, and the run goes on with football-data.co.uk alone."""
    if now is None and wall is None:
        wall = lambda: datetime.now(timezone.utc)          # noqa: E731 - a real run: each row its own moment
    now = _aware(now)
    unreadable = None
    try:
        snap = load(path)
    except Exception as err:                       # noqa: BLE001 - logged; rebuilt from this fetch
        unreadable = f"{err.__class__.__name__}: {str(err)[:120]}"
        snap = empty()
    got = fetch(now, days_back, days_ahead, get=get, leagues=leagues, log=log, wall=wall)
    if unreadable is not None:
        if not got["ok"]:
            log(f"   feeds: the snapshot on file is unreadable ({unreadable}) and no request answered; "
                f"left as it is")
            return snap
        log(f"   feeds: the snapshot on file was unreadable ({unreadable}); rebuilt from this fetch")
    distrust_empty(snap.get("rows", []), got, log=log, previous=(snap.get("last_run") or {}).get("failed"))
    rows = merge(snap.get("rows", []), got, now, log=log)
    requests = dict(snap.get("requests") or {})
    requests.update({k: got["at"] for k in got["ok"]})
    on_file = {r["request"] for r in rows}
    oldest = (now - timedelta(days=KEEP_BEHIND)).strftime("%Y%m%d")
    requests = {k: v for k, v in requests.items()
                if k in on_file or k in got["ok"] or (k.startswith("espn:") and k.rsplit(":", 1)[1] >= oldest)}
    snap = {"schema": SCHEMA, "requests": requests, "rows": rows,
            "last_run": {"at": got["at"], "answered": len(got["ok"]), "failed": dict(sorted(got["failed"].items())),
                         "unmapped": got.get("unmapped", [])}}
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


def _day(row):
    hint = row.get("date_hint")
    return date.fromisoformat(hint) if hint else None


def legs(espn_rows, fd_rows):
    """One pairing's rows in a league where pairings repeat (REPEATS) -> one
    ([ESPN row], [fixturedownload row]) per match: every ESPN event is a
    match, and a fixturedownload row is the ESPN event's nearest in date
    within PAIR_DAYS (closest pairs first, each row once) or a match of its
    own."""
    gaps = sorted((abs((_day(e) - _day(f)).days), i, j) for i, e in enumerate(espn_rows)
                  for j, f in enumerate(fd_rows) if _day(e) and _day(f) and abs((_day(e) - _day(f)).days) <= PAIR_DAYS)
    used_e, used_f, out = set(), set(), []
    for _, i, j in gaps:
        if i in used_e or j in used_f:
            continue
        used_e.add(i)
        used_f.add(j)
        out.append(([espn_rows[i]], [fd_rows[j]]))
    out += [([e], []) for i, e in enumerate(espn_rows) if i not in used_e]
    out += [([], [f]) for j, f in enumerate(fd_rows) if j not in used_f]
    return out


def reconcile(espn_rows, fd_rows, log=print):
    """One row per match from both feeds' rows: per pairing (league, season,
    home, away) - a pairing is played once a season - except in a league
    where pairings repeat (REPEATS), where each of a pairing's matches is
    one row (legs: by date).

    Each row: match_id, league, season, home, away, uk_date, date_hint,
    kickoff_utc (ESPN's valid time, else fixturedownload's: the match's
    date, id and shown time; None without one, or for a postponed or
    cancelled match), kickoff_first (the earlier of the two: no pick stays
    open past it), kickoff_espn, kickoff_fd, time_confirmed, status
    (MATCH_STATUSES), status_espn,
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
        es, fs = groups[key]
        for e_rows, f_rows in (legs(es, fs) if key[0] in REPEATS else [(es, fs)]):
            row = _reconcile_one(key, e_rows, f_rows)
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
    # The match's time - its date, its id, the time shown - is ESPN's valid
    # one where there is one (ESPN tracks a move; fixturedownload's later
    # rounds were measured stale), else fixturedownload's. The earlier of the
    # two still closes a pick (kickoff_first; soccer_schedule).
    when = ke or kf
    first = times[0] if times else None
    if ke and kf and ke != kf:
        notes.append(f"kickoff differs: ESPN {ke}, fixturedownload {kf}; ESPN's is the match's, "
                     f"the earlier closes a pick")
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
        "kickoff_first": None if status in ("postponed", "cancelled") else first,
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
        sync(leagues=FEED_LEAGUES if "--all" in sys.argv else LIVE_LEAGUES)
    summary(reconciled(log=lambda *_: None))
