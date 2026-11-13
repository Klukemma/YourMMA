"""The fixture feeds for every league beyond the top five (soccer_feed.py), on
the league probe's real responses (tests/fixtures/feeds/leagues/,
probe_leagues.py, GitHub's machines 2026-10-29 12:43 UTC): the club tables,
the leagues asked about and why the others are not, the season convention,
fixturedownload's placeholders, pairings that repeat, knock-out phases, and
every match a feed and football-data.co.uk both have.

football-data.co.uk is read as the same probe downloaded it (its CSVs in the
same folder) - and, for the five second tiers the probe did not download,
from soccer/data's snapshot frozen on 2026-10-29
(fixtures/football_data_second_tiers_2025_to_20261028.csv.gz) - so a gate
test checks the code, never a correction posted later. A test that pins what
the samples said about matches still to be played skips if the samples are
re-probed."""

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import cache
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import soccer_data as sd
import soccer_feed as sf

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROBE = FIXTURES / "feeds" / "leagues"
REPORT = json.loads((PROBE / "probe_report.json").read_text())
PROBED = REPORT["probed"]
NOW = datetime(2026, 10, 29, 12, 42, 59, tzinfo=timezone.utc)
AS_PROBED = pytest.mark.skipif(not PROBED.startswith("2026-10-29"),
                               reason="samples re-probed: the 2026-10-29 evidence no longer applies")
NEW = [code for code in sf.FEED_LEAGUES if code not in sd.TOP]
SECOND_TIERS = ["E1", "SP2", "I2", "D2", "F2"]
NEW_MAIN = ["E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "N1", "B1", "P1", "T1", "G1"]
CUPS = {"champions-league", "europa-league", "conference-league"}       # fixturedownload, not leagues
EASTERN = ZoneInfo("America/New_York")


def raw(name):
    return gzip.decompress((PROBE / f"{name}.gz").read_bytes())


def board_events(code):
    return json.loads(raw(f"espn_{sf.ESPN_SLUGS[code]}_boards.json"))


def teams(slug):
    body = json.loads(raw(f"espn_{slug}_teams.json"))
    return body["sports"][0]["leagues"][0] if body.get("sports") else {}


def gz_json(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def fxd(code, year):
    path = PROBE / f"fxd_{sf.FD_SLUGS[code]}_{year}.json.gz"
    return gz_json(path) if path.exists() else None


@cache
def football_data():
    """(results, fixtures) of every league outside the ten as the probe
    downloaded them, plus the five second tiers' frozen results."""
    parts = [sd.frame(sd.parse(raw(f"fd_{div}_{code}.csv"), int("20" + code[:2]), leagues=[div]))
             for div in NEW_MAIN for code in ("2526", "2627")]
    parts += [sd.frame(sd.parse_extra(raw(f"fd_new_{code}.csv"), league=code)) for code in sd.EXTRA]
    tiers = pd.read_csv(FIXTURES / "football_data_second_tiers_2025_to_20261028.csv.gz")
    tiers["kickoff"] = pd.to_datetime(tiers["kickoff"])
    results = pd.concat(parts + [tiers], ignore_index=True)
    fixtures = pd.concat([sd.frame(sd.parse(raw("fd_fixtures.csv"), leagues=list(sd.ALL_LEAGUES))),
                          sd.frame(sd.parse_extra(raw("fd_new_league_fixtures.csv")))], ignore_index=True)
    return results, fixtures


def clubs_of(code, season):
    results, fixtures = football_data()
    rows = pd.concat([results[(results["league"] == code) & (results["season"] == season)],
                      fixtures[(fixtures["league"] == code) & (fixtures["season"] == season)]])
    return set(rows["home"]) | set(rows["away"])


@cache
def espn_rows():
    warnings, rows = [], []
    for code in NEW:
        if code in sf.ESPN_SLUGS:
            rows += sf.parse_espn({"events": board_events(code)}, code, request=f"espn:{code}:probe",
                                  seen_at=PROBED, log=warnings.append)
    return rows, warnings


@cache
def fd_rows(year):
    warnings, rows = [], []
    for code in NEW:
        if code in sf.FD_SLUGS and fxd(code, year) is not None:
            rows += sf.parse_fixturedownload(fxd(code, year), code, year, request=f"fixturedownload:{code}:{year}",
                                             seen_at=PROBED, log=warnings.append)
    return rows, warnings


# ------------------------------------------------------------- the leagues asked about

def test_every_league_espn_or_fixturedownload_covers_is_asked_and_every_other_says_why():
    for code, meta in sd.ALL_LEAGUES.items():
        if meta.espn is None:
            assert code == "POL" and REPORT["requests"]["espn_pol.1_teams.json"]["status"] == 404
            continue
        assert (code in sf.ESPN_SLUGS) != (code in sf.ESPN_NOT_ASKED), code
        if code in sf.ESPN_SLUGS:
            assert sf.ESPN_SLUGS[code] == meta.espn                      # the registry's code
    # Each league asked about had this season's clubs and its games; each not
    # asked had neither (no clubs, or last season's) - while football-data.co.uk
    # had matches of FIN, SC2 and ROU in the same days.
    for code in NEW:
        if code in sf.ESPN_SLUGS:
            assert teams(sf.ESPN_SLUGS[code])["season"]["year"] == 2026 and board_events(code), code
    for code in sf.ESPN_NOT_ASKED:
        slug = sd.ALL_LEAGUES[code].espn
        assert json.loads(raw(f"espn_{slug}_boards.json")) == [], code
        listed = teams(slug).get("teams", [])
        assert not listed or teams(slug)["season"]["year"] == 2025, code
    results, fixtures = football_data()
    both = pd.concat([results, fixtures]).drop_duplicates("match_id", keep="first")
    window = both[(both["kickoff"] >= "2026-10-26 04:00") & (both["kickoff"] < "2026-11-09 05:00")]
    assert dict(Counter(window.loc[window["league"].isin(list(sf.ESPN_NOT_ASKED)), "league"])) == \
        {"FIN": 3, "SC2": 1, "ROU": 1}
    # fixturedownload: every slug asked about answered for 2026-27, and every
    # one that answered is asked about (or is a European cup).
    answered = {k[len("fxd_"):].rsplit("_", 1)[0] for k, v in REPORT["requests"].items()
                if k.startswith("fxd_") and v["status"] == 200}
    for code, slug in sf.FD_SLUGS.items():
        if code not in sd.TOP:
            assert REPORT["requests"][f"fxd_{slug}_2026.json"]["status"] == 200, code
    assert answered == {s for c, s in sf.FD_SLUGS.items() if c not in sd.TOP} | CUPS
    assert set(sf.LOCAL) == set(sf.FD_SLUGS)
    assert sf.FEED_LEAGUES[:5] == sf.LIVE_LEAGUES == tuple(sd.TOP)
    assert len(sf.FEED_LEAGUES) == 31


def test_espns_own_league_index_lists_every_league_asked_about_and_none_of_the_others():
    """A second, independent line of evidence for ESPN_NOT_ASKED: ESPN's core
    API index of its soccer leagues (fetched by the same probe, 220 of them)
    lists all 31 codes asked about and none of the seven that are not -
    sco.3, sco.4, rou.1, sui.1, fin.1, irl.1 - nor pol.1."""
    index = json.loads(raw("espn_leagues.json"))
    assert REPORT["requests"]["espn_leagues.json"]["status"] == 200
    listed = {item["$ref"].split("/leagues/")[1].split("?")[0] for item in index["items"]}
    assert len(listed) == index["count"] == 220
    assert set(sf.ESPN_SLUGS.values()) <= listed and len(sf.ESPN_SLUGS) == 31
    assert not {sd.ALL_LEAGUES[code].espn for code in sf.ESPN_NOT_ASKED} & listed
    assert "pol.1" not in listed
    assert sorted(sd.ALL_LEAGUES[code].espn for code in sf.ESPN_NOT_ASKED) == \
        ["fin.1", "irl.1", "rou.1", "sco.3", "sco.4", "sui.1"]


def test_a_fetch_of_every_league_asks_each_feed_where_it_covers_the_league():
    reqs = sf.requests_for(NOW, leagues=sf.FEED_LEAGUES)
    espn = [r for r in reqs if r[3] == "espn"]
    fd = [r for r in reqs if r[3] == "fixturedownload"]
    assert (len(espn), len(fd), len(reqs)) == (31 * 15, 12, 477)
    assert {r[2] for r in espn} == set(sf.ESPN_SLUGS) and {r[2] for r in fd} == set(sf.FD_SLUGS)
    urls = {r[0]: r[1] for r in reqs}
    assert urls["espn:USA:20261101"] == ("https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/"
                                         "scoreboard?dates=20261101")
    assert urls["fixturedownload:SC0:2026"] == "https://fixturedownload.com/feed/json/scottish-premiership-2026"
    assert urls["fixturedownload:USA:2026"] == "https://fixturedownload.com/feed/json/mls-2026"
    # The default is still the five the app predicts (80 requests).
    assert len(sf.requests_for(NOW)) == 80


def test_the_app_asks_about_its_leagues_that_have_a_feed_and_never_rus():
    """leagues_for(in_app()): the 37 leagues the app carries, less the seven
    no feed covers - 30, in the registry's order (the five first, so a fetch
    its budget cuts short loses the others first), 462 requests. RUS has a
    feed and is out of the app: never asked, so its feed-only matches can
    never be picked by the model alone."""
    import soccer_leagues as slg
    app = slg.in_app()
    asked = sf.leagues_for(app)
    assert len(app) == 37 and "RUS" not in app
    assert len(asked) == 30 and "RUS" not in asked and asked[:5] == tuple(sd.TOP)
    assert set(asked) == set(sf.FEED_LEAGUES) - {"RUS"}
    assert list(asked) == [c for c in sf.FEED_LEAGUES if c in asked]            # the registry's order
    assert set(app) - set(asked) == {"SC2", "SC3", "FIN", "IRL", "POL", "ROU", "SWZ"}
    reqs = sf.requests_for(NOW, leagues=asked)
    assert len(reqs) == 30 * 15 + 12 == 462
    assert sum(r[3] == "fixturedownload" for r in reqs) == 12 and not any(r[2] == "RUS" for r in reqs)
    # A league no feed covers asks nothing (the seven passed straight from in_app()).
    assert sf.requests_for(NOW, leagues=app) == reqs
    # Any order, any duplicates, any code no feed has: the registry's order back.
    assert sf.leagues_for(["USA", "SC2", "E0", "USA", "XX"]) == ("E0", "USA")
    assert sf.leagues_for([]) == ()


def test_each_league_says_which_feed_covers_it_and_what_that_leaves_out():
    """coverage(code): ESPN's code, fixturedownload's name, or why neither -
    for every league (RUS too). In the app: both feeds 12, ESPN alone 18,
    none 7. coverage_caveat adds what a league's cards can rest on (no
    second source to confirm a time; no live check at all) beside its
    measured caveats, never in place of one."""
    import soccer_leagues as slg
    for code in sd.ALL_LEAGUES:
        c = sf.coverage(code)
        assert set(c) == {"espn", "fixturedownload", "none_because"}, code
        assert c["espn"] == sf.ESPN_SLUGS.get(code) and c["fixturedownload"] == sf.FD_SLUGS.get(code), code
        assert (c["none_because"] is None) == bool(c["espn"] or c["fixturedownload"]), code
    kind = {code: ("both" if c["espn"] and c["fixturedownload"] else "espn" if c["espn"] else
                   "fixturedownload" if c["fixturedownload"] else "none")
            for code, c in ((code, sf.coverage(code)) for code in slg.in_app())}
    assert Counter(kind.values()) == {"both": 12, "espn": 18, "none": 7}
    assert {c for c, k in kind.items() if k == "both"} == {"E0", "SP1", "I1", "D1", "F1", "E1", "F2", "SC0", "N1",
                                                           "P1", "T1", "USA"}
    assert {c for c, k in kind.items() if k == "none"} == {"SC2", "SC3", "FIN", "IRL", "POL", "ROU", "SWZ"}
    assert sf.coverage("POL")["none_because"] == sf.NO_ESPN["POL"] and "404" in sf.NO_ESPN["POL"]
    assert sf.coverage("SC2")["none_because"] == sf.ESPN_NOT_ASKED["SC2"]
    assert sf.coverage("RUS") == {"espn": "rus.1", "fixturedownload": None, "none_because": None}
    assert set(sf.FD_SLUGS) <= set(sf.ESPN_SLUGS)                 # no league has fixturedownload alone
    for code, k in kind.items():
        note = sf.coverage_caveat(code)
        assert note == (None if k == "both" else sf.COVERAGE_CAVEATS[k]), code
        assert note is None or note not in slg.caveats(code), code          # added, not already said
    with pytest.raises(KeyError):
        sf.coverage("XX")


def test_each_feed_takes_football_datas_season_for_the_league():
    """football-data.co.uk names a season by the year it starts - a calendar
    year where the league plays one (soccer_data.season_for): MLS's January
    request is for 2027, the Championship's still for 2026-27; and the J1
    League is August-May since 2026."""
    january = datetime(2027, 1, 15, 12, 0, tzinfo=timezone.utc)
    keys = {r[0]: r for r in sf.requests_for(january, leagues=["USA", "E1", "ARG", "JPN"])}
    assert keys["fixturedownload:USA:2027"][1].endswith("/mls-2027") and keys["fixturedownload:USA:2027"][4] == 2027
    assert keys["fixturedownload:E1:2026"][1].endswith("/championship-2026")
    assert {r[4] for r in keys.values() if r[2] == "ARG"} == {2027} and {r[4] for r in keys.values()
                                                                        if r[2] == "JPN"} == {2026}
    for code, when, season in (("USA", "2027-03-01T00:30Z", 2027), ("ARG", "2027-02-01T22:00Z", 2027),
                               ("BRA", "2026-12-06T19:00Z", 2026), ("JPN", "2027-03-06T05:00Z", 2026),
                               ("E1", "2027-03-06T15:00Z", 2026), ("USA", "2026-12-05T20:00Z", 2026)):
        ev = json.loads(json.dumps(board_events(code)[0]))
        ev["date"] = ev["competitions"][0]["date"] = when
        ev["season"]["year"] = season
        warnings = []
        row = sf.parse_espn({"events": [ev]}, code, log=warnings.append)[0]
        assert row["season"] == season and not warnings, (code, when)
        assert sd.season_for(code, sf.uk_date(sf.parse_utc(when)).isoformat()) == season, (code, when)
    # Every league ESPN is asked about, each on one of its own events moved to
    # mid-January 2027: the new season in a calendar-year league (and the
    # season ESPN files it in, where ESPN says otherwise, only warned), the
    # one under way in every other - the J1 League included since 2026.
    calendar = {code for code, meta in sd.ALL_LEAGUES.items() if meta.calendar == "calendar-year"}
    assert calendar == {"ARG", "BRA", "CHN", "FIN", "IRL", "NOR", "SWE", "USA"}
    for code in sf.ESPN_SLUGS:
        events = board_events(code) if code not in sd.TOP else \
            gz_json(FIXTURES / "feeds" / f"espn_{code}_board_20261025.json.gz")["events"]
        ev = json.loads(json.dumps(events[0]))
        ev["date"] = ev["competitions"][0]["date"] = "2027-01-16T15:00Z"
        warnings = []
        row = sf.parse_espn({"events": [ev]}, code, log=warnings.append)[0]
        assert row["season"] == (2027 if code in calendar else 2026), code
        assert bool(warnings) == (code in calendar), code             # ESPN's own year still 2026 there
    # An MLS 2027 fixturedownload row is in its season, not "outside" it.
    warnings = []
    rows = sf.parse_fixturedownload([dict(fxd("USA", 2026)[0], DateUtc="2027-03-01 00:30:00Z", HomeTeamScore=None,
                                          AwayTeamScore=None)], "USA", 2027, log=warnings.append)
    assert rows[0]["season"] == 2027 and not warnings


# ------------------------------------------------------------- the club tables

def test_every_espn_event_in_the_samples_maps_and_reads_cleanly():
    rows, warnings = espn_rows()
    assert not warnings
    events = {code: len(board_events(code)) for code in NEW if code in sf.ESPN_SLUGS}
    assert sum(events.values()) == len(rows) == 531
    assert events == {"E1": 36, "SP2": 22, "I2": 30, "D2": 18, "F2": 20, "E2": 20, "E3": 15, "EC": 25, "SC0": 16,
                      "SC1": 11, "N1": 18, "B1": 19, "P1": 19, "T1": 21, "G1": 14, "ARG": 33, "AUT": 12,
                      "BRA": 22, "CHN": 8, "DNK": 13, "JPN": 20, "MEX": 19, "NOR": 16, "RUS": 16, "SWE": 24,
                      "USA": 44}
    assert {r["status"] for r in rows} == {"scheduled", "final", "postponed"}
    for r in rows:
        assert set(r) == set(sf.ROW_FIELDS) and r["season"] == 2026
        if r["time_confirmed"]:
            assert r["uk_date"] == sf.uk_date(sf.parse_utc(r["kickoff_utc"])).isoformat() == r["date_hint"]
            assert r["match_id"] == sd.match_id(r["league"], pd.Timestamp(r["uk_date"]).date(), r["home"], r["away"])
        else:
            assert (r["kickoff_utc"], r["uk_date"], r["match_id"]) == (None, None, None)
    # The only timeless events: Argentina's, whose later rounds ESPN had not timed.
    assert Counter(r["league"] for r in rows if not r["time_confirmed"]) == {"ARG": 15}


def test_the_espn_tables_hold_each_leagues_clubs_one_to_one():
    """Every club ESPN listed for a league (its teams page and every event)
    is in the table by its id, under the same display name; the table is one
    to one; and its clubs are exactly the ones football-data.co.uk has that
    season."""
    for code in NEW:
        if code not in sf.ESPN_SLUGS:
            continue
        table = sf.ESPN_CLUBS[code]
        assert len({club for _, club in table.values()}) == len(table), code
        listed = {t["team"]["id"]: t["team"]["displayName"] for t in teams(sf.ESPN_SLUGS[code])["teams"]}
        on_boards = {c["team"]["id"]: c["team"]["displayName"] for e in board_events(code)
                     for c in e["competitions"][0]["competitors"]}
        for tid, name in {**listed, **on_boards}.items():
            assert tid in table and table[tid][0] == name, (code, tid, name)
        assert set(listed) == set(table), code                          # nothing in the table ESPN did not list
        assert {club for _, club in table.values()} == clubs_of(code, 2026), code


# Where a pairing is played once in the phase under way on 2026-10-29, by
# league: (the phase's first day, whether home and away are told apart). The
# season so far - a double round robin before any split - but once either way
# round in Liga MX's Apertura and Argentina's Clausura (both from July). Not
# SC0 or SC1 (three and four meetings a season) or MLS (an unbalanced schedule).
ONCE_IN_PHASE = {**{code: ("2026-07-01", True) for code in ("E1", "SP2", "I2", "D2", "F2", "E2", "E3", "EC", "N1",
                                                              "B1", "P1", "T1", "G1", "AUT", "DNK", "JPN", "RUS")},
                 **{code: ("2026-01-01", True) for code in ("BRA", "CHN", "NOR", "SWE")},
                 "MEX": ("2026-07-01", False), "ARG": ("2026-07-01", False)}
# By league: how many of the table's clubs a match confirms; and (caught,
# pairs) - of every pair of the table's clubs, how many the pairing check
# would catch were the two swapped in the table.
CONFIRMED_BY_A_MATCH = {"E1": 24, "SP2": 2, "I2": 20, "D2": 0, "F2": 18, "E2": 10, "E3": 4, "EC": 24, "SC0": 12,
                        "SC1": 2, "N1": 18, "B1": 0, "P1": 18, "T1": 18, "G1": 0, "ARG": 6, "AUT": 0, "BRA": 20,
                        "CHN": 0, "DNK": 2, "JPN": 0, "MEX": 2, "NOR": 0, "RUS": 0, "SWE": 16, "USA": 30}
SWAPS_CAUGHT = {"E1": (231, 276), "SP2": (160, 231), "I2": (153, 190), "D2": (110, 153), "F2": (121, 153),
                "E2": (158, 276), "E3": (142, 276), "EC": (215, 276), "N1": (106, 153), "B1": (105, 153),
                "P1": (107, 153), "T1": (110, 153), "G1": (74, 91), "ARG": (332, 435), "AUT": (63, 66),
                "BRA": (190, 190), "CHN": (120, 120), "DNK": (66, 66), "JPN": (135, 190), "MEX": (152, 153),
                "NOR": (120, 120), "RUS": (25, 120), "SWE": (120, 120)}


def _clashes(events, met, swap=None):
    """The events (home, away, day) whose pairing - with two clubs swapped,
    where `swap` names them - football-data.co.uk has, but on other days."""
    relabel = {swap[0]: swap[1], swap[1]: swap[0]} if swap else {}
    out = []
    for home, away, day in events:
        days = met.get((relabel.get(home, home), relabel.get(away, away)))
        if days and day not in days:
            out.append((home, away, day))
    return out


def test_the_espn_tables_are_confirmed_by_matches_beyond_the_names():
    """Names proposed every pair; matches check them. (1) A club in an ESPN
    event that football-data.co.uk has under the same id at the same kickoff,
    or that reconcile joins to fixturedownload's row at the same kickoff
    (fixturedownload's tables are held to football-data.co.uk's match ids in
    test_fixturedownload_agrees_with_football_data_on_the_matches_played): a
    club plays one match at a time, so a wrong pair could not join - 246 of
    the 488 clubs. (2) Where a pairing is played once in the phase under way
    (ONCE_IN_PHASE), no event in the samples is, through the tables, a pairing
    football-data.co.uk already has from another day; swapping two clubs of a
    league in its table breaks that for 3115 of the 4114 pairs (RUS 25 of 120:
    football-data.co.uk's 2026-27 file stops at 2026-08-02, 16 matches). Left
    to the names alone: SC1's eight unmatched clubs."""
    rows, _ = espn_rows()
    matches = sf.reconcile(rows, fd_rows(2026)[0], log=lambda *_: None)
    results, fixtures = football_data()
    by_id = {g.match_id: g for g in pd.concat([results, fixtures]).drop_duplicates("match_id").itertuples()}
    confirmed = defaultdict(set)
    for m in matches:
        g = by_id.get(m["match_id"]) if m["espn_id"] and m["match_id"] else None
        if (g is not None and pd.Timestamp(m["kickoff_espn"]).tz_localize(None) == g.kickoff) or \
                (m["espn_id"] and m["fd_id"] and m["kickoff_espn"] and m["kickoff_espn"] == m["kickoff_fd"]):
            confirmed[m["league"]] |= {m["home"], m["away"]}
    tables = {code: {club for _, club in sf.ESPN_CLUBS[code].values()} for code in NEW if code in sf.ESPN_SLUGS}
    assert {code: len(confirmed[code] & clubs) for code, clubs in tables.items()} == CONFIRMED_BY_A_MATCH
    assert (sum(CONFIRMED_BY_A_MATCH.values()), sum(map(len, tables.values()))) == (246, 488)
    assert sorted(tables["SC1"] - confirmed["SC1"]) == ["Arbroath", "Ayr", "Inverness C", "Livingston", "Morton",
                                                        "Partick", "Queens Park", "Raith Rvs"]

    caught = {}
    for code, (start, ordered) in ONCE_IN_PHASE.items():
        met = defaultdict(set)                             # pairing -> the days football-data.co.uk has it
        for g in results[(results["league"] == code) & (results["date"] >= start)].itertuples():
            met[(g.home, g.away)].add(g.date)
            if not ordered:
                met[(g.away, g.home)].add(g.date)
        events = [(r["home"], r["away"], r["uk_date"] or r["date_hint"]) for r in rows if r["league"] == code]
        assert _clashes(events, met) == [], code
        pairs = list(combinations(sorted(tables[code]), 2))
        caught[code] = (sum(bool(_clashes(events, met, p)) for p in pairs), len(pairs))
    assert caught == SWAPS_CAUGHT
    assert tuple(map(sum, zip(*SWAPS_CAUGHT.values()))) == (3115, 4114)


def test_the_fixturedownload_tables_map_each_season_one_to_one():
    seen = 0
    for code in NEW:
        if code not in sf.FD_SLUGS:
            continue
        for year in (2025, 2026):
            feed = fxd(code, year)
            if feed is None:
                continue
            names = {r[k] for r in feed for k in ("HomeTeam", "AwayTeam")}
            mapped = {sf.FD_CLUBS[code][sf._nfc(n)] for n in names}
            assert len(mapped) == len(names), (code, year)                  # one to one within a season
            assert mapped == clubs_of(code, year), (code, year)
            seen += 1
    assert seen == 12                                       # Ligue 2 and the Scottish Premiership: 2026-27 only
    _, warnings = fd_rows(2025)
    assert not warnings and not fd_rows(2026)[1]


# ------------------------------------------------------------- against football-data.co.uk

# Matches ESPN and football-data.co.uk both have, by league: the 69 of the
# league probe's own files (soccer_data's ESPN_PAIRED, there paired by name
# likeness; here through the club tables and reconcile) and the second
# tiers' 13 from the frozen snapshot.
ESPN_AND_FOOTBALL_DATA = {"ARG": 3, "BRA": 12, "DNK": 1, "MEX": 1, "SWE": 8, "USA": 14, "E2": 5, "E3": 2,
                          "EC": 12, "SC0": 6, "SC1": 1, "P1": 1, "T1": 3, "SP2": 1, "I2": 10, "F2": 2}


def test_espn_has_every_match_football_data_has_in_its_days_under_its_id_kickoff_and_score():
    """Every football-data.co.uk match (results and fixtures) of a covered
    league in the days ESPN was asked about (Oct 26 - Nov 8, ESPN's days are
    the US Eastern dates) is a match reconcile makes of the feeds, under the
    same id, at the same kickoff - football-data.co.uk's UK clock read as
    UTC-or-summer-time equals ESPN's UTC - and, where both have a result,
    with the same score."""
    rows, _ = espn_rows()
    matches = sf.reconcile(rows, fd_rows(2026)[0], log=lambda *_: None)
    by_id = {m["match_id"]: m for m in matches if m["espn_id"]}
    results, fixtures = football_data()
    both = pd.concat([results[results["season"] == 2026], fixtures]).drop_duplicates("match_id", keep="first")
    window = both[both["league"].isin([c for c in NEW if c in sf.ESPN_SLUGS])
                  & (both["kickoff"] >= "2026-10-26 04:00") & (both["kickoff"] < "2026-11-09 05:00")]
    assert dict(Counter(window["league"])) == ESPN_AND_FOOTBALL_DATA
    assert len(window) == 82
    # the 69 soccer_data's notes list (tests/test_soccer_leagues.py's ESPN_PAIRED), reproduced through reconcile
    assert sum(n for code, n in ESPN_AND_FOOTBALL_DATA.items() if code not in SECOND_TIERS) == 69
    scored = 0
    for g in window.itertuples():
        m = by_id[g.match_id]
        assert (m["league"], m["season"], m["home"], m["away"]) == (g.league, g.season, g.home, g.away)
        assert pd.Timestamp(m["kickoff_espn"]).tz_localize(None) == g.kickoff, g.match_id
        if isinstance(g.result, str):
            scored += 1
            assert m["status"] == "final" and (m["hg"], m["ag"], m["result"]) == (int(g.hg), int(g.ag), g.result)
    assert scored == 48


# (league, fixturedownload's season) -> (rows, played, the same match_id,
# the same kickoff, the same score) against football-data.co.uk's results;
# every played row pairs with one of its matches.
FD_AGREES = {("E1", 2025): (552, 552, 552, 550, 552), ("N1", 2025): (306, 305, 305, 301, 305),
             ("P1", 2025): (306, 306, 306, 305, 306), ("T1", 2025): (306, 304, 304, 300, 304),
             ("USA", 2025): (510, 510, 509, 496, 510),
             ("E1", 2026): (552, 144, 144, 144, 144), ("F2", 2026): (306, 90, 90, 90, 90),
             ("SC0", 2026): (198, 60, 60, 60, 60), ("N1", 2026): (306, 89, 89, 89, 89),
             ("P1", 2026): (306, 81, 81, 81, 81), ("T1", 2026): (306, 81, 81, 80, 81),
             ("USA", 2026): (510, 466, 466, 463, 463)}
# The kickoffs that differ: minutes fixturedownload's is later (earlier < 0).
FD_KICKOFF_DIFFERS = {("E1", 2025): [15, 150], ("N1", 2025): [10, 15, 15, 25], ("P1", 2025): [15],
                      ("T1", 2025): [2, 3, 60, 60], ("T1", 2026): [3],
                      ("USA", 2025): [-120, -55, -10, -10, 15, 38, 45, 45, 66, 70, 88, 91, 131, 193],
                      ("USA", 2026): [60, 60, 95]}


def test_fixturedownload_agrees_with_football_data_on_the_matches_played():
    """The kickoffs that differ (29, fixturedownload's later in 25): minutes
    in E1, N1, P1 and T1 (late starts) but E1's Norwich v Ipswich of
    11/04/2026 (2h30m) and the T1 pair of 14/09/2025 football-data.co.uk has
    an hour early; MLS 2025's 14, from Austin v Seattle two hours earlier
    (the one match_id that differs, across UK midnight) to 3h13m later (odd
    minutes, like a start after a weather delay); MLS 2026's Inter Miami v
    Atlanta (1h35m) and the two in the hour the UK clock repeats
    (25/10/2026). The scores: every one, except three
    MLS 2026 matches - Galaxy v St. Louis and LAFC v Salt Lake of
    23/07/2026 (3-1 and 1-3 here, 1-3 and 3-1 there: the two look swapped
    in one source) and Orlando v Chicago of 20/08/2026 (1-0 here, 1-2
    there). Which is right is not settled here; a feed score is never a
    result alone (ESPN must call it final - reconcile)."""
    results, _ = football_data()
    got, differ = {}, defaultdict(list)
    for year in (2025, 2026):
        rows, _ = fd_rows(year)
        for code in {r["league"] for r in rows}:
            mine = [r for r in rows if r["league"] == code]
            theirs = defaultdict(list)
            for g in results[(results["league"] == code) & (results["season"] == year)].itertuples():
                theirs[(g.home, g.away)].append(g)
            played = [r for r in mine if r["status"] == "final"]
            ids = kicks = scores = 0
            for r in played:
                g = min(theirs[(r["home"], r["away"])],                     # the nearest by date (MLS, Scotland repeat)
                        key=lambda g: abs((pd.Timestamp(g.date) - pd.Timestamp(r["date_hint"])).days))
                ids += r["match_id"] == g.match_id
                gap = (pd.Timestamp(r["kickoff_utc"]).tz_localize(None) - g.kickoff).total_seconds() / 60
                kicks += gap == 0
                if gap:
                    differ[(code, year)].append(int(gap))
                scores += (r["hg"], r["ag"]) == (int(g.hg), int(g.ag))
            got[(code, year)] = (len(mine), len(played), ids, kicks, scores)
    assert got == FD_AGREES
    assert {k: sorted(v) for k, v in differ.items()} == FD_KICKOFF_DIFFERS
    assert sum(map(len, differ.values())) == 29 and sum(g > 0 for v in differ.values() for g in v) == 25


def test_both_feeds_agree_on_every_match_espn_listed_but_two_espn_moved():
    """ESPN v fixturedownload on the 174 matches both list (the seven
    leagues with both feeds): every kickoff equal (and confirmed) but two
    Ligue 2 matches ESPN had moved - Reims v Sochaux to Monday, Montpellier
    v Metz to Friday - that fixturedownload still had on Saturday at 13:00
    UTC. No score contradicts."""
    rows, _ = espn_rows()
    matches = sf.reconcile(rows, fd_rows(2026)[0], log=lambda *_: None)
    both = [m for m in matches if m["sources"] == ["espn", "fixturedownload"]]
    assert len(both) == 174 and not [m for m in matches if m["league"] in sf.FD_SLUGS and m["sources"] == ["espn"]]
    differ = sorted((m["league"], m["home"], m["away"], m["kickoff_espn"], m["kickoff_fd"]) for m in both
                    if m["kickoff_espn"] != m["kickoff_fd"])
    assert differ == [("F2", "Montpellier", "Metz", "2026-11-06T19:00:00Z", "2026-11-07T13:00:00Z"),
                      ("F2", "Reims", "Sochaux", "2026-11-02T19:45:00Z", "2026-10-31T13:00:00Z")]
    assert sum(m["time_confirmed"] for m in both) == 172
    assert not [m for m in matches if m["status"] == "disputed"]


# ------------------------------------------------------------- placeholders

def test_no_played_match_in_the_samples_is_taken_for_a_placeholder():
    for code in NEW:
        if code not in sf.FD_SLUGS:
            continue
        for year in (2025, 2026):
            for r in fxd(code, year) or []:
                if r["HomeTeamScore"] is not None:
                    assert not sf.is_placeholder(sf.parse_utc(r["DateUtc"]), code), (code, r)
    # MLS kicks off at 00:00 UTC for real - 8pm Eastern summer time.
    midnight = {year: sum(sf.parse_utc(r["DateUtc"]).strftime("%H:%M") == "00:00" and r["HomeTeamScore"] is not None
                          for r in fxd("USA", year)) for year in (2025, 2026)}
    assert midnight == {2025: 4, 2026: 5} and sf.LOCAL["USA"] is None


@AS_PROBED
def test_the_placeholders_in_the_samples_are_whole_unscheduled_rounds():
    """Midnight UTC in the Eredivisie and the Primeira Liga, local midnight in
    Ligue 2, the Primeira Liga's summer time and the Super Lig. The
    Championship's and the Scottish Premiership's unscheduled rounds read
    Saturday 15:00 UK (or a midweek 19:45) - real kickoff times, which
    cannot be told apart, as in the Premier League: never confirmed alone.
    One placeholder is a single match: the Eredivisie's Twente v Sparta,
    whose round 16 is timed for Dec 12-13, filed alone at 00:00 UTC on
    Saturday 2027-02-06 - moved, with no time yet: no time, as it should."""
    counts, moved = {}, []
    for code in NEW:
        if code not in sf.FD_SLUGS:
            continue
        rows = sf.parse_fixturedownload(fxd(code, 2026), code, 2026, log=lambda *_: None)
        holders = [r for r in rows if r["kickoff_utc"] is None]
        counts[code] = len(holders)
        assert all(r["status"] == "scheduled" and r["time_confirmed"] is False for r in holders)
        for rnd in {r["round"] for r in holders}:
            timed = [r for r in rows if r["round"] == rnd and r["kickoff_utc"]]
            if timed:
                moved += [(code, r["home"], r["away"], r["date_hint"]) for r in holders if r["round"] == rnd]
    assert counts == {"E1": 0, "F2": 171, "SC0": 0, "N1": 100, "P1": 198, "T1": 162, "USA": 0}
    assert moved == [("N1", "Twente", "Sparta Rotterdam", "2027-02-06")]


# ------------------------------------------------------------- pairings that repeat

def test_the_leagues_where_pairings_repeat_are_the_ones_football_data_says():
    """REPEATS: every league in which football-data.co.uk has one pairing
    twice in its 2025 season (2025-26, or 2025 for a calendar-year league),
    on the probe's files and the frozen snapshots - and no other."""
    results, _ = football_data()
    top = pd.read_csv(FIXTURES / "football_data_top5_2025_to_20261021.csv.gz")
    both = pd.concat([results, top])
    s25 = both[both["season"] == 2025]
    counts = s25.groupby(["league", "home", "away"]).size()
    repeat = set(counts[counts > 1].index.get_level_values("league"))
    assert repeat == set(sf.REPEATS)
    assert set(s25["league"]) == set(sd.ALL_LEAGUES)                      # every league had its 2025 season read


def test_where_pairings_repeat_each_meeting_is_its_own_match():
    """The Scottish Premiership's fixturedownload season lists 66 pairings
    twice (33 rounds of 12 clubs): never warned there, every row a match,
    each of ESPN's 16 events beside the row of its own date."""
    warnings = []
    rows = sf.parse_fixturedownload(fxd("SC0", 2026), "SC0", 2026, log=warnings.append)
    assert sum(n > 1 for n in Counter((r["home"], r["away"]) for r in rows).values()) == 66 and not warnings
    espn = [r for r in espn_rows()[0] if r["league"] == "SC0"]
    matches = sf.reconcile(espn, rows, log=warnings.append)
    assert len(matches) == 198 and not warnings
    paired = [m for m in matches if m["espn_id"]]
    assert len(paired) == 16 and all(m["fd_id"] and m["kickoff_espn"] == m["kickoff_fd"] for m in paired)
    assert len({m["match_id"] for m in matches if m["match_id"]}) == sum(bool(m["match_id"]) for m in matches)


def _sc0_event(eid, when, kind="STATUS_SCHEDULED"):
    ev = json.loads(json.dumps(next(e for e in board_events("SC0") if e["competitions"][0]["status"]["type"]["name"]
                                    == "STATUS_SCHEDULED")))
    ev["id"], ev["date"] = eid, when
    comp = ev["competitions"][0]
    comp["id"], comp["date"] = eid, when
    for c in comp["competitors"]:
        tid = "257" if c["homeAway"] == "home" else "256"                    # Rangers v Celtic
        c["id"] = c["team"]["id"] = tid
        c["team"]["displayName"] = sf.ESPN_CLUBS["SC0"][tid][0]
    return ev


def test_two_meetings_of_one_pairing_are_two_matches_and_each_source_joins_its_own():
    """Made up on real events: Rangers v Celtic twice within the feeds'
    window (a split's extra round). Each ESPN event is a match; each
    fixturedownload row joins the ESPN event of its date (within PAIR_DAYS,
    nearest first) - here one a day apart - and a row with no ESPN event
    near it is a match of its own. In a league where pairings never repeat,
    the pairing is still one match (the Premier League's rule, unchanged)."""
    espn = sf.parse_espn({"events": [_sc0_event("1", "2026-11-01T12:00Z"), _sc0_event("2", "2026-12-06T12:00Z")]},
                         "SC0", log=lambda *_: None)
    fd = sf.parse_fixturedownload([
        {"MatchNumber": 70, "RoundNumber": 12, "DateUtc": "2026-11-01 12:00:00Z", "HomeTeam": "Rangers",
         "AwayTeam": "Celtic", "HomeTeamScore": None, "AwayTeamScore": None},
        {"MatchNumber": 140, "RoundNumber": 24, "DateUtc": "2026-12-05 15:00:00Z", "HomeTeam": "Rangers",
         "AwayTeam": "Celtic", "HomeTeamScore": None, "AwayTeamScore": None},
        {"MatchNumber": 190, "RoundNumber": 32, "DateUtc": "2027-01-02 15:00:00Z", "HomeTeam": "Rangers",
         "AwayTeam": "Celtic", "HomeTeamScore": None, "AwayTeamScore": None}], "SC0", 2026, log=lambda *_: None)
    warnings = []
    matches = sf.reconcile(espn, fd, log=warnings.append)
    got = [(m["espn_id"], m["fd_id"], m["match_id"], m["kickoff_utc"], m["kickoff_first"]) for m in matches]
    assert got == [("1", "2026-70", "SC0-20261101-rangers-celtic", "2026-11-01T12:00:00Z", "2026-11-01T12:00:00Z"),
                   ("2", "2026-140", "SC0-20261206-rangers-celtic", "2026-12-06T12:00:00Z", "2026-12-05T15:00:00Z"),
                   (None, "2026-190", "SC0-20270102-rangers-celtic", "2027-01-02T15:00:00Z", "2027-01-02T15:00:00Z")]
    assert len(warnings) == 1 and "kickoff differs" in warnings[0]          # the second one's, by a day
    # Four days apart is not one match (PAIR_DAYS = 3): two rows, neither confirmed.
    far = sf.parse_fixturedownload([{"MatchNumber": 70, "RoundNumber": 12, "DateUtc": "2026-11-05 19:45:00Z",
                                     "HomeTeam": "Rangers", "AwayTeam": "Celtic", "HomeTeamScore": None,
                                     "AwayTeamScore": None}], "SC0", 2026, log=lambda *_: None)
    apart = sf.reconcile(espn[:1], far, log=lambda *_: None)
    assert [m["sources"] for m in apart] == [["espn"], ["fixturedownload"]]
    # The same in the Championship (pairings never repeat): one match, warned.
    e1 = [dict(r, league="E1", home="Watford", away="Millwall") for r in espn]
    one = sf.reconcile(e1, [], log=lambda *_: None)
    assert len(one) == 1 and "lists this pairing 2 times" in one[0]["warnings"][0]


# ------------------------------------------------------------- knock-out phases

def test_a_play_off_or_cup_tie_is_skipped_in_a_division_football_data_files_without_them():
    """ESPN names a knock-out phase in the event's season slug (KNOCKOUT): all
    123 play-off and cup-tie events the probe saved match it and none of the
    661 league events of the 31 leagues does. In a main division (whose
    football-data.co.uk file leaves play-offs out) such an event is skipped
    before its clubs are looked up - a lower-league club in it is not a club
    the table is missing; in an extra league (whose file keeps MLS Cup, the
    liguilla, ...) it is read."""
    knockout = [e for lg in ("usa.usl.1", "eng.fa", "eng.league_cup", "esp.copa_del_rey", "ger.dfb_pokal")
                for e in json.loads(raw(f"espn_{lg}_boards.json"))]
    assert len(knockout) == 123 and all(sf.KNOCKOUT.search(e["season"]["slug"]) for e in knockout)
    league = [e for code in NEW if code in sf.ESPN_SLUGS for e in board_events(code)]
    top = {}
    for path in (FIXTURES / "feeds").glob("espn_*_board_2*.json.gz"):
        top.update({e["id"]: e for e in gz_json(path)["events"]})
    assert len(league) + len(top) == 661
    assert not any(sf.KNOCKOUT.search(e["season"]["slug"]) for e in league + list(top.values()))
    warnings, missing = [], set()
    assert sf.parse_espn({"events": knockout}, "E1", log=warnings.append, missing=missing) == []
    assert len(warnings) == 123 and all("play-off or cup tie" in w for w in warnings) and not missing
    playoff = dict(board_events("USA")[0], season={"slug": "mls-cup-playoffs---round-one", "type": 1, "year": 2026})
    assert len(sf.parse_espn({"events": [playoff]}, "USA", log=lambda *_: None)) == 1
    assert sf.knockout_phase(playoff, "USA") is None
    assert sf.knockout_phase(dict(playoff, season={"slug": "championship-playoffs"}), "E1") == "championship-playoffs"


def test_a_knock_out_phase_skipped_is_listed_in_the_snapshot_not_only_logged(tmp_path):
    """A division's feed going quiet because ESPN names its phase a play-off
    (Belgium's or Scotland's split, if ESPN labels it so - not measured)
    must show: every phase skipped is in last_run.skipped_phase as
    "league:slug", once, beside the board's league events read as before.
    An extra league reads the same events, and lists nothing."""
    knockout = [e for lg in ("usa.usl.1", "eng.fa") for e in json.loads(raw(f"espn_{lg}_boards.json"))][::20]
    slugs = {e["season"]["slug"] for e in knockout}
    league = board_events("B1")[:3]
    split = [dict(e, season=dict(e["season"], slug="jupiler-pro-league-champions-play-offs")) for e in league[:1]]

    def get(url):
        if "/soccer/bel.1/" in url:
            return json.dumps({"events": knockout + split + league[1:]}).encode()
        raise ConnectionError(url)
    snap = sf.sync(NOW, path=tmp_path / "feed.json", get=get, leagues=["B1"], days_back=0, days_ahead=1,
                   log=lambda *_: None)
    run = snap["last_run"]
    assert run["skipped_phase"] == sorted({f"B1:{s}" for s in slugs} | {"B1:jupiler-pro-league-champions-play-offs"})
    assert {r["source_id"] for r in snap["rows"]} == {e["id"] for e in league[1:]}
    assert sf.load(tmp_path / "feed.json")["last_run"]["skipped_phase"] == run["skipped_phase"]
    # Nothing skipped: an empty list, never a missing key.
    got = sf.fetch(NOW, get=probe_getter(), leagues=["E2"], days_back=3, log=lambda *_: None)
    assert got["skipped_phase"] == []
    # An extra league's file keeps its play-offs: the same phase is read there.
    phases = set()
    usa = [dict(board_events("USA")[0], season={"slug": "mls-cup-playoffs---round-one", "type": 1, "year": 2026})]
    assert len(sf.parse_espn({"events": usa}, "USA", log=lambda *_: None, phases=phases)) == 1 and not phases


# ------------------------------------------------------------- a fetch of every league

def probe_getter(calls=None):
    """Each feed URL answered from the league probe: an ESPN day with the
    events whose US Eastern date it is (ESPN's day: on all 658 days of the
    probe's boards with events, the days that answered empty are exactly
    those), a day not probed as a failure; fixturedownload's 2026 files."""
    slug_of = {slug: code for code, slug in sf.ESPN_SLUGS.items()}
    fd_of = {slug: code for code, slug in sf.FD_SLUGS.items()}

    def get(url):
        if calls is not None:
            calls.append(url)
        if "/scoreboard?dates=" in url:
            slug, day = url.split("/soccer/")[1].split("/scoreboard?dates=")
            if f"espn_{slug}_board_{day}" not in REPORT["requests"]:
                raise ConnectionError(f"not probed: {url}")
            events = [e for e in board_events(slug_of[slug])
                      if sf.parse_utc(e["date"]).astimezone(EASTERN).strftime("%Y%m%d") == day]
            return json.dumps({"events": events}).encode()
        slug, year = url.rsplit("/", 1)[1].rsplit("-", 1)
        feed = fxd(fd_of[slug], int(year))
        if feed is None:
            raise ConnectionError(f"not probed: {url}")
        return json.dumps(feed).encode()
    return get


def test_the_espn_days_are_the_us_eastern_dates():
    days = checked = 0
    for key, entry in REPORT["requests"].items():
        if not key.endswith("_boards.json") or not entry["events"]:
            continue
        slug = key[len("espn_"):-len("_boards.json")]
        events = json.loads(raw(f"espn_{slug}_boards.json"))
        sizes = {k.rsplit("_", 1)[1]: v["bytes"] for k, v in REPORT["requests"].items()
                 if k.startswith(f"espn_{slug}_board_")}
        empty = min(sizes.values())
        dated = {sf.parse_utc(e["date"]).astimezone(EASTERN).strftime("%Y%m%d") for e in events}
        for day, size in sizes.items():
            checked += 1
            days += (day in dated) == (size > empty + 50)
    assert days == checked == 658


def test_a_sync_of_every_new_league_from_the_probe(tmp_path):
    calls = []
    snap = sf.sync(NOW, path=tmp_path / "feed.json", get=probe_getter(calls), leagues=NEW, days_back=3,
                   log=lambda *_: None)
    run = snap["last_run"]
    assert len(calls) == run["answered"] == 26 * 14 + 7 and not run["failed"] and run["unmapped"] == []
    assert run["skipped_phase"] == []                       # no play-off or cup tie on a league's board
    espn = [r for r in snap["rows"] if r["source"] == "espn"]
    assert len(espn) == 531
    matches = sf.reconciled(snap, log=lambda *_: None)
    assert {m["league"] for m in matches} == set(NEW)
    # "unconfirmed": a fixturedownload score from before ESPN's days (ESPN not asked).
    assert {m["status"] for m in matches} == {"scheduled", "final", "postponed", "unconfirmed"}
    assert all(not m["espn_id"] for m in matches if m["status"] == "unconfirmed")
    finals = [m for m in matches if m["status"] == "final"]
    assert len(finals) == sum(r["status"] == "final" for r in espn) == 74
    assert all(m["provisional"] and m["result_seen_at"] == "2026-10-29T12:42:59Z" for m in finals)


def test_an_outage_of_every_feed_stops_after_twenty_requests(tmp_path):
    """Both feeds timing out, every league asked: five leagues' three
    scoreboard days and season file each, then both feeds are taken for down
    - 20 requests, not 477 (or ten minutes of them); nothing on file lost."""
    path = tmp_path / "feed.json"
    first = sf.sync(NOW, path=path, get=probe_getter(), leagues=NEW, days_back=3, log=lambda *_: None)
    calls, lines = [], []

    def down(url):
        calls.append(url)
        raise ConnectionError("timed out")
    second = sf.sync(NOW, path=path, get=down, leagues=sf.FEED_LEAGUES, log=lines.append)
    assert len(calls) == 5 * sf.STRAIGHT_FAILS + sf.FEED_DOWN == 20
    assert set(second["last_run"]["failed"]) == {k for k, *_ in sf.requests_for(NOW, leagues=sf.FEED_LEAGUES)}
    assert second["rows"] == first["rows"]
    assert sum("taken for down" in line for line in lines) == 2
    # ESPN alone down: fixturedownload is still read in every league it covers.
    good, calls = probe_getter(), []

    def espn_down(url):
        calls.append(url)
        if "espn.com" in url:
            raise ConnectionError("timed out")
        return good(url)
    got = sf.fetch(NOW, get=espn_down, leagues=NEW, days_back=3, log=lambda *_: None)
    assert sum("espn.com" in u for u in calls) == 15 and set(got["ok"]) == {f"fixturedownload:{c}:2026"
                                                                            for c in NEW if c in sf.FD_SLUGS}
    # One league's scoreboard gone (a renamed code): that league alone is cut short.
    calls = []

    def one_gone(url):
        calls.append(url)
        if "/soccer/eng.3/" in url:
            raise ConnectionError("404")
        return good(url)
    got = sf.fetch(NOW, get=one_gone, leagues=NEW, days_back=3, log=lambda *_: None)
    assert sum("/eng.3/" in u for u in calls) == sf.STRAIGHT_FAILS
    assert {k for k in got["failed"]} == {k for k, *_ in sf.requests_for(NOW, 3, leagues=["E2"]) if "espn" in k}


# ------------------------------------------------------------- the live snapshot

@pytest.mark.snapshot
def test_every_club_football_data_has_this_season_maps_in_every_feed_of_its_league():
    """The tables hold one season's clubs: the promoted ones must be added
    every summer (calendar-year leagues: every winter). Checked against
    football-data.co.uk's snapshot of each league's current season, so
    reported, never a gate."""
    m, fx = sd.load_matches(leagues="all"), sd.load_fixtures(leagues="all")
    for code in sf.FEED_LEAGUES:
        season = int(m.loc[m["league"] == code, "season"].max())
        rows = pd.concat([m[(m["league"] == code) & (m["season"] == season)],
                          fx[(fx["league"] == code) & (fx["season"] == season)]])
        clubs = set(rows["home"]) | set(rows["away"])
        if code in sf.ESPN_SLUGS:
            assert clubs <= {c for _, c in sf.ESPN_CLUBS[code].values()}, (code, "ESPN", clubs)
        if code in sf.FD_SLUGS:
            assert clubs <= set(sf.FD_CLUBS[code].values()), (code, "fixturedownload", clubs)
