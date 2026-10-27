"""The fixture feeds (soccer_feed.py): ESPN's scoreboard and fixturedownload.com.

Read first from the real responses the probe saved on GitHub's machines on
2026-10-26 (tests/fixtures/feeds/), checked against football-data.co.uk's own
rows for the same matches; then made-up edge cases built on a real event: the
clock change, a postponement, contradicting scores, a missing time, a club
with no mapping, a failed request.

A test that pins what the samples said about matches still to be played at
probe time skips if the samples are ever re-probed (probe_report.json's date
moves); everything about matches already played holds whatever the date."""

import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import soccer_data as sd
import soccer_feed as sf
import soccer_teams as st

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FEEDS = FIXTURES / "feeds"
PROBED = json.loads((FEEDS / "probe_report.json").read_text())["probed"]
AS_PROBED = pytest.mark.skipif(not PROBED.startswith("2026-10-26"),
                               reason="samples re-probed: the 2026-10-26 evidence no longer applies")
NOW = datetime(2026, 10, 26, 13, 13, 23, tzinfo=timezone.utc)


def sample(name):
    with gzip.open(FEEDS / f"{name}.json.gz") as fh:
        return json.loads(fh.read())


def boards(pattern="espn_*_board_*.json.gz"):
    """(league, request key, payload) for every saved ESPN scoreboard."""
    out = []
    for path in sorted(FEEDS.glob(pattern)):
        league, day = path.name[len("espn_"):].split("_board_")
        out.append((league, f"espn:{league}:{day.split('.')[0]}", sample(path.name[:-len(".json.gz")])))
    return out


@pytest.fixture(scope="module")
def espn():
    warnings, rows = [], []
    for league, key, payload in boards("espn_*_board_2*.json.gz"):
        rows += sf.parse_espn(payload, league, request=key, seen_at=PROBED, log=warnings.append)
    return rows, warnings


def fd_rows(season, log=None):
    rows, sink = [], [] if log is None else log
    for league in sd.TOP:
        rows += sf.parse_fixturedownload(sample(f"fixturedownload_{league}_{season}"), league, season,
                                         request=f"fixturedownload:{league}:{season}", seen_at=PROBED,
                                         log=sink.append)
    return rows


@pytest.fixture(scope="module")
def football_data():
    m = sd.load_matches()
    return m[m["league"].isin(sd.TOP) & m["season"].isin([2025, 2026])]


def by_pairing(frame):
    return {(r.league, int(r.season), r.home, r.away): r for r in frame.itertuples()}


def sample_getter(calls=None):
    """A getter that answers each feed URL with the saved response."""
    def get(url):
        if calls is not None:
            calls.append(url)
        for league, slug in sf.ESPN_SLUGS.items():
            if f"/soccer/{slug}/scoreboard?dates=" in url:
                path = FEEDS / f"espn_{league}_board_{url.rsplit('=', 1)[1]}.json.gz"
                return gzip.open(path).read() if path.exists() else None
        for league, slug in sf.FD_SLUGS.items():
            if url.startswith(f"https://fixturedownload.com/feed/json/{slug}-"):
                return gzip.open(FEEDS / f"fixturedownload_{league}_{url.rsplit('-', 1)[1]}.json.gz").read()
        raise ConnectionError(url)
    return get


# A real Premier League event (Crystal Palace 0-2 Newcastle, 2026-10-25), the
# template every made-up event below is copied from.
TEMPLATE = next(e for e in sample("espn_E0_board_20261025")["events"] if e["id"] == "401879254")
SCHEDULED = {"completed": False, "description": "Scheduled", "id": "1", "name": "STATUS_SCHEDULED",
             "state": "pre"}
FULL_TIME = {"completed": True, "description": "Full Time", "id": "28", "name": "STATUS_FULL_TIME",
             "state": "post"}
POSTPONED = {"completed": False, "description": "Postponed", "id": "6", "name": "STATUS_POSTPONED",
             "state": "post"}


def event(eid="9001", home="359", away="363", when="2026-10-31T15:00Z", kind=SCHEDULED, scores=("0", "0"),
          valid=True, names=None):
    """ESPN event: Arsenal (359) v Chelsea (363) unless told otherwise."""
    ev = copy.deepcopy(TEMPLATE)
    ev["id"], ev["date"] = eid, when
    comp = ev["competitions"][0]
    comp["id"], comp["date"], comp["startDate"], comp["timeValid"] = eid, when, when, valid
    for c in comp["competitors"]:
        tid = home if c["homeAway"] == "home" else away
        c["id"] = c["team"]["id"] = tid
        c["team"]["displayName"] = (names or {}).get(tid) or sf.ESPN_CLUBS["E0"].get(tid, ("Nobody FC",))[0]
        c["score"] = scores[0] if c["homeAway"] == "home" else scores[1]
    comp["status"]["type"] = ev["status"]["type"] = dict(kind)
    return ev


def board(*events):
    return {"events": list(events), "leagues": [], "provider": {}}


def fd(home="Arsenal", away="Chelsea", when="2026-10-31 15:00:00Z", score=(None, None), number=84, rnd=9):
    return {"MatchNumber": number, "RoundNumber": rnd, "DateUtc": when, "Location": "", "HomeTeam": home,
            "AwayTeam": away, "Group": None, "HomeTeamScore": score[0], "AwayTeamScore": score[1],
            "Winner": None}


def one_espn(*events, log=None):
    return sf.parse_espn(board(*events), "E0", request="espn:E0:test", seen_at="2026-10-26T13:00:00Z",
                         log=(log if log is not None else []).append)


def one_fd(*rows, log=None, league="E0"):
    return sf.parse_fixturedownload(list(rows), league, 2026, request=f"fixturedownload:{league}:2026",
                                    seen_at="2026-10-26T13:00:00Z", log=(log if log is not None else []).append)


def only(matches):
    assert len(matches) == 1, matches
    return matches[0]


# ------------------------------------------------------------- the real samples

def test_every_saved_scoreboard_parses_cleanly(espn):
    rows, warnings = espn
    assert not warnings
    events = sum(len(p["events"]) for _, _, p in boards("espn_*_board_2*.json.gz"))
    assert len(rows) == events > 100
    for league, key, payload in boards():                       # the default boards too
        names = {e["competitions"][0]["status"]["type"]["name"] for e in payload["events"]}
        assert names <= set(sf.ESPN_STATUS), names
        assert len(sf.parse_espn(payload, league, log=warnings.append)) == len(payload["events"])
    assert not warnings
    assert {r["status"] for r in rows} <= {"scheduled", "final"}
    for r in rows:
        assert set(r) == set(sf.ROW_FIELDS)
        assert r["time_confirmed"] and r["kickoff_utc"].endswith(":00Z")
        assert r["uk_date"] == sf.uk_date(sf.parse_utc(r["kickoff_utc"])).isoformat() == r["date_hint"]
        assert r["match_id"] == sd.match_id(r["league"], pd.Timestamp(r["uk_date"]).date(), r["home"], r["away"])
        assert (r["hg"] is None) == (r["status"] != "final")
        assert r["season"] == 2026


def test_the_round_just_played_reads_as_final_with_its_score(espn):
    rows, _ = espn
    palace = only([r for r in rows if r["source_id"] == "401879254"])
    assert palace == dict(palace, league="E0", season=2026, home="Crystal Palace", away="Newcastle",
                          home_name="Crystal Palace", away_name="Newcastle United",
                          kickoff_utc="2026-10-25T14:00:00Z", uk_date="2026-10-25",
                          match_id="E0-20261025-crystalpalace-newcastle", status="final",
                          status_detail="STATUS_FULL_TIME", hg=0, ag=2, request="espn:E0:20261025")
    clasico = only([r for r in rows if r["league"] == "SP1" and r["home"] == "Barcelona"
                    and r["away"] == "Real Madrid"])
    assert (clasico["kickoff_utc"], clasico["match_id"], clasico["hg"], clasico["ag"]) == \
        ("2026-10-25T20:00:00Z", "SP1-20261025-barcelona-realmadrid", 1, 2)


def test_espn_ids_and_kickoffs_are_football_datas_across_the_clock_change(espn):
    """football-data.co.uk's own fixtures for Oct 23-26 (soccer/data/fixtures.csv
    as fetched 2026-10-25, frozen here): the UK clock went back on the 25th,
    and every ESPN kickoff and id equals theirs on both sides of it."""
    rows, _ = espn
    frozen = pd.read_csv(FIXTURES / "football_data_fixtures_20261023.csv")
    frozen["kickoff"] = pd.to_datetime(frozen["kickoff"])
    theirs = by_pairing(frozen)
    paired = [(r, theirs[(r["league"], r["season"], r["home"], r["away"])]) for r in rows
              if (r["league"], r["season"], r["home"], r["away"]) in theirs]
    assert len(paired) == len(frozen) == 48
    for r, g in paired:
        assert r["match_id"] == g.match_id
        assert pd.Timestamp(r["kickoff_utc"]).tz_localize(None) == g.kickoff
    times = {(g.date, g.time, r["kickoff_utc"]) for r, g in paired}
    assert ("2026-10-24", "15:00", "2026-10-24T14:00:00Z") in times      # British Summer Time
    assert ("2026-10-25", "14:00", "2026-10-25T14:00:00Z") in times      # Greenwich Mean Time


def test_fixturedownload_agrees_with_football_data_on_every_match_already_played(football_data):
    theirs = by_pairing(football_data)
    # 2026-27 to football-data.co.uk's last result at the probe (Oct 21):
    # every played match's id, kickoff and score.
    played = [r for r in fd_rows(2026) if r["status"] == "final" and r["uk_date"] <= "2026-10-21"]
    paired = [(r, theirs[(r["league"], r["season"], r["home"], r["away"])]) for r in played]
    assert len(paired) == 346
    for r, g in paired:
        assert r["match_id"] == g.match_id
        assert pd.Timestamp(r["kickoff_utc"]).tz_localize(None) == g.kickoff
        assert (r["hg"], r["ag"]) == (int(g.hg), int(g.ag))
    # 2025-26, the whole season: every score equal; the date one day apart on
    # one match (La Liga, Mallorca v Valencia) and the kickoff on nine - which
    # is why a fixturedownload time is never confirmed on its own.
    rows = fd_rows(2025)
    assert len(rows) == 1752
    paired = [(r, theirs[(r["league"], r["season"], r["home"], r["away"])]) for r in rows]
    assert all((r["hg"], r["ag"]) == (int(g.hg), int(g.ag)) for r, g in paired if r["status"] == "final")
    assert sum(r["status"] == "final" for r in rows) == 1751           # one Ligue 1 score never posted
    assert sum(r["match_id"] != g.match_id for r, g in paired) <= 1
    assert sum(pd.Timestamp(r["kickoff_utc"]).tz_localize(None) != g.kickoff for r, g in paired) <= 9


@pytest.mark.snapshot
def test_the_provisional_results_are_football_datas_once_it_posts_them(espn, football_data):
    """The round the feeds had first (Oct 23-26): once football-data.co.uk
    posts it, every provisional result must be its result, under its id.
    Depends on its publishing clock, so reported, never a gate."""
    rows, _ = espn
    matches = sf.reconcile(rows, fd_rows(2026), log=lambda *_: None)
    theirs = by_pairing(football_data[football_data["result"].notna()])
    landed = [(m, theirs[k]) for m in matches if m["status"] == "final"
              and (k := (m["league"], m["season"], m["home"], m["away"])) in theirs]
    for m, g in landed:
        assert (m["match_id"], m["hg"], m["ag"], m["result"]) == (g.match_id, int(g.hg), int(g.ag), g.result)


@AS_PROBED
def test_both_feeds_agree_on_every_match_espn_was_asked_about(espn):
    rows, _ = espn
    fd2026 = fd_rows(2026)
    matches = {(m["league"], m["home"], m["away"]): m
               for m in sf.reconcile(rows, fd2026, log=lambda *_: None)}
    covered = [matches[(r["league"], r["home"], r["away"])] for r in rows]
    assert len(covered) == 130
    for m in covered:
        assert m["sources"] == ["espn", "fixturedownload"]
        assert m["kickoff_espn"] == m["kickoff_fd"] == m["kickoff_utc"] and m["time_confirmed"]
        assert m["status"] in ("scheduled", "final") and not m["warnings"]
    finals = [m for m in covered if m["status"] == "final"]
    assert len(finals) == 46
    assert all(m["result_sources"] == ["espn", "fixturedownload"] and m["provisional"] for m in finals)


def test_every_club_in_the_samples_maps_one_to_one_to_football_datas(football_data):
    fixtures = sd.load_fixtures()
    for league in sd.TOP:
        espn_table = sf.ESPN_CLUBS[league]
        assert len({club for _, club in espn_table.values()}) == len(espn_table), league
        teams = sample(f"espn_{league}_teams")["sports"][0]["leagues"][0]
        season = int(teams["season"]["year"])
        listed = {t["team"]["id"]: t["team"]["displayName"] for t in teams["teams"]}
        on_boards = {c["team"]["id"]: c["team"]["displayName"]
                     for lg, _, payload in boards() if lg == league
                     for e in payload["events"] for c in e["competitions"][0]["competitors"]}
        for tid, name in {**listed, **on_boards}.items():
            assert tid in espn_table and espn_table[tid][0] == name, (league, tid, name)
        theirs = football_data[(football_data["league"] == league) & (football_data["season"] == season)]
        clubs = set(theirs["home"]) | set(theirs["away"])
        clubs |= set(fixtures.loc[(fixtures["league"] == league) & (fixtures["season"] == season), "home"])
        assert {espn_table[t][1] for t in listed} == clubs, league

        for year in (2025, 2026):
            names = {r[k] for r in sample(f"fixturedownload_{league}_{year}") for k in ("HomeTeam", "AwayTeam")}
            mapped = {sf.FD_CLUBS[league][sf._nfc(n)] for n in names}
            assert len(mapped) == len(names), (league, year)             # one to one within a season
            theirs = football_data[(football_data["league"] == league) & (football_data["season"] == year)]
            assert mapped == set(theirs["home"]) | set(theirs["away"]), (league, year)
        # every club the tables name has a badge in the app
        for club in {c for _, c in espn_table.values()} | set(sf.FD_CLUBS[league].values()):
            assert club in st.TEAMS, club


def test_a_fixturedownload_placeholder_has_no_kickoff_and_no_date():
    # No real kickoff in a complete season is ever taken for a placeholder.
    for league in sd.TOP:
        for raw in sample(f"fixturedownload_{league}_2025"):
            assert not sf.is_placeholder(sf.parse_utc(raw["DateUtc"]), league), raw
    # Real rows from the 2026-27 feeds: La Liga's 00:00 UTC on a nominal
    # Sunday, Ligue 1's local midnight (23:00 UTC the Friday before), and the
    # Premier League's Saturday 15:00, which is also a real kickoff time.
    rows = one_fd(fd("Sevilla FC", "Getafe CF", "2026-11-29 00:00:00Z", number=136, rnd=14), league="SP1")
    rows += one_fd(fd("Stade Rennais FC", "Paris Saint-Germain", "2027-03-05 23:00:00Z", number=205, rnd=23),
                   league="F1")
    for r in rows:
        assert (r["kickoff_utc"], r["uk_date"], r["match_id"], r["time_confirmed"]) == (None, None, None, False)
    assert [r["date_hint"] for r in rows] == ["2026-11-29", "2027-03-06"]
    epl = only(one_fd(fd("Bournemouth", "Leeds", "2026-10-31 15:00:00Z", number=81)))
    assert (epl["kickoff_utc"], epl["uk_date"], epl["time_confirmed"]) == ("2026-10-31T15:00:00Z", "2026-10-31", False)


@AS_PROBED
def test_the_placeholders_in_the_samples_are_whole_unscheduled_rounds():
    counts = {}
    for league in sd.TOP:
        rows = one_fd(*sample(f"fixturedownload_{league}_2026"), league=league)
        holders = [r for r in rows if r["kickoff_utc"] is None]
        counts[league] = len(holders)
        assert all(r["status"] == "scheduled" for r in holders)
        rounds = {r["round"] for r in holders}
        assert not any(r["kickoff_utc"] for r in rows if r["round"] in rounds), league
    assert counts == {"E0": 0, "SP1": 250, "I1": 190, "D1": 198, "F1": 144}


def test_a_pairing_fixturedownload_lists_twice_is_warned_and_the_played_one_used():
    # Real: 2026-27 Ligue 1 has Rennes v Paris SG twice (rounds 1 and 23) and
    # Paris SG v Rennes never - home and away swapped in round 23.
    warnings = []
    rows = one_fd(fd("Stade Rennais FC", "Paris Saint-Germain", "2026-08-23 18:45:00Z", (2, 2), 7, 1),
                  fd("Stade Rennais FC", "Paris Saint-Germain", "2027-03-05 23:00:00Z", (None, None), 205, 23),
                  league="F1", log=warnings)
    assert any("listed 2 times" in w for w in warnings)
    m = only(sf.reconcile([], rows, log=warnings.append))
    assert m["fd_id"] == "2026-7" and m["score_fd"] == [2, 2] and m["match_id"] == "F1-20260823-rennes-parissg"
    assert any("2026-205" in w and "the one played" in w for w in m["warnings"])


# ------------------------------------------------------------- made-up edge cases

def test_the_uk_date_follows_the_uk_clock_through_both_changes():
    cases = {"2026-03-28T15:00Z": "2026-03-28",    # GMT
             "2026-03-29T14:00Z": "2026-03-29",    # BST from 01:00 UTC that day: 15:00 UK
             "2026-10-24T14:00Z": "2026-10-24",    # 15:00 BST
             "2026-10-25T14:00Z": "2026-10-25",    # GMT again: 14:00 UK
             "2026-08-15T23:30Z": "2026-08-16",    # 00:30 BST - the UK day, not the UTC day
             "2026-11-14T23:30Z": "2026-11-14"}
    for text, day in cases.items():
        moment = sf.parse_utc(text)
        assert sf.uk_date(moment).isoformat() == day
        # football-data.co.uk's reading of that UK date and clock is the same moment
        clock = moment.astimezone(sd.UK).strftime("%H:%M")
        assert sd.kickoff_utc(sf.uk_date(moment), clock) == pd.Timestamp(moment).tz_localize(None)
    late = only(one_espn(event(when="2026-08-15T23:30Z")))
    assert (late["uk_date"], late["match_id"]) == ("2026-08-16", "E0-20260816-arsenal-chelsea")


def test_a_postponed_match_has_no_kickoff_and_no_result():
    espn_row = one_espn(event(kind=POSTPONED))
    fd_row = one_fd(fd())
    m = only(sf.reconcile(espn_row, fd_row, log=lambda *_: None))
    assert (m["status"], m["kickoff_utc"], m["result"]) == ("postponed", None, None)
    assert m["kickoff_espn"] == "2026-10-31T15:00:00Z" and m["match_id"] == "E0-20261031-arsenal-chelsea"
    assert not m["warnings"]
    # ESPN's postponement stands against a score from fixturedownload, warned;
    # and against fixturedownload having moved it to an unscheduled round.
    scored = only(sf.reconcile(espn_row, one_fd(fd(score=(1, 0))), log=lambda *_: None))
    assert scored["status"] == "postponed" and scored["result"] is None and scored["warnings"]
    moved = only(sf.reconcile(espn_row, one_fd(fd(when="2027-01-16 00:00:00Z")), log=lambda *_: None))
    assert moved["status"] == "postponed" and moved["kickoff_utc"] is None and moved["kickoff_fd"] is None


def test_a_result_needs_the_sources_not_to_contradict():
    final = one_espn(event(kind=FULL_TIME, scores=("2", "1")))
    agree = only(sf.reconcile(final, one_fd(fd(score=(2, 1))), log=lambda *_: None))
    assert (agree["status"], agree["hg"], agree["ag"], agree["result"]) == ("final", 2, 1, "H")
    assert agree["provisional"] and agree["result_sources"] == ["espn", "fixturedownload"]
    assert agree["result_seen_at"] == "2026-10-26T13:00:00Z" and not agree["warnings"]

    not_yet = only(sf.reconcile(final, one_fd(fd()), log=lambda *_: None))
    assert not_yet["status"] == "final" and not_yet["result_sources"] == ["espn"]
    espn_only = only(sf.reconcile(final, [], log=lambda *_: None))
    assert espn_only["status"] == "final" and espn_only["sources"] == ["espn"]

    warnings = []
    clash = only(sf.reconcile(final, one_fd(fd(score=(1, 1))), log=warnings.append))
    assert clash["status"] == "disputed" and clash["result"] is None and clash["hg"] is None
    assert (clash["score_espn"], clash["score_fd"]) == ([2, 1], [1, 1]) and warnings

    # A score only fixturedownload has is not a result: warned where ESPN
    # says otherwise, quiet where ESPN was not asked about the match.
    warnings = []
    stale = only(sf.reconcile(one_espn(event()), one_fd(fd(score=(2, 0))), log=warnings.append))
    assert stale["status"] == "unconfirmed" and stale["result"] is None and warnings
    quiet = []
    alone = only(sf.reconcile([], one_fd(fd(score=(2, 0))), log=quiet.append))
    assert alone["status"] == "unconfirmed" and alone["result"] is None and not quiet


def test_a_missing_time_is_missing():
    timeless = only(one_espn(event(valid=False)))
    assert (timeless["kickoff_utc"], timeless["uk_date"], timeless["match_id"], timeless["time_confirmed"]) == \
        (None, None, None, False)
    assert timeless["date_hint"] == "2026-10-31"
    # fixturedownload's time is used - and is not confirmed.
    m = only(sf.reconcile([timeless], one_fd(fd(when="2026-10-31 12:30:00Z")), log=lambda *_: None))
    assert (m["kickoff_utc"], m["time_confirmed"], m["match_id"]) == \
        ("2026-10-31T12:30:00Z", False, "E0-20261031-arsenal-chelsea")
    # Neither source has a time: no kickoff, no id, nothing to freeze a pick at.
    none = only(sf.reconcile([timeless], one_fd(fd(when="2026-10-31 00:00:00Z")), log=lambda *_: None))
    assert (none["kickoff_utc"], none["match_id"], none["status"], none["date_hint"]) == \
        (None, None, "scheduled", "2026-10-31")


def test_differing_kickoffs_keep_both_and_freeze_at_the_earlier():
    warnings = []
    m = only(sf.reconcile(one_espn(event(when="2026-10-31T15:00Z")), one_fd(fd(when="2026-10-31 12:30:00Z")),
                          log=warnings.append))
    assert (m["kickoff_utc"], m["kickoff_espn"], m["kickoff_fd"], m["time_confirmed"]) == \
        ("2026-10-31T12:30:00Z", "2026-10-31T15:00:00Z", "2026-10-31T12:30:00Z", False)
    assert warnings and "earlier" in m["warnings"][0]
    # Agreeing times are confirmed only with ESPN's timeValid.
    same = only(sf.reconcile(one_espn(event()), one_fd(fd()), log=lambda *_: None))
    assert same["time_confirmed"] and same["kickoff_utc"] == "2026-10-31T15:00:00Z"
    # ESPN's valid time against a fixturedownload placeholder: kept, not confirmed.
    unset = only(sf.reconcile(one_espn(event()), one_fd(fd(when="2026-10-31 00:00:00Z")), log=lambda *_: None))
    assert unset["kickoff_utc"] == "2026-10-31T15:00:00Z" and not unset["time_confirmed"]


def test_an_unmapped_club_is_skipped_with_a_warning():
    warnings = []
    assert one_espn(event(home="99999"), log=warnings) == []
    assert any("99999" in w and "Nobody FC" in w for w in warnings)
    warnings = []
    assert one_fd(fd(home="Arsenal FC"), log=warnings) == []
    assert any("'Arsenal FC'" in w for w in warnings)
    # Nothing is matched by likeness: a near name is still unmapped.
    assert sf.fd_club("E0", "Man United", log=lambda *_: None) is None
    # A known ESPN id under a new name is the same club, mapped by its id - and warned.
    warnings = []
    renamed = only(one_espn(event(names={"359": "Arsenal FC"}), log=warnings))
    assert renamed["home"] == "Arsenal" and renamed["home_name"] == "Arsenal FC" and warnings


def test_an_unknown_or_inconsistent_espn_status_is_never_final():
    warnings = []
    say = lambda **kw: sf.espn_status(kw, log=warnings.append)
    assert say(name="STATUS_SCHEDULED", state="pre", completed=False) == ("scheduled", "STATUS_SCHEDULED")
    assert say(name="STATUS_FULL_TIME", state="post", completed=True) == ("final", "STATUS_FULL_TIME")
    assert say(name="STATUS_POSTPONED", state="post", completed=False)[0] == "postponed"
    assert say(name="STATUS_CANCELED", state="post", completed=False)[0] == "cancelled"
    assert say(name="STATUS_ABANDONED", state="post", completed=False)[0] == "abandoned"
    assert say(name="STATUS_SECOND_HALF", state="in", completed=False)[0] == "live"
    assert [say(name="STATUS_DELAYED", state=s, completed=False)[0] for s in ("pre", "in")] == ["scheduled", "live"]
    assert not warnings
    assert say(name="STATUS_FULL_TIME", state="in", completed=False)[0] == "unknown"
    assert say(name="STATUS_FINAL_PEN", state="post", completed=True)[0] == "unknown"
    assert say(name="STATUS_SOMETHING_NEW", state="post", completed=True)[0] == "unknown"
    assert len(warnings) == 3
    odd = only(one_espn(event(kind=dict(FULL_TIME, name="STATUS_SOMETHING_NEW"), scores=("3", "0"))))
    assert odd["status"] == "unknown" and odd["hg"] is None
    m = only(sf.reconcile([odd], one_fd(fd(score=(3, 0))), log=lambda *_: None))
    assert m["status"] == "unconfirmed" and m["result"] is None


def test_espn_is_asked_one_day_at_a_time():
    reqs = sf.requests_for(NOW)
    espn = [r for r in reqs if r[3] == "espn"]
    assert len(espn) == 5 * 15 and len(reqs) == 5 * 16
    days = sorted({r[0].rsplit(":", 1)[1] for r in espn})
    assert days[0] == "20261022" and days[-1] == "20261105"
    for key, url, league, kind, season in reqs:
        if kind == "espn":
            assert url == ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                           f"{sf.ESPN_SLUGS[league]}/scoreboard?dates={key.rsplit(':', 1)[1]}")
        else:
            assert url == f"https://fixturedownload.com/feed/json/{sf.FD_SLUGS[league]}-2026" and season == 2026


def test_a_failed_request_keeps_what_is_on_file(tmp_path):
    path = tmp_path / "feed.json"
    calls = []
    first = sf.sync(NOW, path=path, get=sample_getter(calls), log=lambda *_: None)
    assert len(calls) == 80 and not first["last_run"]["failed"]
    assert sum(r["status"] == "final" for r in first["rows"] if r["source"] == "espn") == 46
    before = {sf.row_key(r): r for r in first["rows"]}

    good = sample_getter()

    def flaky(url):
        if "/eng.1/" in url:
            raise ConnectionError("timed out")
        if "la-liga" in url:
            return b"<html><body>Service unavailable</body></html>"
        if "serie-a" in url:
            return b"[]"                                      # parses, but no rows: a format change
        if "ger.1" in url and url.endswith("20261025"):
            return b'{"error": "rate limited"}'
        return good(url)

    later = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
    lines = []
    second = sf.sync(later, path=path, get=flaky, log=lines.append)
    failed = set(second["last_run"]["failed"])
    every_e0_day = {key for key, *_ in sf.requests_for(later) if key.startswith("espn:E0:")}
    assert failed == every_e0_day | {"fixturedownload:SP1:2026", "fixturedownload:I1:2026", "espn:D1:20261025"}
    assert any("kept what is on file" in line for line in lines)
    after = {sf.row_key(r): r for r in second["rows"]}
    assert after == before                                     # nothing lost, nothing restamped
    assert second["requests"]["espn:E0:20261025"] == "2026-10-26T13:13:23Z"
    assert second["requests"]["espn:F1:20261025"] == "2026-10-26T18:00:00Z"
    assert sf.load(path) == second


def test_an_outage_is_cut_short_and_wipes_nothing(tmp_path):
    """Every request timing out: three tries per league and feed, not 80
    thirty-second timeouts; nothing on file is lost. A slow fetch stops
    starting requests once past its budget."""
    path = tmp_path / "feed.json"
    first = sf.sync(NOW, path=path, get=sample_getter(), log=lambda *_: None)
    calls = []

    def down(url):
        calls.append(url)
        raise ConnectionError("timed out")

    later = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
    lines = []
    second = sf.sync(later, path=path, get=down, log=lines.append)
    assert len(calls) == 5 * (sf.STRAIGHT_FAILS + 1)            # 3 scoreboard days and the season file, per league
    assert set(second["last_run"]["failed"]) == {k for k, *_ in sf.requests_for(later)}
    assert second["last_run"]["answered"] == 0
    assert second["rows"] == first["rows"] and second["requests"] == first["requests"]
    assert sum("failures in a row" in line for line in lines) == 5

    tick = iter(range(0, 10_000, 100))                          # 0 at the start, then 100 s a request
    got = sf.fetch(later, get=sample_getter(), log=lambda *_: None, clock=lambda: next(tick), budget=250)
    assert len(got["ok"]) == 2 and len(got["failed"]) == 78
    assert all(why.startswith("skipped: the fetch ran past") for why in got["failed"].values())


def test_a_refetch_moves_a_moved_match_and_never_drops_a_result():
    played = one_espn(event(eid="1", kind=FULL_TIME, scores=("1", "0"), when="2026-10-25T14:00Z"))
    upcoming = one_espn(event(eid="2", home="370", away="364", when="2026-10-31T15:00Z"))
    for r in played:
        r["request"] = "espn:E0:20261025"
    for r in upcoming:
        r["request"] = "espn:E0:20261031"
    old = played + upcoming
    # The 25th's board now answers empty; the 31st no longer lists event 2,
    # which has moved to Sunday the 1st, 16:30.
    moved = one_espn(event(eid="2", home="370", away="364", when="2026-11-01T16:30Z"))
    for r in moved:
        r["request"], r["seen_at"] = "espn:E0:20261101", "2026-10-27T09:00:00Z"
    lines = []
    rows = sf.merge(old, {"at": "2026-10-27T09:00:00Z", "rows": moved,
                          "ok": {"espn:E0:20261025": 0, "espn:E0:20261031": 0, "espn:E0:20261101": 1}},
                    log=lines.append)
    assert [(r["source_id"], r["kickoff_utc"], r["seen_at"]) for r in rows] == [
        ("1", "2026-10-25T14:00:00Z", "2026-10-26T13:00:00Z"),          # the result stays
        ("2", "2026-11-01T16:30:00Z", "2026-10-27T09:00:00Z")]          # moved, once, newly seen
    assert any("result is kept" in line for line in lines)
    # Unchanged on a later fetch: it keeps the moment it was first seen.
    again = [dict(r, seen_at="2026-10-28T09:00:00Z") for r in moved]
    rows = sf.merge(rows, {"at": "2026-10-28T09:00:00Z", "rows": again, "ok": {"espn:E0:20261101": 1}})
    assert rows[1]["seen_at"] == "2026-10-27T09:00:00Z"


def test_the_snapshot_is_stable_sorted_and_has_no_nan(tmp_path):
    rows = one_espn(event(eid="2", when="2026-11-01T16:30Z"), event(eid="1", home="370", away="364"))
    rows[0]["hg"] = float("nan")
    snap = {"schema": sf.SCHEMA, "requests": {"espn:E0:test": "2026-10-26T13:00:00Z"}, "last_run": None,
            "rows": rows}
    path = sf.save(snap, tmp_path / "feed.json")
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and json.loads(text)["rows"][1]["hg"] is None
    assert [r["source_id"] for r in json.loads(text)["rows"]] == ["1", "2"]       # by date
    assert len(text.splitlines()) == 1 + 3 + 1 + 2 + 2           # {, the header, "rows": [, a line a row, ], }
    sf.save(sf.load(path), path)
    assert path.read_text(encoding="utf-8") == text
    assert sf.load(tmp_path / "missing.json") == sf.empty()
    assert json.loads(sf.dumps(sf.empty()))["rows"] == []
