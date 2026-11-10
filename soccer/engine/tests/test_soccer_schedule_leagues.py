"""soccer_schedule beyond the top five: the leagues the run carries, and the
leagues where one pairing is played more than once a season
(soccer_feed.REPEATS).

Two rules are checked here. The leagues passed are the leagues joined: a
feed match or a fixture of any other league (RUS, out of the app) never
reaches the frame, and told nothing the schedule is the five's, as before.
And where pairings repeat, a match is its pairing AND its day: a meeting
played earlier in the season does not make the next one "played", two
meetings in the feeds' window are two matches, a football-data.co.uk
fixture joins the feed match nearest its date within PAIR_DAYS, and a
status, a postponement or a result belongs to its own meeting only. The
rows are made up in the feeds' and football-data.co.uk's own schemas; the
last test runs the league probe's real week."""

from collections import Counter

import pandas as pd

import soccer_data as sd
import soccer_feed as sfeed
import soccer_leagues as slg
import soccer_schedule as ssch

SILENT = lambda *_: None                     # noqa: E731
NOW = pd.Timestamp("2026-11-06 12:00")
SEEN = "2026-11-06T11:00:00Z"
DETAIL = {"scheduled": "STATUS_SCHEDULED", "final": "STATUS_FULL_TIME", "postponed": "STATUS_POSTPONED"}


def _when(kickoff):
    k = pd.Timestamp(kickoff)
    return k, f"{k:%Y-%m-%dT%H:%M:%SZ}", k.tz_localize("UTC").tz_convert(sd.UK).date().isoformat()


def espn(league, home, away, kickoff, status="scheduled", score=None, eid=None):
    """An ESPN scoreboard row as soccer_feed.parse_espn writes it (a valid time)."""
    k, iso, day = _when(kickoff)
    final = status == "final"
    return sfeed._row(source="espn", source_id=eid or f"{home}-{away}-{day}", request=f"espn:{league}:{k:%Y%m%d}",
                      league=league, season=sd.season_for(league, day), home=home, away=away, home_name=home,
                      away_name=away, kickoff_utc=iso, time_confirmed=True, uk_date=day, date_hint=day,
                      status=status, status_detail=DETAIL[status], hg=score[0] if final else None,
                      ag=score[1] if final else None, seen_at=SEEN)


def feed(*rows):
    return sfeed.reconcile([r for r in rows if r["source"] == "espn"],
                           [r for r in rows if r["source"] == "fixturedownload"], log=SILENT)


def _row(league, home, away, kickoff, **kw):
    k, _, day = _when(kickoff)
    row = {"match_id": sd.match_id(league, pd.Timestamp(day).date(), home, away), "league": league,
           "season": sd.season_for(league, day), "date": day,
           "time": k.tz_localize("UTC").tz_convert(sd.UK).strftime("%H:%M"), "kickoff": k, "home": home,
           "away": away}
    row.update(kw)
    return row


def result(league, home, away, kickoff, hg, ag):
    """A football-data.co.uk result."""
    return _row(league, home, away, kickoff, hg=hg, ag=ag, result="H" if hg > ag else "A" if hg < ag else "D",
                avg_h=2.1, avg_d=3.4, avg_a=3.5)


def fixture(league, home, away, kickoff):
    """A football-data.co.uk fixture, priced."""
    return _row(league, home, away, kickoff, avg_h=1.9, avg_d=3.5, avg_a=4.2, max_h=2.0, max_d=3.7, max_a=4.5)


def run(results, fixtures, rows, now=NOW, log=SILENT, **kw):
    fx = sd.frame(fixtures)
    fx["fetched_at"] = pd.Timestamp("2026-11-06 10:00")
    return ssch.schedule(sd.frame(results), fx, feed(*rows), now, log=log, **kw)


def to_play(s):
    return sorted(s.frame.loc[s.frame["result"].isna(), "match_id"])


def K(day, league="SC0", home="Rangers", away="Celtic", season=2026):
    return ssch.key(league, season, home, away, day)


# ------------------------------------------------------------- the leagues passed

def test_the_leagues_passed_are_the_leagues_joined_and_rus_never_is():
    """Told the app's leagues (in_app()), the feeds' matches and the fixtures
    of every other league are left out - RUS's above all: it has a feed and
    is out of the app, so a feed-only RUS match would otherwise be picked by
    the model alone, unpriced. Told nothing: the feeds' five and every
    fixture given, as before."""
    rows = [espn("E0", "Arsenal", "Chelsea", "2026-11-08 15:00"), espn("B1", "Club Brugge", "Anderlecht",
                                                                     "2026-11-08 13:00"),
            espn("RUS", "Spartak Moscow", "CSKA Moscow", "2026-11-08 13:00")]
    fx = [fixture("RUS", "Zenit", "Krasnodar", "2026-11-09 16:00"), fixture("E0", "Leeds", "Stoke", "2026-11-09 16:00")]
    default = run([], fx, rows)
    assert to_play(default) == ["E0-20261108-arsenal-chelsea", "E0-20261109-leeds-stoke",
                                "RUS-20261109-zenit-krasnodar"]
    assert set(default.status) == {("E0", 2026, "Arsenal", "Chelsea")}
    lines = []
    app = run([], fx, rows, leagues=slg.in_app(), log=lines.append)
    assert to_play(app) == ["B1-20261108-clubbrugge-anderlecht", "E0-20261108-arsenal-chelsea",
                            "E0-20261109-leeds-stoke"]
    assert app.info["B1-20261108-clubbrugge-anderlecht"]["source"] == ssch.FEED_SOURCE
    assert not any(k[0] == "RUS" for k in app.status) and not any("RUS" in m for m in app.info)
    assert any("1 match of leagues the app does not carry left out" in line for line in lines)
    # RUS's results stay in the frame: the model learns the extra leagues' regression from them.
    res = [result("RUS", "Zenit", "Krasnodar", "2026-08-02 16:00", 1, 0)]
    assert "RUS" in set(run(res, fx, rows, leagues=slg.in_app()).frame["league"])


def test_the_five_alone_are_as_before_whatever_the_leagues_argument_says():
    """Top-five matches only: the Schedule told nothing, told the five and
    told the app's leagues is the same in every field (the five's run, bit
    for bit) - other leagues' feed matches only ever drop out."""
    res = [result("E0", "Arsenal", "Chelsea", "2025-10-04 14:00", 1, 0),
           result("SP1", "Sevilla", "Betis", "2026-10-31 20:00", 2, 2)]
    fx = [fixture("E0", "Leeds", "Stoke", "2026-11-09 16:00"), fixture("SP1", "Getafe", "Girona", "2026-11-04 20:00"),
          fixture("D1", "Mainz", "Koln", "2026-11-07 14:30")]
    rows = [espn("E0", "Arsenal", "Chelsea", "2026-11-08 15:00"), espn("E0", "Leeds", "Stoke", "2026-11-09 15:00"),
            espn("SP1", "Sevilla", "Betis", "2026-10-31 20:00", status="final", score=(2, 2)),
            espn("I1", "Roma", "Lazio", "2026-11-01 19:45", status="final", score=(1, 0)),
            espn("D1", "Mainz", "Koln", "2026-11-07 14:30", status="postponed"),
            espn("SC0", "Rangers", "Celtic", "2026-11-08 12:00"), espn("USA", "Austin", "Seattle", "2026-11-08 01:00")]
    told = [run(res, fx, rows), run(res, fx, rows, leagues=sd.TOP), run(res, fx, rows, leagues=list(reversed(sd.TOP))),
            run(res, fx, rows, leagues=[c for c in slg.in_app() if c in sd.TOP])]
    first = told[0]
    assert to_play(first) == ["E0-20261108-arsenal-chelsea", "E0-20261109-leeds-stoke"] and first.blocked
    for other in told[1:]:
        pd.testing.assert_frame_equal(first.frame, other.frame)
        pd.testing.assert_frame_equal(first.provisional, other.provisional)
        for f in ("info", "off", "tbc", "disputed", "status", "feed_kickoff", "blocked", "disputes"):
            assert getattr(first, f) == getattr(other, f), f
    assert all(len(k) == 4 for k in first.status)                  # the five's keys: the pairing, as before


# ------------------------------------------------------------- pairings that repeat

def test_where_pairings_repeat_a_meeting_played_earlier_does_not_drop_the_next():
    """A fixture whose pairing already has a result this season: in a league
    where a pairing is played once that is football-data.co.uk listing a
    played match, and it is dropped (as before); where pairings repeat it is
    the next meeting, and stands (on 2026-11-06's files: ARG Argentinos Jrs v
    Huracan, FIN Gnistan v VPS, HJK v AC Oulu, KuPS v Inter Turku were lost
    this way). Only a result within PAIR_DAYS of it is the same match."""
    res = [result("SC0", "Rangers", "Celtic", "2026-09-20 12:00", 1, 1),
           result("FIN", "HJK", "AC Oulu", "2026-04-11 13:00", 2, 0),
           result("E1", "Watford", "Millwall", "2026-09-20 14:00", 0, 0)]
    fx = [fixture("SC0", "Rangers", "Celtic", "2026-11-08 12:00"), fixture("FIN", "HJK", "AC Oulu", "2026-11-08 13:00"),
          fixture("E1", "Watford", "Millwall", "2026-11-08 15:00")]
    s = run(res, fx, [], leagues=slg.in_app())
    assert to_play(s) == ["FIN-20261108-hjk-acoulu", "SC0-20261108-rangers-celtic"]
    # The same match, its result dated a day before the fixtures file's date: posted, dropped.
    early = res + [result("SC0", "Rangers", "Celtic", "2026-11-07 12:00", 2, 0)]
    assert to_play(run(early, fx, [], now=pd.Timestamp("2026-11-07 18:00"), leagues=slg.in_app())) == \
        ["FIN-20261108-hjk-acoulu"]


def test_where_pairings_repeat_each_meeting_is_its_own_match_and_a_fixture_joins_the_nearest():
    """Rangers v Celtic twice in the window (the split's extra round), and
    once in September. football-data.co.uk's fixture of Nov 8 joins ESPN's
    event of Nov 9 (a day apart: the same match; the earlier time closes
    the pick); ESPN's Nov 15 is the next meeting, added with no price. By
    the pairing alone they would be one match, and the fixture lost."""
    res = [result("SC0", "Rangers", "Celtic", "2026-09-20 12:00", 1, 1)]
    fx = [fixture("SC0", "Rangers", "Celtic", "2026-11-08 12:00")]
    rows = [espn("SC0", "Rangers", "Celtic", "2026-11-09 12:00", eid="1"),
            espn("SC0", "Rangers", "Celtic", "2026-11-15 15:00", eid="2")]
    s = run(res, fx, rows, leagues=slg.in_app())
    assert to_play(s) == ["SC0-20261108-rangers-celtic", "SC0-20261115-rangers-celtic"]
    fd = s.info["SC0-20261108-rangers-celtic"]
    assert fd["source"] == ssch.FD_SOURCE and fd["feed_id"] == "SC0-20261109-rangers-celtic"
    assert fd["freeze_at"] == pd.Timestamp("2026-11-08 12:00")
    assert fd["kickoff_sources"] == {ssch.FD_SOURCE: pd.Timestamp("2026-11-08 12:00"),
                                     "ESPN": pd.Timestamp("2026-11-09 12:00")}
    nxt = s.info["SC0-20261115-rangers-celtic"]
    assert nxt["source"] == ssch.FEED_SOURCE and nxt["moved_from"] is None
    assert nxt["freeze_at"] == pd.Timestamp("2026-11-15 00:00")          # ESPN alone: not confirmed, 00:00 UK
    # Each meeting under its own key, and the fixture's own day beside its feed match's.
    assert s.status == {K("2026-11-09"): "scheduled", K("2026-11-15"): "scheduled", K("2026-11-08"): "scheduled"}
    assert s.feed_kickoff[K("2026-11-08")] == s.feed_kickoff[K("2026-11-09")] == pd.Timestamp("2026-11-09 12:00")
    assert ("SC0", 2026, "Rangers", "Celtic") not in s.status        # the pairing alone names no match here
    # find_key: the meeting nearest within PAIR_DAYS, never further.
    assert ssch.find_key(s.status, K("2026-11-10")) == K("2026-11-09")
    assert ssch.find_key(s.status, K("2026-11-13")) == K("2026-11-15")
    assert ssch.find_key(s.status, K("2026-11-19")) is None
    pick = {"league": "SC0", "season": 2026, "home": "Rangers", "away": "Celtic", "kickoff": "2026-11-08T12:00:00Z"}
    assert ssch.key_of(pick) == K("2026-11-08") and ssch.lookup(s.feed_kickoff, ssch.key_of(pick)) == \
        pd.Timestamp("2026-11-09 12:00")


def test_where_pairings_repeat_a_postponement_or_a_result_is_that_meetings_alone():
    """Three meetings: September's posted by football-data.co.uk, Nov 1's
    final on the feeds only, Nov 8's postponed. Nov 1's is a provisional
    result (September's result does not make it posted); the postponement
    blocks Nov 8's fixture and nothing else; Nov 15's is still to play, and
    the tables' frame keeps it beside Nov 1's result. Where a pairing is
    played once, a feed final of a pairing football-data.co.uk has posted is
    no result, as before - and where pairings repeat, so is a feed final a
    day from football-data.co.uk's own result of it (dated apart)."""
    res = [result("SC0", "Rangers", "Celtic", "2026-09-20 12:00", 1, 1),
           result("SC0", "Hearts", "Hibernian", "2026-11-01 19:45", 0, 1),
           result("E1", "Watford", "Millwall", "2026-09-20 14:00", 0, 0)]
    rows = [espn("SC0", "Rangers", "Celtic", "2026-11-01 12:00", status="final", score=(2, 1), eid="1"),
            espn("SC0", "Rangers", "Celtic", "2026-11-08 12:00", status="postponed", eid="2"),
            espn("SC0", "Rangers", "Celtic", "2026-11-15 15:00", eid="3"),
            espn("SC0", "Hearts", "Hibernian", "2026-11-02 00:15", status="final", score=(0, 1)),
            espn("E1", "Watford", "Millwall", "2026-11-01 15:00", status="final", score=(3, 0))]
    fx = [fixture("SC0", "Rangers", "Celtic", "2026-11-08 12:00")]
    s = run(res, fx, rows, leagues=slg.in_app())
    assert list(s.provisional["match_id"]) == ["SC0-20261101-rangers-celtic"]
    assert (s.provisional.at[0, "hg"], s.provisional.at[0, "ag"]) == (2, 1)
    assert s.blocked == ["SC0-20261108-rangers-celtic"]
    assert [(o["match_id"], o["status"], o["key"]) for o in s.off] == \
        [("SC0-20261108-rangers-celtic", "postponed", K("2026-11-08"))]
    assert to_play(s) == ["SC0-20261115-rangers-celtic"]
    assert {k: v for k, v in s.status.items() if k[2] == "Rangers"} == \
        {K("2026-11-01"): "final", K("2026-11-08"): "postponed", K("2026-11-15"): "scheduled"}
    shown = ssch.shown(s.frame, s.provisional)
    rc = shown[(shown["league"] == "SC0") & (shown["home"] == "Rangers")]
    assert sorted(zip(rc["date"], rc["result"].fillna("-"), rc["provisional"])) == \
        [("2026-09-20", "D", False), ("2026-11-01", "H", True), ("2026-11-15", "-", False)]
    # A provisional result replaces the row of its own meeting only (a day apart: the same match).
    frame = pd.concat([s.frame, sd.frame([fixture("SC0", "Rangers", "Celtic", "2026-11-02 12:00")])],
                      ignore_index=True)
    rc = ssch.shown(frame, s.provisional)
    assert sorted(rc.loc[(rc["home"] == "Rangers") & rc["result"].isna(), "date"]) == ["2026-11-15"]


def test_where_pairings_repeat_a_move_is_seen_within_pair_days_only():
    """football-data.co.uk's time passed with no result while ESPN lists the
    match later: within PAIR_DAYS that is the match moved (ESPN's date
    stands, moved_from says from when); a week on it may be the pairing's
    next meeting, so it stands as a match of its own and the passed fixture
    goes like any other."""
    fx = [fixture("SC0", "Rangers", "Celtic", "2026-11-04 19:45"), fixture("SC0", "Hearts", "Hibernian",
                                                                           "2026-11-04 19:45")]
    rows = [espn("SC0", "Rangers", "Celtic", "2026-11-07 15:00"),
            espn("SC0", "Hearts", "Hibernian", "2026-11-11 19:45")]
    s = run([], fx, rows, leagues=slg.in_app())
    assert to_play(s) == ["SC0-20261107-rangers-celtic", "SC0-20261111-hearts-hibernian"]
    moved = s.info["SC0-20261107-rangers-celtic"]
    assert moved["source"] == ssch.FEED_SOURCE and moved["moved_from"] == pd.Timestamp("2026-11-04 19:45")
    assert moved["kickoff_sources"] == {ssch.FD_SOURCE: pd.Timestamp("2026-11-04 19:45"),
                                        "ESPN": pd.Timestamp("2026-11-07 15:00")}
    assert s.info["SC0-20261111-hearts-hibernian"]["moved_from"] is None
    # A pick logged on football-data.co.uk's date finds the moved match's ESPN time.
    assert s.feed_kickoff[K("2026-11-04")] == pd.Timestamp("2026-11-07 15:00")
    assert ssch.lookup(s.feed_kickoff, K("2026-11-04", home="Hearts", away="Hibernian")) is None


def test_the_keys_and_the_lookups():
    """key(): the pairing where a pairing is played once, the pairing and its
    UK date where pairings repeat - the date of the kickoff on the UK clock,
    as a match_id has it (an MLS kickoff at 23:30 UTC in July is the next
    day in the UK). is_played reads a plain set of pairings the old way."""
    assert ssch.key("E0", 2026.0, "Arsenal", "Chelsea", "2026-11-08") == ("E0", 2026, "Arsenal", "Chelsea")
    assert ssch.key("USA", 2026, "Austin", "Seattle", pd.Timestamp("2026-07-12 18:00")) == \
        ("USA", 2026, "Austin", "Seattle", "2026-07-12")
    pick = {"league": "USA", "season": 2026, "home": "Austin", "away": "Seattle", "kickoff": "2026-07-11T23:30:00Z"}
    assert ssch.key_of(pick) == ("USA", 2026, "Austin", "Seattle", "2026-07-12")
    assert ssch.key_of(dict(pick, league="E0")) == ("E0", 2026, "Austin", "Seattle")
    assert ssch.find_key({("E0", 2026, "Arsenal", "Chelsea")}, ("E0", 2026, "Arsenal", "Chelsea")) is not None
    assert ssch.find_key({("E0", 2026, "Arsenal", "Chelsea")}, ("E0", 2026, "Chelsea", "Arsenal")) is None
    assert ssch.find_key({K("2026-11-08")}, K(None)) is None
    played = ssch.played_pairs(sd.frame([result("SC0", "Rangers", "Celtic", "2026-09-20 12:00", 1, 1)]))
    assert played == {("SC0", 2026, "Rangers", "Celtic"): ["2026-09-20"]}
    assert ssch.is_played(played, "SC0", 2026, "Rangers", "Celtic", "2026-09-22")
    assert not ssch.is_played(played, "SC0", 2026, "Rangers", "Celtic", "2026-11-08")
    assert not ssch.is_played(played, "SC0", 2026, "Rangers", "Celtic", None)
    assert ssch.is_played({("SC0", 2026, "Rangers", "Celtic")}, "SC0", 2026, "Rangers", "Celtic", "2026-11-08")


# ------------------------------------------------------------- the league probe's week

def test_the_probes_week_of_every_app_league_joins_each_match_once(tmp_path):
    """The league probe's real responses (2026-10-29), synced for the app's
    leagues beyond the five and joined to football-data.co.uk's files as the
    same probe downloaded them: no RUS anywhere, every match to play once,
    each feed match under its own key (none shares one: so no meeting of a
    repeated pairing is lost), every football-data.co.uk fixture in ESPN's
    days of a league ESPN covers joined to the feed match of its own id, and
    every feed match still to play in the window a match to play, once - in
    the leagues where pairings repeat too."""
    from test_soccer_feed_leagues import NOW as PROBED, football_data, probe_getter
    asked = [c for c in sfeed.leagues_for(slg.in_app()) if c not in sd.TOP]
    snap = sfeed.sync(PROBED, path=tmp_path / "feed.json", get=probe_getter(), leagues=asked, days_back=3,
                      log=SILENT)
    matches = sfeed.reconciled(snap, log=SILENT)
    keys = [ssch.feed_key(m) for m in matches]
    assert len(keys) == len(set(keys))
    results, fixtures = football_data()
    fixtures = fixtures.assign(fetched_at=pd.Timestamp("2026-10-29 12:00"))
    now = pd.Timestamp(PROBED).tz_convert("UTC").tz_localize(None)
    s = ssch.schedule(results, fixtures, matches, now, log=SILENT, leagues=slg.in_app())
    up = s.frame[s.frame["match_id"].isin(list(s.info))]                # the matches to play
    assert len(up) == len(s.info) and "RUS" not in set(up["league"])
    assert set(up["league"]) - set(sd.TOP) and set(up["league"]) & set(sfeed.REPEATS)
    fed = {c for c in asked if c in sfeed.ESPN_SLUGS}
    window = up[up["league"].isin(fed) & (up["kickoff"] < pd.Timestamp("2026-11-09 05:00"))]
    joined = [mid for mid in window["match_id"] if s.info[mid]["source"] == ssch.FD_SOURCE]
    assert joined and all(s.info[mid]["feed_id"] == mid for mid in joined)
    # Every feed match still to play in the window is a match to play, once -
    # in the leagues where pairings repeat as everywhere else.
    feed_ids = Counter(v["feed_id"] for v in s.info.values() if v["feed_id"])
    ahead = [m for m in matches if m["status"] == "scheduled" and m["kickoff_utc"]
             and now < ssch.naive(m["kickoff_utc"]) <= now + pd.Timedelta(days=10)]
    assert all(feed_ids[m["match_id"]] == 1 for m in ahead)
    assert sum(m["league"] in sfeed.REPEATS for m in ahead) > 50
    for m in matches:
        if m["status"] == "scheduled" and m["uk_date"]:
            assert s.status[ssch.feed_key(m)] == "scheduled"
