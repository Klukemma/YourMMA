"""The soccer run end to end on a small made-up league: what it logs, what it
grades, and the files the app reads.

The model is stood in for by a fixed rule, so these check the plumbing -
point in time, the log, the payloads - not the model, which the walk-forward
research in experiments/ measures.

The second half runs the same league with the free feeds (soccer_feed's
snapshot, written here from made-up rows in the feed's own schema) beside
football-data.co.uk: a round the feeds list days before football-data.co.uk
posts it with prices, kickoffs the sources disagree on, a time not
confirmed, a postponement, provisional results, a feed that is down."""

import json

import numpy as np
import pandas as pd
import pytest

import soccer_backtest as bt
import soccer_data as sd
import soccer_feed as sfeed
import soccer_log as sl
import soccer_model as smod
import soccer_predict as sp
import soccer_schedule as ssch

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford", "Leeds", "Stoke"]


def _league(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2005, 2027):
        day = pd.Timestamp(f"{season}-08-10")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // 5))
            if season == 2026 and date > pd.Timestamp("2026-10-20"):
                break
            # Each home-away pairing is played once a league season: the
            # fixtures below are still to come in 2026.
            if season == 2026 and (h, a) in (("Arsenal", "Chelsea"), ("Leeds", "Stoke")):
                continue
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00",
                         "kickoff": date + pd.Timedelta(hours=14), "home": h, "away": a,
                         "hg": hg, "ag": ag, "result": "H" if hg > ag else "A" if hg < ag else "D",
                         "avg_h": 2.2, "avg_d": 3.3, "avg_a": 3.4, "max_h": 2.3, "max_d": 3.5, "max_a": 3.6,
                         "avg_o25": 1.9, "avg_u25": 1.9})
    return sd.frame(rows)


def _fixtures():
    rows = [{"match_id": sd.match_id("E0", pd.Timestamp("2026-10-31").date(), "Arsenal", "Chelsea"), "league": "E0",
             "season": 2026, "date": "2026-10-31", "time": "15:00", "kickoff": pd.Timestamp("2026-10-31 15:00"),
             "home": "Arsenal", "away": "Chelsea", "avg_h": 1.8, "avg_d": 3.6, "avg_a": 4.5,
             "max_h": 1.9, "max_d": 3.8, "max_a": 4.8, "avg_o25": 1.8, "avg_u25": 2.0},
            {"match_id": sd.match_id("E0", pd.Timestamp("2026-11-01").date(), "Leeds", "Stoke"), "league": "E0",
             "season": 2026, "date": "2026-11-01", "time": "16:30", "kickoff": pd.Timestamp("2026-11-01 16:30"),
             "home": "Leeds", "away": "Stoke"}]                      # no price posted
    fx = sd.frame(rows)
    fx["fetched_at"] = pd.Timestamp("2026-10-30 16:00")
    return fx


def _stub(frame, model=None):
    """A fixed rule standing in for the model: a little more home than the
    market, and the league's base rate where there is no price; with the
    expected goals and Dixon-Coles rho the real model also returns
    (soccer_model.predict), which the score matrix is built from."""
    out = pd.DataFrame({"p_h": 0.47, "p_d": 0.27, "p_a": 0.26, "p_o25": 0.52,
                        "lam": 1.5, "nu": 1.1, "rho": -0.06}, index=frame.index)
    top = frame["league"].isin(sd.TOP) & (frame["season"] > 2005)
    out.loc[~top] = np.nan
    return out


@pytest.fixture
def run(tmp_path, monkeypatch):
    """go(now, results=, fixtures=, feed=, sync=, lines=) runs the engine once
    and returns (matches, tables, performance, history) as the app and the
    log hold them. results/fixtures replace football-data.co.uk's files from
    then on; feed (source rows) rewrites the feeds' snapshot. go.frames holds
    every frame the model was given in the last run. The feeds' snapshot is
    always this test's own: never soccer/data/feed.json."""
    state = {"matches": _league(), "fixtures": _fixtures()}
    frames = []
    monkeypatch.setattr(sd, "load_matches", lambda *a, **k: state["matches"].copy())
    monkeypatch.setattr(sd, "load_fixtures", lambda *a, **k: state["fixtures"].copy())

    def model(frame, *a, **k):
        frames.append(frame.copy())
        return _stub(frame)
    monkeypatch.setattr(smod, "predict", model)
    feed_path = tmp_path / "feed.json"

    def go(now, results=None, fixtures=None, feed=None, sync=False, lines=None):
        if results is not None:
            state["matches"] = results
        if fixtures is not None:
            state["fixtures"] = fixtures
        if feed is not None:
            write_feed(feed_path, feed)
        frames.clear()
        sp.run(sync=sync, now=pd.Timestamp(now), out_dir=tmp_path, log_path=tmp_path / "history.json",
               log=lines.append if lines is not None else lambda *_: None, feed_path=feed_path)
        read = lambda n: json.loads((tmp_path / f"{n}.json").read_text())
        return read("matches"), read("tables"), read("performance"), json.loads((tmp_path / "history.json").read_text())
    go.frames, go.feed_path, go.dir = frames, feed_path, tmp_path
    return go


def test_the_week_is_picked_logged_and_written_for_the_app(run):
    m, tables, perf, log = run("2026-10-30 18:00")
    ups = {e["match_id"]: e for e in m["matches"]}
    assert len(ups) == 2 and set(log["picks"]) == set(ups)
    priced = next(e for e in ups.values() if e["home"]["name"] == "Arsenal")
    assert abs(priced["p_h"] + priced["p_d"] + priced["p_a"] - 1) < 1e-9
    assert priced["pick"] in ("H", "D", "A") and priced["market"] is not None
    assert priced["prices"]["avg"]["h"] == 1.8 and priced["prices_at"] == "2026-10-30T16:00:00Z"
    assert priced["home"]["code"] == "ARS" and priced["home_form"] and priced["home_position"]
    # No price: the model's own chances, said so - never a market of zeros.
    bare = next(e for e in ups.values() if e["home"]["name"].startswith("Leeds"))
    assert bare["market"] is None and bare["p_h"] == pytest.approx(0.47)
    assert m["league_order"] == list(sd.TOP) and "E0" in m["leagues"]
    assert tables["leagues"][0]["rows"][0]["club"]["code"]
    assert perf["backtest"]["periods"]["confirm"] == list(bt.CONFIRM)
    assert perf["track_record"]["pending"] == 2


def test_a_pick_is_frozen_at_kickoff_and_graded_when_the_result_lands(run):
    _, _, _, before = run("2026-10-30 18:00")
    first = before["picks"]
    # Saturday afternoon, after the first kickoff: that pick may not change.
    _, _, _, during = run("2026-10-31 15:30")
    arsenal = next(k for k in first if "arsenal" in k)
    assert during["picks"][arsenal]["last_logged"] == first[arsenal]["last_logged"]
    # The result lands in the next sync.
    played = _league()
    extra = _fixtures().iloc[[0]].drop(columns=["fetched_at"]).assign(hg=0, ag=2, result="A")
    m, _, perf, after = run("2026-11-02 09:00", results=pd.concat([played, sd.frame(extra.to_dict("records"))]))
    p = after["picks"][arsenal]
    assert p["final"] == {"home": 0, "away": 2, "result": "A"} and p["correct"] == (p["pick"] == "A")
    shown = next(e for e in m["matches"] if e["match_id"] == arsenal)
    assert shown["status"] == "final" and shown["score"] == {"home": 0, "away": 2}
    assert perf["track_record"]["settled"] == 1


def test_no_payload_carries_a_nan(run):
    m, tables, perf, _ = run("2026-10-30 18:00")
    for payload in (m, tables, perf):
        text = json.dumps(payload, allow_nan=False)      # raises on NaN
        assert "NaN" not in text and payload["schema"] == 1 and payload["generated"]


def _strict(text):
    """json.loads that refuses NaN and infinity, wherever they are."""
    def refuse(c):
        raise ValueError(f"{c} in a payload")
    return json.loads(text, parse_constant=refuse)


def test_goals_and_scores_ride_with_every_pick_and_are_logged_with_it(run):
    m, _, _, log = run("2026-10-30 18:00")
    ups = {e["home"]["name"]: e for e in m["matches"]}
    priced, bare = ups["Arsenal"], next(e for k, e in ups.items() if k.startswith("Leeds"))
    # Priced home/draw/away and over/under: the market-implied matrix.
    assert priced["score_source"] == "market" and priced["scores"]["reason"] is None
    assert priced["scores"]["miss"] < 0.04 and priced["scores"]["miss_on"] in ("home win", "draw", "away win", "over 2.5")
    # No price: the model's own matrix, said so; its 1X2 is the model's.
    assert bare["score_source"] == "model" and bare["scores"]["reason"] == "no_price" and bare["scores"]["miss"] is None
    for e in (priced, bare):
        assert 0 < e["p_o35"] < e["scores"]["o25"] < e["p_o15"] < 1 and 0 < e["p_btts"] < 1
        top = e["scores"]["top"]
        assert len(top) == 3 and top[0]["p"] >= top[1]["p"] >= top[2]["p"] > 0
        assert abs(sum(e["scores"]["matrix"].values()) - 1) < 1e-9
        # The pick's own chances stay as they were: the matrix's are beside them.
        assert e["p_h"] == e["pick_prob"] or e["pick"] != "H"
    assert priced["p_o25"] == priced["market_o25"]                     # over 2.5 stays the market's
    # The band from the pick's own chance: measured on the market's picks
    # where the pick is the market's, on the model's own where it is not.
    assert 0.5 <= priced["pick_prob"] < 0.6 and (priced["band"], priced["band_basis"]) == ("SOLID", "pick")
    assert bare["pick_prob"] == pytest.approx(0.47) and (bare["band"], bare["band_basis"]) == ("LEAN", "model")
    for e in (priced, bare):
        p = log["picks"][e["match_id"]]
        assert {k: p[k] for k in sp.SCORE_KEYS} == {k: e[k] for k in sp.SCORE_KEYS}
    # The measured table travels with the matches, and the caveat with it.
    sm = m["scores_measured"]
    assert set(sm["periods"]) == {"2006-2015", "2016-2025"} and sm["fit_max_miss"] == 0.04
    assert set(sm["bands"]) == {"pick", "model"} and sm["band_edges"] == [0.4, 0.5, 0.6]
    assert "Both teams to score runs low" in m["caveats"]["scores"]


def test_no_payload_carries_a_nan_with_the_score_numbers_in_it(run):
    run("2026-10-30 18:00")
    for name in ("matches", "tables", "performance"):
        payload = _strict((run.dir / f"{name}.json").read_text())
        assert payload["schema"] == 1
    _strict((run.dir / "history.json").read_text())
    m = _strict((run.dir / "matches.json").read_text())
    for e in m["matches"]:
        if e["pick"]:
            assert all(isinstance(e[k], float) for k in ("p_btts", "p_o15", "p_o35"))


def test_both_teams_to_score_is_graded_in_the_live_record(run):
    run("2026-10-30 18:00")
    played = _league()
    extra = _fixtures().iloc[[0]].drop(columns=["fetched_at"]).assign(hg=0, ag=2, result="A")
    _, _, perf, after = run("2026-11-02 09:00", results=pd.concat([played, sd.frame(extra.to_dict("records"))]))
    p = next(v for k, v in after["picks"].items() if "arsenal" in k)
    assert p["btts_correct"] == (p["p_btts"] < 0.5)                    # 0-2: not both
    rec = perf["track_record"]
    assert rec["btts_settled"] == 1 and rec["btts_correct"] == int(p["btts_correct"])


def test_a_frozen_pick_shows_the_score_numbers_it_was_logged_with(run):
    _, _, _, before = run("2026-10-30 18:00")
    arsenal = next(k for k in before["picks"] if "arsenal" in k)
    m, _, _, _ = run("2026-10-31 15:30")
    shown = next(e for e in m["matches"] if e["match_id"] == arsenal)
    assert {k: shown[k] for k in sp.SCORE_KEYS} == {k: before["picks"][arsenal][k] for k in sp.SCORE_KEYS}


def test_the_standalone_page_carries_the_data(run, tmp_path):
    import soccer_standalone
    run("2026-10-30 18:00")
    out = soccer_standalone.build(tmp_path / "x.html", tmp_path, log=lambda *_: None)
    html = out.read_text()
    assert "window.__YOURSOCCER_DATA__" in html and '<script type="module" src="app.js">' not in html
    assert 'rel="manifest"' not in html


def test_a_kicked_off_pick_stays_on_screen_until_its_result_lands(run):
    run("2026-10-30 18:00")
    m, _, perf, _ = run("2026-10-31 15:30")
    shown = {e["match_id"]: e for e in m["matches"]}
    arsenal = next(k for k in shown if "arsenal" in k)
    assert shown[arsenal]["status"] == "started" and shown[arsenal]["pick"] and "score" not in shown[arsenal]
    assert perf["track_record"]["pending"] == 2


def test_latest_results_never_list_a_match_that_had_a_pick(run):
    run("2026-10-30 18:00")
    played = _league()
    extra = _fixtures().drop(columns=["fetched_at"]).assign(hg=[0, 1], ag=[2, 1], result=["A", "D"])
    m, _, _, _ = run("2026-11-05 09:17", results=pd.concat([played, sd.frame(extra.to_dict("records"))]))
    logged = {e["match_id"] for e in m["matches"] if e["status"] == "final"}
    assert len(logged) == 2
    assert not logged & {r["match_id"] for r in m["results"]}
    assert all(e.get("prices_at") is None for e in m["matches"] if e["home"]["name"].startswith("Leeds"))


# ------------------------------------------------------------- with the free feeds

SEEN = "2026-10-26T13:00:00Z"
DETAIL = {"scheduled": "STATUS_SCHEDULED", "live": "STATUS_SECOND_HALF", "final": "STATUS_FULL_TIME",
          "postponed": "STATUS_POSTPONED", "abandoned": "STATUS_ABANDONED"}
MONDAY = "2026-10-26 23:00"        # football-data.co.uk has not posted the weekend yet
FRIDAY = "2026-10-30 18:00"        # it has (the fixtures' fetched_at is 16:00)
AC = sd.match_id("E0", pd.Timestamp("2026-10-31").date(), "Arsenal", "Chelsea")
LS = sd.match_id("E0", pd.Timestamp("2026-11-01").date(), "Leeds", "Stoke")
THROUGH = _league()["date"].max()  # football-data.co.uk's latest result here: what the model's ratings include


def _when(kickoff):
    k = pd.Timestamp(kickoff)
    return k, f"{k:%Y-%m-%dT%H:%M:%SZ}", k.tz_localize("UTC").tz_convert(sd.UK).date().isoformat()


def espn(home, away, kickoff, status="scheduled", score=None, valid=True, eid=None, seen=SEEN):
    """An ESPN scoreboard row as soccer_feed.parse_espn writes it."""
    k, iso, day = _when(kickoff)
    final = status == "final"
    return sfeed._row(source="espn", source_id=eid or f"{home}-{away}", request=f"espn:E0:{k:%Y%m%d}",
                      league="E0", season=2026, home=home, away=away, home_name=home, away_name=away,
                      kickoff_utc=iso if valid else None, time_confirmed=valid, uk_date=day if valid else None,
                      date_hint=day, status=status, status_detail=DETAIL[status],
                      hg=score[0] if final else None, ag=score[1] if final else None, seen_at=seen)


def fdl(home, away, kickoff=None, score=None, hint=None, seen=SEEN):
    """A fixturedownload.com row; kickoff None is its placeholder: no time,
    only the round's nominal day (`hint`)."""
    iso = day = None
    if kickoff is not None:
        _, iso, day = _when(kickoff)
    return sfeed._row(source="fixturedownload", source_id=f"2026-{home[:3]}{away[:3]}",
                      request="fixturedownload:E0:2026", league="E0", season=2026, home=home, away=away,
                      home_name=home, away_name=away, kickoff_utc=iso, time_confirmed=False, uk_date=day,
                      date_hint=day or hint, status="final" if score else "scheduled",
                      hg=score[0] if score else None, ag=score[1] if score else None, round=9, seen_at=seen)


def both(home, away, kickoff):
    """Both feeds listing the match at the same time: a confirmed kickoff."""
    return [espn(home, away, kickoff), fdl(home, away, kickoff)]


def write_feed(path, rows, at=SEEN):
    sfeed.save({"schema": sfeed.SCHEMA, "requests": {}, "last_run": {"at": at, "answered": 1, "failed": {}},
                "rows": rows}, path)


def none_posted():
    return _fixtures().iloc[:0]


def weekend():
    """The feeds' weekend: the two matches football-data.co.uk posts on Friday, at its times."""
    return both("Arsenal", "Chelsea", "2026-10-31 15:00") + both("Leeds", "Stoke", "2026-11-01 16:30")


def played_ac(seen="2026-10-31T21:47:00Z", score=(0, 2)):
    """Saturday evening: both feeds have Arsenal v Chelsea's score."""
    return [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="final", score=score, seen=seen),
            fdl("Arsenal", "Chelsea", "2026-10-31 15:00", score=score)] + both("Leeds", "Stoke", "2026-11-01 16:30")


def posted_ac(hg, ag):
    """football-data.co.uk's results with Arsenal v Chelsea in."""
    row = _fixtures().iloc[[0]].drop(columns=["fetched_at"]).assign(hg=hg, ag=ag,
                                                                     result="H" if hg > ag else "A" if hg < ag else "D")
    return pd.concat([_league(), sd.frame(row.to_dict("records"))], ignore_index=True)


def by_id(m):
    return {e["match_id"]: e for e in m["matches"]}


def test_the_feeds_round_is_shown_and_picked_by_the_model_alone_before_any_price(run):
    m, _, perf, log = run(MONDAY, fixtures=none_posted(), feed=weekend())
    ups = by_id(m)
    assert set(ups) == {AC, LS} == set(log["picks"])
    for e in ups.values():
        assert e["status"] == "upcoming" and e["pick"] in ("H", "D", "A") and not e.get("locked")
        # No price: the model's own chances, labelled, never a market of zeros.
        assert e["market"] is None and e["prices_at"] is None and e["price_status"] == "not_yet"
        assert e["p_h"] == pytest.approx(0.47) and e["model"]["p_d"] == pytest.approx(0.27)
        assert e["fixture_source"] == ssch.FEED_SOURCE and e["time_confirmed"] is True
        assert e["freeze_at"] == e["kickoff"]
        p = log["picks"][e["match_id"]]
        assert p["market"] is None and p["price_source"] is None and p["fixture_source"] == ssch.FEED_SOURCE
    assert ups[AC]["kickoff"] == "2026-10-31T15:00:00Z" and ups[LS]["kickoff"] == "2026-11-01T16:30:00Z"
    # The model was given them as matches to play: no price, no result.
    frame = run.frames[0]
    rows = frame[frame["match_id"].isin([AC, LS])]
    assert len(rows) == 2 and rows["result"].isna().all() and rows["avg_h"].isna().all()
    assert perf["track_record"]["pending"] == 2 and m["feeds_at"] == SEEN
    assert "feeds" in m["caveats"] and "provisional" in m["caveats"]["feeds"]


def test_football_datas_priced_fixture_takes_over_the_same_match_before_kickoff(run):
    _, _, _, monday = run(MONDAY, fixtures=none_posted(), feed=weekend())
    m, _, perf, friday = run(FRIDAY, fixtures=_fixtures())
    e = by_id(m)[AC]
    assert e["market"] is not None and e["price_status"] == "posted" and e["fixture_source"] == "football-data.co.uk"
    assert e["prices_at"] == "2026-10-30T16:00:00Z" and abs(e["p_h"] + e["p_d"] + e["p_a"] - 1) < 1e-9
    # One match, one pick, under one id: the market's replaced the model's.
    assert set(friday["picks"]) == {AC, LS} and not any(p.get("void") for p in friday["picks"].values())
    p = friday["picks"][AC]
    assert p["market"] == e["market"] and p["pick"] == e["pick"] and p["price_source"] == "football-data.co.uk"
    assert monday["picks"][AC]["market"] is None
    assert p["first_logged"] == monday["picks"][AC]["first_logged"] == "2026-10-26T23:00:00Z"
    assert p["last_logged"] == "2026-10-30T18:00:00Z"
    # Posted with no price: still the model alone, said so.
    assert by_id(m)[LS]["market"] is None and by_id(m)[LS]["price_status"] == "none"
    assert perf["track_record"]["pending"] == 2


def test_a_feed_date_football_data_posts_differently_moves_the_pick_and_voids_nothing(run):
    monday_ls = sd.match_id("E0", pd.Timestamp("2026-11-02").date(), "Leeds", "Stoke")
    feed = both("Arsenal", "Chelsea", "2026-10-31 15:00") + both("Leeds", "Stoke", "2026-11-02 20:00")
    _, _, _, monday = run(MONDAY, fixtures=none_posted(), feed=feed)
    assert set(monday["picks"]) == {AC, monday_ls}
    # football-data.co.uk posts it for Sunday 16:30; the feeds still say Monday.
    m, _, perf, friday = run(FRIDAY, fixtures=_fixtures())
    assert set(friday["picks"]) == {AC, LS}
    p = friday["picks"][LS]
    assert p["id_was"] == monday_ls and p["first_logged"] == "2026-10-26T23:00:00Z" and not p.get("void")
    e = by_id(m)[LS]
    assert e["kickoff"] == e["freeze_at"] == "2026-11-01T16:30:00Z"
    assert e["kickoff_sources"] == {"football-data.co.uk": "2026-11-01T16:30:00Z", "ESPN": "2026-11-02T20:00:00Z",
                                    "fixturedownload.com": "2026-11-02T20:00:00Z"}
    assert perf["track_record"]["pending"] == 2


def test_when_the_sources_disagree_on_the_kickoff_the_pick_freezes_at_the_earlier(run):
    # football-data.co.uk: Arsenal v Chelsea 15:00, Leeds v Stoke 16:30.
    feed = both("Arsenal", "Chelsea", "2026-10-31 12:30") + both("Leeds", "Stoke", "2026-11-01 18:00")
    m, _, _, friday = run(FRIDAY, feed=feed)
    e = by_id(m)[AC]
    assert e["kickoff"] == e["freeze_at"] == "2026-10-31T12:30:00Z" and e["time_confirmed"] is True
    assert e["kickoff_sources"] == {"football-data.co.uk": "2026-10-31T15:00:00Z", "ESPN": "2026-10-31T12:30:00Z",
                                    "fixturedownload.com": "2026-10-31T12:30:00Z"}
    assert e["market"] is not None and friday["picks"][AC]["freeze_at"] == "2026-10-31T12:30:00Z"
    assert by_id(m)[LS]["kickoff"] == by_id(m)[LS]["freeze_at"] == "2026-11-01T16:30:00Z"   # the earlier either way
    # 13:00: past the earlier kickoff, before football-data.co.uk's - frozen.
    m, _, _, sat = run("2026-10-31 13:00")
    assert sat["picks"][AC] == friday["picks"][AC]
    shown = by_id(m)[AC]
    assert shown["status"] == "started" and shown["pick"] == friday["picks"][AC]["pick"]
    later_time = dict(sat["picks"][AC], pick="D", freeze_at=None, kickoff="2026-10-31T15:00:00Z")
    assert not sl.log_pick(sat, later_time, pd.Timestamp("2026-10-31 13:00"))


def test_a_time_the_feeds_do_not_confirm_freezes_the_pick_at_midnight_uk(run):
    # ESPN has 15:00, fixturedownload 17:30: not confirmed, and the earlier is kept.
    feed = [espn("Arsenal", "Chelsea", "2026-10-31 15:00"), fdl("Arsenal", "Chelsea", "2026-10-31 17:30")]
    m, _, _, log = run(MONDAY, fixtures=none_posted(), feed=feed)
    e = by_id(m)[AC]
    assert e["time_confirmed"] is False and e["kickoff"] == "2026-10-31T15:00:00Z"
    assert e["freeze_at"] == "2026-10-31T00:00:00Z"            # GMT since Oct 25: midnight UK is midnight UTC
    assert log["picks"][AC]["freeze_at"] == e["freeze_at"] and log["picks"][AC]["time_confirmed"] is False
    # In summer time, midnight UK is 23:00 UTC the day before.
    assert ssch.freeze_at(pd.Timestamp("2026-10-24 14:00"), False) == pd.Timestamp("2026-10-23 23:00")
    _, _, _, fri = run("2026-10-30 23:30")                     # still open
    assert fri["picks"][AC]["last_logged"] == "2026-10-30T23:30:00Z"
    m, _, _, sat = run("2026-10-31 00:30")                     # closed, hours before the listed kickoff
    e = by_id(m)[AC]
    assert e["locked"] and e["status"] == "upcoming" and e["pick"] == fri["picks"][AC]["pick"]
    assert sat["picks"][AC] == fri["picks"][AC]
    # football-data.co.uk then posts it for 15:00, priced: the pick froze at
    # midnight and stays as it was logged.
    m, _, _, later = run("2026-10-31 09:00", fixtures=_fixtures())
    e = by_id(m)[AC]
    assert later["picks"][AC] == fri["picks"][AC]
    assert e["locked"] and e["market"] is None and e["pick"] == fri["picks"][AC]["pick"]
    assert e["freeze_at"] == "2026-10-31T00:00:00Z"
    # A match first listed after midnight with a time not confirmed is never picked.
    late = feed + [espn("Watford", "Leeds", "2026-10-31 17:30"), fdl("Watford", "Leeds", "2026-10-31 20:00")]
    m, _, _, log = run("2026-10-31 09:30", feed=late)
    wl = next(e for e in m["matches"] if e.get("home_key") == "Watford")
    assert wl["locked"] and wl["pick"] is None and "midnight UK" in wl["no_pick_reason"]
    assert not any(p["home"] == "Watford" for p in log["picks"].values())


def test_a_postponed_match_is_not_picked_and_a_pick_logged_before_is_held(run):
    _, _, _, monday = run(MONDAY, fixtures=none_posted(), feed=weekend())
    first = monday["picks"][AC]
    off = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="postponed"),
           fdl("Arsenal", "Chelsea", "2026-10-31 15:00")] + both("Leeds", "Stoke", "2026-11-01 16:30")
    m, _, perf, thu = run("2026-10-29 09:00", feed=off)
    assert thu["picks"][AC] == first                           # not re-logged, not graded, not void
    e = by_id(m)[AC]
    assert e["status"] == "postponed" and e["held"] and e["pick"] == first["pick"]
    assert e["logged_at"] == first["last_logged"] and e["logged_for"] == "2026-10-31T15:00:00Z"
    assert perf["track_record"]["pending"] == 2
    # football-data.co.uk posts it, priced, on the date ESPN says is postponed: not picked.
    m, _, _, fri = run(FRIDAY, fixtures=_fixtures())
    assert fri["picks"][AC] == first and by_id(m)[AC]["status"] == "postponed"
    assert [e["match_id"] for e in m["matches"]].count(AC) == 1
    # Past its date: still postponed with the pick held - never "kicked off".
    m, _, perf, sun = run("2026-11-01 09:00")
    e = by_id(m)[AC]
    assert e["status"] == "postponed" and e["held"] and sun["picks"][AC] == first
    assert perf["track_record"]["pending"] == 2 and perf["track_record"]["settled"] == 0
    # Rescheduled for Wednesday the 4th, which football-data.co.uk has not
    # posted: the new date is picked (no price), the old pick is void - one
    # pick for the match.
    new = sd.match_id("E0", pd.Timestamp("2026-11-04").date(), "Arsenal", "Chelsea")
    moved = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="postponed", eid="old"),
             espn("Arsenal", "Chelsea", "2026-11-04 19:45", eid="new"),
             fdl("Arsenal", "Chelsea", "2026-11-04 19:45")] + both("Leeds", "Stoke", "2026-11-01 16:30")
    m, _, perf, mon = run("2026-11-02 09:00", feed=moved)
    assert mon["picks"][AC]["void"] == "rescheduled" and not mon["picks"][new].get("void")
    pair = [p for p in mon["picks"].values() if p["home"] == "Arsenal" and not p.get("void")]
    assert len(pair) == 1 and by_id(m)[new]["status"] == "upcoming" and by_id(m)[new]["price_status"] == "not_yet"
    assert AC not in by_id(m)


def test_a_match_whose_date_the_feeds_withdraw_keeps_its_pick_on_an_undated_card(run):
    _, _, _, monday = run(MONDAY, fixtures=none_posted(), feed=weekend())
    # ESPN drops Leeds v Stoke; fixturedownload's round loses its time.
    feed = both("Arsenal", "Chelsea", "2026-10-31 15:00") + [fdl("Leeds", "Stoke", None, hint="2026-11-01")]
    m, _, perf, log = run("2026-10-28 09:00", feed=feed)
    tbc = [e for e in m["matches"] if e["status"] == "tbc"]
    assert len(tbc) == 1 and tbc[0]["match_id"] is None and tbc[0]["kickoff"] is None
    assert tbc[0]["held"] and tbc[0]["pick"] == monday["picks"][LS]["pick"]
    assert tbc[0]["logged_for"] == "2026-11-01T16:30:00Z" and log["picks"][LS] == monday["picks"][LS]
    assert tbc[0]["home_form"] and tbc[0]["home_position"]     # never "no matches yet"
    assert perf["track_record"]["pending"] == 2


def test_a_feed_result_grades_provisionally_then_football_data_confirms_it(run):
    _, _, _, friday = run(FRIDAY, feed=weekend())
    pick = friday["picks"][AC]["pick"]
    m, tables, perf, sat = run("2026-10-31 21:47", feed=played_ac())
    p = sat["picks"][AC]
    assert p["final"]["provisional"] is True and (p["final"]["home"], p["final"]["away"]) == (0, 2)
    assert p["final"]["source"] == "ESPN + fixturedownload.com" and p["final"]["seen_at"] == "2026-10-31T21:47:00Z"
    assert p["correct"] == (pick == "A") and p["last_logged"] == friday["picks"][AC]["last_logged"]
    e = by_id(m)[AC]
    assert e["status"] == "final" and e["provisional"] is True and e["score"] == {"home": 0, "away": 2}
    assert e["result_source"] == "ESPN + fixturedownload.com" and e["pick"] == pick
    assert perf["track_record"]["settled"] == 1 and perf["track_record"]["provisional"] == 1
    assert perf["track_record"]["by_week"][-1]["provisional"] == 1
    assert m["provisional_results"] == 1 and m["results_through"] == "2026-10-31"
    assert m["model_through"] == m["data_through"] == THROUGH
    assert AC not in {r["match_id"] for r in m["results"]}      # shown with its pick, not twice
    arsenal = next(r for r in tables["leagues"][0]["rows"] if r["team"] == "Arsenal")
    assert arsenal["provisional"] == 1 and arsenal["form"][0] == dict(arsenal["form"][0], provisional=True)
    assert tables["leagues"][0]["provisional"] == 1 and tables["leagues"][0]["model_through"] == THROUGH
    # Monday: football-data.co.uk posts the same score. The grade stands, confirmed.
    m, tables, perf, mon = run("2026-11-02 09:17", results=posted_ac(0, 2))
    p = mon["picks"][AC]
    assert not p["final"].get("provisional") and p["final"]["provisional_was"]["confirmed"] is True
    assert p["correct"] == (pick == "A") and not p.get("corrections")
    e = by_id(m)[AC]
    assert e["provisional"] is False and e["result_source"] == "football-data.co.uk"
    assert perf["track_record"]["settled"] == 1 and perf["track_record"]["provisional"] == 0
    assert m["provisional_results"] == 0 and tables["leagues"][0]["provisional"] == 0


def test_football_data_corrects_a_provisional_result_and_disagreeing_feeds_withdraw_one(run):
    _, _, _, friday = run(FRIDAY, feed=weekend())
    pick = friday["picks"][AC]["pick"]
    run("2026-10-31 21:47", feed=played_ac())
    # Sunday: fixturedownload has 1-2 now, ESPN still 0-2 - no result stands.
    disputed = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="final", score=(0, 2)),
                fdl("Arsenal", "Chelsea", "2026-10-31 15:00", score=(1, 2))] + both("Leeds", "Stoke", "2026-11-01 16:30")
    _, _, perf, sun = run("2026-11-01 09:00", feed=disputed)
    p = sun["picks"][AC]
    assert "final" not in p and p["corrections"][-1]["by"] == "the feeds disagree" and p["corrections"][-1]["now"] is None
    assert perf["track_record"]["settled"] == 0 and perf["track_record"]["pending"] == 2
    # The feeds agree on 0-2 again; then football-data.co.uk posts 1-1: that is the result.
    run("2026-11-01 12:00", feed=played_ac())
    m, _, perf, mon = run("2026-11-02 09:17", results=posted_ac(1, 1))
    p = mon["picks"][AC]
    assert p["final"]["result"] == "D" and p["correct"] == (pick == "D") and not p["final"].get("provisional")
    fix = p["corrections"][-1]
    assert fix["by"] == "football-data.co.uk" and fix["at"] == "2026-11-02T09:17:00Z"
    assert (fix["was"]["home"], fix["was"]["away"], fix["was"]["provisional"]) == (0, 2, True)
    assert (fix["now"]["home"], fix["now"]["away"], fix["now"]["result"]) == (1, 1, "D")
    assert perf["track_record"]["corrected"] == 1 and perf["track_record"]["provisional"] == 0
    e = by_id(m)[AC]
    assert e["score"] == {"home": 1, "away": 1} and e["corrections"] and e["provisional"] is False


def test_a_provisional_result_never_reaches_the_model(run):
    run(FRIDAY, feed=weekend())
    m, tables, _, _ = run("2026-10-31 21:47", feed=played_ac())
    assert len(run.frames) == 1
    frame = run.frames[0]
    # Not as a result, and not as a match still to play either.
    assert not (frame["match_id"] == AC).any()
    assert not ((frame["home"] == "Arsenal") & (frame["away"] == "Chelsea") & (frame["season"] == 2026)).any()
    done = frame[frame["result"].notna()]
    assert set(done["match_id"]) == set(_league()["match_id"])   # football-data.co.uk's results, exactly
    # ...while the tables, which are shown, count it, flagged.
    assert tables["leagues"][0]["provisional"] == 1 and m["model_through"] == THROUGH


def test_running_twice_changes_nothing(run):
    run(FRIDAY, feed=weekend())
    run("2026-10-31 21:47", feed=played_ac())
    files = lambda: {n: (run.dir / f"{n}.json").read_bytes() for n in ("matches", "tables", "performance", "history")}
    first, feed = files(), run.feed_path.read_bytes()
    run("2026-10-31 21:47")
    assert files() == first and run.feed_path.read_bytes() == feed
    # A later run with nothing new neither regrades nor adds a pick.
    _, _, _, later = run("2026-10-31 23:00")
    before = json.loads(first["history"])
    assert later["picks"][AC] == before["picks"][AC] and set(later["picks"]) == set(before["picks"])


def test_a_feed_that_is_down_keeps_its_snapshot_and_never_stops_the_run(run, monkeypatch):
    monkeypatch.setattr(sd, "sync", lambda *a, **k: None)
    real, calls = sfeed.sync, []

    def down(url):
        calls.append(url)
        raise ConnectionError("timed out")
    monkeypatch.setattr(sfeed, "sync", lambda **kw: real(get=down, **kw))
    run(MONDAY, fixtures=none_posted(), feed=weekend())
    rows = sfeed.load(run.feed_path)["rows"]
    m, _, _, log = run("2026-10-27 09:00", sync=True)
    snap = sfeed.load(run.feed_path)
    assert calls and snap["rows"] == rows                        # asked, failed, nothing lost
    assert snap["last_run"]["answered"] == 0 and len(snap["last_run"]["failed"]) == 80
    assert {AC, LS} <= set(by_id(m)) and set(log["picks"]) == {AC, LS}
    # A snapshot that cannot be read, and nothing answered to rebuild it
    # from: football-data.co.uk alone, and the file left as it is (one that
    # answers rebuilds it: test_soccer_feed).
    run.feed_path.write_text("{ not json")
    lines = []
    m, _, _, log = run("2026-10-27 10:00", sync=True, lines=lines)
    assert run.feed_path.read_text() == "{ not json"
    assert any("unreadable" in line and "left as it is" in line for line in lines)
    assert any("football-data.co.uk alone" in line for line in lines)
    assert set(log["picks"]) == {AC, LS} and not any(p.get("void") for p in log["picks"].values())
    # Feed rows the join cannot read: football-data.co.uk alone, said so.
    write_feed(run.feed_path, weekend())
    monkeypatch.setattr(sfeed, "reconciled", lambda *a, **k: [{"league": "E0", "season": 2026}])
    lines = []
    m, _, _, _ = run(FRIDAY, fixtures=_fixtures(), lines=lines)
    assert any("could not be joined" in line for line in lines)
    assert by_id(m)[AC]["fixture_source"] == "football-data.co.uk" and by_id(m)[AC]["market"] is not None


# ------------------------------------------------------------- the reviews' cases

def fixtures_at(ls_kickoff=None, drop=()):
    """football-data.co.uk's fixtures with Leeds v Stoke moved to `ls_kickoff`
    (its id with it), and the ids in `drop` left out."""
    fx = _fixtures()
    if ls_kickoff is not None:
        k = pd.Timestamp(ls_kickoff)
        i = fx.index[fx["home"] == "Leeds"][0]
        fx.loc[i, ["date", "time", "kickoff", "match_id"]] = [
            f"{k:%Y-%m-%d}", f"{k:%H:%M}", k, sd.match_id("E0", k.date(), "Leeds", "Stoke")]
    return fx[~fx["match_id"].isin(drop)].reset_index(drop=True)


def test_a_frozen_pick_whose_id_changes_moves_as_it_is_and_nothing_new_is_logged(run):
    """The feeds disagree on Leeds v Stoke (ESPN Sat 17:30, fixturedownload Sun
    16:30): ESPN's date is the match's, not confirmed, so the pick closes at
    midnight UK on Saturday. football-data.co.uk then posts it for Sunday -
    after that midnight. The frozen pick moves to its id as it is; no pick is
    written after it froze, and nothing is voided."""
    sat_ls = sd.match_id("E0", pd.Timestamp("2026-10-31").date(), "Leeds", "Stoke")
    feed = both("Arsenal", "Chelsea", "2026-10-31 15:00") + [espn("Leeds", "Stoke", "2026-10-31 17:30"),
                                                             fdl("Leeds", "Stoke", "2026-11-01 16:30")]
    _, _, _, monday = run(MONDAY, fixtures=none_posted(), feed=feed)
    first = monday["picks"][sat_ls]
    assert first["freeze_at"] == "2026-10-31T00:00:00Z" and first["kickoff"] == "2026-10-31T17:30:00Z"
    m, _, perf, sat = run("2026-10-31 07:17", fixtures=_fixtures())
    assert sat_ls not in sat["picks"] and set(sat["picks"]) == {AC, LS}
    p = sat["picks"][LS]
    assert p["id_was"] == sat_ls and not p.get("void")
    assert {k: p[k] for k in sl.FIELDS if k not in sl.TIME_FIELDS + ("match_id",)} == \
        {k: first[k] for k in sl.FIELDS if k not in sl.TIME_FIELDS + ("match_id",)}
    assert (p["last_logged"], p["first_logged"], p["freeze_at"]) == \
        (first["last_logged"], first["first_logged"], first["freeze_at"])
    assert (p["kickoff"], p["kickoff_was"]) == ("2026-11-01T16:30:00Z", "2026-10-31T17:30:00Z")
    e = by_id(m)[LS]
    assert e["locked"] and e["pick"] == first["pick"] and e["kickoff"] == "2026-11-01T16:30:00Z"
    assert e["freeze_at"] == "2026-10-31T00:00:00Z"
    # Sunday's result lands: one pick, graded once.
    row = fixtures_at().iloc[[1]].drop(columns=["fetched_at"]).assign(hg=1, ag=1, result="D")
    _, _, perf, mon = run("2026-11-02 09:17", results=pd.concat([_league(), sd.frame(row.to_dict("records"))]))
    assert mon["picks"][LS]["final"]["result"] == "D" and perf["track_record"]["settled"] == 1


def test_a_lone_stale_fixturedownload_time_closes_the_pick_but_never_dates_the_match(run):
    """football-data.co.uk and ESPN have Leeds v Stoke on Monday 20:00,
    fixturedownload still on Saturday 19:45 (Ligue 1's Lorient v Brest, real).
    The match is Monday's - its id, its day, a confirmed time - and the pick
    closes at fixturedownload's earlier time. Past that, the card is locked,
    never "kicked off", until Monday's kickoff."""
    mon_ls = sd.match_id("E0", pd.Timestamp("2026-11-02").date(), "Leeds", "Stoke")
    feed = both("Arsenal", "Chelsea", "2026-10-31 15:00") + [espn("Leeds", "Stoke", "2026-11-02 20:00"),
                                                             fdl("Leeds", "Stoke", "2026-10-31 19:45")]
    m, _, _, fri = run(FRIDAY, fixtures=fixtures_at("2026-11-02 20:00"), feed=feed)
    e = by_id(m)[mon_ls]
    assert (e["kickoff"], e["time_confirmed"], e["freeze_at"]) == ("2026-11-02T20:00:00Z", True, "2026-10-31T19:45:00Z")
    assert e["kickoff_sources"] == {"football-data.co.uk": "2026-11-02T20:00:00Z", "ESPN": "2026-11-02T20:00:00Z",
                                    "fixturedownload.com": "2026-10-31T19:45:00Z"}
    p = fri["picks"][mon_ls]
    assert (p["kickoff"], p["freeze_at"], p["kickoff_sources"]) == (e["kickoff"], e["freeze_at"],
                                                                    e["kickoff_sources"])
    m, _, _, sat = run("2026-10-31 21:47")
    e = by_id(m)[mon_ls]
    assert e["status"] == "upcoming" and e["locked"] and e["pick"] == p["pick"] and sat["picks"][mon_ls] == p
    m, _, _, _ = run("2026-11-02 21:47")
    assert by_id(m)[mon_ls]["status"] == "started" and by_id(m)[mon_ls]["kickoff"] == "2026-11-02T20:00:00Z"


def test_a_feed_only_match_is_dated_by_espn_and_closes_at_the_earlier_listing(run):
    feed = [espn("Leeds", "Stoke", "2026-11-02 20:00"), fdl("Leeds", "Stoke", "2026-10-31 19:45")]
    m, _, _, log = run(MONDAY, fixtures=none_posted(), feed=feed)
    mon_ls = sd.match_id("E0", pd.Timestamp("2026-11-02").date(), "Leeds", "Stoke")
    e = by_id(m)[mon_ls]
    assert (e["kickoff"], e["time_confirmed"], e["freeze_at"]) == ("2026-11-02T20:00:00Z", False, "2026-10-31T19:45:00Z")
    assert e["kickoff_sources"] == {"ESPN": "2026-11-02T20:00:00Z", "fixturedownload.com": "2026-10-31T19:45:00Z"}
    m, _, _, _ = run("2026-10-31 21:47")
    e = by_id(m)[mon_ls]
    assert e["status"] == "upcoming" and e["locked"] and e["pick"] == log["picks"][mon_ls]["pick"]


def test_fixturedownload_alone_never_moves_a_football_data_fixture(run):
    """football-data.co.uk has Arsenal v Chelsea Saturday 15:00; only
    fixturedownload lists it, stale, for Sunday (no ESPN row). Its
    "scheduled" means no score posted yet, not a move: the priced pick is
    not voided, no pick is logged after the real kickoff, and the result
    grades it."""
    _, _, _, fri = run(FRIDAY, fixtures=_fixtures(), feed=[fdl("Arsenal", "Chelsea", "2026-11-01 13:00")])
    first = fri["picks"][AC]
    assert first["market"] is not None
    lines = []
    m, _, _, sat = run("2026-10-31 21:47", lines=lines)
    assert not any("moved" in line for line in lines)
    assert sat["picks"][AC] == first and len(sat["picks"]) == 2
    assert by_id(m)[AC]["status"] == "started"
    _, _, perf, mon = run("2026-11-02 09:17", results=posted_ac(0, 2))
    assert mon["picks"][AC]["final"]["result"] == "A" and not mon["picks"][AC].get("void")
    assert perf["track_record"]["settled"] == 1 and len(mon["picks"]) == 2


def test_espn_moving_a_football_data_fixture_after_its_time_is_shown_on_espns_date(run):
    """football-data.co.uk's file still has Saturday 15:00; ESPN lists the
    match for Monday (fixturedownload agrees with the old slot). Past
    Saturday 15:00 with no result, ESPN's date stands: the match is listed
    again, on Monday, with every source's time - never dropped silently. The
    Friday pick froze at Saturday 15:00, and ESPN's Monday is still to come,
    so the match is not known to be off (it may have been played on Saturday
    with its result not in yet): the pick moves to Monday's id as it is,
    locked - one match, one pick, nothing logged after it froze."""
    feed = [espn("Arsenal", "Chelsea", "2026-11-02 19:45"), fdl("Arsenal", "Chelsea", "2026-10-31 15:00")] + \
        both("Leeds", "Stoke", "2026-11-01 16:30")
    m, _, _, fri = run(FRIDAY, fixtures=_fixtures(), feed=feed)
    e = by_id(m)[AC]
    assert (e["kickoff"], e["freeze_at"]) == ("2026-10-31T15:00:00Z", "2026-10-31T15:00:00Z")
    lines = []
    m, _, _, sun = run("2026-11-01 09:00", lines=lines)
    new = sd.match_id("E0", pd.Timestamp("2026-11-02").date(), "Arsenal", "Chelsea")
    assert any("moved - its date stands" in line for line in lines)
    e = by_id(m)[new]
    assert e["status"] == "upcoming" and e["kickoff"] == "2026-11-02T19:45:00Z"
    assert e["kickoff_sources"] == {"football-data.co.uk": "2026-10-31T15:00:00Z", "ESPN": "2026-11-02T19:45:00Z",
                                    "fixturedownload.com": "2026-10-31T15:00:00Z"}
    assert e["locked"] and e["freeze_at"] == "2026-10-31T15:00:00Z"
    p = sun["picks"][new]
    assert AC not in sun["picks"] and p["id_was"] == AC and not p.get("void")
    assert p["last_logged"] == fri["picks"][AC]["last_logged"] and p["pick"] == fri["picks"][AC]["pick"]
    assert sum(1 for p in sun["picks"].values() if p["home"] == "Arsenal" and not p.get("void")) == 1


def test_a_provisional_grade_is_withdrawn_when_espn_no_longer_calls_it_final(run):
    run(FRIDAY, feed=weekend())
    _, _, _, sat = run("2026-10-31 21:47", feed=played_ac())
    assert sat["picks"][AC]["final"]["provisional"]
    abandoned = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="abandoned"),
                 fdl("Arsenal", "Chelsea", "2026-10-31 15:00", score=(0, 2))] + both("Leeds", "Stoke", "2026-11-01 16:30")
    m, _, perf, sun = run("2026-11-01 09:00", feed=abandoned)
    p = sun["picks"][AC]
    assert "final" not in p and "correct" not in p and not p.get("void")
    assert p["corrections"][-1]["by"] == "the feeds now say abandoned" and p["corrections"][-1]["now"] is None
    assert perf["track_record"]["settled"] == 0 and perf["track_record"]["pending"] == 2
    e = by_id(m)[AC]
    assert e["status"] == "abandoned" and e["held"] and "score" not in e


def test_a_wrong_early_fixturedownload_time_on_a_result_never_voids_a_pick(run):
    """The pick is logged Saturday 07:17 for a 15:00 kickoff both
    football-data.co.uk and ESPN give; the result rows carry ESPN's 15:00 and
    a wrong fixturedownload 07:00. The provisional path checks ESPN's time,
    the final's own: graded, not void."""
    run("2026-10-31 07:17", fixtures=_fixtures(), feed=weekend())
    result = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="final", score=(0, 2)),
              fdl("Arsenal", "Chelsea", "2026-10-31 07:00", score=(0, 2))] + both("Leeds", "Stoke", "2026-11-01 16:30")
    _, _, perf, sat = run("2026-10-31 21:47", feed=result)
    p = sat["picks"][AC]
    assert not p.get("void") and p["final"]["provisional"] and p["last_logged"] == "2026-10-31T07:17:00Z"
    _, _, perf, mon = run("2026-11-02 09:17", results=posted_ac(0, 2))
    assert mon["picks"][AC]["final"]["provisional_was"]["confirmed"] and perf["track_record"]["settled"] == 1


def test_a_priced_pick_is_never_replaced_by_an_unpriced_one(run):
    """football-data.co.uk's fixtures file (replaced whole on every sync)
    stops listing a match it had priced while the feeds still list it: the
    pick stays the market's - only its time follows the feeds - and the card
    shows the pick that counts."""
    _, _, _, fri = run(FRIDAY, fixtures=_fixtures(), feed=weekend())
    first = fri["picks"][AC]
    assert first["market"] is not None and first["price_source"] == "football-data.co.uk"
    m, _, _, sat = run("2026-10-31 07:17", fixtures=fixtures_at(drop=(AC,)))
    p = sat["picks"][AC]
    assert {k: p[k] for k in p if k not in sl.TIME_FIELDS} == {k: first[k] for k in first if k not in sl.TIME_FIELDS}
    assert p["last_logged"] == first["last_logged"]
    e = by_id(m)[AC]
    assert e["fixture_source"] == ssch.FEED_SOURCE and e["market"] == first["market"] and e["pick"] == first["pick"]
    assert e["price_status"] == "posted" and e["pick_prob"] == first["pick_prob"]


def test_feeds_that_contradict_each_other_on_a_score_say_so_on_the_card(run):
    run(FRIDAY, feed=weekend())
    clash = [espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="final", score=(1, 1)),
             fdl("Arsenal", "Chelsea", "2026-10-31 15:00", score=(2, 1))] + both("Leeds", "Stoke", "2026-11-01 16:30")
    m, _, perf, sat = run("2026-10-31 21:47", feed=clash)
    e = by_id(m)[AC]
    assert e["status"] == "started" and "score" not in e
    assert e["disputed"] == {"ESPN": "1-1", "fixturedownload.com": "2-1"}
    assert "final" not in sat["picks"][AC] and perf["track_record"]["settled"] == 0


def test_a_real_run_logs_at_a_moment_no_earlier_than_the_data_it_read(run, monkeypatch):
    """With no `now` (the runner), the feeds stamp each row as its request
    answers and the clock is read again after the sync: a pick's log time
    is never before the prices and kickoffs it was built from."""
    run(MONDAY, fixtures=none_posted(), feed=weekend())
    clock = iter([pd.Timestamp("2026-10-30 18:00"), pd.Timestamp("2026-10-30 18:11")])
    monkeypatch.setattr(sp, "now_utc", lambda: next(clock))
    seen = {}
    monkeypatch.setattr(sd, "sync", lambda *a, **k: seen.setdefault("football-data", True))
    monkeypatch.setattr(sfeed, "sync", lambda now=None, **k: seen.setdefault("feeds_now", now))
    sp.run(sync=True, now=None, out_dir=run.dir, log_path=run.dir / "history.json", log=lambda *_: None,
           feed_path=run.feed_path)
    log = json.loads((run.dir / "history.json").read_text())
    assert seen == {"football-data": True, "feeds_now": None}
    assert log["picks"][AC]["last_logged"] == "2026-10-30T18:11:00Z"


def test_a_feed_final_before_its_own_kickoff_is_no_result():
    """A row ESPN calls final while its kickoff is still to come (a feed
    glitch) grades nothing."""
    early = sfeed.reconcile([espn("Arsenal", "Chelsea", "2026-10-31 15:00", status="final", score=(0, 2))],
                            [fdl("Arsenal", "Chelsea", "2026-10-31 15:00")], log=lambda *_: None)
    assert early[0]["status"] == "final"
    assert ssch.provisional_results(early, set(), pd.Timestamp("2026-10-31 14:00")).empty
    assert len(ssch.provisional_results(early, set(), pd.Timestamp("2026-10-31 17:00"))) == 1
