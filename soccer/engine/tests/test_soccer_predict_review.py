"""The pipeline's fixes from the review of the switch to every league, on the
made-up world of test_soccer_predict_leagues (E0, NOR, FIN, SWE - each with
its own decisions) and on small frames of their own:

  * the top five's pooled Record blocks, calibration and the global scores
    caveat are measured over the decided leagues, not the league set of the
    day: in summer the five leave the app and the blocks stay;
  * the kicked-off picks window is each league's own, so a league whose
    results come later keeps its graded pick on screen, and a result is
    never dropped because a pick was logged that is not on screen;
  * rule (a) on the data of the day counts the feeds' matches still to play
    as posted fixtures - a winter break does not drop a league mid-season -
    and not_in_app says why a league is out;
  * a group whose every number is the model's own has a caveat that says so;
  * a raw-price league's weights are None, as soccer_leagues.SHIPPED holds;
  * badge codes are unique over every season of a league, so they stay put;
  * football-data.co.uk's download problems reach matches.json."""

import json

import numpy as np
import pandas as pd
import pytest

import soccer_data as sd
import soccer_export as sx
import soccer_leagues as slg
import soccer_teams as st

# the made-up world and its run (the fixture is registered here by the import)
from test_soccer_predict_leagues import E0_TEAMS, NOR_TEAMS, _rows, by_home, run, world  # noqa: F401


# ------------------------------------------------------------- the scores caveat

def _measure(market):
    """A soccer_scores.measure of two periods: `market` matches read off the
    market's matrix in each, the rest the model's own (no over/under)."""
    def period(n_market):
        b = {"rate": 0.53, "deciles": 10, "deciles_low": 9,
             "shown": {"log_loss": 0.6856, "mean_p": 0.514}, "model": {"log_loss": 0.6856 if not n_market else 0.6870,
                                                                      "mean_p": 0.514 if not n_market else 0.511},
             "base": {"log_loss": 0.6873, "mean_p": 0.52}}
        ok = {"shown": {"log_loss": 0.55}, "base": {"log_loss": 0.56}, "model": {"log_loss": 0.551}}
        return {"matches": 1000, "market": n_market, "model": {"no_over_under": 1000 - n_market, "no_price": 0},
                "btts": b, "o15": ok, "o35": ok,
                "exact": {"shown": {"log_loss": 2.86, "top1_hit": 0.124}, "base": {"log_loss": 2.93}}}
    return {"periods": {"2013-2018": period(market), "2019-2025": period(market)}, "fit_max_miss": 0.04}


def test_a_group_whose_every_number_is_the_models_own_says_so_and_compares_nothing():
    text = sx.scores_caveat(_measure(0), scope="extra leagues")
    assert text.startswith("Every one of these numbers in the extra leagues is the model's own")
    assert "no over/under 2.5 price is posted for them" in text
    # no market matrix, no fit misses, no model-against-itself comparison
    assert "reproduces the market's" not in text and "fit no score matrix" not in text
    assert " against " not in text and "a little less accurate" not in text
    # what was measured stays: every number read from the measure
    assert "withheld and shown as missing" in text and "51.4%" in text and "53.0%" in text
    assert "about 2 points short" in text and "12.4%" in text
    # a group where the market's matrix was read keeps the comparison
    mixed = sx.scores_caveat(_measure(900), scope="top five")
    assert "reproduces the market's home/draw/away chances" in mixed
    assert "log loss 0.6870 against 0.6856" in mixed


def test_the_extra_leagues_caveat_in_the_run_is_the_models_own(run):
    m, *_ = run()
    # NOR and FIN have no over/under price anywhere: every number is the model's
    assert m["scores_measured"]["extra"]["periods"]
    assert all(p["market"] == 0 for p in m["scores_measured"]["extra"]["periods"].values() if p["matches"])
    assert m["groups"]["extra"]["scores_caveat"].startswith("Every one of these numbers in the extra leagues")
    assert "against" not in m["groups"]["extra"]["scores_caveat"]
    assert m["groups"]["top"]["scores_caveat"].startswith("Both teams to score, over 1.5")


# ------------------------------------------------------------- the weights

def test_a_raw_price_leagues_weights_are_none_as_shipped(run):
    _, _, perf, _, out = run()
    w = perf["backtest"]["weights"]
    assert w["FIN"] is None and slg.SHIPPED["FIN"]["pick"] == "raw"
    assert w["E0"] == {"season": 2026, "w": [round(v, 4) for v in out["weights"][("E0", 2026)]]}


# ------------------------------------------------------------- the badges

def _two_seasons():
    """A league whose namesake clubs ("Testford" and "Testford Athletic",
    both TES by their first letters, neither in soccer_teams.TEAMS) meet in
    one season only: Testford is relegated after 2024."""
    clubs = {2024: ["Testford", "Testford Athletic", "Arbury", "Bexwell"],
             2025: ["Testford Athletic", "Arbury", "Bexwell", "Cranby"]}
    rows = []
    for season, teams in clubs.items():
        for k, (h, a) in enumerate((h, a) for h in teams for a in teams if h != a):
            d = pd.Timestamp(f"{season}-08-10") + pd.Timedelta(days=7 * k)
            rows.append({"match_id": sd.match_id("SC1", d.date(), h, a), "league": "SC1", "season": season,
                         "date": d.strftime("%Y-%m-%d"), "kickoff": d + pd.Timedelta(hours=14), "home": h,
                         "away": a, "hg": 1, "ag": 0, "result": "H"})
    return sd.frame(rows)


def test_badge_codes_are_unique_over_every_season_and_do_not_move_with_the_season_shown():
    frame = _two_seasons()
    assert all(k not in st.TEAMS for k in ("Testford", "Testford Athletic"))
    codes = {s: sx.View(frame, {"SC1": s}).chip("SC1", "Testford Athletic")["code"] for s in (2024, 2025)}
    # one season's clubs alone would give Testford Athletic TES in 2025, where Testford is gone
    assert st.codes_for({"Testford Athletic", "Arbury", "Bexwell", "Cranby"})["Testford Athletic"]["code"] == "TES"
    assert codes[2024] == codes[2025] != "TES"
    view = sx.View(frame)
    assert view.chip("SC1", "Testford")["code"] == "TES"
    assert {view.chip("SC1", k)["code"] for k in ("Testford", "Testford Athletic", "Arbury", "Bexwell", "Cranby")} \
        == {d["code"] for d in st.by_league(frame)["SC1"].values()}
    # a club not on file takes a code free of the league's, whatever order it is asked in
    a, b = sx.View(frame), sx.View(frame)
    first = [a.chip("SC1", "Cranbrook")["code"], a.chip("SC1", "Cranley")["code"]]
    second = [b.chip("SC1", "Cranley")["code"], b.chip("SC1", "Cranbrook")["code"]][::-1]
    assert first == second and "CRA" not in first        # Cranby holds CRA


# ------------------------------------------------------------- the Record in summer

def _before(frame, day):
    return frame[frame["date"] < day].reset_index(drop=True)


def test_in_summer_the_top_fives_record_calibration_and_scores_caveat_do_not_go_empty(run):
    """Every summer the top five leave the app for weeks: no result in 45
    days and no fixture posted. Their pooled Record blocks, calibration and
    the global scores caveat are measured on the five whatever the day's
    league set is - the same numbers as in October - and the five keep
    their Record rows, marked out of the app."""
    run.state["fixtures"] = run.state["fixtures"].iloc[:0]
    m, _, perf, _, out = run("2026-07-15 12:00", results=_before(world(), "2026-07-15"))
    assert m["league_order"] == ["FIN", "NOR"] and "E0" in m["not_in_app"]
    assert out["on_record"] == ["E0", "FIN", "NOR", "SWE"]
    b = perf["backtest"]
    assert b["result"]["blend"]["2016-2025"]["matches"] > 0 and b["result"]["model"]["2006-2015"]["matches"] > 0
    assert b["calibration"] and b["over_2_5"]["2016-2025"]["matches"] > 0
    assert "scores" in m["caveats"] and "top" in m["scores_measured"] and m["scores_measured"]["periods"]
    by = {r["code"]: r for r in b["by_league"]}
    assert by["E0"]["in_app"] is False and by["E0"]["confirm"]["matches"] > 0 and by["NOR"]["in_app"] is True
    # ... and they are October's numbers, when E0 is in the app
    run.state["fixtures"] = __import__("test_soccer_predict_leagues").fixtures()
    m_oct, _, perf_oct, _, out_oct = run(results=world())
    assert "E0" in m_oct["league_order"]
    o = perf_oct["backtest"]
    assert b["result"] == o["result"] and b["calibration"] == o["calibration"] and b["over_2_5"] == o["over_2_5"]
    assert b["value"] == o["value"] and b["pooled"] == o["pooled"] and b["by_group"] == o["by_group"]
    assert m["caveats"]["scores"] == m_oct["caveats"]["scores"] and m["caveats"]["model"] == m_oct["caveats"]["model"]
    assert m["scores_measured"]["top"] == m_oct["scores_measured"]["top"]


# ------------------------------------------------------------- the kicked-off picks, per league

def _e0_after(after, through, seed=9):
    """E0's rounds after `after` through `through` (Arsenal v Chelsea left to its fixture)."""
    return [r for r in _rows("E0", [2026], "08-10", E0_TEAMS, "pre", per_day=5, end=through,
                             skip={("Arsenal", "Chelsea")}, seed=seed) if r["date"] > after]


def _played(fixture, hg, ag):
    return dict({k: v for k, v in fixture.items() if k != "fetched_at"}, hg=hg, ag=ag,
                result="H" if hg > ag else "A" if hg < ag else "D")


def _never_neither(m, results, log):
    """Every result in its league's latest round whose match has a logged
    pick is on screen, as the graded pick or as a result row - never
    neither."""
    shown = {e["match_id"] for e in m["matches"]} | {r["match_id"] for r in m["results"]}
    since = {c: (pd.Timestamp(v["results_through"]) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
             for c, v in m["leagues"].items() if v.get("results_through")}
    picked = {p["match_id"] for p in log["picks"].values() if not p.get("void")}
    for r in results.itertuples():
        if r.match_id in picked and r.league in since and r.date >= since[r.league]:
            assert r.match_id in shown, r.match_id


def test_a_league_whose_results_come_later_keeps_its_graded_pick_on_the_matches_tab(run):
    """E0's results land within a day; FIN's (no feed) days later. Its pick
    stays on the Matches tab while its round's result is to come, and is
    shown graded, with its score, when it lands - never dropped because E0
    moved the window on, and never hidden behind the pick."""
    fx = __import__("test_soccer_predict_leagues").fixtures()
    run()                                                   # Friday: the weekend's picks logged
    e0_ac, fin_hjk = fx.iloc[0].to_dict(), fx.iloc[2].to_dict()
    later = sd.frame(_e0_after("2026-10-20", "2026-11-09") + [_played(e0_ac, 2, 1)])
    results = pd.concat([world(), later], ignore_index=True)
    m, _, _, log, _ = run("2026-11-12 09:00", results=results)
    assert m["data_through"] >= "2026-11-09" and m["leagues"]["FIN"]["results_through"] < "2026-11-01"
    fin = {e["match_id"]: e for e in m["matches"]}.get(fin_hjk["match_id"])
    assert fin is not None and fin["status"] == "started"   # E0's window alone starts Nov 5: it was dropped
    _never_neither(m, results, log)
    # FIN's result lands two days later: graded, shown with its score, not listed twice
    results = pd.concat([results, sd.frame([_played(fin_hjk, 0, 1)])], ignore_index=True)
    m, _, _, log, _ = run("2026-11-14 09:00", results=results)
    fin = {e["match_id"]: e for e in m["matches"]}[fin_hjk["match_id"]]
    assert fin["status"] == "final" and fin["score"] == {"home": 0, "away": 1} and fin["correct"] is not None
    assert fin_hjk["match_id"] not in {r["match_id"] for r in m["results"]}
    _never_neither(m, results, log)


def test_a_result_whose_logged_pick_is_not_on_screen_is_still_listed(run, monkeypatch):
    """The results list hides a result only behind a pick on screen: a pick
    logged but not shown (here: outside every window) never hides it."""
    import soccer_predict as sp
    run()
    fx = __import__("test_soccer_predict_leagues").fixtures()
    fin_hjk = fx.iloc[2].to_dict()
    results = pd.concat([world(), sd.frame([_played(fin_hjk, 2, 2)])], ignore_index=True)
    # The windows shrunk to nothing: no kicked-off pick is on screen
    monkeypatch.setattr(sp, "BEHIND", pd.Timedelta(days=-30))
    m, _, _, log, _ = run("2026-11-02 09:00", results=results)
    assert log["picks"][fin_hjk["match_id"]]["final"]["result"] == "D"
    assert fin_hjk["match_id"] not in {e["match_id"] for e in m["matches"]}
    monkeypatch.setattr(sp, "BEHIND", pd.Timedelta(days=4))
    m, _, _, log, _ = run("2026-11-02 09:00", results=results)
    assert fin_hjk["match_id"] in {e["match_id"] for e in m["matches"]}
    assert fin_hjk["match_id"] not in {r["match_id"] for r in m["results"]}


# ------------------------------------------------------------- rule (a) over a break

def _espn(code, home, away, kickoff, seen="2026-12-19T10:00:00Z"):
    import soccer_feed as sfeed
    k = pd.Timestamp(kickoff)
    return sfeed._row(source="espn", source_id=f"{home}-{away}", request=f"espn:{code}:{k:%Y%m%d}", league=code,
                      season=sd.season_for(code, k), home=home, away=away, home_name=home, away_name=away,
                      kickoff_utc=f"{k:%Y-%m-%dT%H:%M:%SZ}", time_confirmed=True, uk_date=k.strftime("%Y-%m-%d"),
                      date_hint=k.strftime("%Y-%m-%d"), status="scheduled", status_detail="STATUS_SCHEDULED",
                      hg=None, ag=None, seen_at=seen)


def test_a_break_with_the_next_round_on_the_feeds_does_not_drop_a_league_and_not_in_app_says_why(run, tmp_path):
    """NOR's last result is ~12 weeks old and football-data.co.uk has posted
    nothing since November: out, and not_in_app says why. The feeds list
    its next match (Bodo/Glimt v Molde, which never was played on its
    November date): it is in the app, and that match is picked by the model
    alone."""
    import soccer_feed as sfeed
    m, *_ = run("2026-12-20 12:00")
    assert "NOR" not in m["league_order"]
    why = m["not_in_app"]["NOR"]
    assert why["caveats"] == slg.caveats("NOR") and why["season_check"]["fixtures_posted"] == 0
    assert why["season_check"]["last_result"] == world().query("league == 'NOR'")["date"].max()
    assert why["season_check"]["days_since"] > slg.RECENT_DAYS
    assert why["why"].startswith("Not in the app: no current season in the data") and "45 days" in why["why"]
    sfeed.save({"schema": sfeed.SCHEMA, "requests": {}, "last_run": {"at": "2026-12-19T10:00:00Z", "answered": 1,
                                                                        "failed": {}},
                "rows": [_espn("NOR", "Bodo/Glimt", "Molde", "2026-12-27 15:00")]}, tmp_path / "feed.json")
    m, _, perf, log, out = run("2026-12-20 12:00")
    assert "NOR" in m["league_order"] and "NOR" not in m["not_in_app"]
    back = by_home(m)["Bodo/Glimt"]
    assert back["status"] == "upcoming" and back["pick"] and back["basis"] == "model"
    assert back["price_status"] == "not_yet" and back["match_id"] in log["picks"]
    # the leagues still out say why; the Record keeps every decided league
    assert set(m["not_in_app"]) == {"E0", "FIN", "SWE"} and all("why" in v for v in m["not_in_app"].values())
    assert [r["code"] for r in perf["backtest"]["by_league"]] == ["E0", "FIN", "NOR", "SWE"]


# ------------------------------------------------------------- download problems

def test_footballdata_download_problems_are_read_off_the_real_sync(tmp_path, monkeypatch):
    """soccer_data.sync on a made-up site failing every way it can: each
    problem reaches sync_all's list per league, with what and why."""
    import soccer_feed as sfeed
    import soccer_predict as sp
    monkeypatch.setattr(sfeed, "sync", lambda **k: None)
    root = tmp_path / "matches"
    sd.store_season(root, "ARG", 2025, _rows("ARG", [2025], "02-01", NOR_TEAMS[:4], "close"), log=lambda *_: None)
    clock = [0.0]

    def fetch(url):
        if url == sd.FIXTURES_URL:
            raise ConnectionError("down")
        if url == sd.EXTRA_FIXTURES_URL:
            return b""                                        # no longer published
        if url.endswith("/E0.csv"):
            return b"Div,Date,HomeTeam,AwayTeam,FTHG\nE0,01/08/2026,Arsenal"       # cut off mid-row
        if url.endswith("/D1.csv"):
            raise TimeoutError()
        if url.endswith("/new/ARG.csv"):
            clock[0] = 10_000                                 # ... and then the budget is spent
            return b""
        raise AssertionError(f"asked after the budget was spent: {url}")
    lines = []
    got = sp.sync_all(pd.Timestamp("2026-11-06 19:30"), tmp_path / "feed.json", log=lines.append,
                      data={"leagues": ["E0", "D1", "ARG", "AUT"], "fetch": fetch, "matches_path": root,
                            "fixtures_path": tmp_path / "fixtures.csv", "clock": lambda: clock[0],
                            "today": "2026-11-06"})
    at = "2026-11-06T19:30:00Z"
    assert sorted((p["league"], p["what"], p["why"]) for p in got) == sorted([
        ("E0", "fixtures", "download failed (ConnectionError)"),
        ("D1", "fixtures", "download failed (ConnectionError)"),
        ("ARG", "extra-league fixtures", "not published (kept what is on file)"),
        ("AUT", "extra-league fixtures", "not published (kept what is on file)"),
        ("E0", "E0 2026", "cut off part-way through its last row"),
        ("D1", "D1 2026", "download failed (TimeoutError)"),
        ("ARG", "ARG", "not published (kept what is on file)"),
        ("AUT", "AUT", f"not asked: {sd.DOWNLOAD_BUDGET} s spent")])
    assert all(p["at"] == at for p in got)
    # a season the site has not started publishing is not a problem
    assert sp.data_problems(["   E0 2027: not published"], at, ["E0"], root) == []


def test_the_download_problems_go_into_matches_json(run, monkeypatch, tmp_path):
    import soccer_predict as sp
    problems = [{"league": "NOR", "what": "NOR", "why": "download failed (TimeoutError)", "at": "2026-10-30T18:00:00Z"}]
    monkeypatch.setattr(sp, "sync_all", lambda *a, **k: problems)
    sp.run(sync=True, now=pd.Timestamp("2026-10-30 18:00"), out_dir=tmp_path, log_path=tmp_path / "history.json",
           log=lambda *_: None, feed_path=tmp_path / "feed.json")
    feeds = json.loads((tmp_path / "matches.json").read_text())["feeds"]
    assert feeds["data_problems"] == problems and feeds["data_synced_at"] == "2026-10-30T18:00:00Z"
    m, *_ = run()                                             # no sync: none known, and said so
    assert m["feeds"]["data_problems"] == [] and m["feeds"]["data_synced_at"] is None
