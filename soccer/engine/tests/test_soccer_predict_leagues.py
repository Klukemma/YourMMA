"""The run switched to every league in the app, on a small made-up world:
the top five's E0, two calendar-year extra leagues (NOR; FIN, whose
pairings repeat in a season and which no feed covers) and SWE, measured but
with no current season - each with its own decisions (soccer_leagues).

The model is stood in for by a fixed rule, as in test_soccer_predict: these
check that every league reaches the payloads with its own season, dates,
decisions, withheld numbers and record, and that a repeated pairing is two
matches - not the model."""

import json

import numpy as np
import pandas as pd
import pytest

import soccer_data as sd
import soccer_export as sx
import soccer_feed as sfeed
import soccer_leagues as slg
import soccer_log as sl
import soccer_model as smod
import soccer_predict as sp
import soccer_scores as ssc

NOW = "2026-10-30 18:00"
E0_TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford", "Leeds", "Stoke"]
NOR_TEAMS = ["Bodo/Glimt", "Molde", "Brann", "Rosenborg", "Viking", "Lillestrom", "Tromso", "Sarpsborg"]
FIN_TEAMS = ["HJK", "Inter Turku", "KuPS", "SJK", "Ilves", "VPS"]
SWE_TEAMS = ["Malmo FF", "AIK", "Djurgarden", "Hammarby"]


def _rows(code, seasons, md, teams, price, per_day=None, end=None, skip=(), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    per_day = per_day or len(teams) // 2
    for season in seasons:
        day = pd.Timestamp(f"{season}-{md}")
        pairs = [(h, a) for h in teams for a in teams if h != a]
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // per_day))
            if end is not None and season == max(seasons) and date > pd.Timestamp(end):
                break
            if season == max(seasons) and (h, a) in skip:
                continue
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            r = {"match_id": sd.match_id(code, date.date(), h, a), "league": code, "season": season,
                 "date": date.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": date + pd.Timedelta(hours=14),
                 "home": h, "away": a, "hg": hg, "ag": ag, "result": "H" if hg > ag else "A" if hg < ag else "D"}
            if price == "pre":
                r.update(avg_h=2.2, avg_d=3.3, avg_a=3.4, max_h=2.3, max_d=3.5, max_a=3.6, avg_o25=1.9, avg_u25=1.9)
            else:                                        # an extra league's history: the close only
                r.update(avgc_h=2.2, avgc_d=3.3, avgc_a=3.4)
            out.append(r)
    return out


def world():
    rows = (_rows("E0", range(2005, 2027), "08-10", E0_TEAMS, "pre", per_day=5, end="2026-10-20",
                  skip={("Arsenal", "Chelsea")})
            + _rows("NOR", range(2012, 2027), "03-20", NOR_TEAMS, "close", per_day=2, end="2026-10-25", seed=1,
                    skip={("Bodo/Glimt", "Molde")})
            + _rows("FIN", range(2012, 2027), "04-05", FIN_TEAMS, "close", per_day=1, end="2026-10-25", seed=2)
            + _rows("SWE", range(2012, 2026), "04-01", SWE_TEAMS, "close", seed=3))
    return sd.frame(rows)


def _fixture(code, day, time, home, away, priced=True):
    k = pd.Timestamp(f"{day} {time}")
    r = {"match_id": sd.match_id(code, k.date(), home, away), "league": code, "season": sd.season_for(code, day),
         "date": day, "time": time, "kickoff": k, "home": home, "away": away}
    if priced:                                           # live fixtures carry the pre-match price, every league
        r.update(avg_h=1.8, avg_d=3.6, avg_a=4.5, max_h=1.9, max_d=3.8, max_a=4.8)
        if code in sd.TOP:
            r.update(avg_o25=1.8, avg_u25=2.0)
    return r


def fixtures():
    fx = sd.frame([_fixture("E0", "2026-10-31", "15:00", "Arsenal", "Chelsea"),
                   _fixture("NOR", "2026-11-01", "16:00", "Bodo/Glimt", "Molde"),
                   # FIN: HJK met Inter Turku in April - this is their next meeting, not a played match
                   _fixture("FIN", "2026-11-01", "13:00", "HJK", "Inter Turku"),
                   _fixture("FIN", "2026-11-01", "15:00", "KuPS", "SJK", priced=False)])
    fx["fetched_at"] = pd.Timestamp("2026-10-30 16:00")
    return fx


def _stub(frame):
    out = pd.DataFrame({"p_h": 0.47, "p_d": 0.27, "p_a": 0.26, "p_o25": 0.52, "lam": 1.5, "nu": 1.1, "rho": -0.06},
                       index=frame.index)
    first = frame.groupby("league")["season"].transform("min")
    out.loc[frame["season"] == first] = np.nan
    out.attrs["ratings"] = {t: {"league": "E0", "egd": 0.1} for t in E0_TEAMS}
    return out


@pytest.fixture
def run(tmp_path, monkeypatch):
    shipped = {"E0": slg.SHIPPED["E0"], "NOR": slg.SHIPPED["NOR"],
               # FIN as decided, but with no model-alone pick: an unpriced match waits for its price
               "FIN": dict(slg.SHIPPED["FIN"], model_alone=False), "SWE": slg.SHIPPED["SWE"]}
    monkeypatch.setattr(slg, "SHIPPED", shipped)
    monkeypatch.setattr(smod, "MIN_TRAIN", 300)
    state = {"matches": world(), "fixtures": fixtures()}
    asked = {}

    def load_matches(*a, **k):
        asked["matches"] = k.get("leagues")
        return state["matches"].copy()

    def load_fixtures(*a, **k):
        asked["fixtures"] = k.get("leagues")
        return state["fixtures"].copy()
    monkeypatch.setattr(sd, "load_matches", load_matches)
    monkeypatch.setattr(sd, "load_fixtures", load_fixtures)

    def model(frame, leagues=None, **k):
        asked["model_leagues"] = leagues
        return _stub(frame)
    monkeypatch.setattr(smod, "predict", model)

    def go(now=NOW, results=None):
        if results is not None:
            state["matches"] = results
        out = sp.run(sync=False, now=pd.Timestamp(now), out_dir=tmp_path, log_path=tmp_path / "history.json",
                     log=lambda *_: None, feed_path=tmp_path / "feed.json")
        read = lambda n: json.loads((tmp_path / f"{n}.json").read_text())
        return read("matches"), read("tables"), read("performance"), sl.load(tmp_path / "history.json"), out
    go.asked, go.state = asked, state
    return go


def by_home(m):
    return {e.get("home_key") or e["home"]["name"]: e for e in m["matches"]}


def test_every_league_with_a_current_season_is_run_and_shown_with_its_own_decisions(run):
    m, tables, perf, log, out = run()
    assert run.asked == {"matches": "all", "fixtures": "all", "model_leagues": slg.MODEL_LEAGUES}
    # SWE was measured but its season is over: out of the app, said so. In
    # soccer_data's order.
    assert m["league_order"] == out["leagues"] == ["E0", "FIN", "NOR"]
    assert set(m["leagues"]) == {"E0", "NOR", "FIN"} and set(m["not_in_app"]) == {"SWE"}
    assert m["not_in_app"]["SWE"]["caveats"] == slg.caveats("SWE")
    assert {g: {k: v for k, v in x.items() if k != "scores_caveat"} for g, x in m["groups"].items()} == {
        "top": {"name": "top five", "codes": ["E0"]}, "extra": {"name": "extra leagues", "codes": ["FIN", "NOR"]}}
    # each group's scores caveat from its own measure, scoped; the global one stays
    assert "in the extra leagues" in m["groups"]["extra"]["scores_caveat"]
    assert "withheld and shown as missing" in m["groups"]["extra"]["scores_caveat"]
    assert "Both teams to score runs low" in m["caveats"]["scores"]
    for code in ("E0", "NOR", "FIN"):
        lg = m["leagues"][code]
        assert lg["caveats"] == slg.caveats(code) and lg["group"] == slg.group_of(code)
        assert lg["pick"] == slg.SHIPPED[code]["pick"] and lg["withheld"]["model"] == list(slg.withheld(code, "model"))
        assert lg["coverage"] == sfeed.coverage(code) and lg["coverage_caveat"] == sfeed.coverage_caveat(code)
    # each on its own calendar and its own latest result
    assert (m["leagues"]["E0"]["season"], m["leagues"]["E0"]["season_label"]) == (2026, "2026-27 season")
    assert (m["leagues"]["NOR"]["season"], m["leagues"]["NOR"]["season_label"]) == (2026, "2026 season")
    assert m["leagues"]["FIN"]["coverage_caveat"] == sfeed.COVERAGE_CAVEATS["none"]
    through = {c: run.state["matches"].query("league == @c")["date"].max() for c in ("E0", "NOR", "FIN")}
    assert {c: m["leagues"][c]["model_through"] for c in through} == through
    assert m["data_through"] == m["model_through"] == max(through.values())
    # the bands each card is read against, per group, and the floors
    assert m["bands"]["extra"]["period"] == "2019-2025" and m["bands"]["top"]["period"] == "2016-2025"
    assert m["bands"]["extra"]["pick"] == slg.BANDS["extra"]["pick"]
    assert m["band_floors"] == {name: floor for name, floor in ssc.BANDS}
    # the Method tab's measure, per group; the top five's where the app read it before
    sm = m["scores_measured"]
    assert set(sm["top"]["periods"]) == {"2006-2015", "2016-2025"}
    assert set(sm["extra"]["periods"]) == {"2013-2018", "2019-2025"}
    assert sm["periods"] == sm["top"]["periods"] and sm["bands"] == sm["top"]["bands"]
    # tables: each league with its own season, caveats and what its table is not
    t = {x["code"]: x for x in tables["leagues"]}
    assert set(t) == {"E0", "NOR", "FIN"} and t["NOR"]["season_label"] == "2026 season"
    assert t["FIN"]["table_note"] == sx.TABLE_NOTES["FIN"] and t["E0"]["table_note"] is None
    assert t["FIN"]["caveats"] == slg.caveats("FIN")


def test_each_entry_carries_its_basis_and_withholds_what_its_league_did_not_ship(run):
    m, _, _, log, _ = run()
    ups = by_home(m)
    e0, nor, fin, bare = ups["Arsenal"], ups["Bodo/Glimt"], ups["HJK"], ups["KuPS"]
    assert e0["basis"] == "recalibrated" and e0["price_kind"] == "pre" and e0["p_o25"] is not None
    assert fin["basis"] == "raw" and fin["pick"] and fin["price_kind"] == "pre"
    # NOR's model did not ship both teams to score or over 2.5: null, and listed
    assert nor["score_source"] == "model" and nor["p_btts"] is None and nor["p_o25"] is None
    assert nor["model_o25"] is None and set(nor["scores"]["withheld"]) == {"btts", "o25"}
    assert nor["p_o15"] is not None and nor["scores"]["top"]
    assert "withheld" not in e0["scores"]
    # FIN gives no model-alone pick (as decided here): no price, no pick, said so
    assert bare["pick"] is None and bare["basis"] == "unpriced"
    assert bare["no_pick_reason"].startswith("No pick until the price is posted")
    assert bare["match_id"] not in log["picks"]
    for e in (e0, nor, fin):
        p = log["picks"][e["match_id"]]
        assert (p["basis"], p["price_kind"], p["group"]) == (e["basis"], e["price_kind"], slg.group_of(e["league"]))
        assert p["tier"] == sd.ALL_LEAGUES[e["league"]].tier
        assert p["scores"] == e["scores"]
    # every NaN-free, the badges unique within each league
    assert "NaN" not in json.dumps(m)
    for code in ("NOR", "FIN"):
        cards = [e for e in m["matches"] if e["league"] == code]
        assert all(e["home"]["code"] != e["away"]["code"] for e in cards)


def test_a_repeated_pairing_is_picked_and_graded_on_its_own_meeting(run):
    """FIN plays HJK v Inter Turku more than once a season: April's result is
    another match. The November meeting is listed, picked, logged and - its
    result in - graded on that result, never voided on April's."""
    m, _, _, log, _ = run()
    hjk = by_home(m)["HJK"]
    april = run.state["matches"].query("league == 'FIN' and season == 2026 and home == 'HJK' and away == 'Inter Turku'")
    assert len(april) == 1 and april["date"].iloc[0] < "2026-06-01"
    p = log["picks"][hjk["match_id"]]
    assert not p.get("void") and "final" not in p
    played = pd.concat([run.state["matches"], sd.frame([dict(
        {k: v for k, v in fixtures().iloc[2].to_dict().items() if k != "fetched_at"}, hg=0, ag=1, result="A")])],
        ignore_index=True)
    m, _, perf, log, _ = run("2026-11-02 09:00", results=played)
    p = log["picks"][hjk["match_id"]]
    assert p["final"]["result"] == "A" and not p.get("void")
    shown = {e["match_id"]: e for e in m["matches"]}[hjk["match_id"]]
    assert shown["status"] == "final" and shown["score"] == {"home": 0, "away": 1}
    rec = perf["track_record"]
    assert rec["by_league"]["FIN"]["settled"] == 1 and rec["by_group"]["extra"]["settled"] == 1
    assert rec["by_basis"]["raw"]["settled"] == 1 and rec["settled"] == 1
    # April's result is in the results list only if it is in the league's last round: never twice
    assert hjk["match_id"] not in {r["match_id"] for r in m["results"]}


def test_the_record_tab_scores_every_league_on_its_own_periods(run):
    _, _, perf, _, out = run()
    b = perf["backtest"]
    assert [r["code"] for r in b["by_league"]] == ["E0", "FIN", "NOR"]
    nor = next(r for r in b["by_league"] if r["code"] == "NOR")
    assert nor["periods"] == {"dev": [2013, 2018], "confirm": [2019, 2025]} and nor["price"] == "closing"
    assert nor["matches"] == nor["confirm"]["matches"] > 0 and nor["accuracy"] == nor["confirm"]["accuracy"]
    assert b["result"]["scope"] == b["over_2_5"]["scope"] == b["value"]["scope"] == b["calibration_scope"] == "top five"
    assert set(b["by_group"]) == {"top", "extra"}
    assert set(b["weights"]) == {"E0", "NOR", "FIN"} and b["weights"]["FIN"]["w"] is None      # FIN's pick is raw
    assert b["weights"]["E0"]["season"] == 2026


def test_the_sync_asks_the_feeds_about_the_apps_leagues_a_feed_covers_and_never_rus(monkeypatch):
    seen = {}
    monkeypatch.setattr(sd, "sync", lambda *a, **k: None)
    monkeypatch.setattr(sfeed, "sync", lambda **k: seen.update(k))
    sp.sync_all(None, "feed.json", log=lambda *_: None)
    assert seen["leagues"] == sfeed.leagues_for(slg.in_app())
    assert "RUS" not in seen["leagues"] and len(seen["leagues"]) == 30 and set(seen["leagues"]) >= set(sd.TOP)
    assert len(sfeed.requests_for(pd.Timestamp(NOW), leagues=seen["leagues"])) == 462


def test_a_calendar_year_league_between_seasons_shows_the_one_it_finished():
    frame = sd.frame(_rows("NOR", [2025, 2026], "03-20", NOR_TEAMS, "close")
                     + _rows("E0", [2025, 2026], "08-10", E0_TEAMS, "pre", end="2026-10-20"))
    seasons = sp.league_seasons(frame, ["NOR", "E0"], pd.Timestamp("2027-02-10"))
    # in February NOR's 2027 season has not started; E0's 2026-27 is on
    assert seasons == {"NOR": 2026, "E0": 2026}
    assert sx.season_label_of("NOR", 2026) == "2026 season" and sx.season_label_of("E0", 2026) == "2026-27 season"
