"""The same-day rule, its day, the live builder's clock and lock, and grading."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import build_sameday as bs  # noqa: E402
import sameday as sdy  # noqa: E402


def L(sport, key, p, dec, p_market=None, day="2026-11-08", kickoff="2026-11-08T18:00:00Z", result=None):
    return sdy.leg(sport, key, day, f"{key} event", f"{key} pick", p, dec,
                   p_market=p if p_market is None else p_market, kickoff=kickoff, result=result)


def test_a_ticket_pays_3_to_5x_with_at_most_six_legs_one_per_event():
    legs = [L("NFL", f"g{i}", 0.70, 1.40) for i in range(8)] + [L("Soccer", "m1", 0.55, 1.80)]
    t = sdy.build(legs)
    assert t is not None and 3.0 <= t["multiplier"] <= 5.0 and t["n_legs"] <= 6
    assert len({l["key"] for l in t["legs"]}) == t["n_legs"]


def test_legs_under_half_a_chance_or_without_a_price_are_never_used():
    assert sdy.leg("NFL", "x", "2026-11-08", "e", "p", 0.7, None) is None
    assert sdy.leg("NFL", "x", "2026-11-08", "e", "p", 0.7, 1.0) is None
    t = sdy.build([L("NFL", "a", 0.45, 3.5), L("NFL", "b", 0.40, 4.0)])
    assert t is None


def test_the_likeliest_combination_wins_and_ties_go_to_fewer_legs():
    a, b = L("NFL", "a", 0.62, 1.75), L("NFL", "b", 0.62, 1.75)      # 3.06x, 0.384
    c, d, e = (L("Soccer", k, 0.70, 1.45) for k in "cde")               # 3.05x, 0.343
    t = sdy.build([a, b, c, d, e])
    assert {l["key"] for l in t["legs"]} == {"NFL:a", "NFL:b"}
    # exactly tied products: the two-leg ticket, not the four-leg one
    x = [L("UFC", "x", 0.64, 1.75), L("UFC", "y", 0.64, 1.75)]
    y = [L("NFL", f"q{i}", 0.8, 1.32) for i in range(4)]                 # 0.4096 = 0.64^2
    assert sdy.build(x + y)["n_legs"] == 2


def test_the_market_only_control_reads_no_model():
    legs = [L("NFL", "a", 0.90, 1.75, p_market=0.55), L("NFL", "b", 0.90, 1.75, p_market=0.55),
            L("NFL", "c", 0.55, 1.75, p_market=0.60), L("NFL", "d", 0.55, 1.75, p_market=0.60)]
    m = sdy.build(legs, objective="p_market")
    assert {l["key"] for l in m["legs"]} == {"NFL:c", "NFL:d"}
    assert {l["key"] for l in sdy.build(legs)["legs"]} == {"NFL:a", "NFL:b"}


def test_the_day_is_eastern_so_night_games_and_late_cards_stay_on_their_day():
    assert sdy.et_day("2026-11-06T01:15:00Z") == "2026-11-05"        # Thursday night football
    assert sdy.et_day("2026-11-01T00:00:00Z") == "2026-10-31"        # a UFC card's first bell, 20:00 EDT
    assert sdy.et_day("2026-11-08T14:30:00Z") == "2026-11-08"        # a London game, morning ET


def test_void_and_unknown_legs_settle_as_books_do():
    won = L("NFL", "a", 0.6, 2.0, result={"winner": "pick"})
    void = L("NFL", "b", 0.6, 1.8, result={"winner": None})
    lost = L("NFL", "c", 0.6, 1.8, result={"winner": "other"})
    open_ = L("NFL", "d", 0.6, 1.8, result=None)
    assert sdy.settle({"legs": [won, void]}) == {"status": "won", "paid": 2.0, "profit": 1.0}
    assert sdy.settle({"legs": [won, open_]})["status"] == "open"
    assert sdy.settle({"legs": [open_, lost]})["status"] == "lost"


NOW = datetime(2026, 11, 8, 17, 0, tzinfo=timezone.utc)


def test_a_started_game_is_never_a_leg_whatever_its_status_says():
    week = {"games": [
        {"game_id": "early", "kickoff": "2026-11-08T14:30:00Z", "status": "upcoming", "home": "ATL",
         "away": "CIN", "pick": "CIN", "pick_prob": 0.7, "model_prob": 0.3, "market_prob": 0.32,
         "market": {"home_ml": 150, "away_ml": -170}},
        {"game_id": "late", "kickoff": "2026-11-08T18:00:00Z", "status": "upcoming", "home": "IND",
         "away": "DAL", "pick": "DAL", "pick_prob": 0.69, "model_prob": 0.47, "market_prob": 0.32,
         "market": {"home_ml": 200, "away_ml": -240}}]}
    legs = bs.nfl_legs(week, NOW)
    assert [l["names"]["game_id"] for l in legs] == ["late"]
    assert legs[0]["decimal"] == pytest.approx(1 + 100 / 240)
    assert legs[0]["p_market"] == pytest.approx(0.68) and legs[0]["p_model"] == pytest.approx(0.53)


def test_soccer_is_priced_at_the_average_never_the_best_and_unpriced_rounds_are_skipped():
    data = {"matches": [
        {"match_id": "m1", "league": "E0", "kickoff": "2026-11-08T19:00:00Z", "status": "upcoming",
         "pick": "H", "pick_prob": 0.7, "price_status": "posted", "market": {"p_h": 0.68},
         "prices": {"avg": {"h": 1.40, "d": 4.5, "a": 7.0}, "best": {"h": 1.48}},
         "home": {"name": "Arsenal"}, "away": {"name": "Hull City"}},
        {"match_id": "m2", "league": "E0", "kickoff": "2026-11-08T19:00:00Z", "status": "upcoming",
         "pick": "H", "pick_prob": 0.7, "price_status": "not_yet", "home": {"name": "A"}, "away": {"name": "B"}}]}
    legs = bs.soccer_legs(data, NOW)
    assert len(legs) == 1 and legs[0]["decimal"] == 1.40 and legs[0]["pick"] == "Arsenal"


def _row(day, first_start, legs=(("NFL:g", 1.5),)):
    return {"day": day, "built_at": "2026-11-08T12:00:00Z", "status": "pending", "multiplier": 3.2,
            "first_start": first_start,
            "legs": [{"sport": "NFL", "key": k, "decimal": d, "pick": "X", "names": {}} for k, d in legs]}


def test_a_ticket_is_locked_at_its_first_start_and_never_replaced():
    history = {"tickets": [_row("2026-11-08", "2026-11-08T14:30:00Z")]}
    legs = [L("NFL", f"g{i}", 0.7, 1.45, kickoff="2026-11-08T21:00:00Z") for i in range(5)]
    bs.update(history, NOW, legs)
    assert len([r for r in history["tickets"] if r["day"] == "2026-11-08"]) == 1


def test_before_its_first_start_a_run_that_builds_nothing_supersedes_with_a_none_row():
    history = {"tickets": [_row("2026-11-08", "2026-11-08T21:00:00Z")]}
    bs.update(history, NOW, [])
    assert bs.latest(history, "2026-11-08")["status"] == "none"
    graded = bs.grade_all(history)
    assert graded == [] and history["tickets"][0]["status"] == "superseded"


def test_the_page_payload_has_no_nan_and_says_why_a_day_has_no_ticket(monkeypatch):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    history = {"tickets": []}
    show = bs.update(history, NOW, [])
    out = bs.payload(history, show, NOW, bs.grade_all(history))
    text = json.dumps(out, allow_nan=False)
    assert out["tickets"][0]["status"] == "none" and "3-5x" in out["tickets"][0]["missing_reason"]
    assert "hard to beat" in text


def test_grading_reads_each_sport_and_leaves_unknown_results_open():
    res = {"ufc": None, "nfl": {"g1": 7.0, "g2": 0.0}, "soccer": {"m1": "D", "m2": "void"}}
    nfl_home = {"sport": "NFL", "pick": "X", "names": {"game_id": "g1", "home": True}}
    nfl_tie = {"sport": "NFL", "pick": "X", "names": {"game_id": "g2", "home": False}}
    soc = {"sport": "Soccer", "pick": "Draw", "names": {"match_id": "m1", "pick": "D"}}
    soc_void = {"sport": "Soccer", "pick": "A", "names": {"match_id": "m2", "pick": "H"}}
    soc_open = {"sport": "Soccer", "pick": "A", "names": {"match_id": "m9", "pick": "H"}}
    ufc = {"sport": "UFC", "pick": "A", "names": {"red": "A", "blue": "B", "event_date": "2026-11-07"}}
    assert bs.leg_result(nfl_home, res) == {"winner": "pick"}
    assert bs.leg_result(nfl_tie, res) == {"winner": None}
    assert bs.leg_result(soc, res) == {"winner": "pick"}
    assert bs.leg_result(soc_void, res) == {"winner": None}
    assert bs.leg_result(soc_open, res) is None
    assert bs.leg_result(ufc, res) is None


def test_soccer_legs_come_only_from_the_leagues_the_rule_was_measured_on():
    """The soccer app shows 37 leagues; the same-day rule was measured on the
    top five, so a Norwegian match is never a leg."""
    m = {"match_id": "x", "kickoff": "2026-11-08T19:00:00Z", "status": "upcoming", "pick": "H",
         "pick_prob": 0.7, "price_status": "posted", "market": {"p_h": 0.68},
         "prices": {"avg": {"h": 1.40}}, "home": {"name": "A"}, "away": {"name": "B"}}
    assert len(bs.soccer_legs({"matches": [dict(m, league="E0")]}, NOW)) == 1
    assert bs.soccer_legs({"matches": [dict(m, league="NOR")]}, NOW) == []


# ---------------------------------------------------------------- the soccer log, whole

SOCCER_ENGINE = ENGINE.parents[1] / "soccer" / "engine"


def _soccer_pick(mid, kickoff, **kw):
    p = {"match_id": mid, "league": "E0", "season": 2026, "kickoff": kickoff, "home": "Arsenal",
         "away": "Chelsea", "pick": "H", "pick_prob": 0.6}
    p.update(kw)
    return p


def _save_with_soccer_log(log, path):
    """Write `log` with soccer_log.save itself, in its own process on a clean
    import path (the soccer and UFC engines share package names), so the
    split is the one the soccer runner writes."""
    import os
    import subprocess
    src = path.parent / "log_in.json"
    src.write_text(json.dumps(log))
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import soccer_log as sl; "
            "sl.save(json.loads(open(sys.argv[2]).read()), sys.argv[3]); "
            "print(json.dumps({'picks': sorted(sl.load(sys.argv[3])['picks']), 'late': sl.LATE, "
            "'record': sl.FD, 'schema': sl.SCHEMA}))")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SOCCER_PREDICTIONS_LOG=str(path))
    env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, "-c", code, str(SOCCER_ENGINE), str(src), str(path)],
                         capture_output=True, text=True, env=env, cwd=path.parent, check=True)
    return json.loads(out.stdout)


def test_a_settled_ticket_stays_settled_once_its_soccer_leg_is_archived(tmp_path, monkeypatch):
    """soccer_log.save moves a pick settled for good to prediction_history/
    YYYY-MM.json once its kickoff is 28 days before the newest pick's, and
    every run grades every ticket again: the grader reads the whole log."""
    path = tmp_path / "prediction_history.json"
    old = _soccer_pick("E0-20261101-arsenal-chelsea", "2026-11-01T15:00:00Z",
                       final={"home": 2, "away": 0, "result": "H"}, correct=True)
    gone = _soccer_pick("E0-20261102-leeds-stoke", "2026-11-02T15:00:00Z", home="Leeds", away="Stoke",
                        void="no result")
    new = _soccer_pick("E0-20261210-everton-fulham", "2026-12-10T20:00:00Z", home="Everton", away="Fulham")
    log = {"schema": 1, "picks": {p["match_id"]: p for p in (old, gone, new)}}
    saved = _save_with_soccer_log(log, path)
    # the names this reader mirrors are soccer_log's own
    assert (saved["late"], saved["record"], saved["schema"]) == (bs.SOCCER_LATE, bs.SOCCER_RECORD,
                                                                 bs.SOCCER_LOG_SCHEMA)
    main = json.loads(path.read_text())["picks"]
    assert sorted(main) == [new["match_id"]]                         # the split did happen
    assert (tmp_path / "prediction_history" / "2026-11.json").exists()
    assert sorted(bs.soccer_picks(path)) == saved["picks"]           # what soccer_log.load reads
    assert bs.soccer_results(path) == {old["match_id"]: "H", gone["match_id"]: "void"}

    leg = {"sport": "Soccer", "key": "Soccer:" + old["match_id"], "pick": "Arsenal", "decimal": 3.2,
           "names": {"match_id": old["match_id"], "pick": "H"}}
    history = {"tickets": [{"day": "2026-11-01", "status": "won", "multiplier": 3.2, "paid": 3.2, "profit": 2.2,
                            "first_start": "2026-11-01T15:00:00Z", "legs": [leg]}]}
    monkeypatch.setattr(bs, "SOCCER_LOG", path)
    monkeypatch.setattr(bs, "UFC_ARCHIVE", tmp_path / "none.csv")
    monkeypatch.setattr(bs, "NFL_GAMES", tmp_path / "none.csv.gz")
    graded = bs.grade_all(history)
    assert graded[0]["status"] == "won" and graded[0]["profit"] == pytest.approx(2.2)
    assert bs.record(graded)["settled"] == 1 and bs.record(graded)["pending"] == 0


def test_a_soccer_log_file_that_does_not_read_stops_the_run(tmp_path):
    """Read as empty, it would send every ticket with a leg in it back to
    pending and write that down."""
    path = tmp_path / "prediction_history.json"
    path.write_text('{"picks": {}, "schema": 1}')
    (tmp_path / "prediction_history").mkdir()
    (tmp_path / "prediction_history" / "2026-10.json").write_text('{"picks": {')
    with pytest.raises(ValueError):
        bs.soccer_picks(path)
    assert bs.soccer_picks(tmp_path / "missing.json") == {}           # no soccer yet: no picks


def test_a_void_soccer_pick_is_a_void_leg_unless_football_data_may_restore_it(tmp_path):
    """soccer_log writes "void" (it never writes "status"); a pick voided as
    logged after kickoff on the feeds' time waits for football-data.co.uk's."""
    k = "2026-11-08T15:00:00Z"
    picks = {
        "off": _soccer_pick("off", k, void="no result"),
        "moved": _soccer_pick("moved", k, void="rescheduled"),
        "dup": _soccer_pick("dup", k, void="duplicate"),
        "late_fd": _soccer_pick("late_fd", k, void="logged after kickoff", void_by="football-data.co.uk"),
        "late_feed": _soccer_pick("late_feed", k, void="logged after kickoff", void_by="ESPN"),
        "open": _soccer_pick("open", k),
        "prov": _soccer_pick("prov", k, final={"home": 1, "away": 1, "result": "D", "provisional": True}),
    }
    path = tmp_path / "prediction_history.json"
    path.write_text(json.dumps({"schema": 1, "picks": picks}))
    res = bs.soccer_results(path)
    assert res == {"off": "void", "moved": "void", "dup": "void", "late_fd": "void", "prov": "D"}
    leg = {"sport": "Soccer", "pick": "A", "names": {"match_id": "off", "pick": "A"}}
    assert bs.leg_result(leg, {"soccer": res}) == {"winner": None}
    late = dict(leg, names={"match_id": "late_feed", "pick": "A"})
    assert bs.leg_result(late, {"soccer": res}) is None
    # and a ticket whose other leg won settles on it, the void leg at 1.0
    t = sdy.settle({"legs": [L("NFL", "a", 0.6, 2.0, result={"winner": "pick"}),
                             L("Soccer", "off", 0.6, 1.8, result=bs.leg_result(leg, {"soccer": res}))]})
    assert t == {"status": "won", "paid": 2.0, "profit": 1.0}


def test_a_moved_soccer_match_is_graded_under_its_new_id(tmp_path):
    """soccer_log moves a pick to the id its match changed to ("id_was" the
    first): a leg keyed by the old id reads the result there."""
    picks = {"F1-20261101-lyon-angers": _soccer_pick(
        "F1-20261101-lyon-angers", "2026-11-01T16:15:00Z", league="F1", id_was="F1-20261031-lyon-angers",
        final={"home": 2, "away": 0, "result": "H"})}
    path = tmp_path / "prediction_history.json"
    path.write_text(json.dumps({"schema": 1, "picks": picks}))
    res = bs.soccer_results(path)
    assert res == {"F1-20261101-lyon-angers": "H", "F1-20261031-lyon-angers": "H"}


# ---------------------------------------------------------------- the underdog ticket, live

def _card(first_start="2026-11-14T23:00:00Z", fights=()):
    return {"event": {"date": "2026-11-14", "name": "UFC X", "first_start": first_start}, "fights": list(fights)}


def _fight(red, blue, pick, win_prob, odds, odds_other, median=None, books=None, model=None):
    f = {"number": 1, "red": red, "blue": blue, "pick": pick, "win_prob": win_prob,
         "win_prob_model": model, "odds": odds, "odds_other": odds_other,
         "decimal": sdy.pp.decimal_from_american(odds), "decimal_other": sdy.pp.decimal_from_american(odds_other)}
    if median:
        f.update(odds_median=median[0], odds_other_median=median[1], books_median=books,
                 decimal_median=sdy.pp.decimal_from_american(median[0]),
                 decimal_other_median=sdy.pp.decimal_from_american(median[1]))
    return f


THU = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)


def test_a_ufc_underdog_is_the_other_corner_at_the_median_pair_when_the_card_has_one():
    fav = _fight("Ciryl Gané", "Josh Hokit", "Ciryl Gané", 0.67, -195, 185, median=(-210, 170), books=4,
                 model=0.71)
    (d,) = bs.ufc_dogs(_card(fights=[fav]), THU)
    assert d["pick"] == "Josh Hokit" and d["dog"] and d["day"] == "2026-11-14"
    assert d["decimal"] == pytest.approx(2.70) and d["price_source"] == "median of 4 books"
    ia, ib = 1 / (1 + 100 / 210), 1 / 2.70
    assert d["p_market"] == pytest.approx(ib / (ia + ib))           # no-vig at the pair it is priced at
    assert d["p"] == pytest.approx(0.33) and d["p_model"] == pytest.approx(0.29)
    assert d["key"] == "UFC:2026-11-14:Ciryl Gané:Josh Hokit"       # the leg's key: the fill skips the bout
    assert d["lot_key"] == "ciryl gane|josh hokit"
    assert d["names"] == {"red": "Ciryl Gané", "blue": "Josh Hokit", "event_date": "2026-11-14"}


def test_without_a_median_from_two_books_a_ufc_dog_is_priced_at_the_best_and_flagged():
    one = _fight("A Red", "B Blue", "A Red", 0.70, -250, 210, median=(-250, 210), books=1)
    none_ = _fight("C Red", "D Blue", "C Red", 0.70, -250, 210)
    a, b = bs.ufc_dogs(_card(fights=[one, none_]), THU)
    assert a["price_source"] == b["price_source"] == "best of books" and a["decimal"] == pytest.approx(3.10)


def test_a_ufc_pick_the_market_rates_the_underdog_is_itself_the_dog_and_level_prices_give_none():
    picked_dog = _fight("E Red", "F Blue", "F Blue", 0.52, 120, -140, model=0.6)
    level = _fight("G Red", "H Blue", "G Red", 0.55, -110, -110)
    (d,) = bs.ufc_dogs(_card(fights=[picked_dog, level]), THU)
    assert d["pick"] == "F Blue" and d["p"] == pytest.approx(0.52) and d["decimal"] == pytest.approx(2.20)
    assert d["p_market"] < 0.5 and d["p_model"] == pytest.approx(0.6)


def test_a_started_card_gives_no_underdog():
    f = _fight("A Red", "B Blue", "A Red", 0.70, -250, 210)
    assert bs.ufc_dogs(_card("2026-11-12T15:00:00Z", [f]), THU) == []
    assert bs.ufc_legs(_card("2026-11-12T15:00:00Z", [f]), THU) == []


def _game(gid, kickoff, home, away, pick, pick_prob, home_ml, away_ml, model_prob=0.5):
    return {"game_id": gid, "kickoff": kickoff, "status": "upcoming", "home": home, "away": away,
            "pick": pick, "pick_prob": pick_prob, "model_prob": model_prob, "market_prob": 0.5,
            "market": {"home_ml": home_ml, "away_ml": away_ml}}


def test_an_nfl_underdog_is_priced_at_the_consensus_and_its_chance_is_one_less_the_picks():
    week = {"games": [
        _game("2026_10_WAS_NYG", "2026-11-13T01:15:00Z", "NYG", "WAS", "NYG", 0.555, -128.5, 108, 0.573),
        _game("2026_10_ATL_DEN", "2026-11-15T18:00:00Z", "DEN", "ATL", "ATL", 0.53, -140, 120, 0.40),
        _game("started", "2026-11-12T15:00:00Z", "KC", "BUF", "KC", 0.6, -200, 170),
        _game("level", "2026-11-15T18:00:00Z", "SF", "SEA", "SF", 0.5, -110, -110)]}
    a, b = bs.nfl_dogs(week, THU)
    assert (a["pick"], a["names"]) == ("WAS", {"game_id": "2026_10_WAS_NYG", "home": False})
    assert a["p"] == pytest.approx(1 - 0.555) and a["decimal"] == pytest.approx(2.08)
    assert a["p_model"] == pytest.approx(1 - 0.573) and a["day"] == "2026-11-12"
    assert a["lot_key"] == "2026_10_WAS_NYG" and a["price_source"] == "consensus moneyline"
    # the pick is the market's underdog: it is the dog, at the pick's chance
    assert (b["pick"], b["p"], b["names"]["home"]) == ("ATL", pytest.approx(0.53), False)
    assert b["p_model"] == pytest.approx(0.60)
    # graded as a leg: ATL (away) wins when the home margin is negative
    assert bs.leg_result(b, {"nfl": {"2026_10_ATL_DEN": -3.0}}) == {"winner": "pick"}


def _match(mid, league="E0", kickoff="2026-11-14T15:00:00Z", qh=0.30, qd=0.27, qa=0.43, ph=0.28, pa=0.45,
           avg=(3.2, 3.5, 2.25), posted="posted"):
    return {"match_id": mid, "league": league, "kickoff": kickoff, "status": "upcoming", "pick": "A",
            "pick_prob": pa, "p_h": ph, "p_d": 1 - ph - pa, "p_a": pa, "price_status": posted,
            "market": {"p_h": qh, "p_d": qd, "p_a": qa},
            "prices": {"avg": {"h": avg[0], "d": avg[1], "a": avg[2]}},
            "home": {"name": "Hull City"}, "away": {"name": "Arsenal"}}


def test_a_soccer_underdog_is_the_less_likely_team_never_the_draw():
    # the draw is the least likely outcome here, and still never the dog
    data = {"matches": [_match("m1", qh=0.30, qd=0.25, qa=0.45),
                        _match("level", qh=0.36, qd=0.28, qa=0.36),
                        _match("nor", league="NOR"),
                        _match("gone", kickoff="2026-11-12T12:00:00Z"),
                        _match("unpriced", posted="not_yet")]}
    (d,) = bs.soccer_dogs(data, THU)
    assert d["pick"] == "Hull City" and d["names"] == {"match_id": "m1", "pick": "H"}
    assert d["decimal"] == 3.2 and d["p"] == pytest.approx(0.28) and d["p_market"] == pytest.approx(0.30)
    assert d["lot_key"] == "m1" and d["price_source"] == "average price"
    assert bs.leg_result(d, {"soccer": {"m1": "H"}}) == {"winner": "pick"}
    assert bs.leg_result(d, {"soccer": {"m1": "D"}}) == {"winner": "other"}


def _dogday(now=THU):
    """A Thursday with one NFL game (no ticket fits) and a Saturday card."""
    card = _card(fights=[_fight(f"R{i}", f"B{i}", f"R{i}", 0.75, -300, 250, median=(-320, 240), books=3)
                         for i in range(4)])
    week = {"games": [_game("2026_10_WAS_NYG", "2026-11-13T01:15:00Z", "NYG", "WAS", "NYG", 0.555, -128.5, 108)]}
    legs = bs.ufc_legs(card, now) + bs.nfl_legs(week, now)
    dogs = bs.ufc_dogs(card, now) + bs.nfl_dogs(week, now)
    return legs, dogs


def test_a_thursday_with_one_game_still_shows_saturdays_tickets(monkeypatch):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    legs, dogs = _dogday()
    history, why = {"tickets": []}, {}
    show = bs.update(history, THU, legs, dogs, why)
    assert show == ["2026-11-12", "2026-11-14"]
    out = bs.payload(history, show, THU, bs.grade_all(history), why)
    thu, sat = out["days"]
    assert thu["today"] and thu["likeliest"]["status"] == "none" and thu["underdog"]["status"] == "none"
    assert "Saturday, Nov 14" in thu["likeliest"]["missing_reason"]
    assert thu["underdog"]["none_reason"] == "no_fill" and "Saturday, Nov 14" in thu["underdog"]["missing_reason"]
    assert thu["likeliest"]["next_day"] == thu["underdog"]["next_day"] == "2026-11-14"
    assert sat["likeliest"]["status"] == "open" and sat["underdog"]["status"] == "open"
    dog_leg = [l for l in sat["underdog"]["legs"] if l["dog"]]
    assert len(dog_leg) == 1 and dog_leg[0]["price_source"] == "median of 3 books"
    assert dog_leg[0]["decimal"] == pytest.approx(3.40)
    assert sat["underdog"]["dog"]["candidates"] == 4 and sat["underdog"]["lands_1_in"] >= 3
    assert [t["day"] for t in out["tickets"]] == show                # the old shape, the likeliest's
    json.dumps(out, allow_nan=False)


def test_days_beyond_the_window_are_not_shown():
    far = bs.ufc_legs(_card("2026-11-21T23:00:00Z", [_fight("A", "B", "A", 0.7, -250, 210)]), THU)
    assert far and bs.days_shown({"tickets": []}, THU, far) == ["2026-11-12"]
    near = bs.ufc_legs(_card("2026-11-20T01:00:00Z", [_fight("A", "B", "A", 0.7, -250, 210)]), THU)
    assert bs.days_shown({"tickets": []}, THU, near) == ["2026-11-12", "2026-11-19"]


def test_no_underdog_in_the_band_is_told_apart_from_no_fill(monkeypatch):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    heavy = _card(fights=[_fight(f"R{i}", f"B{i}", f"R{i}", 0.85, -600, 450) for i in range(3)])  # dogs at 5.5
    history, why = {"tickets": []}, {}
    show = bs.update(history, THU, bs.ufc_legs(heavy, THU), bs.ufc_dogs(heavy, THU), why)
    out = bs.payload(history, show, THU, bs.grade_all(history), why)
    sat = out["days"][1]
    assert sat["underdog"]["status"] == "none" and sat["underdog"]["none_reason"] == "no_dog"
    assert "2.00-4.00" in sat["underdog"]["missing_reason"]


def _dog_row(day, first_start, dog_key="NFL:d", fill=(("NFL:f", 1.3),)):
    legs = [{"sport": "NFL", "key": dog_key, "decimal": 2.6, "pick": "DOG", "dog": True, "p": 0.36,
             "p_market": 0.35, "names": {"game_id": dog_key[4:], "home": False}}]
    legs += [{"sport": "NFL", "key": k, "decimal": d, "pick": "FAV", "p": 0.8, "p_market": 0.77,
              "names": {"game_id": k[4:], "home": True}} for k, d in fill]
    return {"day": day, "kind": "underdog", "built_at": "2026-11-08T12:00:00Z", "status": "pending",
            "multiplier": 3.38, "p": 0.3, "p_market": 0.27, "first_start": first_start, "legs": legs,
            "dog": {"key": dog_key, "pick": "DOG", "rank": 0, "candidates": 3, "salt": sdy.DOG_SALT,
                    "version": sdy.DOG_VERSION}}


def test_old_rows_are_read_as_the_likeliest_ticket():
    old = _row("2026-11-08", "2026-11-08T21:00:00Z")                 # no "kind": written before v1
    history = {"tickets": [old]}
    assert bs.kind_of(old) == "likeliest" and "kind" not in old
    assert bs.latest(history, "2026-11-08") is old and bs.latest(history, "2026-11-08", "likeliest") is old
    assert bs.latest(history, "2026-11-08", "underdog") is None


def test_each_kind_is_locked_at_its_own_first_start():
    # the underdog's first leg has started (14:30); the likeliest's has not (21:00)
    history = {"tickets": [_row("2026-11-08", "2026-11-08T21:00:00Z"),
                           _dog_row("2026-11-08", "2026-11-08T14:30:00Z")]}
    legs = [L("NFL", f"g{i}", 0.7, 1.45, kickoff="2026-11-08T22:00:00Z") for i in range(5)]
    dogs = [sdy.dog("NFL", "x", "2026-11-08", "x event", "X", 0.35, 2.8, 0.33, lot_key="x",
                    kickoff="2026-11-08T22:00:00Z")]
    bs.update(history, NOW, legs, dogs)
    assert bs.latest(history, "2026-11-08", "underdog") is history["tickets"][1]       # locked: untouched
    new = bs.latest(history, "2026-11-08", "likeliest")
    assert new is not history["tickets"][0] and new["kind"] == "likeliest" and new["status"] == "pending"


def test_a_none_supersedes_only_its_own_kind_and_each_kind_is_graded_on_its_own(monkeypatch):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "soccer": {},
                                                  "nfl": {"g": 3.0, "d": 7.0, "f": 7.0}})
    lk = _row("2026-11-08", "2026-11-08T21:00:00Z")
    lk.pop("kind", None)
    dog = _dog_row("2026-11-08", "2026-11-08T21:30:00Z")
    history = {"tickets": [lk, dog]}
    # a run before either starts where only the likeliest still fits
    legs = [L("NFL", f"g{i}", 0.7, 1.45, kickoff="2026-11-08T22:00:00Z") for i in range(5)]
    bs.update(history, NOW, legs, [])
    assert bs.latest(history, "2026-11-08", "underdog")["status"] == "none"
    assert bs.latest(history, "2026-11-08", "underdog")["none_reason"] == "no_dog"
    assert bs.latest(history, "2026-11-08", "likeliest")["status"] == "pending"
    graded = bs.grade_all(history)
    assert dog["status"] == "superseded" and lk["status"] == "superseded"
    assert [bs.kind_of(r) for r in graded] == ["likeliest"]

    # graded independently, each from its own legs
    history = {"tickets": [_row("2026-11-07", "2026-11-07T18:00:00Z", legs=(("NFL:g", 3.2),)),
                           _dog_row("2026-11-07", "2026-11-07T18:00:00Z", dog_key="NFL:d")]}
    history["tickets"][0]["legs"][0]["names"] = {"game_id": "g", "home": True}
    history["tickets"][1]["legs"][0]["names"] = {"game_id": "d", "home": True}   # the dog is home and won
    graded = bs.grade_all(history)
    by = {bs.kind_of(r): r for r in graded}
    assert by["likeliest"]["status"] == "won" and by["underdog"]["status"] == "won"
    assert by["underdog"]["paid"] == pytest.approx(2.6 * 1.3)
    rec = {k: bs.record(graded, k) for k in bs.KINDS}
    assert rec["likeliest"]["settled"] == rec["underdog"]["settled"] == 1
    assert rec["underdog"]["units"] == pytest.approx(2.6 * 1.3 - 1, abs=0.01)
    assert rec["underdog"]["implied_roi"] == pytest.approx(0.27 * 3.38 - 1, abs=1e-4)
    assert bs.record(graded)["settled"] == 2


def test_the_underdog_row_logs_its_draw_and_an_unchanged_draw_is_not_logged_again(monkeypatch):
    legs, dogs = _dogday()
    history = {"tickets": []}
    bs.update(history, THU, legs, dogs)
    rows = [r for r in history["tickets"] if r["kind"] == "underdog"]
    assert len(rows) == 1
    d = rows[0]["dog"]
    assert set(d) >= {"key", "pick", "lot_key", "rank", "candidates", "salt", "version"}
    assert (d["salt"], d["version"], d["candidates"]) == (sdy.DOG_SALT, 1, 4)
    # the draw is fixed: recomputed from the logged lot keys alone, the same dog comes first
    sat = [x for x in dogs if x["day"] == "2026-11-14"]
    assert sdy.dog_order(sat)[d["rank"]]["key"] == d["key"]
    bs.update(history, THU, legs, dogs)
    assert len(history["tickets"]) == len(rows) + 1                    # + Saturday's likeliest, nothing new


def test_the_page_says_when_the_two_tickets_cannot_both_win(monkeypatch):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    lk = dict(_row("2026-11-12", "2026-11-12T23:00:00Z", legs=(("NFL:d", 1.6), ("NFL:f2", 2.0))), kind="likeliest")
    for l in lk["legs"]:
        l.update(pick="FAV", p=0.6, p_market=0.6)
    dog = _dog_row("2026-11-12", "2026-11-12T23:00:00Z", dog_key="NFL:d")
    history = {"tickets": [lk, dog]}
    out = bs.payload(history, ["2026-11-12"], THU, bs.grade_all(history))
    day = out["days"][0]
    assert day["cannot_both_win"] is True
    assert day["underdog"]["dog"]["candidates"] == 3 and day["underdog"]["lands_1_in"] == 4
    assert [l["dog"] for l in day["underdog"]["legs"]] == [True, False]


def test_the_underdog_record_is_not_measured_until_the_backtest_writes_this_rules_block(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    base = {"results": {"same_day": {}, "market_only": {}}, "periods": {},
            "splits": {"same_day": {"multipliers": {}, "by_sports": {}, "by_mix": {}, "by_weekday": {},
                                    "by_year": {}}},
            "caveats": ["The market is hard to beat."]}
    path = tmp_path / "bt.json"
    monkeypatch.setattr(bs, "BACKTEST", path)
    history = {"tickets": []}
    show = bs.update(history, NOW, [])

    path.write_text(json.dumps(base))
    m = bs.payload(history, show, NOW, [])["measured"]
    assert m["underdog"] is None and "Not measured yet" in m["underdog_missing_reason"]
    assert m["likeliest"]["rule"] == {} and "rule" in m                 # as before, and for an older page

    other = dict(base, dog={"version": 2, "salt": "sameday-dog-v2", "results": {"live": {"all": {"tickets": 9}}}})
    path.write_text(json.dumps(other))
    assert bs.payload(history, show, NOW, [])["measured"]["underdog"] is None

    good = dict(base, dog={"version": 1, "salt": sdy.DOG_SALT, "rule": "v1",
                           "results": {"live": {"all": {"tickets": 1465, "roi": float("nan")}},
                                       "salts": {"all": {"p5": -0.2, "p95": 0.1}}, "gap": {}, "paired": {}},
                           "caveats": ["Dog caveat."]})
    path.write_text(json.dumps(good))                    # json.dumps writes the NaN as NaN
    out = bs.payload(history, show, NOW, [])
    u = out["measured"]["underdog"]
    assert u["live"]["all"]["tickets"] == 1465 and u["live"]["all"]["roi"] is None
    assert set(u["report_only"]) == {"gap", "likeliest_dog", "market_only", "paired", "both"}
    assert "Dog caveat." in out["caveats"] and "The market is hard to beat." in out["caveats"]
    json.dumps(out, allow_nan=False)


def test_the_likeliest_tickets_price_implied_return_comes_from_the_backtests_own_tickets(tmp_path):
    """Outcome-free: each ticket's market chance times its multiplier, less
    1, averaged; left missing when the ticket list and the json disagree."""
    import csv
    import gzip
    path = tmp_path / "t.csv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "day", "multiplier", "p_market", "status", "profit"])
        w.writerow(["same_day", "2020-01-01", 3.0, 0.30, "won", 2.0])
        w.writerow(["same_day", "2025-01-01", 4.0, 0.20, "lost", -1.0])
        w.writerow(["market_only", "2025-01-01", 3.5, 0.25, "lost", -1.0])
    bt = {"periods": {"all": ["2016-08-01", "2026-08-08"], "last_3_years": ["2023-08-09", "2026-08-08"]},
          "results": {"same_day": {"all": {"tickets": 2}, "last_3_years": {"tickets": 1}},
                      "market_only": {"all": {"tickets": 2}, "last_3_years": {"tickets": 1}}}}
    got = bs.implied_from_tickets(bt, path)
    assert got["same_day"] == {"all": pytest.approx((0.9 - 1 + 0.8 - 1) / 2), "last_3_years": pytest.approx(-0.2)}
    assert got["market_only"] == {"last_3_years": pytest.approx(-0.125)}          # "all" counts 1, not 2
    assert bs.implied_from_tickets(bt, tmp_path / "missing.csv.gz") == {}
