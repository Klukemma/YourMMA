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
    assert bs.leg_result(nfl_tie, res) == {"winner": None, "void_reason": "tie"}
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
    out = bs.payload(history, show, NOW, [])
    m = out["measured"]
    assert m["underdog"] is None and "Not measured yet" in m["underdog_missing_reason"]
    assert m["underdog_missing_reason"] in out["caveats"] and set(bs.DOG_CAVEATS) <= set(out["caveats"])
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
    assert not set(bs.DOG_CAVEATS) & set(out["caveats"])            # the block's own caveats say it
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


def test_a_ufc_underdog_is_graded_as_its_own_corner():
    """The dog's names are the bout's corners and its pick the dog's name, so
    leg_result grades the dog's side - not the app's pick."""
    import types
    fav = _fight("Ciryl Gané", "Josh Hokit", "Ciryl Gané", 0.67, -195, 185)
    (d,) = bs.ufc_dogs(_card(fights=[fav]), THU)
    norm = sdy._norm_name
    grade = types.SimpleNamespace(
        norm_name=norm,
        find_result=lambda bout, index: (None, index[(bout["red_corner"], bout["blue_corner"])], None),
        _corner=lambda w, red, blue: "red" if norm(w) == norm(red) else "blue" if norm(w) == norm(blue) else None)
    won = {"ufc": (grade, {("Ciryl Gané", "Josh Hokit"): "Josh Hokit"})}
    lost = {"ufc": (grade, {("Ciryl Gané", "Josh Hokit"): "Ciryl Gane"})}
    assert bs.leg_result(d, won) == {"winner": "pick"}
    assert bs.leg_result(d, lost) == {"winner": "other"}


# ---------------------------------------------------------------- live mechanics (review fixes)

def _flip_week(flipped):
    """NFL games on Saturday. Unflipped: d is home +100 / away -120 and
    the blend picks the away favourite; flipped, the line mirrors (home -120
    / away +100) and the blend picks home. Every price is the same, only
    the sides move."""
    if not flipped:
        d = _game("d", "2026-11-14T18:00:00Z", "NYG", "WAS", "WAS", 0.55, 100, -120)
    else:
        d = _game("d", "2026-11-14T18:00:00Z", "NYG", "WAS", "NYG", 0.55, -120, 100)
    favs = [_game(f"f{i}", "2026-11-14T21:00:00Z", f"H{i}", f"A{i}", f"H{i}", 0.85, -600, 450) for i in range(4)]
    return {"games": [d] + favs}                 # the favourites' dogs, at 5.50, are out of the band


def test_a_pick_or_dog_that_flips_sides_at_the_same_price_is_logged_again():
    """The event key is the same for both sides: a flip at an equal price
    must not read as the same ticket, or the old side is shown and graded."""
    history = {"tickets": []}
    for flipped in (False, True):
        week = _flip_week(flipped)
        bs.update(history, THU, bs.nfl_legs(week, THU), bs.nfl_dogs(week, THU))
    lk = [r for r in history["tickets"] if r["kind"] == "likeliest" and r["day"] == "2026-11-14"]
    ud = [r for r in history["tickets"] if r["kind"] == "underdog" and r["day"] == "2026-11-14"]
    assert len(lk) == 2 and len(ud) == 2
    assert [l["decimal"] for l in lk[0]["legs"]] == [l["decimal"] for l in lk[1]["legs"]]     # same prices
    assert "WAS" in {l["pick"] for l in lk[0]["legs"]} and "NYG" in {l["pick"] for l in lk[1]["legs"]}
    assert (ud[0]["dog"]["pick"], ud[1]["dog"]["pick"]) == ("NYG", "WAS")
    assert ud[0]["legs"][0]["decimal"] == ud[1]["legs"][0]["decimal"] == pytest.approx(2.0)
    # and nothing new when nothing moved
    week = _flip_week(True)
    n = len(history["tickets"])
    bs.update(history, THU, bs.nfl_legs(week, THU), bs.nfl_dogs(week, THU))
    assert len(history["tickets"]) == n


def _moved_files(kickoff):
    """Saturday: Hull City (home, 3.20) the only underdog in the band, and
    NFL favourites at 1.17 to fill it (their dogs, at 5.50, are out)."""
    favs = [_game(f"f{i}", "2026-11-14T21:00:00Z", f"H{i}", f"A{i}", f"H{i}", 0.85, -600, 450) for i in range(2)]
    return {"card": None, "week": {"games": favs}, "matches": {"matches": [_match("m1", kickoff=kickoff)]}}


def _run(history, now, files):
    now = datetime.fromisoformat(now.replace("Z", "+00:00"))
    starts = bs.starts_on_file(files)
    bs.update(history, now, bs.live_legs(now, files), bs.live_dogs(now, files), starts=starts)
    return now, starts


def test_a_kickoff_moved_earlier_is_logged_and_locks_the_ticket_at_the_new_start():
    history = {"tickets": []}
    _run(history, "2026-11-14T09:00:00Z", _moved_files("2026-11-14T15:00:00Z"))
    (first,) = [r for r in history["tickets"] if r["kind"] == "underdog"]
    assert first["first_start"] == "2026-11-14T15:00:00Z" and first["dog"]["pick"] == "Hull City"
    # 11:00: the match is moved to 12:30 at the same prices - a new row, with the new first start
    _run(history, "2026-11-14T11:00:00Z", _moved_files("2026-11-14T12:30:00Z"))
    now = bs.latest(history, "2026-11-14", "underdog")
    assert now is not first and now["first_start"] == "2026-11-14T12:30:00Z"
    # 13:00: under way - locked, never rebuilt without its dog
    n = len(history["tickets"])
    _run(history, "2026-11-14T13:00:00Z", _moved_files("2026-11-14T12:30:00Z"))
    assert len(history["tickets"]) == n


def test_a_ticket_is_locked_when_the_files_say_a_leg_has_started_whatever_its_logged_start(monkeypatch):
    """No run between the move and the new kickoff: the row still says
    15:00, the files say 12:30, and at 13:00 the ticket is in play."""
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    history = {"tickets": []}
    _run(history, "2026-11-14T09:00:00Z", _moved_files("2026-11-14T15:00:00Z"))
    row = bs.latest(history, "2026-11-14", "underdog")
    files = _moved_files("2026-11-14T12:30:00Z")
    now, starts = _run(history, "2026-11-14T13:00:00Z", files)
    assert bs.latest(history, "2026-11-14", "underdog") is row             # not superseded by a "none"
    assert bs.locked(row, now, starts) and not bs.locked(row, now)        # the files decide it
    out = bs.payload(history, ["2026-11-14"], now, bs.grade_all(history), starts=starts)
    assert out["days"][0]["underdog"]["status"] == "locked"
    # without the files' start it would have been rebuilt (here: to "none", its dog gone)
    h2 = {"tickets": [dict(row)]}
    bs.update(h2, now, bs.live_legs(now, files), bs.live_dogs(now, files))
    assert bs.latest(h2, "2026-11-14", "underdog")["status"] == "none"


def test_a_corrupt_app_file_stops_the_run_and_a_missing_one_is_no_events(tmp_path, monkeypatch):
    week = tmp_path / "week.json"
    monkeypatch.setattr(bs, "UFC_CARD", tmp_path / "card.json")              # missing: no UFC events
    monkeypatch.setattr(bs, "NFL_WEEK", week)
    monkeypatch.setattr(bs, "SOCCER_MATCHES", tmp_path / "matches.json")
    assert bs.load_app_files() == {"card": None, "week": None, "matches": None}
    week.write_text('{"games": [{"game_id": "2026_11_KC_BUF"')               # committed truncated
    with pytest.raises(ValueError):
        bs.load_app_files()
    week.write_text('{"fixtures": []}')                                     # its list renamed
    with pytest.raises(ValueError):
        bs.load_app_files()
    week.write_text('{"games": []}')                                        # an empty week reads
    assert bs.load_app_files()["week"] == {"games": []}


def test_main_writes_nothing_when_an_app_file_does_not_read(tmp_path, monkeypatch):
    """A logged, unstarted ticket survives a corrupt week.json: the run
    stops before history.json or the page's file is written."""
    hist, out = tmp_path / "history.json", tmp_path / "sameday.json"
    logged = {"schema": 1, "tickets": [_row("2026-11-15", "2026-11-15T18:00:00Z")]}
    hist.write_text(json.dumps(logged))
    (tmp_path / "week.json").write_text("{not json")
    for name, path in (("HISTORY", hist), ("OUT", out), ("NFL_WEEK", tmp_path / "week.json"),
                       ("UFC_CARD", tmp_path / "card.json"), ("SOCCER_MATCHES", tmp_path / "matches.json")):
        monkeypatch.setattr(bs, name, path)
    monkeypatch.setattr(sys, "argv", ["build_sameday.py", "--now", "2026-11-15T12:00:00Z"])
    with pytest.raises(ValueError):
        bs.main()
    assert json.loads(hist.read_text()) == logged and not out.exists()


def test_a_history_file_that_does_not_read_stops_the_run(tmp_path):
    """Read as empty, the run would write an empty log over the record."""
    path = tmp_path / "history.json"
    assert bs.load_history(path) == {"schema": 1, "tickets": []}            # none yet: a fresh log
    path.write_text('{"schema": 1, "tickets": [')
    with pytest.raises(ValueError):
        bs.load_history(path)
    path.write_text('{"schema": 1}')
    with pytest.raises(ValueError):
        bs.load_history(path)


def _ufc_archive(path, rows):
    """A results archive in the columns engine/grade.py's load_results reads."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "r_name", "b_name", "winner", "method"])
        w.writerows(rows)
    return path


def _cancel_history():
    def leg(red, blue, pick, dec, dog=False):
        return {"sport": "UFC", "key": f"UFC:2026-11-14:{red}:{blue}", "event": f"{red} vs {blue}", "pick": pick,
                "decimal": dec, "kickoff": "2026-11-14T23:00:00Z", "dog": dog or None,
                "names": {"red": red, "blue": blue, "event_date": "2026-11-14"}}
    return {"tickets": [{"day": "2026-11-14", "kind": "underdog", "status": "pending", "multiplier": 4.0,
                         "p_market": 0.25, "first_start": "2026-11-14T23:00:00Z",
                         "legs": [leg("Ciryl Gane", "Josh Hokit", "Josh Hokit", 3.2, dog=True),
                                  leg("Ed Echo", "Tom Trot", "Ed Echo", 1.25)]}]}


def _grade_ufc(tmp_path, monkeypatch, rows, now):
    monkeypatch.setattr(bs, "UFC_ARCHIVE", _ufc_archive(tmp_path / "ufc.csv", rows))
    monkeypatch.setattr(bs, "UFC_LOG", tmp_path / "no_log.json")
    monkeypatch.setattr(bs, "NFL_GAMES", tmp_path / "none.csv.gz")
    monkeypatch.setattr(bs, "SOCCER_LOG", tmp_path / "none.json")
    history = _cancel_history()
    graded = bs.grade_all(history, datetime.fromisoformat(now.replace("Z", "+00:00")))
    return history["tickets"][0], graded


CARD = [("2026-11-14", "Ciryl Gane", "Josh Hokit", "Josh Hokit", "KO/TKO"),
        ("2026-11-14", "Al Alpha", "Bo Bravo", "Al Alpha", "Decision - Unanimous"),
        ("2026-11-14", "Cy Charlie", "Dee Delta", "Dee Delta", "Submission")]


def test_a_ufc_bout_that_never_happened_is_void_once_its_card_is_complete(tmp_path, monkeypatch):
    """engine/grade.py's rule, reused: the card has results in, it is a day
    past its first bell, and neither fighter has a result near it - the leg
    is void (books return it), never open for good."""
    row, _ = _grade_ufc(tmp_path, monkeypatch, CARD, "2026-11-15T05:00:00Z")      # card not complete yet
    assert row["status"] == "pending" and row["legs"][1]["outcome"] is None
    row, graded = _grade_ufc(tmp_path, monkeypatch, CARD, "2026-11-16T05:00:00Z")
    assert row["status"] == "won" and row["paid"] == pytest.approx(3.2)             # the void leg at 1.0
    assert row["legs"][1]["outcome"] == "void" and row["legs"][1]["void_reason"] == "cancelled"
    assert bs.record(graded, "underdog")["settled"] == 1


def test_a_missing_bout_stays_open_when_a_fighter_fought_near_the_card_or_no_results_are_in(tmp_path, monkeypatch):
    # Ed Echo has a result on the card under another spelling of the bout: not cancelled, a name to reconcile
    renamed = CARD + [("2026-11-14", "Edward Echo", "Tom Trot", "Edward Echo", "KO/TKO"),
                      ("2026-11-14", "Ed Echo", "Thomas Trot", "Ed Echo", "KO/TKO")]
    row, _ = _grade_ufc(tmp_path, monkeypatch, renamed, "2026-12-30T05:00:00Z")
    assert row["legs"][1]["outcome"] is None and row["status"] == "pending"
    # the card's results are not in yet: a missing bout means only "not yet"
    row, _ = _grade_ufc(tmp_path, monkeypatch, [("2026-10-01", "X Ray", "Zed Zulu", "X Ray", "KO/TKO")],
                        "2026-12-30T05:00:00Z")
    assert [l["outcome"] for l in row["legs"]] == [None, None] and row["status"] == "pending"


def test_a_ticket_still_in_play_after_midnight_stays_on_the_page(monkeypatch):
    """The 05:23 UTC run is 00:23 ET Sunday, with Saturday's main event on:
    Saturday is shown before today, flagged in play, and never rebuilt."""
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    sat = dict(_row("2026-11-14", "2026-11-14T23:00:00Z"), kind="likeliest")
    sat["legs"][0].update(p=0.6, p_market=0.58)
    sat_none = {"day": "2026-11-14", "kind": "underdog", "built_at": "2026-11-14T22:00:00Z", "status": "none",
                "multiplier": None, "legs": [], "none_reason": "no_fill"}
    fri = dict(_row("2026-11-13", "2026-11-13T23:00:00Z"), kind="likeliest", status="won")      # settled: gone
    thu = dict(_row("2026-11-12", "2026-11-12T23:00:00Z"), kind="likeliest")                   # 3 days back
    history = {"tickets": [thu, fri, sat, sat_none]}
    now = datetime(2026, 11, 15, 5, 23, tzinfo=timezone.utc)
    show = bs.update(history, now, [], [])
    assert show == ["2026-11-14", "2026-11-15"]
    assert history["tickets"] == [thu, fri, sat, sat_none]              # nothing logged for a past day
    out = bs.payload(history, show, now, bs.grade_all(history, now))
    past, today = out["days"]
    assert past["past"] and past["in_play"] and not past["today"]
    assert past["likeliest"]["status"] == "locked" and past["underdog"]["none_reason"] == "no_fill"
    assert today["today"] and not today["past"] and not today["in_play"]
    assert "No likeliest ticket in the next 7 days" in today["likeliest"]["missing_reason"]
    assert [t["day"] for t in out["tickets"]] == ["2026-11-15"]           # the old shape: today first
    # once graded it leaves the days shown (it is in the record's last tickets instead)
    sat["status"] = fri["status"] = "won"                 # (grade_all regraded fri, legs unknown here)
    assert bs.days_shown(history, now, []) == ["2026-11-15"]


def test_the_not_measured_reasons_carry_no_file_path(monkeypatch, tmp_path):
    """Reader-facing text: a path is an unbreakable token that runs past a
    phone's card; "salt" is a word the page never defines."""
    import re
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    reasons = [bs.dog_measured({})[1], bs.dog_measured({"dog": {"version": 2, "salt": "x"}})[1],
               bs.dog_measured({"dog": {"version": 1, "salt": sdy.DOG_SALT, "results": {"live": {}}}})[1]]
    monkeypatch.setattr(bs, "BACKTEST", tmp_path / "none.json")
    history = {"tickets": []}
    out = bs.payload(history, bs.update(history, NOW, []), NOW, [])
    for text in reasons + out["caveats"]:
        assert not re.search(r"\w/\w|\.py\b|\.json\b", text), text
        assert "salt" not in text.lower(), text
    assert all(r.startswith(("Not measured yet", "Not shown")) for r in reasons)


# ---------------------------------------------------------------- tickets v2, live

SAT = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


def _ou_match(mid, league="E0", o25=1.80, u25=2.05, market_o25=0.53, kickoff="2026-11-14T15:00:00Z", **kw):
    m = _match(mid, league=league, kickoff=kickoff)
    m["prices"]["avg"].update(o25=o25, u25=u25)
    m["market_o25"] = market_o25
    m.update(kw)
    return m


def test_an_over_under_leg_is_the_side_the_average_pair_favours_and_only_where_the_app_ships_it():
    missing = {}
    old = _match("old")                                          # written before the O/U prices were kept
    data = {"matches": [_ou_match("over"), _ou_match("under", o25=2.20, u25=1.68, market_o25=0.43),
                        _ou_match("held", league="SC2", market_o25=None),
                        _ou_match("level", o25=1.90, u25=1.90), old,
                        _ou_match("gone", kickoff="2026-11-14T11:00:00Z")]}
    over, under = bs.soccer_ou_legs(data, SAT, missing)
    ia, ib = 1 / 1.80, 1 / 2.05
    assert (over["pick"], over["bet"], over["decimal"]) == ("Over 2.5 goals", "OU25", 1.80)
    assert over["p"] == pytest.approx(ia / (ia + ib)) == over["p_market"]
    assert over["key"] == "Soccer:over" and over["market"] == "moneyline"      # one leg per match, any market
    assert (under["pick"], under["decimal"], under["names"]["over"]) == ("Under 2.5 goals", 1.68, False)
    assert missing[("2026-11-14", "OU25")] == {bs.OU_WITHHELD: 1, bs.LEVEL: 1, bs.OU_NOT_POSTED: 1}


def test_nfl_spread_and_total_legs_are_the_no_vig_side_at_the_consensus_line():
    def game(gid, market, **kw):
        g = _game(gid, "2026-11-15T18:00:00Z", "NYG", "BUF", "BUF", 0.6, 150, -170)
        g["market"].update(market)
        g.update(kw)
        return g
    priced = {"spread": -3.5, "spread_price_home": 100, "spread_price_away": -120, "spread_price_missing": None,
              "total": 44.5, "total_price_over": -105, "total_price_under": -115, "total_price_missing": None}
    missing = {}
    week = {"games": [
        game("g1", priced),
        game("g2", dict(priced, spread=0.0, total_price_over=None, total_price_under=None,
                        total_price_missing="no book priced both sides at the line")),
        game("g3", dict(priced, spread=2.0, spread_price_home=-110, spread_price_away=-110, total_price_over=-110,
                        total_price_under=-110), spread_pick={"team": "NYG", "line": -2.0},
             total_pick={"side": "OVER", "line": 44.5}),
        game("g4", {"spread": 1.0, "total": 40.0})]}               # a file from before the prices were kept
    legs = {(l["names"]["game_id"], l["bet"]): l for l in bs.nfl_line_legs(week, THU, missing)}
    s, t = legs[("g1", "SPREAD")], legs[("g1", "TOTAL")]
    ia, ib = 1 / 2.0, 1 / (1 + 100 / 120)
    assert (s["pick"], s["decimal"], s["names"]["home"]) == ("BUF -3.5", pytest.approx(1 + 100 / 120), False)
    assert s["p"] == pytest.approx(ib / (ia + ib)) and s["key"] == "NFL:g1"
    assert (t["pick"], t["names"]["over"]) == ("Under 44.5", False)
    # level prices: the shipped model's side at 0.5
    assert (legs[("g3", "SPREAD")]["pick"], legs[("g3", "SPREAD")]["p"]) == ("NYG -2", 0.5)
    assert legs[("g3", "TOTAL")]["pick"] == "Over 44.5"
    assert ("g2", "SPREAD") not in legs and ("g2", "TOTAL") not in legs and ("g4", "SPREAD") not in legs
    assert missing[("2026-11-15", "SPREAD")] == {bs.NFL_ZERO_SPREAD: 1, bs.NFL_NOT_IN_FILE: 1}
    assert missing[("2026-11-15", "TOTAL")] == {"no book priced both sides at the line": 1, bs.NFL_NOT_IN_FILE: 1}


def _method_card(est=2.6, chance=0.30, reason=None):
    f = _fight("Ciryl Gané", "Josh Hokit", "Ciryl Gané", 0.67, -195, 185)
    f["method_legs"] = [{"corner": c, "fighter": n, "type": t, "chance": None if reason else chance * (i + 1) / 3,
                         "market_chance": None if reason else 0.9 * chance, "est_decimal": None if reason else est + i,
                         "est_american": None, "error_pct": None if reason else 21.4, "source": "estimated",
                         "moneyline_source": None if reason else "median of 4 books", "missing_reason": reason}
                        for c, n in (("red", "Ciryl Gané"), ("blue", "Josh Hokit"))
                        for i, t in enumerate(sdy.METHOD_TYPES)]
    return _card(fights=[f])


def test_a_method_leg_is_the_cards_estimate_on_the_fights_own_key():
    missing = {}
    legs = bs.ufc_method_legs(_method_card(), THU, missing)
    assert len(legs) == 6 and not missing
    ko = legs[0]
    assert (ko["pick"], ko["bet"], ko["bet_detail"]) == ("Ciryl Gané by KO/TKO", "METHOD", "KO/TKO")
    assert ko["key"] == "UFC:2026-11-14:Ciryl Gané:Josh Hokit"            # the winner leg's: one leg per fight
    assert ko["estimated"] and ko["decimal"] == ko["est_decimal"] == 2.6 and ko["error_pct"] == 21.4
    assert ko["lot_key"] == sdy.method_lot_key(["Ciryl Gané", "Josh Hokit"], "Ciryl Gané", "KO/TKO")
    assert ko["names"] == {"red": "Ciryl Gané", "blue": "Josh Hokit", "event_date": "2026-11-14",
                           "corner": "red", "type": "KO/TKO"}
    assert ko["p_market"] == pytest.approx(0.27) and ko["day"] == "2026-11-14"
    missing = {}
    assert bs.ufc_method_legs(_method_card(reason="No division named for this bout."), THU, missing) == []
    assert missing == {("2026-11-14", "METHOD"): {"No division named for this bout.": 6}}
    missing = {}
    assert bs.ufc_method_legs(_card(fights=[_fight("A", "B", "A", 0.7, -250, 210)]), THU, missing) == []
    assert missing == {("2026-11-14", "METHOD"): {bs.METHOD_NOT_IN_FILE: 6}}
    assert bs.ufc_method_legs(_method_card(), datetime(2026, 11, 15, tzinfo=timezone.utc)) == []   # started


def test_the_v2_winner_legs_take_every_soccer_league_and_v1_still_the_top_five():
    files = {"card": None, "week": None, "matches": {"matches": [_match("e0"), _match("nor", league="NOR")]}}
    v2 = bs.live_v2(THU, files)
    assert sorted(l["key"] for l in v2["legs"] if l["bet"] == "1X2") == ["Soccer:e0", "Soccer:nor"]
    assert sorted(d["key"] for d in v2["dogs"]) == ["Soccer:e0", "Soccer:nor"]
    assert [l["key"] for l in bs.live_legs(THU, files)] == ["Soccer:e0"]
    assert all(l["market"] == "moneyline" for l in v2["legs"])


def _v2_files():
    """Saturday: a UFC card with four favourites and their props, NFL
    favourites on Sunday with spreads, and soccer matches with over/under."""
    fights = []
    for i in range(4):
        f = _fight(f"R{i}", f"B{i}", f"R{i}", 0.75, -300, 250, median=(-320, 240), books=3)
        f["number"] = i + 1
        f["method_legs"] = [{"corner": c, "fighter": n, "type": t, "chance": 0.75 * s if c == "red" else 0.25 * s,
                             "market_chance": 0.7 * s if c == "red" else 0.23 * s,
                             "est_decimal": 1 / (0.7 * s if c == "red" else 0.23 * s) / 1.2,
                             "est_american": None, "error_pct": 20.0, "source": "estimated",
                             "moneyline_source": "median of 3 books", "missing_reason": None}
                            for c, n in (("red", f"R{i}"), ("blue", f"B{i}"))
                            for t, s in zip(sdy.METHOD_TYPES, (0.45, 0.15, 0.40))]
        fights.append(f)
    matches = [_ou_match(f"m{i}", o25=1.55, u25=2.45, market_o25=0.61) for i in range(3)]
    return {"card": _card(fights=fights), "week": None, "matches": {"matches": matches}}


def _v2_run(history, now, files, why=None):
    starts = bs.starts_on_file(files)
    return bs.update(history, now, bs.live_legs(now, files), bs.live_dogs(now, files), why, starts,
                     v2=bs.live_v2(now, files))


def test_the_four_v2_kinds_are_built_and_logged_beside_v1_which_is_unchanged():
    files = _v2_files()
    history, why = {"tickets": []}, {}
    _v2_run(history, THU, files, why)
    sat = {r["kind"]: r for r in history["tickets"] if r["day"] == "2026-11-14"}
    assert set(sat) == set(bs.KINDS)
    for kind in bs.KINDS_V2:
        assert sat[kind]["version"] == sdy.V2_VERSION and sat[kind]["status"] == "pending"
        assert 3.0 <= sat[kind]["multiplier"] <= 5.0
    assert any(l["bet"] != "1X2" for l in sat["mixed"]["legs"])
    props = sat["props"]
    assert sum(l["bet"] == "METHOD" for l in props["legs"]) == 1 and props["estimated"]
    assert props["props"]["salt"] == sdy.PROPS_SALT and props["props"]["band"] == list(sdy.PROPS_LIVE_BAND)
    assert sdy.PROPS_LIVE_BAND[0] <= props["multiplier"] <= sdy.PROPS_LIVE_BAND[1]
    assert sat["underdog_v2"]["dog"]["salt"] == sdy.DOG_SALT_V2 and sat["underdog_v2"]["dog"]["version"] == 2
    assert all(l["bet"] in sdy.LEG_MARKETS for r in sat.values() if r["kind"] in bs.KINDS_V2 for l in r["legs"])
    # v1 rows are what they were with no v2 pool at all
    h1 = {"tickets": []}
    bs.update(h1, THU, bs.live_legs(THU, files), bs.live_dogs(THU, files), starts=bs.starts_on_file(files))
    v1 = [r for r in history["tickets"] if r["kind"] in bs.KINDS_V1]
    assert [dict(r, built_at=None) for r in v1] == [dict(r, built_at=None) for r in h1["tickets"]]
    assert "bet" not in v1[0]["legs"][0]                              # v1 rows keep their shape
    # nothing moved: nothing logged again
    n = len(history["tickets"])
    _v2_run(history, THU, files)
    assert len(history["tickets"]) == n
    # the props draw is recomputable from the logged lot keys
    legs = [l for l in bs.live_v2(THU, files)["legs"] if l["day"] == "2026-11-14"]
    cands = [m for m in legs if m["bet"] == "METHOD" and sdy.PROPS_PRICE[0] <= m["decimal"] <= sdy.PROPS_PRICE[1]]
    assert sdy.dog_order(cands, sdy.PROPS_SALT)[props["props"]["rank"]]["lot_key"] == props["props"]["lot_key"]


def test_each_v2_kind_is_locked_at_its_own_first_start_and_a_none_supersedes_only_its_kind():
    files = _v2_files()
    history = {"tickets": []}
    _v2_run(history, THU, files)
    before = {k: bs.latest(history, "2026-11-14", k) for k in bs.KINDS}
    # the soccer matches start at 15:00; the card at 23:00. At 16:00 a mixed
    # ticket with a soccer leg is locked; the props ticket (all UFC here) not
    at = datetime(2026, 11, 14, 16, 0, tzinfo=timezone.utc)
    files["card"]["fights"] = files["card"]["fights"][:1]             # three bouts gone
    _v2_run(history, at, files)
    after = {k: bs.latest(history, "2026-11-14", k) for k in bs.KINDS}
    for k in bs.KINDS:
        if bs.locked(before[k], at, bs.starts_on_file(files)):
            assert after[k] is before[k], k
        else:
            assert after[k] is not before[k], k
    assert after["props"]["kind"] == "props" and after["props"]["status"] in ("none", "pending")
    graded = bs.grade_all(history)
    assert {bs.kind_of(r) for r in graded} <= set(bs.KINDS)
    assert all(r["status"] == "superseded" for r in history["tickets"]
               if r is not after[bs.kind_of(r)] and r["day"] == "2026-11-14" and r["status"] != "none")


def test_without_a_v2_pool_no_v2_row_is_touched():
    files = _v2_files()
    history = {"tickets": []}
    _v2_run(history, THU, files)
    rows = list(history["tickets"])
    bs.update(history, THU, [], [])                                  # every v1 kind to none, v2 left alone
    for k in bs.KINDS_V2:
        assert bs.latest(history, "2026-11-14", k) in rows


def test_the_props_ticket_is_on_a_ufc_day_only():
    files = {"card": None, "week": None, "matches": {"matches": [_ou_match(f"m{i}") for i in range(6)]}}
    history, why = {"tickets": []}, {}
    _v2_run(history, THU, files, why)
    assert bs.latest(history, "2026-11-14", "props") is None and why[("2026-11-14", "props")] == "no_card"
    out = bs.payload(history, bs.days_shown(history, THU, bs.live_v2(THU, files)["legs"]), THU, [], why)
    assert all(x["props"] is None for x in out["days"]) and not any(x["ufc_day"] for x in out["days"])


def test_grading_per_market():
    res = {"ufc": None, "nfl": {"g": 3.0}, "nfl_total": {"g": 44.0}, "soccer": {"m": "H", "v": "void"},
           "soccer_goals": {"m": (2.0, 1.0), "v": "void", "n": None}}
    ou = lambda mid, over: {"sport": "Soccer", "bet": "OU25", "names": {"match_id": mid, "over": over, "line": 2.5}}
    assert bs.leg_result(ou("m", True), res) == {"winner": "pick"}
    assert bs.leg_result(ou("m", False), res) == {"winner": "other"}
    assert bs.leg_result(ou("v", True), res) == {"winner": None}
    assert bs.leg_result(ou("n", True), res) is None and bs.leg_result(ou("x", True), res) is None
    sp = lambda line, home: {"sport": "NFL", "bet": "SPREAD", "names": {"game_id": "g", "home": home, "line": line}}
    assert bs.leg_result(sp(2.5, True), res) == {"winner": "pick"}            # home won by 3, laid 2.5
    assert bs.leg_result(sp(3.5, True), res) == {"winner": "other"}
    assert bs.leg_result(sp(3.5, False), res) == {"winner": "pick"}           # away +3.5 covers
    assert bs.leg_result(sp(3.0, False), res) == {"winner": None, "void_reason": "push"}
    to = lambda line, over: {"sport": "NFL", "bet": "TOTAL", "names": {"game_id": "g", "over": over, "line": line}}
    assert bs.leg_result(to(43.5, True), res) == {"winner": "pick"}
    assert bs.leg_result(to(44.5, True), res) == {"winner": "other"}
    assert bs.leg_result(to(44.0, False), res) == {"winner": None, "void_reason": "push"}
    assert bs.leg_result(to(44.5, True), {"nfl": {}, "nfl_total": {}}) is None
    # a v1 leg and a v2 winner leg grade alike
    w = {"sport": "Soccer", "pick": "A", "names": {"match_id": "m", "pick": "H"}}
    assert bs.leg_result(w, res) == bs.leg_result(dict(w, bet="1X2"), res) == {"winner": "pick"}


def test_a_method_leg_wins_only_by_its_type_for_its_corner(tmp_path, monkeypatch):
    rows = [("2026-11-14", "Ciryl Gane", "Josh Hokit", "Josh Hokit", "KO/TKO"),
            ("2026-11-14", "Al Alpha", "Bo Bravo", "Al Alpha", "Decision - Unanimous"),
            ("2026-11-14", "Cy Charlie", "Dee Delta", "Dee Delta", "Submission"),
            ("2026-11-14", "Ed Echo", "Fy Fox", "", "Decision - Split"),
            ("2026-11-14", "Gus Golf", "Hal Hotel", "Gus Golf", "DQ")]
    monkeypatch.setattr(bs, "UFC_ARCHIVE", _ufc_archive(tmp_path / "ufc.csv", rows))
    for name in ("NFL_GAMES", "SOCCER_LOG", "UFC_LOG"):
        monkeypatch.setattr(bs, name, tmp_path / "none")
    def m(red, blue, corner, mtype):
        return {"sport": "UFC", "bet": "METHOD", "kickoff": "2026-11-14T23:00:00Z",
                "names": {"red": red, "blue": blue, "event_date": "2026-11-14", "corner": corner, "type": mtype}}
    res = bs._results()
    logged = {"tickets": [{"legs": [m("Ciryl Gane", "Josh Hokit", "red", "KO/TKO"),
                                    m("Ivy India", "Jo Juliet", "red", "KO/TKO")]}]}
    res = dict(res, **bs.ufc_cancel_index(logged, res["ufc"], log={}))
    assert bs.leg_result(m("Ciryl Gané", "Josh Hokit", "blue", "KO/TKO"), res) == {"winner": "pick"}
    assert bs.leg_result(m("Ciryl Gané", "Josh Hokit", "blue", "Decision"), res) == {"winner": "other"}
    assert bs.leg_result(m("Ciryl Gané", "Josh Hokit", "red", "KO/TKO"), res) == {"winner": "other"}
    assert bs.leg_result(m("Al Alpha", "Bo Bravo", "red", "Decision"), res) == {"winner": "pick"}
    assert bs.leg_result(m("Cy Charlie", "Dee Delta", "blue", "Submission"), res) == {"winner": "pick"}
    assert bs.leg_result(m("Ed Echo", "Fy Fox", "red", "Decision"), res) == {"winner": None,
                                                                         "void_reason": "draw or no contest"}
    assert bs.leg_result(m("Gus Golf", "Hal Hotel", "red", "KO/TKO"), res) == {"winner": "other"}   # a DQ
    gone = m("Ivy India", "Jo Juliet", "red", "KO/TKO")                      # never fought
    assert bs.leg_result(gone, res, datetime(2026, 11, 14, 23, 30, tzinfo=timezone.utc)) is None
    assert bs.leg_result(gone, res, datetime(2026, 11, 16, 5, tzinfo=timezone.utc)) == {
        "winner": None, "void_reason": "cancelled"}


def test_soccer_results_can_carry_the_finals_goals(tmp_path):
    picks = {"a": _soccer_pick("a", "2026-11-08T15:00:00Z", final={"home": 3, "away": 1, "result": "H"}),
             "b": _soccer_pick("b", "2026-11-08T15:00:00Z", void="no result"),
             "c": _soccer_pick("c", "2026-11-08T15:00:00Z", final={"result": "D"})}
    path = tmp_path / "prediction_history.json"
    path.write_text(json.dumps({"schema": 1, "picks": picks}))
    assert bs.soccer_results(path) == {"a": "H", "b": "void", "c": "D"}
    assert bs.soccer_results(path, with_goals=True) == {"a": {"result": "H", "goals": (3.0, 1.0)}, "b": "void",
                                                         "c": {"result": "D", "goals": None}}


def test_the_v2_page_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "_results", lambda: {"ufc": None, "nfl": {}, "soccer": {}})
    monkeypatch.setattr(bs, "BACKTEST", tmp_path / "none.json")
    files = _v2_files()
    files["week"] = {"games": [_game("g1", "2026-11-15T18:00:00Z", "NYG", "BUF", "BUF", 0.6, 150, -170)]}
    history, why = {"tickets": []}, {}
    show = _v2_run(history, THU, files, why)
    out = bs.payload(history, show, THU, bs.grade_all(history), why)
    text = json.dumps(out, allow_nan=False)
    sat = next(x for x in out["days"] if x["day"] == "2026-11-14")
    sun = next(x for x in out["days"] if x["day"] == "2026-11-15")
    assert sat["ufc_day"] and sat["props"]["status"] == "open" and sun["props"] is None
    p = sat["props"]
    assert p["estimated"] and p["props"]["candidates"] >= 1
    leg = next(l for l in p["legs"] if l["bet"] == "METHOD")
    assert leg["estimated"] and leg["error_pct"] == 20.0 and leg["market"] == "Method"
    assert leg["label"].endswith(("by KO/TKO", "by Submission", "by Decision"))
    assert "estimator" in leg["reason"]
    win = next(l for l in sat["likeliest_v2"]["legs"])
    assert win["label"].endswith(" to win") and win["market"] == "Moneyline" and not win["estimated"]
    mixed_bets = {l["bet"] for l in sat["mixed"]["legs"]}
    assert mixed_bets - {"1X2"}
    assert sun["missing"] == {"SPREAD": {bs.NFL_NOT_IN_FILE: 1}, "TOTAL": {bs.NFL_NOT_IN_FILE: 1}}
    assert set(out["records"]) == set(bs.KINDS) and out["rule"]["v2"]["props_live_band"] == [3.6, 4.2]
    m = out["measured_v2"]
    assert m["kinds"] == {k: None for k in bs.KINDS_V2}
    assert all(r.startswith("Not measured yet") for r in m["missing_reason"].values())
    assert set(bs.V2_CAVEATS) <= set(out["caveats"]) and "hard to beat" in text
    # the v1 shape the older page reads is all still there
    assert {"likeliest", "underdog", "cannot_both_win"} <= set(sat) and out["tickets"][0]["kind"] == "likeliest"


def test_the_v2_measured_block_is_read_defensively():
    assert all(v is None for v in bs.v2_measured({})[0].values())
    rec = {"all": {"tickets": 2400, "wins": 700, "roi": -0.1, "implied_roi": -0.15},
           "last_3_years": {"tickets": 0}}
    bt = {"periods": {"all": ["2016-08-01", "2026-08-08"]},
          "v2": {"version": 2, "kinds": {
              "likeliest_v2": {"results": rec},
              "underdog_v2": {"salt": sdy.DOG_SALT_V2, "results": {"live": rec, "salts": {"all": {"roi": -0.12}}}},
              "props": {"salt": "another", "as_live": rec},
              "mixed": {"results": {"all": {"tickets": 0}}}},
              "caveats": ["V2 caveat."]}}
    got, why = bs.v2_measured(bt)
    assert got["likeliest_v2"]["headline"]["all"]["tickets"] == 2400 and why["likeliest_v2"] is None
    assert got["underdog_v2"]["headline_from"] == "live" and got["underdog_v2"]["salts"]["all"]["roi"] == -0.12
    assert got["props"] is None and why["props"].startswith("Not shown")
    assert got["mixed"] is None and why["mixed"].startswith("Not measured yet")
    assert got["likeliest_v2"]["periods"] == bt["periods"]
    got, why = bs.v2_measured(dict(bt, v2=dict(bt["v2"], version=3)))
    assert all(v is None for v in got.values()) and all(w.startswith("Not shown") for w in why.values())
    import re
    for w in list(why.values()) + list(bs.v2_measured({})[1].values()) + list(bs.v2_measured(bt)[1].values()):
        if w:
            assert not re.search(r"\w/\w|\.py\b|\.json\b", w) and "salt" not in w.lower(), w


def test_the_method_estimate_reads_the_median_pair_when_the_card_has_one():
    """The estimator was fitted on one book's closing pair; live, the card's
    median pair across books is the closest match to it, and each leg says
    which pair it read."""
    sys.path.insert(0, str(ENGINE.parents[1] / "engine"))
    import pandas as pd
    import app_export as ax
    import method_price as mp
    est = ax.method_estimator()
    if est["model"] is None:
        pytest.skip(est["missing"])
    archive = pd.DataFrame([("A One", "B Two", "2026-01-10", "A One", "KO/TKO")],
                           columns=["r_name", "b_name", "date", "winner", "method"])
    fight = {"red": "A One", "blue": "B Two", "pick": "A One", "win_prob": 0.66, "market_prob": 0.62,
             "rounds_scheduled": 3, "decimal": 1.55, "decimal_other": 2.75,
             "decimal_median": 1.50, "decimal_other_median": 2.60, "books_median": 5}

    def p_red_of(legs):
        """The p_red an estimate was built from: the one whose prediction it equals."""
        for p in (0.62, 1 / 1.50 / (1 / 1.50 + 1 / 2.60), 1 / 1.55 / (1 / 1.55 + 1 / 2.75)):
            e = mp.predict({"p_red": p, "division": "lightweight", "rounds_scheduled": 3,
                            "facts_red": sdy.pp.facts(archive, "A One", "2026-11-14"),
                            "facts_blue": sdy.pp.facts(archive, "B Two", "2026-11-14")}, est["model"])
            if all(abs(l["est_decimal"] - e[l["corner"]][l["type"]]["decimal"]) < 1e-5 for l in legs):
                return p
        return None
    median = ax.method_legs_payload(fight, "lightweight", archive, "2026-11-14", est)
    assert p_red_of(median) == pytest.approx(1 / 1.50 / (1 / 1.50 + 1 / 2.60))
    assert {l["moneyline_source"] for l in median} == {"median of 5 books"}
    best = ax.method_legs_payload({k: v for k, v in fight.items() if "median" not in k}, "lightweight", archive,
                                  "2026-11-14", est)
    assert p_red_of(best) == pytest.approx(1 / 1.55 / (1 / 1.55 + 1 / 2.75))
    assert {l["moneyline_source"] for l in best} == {"best of books"}
    bare = {k: v for k, v in fight.items() if "decimal" not in k and "median" not in k}
    assert {l["moneyline_source"] for l in ax.method_legs_payload(bare, "lightweight", archive, "2026-11-14", est)} \
        == {"card market chance"}


def test_the_v2_measured_block_in_the_backtests_shape():
    """sameday_backtest.measure_v2's block: kinds likeliest / underdog /
    mixed / props_as_live / props_as_priced, each reading split by league
    set; the headline is every league's, the other beside it."""
    rec = lambda n, roi: {"tickets": n, "wins": n // 3, "roi": roi, "implied_roi": -0.15, "days": 3660,
                          "days_with_ticket": n, "share_of_days": round(n / 3660, 4)}
    per = lambda n, roi: {"all": rec(n, roi), "first": rec(n // 2, roi), "last_3_years": rec(n // 2, roi)}
    split = lambda a, b: {"all_leagues": a, "without_extra_leagues": b}
    salts = {"all": {"roi": -0.11, "roi_low": -0.15, "roi_high": -0.07, "implied_roi": -0.14, "salts": 1000}}
    bt = {"v2": {"version": 2, "periods": {"all": ["2016-08-01", "2026-08-08"]}, "headline": "Tickets v2: ...",
                 "activity": {"all_leagues": {"all": {"tickets": 7000, "tickets_per_day": 1.9, "share_of_days": 0.74}},
                              "v1": {"all": {"tickets": 2700, "tickets_per_day": 0.74}}, "note": "n"},
                 "kinds": {"likeliest": {"live": split(per(2400, -0.1), per(2000, -0.08)),
                                         "market_only": split(per(2400, -0.12), per(2000, -0.1)),
                                         "splits": {"by_bets": {"note": "d", "1X2": rec(2400, -0.1)}}},
                           "underdog": {"salt": sdy.DOG_SALT_V2, "live": split(per(2500, -0.05), per(2100, -0.06)),
                                        "salts": split(salts, salts)},
                           "mixed": {"live": split(per(2300, -0.14), per(2000, -0.13))},
                           "props_as_live": {"salt": sdy.PROPS_SALT, "label": "as live",
                                             "periods": {"all": ["2016-08-01", "2024-12-07"]},
                                             "live": split(per(1200, -0.2), per(1200, -0.2)),
                                             "salts": split(salts, salts)},
                           "props_as_priced": {"salt": sdy.PROPS_SALT, "live": split(per(1100, -0.25), per(1100, -0.25))}},
                 "caveats": ["V2 caveat."]}}
    got, why = bs.v2_measured(bt)
    assert all(w is None for w in why.values())
    assert got["likeliest_v2"]["headline"]["all"]["tickets"] == 2400
    assert got["likeliest_v2"]["without_extra"]["all"]["tickets"] == 2000
    assert got["likeliest_v2"]["market_only"]["all"]["roi"] == -0.12
    assert got["underdog_v2"]["salts"]["all"]["roi"] == -0.11 and "salts" not in got["mixed"]
    assert got["props"]["headline"]["all"]["tickets"] == 1200 and got["props"]["as_priced"]["all"]["tickets"] == 1100
    assert got["props"]["periods"]["all"][1] == "2024-12-07" and got["props"]["label"] == "as live"
    act = bs.v2_activity(bt)
    assert act["v2"]["all"]["tickets_per_day"] == 1.9 and act["v1"]["all"]["tickets_per_day"] == 0.74
    assert bs.v2_activity({}) is None
    other = json.loads(json.dumps(bt))
    other["v2"]["kinds"]["underdog"]["salt"] = "sameday-dog-v9"
    assert bs.v2_measured(other)[0]["underdog_v2"] is None
