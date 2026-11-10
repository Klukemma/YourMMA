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
