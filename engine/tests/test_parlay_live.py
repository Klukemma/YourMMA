"""The three parlay tickets on the live card, their log, and their grading.

A parlay's arithmetic is where optimism hides, and a live one has two more
places to go wrong: a price on the wrong side of a bout, and a ticket graded
on a result it was logged after. Everything here runs on small fixtures, so
none of it needs the twenty-minute engine.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
sys.path.insert(0, str(ENGINE))

import app_export as ax
import grade
import parlay_profiles as pp

DATE = "2026-11-07"


def _fight(number, red, blue, pick, win_prob, win_prob_model, odds_pick, odds_other):
    return {"number": number, "red": red, "blue": blue, "pick": pick,
            "win_prob": win_prob, "win_prob_model": win_prob_model,
            "odds": odds_pick, "odds_other": odds_other}


@pytest.fixture
def card():
    """Six safe favourites at -250, one long underdog at +300 the model rates
    well above the market, and one +150 underdog it rates above the market
    by enough to be a risky leg: a card that makes all three bands."""
    fights = [_fight(i, f"Red {i}", f"Blue {i}", f"Red {i}", 0.70, 0.72, -250, 200)
              for i in range(1, 7)]
    fights.append(_fight(7, "Red 7", "Blue 7", "Red 7", 0.70, 0.55, -400, 300))
    fights.append(_fight(8, "Red 8", "Blue 8", "Red 8", 0.55, 0.45, -180, 150))
    return fights


def test_three_tickets_in_band_order(card):
    out = ax.parlay_profiles_payload(card, None, None, event_date=DATE)
    assert [e["profile"] for e in out] == ["steady", "risker", "risk_lover"]
    for e in out:
        assert e["label"] == pp.PROFILES[e["profile"]]["label"]
        assert e["target"] == pp.PROFILES[e["profile"]]["target"]
        assert e["band"] == list(pp.BANDS[e["profile"]])


def test_each_multiplier_is_the_product_of_its_legs_and_inside_its_band(card):
    for e in ax.parlay_profiles_payload(card, None, None, event_date=DATE):
        assert e["multiplier"] is not None, e["missing_reason"]
        assert e["missing_reason"] is None
        product = float(np.prod([l["decimal"] for l in e["legs"]]))
        assert e["multiplier"] == pytest.approx(product, rel=1e-6)
        lo, hi = e["band"]
        assert lo <= e["multiplier"] <= hi
        # Rounded to six places on the way out, like every probability here.
        assert e["prob"] == pytest.approx(float(np.prod([l["prob"] for l in e["legs"]])), abs=2e-6)
        for l in e["legs"]:
            # The price as quoted, and the decimal it converts to.
            assert isinstance(l["odds"], int)
            assert pp.decimal_from_american(l["odds"]) == pytest.approx(l["decimal"], rel=1e-6)
            assert l["side"] in ("favourite", "underdog")
            assert l["reason"] and l["fight"] == f"{l['red']} vs {l['blue']}"
            assert l["pick"] == (l["red"] if l["corner"] == "red" else l["blue"])


def test_no_ticket_carries_two_legs_from_one_bout(card):
    for e in ax.parlay_profiles_payload(card, None, None, event_date=DATE):
        numbers = [l["number"] for l in e["legs"]]
        assert len(numbers) == len(set(numbers))


def test_the_pick_side_probabilities_are_turned_round_for_the_other_corner(card):
    """win_prob and win_prob_model arrive for the PICK. Fight 8's pick is red
    on the blend (55%) though the model alone gives red 45%: the blue leg
    must carry the model's 55%, not 45%."""
    legs, quoted, priced = ax._card_legs(card, None, DATE)
    blue8 = next(l for l in legs if l["number"] == 8 and l["side"] == "blue")
    assert blue8["p_model"] == pytest.approx(0.55)
    assert blue8["p"] == pytest.approx(0.45)
    assert quoted[(blue8["key"], "blue")] == 150
    assert priced == 8


def test_a_fight_missing_either_price_contributes_no_leg(card):
    card = card + [_fight(9, "Red 9", "Blue 9", "Red 9", 0.9, 0.9, -900, None),
                   _fight(10, "Red 10", "Blue 10", "Red 10", 0.9, 0.9, None, 600),
                   _fight(11, "Red 11", "Blue 11", "Red 11", 0.9, None, -900, 600)]
    legs, _, priced = ax._card_legs(card, None, DATE)
    assert {l["number"] for l in legs} == set(range(1, 9))
    assert priced == 8
    for e in ax.parlay_profiles_payload(card, None, None, event_date=DATE):
        assert not {l["number"] for l in e["legs"]} & {9, 10, 11}


def test_a_band_the_card_cannot_reach_shows_no_ticket_and_says_why():
    one = [_fight(1, "A", "B", "A", 0.7, 0.72, -250, 200)]
    out = ax.parlay_profiles_payload(one, None, None, event_date=DATE)
    assert [e["profile"] for e in out] == ["steady", "risker", "risk_lover"]
    for e in out:
        assert e["multiplier"] is None and e["prob"] is None and e["legs"] == []
        assert "No combination" in e["missing_reason"]
    unpriced = ax.parlay_profiles_payload(
        [_fight(1, "A", "B", "A", 0.7, 0.72, None, None)], None, None, event_date=DATE)
    assert all(e["multiplier"] is None and "both prices" in e["missing_reason"] for e in unpriced)


def test_without_the_backtest_the_named_profiles_build_and_nothing_is_measured(card, tmp_path):
    assert ax.load_parlay_profiles(tmp_path / "absent.json") is None
    for e in ax.parlay_profiles_payload(card, None, None, event_date=DATE):
        assert e["rule"] == e["profile"]
        assert e["measured"] is None


def _measured(ship):
    summary = {"tickets": 41, "wins": 6, "hit_rate": 0.1463, "expected_hit_rate": 0.17,
               "market_hit_rate": 0.15, "mean_multiplier": 5.4, "mean_legs": 4.2,
               "profit": -4.9, "roi": -0.1195, "roi_low": -0.45, "roi_high": 0.3,
               "biggest_win": 5.1}
    names = ["steady", "risker", "risk_lover", "risk_lover_method"] + [
        f"{k}@{b}" for k in ("mixed", "market_only", "mixed_else") for b in pp.ORDER]
    return {"generated": "2026-11-07T00:00:00Z", "protocol": "fixed before scoring",
            "periods": {"dev": ["2016-11-01", "2023-10-31"], "window": ["2023-11-01", "2026-10-31"]},
            "results": {"dev": {n: dict(summary, tickets=80) for n in names},
                        "window": {n: dict(summary) for n in names}},
            "by_snapshot": {"window": {"close": {}, "open": {}}},
            "cards": {"dev": 300, "window": 140}, "coverage": {},
            "ship": ship, "caveats": ["Closing prices only before 2024."]}


def test_the_backtest_picks_the_rule_and_its_record_rides_with_the_ticket(card):
    measured = _measured({"steady": "mixed", "risker": "risker", "risk_lover": "risk_lover"})
    out = ax.parlay_profiles_payload(card, None, measured, event_date=DATE)
    assert [e["rule"] for e in out] == ["mixed_else@steady", "risker", "risk_lover"]
    assert out[0]["built_by"] in ("mixed", "steady")
    assert out[0]["measured"]["window"]["tickets"] == 41
    assert out[0]["measured"]["dev"]["tickets"] == 80
    assert out[0]["measured"]["periods"]["window"][0] == "2023-11-01"
    assert "mixed" in out[0]["blurb"].lower() or "any legs" in out[0]["blurb"].lower()
    # A rule the pre-registration does not allow is not shipped.
    odd = _measured({"steady": "market_only", "risker": None, "risk_lover": "mixed"})
    assert [ax.shipped_rule(odd, b) for b in pp.ORDER] == ["steady", "risker", "mixed_else@risk_lover"]


@pytest.mark.parametrize("measured", [None, "present"])
def test_the_caveat_says_the_market_is_hard_to_beat_and_prices_are_best_of_books(measured):
    m = _measured({"steady": "steady", "risker": "mixed", "risk_lover": "risk_lover"}) if measured else None
    text = ax.parlay_profiles_caveat(m)
    assert len(text) > 60
    assert "market is hard to beat" in text
    assert "best price seen across the books" in text and "may pay less" in text
    assert "not priced live" in text
    if m is None:
        assert "Not measured yet" in text and "%" not in text
    else:
        # Every figure is the file's: 41 tickets, 6 won, -12% (-45% to +30%).
        assert "41 tickets, 6 won, -12% a ticket (-45% to +30%)" in text
        assert "2023-11-01 to 2026-10-31" in text
        assert "50-100x ticket landed 6 times in 41" in text
        assert "~10x (mixed, else risker)" in text


def test_the_card_carries_both_prices_the_tickets_and_every_caveat(card):
    fights = [dict(card[0], book="FanDuel", book_other="BetMGM",
                   odds_seen="2026-11-05T04:29:16Z", odds_other_seen="2026-11-05T04:29:16Z")]
    tickets = ax.parlay_profiles_payload(card, None, None, event_date=DATE)
    payload = ax.card_payload("E", DATE, fights, parlay_profiles=tickets, profiles_measured=None)
    f = payload["fights"][0]
    assert (f["odds"], f["odds_other"]) == (-250, 200)
    assert f["decimal"] == pytest.approx(1.4) and f["decimal_other"] == pytest.approx(3.0)
    assert (f["book"], f["book_other"]) == ("FanDuel", "BetMGM")
    assert f["odds_seen"] == "2026-11-05T04:29:16Z"
    assert payload["parlay_profiles"] == tickets
    # The old caveat stays; the new one joins it.
    for key in ("model", "simulation", "parlay", "parlay_profiles"):
        assert len(payload["caveats"][key]) > 60
    bare = ax.card_payload("E", DATE, [{"red": "A", "blue": "B"}], profiles_measured=None)
    assert bare["fights"][0]["odds_other"] is None and bare["fights"][0]["decimal"] is None
    assert bare["parlay_profiles"] == []
    json.dumps(payload, allow_nan=False)


def test_the_reasons_read_only_bouts_before_the_card(card):
    archive = pd.DataFrame({
        "date": ["2025-01-01", "2027-01-01"],
        "r_name": ["Blue 7", "Blue 7"], "b_name": ["Someone", "Someone Else"],
        "winner": ["Blue 7", "Blue 7"], "method": ["KO/TKO Punches", "Submission"],
    })
    legs, _, _ = ax._card_legs(card, archive, DATE)
    blue7 = next(l for l in legs if l["number"] == 7 and l["side"] == "blue")
    assert blue7["facts"]["fights"] == 1 and blue7["facts"]["ko_wins"] == 1
    assert blue7["facts"]["sub_wins"] == 0


# ------------------------------------------------------------------ the log

def test_logged_ticket_records_are_well_formed(card):
    tickets = ax.parlay_profiles_payload(card, None, None, event_date=DATE)
    tickets.append(ax.parlay_profiles_payload(card[:1], None, None, event_date=DATE)[0])
    rows = ax.parlay_records(tickets, "UFC Test", DATE, first_start="2026-11-07T22:00:00Z",
                             timestamp="2026-11-07T12:00:00+00:00")
    assert len(rows) == 3, "a band with no ticket logs nothing"
    keys = {"event_name", "event_date", "first_start", "profile", "rule", "band",
            "multiplier", "prob", "legs", "timestamp", "status"}
    for row, ticket in zip(rows, tickets):
        assert keys <= set(row)
        assert row["status"] == "pending" and row["event_date"] == DATE
        assert row["profile"] == ticket["profile"] and row["rule"] == ticket["rule"]
        assert row["multiplier"] == pytest.approx(float(np.prod([l["decimal"] for l in row["legs"]])), rel=1e-6)
        for leg in row["legs"]:
            assert set(leg) == {"red", "blue", "pick", "side", "market", "decimal"}
            assert leg["side"] in ("red", "blue")
            assert leg["pick"] == leg[leg["side"]]
            assert leg["market"] == "moneyline" and leg["decimal"] > 1.0
        assert grade.made_before_the_card(row)
    json.dumps(rows, allow_nan=False)
    late = ax.parlay_records(tickets, "UFC Test", DATE, first_start="2026-11-07T22:00:00Z",
                             timestamp="2026-11-07T23:00:00+00:00")
    assert not grade.made_before_the_card(late[0])
    # A real stamp is UTC and readable.
    stamp = ax.parlay_records(tickets, "UFC Test", DATE)[0]["timestamp"]
    assert datetime.fromisoformat(stamp).utcoffset().total_seconds() == 0


def test_the_engine_logs_tickets_only_under_the_picks_cutoff():
    source = (ENGINE / "predict_card.py").read_text()
    logged = source.index("log_card_predictions(predictions_to_log, EVENT_NAME, EVENT_DATE")
    flag = source.index("_may_log_parlays = True")
    gate = source.rindex("_grade.picks_may_be_logged(", 0, logged)
    assert gate < logged < flag
    # Set nowhere else, and the only call is behind it.
    assert source.count("_may_log_parlays = True") == 1
    call = source.index("    log_parlay_tickets(_parlay_tickets")
    assert source.rindex("if _may_log_parlays and", 0, call) > flag
    assert "parlay_profiles=_parlay_tickets" in source


# ------------------------------------------------------------------ grading

@pytest.fixture
def results(tmp_path):
    csv = tmp_path / "f.csv"
    pd.DataFrame({
        "date": [DATE] * 3 + ["2026-10-31"],
        "r_name": ["Ann One", "Ann Two", "Ann Three", "Old Red"],
        "b_name": ["Bea One", "Bea Two", "Bea Three", "Old Blue"],
        "winner": ["Ann One", "Bea Two", "", "Old Red"],
        "method": ["KO/TKO", "U-DEC", "Decision - Split", "Submission"],
    }).to_csv(csv, index=False)
    return grade.load_results(csv)


def _leg(red, blue, side, decimal, market="moneyline"):
    return {"red": red, "blue": blue, "pick": red if side == "red" else blue,
            "side": side, "market": market, "decimal": decimal}


def _ticket(profile, legs, stamp="2026-11-07T12:00:00+00:00", date=DATE,
            first_start="2026-11-07T22:00:00Z"):
    return {"event_name": "UFC Test", "event_date": date, "first_start": first_start,
            "profile": profile, "rule": profile, "band": list(pp.BANDS[profile]),
            "multiplier": float(np.prod([l["decimal"] for l in legs])), "prob": 0.2,
            "legs": legs, "timestamp": stamp, "status": "pending"}


def test_grade_parlays_settles_won_lost_and_void(results):
    won = _ticket("steady", [_leg("Ann One", "Bea One", "red", 1.5),
                             _leg("Ann Two", "Bea Two", "blue", 2.0)])
    lost = _ticket("risker", [_leg("Ann One", "Bea One", "red", 1.5),
                              _leg("Ann Two", "Bea Two", "red", 3.0)])
    void = _ticket("risk_lover", [_leg("Ann Three", "Bea Three", "red", 4.0)])
    history = {"predictions": [], "parlays": [won, lost, void]}
    graded = grade.grade_parlays(history, results)
    assert len(graded) == 3
    assert (won["status"], won["paid"], won["profit"]) == ("won", pytest.approx(3.0), pytest.approx(2.0))
    assert [l["outcome"] for l in won["legs"]] == ["won", "won"]
    assert (lost["status"], lost["profit"]) == ("lost", -1.0)
    assert [l["outcome"] for l in lost["legs"]] == ["won", "lost"]
    assert (void["status"], void["profit"]) == ("void", 0.0)
    assert void["legs"][0]["void_reason"] == "draw or no contest"


def test_a_void_leg_counts_one_and_a_cancelled_bout_is_void_once_the_card_is_in(results):
    t = _ticket("risk_lover", [_leg("Ann One", "Bea One", "red", 1.5),
                               _leg("Ann Three", "Bea Three", "blue", 3.0),
                               _leg("Gone One", "Gone Two", "red", 5.0)])
    history = {"predictions": [], "parlays": [t]}
    grade.grade_parlays(history, results)
    assert t["status"] == "won"
    assert t["paid"] == pytest.approx(1.5), "void legs pay 1.0, not their price"
    assert t["legs"][2]["outcome"] == "void" and t["legs"][2]["void_reason"] == "cancelled"


def test_a_card_with_no_results_yet_stays_pending(results):
    t = _ticket("steady", [_leg("Next One", "Next Two", "red", 1.5)], date="2026-12-12",
                stamp="2026-12-10T00:00:00+00:00", first_start=None)
    history = {"predictions": [], "parlays": [t]}
    grade.grade_parlays(history, results)
    assert t["status"] == "pending" and t["profit"] is None
    assert t["legs"][0]["outcome"] is None and "void_reason" not in t["legs"][0]


def test_a_lost_leg_loses_the_ticket_even_with_a_leg_still_to_come(results):
    t = _ticket("risker", [_leg("Ann Two", "Bea Two", "red", 3.0),
                           _leg("Next One", "Next Two", "red", 1.5)],
                date=DATE)
    history = {"predictions": [], "parlays": [t]}
    grade.grade_parlays(history, results)
    assert t["status"] == "lost"


def test_only_the_newest_ticket_made_before_the_card_counts(results):
    early = _ticket("steady", [_leg("Ann Two", "Bea Two", "red", 3.0)], stamp="2026-11-06T10:00:00+00:00")
    newest = _ticket("steady", [_leg("Ann One", "Bea One", "red", 1.5)], stamp="2026-11-07T20:00:00+00:00")
    late = _ticket("steady", [_leg("Ann Two", "Bea Two", "blue", 2.0)], stamp="2026-11-08T01:00:00+00:00")
    history = {"predictions": [], "parlays": [early, newest, late]}
    graded = grade.grade_parlays(history, results)
    assert graded == [newest]
    assert newest["status"] == "won"
    assert early["status"] == "superseded" and "profit" not in early
    assert late["status"] == "late"
    live = ax.parlay_profiles_live(history)
    assert live["profiles"]["steady"]["settled"] == 1
    assert live["profiles"]["steady"]["won"] == 1
    assert live["profiles"]["steady"]["units"] == pytest.approx(0.5)
    assert live["profiles"]["risker"]["settled"] == 0 and live["profiles"]["risker"]["roi"] is None
    assert [t["profit"] for t in live["last"]] == [pytest.approx(0.5)]


def test_grade_write_keeps_every_other_key(results, tmp_path):
    path = tmp_path / "history.json"
    pick = {"red_corner": "Ann One", "blue_corner": "Bea One", "predicted_winner": "Ann One",
            "event_date": DATE, "win_probability": 0.6, "method_predicted": "KO/TKO",
            "event_name": "UFC Test", "timestamp": "2026-11-07T10:00:00", "status": "pending"}
    ticket = _ticket("steady", [_leg("Ann One", "Bea One", "red", 1.5)])
    path.write_text(json.dumps({"predictions": [pick], "summary": {}, "parlays": [ticket],
                                "something_else": {"kept": True}}))
    graded, ungraded = grade.grade(grade.deduplicate([pick]), results)
    grade.write_back(graded, ungraded, path, results)
    out = json.loads(path.read_text())
    assert out["something_else"] == {"kept": True}
    assert out["predictions"][0]["status"] == "correct"
    assert out["parlays"][0]["status"] == "won"
    assert out["parlays"][0]["profit"] == pytest.approx(0.5)
    # Graded again, nothing moves: grading is a function of the results.
    grade.write_back(graded, ungraded, path, results)
    assert json.loads(path.read_text()) == out


def test_a_method_leg_settles_on_the_short_decision_labels(results):
    t = _ticket("risk_lover", [_leg("Ann Two", "Bea Two", "blue", 6.0, market="Decision")])
    grade.grade_parlays({"predictions": [], "parlays": [t]}, results)
    assert t["status"] == "won"


# ------------------------------------------------------------------ the record tab

def test_the_record_tab_reads_the_backtest_from_disk(tmp_path, monkeypatch):
    import build_app_data as bad
    monkeypatch.setattr(bad, "EXPERIMENTS", tmp_path)
    empty = bad.parlay_profiles({"predictions": []})
    assert empty["backtest"] is None
    assert empty["live"]["profiles"]["steady"]["settled"] == 0
    assert len(empty["caveats"]["profiles"]) > 60 and len(empty["caveats"]["parlay"]) > 60

    (tmp_path / "parlay_profiles.json").write_text(json.dumps(
        _measured({"steady": "steady", "risker": "mixed", "risk_lover": "risk_lover"})))
    full = bad.parlay_profiles({"predictions": []})
    rows = {(r["band"], r["rule"]): r for r in full["backtest"]["rows"]}
    assert rows[("steady", "steady")]["shipped"] and rows[("steady", "steady")]["role"] == "named"
    assert rows[("risker", "mixed_else@risker")]["shipped"]
    assert rows[("risker", "mixed@risker")]["role"] == "pre-registered"
    assert not rows[("risker", "risker")]["shipped"]
    assert rows[("risk_lover", "market_only@risk_lover")]["role"] == "control"
    assert rows[("risk_lover", "risk_lover_method")]["role"] == "props"
    assert rows[("steady", "steady")]["window"]["tickets"] == 41
    assert full["backtest"]["caveats"] == ["Closing prices only before 2024."]
    json.dumps(full, allow_nan=False)


# ------------------------------------------------------------------ the app

def test_the_app_shows_three_parlays_before_the_doubt_card():
    app = (ROOT / "app" / "app.js").read_text()
    boot = app[app.index("BOOT.card = async"):app.index("function stamp()")]
    assert boot.index("parlayTickets(d)") < boot.index("doubtCard(d.fights")
    assert "parlayCard(" not in app, "the fixed 2/3/4-leg parlays are replaced"
    assert "Three parlays" in app and "Parlay profiles" in app
    assert "The market is hard to beat." in app
    assert "Not measured yet" in app
    assert "parlayRecord(d.parlay_profiles)" in app
    assert "d.caveats.parlay" in app or "cav.parlay" in app, "the old caveat stays visible"


def test_the_parlay_screens_use_only_the_shared_classes():
    """test_desk_ui holds the stylesheet to shared/desk.css; this holds the
    new markup to the classes that sheet already defines."""
    app = (ROOT / "app" / "app.js").read_text()
    css = (ROOT / "shared" / "desk.css").read_text()
    parts = [app[app.index("/* ------------------------------------------------------- three parlays */"):
                 app.index("const BOOT = {};")],
             app[app.index("function parlayRecord(pp)"):app.index("BOOT.record = async")]]
    used = set()
    for part in parts:
        for attr in re.findall(r'class="([^"$]*)"', part):
            used.update(attr.split())
    assert used, "no classes found - the markers moved"
    missing = sorted(c for c in used if not re.search(rf"\.{re.escape(c)}\b", css))
    assert not missing, missing


def test_mixed_finding_nothing_falls_back_to_the_named_profile_and_says_so(card, monkeypatch):
    """Where mixed ships but builds no ticket on a card, the named profile's
    stands in, the entry says which built it, and the record shown is the
    combined procedure's (mixed_else@band), not mixed's alone."""
    import parlay_profiles as pp
    measured = _measured({"steady": "steady", "risker": "risker", "risk_lover": "mixed"})
    real = pp.build

    def no_mixed(legs, profile, band=None):
        return None if profile == "mixed" else real(legs, profile, band)
    monkeypatch.setattr(pp, "build", no_mixed)
    out = ax.parlay_profiles_payload(card, None, measured, event_date=DATE)
    rl = out[2]
    assert rl["rule"] == "mixed_else@risk_lover"
    if rl["legs"]:
        assert rl["built_by"] == "risk_lover"
        assert "mixed found no" in rl["blurb"].lower()
    else:
        assert "mixed rule or the risk lover profile" in rl["missing_reason"]
