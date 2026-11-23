"""Same-game tickets (sgp-v1): one test per clause of nfl_sgp's docstring -
the build order, the cap, the shape of each ticket, the rookie rules, the
reasons a ticket is not built, the voids, the price and the meters."""

import math
import random

import pandas as pd
import pytest

import nfl_sgp as sgp


def _td(gsis_id, team, prob, rookie=False, name=None, decimal=None):
    return {"gsis_id": gsis_id, "name": name or gsis_id, "position": "RB", "team": team,
            "prob": prob, "rookie": rookie, "decimal": decimal}


def _game(td=(), p_home=0.62, spread=3.5, spread_prices=(1.91, 1.91), total=47.5,
          total_prices=(1.87, 1.95), mls=(1.6, 2.4)):
    return sgp.game_legs("KC", "BUF", p_home=p_home, spread_line=spread,
                         home_spread_price=spread_prices[0], away_spread_price=spread_prices[1],
                         total_line=total, over_price=total_prices[0], under_price=total_prices[1],
                         home_ml=mls[0], away_ml=mls[1], td=list(td))


TDS = [_td("a", "KC", 0.45), _td("b", "BUF", 0.38, rookie=True), _td("c", "KC", 0.30, rookie=True),
       _td("d", "BUF", 0.20)]


def _keys(ticket):
    return [leg["key"] for leg in ticket["legs"]]


def _by_kind(tickets):
    return {t["kind"]: t for t in tickets}


# ---------------------------------------------------------------- the legs

def test_win_legs_read_the_blend_and_the_away_side_is_one_minus_it():
    legs = {leg["key"]: leg for leg in _game(p_home=0.62)}
    assert legs["WIN:KC"]["chance"] == pytest.approx(0.62)
    assert legs["WIN:BUF"]["chance"] == pytest.approx(0.38)
    assert not any(leg["kind"] == "WIN" for leg in _game(p_home=None))


def test_spread_and_total_chances_are_the_markets_proportional_no_vig():
    legs = {leg["key"]: leg for leg in _game(spread_prices=(1.80, 2.05), total_prices=(1.87, 1.95))}
    home = (1 / 1.80) / (1 / 1.80 + 1 / 2.05)
    assert legs["SPREAD:KC"]["chance"] == pytest.approx(home)
    assert legs["SPREAD:BUF"]["chance"] == pytest.approx(1 - home)
    assert legs["TOTAL:OVER"]["chance"] == pytest.approx((1 / 1.87) / (1 / 1.87 + 1 / 1.95))
    assert sgp.no_vig(1.91, 1.91) == pytest.approx(0.5)
    # No price, no chance: the leg is not there.
    assert not any(leg["kind"] == "SPREAD" for leg in _game(spread_prices=(1.91, None)))
    assert not any(leg["kind"] == "TOTAL" for leg in _game(total=None))


def test_the_spread_line_is_the_home_sides_expected_margin():
    legs = {leg["key"]: leg for leg in _game(spread=3.5)}
    assert legs["SPREAD:KC"]["line"] == -3.5 and legs["SPREAD:BUF"]["line"] == 3.5
    assert legs["SPREAD:KC"]["label"] == "KC -3.5"


def test_only_td_candidates_with_a_chance_become_legs():
    legs = _game(td=[_td("a", "KC", 0.4), _td("x", "KC", None), _td("y", "BUF", float("nan"))])
    assert [leg["gsis_id"] for leg in legs if leg["kind"] == "TD"] == ["a"]
    leg = next(leg for leg in legs if leg["kind"] == "TD")
    assert leg["key"] == "TD:a" and leg["rookie"] is False and leg["decimal"] is None


def test_american_prices_convert_to_decimal():
    assert sgp.decimal_from_american(150) == pytest.approx(2.5)
    assert sgp.decimal_from_american(-200) == pytest.approx(1.5)
    assert sgp.decimal_from_american(None) is None and sgp.decimal_from_american(50) is None


def test_a_rookie_has_no_game_in_any_earlier_season():
    pg = pd.DataFrame({"player_id": ["vet", "vet", "new"], "season": [2019, 2020, 2020]})
    first = sgp.first_seasons(pg)
    assert not sgp.is_rookie(first, "vet", 2020)
    assert sgp.is_rookie(first, "vet", 2019)            # his first season, live and in the replay
    assert sgp.is_rookie(first, "new", 2020) and sgp.is_rookie(first, "never", 2020)


# ---------------------------------------------------------------- the tickets

def test_tickets_come_in_the_build_order():
    assert [t["kind"] for t in sgp.build_tickets(_game(TDS))] == list(sgp.KINDS)
    assert sgp.KINDS == ("small", "rookie", "medium", "large")


def test_each_ticket_has_its_shape():
    t = _by_kind(sgp.build_tickets(_game(TDS)))
    assert _keys(t["small"]) == ["TD:a", "WIN:KC"]                    # best TD + best result
    assert _keys(t["rookie"]) == ["TD:b", "WIN:BUF"]                  # best rookie + HIS team's WIN
    assert _keys(t["medium"]) == ["TD:a", "WIN:KC", "TOTAL:OVER"]     # + best of TOTAL and other TDs
    # LARGE: TD:a and WIN:KC are in two tickets already; the next best take their slots.
    assert _keys(t["large"]) == ["TD:b", "TD:c", "SPREAD:BUF", "TOTAL:OVER"]
    assert all(len(x["legs"]) == sgp.LEGS[k] for k, x in t.items())


def test_a_slot_takes_the_highest_chance_leg_still_available():
    # The best TD is in SMALL and MEDIUM; LARGE starts from the next one.
    t = _by_kind(sgp.build_tickets(_game(TDS)))
    assert "TD:a" in _keys(t["small"]) and "TD:a" in _keys(t["medium"])
    assert "TD:a" not in _keys(t["large"]) and _keys(t["large"])[0] == "TD:b"
    # MEDIUM's third slot is a TD on another player when it beats the totals.
    t = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.6), _td("e", "KC", 0.55)],
                                         total_prices=(1.91, 1.91))))
    assert _keys(t["medium"]) == ["TD:a", "WIN:KC", "TD:e"]


def test_no_leg_is_in_more_than_two_tickets():
    rng = random.Random(7)
    for _ in range(300):
        tickets = sgp.build_tickets(_random_game(rng))
        uses = {}
        for t in tickets:
            for k in _keys(t):
                uses[k] = uses.get(k, 0) + 1
        assert max(uses.values(), default=0) <= sgp.USE_CAP == 2


def test_a_ticket_holds_one_result_leg_one_total_leg_and_td_legs_on_different_players():
    rng = random.Random(11)
    for _ in range(300):
        for t in sgp.build_tickets(_random_game(rng)):
            kinds = [leg["kind"] for leg in t["legs"]]
            players = [leg["gsis_id"] for leg in t["legs"] if leg["kind"] == "TD"]
            assert sum(k in sgp.RESULT_KINDS for k in kinds) <= 1
            assert kinds.count("TOTAL") <= 1
            assert len(players) == len(set(players))
            if not t["built"]:
                continue
            assert len(t["legs"]) == sgp.LEGS[t["kind"]]
            shape = ["R" if k in sgp.RESULT_KINDS else k for k in kinds]
            assert shape in {"small": [["TD", "R"]], "rookie": [["TD", "R"]],
                             "medium": [["TD", "R", "TOTAL"], ["TD", "R", "TD"]],
                             "large": [["TD", "TD", "R", "TOTAL"]]}[t["kind"]]
            if t["kind"] == "rookie":
                td, win = t["legs"]
                assert td["rookie"] and td["chance"] >= sgp.ROOKIE_FLOOR
                assert win["kind"] == "WIN" and win["team"] == td["team"]


def test_ties_go_to_the_leg_key_alphabetically():
    t = _by_kind(sgp.build_tickets(_game([_td("zed", "KC", 0.4), _td("amy", "BUF", 0.4)])))
    assert _keys(t["small"])[0] == "TD:amy"
    # Level spread prices: both sides 0.5, BUF before KC.
    legs = sgp.game_legs("KC", "BUF", spread_line=3.5, home_spread_price=1.91, away_spread_price=1.91,
                         td=[_td("a", "KC", 0.4)])
    assert _keys(_by_kind(sgp.build_tickets(legs))["small"]) == ["TD:a", "SPREAD:BUF"]


def test_a_rookie_needs_a_chance_of_the_floor_or_more():
    assert sgp.ROOKIE_FLOOR == 0.25
    under = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.5), _td("r", "BUF", 0.2499, rookie=True)])))
    assert not under["rookie"]["built"] and under["rookie"]["reason"] == sgp.NO_ROOKIE
    at = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.5), _td("r", "BUF", 0.25, rookie=True)])))
    assert _keys(at["rookie"]) == ["TD:r", "WIN:BUF"]


def test_the_rookie_ticket_takes_his_own_teams_win_even_as_the_underdog():
    t = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.5), _td("r", "BUF", 0.3, rookie=True)],
                                         p_home=0.8)))
    assert _keys(t["rookie"]) == ["TD:r", "WIN:BUF"]


def test_a_rookie_ticket_identical_to_smalls_moves_to_the_next_rookie():
    tds = [_td("r1", "KC", 0.5, rookie=True), _td("r2", "BUF", 0.3, rookie=True)]
    t = _by_kind(sgp.build_tickets(_game(tds)))
    assert _keys(t["small"]) == ["TD:r1", "WIN:KC"]
    assert _keys(t["rookie"]) == ["TD:r2", "WIN:BUF"]
    # With no next rookie: no ticket, and why.
    t = _by_kind(sgp.build_tickets(_game(tds[:1] + [_td("v", "BUF", 0.3)])))
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp.ROOKIE_IS_SMALL
    # Not identical when SMALL's result leg is another one: the same rookie stays.
    t = _by_kind(sgp.build_tickets(_game(tds[:1], p_home=0.3)))
    assert _keys(t["small"]) == ["TD:r1", "WIN:BUF"] and _keys(t["rookie"]) == ["TD:r1", "WIN:KC"]


def test_no_win_for_the_rookies_team_means_no_rookie_ticket(monkeypatch):
    t = _by_kind(sgp.build_tickets(_game([_td("r", "KC", 0.4, rookie=True)], p_home=None)))
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp.NO_ROOKIE_WIN
    # That WIN no longer available: not the next rookie, no ticket. With v1's
    # cap of two the WIN is always free when ROOKIE is built second, so the
    # branch is exercised with a cap of one.
    monkeypatch.setattr(sgp, "USE_CAP", 1)
    t = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.5), _td("r", "KC", 0.4, rookie=True),
                                          _td("s", "BUF", 0.3, rookie=True)])))
    assert _keys(t["small"]) == ["TD:a", "WIN:KC"]
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp.NO_ROOKIE_WIN


def test_an_empty_slot_leaves_the_ticket_unbuilt_with_its_reason():
    t = _by_kind(sgp.build_tickets(_game([])))
    assert [t[k]["reason"] for k in sgp.KINDS] == [sgp.NO_TD, sgp.NO_ROOKIE, sgp.NO_TD, sgp.NO_TD]
    assert all(not x["built"] and x["legs"] == [] and x["chance"] is None for x in t.values())
    # One TD player: in SMALL and MEDIUM, so none is left for LARGE.
    one = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.4)])))
    assert one["medium"]["built"] and one["large"]["reason"] == sgp.NO_TD
    two = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.4), _td("b", "BUF", 0.3)])))
    assert two["large"]["reason"] == sgp.NO_SECOND_TD
    no_result = _by_kind(sgp.build_tickets(sgp.game_legs("KC", "BUF", td=TDS)))
    assert no_result["small"]["reason"] == sgp.NO_RESULT
    no_total = _by_kind(sgp.build_tickets(_game(TDS, total=None)))
    assert no_total["large"]["reason"] == sgp.NO_TOTAL


def test_a_ticket_is_never_filled_with_a_leg_outside_its_rule():
    # MEDIUM's third slot is a TOTAL or another player's TD - never a second
    # result leg, even with result legs to spare.
    t = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.4)], total=None)))
    assert not t["medium"]["built"] and t["medium"]["reason"] == sgp.NO_THIRD


def test_the_chance_is_the_product_and_the_est_payout_the_product_of_prices():
    t = _by_kind(sgp.build_tickets(_game([_td("a", "KC", 0.45, decimal=2.2), _td("b", "BUF", 0.3),
                                          _td("c", "KC", 0.25, decimal=3.0)])))
    small = t["small"]
    assert small["chance"] == pytest.approx(0.45 * 0.62)
    assert small["est_payout"] == pytest.approx(2.2 * 1.6) and small["est_payout_reason"] is None
    large = t["large"]
    assert large["chance"] == pytest.approx(math.prod(leg["chance"] for leg in large["legs"]))
    assert large["est_payout"] is None and "b anytime TD" in large["est_payout_reason"]


# ---------------------------------------------------------------- grading

def _result(hs, as_, players=None):
    return sgp.outcome("KC", "BUF", hs, as_, players)


def test_a_tie_voids_a_win_leg():
    win = sgp.win_leg("KC", 0.6, 1.6)
    assert sgp.grade_leg(win, _result(20, 20)) == sgp.VOID
    assert sgp.grade_leg(win, _result(23, 20)) == sgp.WON
    assert sgp.grade_leg(sgp.win_leg("BUF", 0.4), _result(23, 20)) == sgp.LOST


def test_a_push_voids_a_spread_or_total_leg():
    assert sgp.grade_leg(sgp.spread_leg("KC", -3, 0.5), _result(23, 20)) == sgp.VOID
    assert sgp.grade_leg(sgp.spread_leg("KC", -3.5, 0.5), _result(23, 20)) == sgp.LOST
    assert sgp.grade_leg(sgp.spread_leg("BUF", 3.5, 0.5), _result(23, 20)) == sgp.WON
    assert sgp.grade_leg(sgp.total_leg("OVER", 43, 0.5), _result(23, 20)) == sgp.VOID
    assert sgp.grade_leg(sgp.total_leg("UNDER", 43.5, 0.5), _result(23, 20)) == sgp.WON
    assert sgp.grade_leg(sgp.total_leg("OVER", 43.5, 0.5), _result(23, 20)) == sgp.LOST


def test_a_td_leg_is_one_rushing_or_receiving_td_and_void_when_he_did_not_play():
    leg = sgp.td_leg("a", "A", "RB", "KC", 0.4)
    assert sgp.grade_leg(leg, _result(23, 20, {"a": 1})) == sgp.WON
    assert sgp.grade_leg(leg, _result(23, 20, {"a": 0})) == sgp.LOST
    assert sgp.grade_leg(leg, _result(23, 20, {"z": 2})) == sgp.VOID
    assert sgp.grade_leg(leg, _result(23, 20, None)) is None        # box score not in yet


def test_a_ticket_with_a_void_leg_pays_on_the_remaining_legs():
    ticket = sgp._ticket("small", [sgp.td_leg("a", "A", "RB", "KC", 0.4, decimal=2.5),
                                   sgp.win_leg("KC", 0.6, 1.6)])
    g = sgp.grade_ticket(ticket, _result(20, 20, {"a": 1}))
    assert g["result"] == sgp.WON and g["payout"] == pytest.approx(2.5)
    assert g["chance_graded"] == pytest.approx(0.4) and g["chance"] == pytest.approx(0.24)
    assert sgp.grade_ticket(ticket, _result(20, 20, {})) ["result"] == sgp.VOID
    assert sgp.grade_ticket(ticket, _result(20, 20, {}))["payout"] == 1.0
    assert sgp.grade_ticket(ticket, _result(23, 20, {"a": 0}))["result"] == sgp.LOST
    assert sgp.grade_ticket(ticket, _result(23, 20, None))["result"] is None
    # A lost leg loses the ticket even while another waits.
    assert sgp.grade_ticket(ticket, _result(17, 20, None))["result"] == sgp.LOST
    assert sgp.grade_ticket(ticket, _result(23, 20, {"a": 2}))["payout"] == pytest.approx(4.0)


# ---------------------------------------------------------------- meters

def _graded(kind, result, chance=0.3, payout=3.0, priced=True):
    return {"kind": kind, "built": True, "reason": None, "legs": [], "chance": chance,
            "chance_graded": chance, "est_payout": payout if priced else None,
            "result": result, "payout": payout}


def test_the_meters_count_hit_rate_chance_and_return_per_kind():
    rows = [_graded("small", sgp.WON, 0.3, 3.0), _graded("small", sgp.LOST, 0.2),
            _graded("small", sgp.VOID, 0.25, 1.0), _graded("small", None),
            sgp._unbuilt("small", sgp.NO_TD), _graded("large", sgp.LOST, 0.05)]
    m = sgp.summarise(rows)
    s = m["small"]
    assert (s["tickets"], s["graded"], s["won"], s["lost"], s["void"], s["pending"]) == (4, 3, 1, 1, 1, 1)
    assert s["not_built"] == 1 and s["not_built_why"] == {sgp.NO_TD: 1}
    assert s["hit_rate"] == pytest.approx(0.5) and s["mean_chance"] == pytest.approx(0.25)
    # $10 a ticket: +20, -10, the void's stake back.
    assert s["profit"] == pytest.approx(10.0) and s["roi"] == pytest.approx(10.0 / 30.0)
    assert m["large"]["roi"] == pytest.approx(-1.0)
    assert m["rookie"]["tickets"] == 0 and m["rookie"]["hit_rate"] is None and m["rookie"]["hit_rate_reason"]
    assert m["rookie"]["roi"] is None and m["rookie"]["roi_reason"] == "no ticket graded yet"


def test_return_is_missing_with_its_reason_unless_every_graded_ticket_has_a_price():
    rows = [_graded("small", sgp.WON), _graded("small", sgp.LOST, priced=False)]
    m = sgp.summarise(rows, no_price_reason=sgp.NO_TD_PRICE)["small"]
    assert m["roi"] is None and m["profit"] is None and m["roi_reason"] == sgp.NO_TD_PRICE
    m = sgp.summarise(rows)["small"]
    assert m["roi"] is None and m["roi_reason"].startswith("1 of 2 graded tickets")
    assert m["hit_rate"] == pytest.approx(0.5)         # hits are measured without prices


# ---------------------------------------------------------------- helpers

def _random_game(rng):
    tds = [_td(f"p{i}", rng.choice(["KC", "BUF"]), round(rng.uniform(0.02, 0.7), 2),
               rookie=rng.random() < 0.3) for i in range(rng.randint(0, 8))]
    price = lambda: rng.choice([None, 1.8, 1.91, 2.05])   # noqa: E731
    return _game(tds, p_home=rng.choice([None, round(rng.uniform(0.1, 0.9), 2)]),
                 spread=rng.choice([None, 3.5, -7, 0]), spread_prices=(price(), price()),
                 total=rng.choice([None, 44.5, 51]), total_prices=(price(), price()))
