"""Same-game tickets (sgp-v2): one test per clause of nfl_sgp2's docstring -
the confident games, the pick's side, the TD floor and the rookie's, the
pick's best TD scorer in SMALL, v1's totals, everything kept from v1, the
two numbers from the pre-registered grids, and the reasons a ticket is not
built."""

import math
import random

import pytest

import nfl_sgp as sgp
import nfl_sgp2 as sgp2

CONF, FLOOR = 0.65, 0.40          # a pair of the grid, so the tests do not hang on the chosen one


def _td(gsis_id, team, prob, rookie=False, decimal=None):
    return {"gsis_id": gsis_id, "name": gsis_id, "position": "RB", "team": team, "prob": prob,
            "rookie": rookie, "decimal": decimal}


def _game(td=(), p_home=0.70, spread=6.5, spread_prices=(1.91, 1.91), total=47.5,
          total_prices=(1.87, 1.95), mls=(1.4, 3.1)):
    return sgp.game_legs("KC", "BUF", p_home=p_home, spread_line=spread,
                         home_spread_price=spread_prices[0], away_spread_price=spread_prices[1],
                         total_line=total, over_price=total_prices[0], under_price=total_prices[1],
                         home_ml=mls[0], away_ml=mls[1], td=list(td))


def _build(legs, p_home=0.70, confident=CONF, td_floor=FLOOR):
    pick, p = sgp2.pick_of("KC", "BUF", p_home)
    return {t["kind"]: t for t in sgp2.build_tickets(legs, pick, p, confident, td_floor)}


def _keys(ticket):
    return [leg["key"] for leg in ticket["legs"]]


# ---------------------------------------------------------------- the two numbers

def test_the_grids_and_the_coverage_floor_are_the_pre_registered_ones():
    assert sgp2.VERSION == "sgp-v2" and sgp.VERSION == "sgp-v1"
    assert sgp2.CONFIDENT_GRID == (0.60, 0.65, 0.70, 0.75)
    assert sgp2.TD_FLOOR_GRID == (0.35, 0.40, 0.45, 0.50)
    assert sgp2.COVERAGE_MIN == 0.25 and sgp2.DEV == (2014, 2019) and sgp2.CONFIRM == (2020, 2025)
    # Chosen from the grids (or not yet, and then nothing is built).
    if sgp2.CONFIDENT is None:
        assert sgp2.TD_FLOOR is None and sgp2.CHOSEN_ON is None
    else:
        assert sgp2.CONFIDENT in sgp2.CONFIDENT_GRID and sgp2.TD_FLOOR in sgp2.TD_FLOOR_GRID
        assert sgp2.CHOSEN_ON is not None


def test_until_the_numbers_are_chosen_no_ticket_is_built():
    t = sgp2.build_tickets(_game([_td("a", "KC", 0.6)]), "KC", 0.7, None, None)
    assert [x["reason"] for x in t] == [sgp2.NOT_CHOSEN] * 4 and not any(x["built"] for x in t)


# ---------------------------------------------------------------- GAMES

def test_the_pick_is_the_blend_side_as_nfl_predict_picks_it():
    assert sgp2.pick_of("KC", "BUF", 0.7) == ("KC", 0.7)
    assert sgp2.pick_of("KC", "BUF", 0.3) == ("BUF", pytest.approx(0.7))
    assert sgp2.pick_of("KC", "BUF", 0.5) == ("KC", 0.5)
    assert sgp2.pick_of("KC", "BUF", None) == (None, None)
    assert sgp2.pick_of("KC", "BUF", float("nan")) == (None, None)


def test_only_a_confident_game_gets_tickets_and_every_other_says_why():
    assert sgp2.eligible(0.65, 0.65) and not sgp2.eligible(0.6499, 0.65)
    assert not sgp2.eligible(None, 0.65) and not sgp2.eligible(0.9, None)
    legs = _game([_td("a", "KC", 0.6), _td("b", "BUF", 0.5)])
    assert _build(legs, p_home=0.65)["small"]["built"]
    below = sgp2.build_tickets(legs, "KC", 0.6499, CONF, FLOOR)
    assert not any(t["built"] for t in below) and [t["kind"] for t in below] == list(sgp.KINDS)
    assert {t["reason"] for t in below} == {sgp2.not_confident(CONF)}
    for c in sgp2.CONFIDENT_GRID:                          # the bar in the words, from the constant
        assert sgp2.not_confident(c) == f"not a confident game: the pick's chance is below {c:.0%}"
    # The away pick is confident by its own chance, 1 - P(home).
    assert _build(legs, p_home=0.30)["small"]["built"]
    none = sgp2.build_tickets(legs, None, None, CONF, FLOOR)
    assert {t["reason"] for t in none} == {sgp2.NO_PICK}


# ---------------------------------------------------------------- RESULT

def test_result_legs_are_only_the_picks_win_and_spread():
    legs = _game([_td("a", "KC", 0.6), _td("b", "KC", 0.5), _td("c", "BUF", 0.45)],
                 spread_prices=(2.2, 1.7))                  # BUF's spread the likelier side
    assert {leg["key"] for leg in sgp2.pool(legs, "KC", FLOOR) if leg["kind"] in sgp.RESULT_KINDS} == {
        "WIN:KC", "SPREAD:KC"}
    t = _build(legs)
    assert _keys(t["small"])[1] == _keys(t["medium"])[1] == "WIN:KC"
    assert "SPREAD:KC" in _keys(t["large"])                 # the pick's WIN is in two tickets already
    # No pick's spread: LARGE has no result leg, though the other team's WIN is there.
    t = _build(_game([_td("a", "KC", 0.6), _td("b", "KC", 0.5), _td("c", "BUF", 0.45)], spread=None))
    assert not t["large"]["built"] and t["large"]["reason"] == sgp2.NO_RESULT


# ---------------------------------------------------------------- TD

def test_a_td_leg_needs_the_floor():
    legs = _game([_td("a", "KC", 0.40), _td("b", "BUF", 0.3999), _td("c", "KC", 0.2)])
    assert [leg["key"] for leg in sgp2.pool(legs, "KC", FLOOR) if leg["kind"] == "TD"] == ["TD:a"]
    t = _build(legs)
    assert _keys(t["small"]) == ["TD:a", "WIN:KC"]
    assert t["large"]["reason"] == sgp2.no_td(FLOOR)          # his one leg is in two tickets already
    t = _build(_game([_td("a", "KC", 0.6), _td("b", "BUF", 0.5), _td("c", "KC", 0.39)]))
    assert t["large"]["reason"] == sgp2.no_second_td(FLOOR)   # c is under the floor
    for f in sgp2.TD_FLOOR_GRID:                            # each reason names the bar it used
        assert all(f"{f:.0%}" in why(f) for why in (sgp2.no_td, sgp2.no_second_td, sgp2.no_third))


def test_the_rookie_leg_keeps_v1s_floor_and_is_on_the_picks_team():
    assert sgp.ROOKIE_FLOOR == 0.25 < min(sgp2.TD_FLOOR_GRID)
    # Under TD_FLOOR but at ROOKIE_FLOOR, on the pick's team: ROOKIE only.
    t = _build(_game([_td("a", "KC", 0.6), _td("b", "BUF", 0.5), _td("r", "KC", 0.30, rookie=True)]))
    assert _keys(t["rookie"]) == ["TD:r", "WIN:KC"]
    assert all("TD:r" not in _keys(t[k]) for k in ("small", "medium", "large"))
    # On the other team: no rookie ticket, whatever his chance.
    for p in (0.30, 0.55):
        t = _build(_game([_td("a", "KC", 0.6), _td("r", "BUF", p, rookie=True)]))
        assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp2.NO_ROOKIE
    # ...though at the floor he is an ordinary TD leg for the other tickets.
    assert "TD:r" in _keys(t["medium"]) + _keys(t["large"])
    t = _build(_game([_td("a", "KC", 0.6), _td("r", "KC", 0.2499, rookie=True)]))
    assert t["rookie"]["reason"] == sgp2.NO_ROOKIE
    assert f"{sgp.ROOKIE_FLOOR:.0%}" in sgp2.NO_ROOKIE and "pick's team" in sgp2.NO_ROOKIE


# ---------------------------------------------------------------- SMALL

def test_small_takes_the_picks_best_td_scorer_else_the_games_best():
    legs = _game([_td("kc", "KC", 0.42), _td("buf", "BUF", 0.55)])
    t = _build(legs)
    assert _keys(t["small"]) == ["TD:kc", "WIN:KC"]          # the pick's own, though BUF's is likelier
    assert _keys(sgp2.game_best_small(legs, "KC", 0.7, CONF, FLOOR)) == ["TD:buf", "WIN:KC"]
    assert _keys(t["medium"])[0] == "TD:buf"                  # MEDIUM keeps v1's best TD
    # No TD on the pick's team clears the floor: the game's best that does.
    t = _build(_game([_td("kc", "KC", 0.38), _td("buf", "BUF", 0.55)]))
    assert _keys(t["small"]) == ["TD:buf", "WIN:KC"]
    # None clears it: no SMALL, and why.
    t = _build(_game([_td("kc", "KC", 0.38)]))
    assert not t["small"]["built"] and t["small"]["reason"] == sgp2.no_td(FLOOR)
    alt = sgp2.game_best_small(_game([_td("kc", "KC", 0.38)]), "KC", 0.7, CONF, FLOOR)
    assert not alt["built"] and alt["reason"] == sgp2.no_td(FLOOR)


# ---------------------------------------------------------------- TOTAL

def test_totals_are_v1s_and_only_where_v1_puts_one():
    legs = _game([_td("a", "KC", 0.6), _td("b", "KC", 0.45)], total_prices=(1.80, 2.05))
    assert {leg["key"] for leg in sgp2.pool(legs, "KC", FLOOR) if leg["kind"] == "TOTAL"} == {
        "TOTAL:OVER", "TOTAL:UNDER"}
    t = _build(legs)
    assert _keys(t["medium"])[2] == "TOTAL:OVER" and _keys(t["small"]) == ["TD:a", "WIN:KC"]
    # MEDIUM's third slot competes with the TD legs: a likelier one takes it.
    t = _build(_game([_td("a", "KC", 0.6), _td("b", "KC", 0.58), _td("c", "BUF", 0.5)],
                     total_prices=(1.80, 2.05)))
    assert _keys(t["medium"]) == ["TD:a", "WIN:KC", "TD:b"]
    assert _keys(t["large"]) == ["TD:b", "TD:c", "SPREAD:KC", "TOTAL:OVER"]   # its fourth slot a total
    t = _build(_game([_td("a", "KC", 0.6), _td("b", "KC", 0.58), _td("c", "BUF", 0.5)], total=None))
    assert t["large"]["reason"] == sgp2.NO_TOTAL == sgp.NO_TOTAL


# ---------------------------------------------------------------- everything else is v1's

def test_tickets_come_in_v1s_order_and_shapes():
    t = sgp2.build_tickets(_game([_td("a", "KC", 0.6), _td("b", "BUF", 0.5)]), "KC", 0.7, CONF, FLOOR)
    assert [x["kind"] for x in t] == list(sgp.KINDS) and sgp2.KINDS is sgp.KINDS
    assert all(len(x["legs"]) == sgp.LEGS[x["kind"]] for x in t if x["built"])


def test_the_rookie_ticket_identical_to_smalls_moves_to_the_next_rookie():
    t = _build(_game([_td("r1", "KC", 0.5, rookie=True), _td("r2", "KC", 0.3, rookie=True)]))
    assert _keys(t["small"]) == ["TD:r1", "WIN:KC"] and _keys(t["rookie"]) == ["TD:r2", "WIN:KC"]
    t = _build(_game([_td("r1", "KC", 0.5, rookie=True), _td("v", "KC", 0.45)]))
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp2.ROOKIE_IS_SMALL


def test_the_rookie_ticket_needs_the_picks_win_available(monkeypatch):
    # With the cap of two the pick's WIN is always free when ROOKIE is built
    # second; a cap of one exercises v1's branch, with v1's words.
    monkeypatch.setattr(sgp, "USE_CAP", 1)
    t = _build(_game([_td("a", "KC", 0.5), _td("r", "KC", 0.3, rookie=True)]))
    assert _keys(t["small"]) == ["TD:a", "WIN:KC"]
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp2.NO_ROOKIE_WIN == sgp.NO_ROOKIE_WIN


def test_an_empty_slot_leaves_its_ticket_unbuilt_with_its_reason():
    t = _build(_game([]))
    assert [t[k]["reason"] for k in sgp.KINDS] == [sgp2.no_td(FLOOR), sgp2.NO_ROOKIE, sgp2.no_td(FLOOR),
                                                   sgp2.no_td(FLOOR)]
    assert all(not x["built"] and x["legs"] == [] and x["chance"] is None for x in t.values())
    t = _build(_game([_td("a", "KC", 0.5)], total=None))
    assert t["medium"]["reason"] == sgp2.no_third(FLOOR)
    t = _build(sgp.game_legs("KC", "BUF", p_home=None, td=[_td("a", "KC", 0.5)]), p_home=None)
    assert t["small"]["reason"] == sgp2.NO_PICK


def test_every_ticket_keeps_the_cap_one_result_one_total_and_distinct_players():
    rng = random.Random(5)
    for _ in range(400):
        legs, p_home = _random_game(rng)
        pick, p = sgp2.pick_of("KC", "BUF", p_home)
        for conf in sgp2.CONFIDENT_GRID:
            for floor in sgp2.TD_FLOOR_GRID:
                tickets = sgp2.build_tickets(legs, pick, p, conf, floor)
                uses = {}
                for t in tickets:
                    for k in _keys(t):
                        uses[k] = uses.get(k, 0) + 1
                    if not t["built"]:
                        continue
                    assert p >= conf
                    kinds = [leg["kind"] for leg in t["legs"]]
                    players = [leg["gsis_id"] for leg in t["legs"] if leg["kind"] == "TD"]
                    assert sum(k in sgp.RESULT_KINDS for k in kinds) == 1 and kinds.count("TOTAL") <= 1
                    assert len(players) == len(set(players)) and len(kinds) == sgp.LEGS[t["kind"]]
                    for leg in t["legs"]:
                        if leg["kind"] in sgp.RESULT_KINDS:
                            assert leg["team"] == pick
                        elif leg["kind"] == "TD" and t["kind"] == "rookie":
                            assert leg["rookie"] and leg["team"] == pick and leg["chance"] >= sgp.ROOKIE_FLOOR
                        elif leg["kind"] == "TD":
                            assert leg["chance"] >= floor
                assert max(uses.values(), default=0) <= sgp.USE_CAP


def test_tickets_are_v1s_so_v1_grades_voids_and_prices_them():
    t = _build(_game([_td("a", "KC", 0.6, decimal=1.9), _td("b", "BUF", 0.5), _td("c", "KC", 0.45)]))
    small = t["small"]
    assert small["chance"] == pytest.approx(0.6 * 0.7) and small["est_payout"] == pytest.approx(1.9 * 1.4)
    assert t["medium"]["chance"] == pytest.approx(math.prod(leg["chance"] for leg in t["medium"]["legs"]))
    assert t["large"]["est_payout"] is None and "b anytime TD" in t["large"]["est_payout_reason"]
    # A tie voids the WIN and the ticket pays on the TD; a player who did not play voids his leg.
    g = sgp.grade_ticket(small, sgp.outcome("KC", "BUF", 20, 20, {"a": 1}))
    assert g["result"] == sgp.WON and g["payout"] == pytest.approx(1.9)
    assert sgp.grade_ticket(small, sgp.outcome("KC", "BUF", 27, 20, {}))["chance_graded"] == pytest.approx(0.7)
    m = sgp.summarise([dict(g, season=2020)] + sgp2.build_tickets([], "KC", 0.5, CONF, FLOOR))
    assert m["small"]["won"] == 1 and m["small"]["not_built"] == 1


def test_the_rules_words_take_their_numbers_from_the_constants_and_claim_no_edge():
    words = sgp2.rules(CONF, FLOOR)
    assert "65%" in words["games"] and "40%" in words["td"] and f"{sgp.ROOKIE_FLOOR:.0%}" in words["td"]
    assert sgp2.rules(None, None) is None
    assert sgp2.OBJECTIVE.startswith("the highest SMALL hit rate") and f"{sgp2.COVERAGE_MIN:.0%}" in sgp2.OBJECTIVE
    assert "Nothing here claims an edge" in sgp2.NO_EDGE and "not bets it rates profitable" in sgp2.NO_EDGE


def test_untouched_says_it_covers_the_two_numbers_and_why_not_smalls_clause():
    """FOUND AFTER THE RULE WAS FIXED: SMALL's pick's-team clause rests on
    the docstring's P(TD | won) figures, whose seasons (SMALL_RESEARCH, as
    the docstring gives them) overlap CONFIRM - so the note says CONFIRM is
    not a clean holdout for that clause, every span from the constants."""
    doc = " ".join(sgp2.__doc__.split())
    lo, hi = sgp2.SMALL_RESEARCH
    assert "P(TD | his team won) was" in doc and f"({lo}-{hi}, the research before v1)" in doc
    note = sgp2.UNTOUCHED_NOTE
    assert note == sgp2.untouched_note() and note.startswith("Untouched means the two numbers")
    assert f"{sgp2.DEV[0]}-{sgp2.DEV[1]} alone" in note and f"a {lo}-{hi} statistic" in note
    both = f"{max(lo, sgp2.CONFIRM[0])}-{min(hi, sgp2.CONFIRM[1])}"
    assert f"for that clause {both} is not a clean holdout" in note
    # Research that does not overlap the seasons measured says nothing.
    assert sgp2.untouched_note(research=(2010, 2013)) is None


# ---------------------------------------------------------------- helpers

def _random_game(rng):
    tds = [_td(f"p{i}", rng.choice(["KC", "BUF"]), round(rng.uniform(0.02, 0.7), 2),
               rookie=rng.random() < 0.3) for i in range(rng.randint(0, 8))]
    price = lambda: rng.choice([None, 1.8, 1.91, 2.05])   # noqa: E731
    p_home = rng.choice([None, round(rng.uniform(0.1, 0.9), 2)])
    return _game(tds, p_home=p_home, spread=rng.choice([None, 3.5, -7, 0]), spread_prices=(price(), price()),
                 total=rng.choice([None, 44.5, 51]), total_prices=(price(), price())), p_home
