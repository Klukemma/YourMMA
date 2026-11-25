"""The same-game tickets live from v2 on (nfl_predict, nfl_export): each game
still to come gets nfl_sgp2's four tickets from its pick and the same legs
v1 read, logged as sgp-v2; v1's tickets already logged stay exactly as they
were logged and are still graded, each ticket by its own version's rule; no
logged ticket is rebuilt after kickoff; every TD candidate carries his
anytime-TD price or why not, logged with the pick; and week.json's "sgp" is
v2's, with the chosen pair and the grid, and v1's meters beside it,
retired. No test touches the network: the TD prices are the hand-written
fixture's (tests/fixtures/odds)."""

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import nfl_export as nx
import nfl_odds as no
import nfl_predict as npd
import nfl_sgp as sgp
import nfl_sgp2 as sgp2

from conftest import ENGINE

sys.path.insert(0, str(ENGINE / "experiments"))
import nfl_sgp_backtest as bk  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "odds"
CLOCK = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
KICKOFF = pd.Timestamp("2026-10-25 20:25")
MARKET = {"source": "the-odds-api", "home_ml": -250.0, "away_ml": 205.0,
          "spread": 6.5, "spread_price_home": -110.0, "spread_price_away": -110.0,
          "spread_price_source": "the-odds-api", "total": 47.5, "total_price_over": -108.0,
          "total_price_under": -112.0, "total_price_source": "the-odds-api"}
# gsis, name, chance (None: a usage group not measured as calibrated). Three
# clear the TD floor (kc1, kc2, lv1); kc2 is KC's rookie, lv1 LV's.
HOME = [("kc1", "Tom Runner", 0.58), ("kc2", "Sam Tight", 0.49), ("kc3", "Nicholas Name", 0.22),
        ("kc4", "Chris Hands", None)]
AWAY = [("lv1", "Max Wide", 0.46), ("lv2", "DJ Catcher", 0.27), ("lv3", "Chris Hands Jr.", 0.12)]
FIRST = {"kc1": 2020, "kc2": 2026, "kc3": 2023, "kc4": 2021, "lv1": 2026, "lv2": 2022, "lv3": 2024}


def _list(rows):
    return [{"gsis_id": g, "name": n, "position": "WR", "prob": p, "tds_season": 0, "games_season": 0,
             "status": "active", "rank": i + 1, "group": "3+" if p is not None else "under 1"}
            for i, (g, n, p) in enumerate(rows)]


def _game(home=HOME, away=AWAY, home_prob=0.7, **kw):
    pick_home = home_prob is not None and home_prob >= 0.5
    g = {"game_id": "2026_07_LV_KC", "season": 2026, "week": 7, "game_type": "REG",
         "kickoff": "2026-10-25T20:25:00Z", "home": "KC", "away": "LV",
         "pick": "KC" if pick_home else "LV",
         "pick_prob": None if home_prob is None else (home_prob if pick_home else 1 - home_prob),
         "home_prob": home_prob, "market": dict(MARKET), "home_td_all": _list(home), "away_td_all": _list(away)}
    g["home_td"] = [dict(p) for p in g["home_td_all"] if p["prob"] is not None][:npd.TD_SHOWN]
    g["away_td"] = [dict(p) for p in g["away_td_all"] if p["prob"] is not None][:npd.TD_SHOWN]
    g.update(kw)
    return g


def _td_prices(game, aliases=None):
    cache = no.empty()
    key, entry = no.parse_td_event(json.loads((FIX / "td_event.json").read_text()))
    no.merge_td(cache, key, entry, now=CLOCK)
    return no.td_prices_for(cache, "LV", "KC", KICKOFF, npd._td_names(game, aliases or {}), now=CLOCK)


def _tickets(game=None, version=npd.LIVE_VERSION):
    game = game or _game()
    return npd.game_tickets(game, game["market"], _td_prices(game), FIRST, version=version)


def _keys(t):
    return [leg["key"] for leg in t["legs"]]


def _by_kind(tickets):
    return {t["kind"]: t for t in tickets}


# ---------------------------------------------------------------- the build

def test_a_confident_game_gets_v2s_four_tickets_from_the_pick_and_the_legs_v1_reads():
    tickets = _tickets()
    assert [t["kind"] for t in tickets] == list(sgp2.KINDS)
    t = _by_kind(tickets)
    assert _keys(t["small"]) == ["TD:kc1", "WIN:KC"]                   # the pick's best scorer
    assert _keys(t["rookie"]) == ["TD:kc2", "WIN:KC"]                  # the pick's team's rookie, not LV's
    assert _keys(t["medium"]) == ["TD:kc1", "SPREAD:KC", "TOTAL:UNDER"]
    assert _keys(t["large"]) == ["TD:kc2", "TD:lv1", "SPREAD:KC", "TOTAL:UNDER"]
    for x in tickets:
        assert x["version"] == sgp2.VERSION == npd.LIVE_VERSION and x["est"] is True
        assert (x["pick"], x["pick_prob"], x["confident_game"]) == ("KC", pytest.approx(0.7), True)
        assert (x["confident"], x["td_floor"]) == (sgp2.CONFIDENT, sgp2.TD_FLOOR)
        assert x["chance"] == pytest.approx(np.prod([leg["chance"] for leg in x["legs"]]))
        assert x["est_payout"] == pytest.approx(np.prod([leg["decimal"] for leg in x["legs"]]))
        assert all(leg["est"] is True for leg in x["legs"])
        assert all(leg["team"] == "KC" for leg in x["legs"] if leg["kind"] in sgp.RESULT_KINDS)
    td = t["small"]["legs"][0]
    assert td["american"] == no.consensus_price([-120, -115, -125]) and td["price_match"] == "normalised"
    assert td["implied"] == pytest.approx(1 / td["decimal"])            # the margin left in


def test_the_pick_is_the_backtests_pick_and_an_away_pick_rides_its_own_side():
    game = _game(home_prob=0.3)                                          # LV at 0.7
    t = _by_kind(_tickets(game))
    assert t["small"]["pick"] == "LV" == sgp2.pick_of("KC", "LV", 0.3)[0]
    assert _keys(t["small"]) == ["TD:lv1", "WIN:LV"]                   # LV's own scorer first
    # LV's only rookie at the floor, with LV's win, is SMALL's ticket.
    assert not t["rookie"]["built"] and t["rookie"]["reason"] == sgp2.ROOKIE_IS_SMALL
    assert all(leg["team"] == "LV" for x in t.values() for leg in x["legs"] if leg["kind"] in sgp.RESULT_KINDS)


def test_a_game_below_the_bar_or_without_a_chance_gets_four_unbuilt_tickets_and_one_reason():
    below = _tickets(_game(home_prob=0.62))
    assert [t["reason"] for t in below] == [sgp2.not_confident(sgp2.CONFIDENT)] * len(sgp2.KINDS)
    assert not any(t["built"] or t["confident_game"] for t in below)
    shown = npd.show_tickets({}, below)
    assert shown["tickets"] == below and shown["tickets_reason"] == sgp2.not_confident(sgp2.CONFIDENT)
    none = _tickets(_game(home_prob=None))
    assert [t["reason"] for t in none] == [sgp2.NO_PICK] * len(sgp2.KINDS)
    # A confident game whose tickets are built carries no single reason.
    assert npd.show_tickets({}, _tickets())["tickets_reason"] is None


def test_live_reads_what_the_v2_backtest_reads():
    """The same numbers through the backtest's reader (a schedule row, its
    candidates frame, a price per game and player - what nfl_sgp2_backtest
    replays) and its pick give the same tickets: one rule, live and replayed."""
    for home_prob in (0.7, 0.3, 0.62):
        game = _game(home_prob=home_prob)
        prices = _td_prices(game)
        live = _tickets(game)
        row = SimpleNamespace(game_id=game["game_id"], home="KC", away="LV", spread_line=MARKET["spread"],
                              home_spread_odds=MARKET["spread_price_home"],
                              away_spread_odds=MARKET["spread_price_away"], total_line=MARKET["total"],
                              over_odds=MARKET["total_price_over"], under_odds=MARKET["total_price_under"],
                              home_ml=MARKET["home_ml"], away_ml=MARKET["away_ml"])
        candidates = pd.DataFrame([{"gsis_id": g, "name": n, "position": "WR", "team": team,
                                    "prob": np.nan if p is None else p,
                                    "rookie": sgp.is_rookie(FIRST, g, game["season"])}
                                   for team, rows in (("KC", HOME), ("LV", AWAY)) for g, n, p in rows])
        decimals = {(game["game_id"], g): sgp.decimal_from_american(p["price"])
                    for g, p in prices.items() if p["price"] is not None}
        replay = sgp2.build_tickets(bk.game_legs(row, home_prob, candidates, decimals),
                                    *sgp2.pick_of("KC", "LV", home_prob))
        assert [(t["kind"], _keys(t), t["chance"], t["est_payout"], t["reason"]) for t in live] == \
            [(t["kind"], _keys(t), t["chance"], t["est_payout"], t["reason"]) for t in replay]


# ---------------------------------------------------------------- the log

def test_v2_is_logged_from_now_on_and_v1_tickets_logged_before_stay_as_they_were():
    log = {"schema": 1, "picks": {}}
    t0, t1, t2 = (pd.Timestamp(x) for x in ("2026-10-24 02:00", "2026-10-24 12:00", "2026-10-25 12:00"))
    # Logged by v1 before v2 went live.
    assert npd.log_pick(log, _game(tickets=_tickets(version=sgp.VERSION)), t0)
    v1 = copy.deepcopy(log["picks"]["2026_07_LV_KC"]["tickets"])
    assert {t["version"] for t in v1} == {sgp.VERSION}
    # The next run before kickoff logs v2's, and keeps v1's untouched.
    assert npd.log_pick(log, _game(tickets=_tickets()), t1)
    entry = log["picks"]["2026_07_LV_KC"]
    assert entry["tickets"][:len(v1)] == v1
    assert [t["version"] for t in entry["tickets"][len(v1):]] == [sgp2.VERSION] * len(sgp2.KINDS)
    # Re-logged before kickoff: v2's newest build counts, v1's still as logged.
    later = _tickets(_game(home=[("kc1", "Tom Runner", 0.40)] + HOME[1:]))
    assert npd.log_pick(log, _game(tickets=later), t2)
    entry = log["picks"]["2026_07_LV_KC"]
    assert entry["tickets"][:len(v1)] == v1 and entry["tickets"][len(v1):] == later
    assert entry["first_logged"] == npd.stamp(t0)
    # Shown before kickoff: v2's as the game's tickets, v1's beside them.
    shown = npd.show_tickets({}, entry["tickets"])
    assert shown["tickets"] == later and shown["tickets_v1"] == v1


def test_a_logged_ticket_is_never_rebuilt_after_kickoff_of_either_version():
    log = {"schema": 1, "picks": {}}
    before, kicked = pd.Timestamp("2026-10-24 12:00"), pd.Timestamp("2026-10-25 20:25")
    assert npd.log_pick(log, _game(tickets=_tickets(version=sgp.VERSION)), before)
    assert npd.log_pick(log, _game(tickets=_tickets()), before)
    logged = copy.deepcopy(log["picks"]["2026_07_LV_KC"]["tickets"])
    rebuilt = _tickets(_game(home=[("kc1", "Tom Runner", 0.10)] + HOME[1:]))
    assert _keys(rebuilt[0]) != _keys(_by_kind(logged[len(sgp.KINDS):])["small"])
    assert not npd.log_pick(log, _game(tickets=rebuilt), kicked)
    assert log["picks"]["2026_07_LV_KC"]["tickets"] == logged
    # A game under way shows the logged ticket objects themselves.
    entry = log["picks"]["2026_07_LV_KC"]
    shown = npd.show_logged({"home": "KC", "away": "LV"}, entry, {})
    assert all(any(s is t for t in entry["tickets"]) for s in shown["tickets"] + shown["tickets_v1"])
    assert [t["version"] for t in shown["tickets"]] == [sgp2.VERSION] * len(sgp2.KINDS)
    assert [t["version"] for t in shown["tickets_v1"]] == [sgp.VERSION] * len(sgp.KINDS)
    # Logged only by v1: shown as v1's, and the game says it has no v2 tickets.
    only_v1 = {"tickets": logged[:len(sgp.KINDS)]}
    shown = npd.show_logged({"home": "KC", "away": "LV"}, only_v1, {})
    assert shown["tickets"] is None and shown["tickets_reason"] == npd.NO_LIVE_TICKETS_LOGGED
    assert shown["tickets_v1"] == only_v1["tickets"]


# ---------------------------------------------------------------- grading

def _box(rows):
    return pd.DataFrame([{"game_id": "2026_07_LV_KC", "player_id": g, "rushing_tds": r, "receiving_tds": c}
                         for g, r, c in rows])


def _snaps(names):
    return pd.DataFrame({"game_id": "2026_07_LV_KC", "player": names,
                         "offense_snaps": [40.0] * len(names), "defense_snaps": [0.0] * len(names)})


def _both_logged(home_score, away_score):
    p = _game(tickets=_tickets(version=sgp.VERSION) + _tickets())
    p["final"] = {"home": home_score, "away": away_score}
    return {"schema": 1, "picks": {p["game_id"]: p}}


def test_v1_and_v2_tickets_of_one_game_are_each_graded_by_their_own_rule(monkeypatch):
    log = _both_logged(27, 20)                    # KC by 7: covers 6.5; 47 under 47.5
    box = _box([("kc1", 1, 0), ("kc2", 0, 0), ("lv1", 0, 1)])
    graded_by = []
    for version, rule in list(npd.TICKET_RULES.items()):
        def spy(t, result, rule=rule, version=version):
            graded_by.append((version, t["version"]))
            return rule.grade_ticket(t, result)
        monkeypatch.setitem(npd.TICKET_RULES, version, SimpleNamespace(grade_ticket=spy))
    assert npd.grade_tickets(log, box, _snaps(["Tom Runner", "Sam Tight", "Max Wide"])) == 1
    assert graded_by and all(rule == ticket for rule, ticket in graded_by)
    assert {v for v, _ in graded_by} == {sgp.VERSION, sgp2.VERSION}
    p = log["picks"]["2026_07_LV_KC"]
    v1 = _by_kind([t for t in p["tickets"] if t["version"] == sgp.VERSION])
    v2 = _by_kind([t for t in p["tickets"] if t["version"] == sgp2.VERSION])
    assert v1["rookie"]["result"] == sgp.LOST                          # v1: LV's rookie with LV's win
    assert v2["rookie"]["result"] == sgp.LOST                          # v2: KC's rookie, kc2, did not score
    assert v2["small"]["result"] == sgp.WON and v2["medium"]["result"] == sgp.WON
    assert v2["large"]["leg_results"] == {"TD:kc2": sgp.LOST, "TD:lv1": sgp.WON,
                                          "SPREAD:KC": sgp.WON, "TOTAL:UNDER": sgp.WON}
    assert v2["small"]["payout"] == pytest.approx(v2["small"]["est_payout"])
    assert p["tickets_graded"] is True and npd.grade_tickets(log, box, _snaps(["Tom Runner"])) == 0


def test_a_ticket_of_a_version_no_rule_is_listed_for_is_left_ungraded():
    log = _both_logged(27, 20)
    stray = dict(_tickets()[0], version="sgp-v0")
    log["picks"]["2026_07_LV_KC"]["tickets"].append(stray)
    npd.grade_tickets(log, _box([("kc1", 1, 0), ("kc2", 0, 0), ("lv1", 0, 1)]),
                      _snaps(["Tom Runner", "Sam Tight", "Max Wide"]))
    assert "result" not in stray
    assert all(t.get("result") for t in log["picks"]["2026_07_LV_KC"]["tickets"] if t is not stray)


# ---------------------------------------------------------------- the TD lists' prices

def test_every_td_candidate_carries_his_matched_price_or_the_missing_reason():
    game = _game()
    npd.attach_td_prices(game, _td_prices(game))
    for side in npd.TD_LISTS:
        assert game[side] and all(set(p["price"]) >= {"american", "decimal", "implied", "best_american",
                                                      "best_book", "books", "as_of", "missing", "est"}
                                  for p in game[side])
    by = {p["gsis_id"]: p["price"] for p in game["home_td_all"] + game["away_td_all"]}
    assert by["kc1"]["american"] == no.consensus_price([-120, -115, -125]) and by["kc1"]["match"] == "normalised"
    assert by["kc1"]["decimal"] == pytest.approx(sgp.decimal_from_american(by["kc1"]["american"]))
    assert by["kc1"]["implied"] == pytest.approx(1 / by["kc1"]["decimal"])
    assert by["kc1"]["best_american"] == -115 and by["kc1"]["best_book"] == "FanDuel" and by["kc1"]["books"] == 3
    assert by["kc1"]["as_of"] == "2026-10-24T12:00:00Z"
    assert by["kc2"]["match"] == "exact" and by["lv1"]["match"] == "exact"
    # Chris Hands (KC, unrated) and Chris Hands Jr. (LV) share a name: neither gets the feed's price.
    assert by["kc4"]["american"] is None and by["kc4"]["missing"] == no.TD_PRICE_MISSING["ambiguous"]
    assert by["lv3"]["missing"] == no.TD_PRICE_MISSING["ambiguous"]
    assert by["kc3"]["american"] is None and by["kc3"]["missing"] == no.TD_PRICE_MISSING["unmatched"]
    # The top three carry the same price as the whole list.
    assert [p["price"] for p in game["home_td"]] == [by[p["gsis_id"]] for p in game["home_td"]]
    # And a ticket's TD leg reads that same price.
    leg = _by_kind(_tickets(game))["small"]["legs"][0]
    assert (leg["american"], leg["decimal"]) == (by["kc1"]["american"], pytest.approx(by["kc1"]["decimal"]))


def test_with_nothing_bought_every_candidate_says_why():
    game = _game()
    far = no.td_prices_for(no.empty(), "LV", "KC", KICKOFF, npd._td_names(game, {}),
                           now=datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))
    npd.attach_td_prices(game, far)
    assert all(p["price"]["missing"] == no.TD_PRICE_MISSING["window"] and p["price"]["american"] is None
               for side in npd.TD_LISTS for p in game[side])


def test_a_logged_entry_keeps_the_price_it_had_before_kickoff():
    log = {"schema": 1, "picks": {}}
    game = _game()
    npd.attach_td_prices(game, _td_prices(game))
    game["tickets"] = _tickets(game)
    assert npd.log_pick(log, game, pd.Timestamp("2026-10-24 12:00"))
    entry = log["picks"]["2026_07_LV_KC"]
    was = copy.deepcopy([p["price"] for p in entry["home_td_all"]])
    # After kickoff the game shows what was logged, whatever is on file now.
    shown = npd.show_logged({"home": "KC", "away": "LV"}, entry, {})
    npd.show_td_prices(shown)
    assert [p["price"] for p in shown["home_td_all"]] == was
    # A pick logged before the lists kept a price says so - and the log is not changed.
    old = copy.deepcopy(entry)
    for side in npd.TD_LISTS:
        for p in old[side]:
            p.pop("price")
    shown = npd.show_td_prices(npd.show_logged({"home": "KC", "away": "LV"}, old, {}))
    assert all(p["price"]["missing"] == npd.NO_TD_PRICE_LOGGED and p["price"]["american"] is None
               for side in npd.TD_LISTS for p in shown[side])
    assert all("price" not in p for side in npd.TD_LISTS for p in old[side])


# ---------------------------------------------------------------- week.json's "sgp"

def _graded_log():
    log = _both_logged(27, 20)
    npd.grade_tickets(log, _box([("kc1", 1, 0), ("kc2", 0, 0), ("lv1", 0, 1)]),
                      _snaps(["Tom Runner", "Sam Tight", "Max Wide"]))
    below = _game(game_id="2026_08_KC_DEN", home_prob=0.6)
    below["tickets"] = _tickets(below)
    log["picks"]["2026_08_KC_DEN"] = below
    log["picks"]["2026_05_OLD"] = {"pick": "KC", "kickoff": "2026-10-01T17:00:00Z"}     # before tickets
    return log


def test_the_block_is_v2s_with_the_chosen_pair_and_the_grid_and_v1_retired_beside_it():
    history, v1_history = nx.load_sgp_history(nx.SGP2_HISTORY), nx.load_sgp_history()
    if history is None or v1_history is None:
        pytest.skip("no history on file yet")
    log = _graded_log()
    block = nx.sgp_block(log, history, v1_history)
    assert block["version"] == sgp2.VERSION and block["rule"] == "nfl/engine/nfl_sgp2.py"
    assert (block["confident"], block["td_floor"], block["chosen_on"]) == (sgp2.CONFIDENT, sgp2.TD_FLOOR,
                                                                          sgp2.CHOSEN_ON)
    assert block["grids"]["confident"] == list(sgp2.CONFIDENT_GRID) and block["grids"]["coverage_min"] == sgp2.COVERAGE_MIN
    h = block["history"]
    assert h["available"] and h["generated"] == history["generated"] and h["kinds"] == history["kinds"]
    assert (h["chosen"]["confident"], h["chosen"]["td_floor"]) == (sgp2.CONFIDENT, sgp2.TD_FLOOR)
    assert len(h["grid"]["rows"]) == len(sgp2.CONFIDENT_GRID) * len(sgp2.TD_FLOOR_GRID)
    assert h["confident_games"] == history["confident_games"] and h["roi_reason"] == sgp.NO_TD_PRICE
    # Live: v2's tickets only, the confident games counted.
    live = block["live"]
    assert live["games"] == 2 and live["graded_games"] == 1 and live["confident_games"] == 1
    assert live["kinds"]["small"]["won"] == 1 and live["kinds"]["small"]["roi"] is not None
    assert live["kinds"]["small"]["not_built_why"] == {sgp2.not_confident(sgp2.CONFIDENT): 1}
    # v1: its own history and live meters, retired.
    v1 = block["v1"]
    assert v1["version"] == sgp.VERSION and v1["retired"] is True and v1["retired_note"] == nx.SGP_V1_RETIRED
    assert v1["history"]["available"] and v1["history"]["kinds"] == v1_history["kinds"]
    assert v1["live"]["games"] == 1 and v1["live"]["kinds"]["rookie"]["lost"] == 1
    # The words, every one from the rules' constants.
    cav = block["caveats"]
    assert cav["edge"] == sgp.NO_EDGE and "market is hard to beat" in cav["edge"]
    assert cav["edge_v2"] == sgp2.NO_EDGE and cav["objective"] == sgp2.OBJECTIVE
    assert cav["rules"] == sgp2.rules() and set(cav["rules"]) >= {"games", "result", "td", "small", "chosen"}
    assert cav["tickets"] == sgp2.ticket_rules() and set(cav["tickets"]) == set(sgp2.KINDS)
    assert f"{sgp2.TD_FLOOR:.0%}" in cav["tickets"]["medium"] and f"{sgp2.CONFIDENT:.0%}" in cav["rules"]["games"]
    for key in ("est", "est_erratum", "bias", "chance", "prices", "exact", "meters", "live", "slots", "rookie",
                "not_in_v1", "roi_priced"):
        assert cav[key] == v1["caveats"][key], key
    assert json.loads(json.dumps(nx._clean(block))) == nx._clean(block)


def test_a_history_of_another_rule_or_pair_says_why_it_is_not_shown():
    log = {"schema": 1, "picks": {}}
    assert "nfl_sgp2_backtest.py" in nx.sgp_block(log, None)["history"]["reason"]
    v1_file = nx.load_sgp_history()
    if v1_file is not None:
        assert sgp.VERSION in nx.sgp_block(log, v1_file)["history"]["reason"]
    other = {"version": sgp2.VERSION, "chosen": {"confident": 0.75, "td_floor": 0.5}}
    why = nx.sgp_block(log, other)["history"]
    assert why["available"] is False and "0.75" in why["reason"] and str(sgp2.CONFIDENT) in why["reason"]
    empty = nx.sgp_block(log, None)
    assert empty["live"]["games"] == 0 and empty["live"]["confident_games"] == 0
    assert empty["v1"]["history"]["available"] is False
