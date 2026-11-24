"""The same-game tickets live (nfl_predict), v1's (sgp-v1, retired since
v2 builds the live tickets - tests/test_nfl_sgp2_live.py): built before
kickoff from what the pick was made with, as the backtest builds them;
logged with the pick and never rebuilt after kickoff; graded from the score
and the box score, a player who took no snap voiding his leg; a TD leg
priced only from its own game's matched price, else missing with the
reason; and each kind metered beside the backtest, with the rule's own
words. v1's tickets already logged are still graded, so all of it still
holds for them; they are built here with version=sgp.VERSION."""

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
# gsis, name, chance (None: a usage group not measured as calibrated)
HOME = [("kc1", "Tom Runner", 0.48), ("kc2", "Sam Tight", 0.36), ("kc3", "Nicholas Name", 0.22),
        ("kc4", "Chris Hands", None)]
AWAY = [("lv1", "Max Wide", 0.33), ("lv2", "DJ Catcher", 0.27), ("lv3", "Chris Hands Jr.", 0.12)]
FIRST = {"kc1": 2020, "kc2": 2019, "kc3": 2023, "kc4": 2021, "lv1": 2026, "lv2": 2022, "lv3": 2024}


def _list(rows):
    return [{"gsis_id": g, "name": n, "position": "WR", "prob": p, "tds_season": 0, "games_season": 0,
             "status": "active", "rank": i + 1, "group": "3+" if p is not None else "under 1"}
            for i, (g, n, p) in enumerate(rows)]


def _game(home=HOME, away=AWAY, **kw):
    g = {"game_id": "2026_07_LV_KC", "season": 2026, "week": 7, "game_type": "REG",
         "kickoff": "2026-10-25T20:25:00Z", "home": "KC", "away": "LV", "pick": "KC", "pick_prob": 0.7,
         "home_prob": 0.7, "market": dict(MARKET), "home_td_all": _list(home), "away_td_all": _list(away)}
    g.update(kw)
    return g


def _td_prices(game, aliases=None):
    cache = no.empty()
    key, entry = no.parse_td_event(json.loads((FIX / "td_event.json").read_text()))
    no.merge_td(cache, key, entry, now=CLOCK)
    return no.td_prices_for(cache, "LV", "KC", KICKOFF, npd._td_names(game, aliases or {}), now=CLOCK)


def _tickets(game=None):
    game = game or _game()
    return npd.game_tickets(game, game["market"], _td_prices(game), FIRST, version=sgp.VERSION)


def _keys(t):
    return [leg["key"] for leg in t["legs"]]


def _by_kind(tickets):
    return {t["kind"]: t for t in tickets}


# ---------------------------------------------------------------- the build

def test_a_game_gets_its_four_tickets_from_what_the_pick_was_made_with():
    tickets = _tickets()
    assert [t["kind"] for t in tickets] == list(sgp.KINDS)
    t = _by_kind(tickets)
    assert _keys(t["small"]) == ["TD:kc1", "WIN:KC"]
    assert _keys(t["rookie"]) == ["TD:lv1", "WIN:LV"]            # lv1's first season is this one
    assert _keys(t["medium"]) == ["TD:kc1", "WIN:KC", "TOTAL:UNDER"]
    assert _keys(t["large"]) == ["TD:kc2", "TD:lv1", "SPREAD:KC", "TOTAL:UNDER"]
    win = t["small"]["legs"][1]
    assert win["chance"] == pytest.approx(0.7)                     # the blend's, not the price's
    assert win["american"] == -250.0 and win["decimal"] == pytest.approx(1.4)
    assert win["price_source"] == "the-odds-api" and win["price_missing"] is None
    assert all(x["version"] == sgp.VERSION and x["est"] is True for x in tickets)
    assert all(leg["est"] is True for x in tickets for leg in x["legs"])
    for x in tickets:
        assert x["chance"] == pytest.approx(np.prod([leg["chance"] for leg in x["legs"]]))
        assert x["est_payout"] == pytest.approx(np.prod([leg["decimal"] for leg in x["legs"]]))


def test_a_td_leg_carries_its_matched_price_and_where_it_came_from():
    td = _by_kind(_tickets())["small"]["legs"][0]
    assert td["gsis_id"] == "kc1" and td["price_match"] == "normalised"     # the feed's "Tom Runner Jr."
    assert td["american"] == no.consensus_price([-120, -115, -125])
    assert td["decimal"] == pytest.approx(sgp.decimal_from_american(td["american"]))
    assert td["best_american"] == -115 and td["best_book"] == "FanDuel" and td["books"] == 3
    assert td["price_as_of"] == "2026-10-24T12:00:00Z" and td["price_source"] == "the-odds-api"


def test_an_unmatched_td_price_stays_missing_with_its_reason():
    game = _game(home=[("kc1", "Thomas Runner", 0.48)] + HOME[1:])
    small = _by_kind(_tickets(game))["small"]
    td = small["legs"][0]
    assert td["gsis_id"] == "kc1" and td["decimal"] is None and td["american"] is None
    assert td["price_missing"] == no.TD_PRICE_MISSING["unmatched"] and td["price_source"] is None
    # The ticket keeps its chance; its est. payout is missing, and says why.
    assert small["chance"] is not None and small["est_payout"] is None
    assert small["est_payout_reason"] == f"no price on file for {td['label']}"


def test_with_no_td_price_bought_yet_every_td_leg_says_why():
    game = _game()
    far = no.td_prices_for(no.empty(), "LV", "KC", KICKOFF, npd._td_names(game, {}),
                           now=datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc))   # before TD_EARLY_DAYS
    tickets = npd.game_tickets(game, game["market"], far, FIRST, version=sgp.VERSION)
    tds = [leg for t in tickets for leg in t["legs"] if leg["kind"] == "TD"]
    assert tds and all(leg["price_missing"] == no.TD_PRICE_MISSING["window"] for leg in tds)
    assert all(t["est_payout"] is None for t in tickets)


def test_a_roster_name_finds_a_price_the_td_list_name_misses():
    game = _game(home=[("kc1", "Thomas Runner", 0.48)] + HOME[1:])
    prices = _td_prices(game, aliases={"kc1": {"Tom Runner Jr."}})
    assert prices["kc1"]["match"] == "exact" and prices["kc1"]["price"] is not None


def test_live_reads_what_the_backtest_read():
    """The same numbers through the backtest's own reader (a schedule row,
    its candidates frame, a price per game and player) give the same
    tickets: the live path and the replay are one rule."""
    game = _game()
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
    replay = sgp.build_tickets(bk.game_legs(row, game["home_prob"], candidates, decimals))
    assert [(t["kind"], _keys(t), t["chance"], t["est_payout"]) for t in live] == \
        [(t["kind"], _keys(t), t["chance"], t["est_payout"]) for t in replay]


def test_no_market_means_no_spread_or_total_legs_and_an_unpriced_win():
    game = _game(market=None)
    t = _by_kind(npd.game_tickets(game, None, _td_prices(game), FIRST, version=sgp.VERSION))
    assert _keys(t["small"]) == ["TD:kc1", "WIN:KC"]
    assert t["small"]["legs"][1]["price_missing"] == npd.NO_PRICE
    assert not t["large"]["built"] and t["large"]["reason"] == sgp.NO_TOTAL


# ---------------------------------------------------------------- the log

def test_a_ticket_logged_before_kickoff_is_never_rebuilt_after_it():
    log = {"schema": 1, "picks": {}}
    before, after = pd.Timestamp("2026-10-24 12:00"), pd.Timestamp("2026-10-25 20:25")
    game = _game(tickets=_tickets())
    assert npd.log_pick(log, game, before)
    logged = copy.deepcopy(log["picks"][game["game_id"]]["tickets"])
    # After kickoff: a different build is refused, the logged one stands.
    rebuilt = _tickets(_game(home=[("kc1", "Tom Runner", 0.10)] + HOME[1:]))
    assert _keys(rebuilt[0]) != _keys(logged[0])
    assert not npd.log_pick(log, _game(tickets=rebuilt), after)
    assert log["picks"][game["game_id"]]["tickets"] == logged
    # A game under way or over shows the logged tickets themselves - v1's
    # beside the live version's, retired, and why it has none of those.
    shown = npd.show_logged({"home": "KC", "away": "LV"}, log["picks"][game["game_id"]], {})
    assert len(shown["tickets_v1"]) == len(logged)
    assert all(a is b for a, b in zip(shown["tickets_v1"], log["picks"][game["game_id"]]["tickets"]))
    assert shown["tickets"] is None and shown["tickets_reason"] == npd.NO_LIVE_TICKETS_LOGGED
    # And with nothing logged, nothing is built: no tickets, and why.
    none = npd.show_logged({"home": "KC", "away": "LV"}, None, {})
    assert none["tickets"] is None and none["tickets_reason"] == npd.NO_TICKETS_LOGGED
    old = npd.show_logged({"home": "KC", "away": "LV"}, {"pick": "KC"}, {})
    assert old["tickets"] is None and old["tickets_reason"] == npd.NO_TICKETS_LOGGED


# ---------------------------------------------------------------- grading

def _logged(home_score, away_score, tickets=None):
    p = _game(tickets=tickets or _tickets())
    p["final"] = {"home": home_score, "away": away_score}
    return {"schema": 1, "picks": {p["game_id"]: p}}


def _box(rows):
    return pd.DataFrame([{"game_id": "2026_07_LV_KC", "player_id": g, "rushing_tds": r, "receiving_tds": c}
                         for g, r, c in rows])


def _snaps(names):
    return pd.DataFrame({"game_id": "2026_07_LV_KC", "player": names,
                         "offense_snaps": [40.0] * len(names), "defense_snaps": [0.0] * len(names)})


def test_grading_waits_for_the_box_score_and_the_snap_counts():
    log = _logged(27, 20)
    assert npd.grade_tickets(log, _box([])) == 0                       # no box score yet
    assert npd.grade_tickets(log, _box([("kc1", 1, 0)])) == 0          # lv1 and kc2: no line, no snaps
    assert "result" not in log["picks"]["2026_07_LV_KC"]["tickets"][0]


def test_tickets_are_graded_from_the_score_and_the_box_score_once():
    log = _logged(27, 20)                    # KC by 7: covers 6.5; 47 under 47.5
    box = _box([("kc1", 1, 0), ("kc2", 0, 0), ("lv1", 0, 1)])
    assert npd.grade_tickets(log, box, _snaps(["Tom Runner", "Sam Tight", "Max Wide"])) == 1
    p = log["picks"]["2026_07_LV_KC"]
    t = _by_kind(p["tickets"])
    assert p["tickets_graded"] is True
    assert t["small"]["result"] == sgp.WON and t["medium"]["result"] == sgp.WON
    assert t["rookie"]["result"] == sgp.LOST                           # Max Wide scored, LV lost
    assert t["large"]["leg_results"] == {"TD:kc2": sgp.LOST, "TD:lv1": sgp.WON,
                                         "SPREAD:KC": sgp.WON, "TOTAL:UNDER": sgp.WON}
    assert t["small"]["payout"] == pytest.approx(t["small"]["est_payout"])
    assert npd.grade_tickets(log, box, _snaps(["Tom Runner"])) == 0     # graded once


def test_a_tie_a_push_and_a_player_who_did_not_play_void_their_legs():
    log = _logged(24, 24)                    # a tie: no winner; 48 over 47.5
    # Tom Runner took no snap: his legs void; the others stand.
    box = _box([("kc2", 1, 0), ("lv1", 0, 0)])
    assert npd.grade_tickets(log, box, _snaps(["Sam Tight", "Max Wide", "Tom Runner"]).assign(
        offense_snaps=[40.0, 40.0, 0.0])) == 1
    t = _by_kind(log["picks"]["2026_07_LV_KC"]["tickets"])
    assert t["small"]["leg_results"] == {"TD:kc1": sgp.VOID, "WIN:KC": sgp.VOID}
    assert t["small"]["result"] == sgp.VOID and t["small"]["payout"] == 1.0
    assert t["medium"]["leg_results"]["TOTAL:UNDER"] == sgp.LOST and t["medium"]["result"] == sgp.LOST
    # The spread: KC -6.5 in a tie lost; Sam Tight scored, Max Wide did not.
    assert t["large"]["result"] == sgp.LOST
    # Whole-number lines: KC by 7 on -7 and 47 points on 47 both push.
    pushed = _logged(27, 20, tickets=_tickets(_game(market=dict(MARKET, spread=7.0, total=47.0))))
    assert npd.grade_tickets(pushed, _box([("kc1", 0, 0), ("kc2", 1, 0), ("lv1", 1, 0)]),
                             _snaps(["Tom Runner", "Sam Tight", "Max Wide"])) == 1
    large = _by_kind(pushed["picks"]["2026_07_LV_KC"]["tickets"])["large"]
    assert large["leg_results"] == {"TD:kc2": sgp.WON, "TD:lv1": sgp.WON,
                                    "SPREAD:KC": sgp.VOID, "TOTAL:UNDER": sgp.VOID}
    assert large["result"] == sgp.WON
    assert large["payout"] == pytest.approx(np.prod([leg["decimal"] for leg in large["legs"][:2]]))


def test_a_won_ticket_with_a_void_leg_pays_on_the_legs_that_stand():
    log = _logged(30, 10)                    # KC by 20; 40 under
    box = _box([("kc1", 2, 0), ("kc2", 1, 0)])
    # Max Wide (lv1) is in neither the box score nor the snap counts: he did not play.
    assert npd.grade_tickets(log, box, _snaps(["Tom Runner", "Sam Tight"])) == 1
    large = _by_kind(log["picks"]["2026_07_LV_KC"]["tickets"])["large"]
    assert large["leg_results"]["TD:lv1"] == sgp.VOID and large["result"] == sgp.WON
    standing = [leg for leg in large["legs"] if leg["key"] != "TD:lv1"]
    assert large["payout"] == pytest.approx(np.prod([leg["decimal"] for leg in standing]))
    assert large["chance_graded"] == pytest.approx(np.prod([leg["chance"] for leg in standing]))


# ---------------------------------------------------------------- the meters

def test_the_meters_read_the_backtest_and_the_log_and_say_the_rules_words():
    log = _logged(27, 20)
    npd.grade_tickets(log, _box([("kc1", 1, 0), ("kc2", 0, 0), ("lv1", 0, 1)]),
                      _snaps(["Tom Runner", "Sam Tight", "Max Wide"]))
    upcoming = _game(game_id="2026_08_KC_DEN", tickets=_tickets())
    log["picks"]["2026_08_KC_DEN"] = upcoming
    log["picks"]["2026_05_OLD"] = {"pick": "KC", "kickoff": "2026-10-01T17:00:00Z"}    # before tickets
    history = nx.load_sgp_history()
    if history is None:
        pytest.skip("no history on file yet")
    block = nx.sgp_v1_block(log, history)
    assert block["version"] == sgp.VERSION and block["kinds"] == list(sgp.KINDS)
    assert block["retired"] is True and block["retired_note"] == nx.SGP_V1_RETIRED
    assert nx.sgp_block(log, None, history)["v1"] == block        # under the live block's "v1"
    assert block["live"]["games"] == 2 and block["live"]["graded_games"] == 1
    live = block["live"]["kinds"]
    assert live["small"]["won"] == 1 and live["small"]["pending"] == 1 and live["rookie"]["lost"] == 1
    assert live["small"]["roi"] is not None                               # every graded leg priced
    assert block["history"]["available"] and block["history"]["generated"] == history["generated"]
    assert block["history"]["kinds"] == history["kinds"] and block["history"]["through"] == history["through"]
    cav = block["caveats"]
    assert cav["edge"] == sgp.NO_EDGE and cav["bias"] == sgp.KNOWN_BIAS and cav["est"] == sgp.EST_PAYOUT
    assert cav["exact"] == sgp.EXACT_PAYOUT and cav["live"] == sgp.LIVE_ROI
    assert cav["est_erratum"] == sgp.EST_ERRATUM          # the correction beside v1's est. words
    assert set(cav["tickets"]) == set(sgp.KINDS) and set(cav["legs"]) == set(sgp.LEG_KINDS)
    assert f"{sgp.ROOKIE_FLOOR:.0%}" in cav["tickets"]["rookie"]


def test_without_a_backtest_of_this_rule_the_history_meter_says_why():
    log = {"schema": 1, "picks": {}}
    none = nx.sgp_v1_block(log, None)["history"]
    assert none["available"] is False and "no same-game ticket backtest" in none["reason"]
    other = nx.sgp_v1_block(log, {"version": "sgp-v0"})["history"]
    assert other["available"] is False and "sgp-v0" in other["reason"]
    live = nx.sgp_v1_block(log, None)["live"]
    assert live["games"] == 0 and live["kinds"]["small"]["roi_reason"] == "no ticket graded yet"


def test_the_block_is_json_safe(tmp_path):
    log = _logged(27, 20)
    path = nx.write("w", {"sgp": nx.sgp_block(log, nx.load_sgp_history(nx.SGP2_HISTORY), nx.load_sgp_history())},
                    tmp_path)
    assert "NaN" not in path.read_text() and json.loads(path.read_text())["sgp"]["v1"]["version"] == sgp.VERSION
