"""Tickets v2 (sameday.py's TICKETS v2): the leg constructor, the likeliest
and underdog kinds over the wider pool, the mixed ticket and the props
ticket - as the docstring fixes them."""

import math
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import sameday as sdy  # noqa: E402

DAY = "2026-11-14"


def W(key, p, dec, sport="Soccer", result=None):
    """A winner leg (v2: bet 1X2)."""
    return sdy.market_leg(sport, key, DAY, f"{key} event", f"{key} pick", p, dec, "1X2", p_market=p,
                          kickoff=f"{DAY}T18:00:00Z", result=result)


def M(key, p, dec, bet="OU25", detail="Over 2.5", sport="Soccer", result=None, est=None, estimated=False):
    return sdy.market_leg(sport, key, DAY, f"{key} event", f"{key} {detail}", p, dec, bet, bet_detail=detail,
                          p_market=p, kickoff=f"{DAY}T18:00:00Z", result=result, est_decimal=est,
                          estimated=estimated,
                          lot_key=sdy.method_lot_key(["A", "B"], "A", detail) if bet == "METHOD" else None)


def P(fight, corner, kind, p, dec, est=None, estimated=False, result=None):
    """A METHOD leg on fight `fight`."""
    return sdy.market_leg("UFC", fight, DAY, f"{fight}", f"{corner} by {kind}", p, dec, "METHOD",
                          bet_detail=kind, p_market=p, kickoff=f"{DAY}T23:00:00Z", result=result,
                          lot_key=sdy.method_lot_key([fight + "a", fight + "b"], corner, kind),
                          est_decimal=est, estimated=estimated)


def test_a_leg_needs_a_price_a_chance_and_a_known_market():
    assert W("a", 0.6, 1.0) is None and W("a", 0.6, float("nan")) is None
    assert W("a", 0.0, 1.8) is None and W("a", 1.0, 1.8) is None
    with pytest.raises(ValueError):
        sdy.market_leg("NFL", "g", DAY, "e", "p", 0.5, 1.9, "BTTS")
    l = M("m1", 0.58, 1.70)
    assert l["market"] == "moneyline" and l["bet"] == "OU25" and l["key"] == "Soccer:m1"


def test_one_leg_per_event_across_markets():
    """A match's winner and its over/under share the event key, so no ticket
    holds both - they move together."""
    legs = [W("m1", 0.70, 1.45), M("m1", 0.62, 1.60), W("m2", 0.66, 1.50), M("m3", 0.60, 1.62)]
    t = sdy.build_mixed(legs)
    keys = [l["key"] for l in t["legs"]]
    assert len(keys) == len(set(keys))


def test_the_likeliest_v2_reads_only_winner_legs():
    legs = [W(f"m{i}", 0.70, 1.45) for i in range(4)] + [M(f"o{i}", 0.80, 1.30) for i in range(4)]
    t = sdy.build_likeliest_v2(legs)
    assert t is not None and all(l["bet"] == "1X2" for l in t["legs"]) and t["profile"] == "same_day_v2"


def test_the_mixed_ticket_always_holds_another_market_and_is_the_likeliest_such():
    winners = [W(f"m{i}", 0.75, 1.36) for i in range(6)]
    ou = M("o1", 0.58, 1.72)
    t = sdy.build_mixed(winners + [ou])
    assert t is not None and any(l["bet"] == "OU25" for l in t["legs"])
    assert 3.0 <= t["multiplier"] <= 5.0 and 1 <= t["n_legs"] <= 6
    assert sdy.build_mixed(winners) is None                     # no other market: no mixed ticket


def test_the_mixed_floors_a_non_method_leg_but_not_a_prop():
    winners = [W(f"m{i}", 0.75, 1.36) for i in range(3)]
    weak_ou = M("o1", 0.45, 2.10)                               # under MIN_P: never used
    assert sdy.build_mixed(winners + [weak_ou]) is None
    prop = P("f1", "A", "Decision", 0.30, 3.10)
    t = sdy.build_mixed([W("m1", 0.75, 1.36), prop])
    assert t is not None and any(l["bet"] == "METHOD" for l in t["legs"])


def test_the_props_ticket_holds_exactly_one_prop_and_never_its_own_fights_winner():
    fill = [W("f1", 0.70, 1.40, sport="UFC"), W("g2", 0.72, 1.38, sport="NFL"), W("m3", 0.68, 1.45)]
    props = [P("f1", "A", "KO/TKO", 0.30, 2.80), P("f1", "A", "Decision", 0.28, 3.10),
             P("f2", "C", "Submission", 0.20, 3.90), P("f2", "D", "KO/TKO", 0.10, 9.0)]
    for salt in sdy.PROPS_SIM_SALTS[:60]:
        t = sdy.build_props(fill, props, salt=salt)
        assert t is not None
        ms = [l for l in t["legs"] if l["bet"] == "METHOD"]
        assert len(ms) == 1 and t["legs"][0]["bet"] == "METHOD"
        assert all(l["key"] != ms[0]["key"] for l in t["legs"][1:])
        assert 3.0 <= t["multiplier"] <= 5.0 and 2 <= t["n_legs"] <= 6
        assert sdy.PROPS_PRICE[0] <= ms[0]["decimal"] <= sdy.PROPS_PRICE[1]   # the 9.0 prop never drawn


def test_the_props_draw_is_weighted_by_chance_and_stable():
    fill = [W("m3", 0.80, 1.45)]
    a, b = P("f1", "A", "KO/TKO", 0.45, 2.30), P("f2", "C", "Decision", 0.30, 2.40)
    first = sum(sdy.build_props(fill, [a, b], salt=s)["method_lot_key"] == a["lot_key"] for s in sdy.PROPS_SIM_SALTS)
    assert abs(first / 1000 - 0.6) < 0.05
    t1 = sdy.build_props(fill, [a, b]); t2 = sdy.build_props(fill, [b, a, P("f9", "Z", "KO/TKO", 0.3, 9.9)])
    assert t1["method_lot_key"] == t2["method_lot_key"] and t1["salt"] == sdy.PROPS_SALT


def test_as_live_chooses_by_the_estimate_and_pays_the_real_price():
    fill = [W("m3", 0.80, 1.50)]
    prop = P("f1", "A", "KO/TKO", 0.30, 3.40, est=2.60)       # estimate 2.60 x 1.50 = 3.90 in the live band
    t = sdy.build_props(fill, [prop], band=sdy.PROPS_LIVE_BAND, price_key="est_decimal")
    assert t is not None and math.isclose(t["searched_multiplier"], 3.9)
    assert math.isclose(t["multiplier"], 3.40 * 1.50)            # paid at the real prop: 5.1, outside the band
    assert sdy.build_props(fill, [prop], band=sdy.PROPS_LIVE_BAND) is None   # by its real price it does not fit


def test_every_market_settles_through_the_same_contract():
    won = [M("o1", 0.6, 1.7, result={"winner": "pick"}), P("f1", "A", "KO/TKO", 0.3, 2.8, result={"winner": "pick"})]
    t = {"legs": won}
    assert sdy.settle(t)["status"] == "won" and math.isclose(sdy.settle(t)["paid"], 1.7 * 2.8)
    push = [M("s1", 0.5, 1.91, bet="SPREAD", detail="BUF -3", sport="NFL", result={"winner": None}),
            W("m1", 0.6, 1.6, result={"winner": "pick"})]
    assert math.isclose(sdy.settle({"legs": push})["paid"], 1.6)
    assert sdy.settle({"legs": [P("f1", "A", "Decision", 0.3, 3.0, result={"winner": "other"})]})["status"] == "lost"


def test_the_underdog_v2_has_its_own_salt():
    fill = [W("m1", 0.8, 1.25)]
    dogs = [sdy.dog("Soccer", f"d{i}", DAY, "e", "dog", 0.33, 3.0, 0.31, lot_key=f"d{i}") for i in range(5)]
    v1 = [sdy.build_dog(fill, dogs)["dog_key"] for _ in range(1)]
    v2 = sdy.build_underdog_v2(fill, dogs)
    assert v2["salt"] == sdy.DOG_SALT_V2 and v2["profile"] == "same_day_dog_v2"
    assert v1[0].startswith("Soccer:d")
