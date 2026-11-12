"""The underdog ticket's rule (sameday.py, v1): the candidates, the lottery,
the fill, and the controls - as the docstring fixes them."""

import math
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import sameday as sdy  # noqa: E402

DAY = "2026-11-14"


def F(key, p, dec, p_market=None, sport="NFL"):
    """A favourite leg (a shipped pick)."""
    return sdy.leg(sport, key, DAY, f"{key} event", f"{key} pick", p, dec,
                   p_market=p if p_market is None else p_market, kickoff=f"{DAY}T18:00:00Z")


def D(key, p, dec, p_market=None, sport="NFL"):
    """An underdog candidate on event `key`."""
    q = p if p_market is None else p_market
    return sdy.dog(sport, key, DAY, f"{key} event", f"{key} dog", p, dec, q, lot_key=key,
                   kickoff=f"{DAY}T18:00:00Z")


def test_the_lottery_number_is_fixed_by_salt_day_and_event_and_never_0_or_1():
    u = sdy.lottery_u(sdy.DOG_SALT, DAY, "2026_10_WAS_NYG")
    assert 0 < u < 1 and u == sdy.lottery_u(sdy.DOG_SALT, DAY, "2026_10_WAS_NYG")
    assert u != sdy.lottery_u(sdy.DOG_SALT, "2026-11-15", "2026_10_WAS_NYG")
    assert u != sdy.lottery_u("sameday-dog-sim0", DAY, "2026_10_WAS_NYG")
    assert len(set(sdy.DOG_SIM_SALTS)) == 1000 and sdy.DOG_SALT not in sdy.DOG_SIM_SALTS


def test_the_ufc_lot_key_ignores_corner_order_case_and_accents():
    a = sdy.dog_lot_key("UFC", corners=["Ciryl Gané", "Josh Hokit"])
    assert a == sdy.dog_lot_key("UFC", corners=["josh  HOKIT", "Ciryl Gane"]) == "ciryl gane|josh hokit"
    assert sdy.dog_lot_key("NFL", game_id="2026_10_WAS_NYG") == "2026_10_WAS_NYG"
    assert sdy.dog_lot_key("Soccer", match_id="D1-20261106-hamburg-mgladbach") == "D1-20261106-hamburg-mgladbach"


def test_a_candidate_is_the_markets_underdog_priced_two_to_four():
    assert D("a", 0.30, 2.00) is not None and D("a", 0.26, 4.00) is not None
    assert D("a", 0.30, 1.99) is None and D("a", 0.24, 4.01) is None
    assert D("a", 0.55, 2.10, p_market=0.50) is None          # level by the market: no underdog
    assert D("a", 0.55, 2.10, p_market=0.47)["p"] == 0.55       # the blend may favour it: still the market's dog


def test_two_candidates_keep_their_order_whatever_else_is_on_the_day():
    a, b = D("a", 0.35, 2.6), D("b", 0.40, 2.3)
    others = [D(f"x{i}", 0.30, 3.0) for i in range(6)]
    for salt in sdy.DOG_SIM_SALTS[:200]:
        two = [d["key"] for d in sdy.dog_order([a, b], salt)]
        many = [d["key"] for d in sdy.dog_order([a, b, *others], salt) if d["key"] in (a["key"], b["key"])]
        assert two == many


def test_the_first_draw_comes_up_in_proportion_to_the_chance():
    """Over the 1,000 simulation salts a 45% dog is first against a 30% one
    about 45/75 = 60% of the time (Efraimidis-Spirakis)."""
    a, b = D("a", 0.45, 2.1), D("b", 0.30, 3.2)
    first = sum(sdy.dog_order([a, b], s)[0]["key"] == a["key"] for s in sdy.DOG_SIM_SALTS)
    assert abs(first / 1000 - 0.6) < 0.05


def test_the_ticket_is_the_dog_plus_the_likeliest_fill_from_other_events():
    dog = D("g1", 0.38, 2.5)
    same_event = F("g1", 0.62, 1.55)                            # the dog's own event: never in the fill
    legs = [same_event, F("g2", 0.80, 1.25), F("g3", 0.70, 1.40), F("g4", 0.55, 1.80)]
    t = sdy.build_dog(legs, [dog])
    keys = [l["key"] for l in t["legs"]]
    assert keys[0] == dog["key"] and t["legs"][0]["dog"] and "NFL:g1" not in keys[1:]
    assert 3.0 <= t["multiplier"] <= 5.0 and 2 <= t["n_legs"] <= 6
    # the likeliest fill in band: 2.5 x 1.25 = 3.125 (0.80) beats 2.5 x 1.40 (0.70)
    assert keys[1:] == ["NFL:g2"] and t["dog_rank"] == 0 and t["salt"] == sdy.DOG_SALT


def test_a_dog_that_cannot_be_filled_gives_way_to_the_next_and_none_fits_gives_none():
    legs = [F("g2", 0.80, 1.25)]
    lone = D("g9", 0.30, 3.9)                                   # 3.9 x 1.25 = 4.875: fits
    big = D("g8", 0.26, 4.0)                                    # 4.0 x 1.25 = 5.0: fits too
    t = sdy.build_dog(legs, [lone, big])
    assert t is not None and t["dog_key"] in (lone["key"], big["key"])
    assert sdy.build_dog([F("g2", 0.90, 1.05)], [D("g9", 0.40, 2.1)]) is None   # 2.2x: nothing fits
    assert sdy.build_dog(legs, []) is None
    # a first-drawn dog with no fit is skipped for the next one
    nofit = D("g7", 0.45, 2.05)                                  # 2.05 x 1.25 = 2.56: no fill
    for salt in sdy.DOG_SIM_SALTS[:50]:
        t = sdy.build_dog(legs, [nofit, lone], salt=salt)
        assert t["dog_key"] == lone["key"]


def test_the_floor_reads_the_shipped_chance_and_the_market_control_its_own():
    legs = [F("g2", 0.80, 1.25)]
    low = D("g9", 0.19, 3.0, p_market=0.31)                      # the model far below the market
    assert sdy.build_dog(legs, [low]) is None
    t = sdy.build_dog(legs, [low], objective="p_market")
    assert t is not None and t["profile"] == "same_day_dog_market_only"


def test_the_market_only_lottery_reads_no_model():
    legs = [F("g2", 0.80, 1.25, p_market=0.78), F("g3", 0.70, 1.40, p_market=0.69)]
    dogs = [D("g8", 0.40, 2.6, p_market=0.36), D("g9", 0.30, 2.7, p_market=0.35)]
    a = sdy.build_dog(legs, dogs, objective="p_market")
    moved = [dict(l, p=0.51) for l in legs]
    moved_dogs = [dict(d, p=0.21) for d in dogs]
    b = sdy.build_dog(moved, moved_dogs, objective="p_market")
    assert [l["key"] for l in a["legs"]] == [l["key"] for l in b["legs"]]


def test_the_report_only_orders_are_deterministic():
    a, b = D("a", 0.40, 2.3, p_market=0.41), D("b", 0.33, 2.9, p_market=0.30)
    assert [d["key"] for d in sdy.dog_order([a, b], order="likeliest")] == ["NFL:a", "NFL:b"]
    assert [d["key"] for d in sdy.dog_order([a, b], order="gap")] == ["NFL:b", "NFL:a"]
    with pytest.raises(ValueError):
        sdy.dog_order([a], order="value")


def test_two_tickets_that_cannot_both_win_are_flagged():
    fav = F("g1", 0.62, 1.55)
    likeliest = {"legs": [fav, F("g2", 0.80, 1.25)]}
    t = sdy.build_dog([F("g3", 0.80, 1.25)], [D("g1", 0.38, 2.5)])
    assert sdy.opposes(t, likeliest)
    assert not sdy.opposes(sdy.build_dog([F("g3", 0.80, 1.25)], [D("g4", 0.38, 2.5)]), likeliest)
    assert math.isclose(t["multiplier"], 2.5 * 1.25)
