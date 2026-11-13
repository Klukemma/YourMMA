"""The typical book's price pair the card carries beside the best one.

The card's odds/odds_other are each side's best price across books, often
from two books; a parlay is bought at one. The same-day underdog ticket
(parlays/engine/build_sameday.py) prices its dog at the median pair instead,
because best-of-books flatters an underdog most. These lift median_price and
with_median_prices out of predict_card rather than import it: importing that
module runs the whole pipeline (as test_odds_matching does).
"""

import re
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
SOURCE = (ENGINE / "predict_card.py").read_text()


def _load():
    start = SOURCE.index("def median_price(")
    end = SOURCE.index("def analyze_fight_value(")
    namespace = {}
    exec(SOURCE[start:end], namespace)
    return namespace["median_price"], namespace["with_median_prices"]


median_price, with_median_prices = _load()


def _side(*prices):
    """A matcher record as match_fighter_to_odds returns it: one offer a book."""
    return {"name": "X", "best_odds": max(prices),
            "odds": [{"book": f"book{i}", "odds": p} for i, p in enumerate(prices)]}


def test_the_median_is_taken_over_decimal_prices_never_american():
    # -105 and +105 straddle even money: an American median would be 0
    m = median_price(_side(-105, 105))
    assert m["decimal"] == pytest.approx((1 + 100 / 105 + 2.05) / 2)
    assert m["american"] == 100 and m["books"] == 2
    odd = median_price(_side(150, 165, 140))
    assert odd == {"american": 150, "decimal": pytest.approx(2.5), "books": 3}


def test_a_typical_book_pays_less_than_the_best_on_an_underdog():
    dog = _side(250, 260, 240, 300)              # one book far out on the dog
    m = median_price(dog)
    assert m["decimal"] == pytest.approx((3.5 + 3.6) / 2) and m["american"] == 255
    assert m["decimal"] < 1 + dog["best_odds"] / 100


def test_nothing_usable_is_none_not_zero():
    assert median_price(None) is None
    assert median_price({"odds": []}) is None
    assert median_price({"odds": [{"book": "a", "odds": None}, {"book": "b", "odds": 0},
                                  {"book": "c", "odds": float("nan")}]}) is None
    assert median_price({"odds": [{"book": "a", "odds": None}, {"book": "b", "odds": -200}]}) == \
        {"american": -200, "decimal": pytest.approx(1.5), "books": 1}


def test_the_card_fields_need_both_sides_and_count_the_thinner_one():
    f = with_median_prices(median_price(_side(-200, -190, -210)), median_price(_side(170, 160)))
    assert f["odds_median"] == -200 and f["odds_other_median"] == 165
    assert f["decimal_median"] == pytest.approx(1.5) and f["decimal_other_median"] == pytest.approx(2.65)
    assert f["books_median"] == 2
    half = with_median_prices(median_price(_side(-200)), None)
    assert set(half) == set(f) and all(v is None for v in half.values())


def test_a_cached_card_has_one_price_a_side_so_its_median_is_the_best():
    """odds_cache serves each card bout as one offer a side (the best it kept),
    so books_median reads 1: the same-day builder must not call that a
    typical book."""
    cached = {"name": "X", "best_odds": 185, "odds": [{"book": "DraftKings", "odds": 185}]}
    m = median_price(cached)
    assert m["american"] == 185 and m["books"] == 1


def test_the_card_gets_them_after_card_payload_and_nothing_else_moves():
    """Written onto the card's fights after card_payload (which emits a fixed
    field list), one fight per _app_fights entry; the best prices the pick,
    the edge and the parlays read are untouched."""
    assert "_card_fight.update(with_median_prices(_med, _med_other))" in SOURCE
    assert SOURCE.index("_app_card = _app_export.card_payload(") < SOURCE.index("_card_fight.update(")
    assert SOURCE.index("_card_fight.update(") < SOURCE.index("_app_export.write_all(APP_DATA_DIR, {'card': _app_card})")
    assert "'odds': _vi.get('best_odds')," in SOURCE and "'odds_other': _other_vi.get('best_odds')," in SOURCE
    # the medians come from the same matcher records as the best prices
    assert re.search(r"median_price\(match_fighter_to_odds\(_pred\['winner'\], CURRENT_ODDS\)", SOURCE)
    assert re.search(r"median_price\(match_fighter_to_odds\(_other_name, CURRENT_ODDS\)", SOURCE)
