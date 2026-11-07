"""Tests for the parlay profiles and their ten-year harness.

A parlay is where optimism hides: a product of prices is easy to get subtly
wrong, a leg that "won" by the wrong method pays nothing, and a backtest
that lets one future fact into a ticket - a price, a result, a record that
includes the fight itself - makes any profile look good. These pin the
arithmetic, the profile rules as written in parlay_profiles.py, and the
harness's point-in-time joins.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import parlay_profiles as pp  # noqa: E402
from experiments import parlay_legs  # noqa: E402
from experiments import parlay_profiles as harness  # noqa: E402


# ---------------------------------------------------------------- a card

# (key, p_model for red, red decimal, blue decimal): favourites the model
# agrees with, underdogs it rates above the market, and fights it does not.
CARD = [
    ("f1", 0.78, 1.30, 3.70),
    ("f2", 0.72, 1.40, 3.05),
    ("f3", 0.70, 1.45, 2.85),
    ("f4", 0.66, 1.55, 2.55),
    ("f5", 0.30, 2.60, 1.55),
    ("f6", 0.45, 2.75, 1.48),
    ("f7", 0.40, 4.00, 1.28),
    ("f8", 0.60, 1.83, 2.00),
    ("f9", 0.80, 1.25, 4.20),
    ("f10", 0.25, 4.50, 1.22),
]


def fight(key, p_model, dec_red, dec_blue, number=1, result=None, facts=None, **extra):
    f = {"key": key, "event": "UFC Test", "date": "2024-01-06", "number": number,
         "red": f"Red {key}", "blue": f"Blue {key}", "p_model": p_model,
         "dec_red": dec_red, "dec_blue": dec_blue, "result": result,
         "facts": facts or {"red": {"wins": 6, "ko_wins": 4, "sub_wins": 1, "last5": ["W", "W"]},
                            "blue": {"wins": 5, "ko_wins": 1, "sub_wins": 3, "ko_losses": 2,
                                     "sub_losses": 1, "last5": ["L", "W"]}}}
    f.update(extra)
    return f


def card_legs(rows=CARD, p_shift=0.0):
    legs = []
    for i, (k, pm, dr, db) in enumerate(rows, 1):
        legs += pp.fight_legs(fight(k, float(np.clip(pm + p_shift, 0.02, 0.98)), dr, db, number=i))
    return legs


# Near pick'em fights: a card of short favourites cannot stack to 40x inside
# seven legs, so the mixed and market-only tickets at the top band are
# tested on this one.
PICKEM = [(f"p{i}", 0.50 + 0.02 * i, 1.85 + 0.03 * i, 1.98 - 0.02 * i) for i in range(1, 10)]


def legs_for(profile, band, p_shift=0.0):
    top = band is not None and tuple(band) == pp.BANDS["risk_lover"]
    return card_legs(PICKEM if top else CARD, p_shift)


ALL = [("steady", None), ("risker", None), ("risk_lover", None)] + [
    (p, pp.BANDS[b]) for p in ("mixed", "market_only") for b in pp.ORDER]


# ---------------------------------------------------------------- tickets

@pytest.mark.parametrize("profile,band", ALL)
def test_a_ticket_multiplies_its_prices_and_lands_in_its_band(profile, band):
    t = pp.build(legs_for(profile, band), profile, band)
    assert t is not None, "the test card is built so every profile has a ticket"
    lo, hi = band or pp.BANDS[profile]
    assert t["multiplier"] == pytest.approx(np.prod([l["decimal"] for l in t["legs"]]))
    assert lo <= t["multiplier"] <= hi
    assert t["band"] == [lo, hi]
    assert t["p"] == pytest.approx(np.prod([l["p"] for l in t["legs"]]))


def test_a_card_that_cannot_reach_the_band_gives_no_ticket():
    legs = card_legs([("a", 0.8, 1.20, 4.50), ("b", 0.8, 1.25, 4.00)])
    for profile in ("steady", "risker", "risk_lover"):
        assert pp.build(legs, profile) is None
    assert pp.build(legs, "mixed", band=pp.BANDS["risk_lover"]) is None


@pytest.mark.parametrize("profile,band", ALL)
def test_one_leg_per_fight(profile, band):
    t = pp.build(legs_for(profile, band), profile, band)
    keys = [l["key"] for l in t["legs"]]
    assert len(keys) == len(set(keys))


def test_steady_legs_are_all_favourites_the_model_agrees_with():
    t = pp.build(card_legs(), "steady")
    for l in t["legs"]:
        assert l["favourite"] and l["p_market"] >= 0.5
        assert l["p_model"] >= 0.5
        assert l["p"] >= pp.SAFE_MIN_P


def test_risker_has_one_or_two_risky_legs_and_steady_legs_otherwise():
    t = pp.build(card_legs(), "risker")
    risky = [l for l in t["legs"] if pp._risky(l)]
    assert 1 <= len(risky) <= 2
    for l in risky:
        assert l["decimal"] >= pp.RISKY_MIN_DECIMAL and l["edge"] >= pp.RISKY_MIN_EDGE
    assert all(pp._safe(l) for l in t["legs"] if l not in risky)


def test_risk_lover_has_one_or_two_long_underdogs_on_two_or_more_core_favourites():
    t = pp.build(card_legs(), "risk_lover")
    longs = [l for l in t["legs"] if not l["favourite"]]
    core = [l for l in t["legs"] if l["favourite"]]
    assert 1 <= len(longs) <= 2
    for l in longs:
        assert l["decimal"] >= pp.LONG_MIN_DECIMAL and l["p_model"] >= l["p_market"]
    assert len(core) >= 2
    assert all(l["p"] >= pp.CORE_MIN_P for l in core)


@pytest.mark.parametrize("band", pp.ORDER)
def test_mixed_contains_a_leg_the_model_rates_above_the_market(band):
    for shift in (0.0, -0.2, 0.15):
        t = pp.build(legs_for("mixed", pp.BANDS[band], shift), "mixed", band=pp.BANDS[band])
        assert t is not None and any(l["edge"] > 0 for l in t["legs"])


@pytest.mark.parametrize("band", pp.ORDER)
def test_market_only_ignores_the_model(band):
    a, b, c = (pp.build(legs_for("market_only", pp.BANDS[band], shift), "market_only",
                        band=pp.BANDS[band]) for shift in (0.0, -0.2, 0.15))
    assert a is not None
    picks = lambda t: [(l["key"], l["side"]) for l in t["legs"]]  # noqa: E731
    assert picks(a) == picks(b) == picks(c)
    assert a["multiplier"] == b["multiplier"] == c["multiplier"]


def test_mixed_and_market_only_need_a_band():
    with pytest.raises(ValueError):
        pp.build(card_legs(), "mixed")


# ---------------------------------------------------------------- settlement

def _leg(side="red", decimal=2.0, winner="red", method="KO/TKO", market="moneyline"):
    return {"side": side, "decimal": decimal, "market": market, "p": 0.5, "p_market": 0.5,
            "result": {"winner": winner, "method": method}}


def test_a_void_leg_pays_one_and_the_rest_are_paid():
    t = {"legs": [_leg(decimal=2.0), _leg(decimal=3.0, winner=None), _leg(decimal=1.5)]}
    s = pp.settle(t)
    assert s["status"] == "won"
    assert s["paid"] == pytest.approx(3.0)
    assert s["profit"] == pytest.approx(2.0)


def test_an_all_void_ticket_returns_the_stake():
    t = {"legs": [_leg(winner=None), _leg(winner=None)]}
    s = pp.settle(t)
    assert s == {"status": "void", "paid": 1.0, "profit": 0.0}


def test_any_lost_leg_loses_the_ticket_even_beside_a_void():
    t = {"legs": [_leg(), _leg(winner=None), _leg(winner="blue")]}
    assert pp.settle(t)["profit"] == -pp.STAKE


def test_a_method_leg_loses_when_the_pick_wins_another_way():
    assert pp.leg_outcome(_leg(market="KO/TKO", method="Decision")) == "lost"
    assert pp.leg_outcome(_leg(market="Submission", method="KO/TKO")) == "lost"
    assert pp.leg_outcome(_leg(market="KO/TKO", method="KO/TKO")) == "won"
    assert pp.leg_outcome(_leg(market="moneyline", method="Decision")) == "won"


def test_an_unsettled_leg_keeps_the_ticket_open_and_out_of_the_record():
    t = {"legs": [_leg(), dict(_leg(), result=None)], "p": 0.25, "p_market": 0.25,
         "multiplier": 4.0, "n_legs": 2}
    assert pp.settle(t)["status"] == "open"
    assert pp.summarise([t]) == {"tickets": 0}


def test_summarise_counts_flat_units():
    won = {"legs": [_leg(decimal=2.0), _leg(decimal=2.5)], "p": 0.3, "p_market": 0.2,
           "multiplier": 5.0, "n_legs": 2}
    lost = dict(won, legs=[_leg(decimal=2.0), _leg(decimal=2.5, winner="blue")])
    s = pp.summarise([won, lost, lost, lost])
    assert s["tickets"] == 4 and s["wins"] == 1
    assert s["profit"] == pytest.approx(4.0 - 3.0)
    assert s["roi"] == pytest.approx(0.25)
    assert s["roi_low"] <= s["roi"] <= s["roi_high"]


# ---------------------------------------------------------------- prices

def test_devig_rejects_bad_overrounds_and_missing_prices():
    assert pp.devig(2.2, 2.2) is None            # sums below 100%: no book quotes that
    assert pp.devig(1.5, 1.5) is None            # 33% over: a stale or mistyped pair
    assert pp.devig(None, 2.0) is None
    assert pp.devig(float("nan"), 1.5) is None   # a pandas gap is not a price
    a, b = pp.devig(1.91, 1.91)
    assert a == pytest.approx(0.5) and a + b == pytest.approx(1.0)


def test_a_side_with_a_nan_price_gives_no_leg_and_no_nan_chance():
    legs = pp.fight_legs(fight("x", 0.6, float("nan"), 2.1))
    assert legs == []


def test_american_and_decimal_round_trip():
    assert pp.decimal_from_american(150) == pytest.approx(2.5)
    assert pp.decimal_from_american(-200) == pytest.approx(1.5)
    assert pp.decimal_from_american(float("nan")) is None
    assert pp.american_from_decimal(2.5) == 150
    assert pp.american_from_decimal(1.5) == -200


def test_the_blend_is_rebuilt_from_the_market_at_the_shipped_weight():
    import market_blend
    leg = pp.fight_legs(fight("x", 0.70, 1.50, 2.70))[0]
    fair = pp.devig(1.50, 2.70)[0]
    assert leg["p"] == pytest.approx(market_blend.blend(0.70, fair))
    assert leg["edge"] == pytest.approx(0.70 - fair)


# ---------------------------------------------------------------- method legs

METHOD = {"ko_red": 4.0, "ko_blue": 6.0, "sub_red": 9.0, "sub_blue": 5.5,
          "decision_red": 4.5, "decision_blue": 3.2}


def test_a_method_leg_carries_the_route_share_of_the_win_chance():
    f = fight("x", 0.45, 2.75, 1.48, **METHOD)
    dog = next(l for l in pp.fight_legs(f) if l["side"] == "red")
    m = pp._as_method(dog)
    mm = dog["method_market"]
    share = mm["KO/TKO"] / sum(mm.values())
    assert m["market"] == "KO/TKO" and m["decimal"] == 4.0
    assert m["p"] == pytest.approx(dog["p"] * share) and m["p"] < dog["p"]
    assert m["p_market"] == pytest.approx(mm["KO/TKO"])
    six = [x for side in pp._method_market(f).values() for x in side.values()]
    assert sum(six) == pytest.approx(1.0)


def test_no_six_way_market_no_method_leg():
    f = fight("x", 0.45, 2.75, 1.48, **dict(METHOD, decision_blue=None))
    dog = next(l for l in pp.fight_legs(f) if l["side"] == "red")
    assert pp._as_method(dog) is None


def test_the_method_variant_records_the_band_it_was_searched_in():
    rows = [(k, pm, dr, db) for k, pm, dr, db in CARD]
    legs = []
    for i, (k, pm, dr, db) in enumerate(rows, 1):
        legs += pp.fight_legs(fight(k, pm, dr, db, number=i, **METHOD))
    t = pp.build(legs, "risk_lover_method")
    assert t is not None
    assert t["band"] == list(pp.BANDS["risk_lover"])
    assert any(l["market"] != "moneyline" for l in t["legs"])
    assert t["multiplier"] == pytest.approx(np.prod([l["decimal"] for l in t["legs"]]))


# ---------------------------------------------------------------- facts and reasons

HISTORY = pd.DataFrame([
    {"date": "2023-01-01", "r_name": "Ann Alpha", "b_name": "X One", "winner": "Ann Alpha", "method": "KO/TKO Punches"},
    {"date": "2023-06-01", "r_name": "X Two", "b_name": "Ann Alpha", "winner": "X Two", "method": "SUB Rear Naked Choke"},
    {"date": "2023-09-01", "r_name": "Ann-Alpha", "b_name": "X Three", "winner": "Ann Alpha", "method": "U-DEC"},
    {"date": "2023-10-01", "r_name": "Ann Alpha", "b_name": "X Four", "winner": "Ann-Alpha", "method": "Decision - Split"},
    {"date": "2024-01-06", "r_name": "Ann Alpha", "b_name": "X Five", "winner": "Ann Alpha", "method": "KO/TKO"},
    {"date": "2024-05-01", "r_name": "X Six", "b_name": "Ann Alpha", "winner": "X Six", "method": "KO/TKO"},
])


def test_facts_never_count_a_fight_on_or_after_the_date():
    f = pp.facts(HISTORY, "Ann Alpha", "2024-01-06")
    assert f["fights"] == 3                      # the 2023-09 row is another spelling's corner
    assert f["wins"] == 2 and f["losses"] == 1
    assert f["ko_wins"] == 1 and f["sub_losses"] == 1
    assert f["last5"] == ["W", "L", "W"]
    assert pp.facts(HISTORY, "Ann Alpha", "2023-01-01")["fights"] == 0


def test_a_winner_spelled_differently_from_the_corner_is_still_a_win():
    f = pp.facts(HISTORY, "Ann Alpha", "2023-12-31")
    assert f["last5"][-1] == "W"                 # "Ann-Alpha" won the 2023-10 fight


@pytest.mark.parametrize("text,expected", [
    ("U-DEC", "Decision"), ("S-DEC", "Decision"), ("M-DEC", "Decision"),
    ("Decision - Unanimous", "Decision"), ("SUB Rear Naked Choke", "Submission"),
    ("SUB Guillotine Choke", "Submission"), ("Submission", "Submission"),
    ("KO/TKO Punches", "KO/TKO"), ("KO/TKO", "KO/TKO"), ("TKO - Doctor's Stoppage", "KO/TKO"),
    ("Overturned", None), ("DQ", None), ("Could Not Continue", None), (None, None),
])
def test_method_class_reads_the_short_labels(text, expected):
    assert pp.method_class(text) == expected


def test_route_is_the_likelier_finish_or_none():
    assert pp.route({"ko_wins": 3, "sub_wins": 1}) == "KO/TKO"
    assert pp.route({"ko_wins": 1, "sub_wins": 2}) == "Submission"
    assert pp.route({"ko_wins": 0, "sub_wins": 0}) is None


GENDERED = re.compile(r"\b(he|him|his|she|her|hers|himself|herself)\b", re.I)


def test_reasons_are_short_and_gender_neutral():
    legs = card_legs()
    f = fight("m", 0.45, 2.75, 1.48, **METHOD)
    legs += [m for m in (pp._as_method(l) for l in pp.fight_legs(f)) if m]
    legs += pp.fight_legs(fight("n", 0.4, 3.0, 1.4, facts={"red": {}, "blue": {}}))
    assert any(l["market"] != "moneyline" for l in legs)
    for l in legs:
        text = pp.reason(l)
        assert text and not GENDERED.search(text), text
        assert len(text) <= 100, text


# ---------------------------------------------------------------- the harness

def _write_fixture(tmp):
    """A two-card world: one price per matching path, one corrupt close with
    a clean open behind it, one unpriced fight, one draw the walk-forward
    never scored, and a fighter who fights twice (and once after)."""
    cache = pd.DataFrame([
        ("c1", "2024-01-06", "UFC A", "Ann Alpha", "Bea Beta", 1.0, 0.70),
        ("c2", "2024-01-06", "UFC A", "Cal Gamma", "Dee Delta", 0.0, 0.40),
        ("c3", "2024-01-06", "UFC A", "Eve Epsilon", "Fay Zeta", 1.0, 0.55),
        ("c4", "2024-01-06", "UFC A", "Gus Eta", "Hal Theta", 0.0, 0.45),
        ("c5", "2024-01-06", "UFC A", "Ivy Iota", "Jo Kappa", 1.0, 0.60),
        ("c6", "2024-03-02", "UFC B", "Ann Alpha", "Kim Lambda", 1.0, 0.65),
    ], columns=["fight_id", "date", "event", "red", "blue", "y", "p_model"])
    cache["year"] = 2024
    for c in ("red_norm", "blue_norm"):
        cache[c] = ""
    for c in ("red_prior_bouts", "blue_prior_bouts", "train_rows"):
        cache[c] = 1
    cache["p_ensemble"] = cache["p_model"]
    cache.to_csv(tmp / "cache.csv", index=False)
    pd.DataFrame([
        ("2024-01-06", "Bea Beta", "Ann Alpha", 250, -300),       # corners reversed
        ("2024-01-07", "Cal Gamma", "Dee Delta", 150, -170),      # UTC-dated, a day on
        ("2024-01-06", "E Epsilon", "Fay Zeta", -120, 100),       # initial + surname
        ("2024-01-06", "C Gamma", "Zed Other", -120, 100),        # initial, other corner wrong
        ("2024-01-06", "Gus Eta", "Hal Theta", 150, 150),         # sums below 100%
        ("2024-03-02", "Ann Alpha", "Kimmy Lambda", -200, 170),   # alias
        ("2024-01-06", "Lu Mu", "Nu Xi", -110, -110),             # the draw
    ], columns=["date", "fighter_a", "fighter_b", "odds_a", "odds_b"]).to_csv(tmp / "odds.csv", index=False)
    pd.DataFrame([
        ("2024-01-06", "Gus Eta", "Hal Theta", 120, -140),
        ("2024-01-06", "Ann Alpha", "Bea Beta", -280, 230),
    ], columns=["date", "fighter_a", "fighter_b", "open_a", "open_b"]).to_csv(tmp / "open.csv", index=False)
    pd.DataFrame([
        ("2024-01-06", "Bea Beta", "Ann Alpha", 700, 200, 800, 150, 1600, 500),
    ], columns=["date", "fighter_a", "fighter_b", "dec_a", "dec_b", "ko_a", "ko_b",
                "sub_a", "sub_b"]).to_csv(tmp / "method.csv", index=False)
    pd.DataFrame([
        ("c0", "2023-05-01", "UFC Z", "Ann Alpha", "Old Foe", "Ann Alpha", "KO/TKO"),
        ("c1", "2024-01-06", "UFC A", "Ann Alpha", "Bea Beta", "Ann Alpha", "KO/TKO"),
        ("c2", "2024-01-06", "UFC A", "Cal Gamma", "Dee Delta", "Dee Delta", "U-DEC"),
        ("c3", "2024-01-06", "UFC A", "Eve Epsilon", "Fay Zeta", "Eve Epsilon", "SUB Armbar"),
        ("c4", "2024-01-06", "UFC A", "Gus Eta", "Hal Theta", "Hal Theta", "KO/TKO"),
        ("c5", "2024-01-06", "UFC A", "Ivy Iota", "Jo Kappa", "Ivy Iota", "Decision - Split"),
        ("d1", "2024-01-06", "UFC A", "Lu Mu", "Nu Xi", None, "Decision - Majority"),
        ("c6", "2024-03-02", "UFC B", "Ann Alpha", "Kim Lambda", "Ann Alpha", "KO/TKO"),
        ("c7", "2024-06-01", "UFC C", "Ann Alpha", "Late Foe", "Late Foe", "KO/TKO"),
    ], columns=["fight_id", "date", "event_name", "r_name", "b_name", "winner",
                "method"]).to_csv(tmp / "archive.csv", index=False)
    (tmp / "aliases.json").write_text(json.dumps({"_comment": "test", "Kimmy Lambda": "Kim Lambda"}))
    return dict(cache_path=tmp / "cache.csv", odds_path=tmp / "odds.csv",
                opening_path=tmp / "open.csv", method_path=tmp / "method.csv",
                archive_path=tmp / "archive.csv", aliases_path=tmp / "aliases.json",
                start="2024-01-01")


@pytest.fixture
def built(tmp_path):
    frame, coverage = parlay_legs.build(**_write_fixture(tmp_path))
    return frame.set_index("fight_id"), coverage


def test_legs_join_prices_point_in_time_and_honestly(built):
    f, cov = built
    assert f.loc["c1", "price_source"] == "close"
    assert f.loc["c1", "dec_red"] == pytest.approx(pp.decimal_from_american(-300))
    assert f.loc["c1", "dec_blue"] == pytest.approx(pp.decimal_from_american(250))
    assert f.loc["c2", "price_source"] == "close"                     # one day off
    assert f.loc["c3", "price_source"] == "close"                     # E Epsilon
    assert f.loc["c4", "price_source"] == "open"                      # corrupt close
    assert f.loc["c4", "dec_red"] == pytest.approx(pp.decimal_from_american(120))
    assert pd.isna(f.loc["c5", "price_source"])                       # nothing guessed
    assert pd.isna(f.loc["c5", "dec_red"]) and pd.isna(f.loc["c5", "dec_blue"])
    assert f.loc["c6", "price_source"] == "close"                     # through the alias
    t = cov["total"]
    assert t["close_rejected"] == 1 and t["close_shifted"] == 1 and t["close_initial"] == 1
    assert t["excluded_fights"] == 1 and t["close_rows_excluded_fight"] == 1
    assert t["close_rows_unmatched"] == 1                             # C Gamma / Zed Other
    assert t["priced_close"] == 4 and t["priced_open"] == 1 and t["unpriced"] == 1


def test_legs_orient_method_props_to_the_corners(built):
    f, cov = built
    assert f.loc["c1", "ko_red"] == pytest.approx(pp.decimal_from_american(150))
    assert f.loc["c1", "ko_blue"] == pytest.approx(pp.decimal_from_american(800))
    assert f.loc["c1", "sub_red"] == pytest.approx(pp.decimal_from_american(500))
    assert f.loc["c1", "decision_red"] == pytest.approx(pp.decimal_from_american(200))
    assert f.loc["c1", "decision_blue"] == pytest.approx(pp.decimal_from_american(700))
    assert cov["total"]["method_priced"] == 1


def test_legs_facts_are_strictly_before_the_fight(built):
    f, _ = built
    first = json.loads(f.loc["c1", "facts_red"])
    second = json.loads(f.loc["c6", "facts_red"])
    assert first["fights"] == 1 and first["ko_wins"] == 1             # 2023 only, not c1 itself
    assert second["fights"] == 2 and second["wins"] == 2              # not the June loss
    assert f.loc["c2", "winner"] == "blue" and f.loc["c3", "method"] == "Submission"
    assert list(f["number"]) == [1, 2, 3, 4, 5, 1]


def test_harness_cards_tickets_and_the_ticket_file_agree(built, tmp_path):
    f, _ = built
    frame = f.reset_index()
    path = tmp_path / "legs.csv.gz"
    parlay_legs.write(frame, path)
    frame = pd.read_csv(path)
    cards = harness.cards(frame)
    assert [(e, d, len(fs)) for e, d, fs, _ in cards] == [("UFC A", "2024-01-06", 5),
                                                          ("UFC B", "2024-03-02", 1)]
    fights = cards[0][2]
    assert fights[0]["result"] == {"winner": "red", "method": "KO/TKO"}
    assert "p_blend" not in fights[0]
    out = harness.run(frame)
    assert set(out["results"]["window"]) == set(harness.NAMES)
    assert out["rows"], "the fixture card is priced enough for some ticket"
    for row in out["rows"]:
        lo, hi = (float(x) for x in row["band"].split("-"))
        assert lo <= row["multiplier"] <= hi
        prod = np.prod([float(x.rsplit("@", 1)[1]) for x in row["legs"].split("|")])
        assert prod == pytest.approx(row["multiplier"], rel=1e-3)


def test_periods_split_on_the_registered_dates():
    assert harness.period_of("2016-10-31", "2026-08-08") is None
    assert harness.period_of("2016-11-01", "2026-08-08") == "dev"
    assert harness.period_of("2023-10-31", "2026-08-08") == "dev"
    assert harness.period_of("2023-11-01", "2026-08-08") == "window"
    assert harness.period_of("2026-08-08", "2026-08-08") == "window"
    assert harness.period_of("2026-08-09", "2026-08-08") is None


def _res(roi, tickets=50):
    return {"tickets": tickets, "roi": roi}


def test_the_ship_rule_needs_mixed_to_win_both_periods():
    base = {name: _res(0.0) for name in harness.NAMES}
    results = {"dev": dict(base), "window": dict(base)}
    results["dev"]["mixed@steady"] = _res(0.10)
    results["window"]["mixed@steady"] = _res(0.05)          # beats both: ships
    results["dev"]["mixed@risker"] = _res(0.50)
    results["window"]["mixed@risker"] = _res(-0.01)         # one period only: does not
    results["dev"]["mixed@risk_lover"] = _res(0.50)
    results["window"]["risk_lover"] = {"tickets": 0}        # nothing to beat: named ships
    results["window"]["mixed@risk_lover"] = _res(0.50)
    assert harness.ship(results) == {"steady": "mixed", "risker": "risker",
                                     "risk_lover": "risk_lover"}


def test_a_tie_keeps_the_named_profile():
    base = {name: _res(0.0) for name in harness.NAMES}
    assert harness.ship({"dev": dict(base), "window": dict(base)}) == {
        b: b for b in pp.ORDER}


def test_the_contract_lists_every_profile_and_band():
    assert harness.NAMES == ("steady", "risker", "risk_lover", "risk_lover_method",
                             "mixed@steady", "mixed@risker", "mixed@risk_lover",
                             "market_only@steady", "market_only@risker",
                             "market_only@risk_lover",
                             # added after the first run, labelled post hoc
                             "mixed_else@steady", "mixed_else@risker",
                             "mixed_else@risk_lover")
    assert harness.PERIODS["dev"] == ("2016-11-01", "2023-10-31")
    assert harness.PERIODS["window"][0] == "2023-11-01"
