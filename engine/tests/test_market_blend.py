"""The blend must not eat the edge, and must not swallow the vig.

Blending with the closing line is the largest single accuracy gain available
in this project - 62.0% to 68.2% walk-forward - and it has two ways of going
quietly wrong.

The first is the vig. Both sides' prices imply probabilities summing to more
than one, and blending with the raw implied number folds the bookmaker's
margin into the prediction, tilting every fight toward the favourite. The
version of this that already existed in predict_fight did exactly that.

The second is the edge. An edge measured against a probability that is itself
three-quarters market is the market measured against itself. The two numbers
stay separate, and these tests hold them apart.
"""

import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from market_blend import (MARKET_WEIGHT, MAX_MARGIN, blend, devig, edge,
                          implied_probability)


# --- reading a price -------------------------------------------------------

def test_implied_probability_both_directions():
    assert implied_probability(-200) == pytest.approx(2 / 3)
    assert implied_probability(+100) == pytest.approx(0.5)
    assert implied_probability(+300) == pytest.approx(0.25)


def test_a_missing_or_zero_price_is_none_not_a_half():
    assert implied_probability(None) is None
    assert implied_probability(0) is None


# --- the vig ---------------------------------------------------------------

def test_the_margin_is_removed():
    """-200 and +170 imply 66.7% and 37.0%, which sum to 103.7%."""
    raw = implied_probability(-200)
    fair = devig(-200, +170)
    assert raw > fair, "de-vigging must reduce a favourite's probability"
    assert fair == pytest.approx(raw / (raw + implied_probability(170)))
    assert 0.6 < fair < 0.66


def test_both_sides_of_a_devigged_pair_sum_to_one():
    a = devig(-200, +170)
    b = devig(+170, -200)
    assert a + b == pytest.approx(1.0)


def test_an_even_market_devigs_to_a_half():
    assert devig(-110, -110) == pytest.approx(0.5)


def test_a_missing_opponent_price_means_no_market_view():
    """One side alone cannot have its margin removed."""
    assert devig(-200, None) is None
    assert devig(None, +170) is None


def test_an_impossible_margin_is_refused():
    """A pair implying a 30% margin is a stale or mistyped line."""
    assert devig(-2000, -2000) is None          # implies ~190%
    assert devig(-110, -110, max_margin=MAX_MARGIN) is not None


# --- the blend -------------------------------------------------------------

def test_the_blend_moves_toward_the_market():
    out = blend(0.40, 0.80, weight=0.75)
    assert out == pytest.approx(0.70)


def test_an_unpriced_fight_keeps_the_model_untouched():
    """Not pushed toward a half: nobody pricing it is not evidence it is
    close."""
    assert blend(0.72, None) == pytest.approx(0.72)


def test_a_weight_of_zero_is_the_model_and_one_is_the_market():
    assert blend(0.3, 0.9, weight=0.0) == pytest.approx(0.3)
    assert blend(0.3, 0.9, weight=1.0) == pytest.approx(0.9)


def test_the_blend_stays_a_probability():
    assert 0.0 < blend(0.999, 0.999) < 1.0
    assert 0.0 < blend(0.001, 0.001) < 1.0


def test_the_shipped_weight_leans_on_the_market():
    """It was measured at 0.75 for every year from 2016. A future change to
    this constant should be a measurement, and this records what it was."""
    assert MARKET_WEIGHT == pytest.approx(0.75)


# --- the edge --------------------------------------------------------------

def test_the_edge_is_measured_from_the_raw_model_not_the_blend():
    """The failure this guards: feeding the blended number back in makes the
    market the yardstick for itself and reports the remainder as an edge."""
    model, market = 0.40, 0.80
    honest = edge(model, market)
    assert honest == pytest.approx(-0.40)

    blended = blend(model, market, weight=0.75)
    flattered = edge(blended, market)
    assert abs(flattered) < abs(honest) / 3, (
        "blending shrinks the apparent edge; the edge must not be computed "
        "from the blended number")


def test_the_edge_is_none_when_there_is_no_market():
    assert edge(0.7, None) is None


def test_blending_shrinks_a_disagreement_without_flipping_it():
    """Direction is what a bet is placed on; it must survive."""
    for model, market in ((0.70, 0.40), (0.30, 0.60), (0.55, 0.45)):
        before = edge(model, market)
        after = blend(model, market) - market
        assert before * after > 0, "the disagreement changed sign"
        assert abs(after) < abs(before)


# --- the wiring ------------------------------------------------------------
# predict_card is not imported: it trains half a dozen models at import time.
# The source is read instead, which is enough to catch the failure that
# matters - the edge quietly being computed from the blended number.

def _source():
    return (ENGINE / "predict_card.py").read_text()


def test_the_shipping_path_blends_and_keeps_the_raw_model():
    src = _source()
    assert "p_win_model = p_win" in src, "the raw model probability is not kept"
    assert "p_win = _blend.blend(p_win, market_devigged)" in src
    assert "'win_prob_model': float(max(p_win_model, 1 - p_win_model))" in src


def test_the_blend_devigs_rather_than_using_the_raw_implied_price():
    """The version this replaced folded the bookmaker's margin into every
    prediction by blending with american_to_implied_prob directly."""
    src = _source()
    assert "_blend.devig(" in src
    blend_block = src[src.index("--- THE CLOSING LINE"):
                      src.index("p_win = _blend.blend(")]
    assert "american_to_implied_prob" not in blend_block


def test_the_edge_reads_the_unblended_probability():
    src = _source()
    assert "model_prob = pred.get('win_prob_model', pred['win_prob'])" in src
    value = src[src.index("def analyze_fight_value"):]
    # analyze_fight_value is the last def in the file, so there may be no
    # following one to slice on; take what follows either way.
    nxt = value.find("\ndef ", 1)
    value = value[:nxt] if nxt != -1 else value
    assert "calculate_value(model_prob, best_odds)" in value
    assert "calculate_value(pred['win_prob']" not in value, (
        "the edge is being measured from the blended probability, which "
        "compares the market with itself")


def test_an_unpriced_fight_is_still_predicted():
    """blend() returns the model untouched without a market, so a fight
    nobody priced keeps a real number rather than losing its prediction."""
    src = _source()
    assert "p_win = _blend.blend(p_win, market_devigged)" in src
    # market_devigged starts as None and only becomes a number when BOTH
    # sides matched, so one-sided prices cannot half-blend.
    assert "market_devigged = None" in src
    assert "if _mine and _theirs:" in src
