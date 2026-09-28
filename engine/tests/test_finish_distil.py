"""The student must refuse rather than guess, and must not read a result.

Training toward the market instead of the outcome is worth +0.053 AUC and
closes two thirds of the distance to the market, measured walk-forward over
4,764 fights. The ways it could go wrong quietly:

  fitting on a handful of prices and still answering every fight
  reading a price that cannot be right - a mis-scrape implying 160%
  learning from prices that were published after the fight it predicts
  being handed a feature matrix in units it was not fitted in

The third is the one that would look like a triumph. The fourth actually
happened: the student was fitted on the raw feature frame and predicted on
the standardised one, and returned a narrow band of confident numbers for a
whole card without complaining. These hold the line.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import finish_distil as fd


def prices(dec_a=300, dec_b=300, ko_a=400, ko_b=400, sub_a=800, sub_b=800):
    return pd.DataFrame([{ "dec_a": dec_a, "dec_b": dec_b, "ko_a": ko_a,
                           "ko_b": ko_b, "sub_a": sub_a, "sub_b": sub_b }])


# --- reading the prices ----------------------------------------------------

def test_implied_probability_both_directions():
    assert fd.implied(-200) == pytest.approx(2 / 3)
    assert fd.implied(+100) == pytest.approx(0.5)


def test_the_six_prices_devig_to_one():
    p, total = fd.market_finish_probability(prices())
    assert total[0] > 1.0, "these props carry a margin; it should show"
    # P(finish) and P(decision) are the two halves of one de-vigged whole.
    assert 0.0 < p[0] < 1.0


def test_shorter_decision_prices_mean_fewer_finishes():
    short_dec, _ = fd.market_finish_probability(prices(dec_a=-200, dec_b=-200))
    long_dec, _ = fd.market_finish_probability(prices(dec_a=900, dec_b=900))
    assert short_dec[0] < long_dec[0]


def test_a_broken_overround_is_thrown_away(tmp_path):
    """A row implying 160% between six prices is a mis-scrape, not a market."""
    path = tmp_path / "m.csv"
    rows = prices().assign(date="2020-01-01", fighter_a="A", fighter_b="B")
    broken = prices(dec_a=-900, dec_b=-900, ko_a=-900, ko_b=-900,
                    sub_a=-900, sub_b=-900).assign(
                        date="2020-02-01", fighter_a="C", fighter_b="D")
    pd.concat([rows, broken]).to_csv(path, index=False)

    key = lambda a, b, d: (a, b)
    targets = fd.load_targets(path, key)
    assert ("A", "B") in targets
    assert ("C", "D") not in targets


def test_a_row_missing_any_price_is_skipped(tmp_path):
    path = tmp_path / "m.csv"
    rows = prices().assign(date="2020-01-01", fighter_a="A", fighter_b="B")
    rows.loc[0, "sub_b"] = None
    rows.to_csv(path, index=False)
    assert fd.load_targets(path, lambda a, b, d: (a, b)) == {}


# --- refusing ---------------------------------------------------------------

def test_too_few_prices_means_no_student():
    """A model fitted on eighty fights still answers every bout on a card,
    and nothing in its output says it was guessing."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = rng.uniform(0.3, 0.7, 100)
    assert fd.fit(X, y) is None
    assert fd.fit(X, y, min_train=50) is not None


def test_rows_without_a_target_do_not_count_toward_the_floor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 5))
    y = np.full(1000, np.nan)
    y[:100] = rng.uniform(0.3, 0.7, 100)
    assert fd.fit(X, y) is None, "NaN targets were counted as training data"


def test_predict_with_no_student_is_none_not_a_half():
    assert fd.predict(None, np.zeros((3, 5))) is None


def test_predictions_stay_probabilities():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 4))
    # A target the model can fit perfectly would otherwise push outside [0,1].
    y = np.clip(X[:, 0] * 5, 0.001, 0.999)
    model = fd.fit(X, y)
    p = fd.predict(model, X)
    assert p.min() >= 0.0 and p.max() <= 1.0


# --- the thing that would look like a triumph -------------------------------

def test_the_student_learns_the_market_not_the_outcome():
    """Given a market that disagrees with what happened, the student must
    follow the market. If it tracked the outcome instead, something is
    feeding it results."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 3))
    market = 1 / (1 + np.exp(-X[:, 0]))          # the teacher
    outcome = (rng.random(2000) < 0.5).astype(float)   # pure noise
    model = fd.fit(X, market)
    p = fd.predict(model, X)
    assert np.corrcoef(p, market)[0, 1] > 0.9
    assert abs(np.corrcoef(p, outcome)[0, 1]) < 0.15


def test_the_student_needs_no_price_to_predict():
    """The whole reason this exists rather than a blend: the method market is
    not available live, so the model has to carry it in its weights."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1000, 3))
    model = fd.fit(X, 1 / (1 + np.exp(-X[:, 0])))
    fresh = rng.normal(size=(5, 3))
    assert fd.predict(model, fresh).shape == (5,)


# --- the wiring ------------------------------------------------------------
# predict_card is not imported here: it trains half a dozen models at import
# time. The source is read instead, which catches the failures that matter -
# the student silently not being used, or quietly moving a split it never
# measured.

def _source():
    return (ENGINE / "predict_card.py").read_text()


def test_the_student_is_trained_only_on_the_training_window():
    """A price from the calibration split is a price from after what the
    model was fitted on."""
    src = _source()
    assert "_student_targets[:train_end_m_full] = _rows[:train_end_m_full]" in src


def test_the_student_supplies_p_finish_in_the_shipping_path():
    src = _source()
    assert "_student = _distil.predict(FINISH_STUDENT, X_pred_s)" in src
    assert "_rebuild_three_way(_packed, np.array([float(_student[0])]))" in src


def test_the_ko_versus_submission_split_is_left_to_the_three_way_model():
    """Distillation measured the binary and nothing else. rebuild_three_way
    keeps the KO:SUB ratio exactly as the classifier had it."""
    src = _source()
    block = src[src.index("THE DISTILLED BINARY"):
                src.index("method_probs = calibrate_method_probs(method_probs)")]
    # Nothing in the block may set a KO or Submission probability from the
    # student; they come out of rebuild_three_way, which preserves the ratio.
    assert "_rebuild_three_way" in block
    assert "'KO/TKO': float(_rebuilt[1])" in block


def test_calibration_still_runs_after_the_student():
    """The student inherits the market's bias - it says 56.5% finishes where
    50.8% happen - and isotonic is what fixes the price."""
    src = _source()
    student_at = src.index("_student = _distil.predict")
    calibrate_at = src.index("method_probs = calibrate_method_probs(method_probs)")
    assert student_at < calibrate_at


def test_a_missing_price_file_does_not_break_the_card():
    src = _source()
    assert "if _METHOD_ODDS_PATH.exists():" in src
    assert "the three-way model supplies " in src


def test_the_calibrator_is_fitted_on_whatever_supplies_p_finish():
    """The mismatch this project has already fixed twice: a calibrator fitted
    on one quantity and applied to another. Once the student exists it is the
    student that produces P(finish), so calibrating the three-way model's
    output would map from a distribution nothing ever emits."""
    src = _source()
    assert "_student_cal = _distil.predict(FINISH_STUDENT, X_cal_m_full)" in src
    block = src[src.index("_student_cal = _distil.predict"):
                src.index("FINISH_CALIBRATOR = _fit_finish_calibrator")]
    assert "_cal_raw_finish = _student_cal" in block
    # And the fallback for a run with no price file.
    assert "xgb_method_prod.predict_proba(" in block


def test_the_student_is_built_before_the_calibrator_needs_it():
    src = _source()
    assert src.index("FINISH_STUDENT = _distil.fit") < \
        src.index("_student_cal = _distil.predict")


# --- the units guard -------------------------------------------------------
# These exist because of a shipped bug: the student was fitted on the raw
# feature frame and predicted on the standardised one. Trees do not fail on
# that - they return a narrow band of confident-looking numbers.

def _trained_student(rng, n=600, columns=12):
    X = rng.normal(loc=50.0, scale=10.0, size=(n, columns))
    target = 0.5 + 0.02 * (X[:, 0] - 50.0) / 10.0
    return fd.fit(X, np.clip(target, 0.05, 0.95)), X


def test_predict_refuses_a_matrix_in_different_units():
    rng = np.random.default_rng(0)
    model, X = _trained_student(rng)
    assert model is not None
    scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    with pytest.raises(ValueError, match="units mismatch"):
        fd.predict(model, scaled)


def test_predict_accepts_the_space_it_was_fitted_on():
    rng = np.random.default_rng(1)
    model, X = _trained_student(rng)
    out = fd.predict(model, X[:20])
    assert out is not None and len(out) == 20


def test_an_unusual_row_still_gets_a_prediction():
    """The guard is not an outlier detector. One strange fighter must not
    stop the card."""
    rng = np.random.default_rng(2)
    model, X = _trained_student(rng)
    strange = X[:1].copy()
    strange[0, 0] += 40.0          # four standard deviations out on one column
    out = fd.predict(model, strange)
    assert out is not None and 0.0 < out[0] < 1.0


def test_the_spread_check_needs_rows_and_says_so():
    """An honest test of a known weakness.

    A handful of rows cannot show a column's spread, so a single card in the
    wrong units gets through this guard. It is caught instead by the held-out
    comparison in predict_card, which refuses a student that does not beat the
    model it replaces. Writing the weakness down here stops someone reading
    the guard above and believing it covers a case it does not.
    """
    rng = np.random.default_rng(4)
    model, X = _trained_student(rng)
    scaled = ((X - X.mean(axis=0)) / X.std(axis=0))[:5]
    assert fd.predict(model, scaled) is not None


def test_wrong_number_of_features_is_refused_by_name():
    rng = np.random.default_rng(3)
    model, X = _trained_student(rng)
    with pytest.raises(ValueError, match="fitted on 12 features"):
        fd.predict(model, X[:5, :8])
