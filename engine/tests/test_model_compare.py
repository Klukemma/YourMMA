"""The shared walk-forward harness must be point-in-time and must not flatter.

Every later idea is judged by experiments/model_compare.py, so these tests
hold it to the three things that make a verdict worth believing: a year's
predictions come from models that never saw that year; the two arms of a
comparison are scored on the same fights with intervals that respect the
clustering of fights into cards and of cards onto fighters; and the market
is matched the way the rest of the project matches it. Synthetic data
throughout, so the whole file runs in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments import model_compare as mc  # noqa: E402


def synthetic(n_per_year=120, years=range(2008, 2016), seed=0, features=4):
    """Fights with a weak real signal in feature 0 and noise elsewhere."""
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for year in years:
        for i in range(n_per_year):
            x = rng.normal(size=features)
            p = 1 / (1 + np.exp(-0.8 * x[0]))
            rows.append({"fight_id": f"{year}-{i:04d}",
                         "date": (pd.Timestamp(f"{year}-01-01")
                                  + pd.Timedelta(days=int(i * 360 / n_per_year))),
                         "event_name": f"E{year}-{i // 10}",
                         "r_name": f"Red {i % 40}", "b_name": f"Blue {i % 40}",
                         "target_win": float(rng.uniform() < p)})
            X.append(x)
    ufc = pd.DataFrame(rows)
    X = pd.DataFrame(np.array(X), columns=[f"f{j}" for j in range(features)])
    return ufc, X, ufc["target_win"].to_numpy(float).copy()


# --- point in time ---------------------------------------------------------

def test_every_year_is_predicted_by_models_fitted_on_earlier_years_only():
    ufc, X, y = synthetic()
    seen = []

    def spy_fitter(X_train, y_train, dates):
        dates = pd.to_datetime(pd.Series(dates))
        seen.append({"max_date": dates.max(), "n": len(X_train)})
        return seen[-1]

    def spy_predictor(models, X_test):
        p = np.full(len(X_test), 0.5)
        return p, p

    out = mc.walk_forward(X, y, ufc, fitter=spy_fitter, predictor=spy_predictor,
                          first_year=2010, min_train=100, min_test=10, verbose=False)
    years = sorted(out["year"].unique())
    assert years == list(range(2010, 2016))
    assert len(seen) == len(years), "one fit per predicted year"
    seen = dict(zip(years, seen))
    for year in out["year"].unique():
        assert seen[year]["max_date"] < pd.Timestamp(f"{year}-01-01")
        # and the training set is exactly every earlier fight, not a window
        assert seen[year]["n"] == int((ufc["date"].dt.year < year).sum())
        assert int((out["year"] == year).sum()) == int((ufc["date"].dt.year == year).sum())
    # every row carries its own date and that date is in the row's year
    assert (pd.to_datetime(out["date"]).dt.year == out["year"]).all()


@pytest.mark.parametrize("recipe", sorted(mc.RECIPES))
def test_a_signal_that_exists_only_in_the_predicted_year_cannot_be_used(recipe):
    """Feature 3 is pure noise before 2013 and the exact answer in 2013. A
    harness that leaked 2013 into its own models would score near 100% on
    2013; the honest one scores like the weak feature-0 signal alone. Both
    recipes: the legacy one as listed, the shipped one on the paired matrix
    (a synthetic fight's swapped orientation is its negation, every feature
    being a red-minus-blue difference - the leak with it)."""
    ufc, X, y = synthetic(n_per_year=150, years=range(2008, 2014), seed=3)
    X = X.copy()
    leak_year = ufc["date"].dt.year == 2013
    X.loc[leak_year, "f3"] = np.where(y[leak_year] == 1, 3.0, -3.0)
    spec = mc.RECIPES[recipe]
    X_in = mc.paired(X, -X) if recipe == "symmetric" else X
    out = mc.walk_forward(X_in, y, ufc, fitter=spec["fit"], predictor=spec["predict"],
                          first_year=2013, min_train=100, min_test=10, verbose=False)
    assert set(out["year"]) == {2013}
    acc = mc.correct_rows(out["y"], out["p_model"]).mean()
    assert acc < 0.80, f"2013 accuracy {acc:.2f} - the harness saw its own answer"
    # the same fitter, in-sample, DOES exploit the leak - so the test bites
    models = spec["fit"](X_in[leak_year], y[leak_year], ufc.loc[leak_year, "date"])
    _, p_in = spec["predict"](models, X_in[leak_year])
    assert mc.correct_rows(y[leak_year], p_in).mean() > 0.95


def test_the_production_recipe_calibrates_through_platt():
    ufc, X, y = synthetic(n_per_year=200, years=range(2008, 2012))
    models = mc.fit_production(X, y, ufc["date"])
    p_ens, p = mc.predict_production(models, X)
    assert p.shape == p_ens.shape == (len(X),)
    assert np.all((p > 0) & (p < 1))
    assert not np.allclose(p, p_ens), "Platt must be applied, not skipped"
    assert models["cal_rows"] == int(len(X) * mc.CAL_FRACTION)
    assert models["train_rows"] + models["cal_rows"] == len(X)


def test_the_calibration_block_is_the_most_recent_tenth_by_date_not_a_random_tenth():
    """predict_card.py:1731-1733 - the last 10% of date-ordered rows calibrate
    (and XGB early-stops on them, and recency weights rise toward them)."""
    ufc, X, y = synthetic(n_per_year=200, years=range(2008, 2012))
    models = mc.fit_production(X, y, ufc["date"])
    dates = ufc["date"].sort_values().reset_index(drop=True)
    cal_rows = int(len(X) * mc.CAL_FRACTION)
    assert models["train_max_date"] <= models["cal_min_date"]
    assert models["cal_min_date"] == dates.iloc[len(X) - cal_rows]
    assert models["cal_max_date"] == dates.iloc[-1]
    assert models["train_min_date"] == dates.iloc[0]
    # the recipe refuses rows that are not in date order, so a shuffled frame
    # cannot quietly calibrate on a random block
    perm = np.random.default_rng(0).permutation(len(X))
    with pytest.raises(ValueError):
        mc.fit_production(X.iloc[perm], y[perm], ufc["date"].iloc[perm])


def test_recency_weights_rise_toward_the_present_and_average_one():
    dates = pd.date_range("2010-01-01", periods=50, freq="30D")
    w = mc.recency_weights(dates)
    assert w.mean() == pytest.approx(1.0)
    assert np.all(np.diff(w) >= 0)
    assert w[-1] / w[0] == pytest.approx(3.0)       # 1 -> 1 + 2 * 1 ** 1.5


# --- the two recipes -------------------------------------------------------

def test_paired_holds_both_orientations_and_halves_takes_them_apart():
    ufc, X, y = synthetic(n_per_year=10, years=range(2010, 2012))
    Xp = mc.paired(X, -X)
    assert list(Xp.columns) == list(X.columns) + [mc.SWAP_PREFIX + c for c in X.columns]
    a, b = mc.halves(Xp)
    assert np.array_equal(a, X.to_numpy(float)) and np.array_equal(b, -X.to_numpy(float))
    with pytest.raises(ValueError):
        mc.paired(X, X.iloc[:-1])
    with pytest.raises(ValueError):
        mc.paired(X, X.rename(columns={"f0": "g0"}))
    with pytest.raises(ValueError):
        mc.halves(np.zeros((3, 5)))


def test_the_shipped_recipe_is_symmetric_and_calibrates_through_its_own_platt():
    ufc, X, y = synthetic(n_per_year=200, years=range(2008, 2012), seed=5)
    Xp = mc.paired(X, -X)
    models = mc.fit_symmetric(Xp, y, ufc["date"])
    e, p = mc.predict_symmetric(models, Xp)
    e2, p2 = mc.predict_symmetric(models, mc.paired(-X, X))
    assert np.allclose(e + e2, 1.0), "the ensemble is exactly symmetric"
    assert p.shape == e.shape == (len(X),) and np.all((p > 0) & (p < 1))
    assert not np.allclose(p, e), "Platt must be applied, not skipped"
    base = mc.fit_production(X, y, ufc["date"])
    assert models["train_rows"] == base["train_rows"], "the same split of FIGHTS"
    assert models["augmented_rows"] == 2 * models["train_rows"]
    with pytest.raises(ValueError):
        mc.fit_symmetric(Xp.iloc[::-1], y[::-1], ufc["date"].iloc[::-1])


def test_the_two_recipes_are_keyed_apart_and_the_legacy_key_is_what_it_always_was():
    ufc, X, y = synthetic(n_per_year=20, years=range(2010, 2012))
    cols = list(X.columns)
    legacy = mc.inputs_hash(X, y, ufc, cols, protocol=mc.RECIPES["legacy"]["protocol"],
                            code=mc.code_fingerprint("legacy"))
    assert legacy == mc.inputs_hash(X, y, ufc, cols), "the defaults are the legacy recipe's"
    assert mc.code_fingerprint() == mc.code_fingerprint("legacy")
    assert mc.code_fingerprint("symmetric") != mc.code_fingerprint("legacy")
    assert mc.PROTOCOL != mc.PROTOCOL_SYMMETRIC
    assert {s["protocol"] for s in mc.RECIPES.values()} == {mc.PROTOCOL, mc.PROTOCOL_SYMMETRIC}
    assert {s["label"] for s in mc.RECIPES.values()} == {"baseline_legacy", "baseline_symmetric"}
    assert mc.RECIPES[mc.DEFAULT_RECIPE]["fit"] is mc.fit_symmetric
    Xp = mc.paired(X, -X)
    symmetric = mc.inputs_hash(Xp, y, ufc, list(Xp.columns), protocol=mc.PROTOCOL_SYMMETRIC,
                               code=mc.code_fingerprint("symmetric"))
    assert symmetric != legacy
    with pytest.raises(KeyError):
        mc.code_fingerprint("something-else")


def test_the_shipped_legacy_cache_is_still_valid_for_its_label():
    """model_compare_baseline_legacy.{csv,json} is the file ideas #3-#7
    were measured against (model_compare_baseline.* until idea #6's fix),
    renamed: its protocol and recipe fingerprint must still be the legacy
    recipe's, or every earlier verdict's yardstick has silently moved."""
    csv_path, meta_path = mc.cache_paths("baseline_legacy")
    if not meta_path.exists():
        pytest.skip("no legacy cache beside the module")
    meta = json.loads(meta_path.read_text())
    assert meta["protocol"] == mc.PROTOCOL
    assert meta["code"] == mc.code_fingerprint("legacy")
    assert meta["label"] == "baseline_legacy"
    head = pd.read_csv(csv_path, nrows=5)
    assert "p_market" not in head.columns and {"p_model", "train_rows"} <= set(head.columns)


def test_predictions_fits_the_shipped_recipe_by_default_and_the_legacy_one_by_name(tmp_path, monkeypatch):
    from experiments import symmetry as sy
    ufc, X, y = synthetic(n_per_year=30, years=range(2009, 2013))
    used = []

    def fake_frames(raw=None, features=None, swap_features=None, verbose=True):
        return {"ufc": ufc, "X": X, "X_swap": -X, "y": y, "feature_cols": list(X.columns)}

    def fake_build_frame(raw=None, features=None, verbose=False):
        return {"ufc": ufc, "X": X, "y": y, "feature_cols": list(X.columns)}

    def fake_walk_forward(X_, y_, ufc_, *, fitter, predictor, verbose=True):
        used.append((fitter, predictor, X_.shape[1]))
        return pd.DataFrame({"fight_id": ufc_["fight_id"], "date": ufc_["date"],
                             "year": ufc_["date"].dt.year, "event": ufc_["event_name"],
                             "red": ufc_["r_name"], "blue": ufc_["b_name"],
                             "red_norm": ufc_["r_name"].str.lower(),
                             "blue_norm": ufc_["b_name"].str.lower(),
                             "y": y_, "p_model": 0.5, "p_ensemble": 0.5,
                             "train_rows": 100})

    monkeypatch.setattr(sy, "frames", fake_frames)
    monkeypatch.setattr(mc, "build_frame", fake_build_frame)
    monkeypatch.setattr(mc, "walk_forward", fake_walk_forward)
    monkeypatch.setattr(mc, "load_prices", lambda path=None, columns=None: {})
    default = mc.predictions(cache_dir=tmp_path, verbose=False)
    assert default.attrs["recipe"] == "symmetric" and default.attrs["label"] == "baseline_symmetric"
    assert used[-1] == (mc.fit_symmetric, mc.predict_symmetric, 2 * X.shape[1])
    legacy = mc.predictions(recipe="legacy", cache_dir=tmp_path, verbose=False)
    assert legacy.attrs["recipe"] == "legacy" and legacy.attrs["label"] == "baseline_legacy"
    assert used[-1] == (mc.fit_production, mc.predict_production, X.shape[1])
    # two files under two protocols, neither servable as the other
    sym = json.loads((tmp_path / "model_compare_baseline_symmetric.json").read_text())
    leg = json.loads((tmp_path / "model_compare_baseline_legacy.json").read_text())
    assert sym["protocol"] == mc.PROTOCOL_SYMMETRIC and leg["protocol"] == mc.PROTOCOL
    assert sym["hash"] != leg["hash"] and sym["code"] != leg["code"]
    assert sym["orientations"] == 2 and sym["features"] == X.shape[1] == leg["features"]
    assert leg["hash"] == mc.inputs_hash(X, y, ufc, list(X.columns))
    n = len(used)
    mc.predictions(cache_dir=tmp_path, verbose=False)
    mc.predictions(recipe="legacy", cache_dir=tmp_path, verbose=False)
    assert len(used) == n, "both baselines served from their own caches"
    with pytest.raises(KeyError):
        mc.predictions(recipe="nope", cache_dir=tmp_path, verbose=False)


def test_decision_set_is_the_fights_whose_training_set_was_large_enough():
    """Later ideas pre-register their PRIMARY bar on this set (the regime
    production is in), all years secondary - see the docstring."""
    frame = pd.DataFrame({"year": [2011, 2017, 2018, 2026],
                          "train_rows": [1239, 3684, 4130, 8082]}, index=[10, 11, 12, 13])
    mask = mc.decision_set(frame)
    assert list(mask) == [False, False, True, True] and list(mask.index) == [10, 11, 12, 13]
    assert list(frame[mask]["year"]) == [2018, 2026]
    assert list(mc.decision_set(frame, min_train_rows=3000)) == [False, True, True, True]
    assert mc.DECISION_MIN_TRAIN_ROWS == 4000
    assert "PRIMARY" in mc.decision_set.__doc__ and "secondary" in mc.decision_set.__doc__
    with pytest.raises(KeyError):
        mc.decision_set(frame.drop(columns=["train_rows"]))


# --- prior bouts -----------------------------------------------------------

def test_prior_bouts_count_either_corner_strictly_before_the_fight():
    ufc = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "r_name": ["A", "B", "A", "C"],
        "b_name": ["B", "C", "C", "A"],
    })
    red, blue = mc.prior_bouts(ufc)
    assert list(red) == [0, 1, 1, 2]       # A:0, B:1, A:1, C:2
    assert list(blue) == [0, 0, 1, 2]      # B:0, C:0, C:1, A:2


def test_build_frame_counts_prior_bouts_on_the_full_archive_including_draws(monkeypatch):
    """A fighter whose only earlier bout was a draw is not a debutant: the
    count is taken before the target_win filter drops the draw."""
    import types
    ufc = pd.DataFrame({
        "fight_id": ["f1", "f2", "f3"],
        "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "event_name": ["E1", "E2", "E3"],
        "r_name": ["A", "A", "C"], "b_name": ["B", "C", "B"],
        "target_win": [np.nan, 1.0, 0.0],          # A-B was a draw
    })
    X = pd.DataFrame({"f0": [0.1, 0.2, 0.3]})
    fake = types.SimpleNamespace(build=lambda raw, verbose=False: {
        "ufc": ufc, "X": X, "feature_cols_winner": ["f0"]})
    monkeypatch.setitem(sys.modules, "feature_frame", fake)
    frame = mc.build_frame(raw=pd.DataFrame())
    assert list(frame["ufc"]["fight_id"]) == ["f2", "f3"], "draw dropped from the valid rows"
    assert list(frame["ufc"]["red_prior_bouts"]) == [1, 1]     # A had the draw; C had f2
    assert list(frame["ufc"]["blue_prior_bouts"]) == [0, 1]    # C debut; B had the draw
    # and walk_forward carries those columns through rather than recounting
    out = mc.walk_forward(frame["X"], frame["y"], frame["ufc"],
                          fitter=lambda X_, y_, d: None,
                          predictor=lambda m, X_: (np.full(len(X_), 0.5),) * 2,
                          first_year=2020, min_train=0, min_test=0, verbose=False)
    assert list(out["red_prior_bouts"]) == [1, 1] and list(out["blue_prior_bouts"]) == [0, 1]


# --- the market ------------------------------------------------------------

def test_market_is_matched_in_either_corner_order_on_the_exact_date(tmp_path):
    odds = tmp_path / "odds.csv"
    pd.DataFrame({
        "date": ["2020-05-09", "2020-05-09", "2020-05-16"],
        "fighter_a": ["Tony Ferguson", "Calvin Kattar", "Alistair Overeem"],
        "fighter_b": ["Justin Gaethje", "Jeremy Stephens", "Walt Harris"],
        "odds_a": [-170, -200, -110], "odds_b": [+150, +170, -110],
    }).to_csv(odds, index=False)
    frame = pd.DataFrame({
        "red_norm": ["tony ferguson", "jeremy stephens", "alistair overeem", "nobody"],
        "blue_norm": ["justin gaethje", "calvin kattar", "walt harris", "anyone"],
        "date": pd.to_datetime(["2020-05-09", "2020-05-09", "2020-05-17", "2020-05-09"]),
    })
    p = mc.market_probabilities(frame, mc.load_prices(odds))
    import market_blend
    assert p[0] == pytest.approx(market_blend.devig(-170, +150))
    assert p[1] == pytest.approx(market_blend.devig(+170, -200))   # flipped corners
    assert np.isnan(p[2]), "a day off is not a match - no tolerance, as residual_harness"
    assert np.isnan(p[3])
    # opening lines live under their own column names and are read the same way
    opening = tmp_path / "opening_odds.csv"
    pd.read_csv(odds).rename(columns={"odds_a": "open_a", "odds_b": "open_b"}).to_csv(
        opening, index=False)
    q = mc.market_probabilities(frame, mc.load_prices(opening, columns=("open_a", "open_b")))
    assert np.array_equal(np.isnan(q), np.isnan(p)) and q[0] == p[0] and q[1] == p[1]


# --- comparison ------------------------------------------------------------

def _arms(n=3000, seed=1, gain=0.03):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < truth).astype(float)
    base = np.clip(truth + rng.normal(0, 0.15, n), 0.02, 0.98)
    cand = np.clip(truth + rng.normal(0, 0.15 - gain * 3, n), 0.02, 0.98)
    events = np.array([f"E{i // 12}" for i in range(n)])
    red = np.array([f"r{rng.integers(0, 400)}" for _ in range(n)])
    blue = np.array([f"b{rng.integers(0, 400)}" for _ in range(n)])
    return y, base, cand, events, red, blue


def test_compare_is_paired_and_a_better_arm_shows_a_negative_log_loss_delta():
    y, base, cand, events, red, blue = _arms()
    r = mc.compare(y, base, cand, events, red=red, blue=blue, alpha=0.05, draws=500)
    d = r["delta"]["log_loss"]
    assert d["mean"] == pytest.approx(r["candidate"]["log_loss"] - r["baseline"]["log_loss"])
    assert d["high"] < 0
    assert d["low"] <= d["mean"] <= d["high"]
    # the reported interval is the wider of the two resampling schemes
    assert d["low"] == min(d["event_low"], d["fighter_low"])
    assert d["high"] == max(d["event_high"], d["fighter_high"])
    assert set(r["delta"]) == {"log_loss", "brier", "accuracy"}


def test_identical_arms_show_a_zero_delta_with_a_zero_width_interval():
    y, base, _, events, red, blue = _arms(n=500)
    r = mc.compare(y, base, base, events, red=red, blue=blue, draws=200)
    for name in ("log_loss", "brier", "accuracy"):
        assert r["delta"][name]["mean"] == 0.0
        assert r["delta"][name]["low"] == 0.0 and r["delta"][name]["high"] == 0.0


def test_a_tighter_family_wise_alpha_widens_the_interval():
    y, base, cand, events, red, blue = _arms(n=1500)
    loose = mc.compare(y, base, cand, events, red=red, blue=blue, alpha=0.05, draws=600)
    tight = mc.compare(y, base, cand, events, red=red, blue=blue, alpha=0.05 / 4, draws=600)
    t, l = tight["delta"]["log_loss"], loose["delta"]["log_loss"]
    assert t["low"] < l["low"] and t["high"] > l["high"]
    # 98.75% against 95%: well beyond percentile noise at 600 draws, so an
    # implementation that ignored alpha could not pass this
    assert (t["high"] - t["low"]) > 1.1 * (l["high"] - l["low"])


def test_event_resampling_is_wider_than_pretending_fights_are_independent():
    """When every fight on a card moves together, resampling fights would
    report an interval too narrow by the shared part."""
    rng = np.random.default_rng(7)
    n_events, per = 150, 10
    shared = rng.normal(0, 1.0, n_events)
    d = np.repeat(shared, per) + rng.normal(0, 0.1, n_events * per)
    events = np.repeat([f"E{i}" for i in range(n_events)], per)
    by_event = mc.bootstrap_interval(d, events, 0.05, draws=800)
    by_fight = mc.bootstrap_interval(d, np.arange(len(d)), 0.05, draws=800)
    assert (by_event[1] - by_event[0]) > 2.5 * (by_fight[1] - by_fight[0])


def test_the_two_arms_must_be_scored_on_the_same_fights():
    y, base, cand, events, _, _ = _arms(n=100)
    with pytest.raises(ValueError):
        mc.compare(y, base, cand[:-1], events)


def test_calibration_deciles_report_predicted_against_observed():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(float)
    rows = mc.calibration_by_decile(y, p)
    assert len(rows) == 10 and sum(r["n"] for r in rows) == 5000
    for r in rows:
        assert abs(r["predicted"] - r["observed"]) < 0.08


# --- cache -----------------------------------------------------------------

def test_the_cache_key_moves_with_the_inputs_and_the_protocol():
    ufc, X, y = synthetic(n_per_year=20, years=range(2010, 2012))
    cols = list(X.columns)
    key = mc.inputs_hash(X, y, ufc, cols)
    assert key == mc.inputs_hash(X.copy(), y.copy(), ufc.copy(), list(cols))
    X2 = X.copy()
    X2.iloc[0, 0] += 1e-6
    assert key != mc.inputs_hash(X2, y, ufc, cols)
    assert key != mc.inputs_hash(X, 1 - y, ufc, cols)
    assert key != mc.inputs_hash(X, y, ufc, cols[::-1])
    assert key != mc.inputs_hash(X, y, ufc, cols, protocol="something-else")
    # the recipe's own source is in the key, so an edit to fit_production
    # misses the cache even if nobody bumps PROTOCOL
    assert key == mc.inputs_hash(X, y, ufc, cols, code=mc.code_fingerprint())
    assert key != mc.inputs_hash(X, y, ufc, cols, code="edited recipe")
    with_bouts = ufc.assign(red_prior_bouts=0, blue_prior_bouts=0)
    assert key != mc.inputs_hash(X, y, with_bouts, cols)
    assert mc.inputs_hash(X, y, with_bouts, cols) != mc.inputs_hash(
        X, y, with_bouts.assign(red_prior_bouts=1), cols)


def test_predictions_are_served_from_cache_only_when_the_hash_matches(tmp_path, monkeypatch):
    ufc, X, y = synthetic(n_per_year=30, years=range(2009, 2013))
    calls = {"fit": 0}

    def fake_build_frame(raw=None, features=None, verbose=False):
        return {"ufc": ufc, "X": X, "y": y, "feature_cols": list(X.columns)}

    def fake_walk_forward(X_, y_, ufc_, verbose=True, **recipe):
        calls["fit"] += 1
        return pd.DataFrame({"fight_id": ufc_["fight_id"], "date": ufc_["date"],
                             "year": ufc_["date"].dt.year, "event": ufc_["event_name"],
                             "red": ufc_["r_name"], "blue": ufc_["b_name"],
                             "red_norm": ufc_["r_name"].str.lower(),
                             "blue_norm": ufc_["b_name"].str.lower(),
                             "y": y_, "p_model": 0.5, "p_ensemble": 0.5})

    monkeypatch.setattr(mc, "build_frame", fake_build_frame)
    monkeypatch.setattr(mc, "walk_forward", fake_walk_forward)
    monkeypatch.setattr(mc, "load_prices", lambda path=None, columns=None: {})
    first = mc.predictions(recipe="legacy", label="t", cache_dir=tmp_path, verbose=False)
    again = mc.predictions(recipe="legacy", label="t", cache_dir=tmp_path, verbose=False)
    assert calls["fit"] == 1 and len(first) == len(again) == len(ufc)
    meta = json.loads((tmp_path / "model_compare_t.json").read_text())
    assert meta["hash"] == mc.inputs_hash(X, y, ufc, list(X.columns))
    assert meta["market_cached"] is False
    assert "p_market" not in pd.read_csv(tmp_path / "model_compare_t.csv").columns
    # the archive changes: the hash misses and the fit is redone
    y[0] = 1 - y[0]
    mc.predictions(recipe="legacy", label="t", cache_dir=tmp_path, verbose=False)
    assert calls["fit"] == 2


def test_a_changed_odds_file_is_seen_on_a_cache_hit_without_a_refit(tmp_path, monkeypatch):
    """The 2025 backfill must show up the next time anyone asks, not after
    somebody remembers to delete the cache."""
    ufc, X, y = synthetic(n_per_year=10, years=range(2009, 2012))
    calls = {"fit": 0}

    def fake_build_frame(raw=None, features=None, verbose=False):
        return {"ufc": ufc, "X": X, "y": y, "feature_cols": list(X.columns)}

    def fake_walk_forward(X_, y_, ufc_, verbose=True, **recipe):
        calls["fit"] += 1
        return pd.DataFrame({"fight_id": ufc_["fight_id"], "date": ufc_["date"],
                             "year": ufc_["date"].dt.year, "event": ufc_["event_name"],
                             "red": ufc_["r_name"], "blue": ufc_["b_name"],
                             "red_norm": ufc_["r_name"].str.lower(),
                             "blue_norm": ufc_["b_name"].str.lower(),
                             "y": y_, "p_model": 0.5, "p_ensemble": 0.5})

    monkeypatch.setattr(mc, "build_frame", fake_build_frame)
    monkeypatch.setattr(mc, "walk_forward", fake_walk_forward)
    columns = ["date", "fighter_a", "fighter_b", "odds_a", "odds_b"]
    odds = tmp_path / "odds.csv"
    pd.DataFrame(columns=columns).to_csv(odds, index=False)
    before = mc.predictions(recipe="legacy", label="t", cache_dir=tmp_path, odds_path=odds,
                            verbose=False)
    assert calls["fit"] == 1 and np.isnan(before["p_market"]).all()
    # the odds file is backfilled for one fight: same archive, same cache
    row = ufc.iloc[5]
    pd.DataFrame([[row["date"].strftime("%Y-%m-%d"), row["r_name"], row["b_name"], -150, 130]],
                 columns=columns).to_csv(odds, index=False)
    after = mc.predictions(recipe="legacy", label="t", cache_dir=tmp_path, odds_path=odds,
                           verbose=False)
    assert calls["fit"] == 1, "a changed odds file is not a reason to refit"
    assert np.isfinite(after["p_market"]).sum() == 1
    assert after.loc[5, "p_market"] == pytest.approx(
        __import__("market_blend").devig(-150, 130))
    assert after.attrs["odds_sha256"] != before.attrs["odds_sha256"]
