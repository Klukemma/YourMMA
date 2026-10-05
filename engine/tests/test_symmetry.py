"""Corner symmetry: the swapped row, the mirror audit and the experiment's bar.

corner_swap.swapped_matrix is the row production would build for a fight
listed the other way round, so it must equal, feature for feature, the row
live_rows builds for the swapped fight - including the three features that
count a corner's earlier appearances and so do NOT mirror under a
whole-archive swap. The experiment's walk-forward must use exactly
model_compare's splits, its training set for arm (B) must be the doubled
block with flipped labels and duplicated weights, and the bar must be the
pre-registered one. Synthetic data where the archive is not needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corner_swap  # noqa: E402
import feature_frame  # noqa: E402
import pending_rows  # noqa: E402
from experiments import model_compare as mc  # noqa: E402
from experiments import symmetry as sy  # noqa: E402
from live_rows import LiveRows  # noqa: E402
from test_model_compare import synthetic  # noqa: E402

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"


# --- the mirror (fast) -----------------------------------------------------

def test_mirror_archive_exchanges_every_corner_column_and_nothing_else():
    raw = pd.DataFrame({"date": ["2020-01-01"], "r_name": ["A"], "b_name": ["B"],
                        "r_kd": [1], "b_kd": [2], "winner": ["B"], "winner_id": ["b"]})
    m = corner_swap.mirror_archive(raw)
    assert list(m.columns) == list(raw.columns)
    assert m["r_name"].iloc[0] == "B" and m["b_name"].iloc[0] == "A"
    assert m["r_kd"].iloc[0] == 2 and m["b_kd"].iloc[0] == 1
    assert m["winner"].iloc[0] == "B" and m["date"].iloc[0] == "2020-01-01"
    with pytest.raises(KeyError, match="no corner twin"):
        corner_swap.mirror_archive(raw.drop(columns=["b_kd"]))


def test_corner_appearances_count_the_other_corners_history_in_row_order():
    red = ["A", "B", "A", "C", "B"]
    blue = ["B", "C", "C", "A", "A"]
    red_of_blue, blue_of_red = corner_swap.corner_appearances(red, blue)
    # row 3: blue is A, who was red twice before; red is C, blue twice before
    assert list(red_of_blue) == [0, 0, 0, 2, 2]
    assert list(blue_of_red) == [0, 1, 0, 2, 1]


def test_corner_history_features_use_the_pipelines_own_arithmetic():
    ufc = pd.DataFrame({"r_name": ["A", "B"], "b_name": ["B", "A"],
                        "r_sapm": [4.0, np.nan], "b_sapm": [2.0, 5.0]})
    out = corner_swap.corner_history_features(ufc)
    # row 1 swapped: red is A (red once before), blue is B (blue once before)
    assert out.loc[1, "data_sparsity_diff"] == pytest.approx(np.log1p(1) - np.log1p(1))
    r_dl, b_dl = np.log1p(5.0 * 12 * 1), np.log1p(3.0 * 12 * 1)   # b_sapm 5, r_sapm NaN -> 3
    assert out.loc[1, "career_damage_diff"] == pytest.approx(r_dl - b_dl)
    assert out.loc[1, "career_damage_level"] == pytest.approx((r_dl + b_dl) / 2)


def test_symmetrise_is_the_average_of_a_fight_and_one_minus_its_swap():
    assert list(corner_swap.symmetrise([0.7, 0.2], [0.3, 0.9])) == pytest.approx([0.7, 0.15])


def test_classify_mirror_names_each_transform_and_flags_what_is_not_clean():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"same": rng.normal(size=50), "neg": rng.normal(size=50),
                      "prob": rng.uniform(size=50), "r_a_vs_b_c": rng.normal(size=50),
                      "b_a_vs_r_c": rng.normal(size=50), "r_t": rng.normal(size=50),
                      "b_t": rng.normal(size=50), "dirty": rng.normal(size=50)})
    M = pd.DataFrame({"same": X["same"], "neg": -X["neg"], "prob": 1 - X["prob"],
                      "r_a_vs_b_c": X["b_a_vs_r_c"], "b_a_vs_r_c": X["r_a_vs_b_c"],
                      "r_t": X["b_t"], "b_t": X["r_t"], "dirty": -X["dirty"] + 0.01})
    out = sy.classify_mirror(X, M).set_index("feature")
    assert out.loc["same", "transform"] == "unchanged"
    assert out.loc["neg", "transform"] == "negated"
    assert out.loc["prob", "transform"] == "1-x"
    assert out.loc["r_a_vs_b_c", "transform"] == "swapped<->b_a_vs_r_c"
    assert out.loc["b_t", "transform"] == "swapped<->r_t"
    assert out.loc["dirty", "transform"].startswith("NOT CLEAN (nearest negated: 100.0%")
    assert out["clean"].tolist() == [True] * 7 + [False]


# --- the mirror (real archive) ----------------------------------------------

@pytest.fixture(scope="module")
def raw():
    if not ARCHIVE.exists():
        pytest.skip("archive not present")
    return pd.read_csv(ARCHIVE, low_memory=False)


@pytest.fixture(scope="module")
def cut(raw):
    """The archive through 2009, built as training would build it."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    cut = raw[days < pd.Timestamp("2010-01-01")].reset_index(drop=True)
    return cut, feature_frame.build(cut, verbose=False)


@pytest.fixture(scope="module")
def audit(cut):
    raw_cut, built = cut
    return sy.mirror_audit(raw_cut, built, verbose=False)


def test_every_feature_is_a_clean_mirror_under_a_whole_archive_swap(audit):
    """With the centres fitted on the swapped frame: every feature is
    unchanged, negated, 1 - x or exchanged with its twin, and the target
    flips on every valid row. The eight interactions are exact twins here
    and not with frozen centres (the centres are corner-specific); the
    three corner-history features mirror here only because the history is
    mirrored with them - the live tests below are what catch those."""
    assert audit["free_clean"].all(), audit.loc[~audit["free_clean"], ["feature", "free_centres"]]
    # a constant column (none of the cut's fights are five-rounders, say)
    # fits more than one transform; the first listed is the one counted
    kinds = audit["free_centres"].str.split(" \\| ").str[0].str.split("<->").str[0]
    assert set(kinds) == {"unchanged", "negated", "1-x", "swapped"}
    assert set(audit.loc[kinds == "1-x", "feature"]) == {"bayesian_prob", "base_prob"}
    assert (kinds == "swapped").sum() == 12
    assert set(audit.loc[~audit["frozen_clean"], "feature"]) == {
        "r_td_vs_b_tdd", "b_td_vs_r_tdd", "r_strike_vs_b_def", "b_strike_vs_r_def",
        "r_sub_vs_b_subdef", "b_sub_vs_r_subdef", "r_power_vs_b_chin", "b_power_vs_r_chin"}
    assert set(audit.loc[audit["corner_history"], "feature"]) == set(corner_swap.CORNER_HISTORY)
    for col in corner_swap.CORNER_HISTORY:
        assert audit.set_index("feature").loc[col, "free_centres"] in ("negated", "unchanged")


def test_the_swapped_matrix_differs_from_the_naive_mirror_only_on_corner_history(cut):
    raw_cut, built = cut
    X_swap = corner_swap.swapped_matrix(raw_cut, built)
    mirrored = feature_frame.build(corner_swap.mirror_archive(raw_cut),
                                   centres=built["INTERACTION_CENTRES"], verbose=False)["X"]
    assert list(X_swap.columns) == built["feature_cols"] and len(X_swap) == len(built["X"])
    same = np.isclose(X_swap.to_numpy(float), mirrored[built["feature_cols"]].to_numpy(float),
                      rtol=1e-9, atol=1e-12).all(axis=0)
    assert set(np.array(built["feature_cols"])[~same]) == set(corner_swap.CORNER_HISTORY)
    for col in corner_swap.CORNER_HISTORY:          # and they really do differ
        assert not np.allclose(X_swap[col], mirrored[col])


@pytest.fixture(scope="module")
def full(raw):
    return feature_frame.build(raw, verbose=False)


@pytest.fixture(scope="module")
def swapped(raw, full):
    return corner_swap.swapped_matrix(raw, full)


def _live_card(raw, full, when):
    """The card of `when` through LiveRows, listed and swapped, as
    predict_card's prefetch builds it: the LiveRows, the fights built and
    the rows (listed, swapped, training index) of each. Fights with a
    corner the archive before the date cannot resolve are skipped, as the
    live path refuses them."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    before, event = raw[days < when], raw[days == when]
    live = LiveRows(raw, full["X"].set_index(full["ufc"]["fight_id"].astype(str)),
                    full["INTERACTION_CENTRES"])
    candidates = pending_rows.fighter_candidates(before)
    history = pending_rows.division_history(before)
    fu = full["ufc"]
    card = []
    for r in event.itertuples():
        try:
            for name in (r.r_name, r.b_name):
                pending_rows.resolve_fighter(name, r.division, candidates, history)
        except KeyError:
            continue
        card.append(r)
    assert len(card) >= 5
    # the card prefetch as predict_card runs it: listed and swapped fights,
    # two passes (a swapped fight holds the same two fighters as its listed one)
    specs = live.prepare_card([(r.r_name, r.b_name, r.total_rounds == 5, bool(r.title_fight))
                               for r in card], when,
                              {i: {"division": r.division} for i, r in enumerate(card)},
                              swapped=True)
    assert len(specs) == 2 * len(card) and live.passes == 2 and not live.refused
    assert specs[len(card)]["red"] == card[0].b_name and specs[len(card)]["blue"] == card[0].r_name
    rows = []
    for r in card:
        swapped_row = live.row(r.b_name, r.r_name, when, r.total_rounds == 5,
                               bool(r.title_fight), {"division": r.division})
        listed_row = live.row(r.r_name, r.b_name, when, r.total_rounds == 5,
                              bool(r.title_fight), {"division": r.division})
        j = fu.index[(fu["date"] == when) & (fu["r_name"] == r.r_name)
                     & (fu["b_name"] == r.b_name)][0]
        rows.append((r, listed_row, swapped_row, j))
    assert live.passes == 2
    return live, rows


def _differing(cols, mine, theirs):
    return [c for c, a, b in zip(cols, mine, theirs)
            if not (np.isclose(a, b, rtol=1e-9, atol=1e-12) or (np.isnan(a) and np.isnan(b)))]


def test_the_live_swapped_row_equals_the_swapped_matrix_row_on_every_feature(raw, full, swapped):
    """The LAST archived event, each fight asked for with the corners
    exchanged through LiveRows - exactly what a symmetrised production
    prediction builds - against corner_swap.swapped_matrix's row for the
    same fight: all 151 features, the corner-history three included. This
    equality is exact only here, where the archive rows were appended by
    the same code a replay runs; the historical-date test below says what
    holds earlier."""
    when = pd.to_datetime(raw["date"], errors="coerce").max()
    _, rows = _live_card(raw, full, when)
    cols = full["feature_cols"]
    for r, listed, swapped_row, j in rows:
        bad = _differing(cols, swapped_row[cols].to_numpy(float)[0],
                         swapped.loc[j, cols].to_numpy(float))
        assert bad == [], f"{r.b_name} vs {r.r_name} (swapped): {bad}"
        # and the swapped row is not the listed row: the corner-history
        # features read the other corner's count, the differences flip
        assert swapped_row["mu_diff"].iloc[0] == pytest.approx(-listed["mu_diff"].iloc[0])


@pytest.fixture(scope="module")
def full_kinds(raw, full):
    """How each feature transforms under the whole-archive swap with free
    centres, classified on the full archive (a constant column may list
    more than one transform; any of them counts)."""
    free = feature_frame.build(corner_swap.mirror_archive(raw), verbose=False)
    out = sy.classify_mirror(full["X"], free["X"], full["feature_cols"]).set_index("feature")
    assert out["clean"].all()
    return out["transform"].str.split(" \\| ").to_dict()


FROZEN_CENTRE_INTERACTIONS = (
    "r_td_vs_b_tdd", "b_td_vs_r_tdd", "r_strike_vs_b_def", "b_strike_vs_r_def",
    "r_sub_vs_b_subdef", "b_sub_vs_r_subdef", "r_power_vs_b_chin", "b_power_vs_r_chin")


def test_on_a_historical_date_the_swapped_live_row_is_the_mirror_of_the_listed_one(raw, full, swapped, full_kinds):
    """An event five years back. The archive's shipped rating series are
    not reproduced bit for bit by a replay (live_parity.json lists
    mx_mass_adv), so neither orientation of the live row equals its
    training row on the rating-derived features - a drift that predates
    the swap and must hit both orientations identically. What the swap
    itself guarantees, and what is checked: (i) the swapped live row is
    the exact mirror of the listed live row (negated, unchanged, 1 - x or
    the twin, as the whole-archive audit classified each feature) on every
    feature outside the corner-history three and the eight frozen-centre
    interactions; (ii) the corner-history three equal swapped_matrix's
    exactly; (iii) the features on which the swapped live row differs from
    swapped_matrix are the ones on which the listed live row differs from
    the training matrix, by the same amounts."""
    days = pd.to_datetime(raw["date"], errors="coerce")
    when = max(d for d in days.dropna().unique() if d <= days.max() - pd.DateOffset(years=5))
    _, rows = _live_card(raw, full, when)
    cols = full["feature_cols"]
    exempt = set(corner_swap.CORNER_HISTORY) | set(FROZEN_CENTRE_INTERACTIONS)
    assert exempt <= set(cols) and len(exempt) == 11

    def mirror_of(li, col, kind):
        if kind.startswith("swapped<->"):
            return li[kind.split("<->")[1]]
        return {"unchanged": li[col], "negated": -li[col], "1-x": 1 - li[col]}[kind]

    for r, listed, swapped_row, j in rows:
        li, sw = listed[cols].iloc[0], swapped_row[cols].iloc[0]
        not_mirrored = []
        for col in cols:
            if col in exempt:
                continue
            expected = [mirror_of(li, col, kind) for kind in full_kinds[col]]
            if not any(np.isclose(sw[col], e, rtol=1e-9, atol=1e-9) for e in expected):
                not_mirrored.append((col, full_kinds[col], float(sw[col]), expected))
        assert not_mirrored == [], f"{r.b_name} vs {r.r_name}: {not_mirrored}"
        for col in corner_swap.CORNER_HISTORY:
            assert sw[col] == pytest.approx(swapped.loc[j, col], rel=1e-9, abs=1e-12), col
        drift_swapped = _differing(cols, sw.to_numpy(float), swapped.loc[j, cols].to_numpy(float))
        drift_listed = _differing(cols, li.to_numpy(float), full["X"].loc[j, cols].to_numpy(float))
        assert drift_swapped == drift_listed, (r.b_name, r.r_name, drift_swapped, drift_listed)
        assert not set(drift_listed) & exempt
        for col in drift_listed:
            assert abs(sw[col] - swapped.loc[j, col]) == pytest.approx(
                abs(li[col] - full["X"].loc[j, col]), rel=1e-6, abs=1e-9), col


# --- the arms (fast) --------------------------------------------------------

def test_the_symmetrised_ensemble_sums_to_one_over_the_two_orientations():
    ufc, X, y = synthetic(n_per_year=150, years=range(2008, 2012), seed=2)
    X_swap = -X
    models = mc.fit_production(X, y, ufc["date"])
    platt = sy.symmetric_platt(models, X.to_numpy(), X_swap.to_numpy(), y)
    e, p = sy.predict_symmetric(models, platt, X.to_numpy(), X_swap.to_numpy())
    e2, p2 = sy.predict_symmetric(models, platt, X_swap.to_numpy(), X.to_numpy())
    assert np.allclose(e + e2, 1.0)
    assert not np.allclose(p, models["platt"].predict_proba(
        mc._average(models["scaler"], models["lr"], models["rf"], models["xgb"], X).reshape(-1, 1))[:, 1])
    # the calibrator may keep a corner prior: p(x) + p(x_swap) is 1 only when
    # platt(0.5) is 0.5
    assert abs(sy.platt_at_half(platt) - 0.5) < 0.5


def test_fit_augmented_doubles_the_training_block_with_flipped_labels_and_duplicated_weights(monkeypatch):
    ufc, X, y = synthetic(n_per_year=100, years=range(2008, 2012), seed=4)
    X_swap = -X
    seen = {}
    real_lr = sy.LogisticRegression

    class SpyLR(real_lr):
        def fit(self, Xs, ys, sample_weight=None):
            if self.C == 0.1:
                seen["X"], seen["y"], seen["w"] = np.asarray(Xs), np.asarray(ys), np.asarray(sample_weight)
            return super().fit(Xs, ys, sample_weight=sample_weight)

    monkeypatch.setattr(sy, "LogisticRegression", SpyLR)
    models = sy.fit_augmented(X, X_swap, y, ufc["date"])
    base = mc.fit_production(X, y, ufc["date"])
    n_tr = base["train_rows"]
    assert models["train_rows"] == n_tr and models["augmented_rows"] == 2 * n_tr
    assert seen["X"].shape == (2 * n_tr, X.shape[1])
    assert np.array_equal(seen["y"][:n_tr], y[:n_tr]) and np.array_equal(seen["y"][n_tr:], 1 - y[:n_tr])
    assert np.allclose(seen["w"][:n_tr], seen["w"][n_tr:])
    assert np.allclose(seen["w"][:n_tr], mc.recency_weights(ufc["date"].iloc[:n_tr]))
    # scaled on the doubled block, the swapped copy is the negation of the listed one
    assert np.allclose(seen["X"][:n_tr], -seen["X"][n_tr:])
    # and the LR it produces is symmetric: p(x) + p(-x) == 1
    Xs = models["scaler"].transform(X)
    assert np.allclose(models["lr"].predict_proba(Xs)[:, 1]
                       + models["lr"].predict_proba(-Xs)[:, 1], 1.0, atol=1e-6)


def test_the_walk_forward_uses_model_compares_splits_exactly():
    ufc, X, y = synthetic(n_per_year=120, years=range(2008, 2016), seed=0)
    reference, mine = [], {"base": [], "aug": []}

    def spy_ref(X_train, y_train, dates):
        reference.append((pd.to_datetime(pd.Series(dates)).max(), len(X_train)))
        return None

    mc.walk_forward(X, y, ufc, fitter=spy_ref,
                    predictor=lambda m, X_: (np.full(len(X_), 0.5),) * 2,
                    first_year=2010, min_train=100, min_test=10, verbose=False)

    def fake_models(X_train, y_train, dates):
        m = mc.fit_production(X_train, y_train, dates)
        mine["base"].append((pd.to_datetime(pd.Series(dates)).max(), len(X_train)))
        return m

    def fake_aug(X_train, X_swap_train, y_train, dates):
        m = mc.fit_production(X_train, y_train, dates)
        mine["aug"].append((pd.to_datetime(pd.Series(dates)).max(), len(X_train), len(X_swap_train)))
        return m

    out, by_year = sy.walk_forward(X, -X, y, ufc, fit_base=fake_models, fit_aug=fake_aug,
                                   first_year=2010, min_train=100, min_test=10, verbose=False)
    assert mine["base"] == reference
    assert [(d, n) for d, n, _ in mine["aug"]] == reference
    assert all(n == m for _, n, m in mine["aug"])
    assert sorted(out["year"].unique()) == list(range(2010, 2016)) == sorted(by_year)
    for year in out["year"].unique():
        assert (pd.to_datetime(out.loc[out["year"] == year, "date"]).dt.year == year).all()
        assert by_year[year]["train_fights"] == int((ufc["date"].dt.year < year).sum())


def test_model_compares_shipped_recipe_is_arm_b_to_the_bit():
    """The harness baseline (model_compare.predictions(), recipe
    "symmetric") must be this file's arm B and not a second implementation
    of it: the same walk-forward, year for year, through model_compare's
    loop on the paired matrix gives the same probabilities."""
    ufc, X, y = synthetic(n_per_year=120, years=range(2008, 2014), seed=0)
    arms, _ = sy.walk_forward(X, -X, y, ufc, first_year=2011, min_train=100, min_test=10,
                              verbose=False)
    mine = mc.walk_forward(mc.paired(X, -X), y, ufc, fitter=mc.fit_symmetric,
                           predictor=mc.predict_symmetric, first_year=2011, min_train=100,
                           min_test=10, verbose=False)
    assert list(mine["fight_id"]) == list(arms["fight_id"])
    assert np.allclose(mine["p_ensemble"], arms["e_B"], rtol=0, atol=1e-12)
    assert np.allclose(mine["p_model"], arms["p_B"], rtol=0, atol=1e-12)
    assert mc.RECIPES["symmetric"]["fit"] is mc.fit_symmetric


def test_frames_takes_a_feature_function_and_needs_a_swap_for_columns_of_its_own(monkeypatch):
    ufc, X, y = synthetic(n_per_year=10, years=range(2010, 2012))
    built = {"ufc": ufc.assign(target_win=y), "X": X, "feature_cols": list(X.columns),
             "feature_cols_winner": ["f0", "f1"], "INTERACTION_CENTRES": {}}

    def fake_build_frame(raw, features=None, verbose=False):
        Xf = features(built)
        return {"ufc": built["ufc"], "X": Xf, "y": y, "feature_cols": list(Xf.columns)}

    monkeypatch.setattr(mc, "build_frame", fake_build_frame)
    monkeypatch.setattr(corner_swap, "swapped_matrix", lambda raw, b: -X)
    fr = sy.frames(raw=pd.DataFrame(), verbose=False)
    assert fr["feature_cols"] == ["f0", "f1"] and np.allclose(fr["X_swap"], -X[["f0", "f1"]])
    fr = sy.frames(raw=pd.DataFrame(), features=lambda b: b["X"][["f2"]], verbose=False)
    assert fr["feature_cols"] == ["f2"] and np.allclose(fr["X_swap"], -X[["f2"]])
    with pytest.raises(KeyError, match="swap_features"):
        sy.frames(raw=pd.DataFrame(), features=lambda b: b["X"].assign(new=1.0), verbose=False)
    fr = sy.frames(raw=pd.DataFrame(), features=lambda b: b["X"].assign(new=1.0),
                   swap_features=lambda raw, b: b["X"].assign(new=-1.0), verbose=False)
    assert list(fr["X_swap"]["new"]) == [-1.0] * len(X)


# --- the bar ---------------------------------------------------------------

def test_passes_bar_needs_the_model_interval_below_zero_and_the_blend_not_worse():
    good = {"mean": -0.004, "low": -0.009, "high": -0.001}
    assert sy.passes_bar(good, {"mean": -0.001})
    assert sy.passes_bar(good, {"mean": 0.0})
    assert not sy.passes_bar(good, {"mean": +0.0001})
    assert not sy.passes_bar({"mean": -0.004, "low": -0.009, "high": 0.0}, {"mean": -0.01})
    assert not sy.passes_bar({"mean": -0.004, "low": -0.009, "high": +0.002}, {"mean": -0.01})


def test_red_bias_is_mean_prediction_minus_observed_red_win_rate():
    assert sy.red_bias([1, 0, 1, 1], [0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.6 - 0.75)


def test_the_registered_plan_hash_matches_the_docstring_and_the_reported_json():
    assert sy.FAMILY == 2 and sy.ALPHA == pytest.approx(0.025)
    if not sy.PLAN_FILE.exists():
        pytest.skip("plan not registered yet")
    registered = json.loads(sy.PLAN_FILE.read_text())
    assert registered["sha256"] == sy.plan_sha256()
    if sy.OUT.exists():
        reported = json.loads(sy.OUT.read_text())
        assert reported["plan"] == sy.__doc__
        assert sy.plan_text(reported["plan"]) == sy.plan_text()
        assert reported["plan_sha256"] == registered["sha256"]
        assert reported["family"] == 2 and reported["alpha"] == pytest.approx(0.025)
        for name, arm in reported["primary"]["arms"].items():
            assert arm["passes"] == (arm["delta"]["log_loss"]["high"] < 0
                                     and arm["blend"]["delta"]["log_loss"]["mean"] <= 0)
        assert reported["passing"] == [a for a, arm in reported["primary"]["arms"].items()
                                       if arm["passes"]]


def test_check_plan_registers_once_and_refuses_a_changed_plan(tmp_path):
    sidecar = tmp_path / "plan.sha256"
    first = sy.check_plan(sidecar)
    assert first["sha256"] == sy.plan_sha256()
    assert sy.check_plan(sidecar)["sha256"] == first["sha256"]
    moved = sy.__doc__.replace("alpha = 0.05 / 2", "alpha = 0.05")
    assert sy.plan_text(moved) != sy.plan_text()
    with pytest.raises(SystemExit):
        sy.check_plan(sidecar, doc=moved)
    annotated = sy.__doc__ + "\nmore notes\n"
    assert sy.check_plan(sidecar, doc=annotated)["sha256"] == first["sha256"]
