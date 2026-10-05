"""Idea #6: does enforcing corner symmetry beat the model as listed?

The red corner is usually the favoured or ranked fighter - it has won 58%
of post-2001 UFC bouts - so a model fitted on fights as listed can learn
corner identity as a shortcut, and nothing forces p(A beats B) to equal
1 - p(B beats A). This measures two ways of enforcing that, against the
production recipe, with the shared walk-forward of model_compare.py.

The swapped row of a fight comes from corner_swap.py: the mirrored archive
through feature_frame.build with the training centres frozen, and the
three features that count a corner's earlier appearances recomputed on the
unchanged history - the row live_rows builds for the swapped fight, equal
to it on every feature on the last archived event and, on earlier dates,
up to the known rating-replay drift that hits both orientations alike
(tests/test_symmetry.py). The mirror audit over every feature is written
beside this file as symmetry_mirror_audit.csv.

PRE-REGISTERED PLAN - written before the first run and reported unchanged.

Rows. The walk-forward of experiments/model_compare.py: every valid fight
of a predicted year (first predicted year 2011, at least 500 earlier
fights, at least 20 in the year), predicted by models fitted on the valid
fights of earlier years only, with the production recipe (LR + RF + XGB
averaged, the last 10% of the fitting rows by date calibrating, recency
weights, Platt), features feature_cols_winner. The swapped row of a fight
is corner_swap.swapped_matrix: the same fight with the corners exchanged
and the archive's history unchanged, interaction centres frozen at the
training values.

Baseline arm. model_compare's cached baseline predictions (p_model), the
production recipe on rows as listed, unchanged. Arm (A) refits the same
models with the same recipe; its own un-symmetrised prediction is compared
with the cache and the largest difference reported, but the comparison is
against the cache.

Candidate arms - a family of two, so alpha = 0.05 / 2:

  (A) swap-averaged prediction. Each year's baseline models, unchanged.
      The ensemble output is symmetrised, e_sym(x) = (e(x) + 1 - e(x_swap))
      / 2, a Platt calibrator (LogisticRegression, C = 1e10) is fitted on
      the same year's calibration block - one row per fight, in the
      orientation listed, as production fits it - on e_sym, and
      p_A = platt(e_sym(x)).
  (B) swap-augmented training. Each year's LR, RF and XGB fitted on the
      training block twice over: every fight as listed with its label, and
      swapped with 1 - label, the fight's recency weight on both copies;
      the scaler fitted on the doubled block; XGB early-stopping on the
      doubled calibration block. The split into training and calibration
      fights is the baseline's (the last 10% of fights by date). The
      prediction is symmetrised and Platt-calibrated exactly as in (A).

Comparison set and bar. PRIMARY: the priced fights (odds.csv prices the
fight; de-vigged closing line, matched as model_compare does) of every
predicted year, identical for every arm and the baseline. Metric: the
model-only log loss. An arm PASSES only if BOTH
  (i)  the wider of the event-resampled and fighter-resampled bootstrap
       intervals of (arm - baseline) log loss, at the family-wise level
       1 - 0.05/2, lies entirely below zero, and
  (ii) the 0.75 market blend of the arm (market_blend.blend, probability
       space, clipped) has a log loss no worse than the blend of the
       baseline on the same fights: the point estimate of
       (arm blend - baseline blend) log loss is <= 0.
Nothing else is a pass.

Secondary, reported, never the bar: accuracy and Brier on the priced set;
model-only log loss, accuracy and Brier on ALL predicted fights (priced and
unpriced); every table again on 2020 onward; the red-corner bias, mean
p_red minus the observed red win rate, for the baseline and each arm on
the priced set and on all fights; log loss by year; and the calibrator's
output at an ensemble value of 0.5 (the corner prior the Platt keeps) by
year. Calibration by decile is recorded in the JSON.

Shipping. Only a passing arm goes to production. (A): predict_fight_prod
builds the swapped live row through live_rows, averages the ensemble
output of the fight with 1 - that of the swapped fight, and calibrates
with a Platt fitted in SECTION 15 on the symmetrised ensemble output of the
calibration block. (B): additionally SECTION 15 fits the winner models on
the doubled training block as above. Method and round models are not
touched. If no arm passes, nothing ships and this file says so.

POST-REGISTRATION NOTES - added after the first run; nothing above this
line changed (its sha256 is in symmetry_plan.sha256 and main() refuses to
run if it moves).

  * Result (first and only registered run, 2026-10-01): BOTH arms pass on
    the primary set (5,897 priced fights, 2011-2026). Model-only log loss
    0.6441 baseline; A 0.6394 (-0.0047, wider interval [-0.0088, -0.0007]);
    B 0.6301 (-0.0140, [-0.0199, -0.0074]). Blend 0.6064 -> A 0.6049
    (-0.0014), B 0.6034 (-0.0030, [-0.0045, -0.0015]). Accuracy 62.1% ->
    63.1% for both. The plan did not say which of two passing arms ships;
    B, the lower log loss, was taken, as stacked_blend.py's rule would.
  * Honest caveats. The gain is front-loaded: on the 2020+ secondary set B
    is -0.0045 model-only with an interval [-0.0104, +0.0014] that crosses
    zero, and the blend gain there is -0.0005. 2019 is worse for both arms.
    The pre-registered bar was all predicted years, and it is met; a
    reader who weights the recent years should know they do not carry it
    alone.
  * The Platt keeps a corner prior: platt(0.5) runs 0.55-0.64 for A and
    0.49-0.61 for B by year, so p(red, blue) + p(blue, red) is not exactly
    1. The ensemble itself is exactly symmetric; the red-corner bias on the
    priced set is -0.0055 baseline, -0.0020 A, -0.0022 B.
  * The baseline refit reproduces the cached predictions to 5e-7 (the
    cache is written at six decimals), so arm A really did use the
    baseline's models.
  * Shipped as production (predict_card.py SECTION 15, predict_fight_prod,
    the SECTION 6 and 9 printouts, live_rows.prepare_card(swapped=True)).
    The shared harness followed: model_compare.predictions() now fits
    arm B by default (recipe "symmetric", importing fit_augmented,
    symmetric_platt and predict_symmetric from here, cached as
    baseline_symmetric and verified to reproduce this file's arm-B
    numbers), and the recipe as listed stays reachable as
    recipe="legacy" (baseline_legacy - the file this run was measured
    against, renamed, hash untouched). main() here keeps using the
    legacy baseline, because the plan's baseline is the recipe as
    listed. Later ideas pre-register their PRIMARY bar on
    model_compare.decision_set (training sets of 4,000+ fights, 2018
    onward), every year a secondary - the lesson of the front-loaded
    gain above.
  * The priced set has NO 2025 fights. odds.csv carries closing lines and
    no source prices 2025 that way (idea #7: only opening lines exist for
    2025, kept apart in opening_odds.csv and never used as closing), so
    the 513 fights predicted for 2025 are all unpriced: they count in the
    all-fights secondaries only, and "2020 onward" on the priced set is
    2020-2024 plus 2026. The primary table is "every predicted year that
    odds.csv prices", 15 years. Review pointed this out after the run; the
    bar is decided and is not re-decided here. When closing lines for 2025
    land, rerun this file and read the 2025 row of the priced by-year
    table as a secondary. Model-only log loss by year on ALL fights,
    including 2025, is reported below (log_loss_by_year_all in the JSON)
    so the missing year is at least visible model-only. What it says:
    2025 (513 fights) baseline 0.6435, A 0.6413, B 0.6444 - the shipped
    arm is 0.0009 WORSE there; over the last four years B is worse on all
    fights in 2023 (+0.0016), 2025 (+0.0009) and 2026 (+0.0018) and better
    in 2024 (-0.0110). That is consistent with the 2020+ intervals that
    cross zero: the recent regime does not carry the gain alone, and the
    year nearest production is a small loss. The registered bar is met on
    the registered set and the decision stands; this is reported so that
    nobody reads the headline as the recent-regime expectation.
  * "The live swapped row equals the swapped training row on every
    feature" is proven on the LAST archived event. On earlier dates both
    orientations of the live row differ from their training rows on the
    same rating-derived features by the same amounts (the archive's
    shipped TrueSkill/MMR/mass series are not reproduced bit for bit by a
    replay - live_parity.json already lists mx_mass_adv), and the swapped
    live row is the exact mirror of the listed live row on every feature
    outside the three corner-history ones and the eight frozen-centre
    interactions. tests/test_symmetry.py checks both.

    python engine/experiments/symmetry.py
"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import corner_swap  # noqa: E402
import feature_frame  # noqa: E402
import market_blend  # noqa: E402
from experiments import model_compare as mc  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
PLAN_FILE = Path(__file__).with_name("symmetry_plan.sha256")
AUDIT_FILE = Path(__file__).with_name("symmetry_mirror_audit.csv")
OPENING_CSV = ENGINE / "data" / "opening_odds.csv"      # OPENING lines, 2010-2025
FAMILY = 2
ALPHA = 0.05 / FAMILY
CONFIRM_FROM = 2020
ARMS = ("A", "B")
ARM_NAMES = {"A": "A swap-averaged", "B": "B swap-augmented"}
PLAN_START = "PRE-REGISTERED PLAN"
PLAN_END = "POST-REGISTRATION NOTES"


# --------------------------------------------------------------------------
# pre-registration
# --------------------------------------------------------------------------

def plan_text(doc=None):
    doc = __doc__ if doc is None else doc
    return doc[doc.index(PLAN_START):doc.index(PLAN_END)]


def plan_sha256(doc=None):
    return hashlib.sha256(plan_text(doc).encode()).hexdigest()


def check_plan(path=PLAN_FILE, doc=None):
    """Register the plan's hash the first time; refuse to run if it has
    moved since."""
    current = plan_sha256(doc)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["sha256"] != current:
            raise SystemExit(
                f"the pre-registered plan has changed since it was registered "
                f"({stored['registered_at']}): {stored['sha256'][:12]} -> {current[:12]}. "
                f"Restore the plan, or register a new experiment under a new name.")
        return stored
    stored = {"sha256": current,
              "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.write_text(json.dumps(stored, indent=1) + "\n")
    return stored


# --------------------------------------------------------------------------
# the mirror audit
# --------------------------------------------------------------------------

def _twin(col, cols):
    """The column a corner-prefixed feature exchanges with under a swap:
    r_striking_trajectory <-> b_striking_trajectory, r_td_vs_b_tdd <->
    b_td_vs_r_tdd. None when there is no such column."""
    if col.startswith("r_"):
        for cand in ("b_" + col[2:], "b_" + col[2:].replace("_b_", "_r_")):
            if cand in cols and cand != col:
                return cand
    if col.startswith("b_"):
        for cand in ("r_" + col[2:], "r_" + col[2:].replace("_r_", "_b_")):
            if cand in cols and cand != col:
                return cand
    return None


def classify_mirror(X, X_mirror, cols=None, rtol=1e-9, atol=1e-9):
    """How each feature transforms under the corner swap: 'unchanged',
    'negated', '1-x', 'swapped<->twin', or 'NOT CLEAN' with the share of
    rows that differ from the nearest of those and the largest gap. A
    feature that fits more than one (a constant column) lists them all."""
    cols = list(X.columns) if cols is None else list(cols)
    rows = []
    for col in cols:
        o, m = X[col].to_numpy(float), X_mirror[col].to_numpy(float)
        kinds = []
        if np.allclose(m, o, rtol=rtol, atol=atol):
            kinds.append("unchanged")
        if np.allclose(m, -o, rtol=rtol, atol=atol):
            kinds.append("negated")
        if np.allclose(m, 1.0 - o, rtol=rtol, atol=atol):
            kinds.append("1-x")
        twin = _twin(col, cols)
        if twin is not None and np.allclose(m, X[twin].to_numpy(float), rtol=rtol, atol=atol):
            kinds.append(f"swapped<->{twin}")
        clean = bool(kinds)
        if not clean:
            candidates = {"unchanged": o, "negated": -o, "1-x": 1.0 - o}
            if twin is not None:
                candidates[f"swapped<->{twin}"] = X[twin].to_numpy(float)
            nearest = min(candidates, key=lambda k: np.abs(m - candidates[k]).mean())
            bad = ~np.isclose(m, candidates[nearest], rtol=rtol, atol=atol)
            kinds.append(f"NOT CLEAN (nearest {nearest}: {bad.mean():.1%} of rows differ, "
                         f"max |gap| {np.abs(m - candidates[nearest]).max():.4g})")
        rows.append({"feature": col, "transform": " | ".join(kinds), "clean": clean})
    return pd.DataFrame(rows)


def mirror_audit(raw, built, verbose=True):
    """The table written as symmetry_mirror_audit.csv: every feature under
    the whole-archive swap with centres fitted on the swapped frame
    (self-consistent mirror) and with the training centres frozen (the row
    production can build), and whether the live swapped row agrees with it
    on the unchanged history (corner_swap.CORNER_HISTORY does not)."""
    mirrored = corner_swap.mirror_archive(raw)
    free = feature_frame.build(mirrored, verbose=False)
    frozen = feature_frame.build(mirrored, centres=built["INTERACTION_CENTRES"], verbose=False)
    cols = built["feature_cols"]
    for other in (free, frozen):
        if other["ufc"]["fight_id"].astype(str).tolist() != built["ufc"]["fight_id"].astype(str).tolist():
            raise RuntimeError("mirrored frame came back in a different row order")
    tw, tm = built["ufc"]["target_win"].to_numpy(), free["ufc"]["target_win"].to_numpy()
    valid = np.isfinite(tw)
    if not (np.array_equal(np.isfinite(tm), valid) and np.array_equal(tm[valid], 1 - tw[valid])):
        raise RuntimeError("the target did not flip under the mirror")
    a = classify_mirror(built["X"], free["X"], cols).rename(
        columns={"transform": "free_centres", "clean": "free_clean"})
    b = classify_mirror(built["X"], frozen["X"], cols).rename(
        columns={"transform": "frozen_centres", "clean": "frozen_clean"})
    out = a.merge(b, on="feature")
    out["winner_model"] = out["feature"].isin(built["feature_cols_winner"])
    out["corner_history"] = out["feature"].isin(corner_swap.CORNER_HISTORY)
    out["live_swap_equals_mirror"] = out["frozen_clean"] & ~out["corner_history"]
    if verbose:
        print(f"  mirror audit: {len(out)} features; free centres: "
              f"{out['free_clean'].sum()} clean; frozen centres: "
              f"{out['frozen_clean'].sum()} clean")
        print("  free-centre transforms: " + ", ".join(
            f"{k}={v}" for k, v in out["free_centres"].str.split("<->|\\(").str[0]
            .str.strip().value_counts().items()))
        print("  not clean with frozen centres: "
              + ", ".join(out.loc[~out["frozen_clean"], "feature"]))
        print("  corner-history features (live swap differs from the mirror): "
              + ", ".join(corner_swap.CORNER_HISTORY))
    return out


# --------------------------------------------------------------------------
# the frames
# --------------------------------------------------------------------------

def frames(raw=None, features=None, swap_features=None, verbose=True):
    """model_compare's frame (valid rows, feature_cols_winner, or what
    `features(built)` returns) plus the swapped matrix aligned to it, and
    the full build for the audit. The swapped matrix is
    corner_swap.swapped_matrix's columns of the frame; a feature function
    that adds columns of its own must pass swap_features(raw, built) ->
    the swapped rows of its matrix (index aligned with built["ufc"]),
    since the swap of a new column is its own to define."""
    raw = mc.load_archive() if raw is None else raw
    captured = {}

    def grab(built):
        captured["built"] = built
        return (features(built) if features is not None
                else built["X"][built["feature_cols_winner"]])

    t0 = time.time()
    frame = mc.build_frame(raw, features=grab, verbose=False)
    built = captured["built"]
    if swap_features is not None:
        X_swap = pd.DataFrame(swap_features(raw, built))[frame["feature_cols"]]
    else:
        missing = [c for c in frame["feature_cols"] if c not in built["feature_cols"]]
        if missing:
            raise KeyError(f"no swapped row for {missing[:5]}: a feature function that adds "
                           f"columns must pass swap_features(raw, built)")
        X_swap = corner_swap.swapped_matrix(raw, built)[frame["feature_cols"]]
    valid = built["ufc"]["target_win"].notna().to_numpy()
    X_swap = X_swap.loc[valid].reset_index(drop=True).astype(float)
    if len(X_swap) != len(frame["X"]):
        raise RuntimeError("swapped matrix is not aligned with the frame")
    frame["X_swap"] = X_swap
    frame["built"] = built
    frame["raw"] = raw
    if verbose:
        print(f"  frames: {len(frame['X']):,} valid fights x {frame['X'].shape[1]} features, "
              f"as listed and swapped ({time.time() - t0:.0f}s)")
    return frame


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def fit_augmented(X, X_swap, y, dates, seed=mc.SEED):
    """mc.fit_production's recipe on the doubled block: the same split of
    FIGHTS (the last CAL_FRACTION by date calibrate), every fight once as
    listed and once swapped with 1 - y, the recency weight of the fight on
    both copies. Hyperparameters are mc.fit_production's, line for line."""
    X, X_swap, y = np.asarray(X, float), np.asarray(X_swap, float), np.asarray(y, float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("fit_augmented needs date-sorted rows")
    n = len(X)
    train_end = int(n * (1.0 - mc.CAL_FRACTION))
    w = mc.recency_weights(dates.iloc[:train_end])
    X_tr = np.vstack([X[:train_end], X_swap[:train_end]])
    y_tr = np.concatenate([y[:train_end], 1.0 - y[:train_end]])
    w_tr = np.concatenate([w, w])
    X_cal = np.vstack([X[train_end:], X_swap[train_end:]])
    y_cal = np.concatenate([y[train_end:], 1.0 - y[train_end:]])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
    lr.fit(X_tr_s, y_tr, sample_weight=w_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                min_samples_split=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr, sample_weight=w_tr)
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        random_state=seed, eval_metric="logloss", verbosity=0,
                        early_stopping_rounds=50)
    xgb.fit(X_tr_s, y_tr, sample_weight=w_tr, eval_set=[(X_cal_s, y_cal)], verbose=False)
    return {"scaler": scaler, "lr": lr, "rf": rf, "xgb": xgb, "platt": None,
            "train_rows": int(train_end), "cal_rows": int(n - train_end),
            "augmented_rows": int(2 * train_end),
            "xgb_best_iteration": int(xgb.best_iteration)}


def ensemble(models, X):
    return mc._average(models["scaler"], models["lr"], models["rf"], models["xgb"], X)


def symmetric_platt(models, X_fit, X_fit_swap, y_fit):
    """The Platt calibrator of the plan: fitted on the calibration block of
    the fit (the last train_rows: rows, one per fight, as listed) on the
    symmetrised ensemble output."""
    te = models["train_rows"]
    e_cal = corner_swap.symmetrise(ensemble(models, X_fit[te:]), ensemble(models, X_fit_swap[te:]))
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(e_cal.reshape(-1, 1), np.asarray(y_fit, float)[te:])
    return platt


def predict_symmetric(models, platt, X, X_swap):
    """(symmetrised ensemble, calibrated probability)."""
    e = corner_swap.symmetrise(ensemble(models, X), ensemble(models, X_swap))
    return e, platt.predict_proba(e.reshape(-1, 1))[:, 1]


def platt_at_half(platt):
    return float(platt.predict_proba([[0.5]])[0, 1])


def walk_forward(X, X_swap, y, ufc, *, fit_base=mc.fit_production, fit_aug=fit_augmented,
                 first_year=mc.FIRST_PREDICTED_YEAR, min_train=mc.MIN_TRAIN,
                 min_test=mc.MIN_TEST, verbose=True):
    """Both arms, year by year, on exactly model_compare.walk_forward's
    splits (tests/test_symmetry.py proves the splits are the same)."""
    X = pd.DataFrame(X).reset_index(drop=True).to_numpy(float)
    X_swap = pd.DataFrame(X_swap).reset_index(drop=True).to_numpy(float)
    y = np.asarray(y, float)
    ufc = ufc.reset_index(drop=True)
    dates = pd.to_datetime(ufc["date"])
    years = dates.dt.year.to_numpy()
    out, by_year = [], {}
    for year in sorted(np.unique(years)):
        if year < first_year:
            continue
        train, test = years < year, years == year
        if train.sum() < min_train or test.sum() < min_test:
            continue
        t0 = time.time()
        base = fit_base(X[train], y[train], dates[train])
        platt_a = symmetric_platt(base, X[train], X_swap[train], y[train])
        e_a, p_a = predict_symmetric(base, platt_a, X[test], X_swap[test])
        _, p_refit = mc.predict_production(base, X[test])
        aug = fit_aug(X[train], X_swap[train], y[train], dates[train])
        platt_b = symmetric_platt(aug, X[train], X_swap[train], y[train])
        e_b, p_b = predict_symmetric(aug, platt_b, X[test], X_swap[test])
        idx = np.flatnonzero(test)
        block = ufc.iloc[idx]
        out.append(pd.DataFrame({
            "fight_id": block["fight_id"].astype(str).to_numpy(),
            "date": dates.iloc[idx].to_numpy(), "year": int(year),
            "event": block.get("event_name", pd.Series("", index=block.index)).to_numpy(),
            "red": block["r_name"].to_numpy(), "blue": block["b_name"].to_numpy(),
            "y": y[idx], "p_refit": p_refit,
            "e_A": e_a, "p_A": p_a, "e_B": e_b, "p_B": p_b,
        }))
        by_year[int(year)] = {
            "train_fights": int(train.sum()), "predicted": int(test.sum()),
            "platt_at_half_A": platt_at_half(platt_a), "platt_at_half_B": platt_at_half(platt_b),
            "xgb_best_iteration_A": base.get("xgb_best_iteration"),
            "xgb_best_iteration_B": aug.get("xgb_best_iteration"),
            "seconds": round(time.time() - t0, 1)}
        if verbose:
            print(f"  {year}: trained on {train.sum():,} fights (B on {2 * base['train_rows']:,} "
                  f"rows), predicted {test.sum():,} ({time.time() - t0:.0f}s; platt(0.5) "
                  f"A {by_year[int(year)]['platt_at_half_A']:.3f} "
                  f"B {by_year[int(year)]['platt_at_half_B']:.3f})", flush=True)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), by_year


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def blend(p_model, p_market):
    return np.array([market_blend.blend(m, k) for m, k in zip(p_model, p_market)])


def passes_bar(model_delta, blend_delta):
    """THE pre-registered bar: the wider interval of (arm - baseline)
    model-only log loss lies entirely below zero AND the point estimate of
    the blended log loss difference is <= 0."""
    return bool(model_delta["high"] < 0) and bool(blend_delta["mean"] <= 0)


def red_bias(y, p):
    """Mean predicted red-corner probability minus the observed red win rate."""
    return float(np.mean(p) - np.mean(y))


def table(frame, label, *, priced, decides, alpha=ALPHA):
    """Every arm against the cached baseline on the rows of `frame`. The
    blended comparison needs prices, so it is only computed when `priced`;
    only the pre-registered table (decides=True) can pass an arm."""
    y = frame["y"].to_numpy()
    base = frame["p_model"].to_numpy()
    events, red, blue = (frame["event"].to_numpy(), frame["red_norm"].to_numpy(),
                         frame["blue_norm"].to_numpy())
    print(f"\n{'=' * 79}\n{label}: {len(frame):,} fights, {frame['event'].nunique():,} events, "
          f"years {frame['year'].min()}-{frame['year'].max()}, alpha {alpha:.4f}"
          + (" - THE BAR" if decides else " (secondary)") + f"\n{'=' * 79}")
    results = {"n": int(len(frame)), "years": [int(v) for v in sorted(frame["year"].unique())],
               "decides": bool(decides), "priced": bool(priced),
               "baseline": {**mc.metrics(y, base), "red_bias": red_bias(y, base)}, "arms": {}}
    if priced:
        market = frame["p_market"].to_numpy()
        base_blend = blend(base, market)
        results["baseline"]["blend"] = mc.metrics(y, base_blend)
        results["market"] = mc.metrics(y, market)
    print(f"  {'arm':<18}{'log loss':>9}{'delta':>8}  {'wider interval':<20}"
          f"{'blend ll':>9}{'delta':>8}{'acc':>7}{'brier':>7}{'red bias':>9}  verdict")
    b = results["baseline"]
    print(f"  {'baseline':<18}{b['log_loss']:>9.4f}{'':>8}  {'':<20}"
          + (f"{b['blend']['log_loss']:>9.4f}" if priced else f"{'-':>9}") + f"{'':>8}"
          f"{b['accuracy']:>7.1%}{b['brier']:>7.4f}{b['red_bias']:>+9.4f}")
    for arm in ARMS:
        p = frame[f"p_{arm}"].to_numpy()
        r = mc.compare(y, base, p, events, red=red, blue=blue, alpha=alpha, label_cand=arm)
        r["red_bias"] = red_bias(y, p)
        r["calibration"] = mc.calibration_by_decile(y, p)
        d = r["delta"]["log_loss"]
        if priced:
            rb = mc.compare(y, base_blend, blend(p, market), events, red=red, blue=blue,
                            alpha=alpha, label_cand=arm)
            r["blend"] = {"metrics": rb[arm], "delta": rb["delta"]}
            bd = rb["delta"]["log_loss"]
            r["passes"] = passes_bar(d, bd) if decides else None
        else:
            r["passes"] = None
        results["arms"][arm] = r
        m = r[arm]
        verdict = ("" if r["passes"] is None else "PASS" if r["passes"] else "no")
        print(f"  {ARM_NAMES[arm]:<18}{m['log_loss']:>9.4f}{d['mean']:>+8.4f}  "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
              + (f"{r['blend']['metrics']['log_loss']:>9.4f}{r['blend']['delta']['log_loss']['mean']:>+8.4f}"
                 if priced else f"{'-':>9}{'':>8}")
              + f"{m['accuracy']:>7.1%}{m['brier']:>7.4f}{r['red_bias']:>+9.4f}  {verdict}")
        print(f"  {'':<18}{'':>17}  event [{d['event_low']:+.4f}, {d['event_high']:+.4f}]  "
              f"fighter [{d['fighter_low']:+.4f}, {d['fighter_high']:+.4f}]")
        acc, bri = r["delta"]["accuracy"], r["delta"]["brier"]
        line = (f"  {'':<18}{'':>17}  accuracy delta {acc['mean']:+.4f} [{acc['low']:+.4f}, "
                f"{acc['high']:+.4f}]; brier delta {bri['mean']:+.4f} [{bri['low']:+.4f}, {bri['high']:+.4f}]")
        if priced:
            bd = r["blend"]["delta"]
            line += (f"; blend ll delta [{bd['log_loss']['low']:+.4f}, {bd['log_loss']['high']:+.4f}], "
                     f"blend acc delta {bd['accuracy']['mean']:+.4f}")
        print(line)
    results["baseline_calibration"] = mc.calibration_by_decile(y, base)
    return results


def by_year_table(frame, label="priced fights"):
    print(f"\n  model-only log loss by year, {label} (baseline / A / B), red win rate, mean p")
    rows = {}
    for year, g in frame.groupby("year"):
        y = g["y"].to_numpy()
        rows[int(year)] = {"n": int(len(g)),
                           "baseline": float(mc.log_loss_rows(y, g["p_model"]).mean()),
                           "A": float(mc.log_loss_rows(y, g["p_A"]).mean()),
                           "B": float(mc.log_loss_rows(y, g["p_B"]).mean()),
                           "red_win_rate": float(y.mean()),
                           "mean_p": {"baseline": float(g["p_model"].mean()),
                                      "A": float(g["p_A"].mean()), "B": float(g["p_B"].mean())}}
        r = rows[int(year)]
        print(f"    {year}: n={r['n']:>4}  {r['baseline']:.4f} / {r['A']:.4f} / {r['B']:.4f}"
              f"   red wins {r['red_win_rate']:.3f}; mean p {r['mean_p']['baseline']:.3f} / "
              f"{r['mean_p']['A']:.3f} / {r['mean_p']['B']:.3f}")
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    registered = check_plan()
    print(f"plan sha256 {registered['sha256'][:12]} registered {registered['registered_at']}")
    print("frames (as listed, swapped)...")
    fr = frames()
    print("mirror audit...")
    audit = mirror_audit(fr["raw"], fr["built"])
    audit.to_csv(AUDIT_FILE, index=False)
    print(f"  wrote {AUDIT_FILE.name}")
    print("baseline walk-forward predictions (cached if the archive is unchanged)...")
    # the plan's baseline: the recipe as listed, model_compare's legacy
    # baseline (its default is this file's arm B since the fix phase)
    base = mc.predictions(raw=fr["raw"], recipe="legacy", verbose=True)
    base["fight_id"] = base["fight_id"].astype(str)
    print("walk-forward, arms A and B...")
    arms, by_year = walk_forward(fr["X"], fr["X_swap"], fr["y"], fr["ufc"])
    merged = base.merge(arms.drop(columns=["date", "year", "event", "red", "blue", "y"]),
                        on="fight_id", how="inner", validate="one_to_one")
    if len(merged) != len(base) or len(merged) != len(arms):
        raise RuntimeError(f"the arms predicted {len(arms)} fights, the cache holds "
                           f"{len(base)}, {len(merged)} in common - not the same splits")
    refit_gap = np.abs(merged["p_refit"] - merged["p_model"])
    print(f"  baseline refit against the cache: max |diff| {refit_gap.max():.2e}, "
          f"mean {refit_gap.mean():.2e}")
    priced = merged[np.isfinite(merged["p_market"])].reset_index(drop=True)
    confirm = priced[priced["year"] >= CONFIRM_FROM].reset_index(drop=True)
    everything = merged.reset_index(drop=True)
    everything_confirm = everything[everything["year"] >= CONFIRM_FROM].reset_index(drop=True)
    print(f"\n{len(priced):,} priced of {len(merged):,} predicted fights; family of {FAMILY}, "
          f"alpha {ALPHA:.4f}")
    priced_by_year = {int(y): int(n) for y, n in priced["year"].value_counts().sort_index().items()}
    unpriced_years = sorted(set(merged["year"].astype(int)) - set(priced_by_year))
    print("  priced fights by year: " + ", ".join(f"{y}={n}" for y, n in priced_by_year.items()))
    if unpriced_years:
        print(f"  predicted years with NO priced fight (outside the primary set and the "
              f"priced secondaries): {', '.join(map(str, unpriced_years))}")
    # The years odds.csv does not price, seen at least once: OPENING lines
    # (opening_odds.csv, kept apart from closing lines) as the market. A
    # secondary, never the bar - an opening line is a weaker forecast than
    # a closing one, and the blend weight was fitted on closing lines.
    opening = merged.assign(p_market=mc.market_probabilities(
        merged, mc.load_prices(OPENING_CSV, columns=("open_a", "open_b"))))
    opening_unpriced = opening[np.isfinite(opening["p_market"])
                               & opening["year"].isin(unpriced_years)].reset_index(drop=True)
    opening_by_year = {int(y): int(n) for y, n in
                       opening_unpriced["year"].value_counts().sort_index().items()}
    print("  of those, fights with an OPENING line: "
          + (", ".join(f"{y}={n}" for y, n in opening_by_year.items()) or "none"))

    results = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "plan": __doc__, "plan_sha256": registered["sha256"],
        "plan_registered_at": registered["registered_at"],
        "odds_sha256": base.attrs.get("odds_sha256"),
        "opening_odds_sha256": mc.file_sha256(OPENING_CSV),
        "family": FAMILY, "alpha": ALPHA, "confirm_from": CONFIRM_FROM,
        "features": int(fr["X"].shape[1]), "valid_fights": int(len(fr["X"])),
        "predicted_rows": int(len(merged)), "priced_rows": int(len(priced)),
        "priced_rows_by_year": priced_by_year,
        "predicted_years_without_prices": unpriced_years,
        "baseline_refit_vs_cache": {"max_abs": float(refit_gap.max()),
                                    "mean_abs": float(refit_gap.mean())},
        "mirror_audit": {
            "features": int(len(audit)),
            "free_centres_clean": int(audit["free_clean"].sum()),
            "frozen_centres_not_clean": audit.loc[~audit["frozen_clean"], "feature"].tolist(),
            "corner_history": list(corner_swap.CORNER_HISTORY),
            "transforms_free_centres": {k: int(v) for k, v in audit["free_centres"]
                                        .str.split("<->|\\(").str[0].str.strip()
                                        .value_counts().items()}},
        "by_year_fit": by_year,
        "primary": table(priced, "PRIMARY - priced fights, all predicted years",
                         priced=True, decides=True),
        "secondary": {
            "priced_confirm": table(confirm, f"SECONDARY - priced fights, {CONFIRM_FROM} onward",
                                    priced=True, decides=False),
            "all_fights": table(everything, "SECONDARY - all predicted fights (priced and unpriced), "
                                "model only", priced=False, decides=False),
            "all_fights_confirm": table(everything_confirm, f"SECONDARY - all predicted fights, "
                                        f"{CONFIRM_FROM} onward, model only", priced=False,
                                        decides=False),
            "log_loss_by_year_priced": by_year_table(priced),
            "log_loss_by_year_all": by_year_table(everything, "all predicted fights"),
            "opening_lines_unpriced_years": {
                "note": "SECONDARY, NOT THE BAR: the years odds.csv prices no fight of, "
                        "scored with opening_odds.csv's OPENING lines as the market. The "
                        "primary set and bar are closing lines only.",
                "years": unpriced_years, "fights_by_year": opening_by_year,
                "table": (table(opening_unpriced,
                                f"SECONDARY - {', '.join(map(str, unpriced_years))} fights "
                                f"with an OPENING line as the market (opening_odds.csv) - "
                                f"NOT THE BAR", priced=True, decides=False)
                          if len(opening_unpriced) else None)},
        },
    }
    passing = [a for a in ARMS if results["primary"]["arms"][a]["passes"]]
    results["passing"] = passing
    if passing:
        results["verdict"] = (f"PASS: {', '.join(ARM_NAMES[a] for a in passing)} - "
                              f"model-only log loss interval below zero and the blend not worse")
    else:
        results["verdict"] = ("NO ARM PASSES: neither symmetrised arm's model-only log loss "
                              "interval lies below the baseline at the family-wise level with "
                              "the blend not worse; nothing ships, predict_card.py unchanged")
    print(f"\nVERDICT: {results['verdict']}")
    OUT.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
