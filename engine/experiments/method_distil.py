"""Can the method model learn what the market knows, from the features it has?

The market prices finish-or-distance at AUC 0.660. This model, on the same
fights, manages 0.594. Something separates those two, and there are only two
possibilities:

    the features carry it and the model is not extracting it
    the features do not carry it at all

The difference matters completely. In the first case the gap is recoverable
and this is how. In the second no amount of training recovers anything,
because a student cannot learn what its inputs do not contain, and the right
conclusion is that the method model is near the limit of what the record can
say.

DISTILLATION asks the question directly. A binary outcome is a very noisy
target: one fight, one bit, and half the information in "this was a 55/45
fight that happened to end early" is thrown away. The market's de-vigged
probability is the same question answered on a continuous scale by people
with money at stake. Training toward it - instead of, or as well as, the
outcome - is a standard way to hand a model a less noisy version of what it
is already trying to learn.

    OUTCOME    trained on the binary, as it ships today
    MARKET     trained on the market's P(finish), a regression
    SOFT(a)    trained on (1-a) * outcome + a * market

Every arm is JUDGED ON THE ACTUAL OUTCOME, walk-forward, each year from
earlier years only. `a` is chosen from earlier years too, so a weight that
flatters the period it was picked on cannot flatter the period it is scored
on.

NO LEAKAGE, AND IT IS WORTH BEING PRECISE ABOUT WHY. The market price is
published before the fight, so a model trained on it has not seen a result.
The student never receives a price at prediction time - it learns to predict
one from the record, which is the whole point. And prices only ever come from
years strictly earlier than the year being predicted.

The universe is the fights that carry a price, so every row has both a target
and an outcome and the arms are compared on identical data.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.method_blend import (PROP_COLUMNS, market_finish_probability,
                                      pair_key)
from experiments.method_model import CLASSES, CONFIRM_FROM, level_columns, rate_columns
from method_calibration import apply_calibrator, fit_calibrator
from name_resolution import norm_name

METHOD_ODDS = ENGINE / "data" / "method_odds.csv"
FIRST_PREDICTED_YEAR = 2014      # the props start in 2012; leave two to train on
MIN_TRAIN = 400
SOFT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (42, 7, 2024)


def params(seed):
    return dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, random_state=seed, verbosity=0)


def fit_arm(X_train, target, X_test, *, binary, seed):
    """One arm. A hard 0/1 target gets a classifier, a soft one a regressor."""
    if binary:
        model = XGBClassifier(objective="binary:logistic",
                              eval_metric="logloss", **params(seed))
        model.fit(X_train, target, verbose=False)
        return model.predict_proba(X_test)[:, 1]
    model = XGBRegressor(objective="reg:squarederror", **params(seed))
    model.fit(X_train, target, verbose=False)
    return np.clip(model.predict(X_test), 0.001, 0.999)


def walk(X, soft_target, outcome, years, weight, *, seed, calibrate=False):
    """Predict each year from earlier years only, at one soft weight.

    `calibrate` fits an isotonic map on a holdout carved from the tail of the
    training window and applies it to that year. A model distilled from the
    market inherits the market's bias along with its ranking - the market says
    56.5% of fights are finished where 50.8% are - and isotonic fixes the
    price without touching the order.
    """
    out_p, out_i = [], []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).to_numpy()
        test = (years == year).to_numpy()
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        rows = np.flatnonzero(train)
        binary = (weight == 0.0)

        if calibrate:
            # The calibrator is fitted on the tail of the training window -
            # later than what the model learned from, earlier than what it
            # predicts. Fitting it on the year being scored would be fitting
            # on the answers.
            cut = int(len(rows) * 0.75)
            fit_rows, hold_rows = rows[:cut], rows[cut:]
            if len(hold_rows) < 150:
                continue
            target = ((1 - weight) * outcome[fit_rows]
                      + weight * soft_target[fit_rows])
            held = fit_arm(X[fit_rows], target, X[hold_rows],
                           binary=binary, seed=seed)
            calibrator = fit_calibrator(held, outcome[hold_rows])
        else:
            calibrator = None

        target = ((1 - weight) * outcome[rows] + weight * soft_target[rows])
        p = fit_arm(X[rows], target, X[test], binary=binary, seed=seed)
        if calibrator is not None:
            p = apply_calibrator(calibrator, p)
        out_p.append(p)
        out_i.append(np.flatnonzero(test))
    return np.concatenate(out_p), np.concatenate(out_i)


def score(label, p, truth, market=None):
    row = {
        "label": label,
        "n": int(len(truth)),
        "accuracy": float(((p > 0.5) == (truth == 1)).mean()),
        "auc": float(roc_auc_score(truth, p)),
        "brier": float(brier_score_loss(truth, np.clip(p, 0, 1))),
        "log_loss": float(log_loss(truth, np.clip(p, 1e-6, 1 - 1e-6))),
    }
    if market is not None:
        # How close the student got to the teacher it never sees at
        # prediction time. A student that cannot reproduce the market from
        # the record is evidence the record does not contain it.
        row["rmse_vs_market"] = float(np.sqrt(np.mean((p - market) ** 2)))
        row["corr_with_market"] = float(np.corrcoef(p, market)[0, 1])
    return row


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/method_distil.json")
    if not METHOD_ODDS.exists():
        sys.exit(f"{METHOD_ODDS} is missing. Run the fetch-method-odds mode.")

    print("building features...")
    import predict_card as engine

    from feature_inventory import finish_level_names, method_rate_names

    added = set(method_rate_names()) | set(finish_level_names())
    mask = engine.ufc_valid["target_method"].isin(CLASSES).values
    full = engine.X_valid.reset_index(drop=True)[mask]
    base = full[[c for c in full.columns if c not in added]].reset_index(drop=True)

    meta = engine.ufc_valid.reset_index(drop=True)[mask].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    order = {c: i for i, c in enumerate(CLASSES)}
    y3 = meta["target_method"].map(order).to_numpy()
    valid = engine.ufc["target_win"].notna().to_numpy()

    def aligned(frame):
        return (frame.reset_index(drop=True)[valid]
                .reset_index(drop=True)[mask].reset_index(drop=True))

    X_all = pd.concat([base, aligned(level_columns(engine)),
                       aligned(rate_columns(engine))], axis=1)

    props = pd.read_csv(METHOD_ODDS)
    props["date"] = pd.to_datetime(props["date"], errors="coerce")
    props = props.dropna(subset=list(PROP_COLUMNS))
    market, _ = market_finish_probability(props)
    priced = {pair_key(a, b, d): p for a, b, d, p
              in zip(props.fighter_a, props.fighter_b, props.date, market)}

    keys = [pair_key(r, b, d) for r, b, d
            in zip(meta.r_name, meta.b_name, meta.date)]
    market_p = np.array([priced.get(k, np.nan) for k in keys])
    have = ~np.isnan(market_p)

    X = X_all[have].to_numpy()
    soft = market_p[have]
    outcome = (y3[have] != 0).astype(float)
    years = meta.loc[have, "date"].dt.year.reset_index(drop=True)
    print(f"  {int(have.sum()):,} fights carry both a price and an outcome "
          f"({years.min()}-{years.max()})")
    print(f"  really finished {outcome.mean():.1%}, market said {soft.mean():.1%}")

    # --- every arm, averaged over seeds -----------------------------------
    print(f"\n  {'arm':<16}{'accuracy':>10}{'AUC':>8}{'Brier':>9}"
          f"{'rmse vs mkt':>13}{'corr':>7}")
    print("  " + "-" * 63)
    results, per_weight = [], {}
    for weight in SOFT_WEIGHTS:
        runs = []
        for seed in SEEDS:
            p, idx = walk(X, soft, outcome, years, weight, seed=seed)
            runs.append((p, idx))
        idx = runs[0][1]
        p = np.mean([r[0] for r in runs], axis=0)
        label = ("OUTCOME" if weight == 0 else "MARKET" if weight == 1
                 else f"SOFT {weight:.2f}")
        row = score(label, p, outcome[idx], market=soft[idx])
        results.append(row)
        per_weight[weight] = (p, idx)
        print(f"  {label:<16}{row['accuracy']:>10.1%}{row['auc']:>8.3f}"
              f"{row['brier']:>9.4f}{row['rmse_vs_market']:>13.4f}"
              f"{row['corr_with_market']:>7.2f}")

    # Distilled and then calibrated on outcomes: the ranking from the market,
    # the price from what actually happened.
    runs = [walk(X, soft, outcome, years, 1.0, seed=s, calibrate=True)
            for s in SEEDS]
    cal_idx = runs[0][1]
    cal_p = np.mean([r[0] for r in runs], axis=0)
    cal_row = score("MARKET+calibrated", cal_p, outcome[cal_idx],
                    market=soft[cal_idx])
    results.append(cal_row)
    per_weight["calibrated"] = (cal_p, cal_idx)
    print(f"  {'MARKET+cal':<16}{cal_row['accuracy']:>10.1%}{cal_row['auc']:>8.3f}"
          f"{cal_row['brier']:>9.4f}{cal_row['rmse_vs_market']:>13.4f}"
          f"{cal_row['corr_with_market']:>7.2f}")
    print(f"  {'':16}says finish {(cal_p > 0.5).mean():.1%} of the time, "
          f"really {outcome[cal_idx].mean():.1%}")

    idx0 = per_weight[0.0][1]
    market_row = score("the market itself", soft[idx0], outcome[idx0])
    print(f"  {'THE MARKET':<16}{market_row['accuracy']:>10.1%}"
          f"{market_row['auc']:>8.3f}{market_row['brier']:>9.4f}")
    results.append(market_row)

    # --- the weight chosen forward ----------------------------------------
    per_weight.pop("calibrated", None)
    chosen, blended, truth_out = [], [], []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR + 2:
            continue
        past = np.array([years.to_numpy()[i] < year for i in idx0])
        now = np.array([years.to_numpy()[i] == year for i in idx0])
        if past.sum() < MIN_TRAIN or now.sum() < 20:
            continue
        best, best_auc = 0.0, -1.0
        for weight, (p, _) in per_weight.items():
            t = outcome[idx0][past]
            if len(np.unique(t)) < 2:
                continue
            auc = roc_auc_score(t, p[past])
            if auc > best_auc:
                best, best_auc = weight, auc
        chosen.append((int(year), best))
        blended.append(per_weight[best][0][now])
        truth_out.append(outcome[idx0][now])

    if blended:
        p = np.concatenate(blended)
        t = np.concatenate(truth_out)
        forward = score("chosen forward", p, t)
        results.append(forward)
        print(f"\n  weight chosen from earlier years: "
              + ", ".join(f"{yr}:{w:.2f}" for yr, w in chosen[-8:]))
        print(f"  {'chosen forward':<16}{forward['accuracy']:>10.1%}"
              f"{forward['auc']:>8.3f}{forward['brier']:>9.4f}")

    outcome_row = next(r for r in results if r["label"] == "OUTCOME")
    best_soft = max((r for r in results if r["label"].startswith(("SOFT", "MARKET"))),
                    key=lambda r: r["auc"])
    gain = best_soft["auc"] - outcome_row["auc"]
    closed = gain / max(market_row["auc"] - outcome_row["auc"], 1e-9)
    print(f"\n  best distilled arm is {best_soft['label']} at AUC "
          f"{best_soft['auc']:.3f}, against {outcome_row['auc']:.3f} for "
          f"training on the outcome.")
    print(f"  the gap to the market was {market_row['auc'] - outcome_row['auc']:.3f}; "
          f"this closes {closed:.0%} of it.")
    if gain > 0.01:
        print("  THE FEATURES CARRY MORE THAN THE OUTCOME-TRAINED MODEL WAS "
              "EXTRACTING.")
    else:
        print("  THE FEATURES DO NOT CARRY IT. A student given a better target\n"
              "  learns no more than one given the outcome, which says the gap\n"
              "  to the market is information this record does not hold - camps,\n"
              "  injuries, styles - and no amount of training recovers it.")

    # An AUC gap of this size on 3,700 fights still deserves an interval:
    # the arms are scored on the same fights, so it is paired.
    best_p, best_idx = per_weight[1.0]
    base_p, base_idx = per_weight[0.0]
    assert np.array_equal(best_idx, base_idx), "arms scored on different fights"
    truth = outcome[base_idx]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        pick = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[pick])) < 2:
            continue
        diffs.append(roc_auc_score(truth[pick], best_p[pick])
                     - roc_auc_score(truth[pick], base_p[pick]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  distilled minus outcome-trained, AUC: "
          f"{best_soft['auc'] - outcome_row['auc']:+.3f}  "
          f"95% [{low:+.3f}, {high:+.3f}]")
    print("  " + ("The interval excludes zero." if low > 0
                  else "THE INTERVAL CROSSES ZERO."))

    out = ENGINE / "experiments" / "method_distil.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": list(SEEDS),
        "fights": int(have.sum()),
        "results": results,
        "chosen_forward": chosen,
        "auc_gain_interval": [float(low), float(high)],
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
