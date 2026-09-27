"""Can the method model be made to say anything?

On UFC Vegas 121 it went 4/9, and 1/6 on the fights whose winner it called
correctly. It said Decision 55% for Hiestand vs Nakamura, which ended by
submission in the second - a fight where two finishers met and the model
offered a number a coin could have produced. That complaint is the reason
this file exists.

THE STRUCTURAL SUSPICION. The method model has no features of its own. It is
trained on X_valid_winner, the matrix built and audited for "who wins", where
55 of 57 columns are differences between the corners. For "who wins" that is
the right shape. For "how does it end" it is close to the wrong one:

    ko_rate_diff          0.0    two men who knock everyone out
    ko_rate_diff          0.0    two men who have never knocked anyone out

Those are the same number and opposite fights. feature_spec already has the
shape that separates them - paired_level, the mean of the two - and the
winner model carries twenty of them. Not one is on the five columns that
decide whether a fight is finished: ko_rate, sub_rate, ko_vulnerability,
been_finished, sub_def. Their `why` lines even say so out loud: "a style
marker more than a predictor of who wins". They were justified for the winner
model and the method model inherited them without being asked.

WHAT IS MEASURED HERE, walk-forward, each year predicted by a model trained
only on earlier years:

    BASE          what ships today: 3-class XGB, class-weighted, on the full
                  feature matrix - which includes the cage-control columns the
                  winner model is denied, so BASE is NOT X_valid_winner
    NO-WEIGHTS    the same without the sample weights
    LEVELS        BASE plus the level of each finish-family feature
    BOTH+W        levels and rates together, class weights kept
    +DIVISION     BOTH+W plus the division's own finish rate, point in time -
                  heavyweight finishes 63.9% and women's strawweight 33.6%,
                  and the models had one binary flag between them
    HIERARCHY+W   the two-stage split with the weights kept
    RATES         BASE plus the point-in-time per-15-minute finish rates, as
                  both level and difference - the numbers the simulator works
                  from and the method model has never seen
    LEVELS+RATES  both, unweighted
    HIERARCHY     P(finish), then P(KO | finish), on LEVELS+RATES
    ALWAYS-DEC    predict the commonest class every time
    BASE-RATE     the training distribution, constant - no skill at all

Accuracy alone would be won by ALWAYS-DEC and would tell us nothing, so the
table also carries macro-F1 (does it ever get a finish right), multiclass log
loss and Brier (are the printed percentages worth printing), and the
calibration of P(finish), which is the number the app shows.

A variant ships only if it beats BASE on log loss AND macro-F1 on the confirm
period, which is 2020 onward and informed nothing here.
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)
from xgboost import XGBClassifier

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from feature_spec import build_all
from method_calibration import (apply_calibrator, balanced_accuracy,
                                choose_threshold, fit_calibrator)

FIRST_PREDICTED_YEAR = 2011
CONFIRM_FROM = 2020
MIN_TRAIN = 500
CLASSES = ["Decision", "KO/TKO", "Submission"]

# The five columns that decide whether a fight is finished rather than who
# wins it, and which ship as differences only.
FINISH_FAMILY = ("ko_rate", "sub_rate", "ko_vulnerability", "been_finished",
                 "sub_def")

# The rate block is not restated here. It is read off feature_inventory, so
# what this experiment measures and what predict_card ships are the same set
# by construction - the first draft of this file listed the columns by hand
# and included cd_kd_per15, which is declared elsewhere already, so the RATES
# variant silently carried two copies of it.


def xgb(seed=42, **kw):
    params = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                  reg_lambda=1.0, random_state=seed, verbosity=0)
    params.update(kw)
    return XGBClassifier(**params)


def level_columns(engine):
    """Rebuild the finish family with level=True and return just the levels.

    Rebuilt through the same build_all that makes the training matrix, rather
    than averaged by hand here, so a change to paired_level reaches this
    experiment instead of leaving it measuring a second definition.
    """
    from dataclasses import replace

    from feature_inventory import all_specs

    wanted = [s for s in all_specs()
              if getattr(s, "name", None) in FINISH_FAMILY]
    missing = set(FINISH_FAMILY) - {s.name for s in wanted}
    if missing:
        raise KeyError(f"finish-family specs not found: {sorted(missing)}")

    levelled = [replace(s, level=True) for s in wanted]
    built = build_all(levelled, engine.ufc)
    return built[[f"{name}_level" for name in FINISH_FAMILY]]


def division_column(engine):
    """The division's own finish rate, point in time.

    Computed here rather than read off the frame: the feature was measured
    and rejected, so predict_card does not build it, and a rejected feature
    should not cost every production run to keep an experiment convenient.
    """
    from division_prior import division_finish_prior

    return pd.DataFrame(
        {"division_finish_prior": division_finish_prior(engine.ufc)},
        index=engine.ufc.index)


def rate_columns(engine):
    """The declared finish-rate block, built by the declaration itself."""
    from feature_inventory import METHOD_RATES

    return build_all(METHOD_RATES, engine.ufc)


def fit_predict(X_tr, y_tr, X_te, weights=None, n_classes=3, seed=42):
    objective = ("multi:softprob" if n_classes > 2 else "binary:logistic")
    extra = {"num_class": n_classes} if n_classes > 2 else {}
    model = xgb(seed=seed, objective=objective, eval_metric="mlogloss", **extra)
    model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
    proba = model.predict_proba(X_te)
    if n_classes == 2 and proba.shape[1] == 2:
        return proba
    return proba


def class_weights_for(y):
    counts = Counter(y)
    n, k = len(y), len(counts)
    table = {c: n / (k * counts[c]) for c in counts}
    return np.array([table[v] for v in y])


def walk_forward(X, y, years, *, weighted, hierarchy=False, seed=42):
    """Predict each year from earlier years only. Returns (proba, index)."""
    out_p, out_i = [], []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        X_tr, y_tr = X[train], y[train]
        X_te = X[test]

        if not hierarchy:
            w = class_weights_for(y_tr) if weighted else None
            p = fit_predict(X_tr, y_tr, X_te, weights=w, n_classes=3, seed=seed)
        else:
            # Stage one: finished at all. Stage two: which kind, among the
            # fights that were finished - fitted on those alone, so the second
            # question is never asked of a decision.
            is_fin_tr = (y_tr != 0).astype(int)
            w1 = class_weights_for(is_fin_tr) if weighted else None
            p_fin = fit_predict(X_tr, is_fin_tr, X_te, weights=w1,
                                n_classes=2)[:, 1]
            fin = y_tr != 0
            if fin.sum() < 50 or len(np.unique(y_tr[fin])) < 2:
                continue
            is_ko_tr = (y_tr[fin] == 1).astype(int)
            w2 = class_weights_for(is_ko_tr) if weighted else None
            p_ko = fit_predict(X_tr[fin], is_ko_tr, X_te, weights=w2,
                               n_classes=2)[:, 1]
            p = np.column_stack([1 - p_fin, p_fin * p_ko, p_fin * (1 - p_ko)])

        out_p.append(p)
        out_i.append(np.flatnonzero(test))
    if not out_p:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    return np.vstack(out_p), np.concatenate(out_i)


def calibrated_walk(X, y, years, *, weighted=True, holdout=0.25):
    """Walk forward again, this time fitting a calibrator and a threshold on
    a holdout carved out of each year's TRAINING window.

    The holdout is the tail of the training data, so it is later than what
    the model learned from and earlier than what it predicts - the only
    arrangement in which a correction is fitted on the model's own mistakes
    without being fitted on the answers it is about to be judged on.
    """
    raw_p, cal_p, idx, thresholds = [], [], [], []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        tr = np.flatnonzero(train)
        cut = int(len(tr) * (1 - holdout))
        fit_rows, hold_rows = tr[:cut], tr[cut:]
        if len(hold_rows) < 100:
            continue

        w = class_weights_for(y[fit_rows]) if weighted else None
        model = xgb(objective="multi:softprob", eval_metric="mlogloss",
                    num_class=3)
        model.fit(X[fit_rows], y[fit_rows], sample_weight=w, verbose=False)

        hold_finish = 1.0 - model.predict_proba(X[hold_rows])[:, 0]
        hold_real = (y[hold_rows] != 0).astype(float)
        calibrator = fit_calibrator(hold_finish, hold_real)
        threshold = choose_threshold(apply_calibrator(calibrator, hold_finish),
                                     hold_real)

        # The model that PREDICTS is refitted on the whole training window;
        # only the correction came from the holdout.
        w_full = class_weights_for(y[train]) if weighted else None
        full = xgb(objective="multi:softprob", eval_metric="mlogloss",
                   num_class=3)
        full.fit(X[train], y[train], sample_weight=w_full, verbose=False)
        p_finish = 1.0 - full.predict_proba(X[test])[:, 0]

        raw_p.append(p_finish)
        cal_p.append(apply_calibrator(calibrator, p_finish))
        idx.append(np.flatnonzero(test))
        thresholds.append((int(year), float(threshold)))
    return (np.concatenate(raw_p), np.concatenate(cal_p),
            np.concatenate(idx), thresholds)


def constant_proba(y_train_like, n):
    counts = Counter(y_train_like)
    total = len(y_train_like)
    row = np.array([counts.get(i, 0) / total for i in range(3)])
    return np.tile(row, (n, 1))


def score(label, proba, y_true):
    """Everything worth knowing about a set of predictions, in one row."""
    pred = proba.argmax(axis=1)
    finished = (y_true != 0).astype(float)
    p_finish = 1.0 - proba[:, 0]
    return {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "finish_brier": float(brier_score_loss(finished, p_finish)),
        "finish_rate_said": float(p_finish.mean()),
        "finish_rate_real": float(finished.mean()),
        # What the app actually prints: the top class and its percentage.
        "called_finish": int((pred != 0).sum()),
        # The binary on its own, thresholded at a half. This is what leading
        # with "finish or distance" would print, and it is not the same thing
        # as the argmax: a fight at KO .27 / SUB .24 / DEC .49 is more likely
        # than not to be finished and prints "Decision", because a decision is
        # one bucket and a finish is two.
        "binary_accuracy": float(((p_finish > 0.5) == (finished == 1)).mean()),
        "binary_auc": float(roc_auc_score(finished, p_finish)),
        "binary_says_finish": float((p_finish > 0.5).mean()),
        "finish_recall": float(((pred != 0) & (y_true != 0)).sum()
                               / max((y_true != 0).sum(), 1)),
        "finish_precision": float(((pred != 0) & (y_true != 0)).sum()
                                  / max((pred != 0).sum(), 1)),
    }


def reliability(proba, y_true, bins=5):
    """Is P(finish) worth its face value? Said against happened, by bucket."""
    p = 1.0 - proba[:, 0]
    real = (y_true != 0).astype(float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        if m.sum() < 20:
            continue
        out.append((float(p[m].mean()), float(real[m].mean()), int(m.sum())))
    return out


def report(rows, title):
    print(f"\n{title}")
    print(f"  {'variant':<14}{'n':>6}{'acc':>8}{'macroF1':>9}{'logloss':>9}"
          f"{'finBrier':>10}{'saidFin':>9}{'realFin':>9}{'finRec':>8}{'finPre':>8}")
    print("  " + "-" * 86)
    for r in rows:
        print(f"  {r['label']:<14}{r['n']:>6}{r['accuracy']:>8.3f}"
              f"{r['macro_f1']:>9.3f}{r['log_loss']:>9.4f}"
              f"{r['finish_brier']:>10.4f}{r['finish_rate_said']:>9.1%}"
              f"{r['finish_rate_real']:>9.1%}{r['finish_recall']:>8.2f}"
              f"{r['finish_precision']:>8.2f}")


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/method_model.json")
    print("building features...")
    import predict_card as engine

    # The shipping method model is handed X_valid, not X_valid_winner: it
    # keeps the cage-control columns the winner model is denied. Measuring
    # against the wrong baseline would credit this experiment with a gain that
    # is really just those four features coming back.
    # BASE is the method model as it was BEFORE this change: everything the
    # full matrix carried, minus the rate block being tested. Reading
    # X_valid now would put the block inside the baseline and measure nothing.
    from feature_inventory import finish_level_names, method_rate_names

    # BOTH blocks come out, not just the rates. An earlier version of this
    # line took only method_rate_names(), so "BASE" still carried the five
    # finish-family levels - it was measuring half the change against the
    # other half and reporting the remainder as noise.
    added = set(method_rate_names()) | set(finish_level_names())
    mask = engine.ufc_valid["target_method"].isin(CLASSES).values
    full = engine.X_valid.reset_index(drop=True)[mask]
    added = added | {"division_finish_prior"}
    base = full[[c for c in full.columns if c not in added]]
    meta = engine.ufc_valid.reset_index(drop=True)[mask].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    years = meta["date"].dt.year

    order = {c: i for i, c in enumerate(CLASSES)}
    y = meta["target_method"].map(order).to_numpy()

    # Both extra blocks are built over every row of ufc, then cut to the rows
    # ufc_valid kept and then to the rows with a method, in that order - the
    # same two cuts base went through.
    valid = engine.ufc["target_win"].notna().to_numpy()

    def aligned(frame):
        return (frame.reset_index(drop=True)[valid]
                .reset_index(drop=True)[mask].reset_index(drop=True))

    levels = aligned(level_columns(engine))
    rates = aligned(rate_columns(engine))
    divisions = aligned(division_column(engine))
    assert set(rates.columns) | set(divisions.columns) == added, \
        "measuring a different set than ships"
    base = base.reset_index(drop=True)
    assert len(levels) == len(base) == len(rates), "row alignment lost"

    with_levels = pd.concat([base, levels], axis=1)
    with_rates = pd.concat([base, rates], axis=1)
    with_both = pd.concat([base, levels, rates], axis=1)
    with_division = pd.concat([base, levels, rates, divisions], axis=1)

    print(f"  fights with a method: {len(base):,}")
    print(f"  distribution: "
          + ", ".join(f"{c} {(y == i).mean():.1%}" for i, c in enumerate(CLASSES)))
    print(f"  base {base.shape[1]} cols, +levels {with_levels.shape[1]}, "
          f"+rates {with_rates.shape[1]}, +both {with_both.shape[1]}")

    runs = {}
    for label, X, weighted, hier in (
            ("BASE", base, True, False),
            ("NO-WEIGHTS", base, False, False),
            ("LEVELS", with_levels, True, False),
            ("RATES", with_rates, True, False),
            ("LEVELS+RATES", with_both, False, False),
            # The obvious candidate: both feature blocks, keeping the class
            # weights. Leaving it out of the first run measured every part of
            # the change except the combination anyone would actually ship.
            ("BOTH+W", with_both, True, False),
            ("+DIVISION", with_division, True, False),
            ("HIERARCHY", with_both, False, True),
            ("HIERARCHY+W", with_both, True, True),
    ):
        print(f"  walking {label} ...", flush=True)
        proba, idx = walk_forward(X.to_numpy(), y, years,
                                  weighted=weighted, hierarchy=hier)
        runs[label] = (proba, idx)

    any_idx = runs["BASE"][1]
    runs["ALWAYS-DEC"] = (np.tile([1.0, 0.0, 0.0], (len(any_idx), 1)), any_idx)
    # A constant at the rate seen BEFORE the predicted period, not over it.
    early = y[(years < FIRST_PREDICTED_YEAR).values]
    runs["BASE-RATE"] = (constant_proba(early, len(any_idx)), any_idx)
    # log_loss will not take a hard zero.
    runs["ALWAYS-DEC"] = (np.clip(runs["ALWAYS-DEC"][0], 1e-6, 1), any_idx)

    all_rows, confirm_rows = [], []
    for label, (proba, idx) in runs.items():
        all_rows.append(score(label, proba, y[idx]))
        keep = (years.to_numpy()[idx] >= CONFIRM_FROM)
        confirm_rows.append(score(label, proba[keep], y[idx][keep]))

    print("\n  finish-or-distance, as a binary at a half (confirm period)")
    print(f"    {'variant':<14}{'accuracy':>10}{'AUC':>8}{'says finish':>13}{'really':>8}")
    print("    " + "-" * 53)
    for r in confirm_rows:
        print(f"    {r['label']:<14}{r['binary_accuracy']:>10.3f}{r['binary_auc']:>8.3f}"
              f"{r['binary_says_finish']:>13.1%}{r['finish_rate_real']:>8.1%}")

    print("\n  is P(finish) worth its face value? (+DIVISION, confirm period)")
    both, idx = runs["+DIVISION"]
    keep = (years.to_numpy()[idx] >= CONFIRM_FROM)
    print(f"    {'said':>8}{'happened':>10}{'n':>7}")
    for said, real, n in reliability(both[keep], y[idx][keep]):
        print(f"    {said:>7.0%}{real:>10.0%}{n:>7}")

    report(all_rows, f"EVERYTHING, {FIRST_PREDICTED_YEAR} ONWARD")
    report(confirm_rows, f"CONFIRM PERIOD, {CONFIRM_FROM} ONWARD "
                         f"(nothing here informed any choice)")

    base_row = next(r for r in confirm_rows if r["label"] == "BASE")
    print("\n  against BASE on the confirm period:")
    for r in confirm_rows:
        if r["label"] == "BASE":
            continue
        d_ll = base_row["log_loss"] - r["log_loss"]     # positive is better
        d_f1 = r["macro_f1"] - base_row["macro_f1"]
        verdict = ("ships" if d_ll > 0 and d_f1 > 0 else "does not ship")
        print(f"    {r['label']:<14} log loss {d_ll:+.4f}  macro-F1 {d_f1:+.4f}"
              f"   {verdict}")

    # ---------------------------------------------------------------- 2 & 6
    print("\n" + "=" * 79)
    print("CALIBRATION AND THRESHOLD, fitted on a holdout inside the training "
          "window")
    print("=" * 79)
    raw, cal, cidx, thresholds = calibrated_walk(
        with_division.to_numpy(), y, years)
    keep = years.to_numpy()[cidx] >= CONFIRM_FROM
    real = (y[cidx] != 0).astype(float)

    def line(label, p, t):
        m = keep
        said = p[m] > t
        acc = float((said == (real[m] == 1)).mean())
        bal = balanced_accuracy(p[m], real[m], t)
        brier = float(np.mean((p[m] - real[m]) ** 2))
        print(f"  {label:<22}{t:>7.2f}{acc:>10.3f}{bal:>10.3f}{brier:>9.4f}"
              f"{said.mean():>13.1%}{real[m].mean():>8.1%}")

    print(f"  {'':22}{'thresh':>7}{'accuracy':>10}{'balanced':>10}{'brier':>9}"
          f"{'says finish':>13}{'really':>8}")
    print("  " + "-" * 71)
    line("raw, half", raw, 0.5)
    line("calibrated, half", cal, 0.5)
    chosen = float(np.median([t for _, t in thresholds]))
    line("calibrated, chosen", cal, chosen)
    line("raw, chosen", raw, chosen)

    print(f"\n  thresholds chosen per year (median {chosen:.2f}): "
          + ", ".join(f"{yr}:{t:.2f}" for yr, t in thresholds[-8:]))

    print("\n  calibrated reliability, confirm period")
    print(f"    {'said':>8}{'happened':>10}{'n':>7}")
    edges = np.quantile(cal[keep], np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cal[keep] >= lo) & (cal[keep] <= hi)
        if m.sum() < 20:
            continue
        print(f"    {cal[keep][m].mean():>7.0%}{real[keep][m].mean():>10.0%}"
              f"{int(m.sum()):>7}")

    # ------------------------------------------------------------ noise
    # The gains being compared are a hundredth of a nat. XGBoost's histogram
    # build is not deterministic across threads, so one run of each variant
    # cannot tell a real difference from the scatter - an earlier run of this
    # file had BOTH+W at +0.0119 and this one has it at +0.0064, same code.
    # Three seeds each, and the spread printed beside the mean.
    print("\n" + "=" * 79)
    print("THE SAME COMPARISON OVER THREE SEEDS")
    print("=" * 79)
    print(f"  {'variant':<14}{'log loss':>22}{'macro-F1':>22}")
    print("  " + "-" * 58)
    noise = {}
    for label, X in (("BASE", base), ("BOTH+W", with_both),
                     ("+DIVISION", with_division)):
        lls, f1s = [], []
        for seed in (42, 7, 2024):
            proba, idx = walk_forward(X.to_numpy(), y, years,
                                      weighted=True, seed=seed)
            m = years.to_numpy()[idx] >= CONFIRM_FROM
            row = score(label, proba[m], y[idx][m])
            lls.append(row["log_loss"]); f1s.append(row["macro_f1"])
        noise[label] = (lls, f1s)
        print(f"  {label:<14}{np.mean(lls):>15.4f} +-{np.std(lls):<6.4f}"
              f"{np.mean(f1s):>13.4f} +-{np.std(f1s):<6.4f}")

    b_ll, b_f1 = noise["BASE"]
    print("\n  against BASE, mean over three seeds, with BASE's own spread "
          "for scale:")
    for label in ("BOTH+W", "+DIVISION"):
        lls, f1s = noise[label]
        d_ll = np.mean(b_ll) - np.mean(lls)
        d_f1 = np.mean(f1s) - np.mean(b_f1)
        verdict = ("clears the noise" if d_ll > 2 * np.std(b_ll)
                   else "INSIDE THE NOISE")
        print(f"    {label:<12} log loss {d_ll:+.4f}  macro-F1 {d_f1:+.4f}"
              f"   {verdict}")

    out = ENGINE / "experiments" / "method_model.csv"
    pd.DataFrame(confirm_rows).to_csv(out, index=False)
    (ENGINE / "experiments" / "method_model.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirm_from": CONFIRM_FROM,
        "all": all_rows,
        "confirm": confirm_rows,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
