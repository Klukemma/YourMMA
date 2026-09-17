"""Can a second layer tell us which predictions to trust?

The base model is ~60% on 2026 and badly overconfident. This asks a narrower
question than "can we predict better": given a prediction, can we predict
whether it is right? If so, acting only on the trustworthy slice raises
accuracy on the fights you actually bet, at the cost of skipping some.

Protocol, the same one that killed the recalibration idea:

    base model   trained once on fights through 2023-12-06
    meta train   2024
    selection    2025          - every variant judged here
    confirmation after 2025-12-06, looked at once

It has to beat three free baselines at the same coverage, or it is not worth
having:

    1. no selection          - the base rate
    2. base confidence       - just take the most confident picks
    3. experience gate       - both fighters with 6+ prior UFC bouts

Baseline 2 matters most. A meta-model that only rediscovers "confident picks
are better" has added nothing over a sort.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.calibrator_mismatch import fit_models, proba
from name_resolution import norm_name

BASE_END = pd.Timestamp('2023-12-06')
META_TRAIN_END = pd.Timestamp('2024-12-31')
SELECT_END = pd.Timestamp('2025-12-06')
COVERAGES = (1.0, 0.5, 0.3, 0.2)


def prior_fight_counts(ufc):
    """Each fighter's UFC bouts before each fight. The strongest signal so far."""
    seen, r_prior, b_prior = {}, [], []
    for row in ufc.itertuples(index=False):
        rn, bn = norm_name(row.r_name), norm_name(row.b_name)
        r_prior.append(seen.get(rn, 0))
        b_prior.append(seen.get(bn, 0))
        seen[rn] = seen.get(rn, 0) + 1
        seen[bn] = seen.get(bn, 0) + 1
    return np.array(r_prior), np.array(b_prior)


def meta_features(X, p_base, r_prior, b_prior):
    """Compact and interpretable - a few hundred training rows will not carry more."""
    conf = np.abs(2 * p_base - 1)
    out = pd.DataFrame({
        'p_base': p_base,
        'confidence': conf,
        'min_prior_fights': np.minimum(r_prior, b_prior),
        'total_prior_fights': r_prior + b_prior,
    }, index=X.index)
    for col in ('combined_uncertainty', 'consistency_diff', 'data_sparsity_diff',
                'is_5rnd', 'is_title', 'age_diff', 'mmr_diff'):
        if col in X.columns:
            out[col] = X[col].values
    if 'mmr_diff' in out:
        out['abs_mmr_diff'] = out['mmr_diff'].abs()
    return out.fillna(0.0)


def accuracy_at_coverage(score, correct, coverage):
    """Accuracy over the top slice by score. Higher score = expected more reliable."""
    n = max(1, int(round(len(score) * coverage)))
    order = np.argsort(-np.asarray(score))[:n]
    return float(np.asarray(correct)[order].mean()), n


def report(name, score, correct):
    cells = []
    for cov in COVERAGES:
        acc, n = accuracy_at_coverage(score, correct, cov)
        cells.append(f"{acc:>7.1%} (n={n:>3})")
    print(f"  {name:<28}" + "".join(f"{c:>18}" for c in cells))


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/meta_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True)
    dates = ufc['date']
    r_prior, b_prior = prior_fight_counts(ufc)

    base_idx = np.where((dates <= BASE_END).values)[0]
    models = fit_models(X.iloc[base_idx], y[base_idx])
    print(f"base model: {len(base_idx):,} fights through {BASE_END.date()}")

    # Out-of-sample base predictions for everything after the base period.
    after = (dates > BASE_END).values
    p_all = np.full(len(X), np.nan)
    p_all[after] = proba(models, X[after], 3)
    correct_all = ((p_all >= 0.5).astype(float) == y).astype(float)

    meta_X = meta_features(X, np.nan_to_num(p_all, nan=0.5), r_prior, b_prior)

    train_m = (after & (dates <= META_TRAIN_END).values)
    select_m = ((dates > META_TRAIN_END) & (dates <= SELECT_END)).values
    confirm_m = (dates > SELECT_END).values
    print(f"meta train : {train_m.sum()} bouts (2024)")
    print(f"selection  : {select_m.sum()} bouts (2025)")
    print(f"confirm    : {confirm_m.sum()} bouts (after {SELECT_END.date()})")

    header = "".join(f"{f'top {int(c*100)}%':>18}" for c in COVERAGES)
    print(f"\n{'SELECTION (2025)':<30}{header}")
    print("-" * (30 + 18 * len(COVERAGES)))

    sel_correct = correct_all[select_m]
    report("no selection (base rate)", np.zeros(select_m.sum()), sel_correct)
    report("base confidence", meta_X.loc[select_m, 'confidence'].values, sel_correct)
    report("experience gate (6+)", (meta_X.loc[select_m, 'min_prior_fights'] >= 6).astype(float).values,
           sel_correct)

    candidates = {
        'meta: logistic': LogisticRegression(max_iter=2000, C=1.0),
        'meta: gbm (depth 2)': GradientBoostingClassifier(max_depth=2, n_estimators=150,
                                                          learning_rate=0.05, random_state=42),
        'meta: gbm (depth 3)': GradientBoostingClassifier(max_depth=3, n_estimators=200,
                                                          learning_rate=0.05, random_state=42),
    }
    scored = {}
    for label, model in candidates.items():
        model.fit(meta_X[train_m], correct_all[train_m])
        s = model.predict_proba(meta_X[select_m])[:, 1]
        scored[label] = s
        report(label, s, sel_correct)

    # Choose on 2025 by accuracy at 30% coverage - the slice you would act on.
    best_label = max(scored, key=lambda k: accuracy_at_coverage(scored[k], sel_correct, 0.3)[0])
    print(f"\nchosen on 2025: {best_label}")

    # Refit on 2024+2025, then look at the confirmation period once.
    final = candidates[best_label]
    fit_m = train_m | select_m
    final.fit(meta_X[fit_m], correct_all[fit_m])

    print(f"\n{'CONFIRMATION (held out)':<30}{header}")
    print("-" * (30 + 18 * len(COVERAGES)))
    con_correct = correct_all[confirm_m]
    report("no selection (base rate)", np.zeros(confirm_m.sum()), con_correct)
    report("base confidence", meta_X.loc[confirm_m, 'confidence'].values, con_correct)
    report("experience gate (6+)",
           (meta_X.loc[confirm_m, 'min_prior_fights'] >= 6).astype(float).values, con_correct)
    report(best_label, final.predict_proba(meta_X[confirm_m])[:, 1], con_correct)

    if hasattr(final, 'feature_importances_'):
        imp = sorted(zip(meta_X.columns, final.feature_importances_),
                     key=lambda kv: -kv[1])[:8]
        print("\n  what the meta-model leans on:")
        for name, v in imp:
            print(f"    {name:<24}{v:>7.3f}")


if __name__ == '__main__':
    main()
