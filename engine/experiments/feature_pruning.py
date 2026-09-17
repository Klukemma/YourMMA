"""Does cutting the dead and harmful features actually help?

The audit found, on 57 winner features:

  - winrate_diff alone is worth +0.093 accuracy; every other feature is
    +0.012 or less, an order of magnitude smaller
  - 15 features have NEGATIVE permutation importance - the model does better
    without them
  - 12 contribute essentially nothing
  - same_cluster is identically zero on every row

Protocol, so the feature set is not chosen by looking at the answer:

  base model   trained on fights through 2023-12-06
  selection    importance measured on 2024, candidate sets scored on 2025
  confirmation 2026, looked at once

Note the audit itself measured importance on 2024 onward, which includes the
confirmation period. That is fine for describing the data but not for choosing,
so importance is recomputed here on 2024 alone.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.calibrator_mismatch import fit_models, proba

BASE_END = pd.Timestamp('2023-12-06')
SELECT_START = pd.Timestamp('2025-01-01')
CONFIRM_START = pd.Timestamp('2025-12-06')


class Wrapped:
    def __init__(self, models):
        self.models = models

    def fit(self, *a):
        return self

    def predict(self, X):
        return (proba(self.models, X, 3) >= 0.5).astype(float)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y, float)).mean())


def evaluate(models, X, y):
    p = proba(models, X, 3)
    picked = (p >= 0.5).astype(float)
    return {
        'accuracy': float((picked == y).mean()),
        'brier': float(np.mean((p - y) ** 2)),
        'n': len(y),
    }


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/prune_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = engine.ufc_valid['date'].reset_index(drop=True)

    train = (dates <= BASE_END).values
    imp_period = ((dates > BASE_END) & (dates < SELECT_START)).values   # 2024
    select = ((dates >= SELECT_START) & (dates < CONFIRM_START)).values  # 2025
    confirm = (dates >= CONFIRM_START).values

    print(f"train {train.sum()}, importance {imp_period.sum()} (2024), "
          f"select {select.sum()} (2025), confirm {confirm.sum()}")

    base = fit_models(X[train], y[train])
    print("\nmeasuring importance on 2024 only...")
    res = permutation_importance(Wrapped(base), X[imp_period], y[imp_period],
                                 n_repeats=5, random_state=42, n_jobs=1)
    imp = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)

    candidates = {
        'all 57 features': list(X.columns),
        'drop negative-importance': list(imp[imp >= 0].index),
        'drop negative and near-zero': list(imp[imp >= 0.0005].index),
        'top 20': list(imp.head(20).index),
        'top 10': list(imp.head(10).index),
        'top 5': list(imp.head(5).index),
        'winrate_diff alone': ['winrate_diff'],
    }

    print(f"\n{'SELECTION (2025)':<32}{'feats':>7}{'acc':>8}{'brier':>9}")
    print("-" * 56)
    scored = {}
    for label, cols in candidates.items():
        if not cols:
            continue
        m = fit_models(X.loc[train, cols], y[train])
        r = evaluate(m, X.loc[select, cols], y[select])
        scored[label] = (cols, r)
        print(f"  {label:<30}{len(cols):>7}{r['accuracy']:>8.1%}{r['brier']:>9.4f}")

    best = max(scored, key=lambda k: scored[k][1]['accuracy'])
    print(f"\nchosen on 2025: {best} ({len(scored[best][0])} features)")

    print(f"\n{'CONFIRMATION (2026, seen once)':<32}{'feats':>7}{'acc':>8}{'brier':>9}")
    print("-" * 56)
    for label in ('all 57 features', best):
        cols = scored[label][0]
        m = fit_models(X.loc[train, cols], y[train])
        r = evaluate(m, X.loc[confirm, cols], y[confirm])
        print(f"  {label:<30}{len(cols):>7}{r['accuracy']:>8.1%}{r['brier']:>9.4f}")

    print(f"\nfeatures kept by '{best}':")
    for c in scored[best][0]:
        print(f"    {c:<30}{imp.get(c, float('nan')):+.4f}")
    print(f"\ndropped ({len(X.columns) - len(scored[best][0])}):")
    dropped = [c for c in X.columns if c not in scored[best][0]]
    for c in dropped:
        print(f"    {c:<30}{imp.get(c, float('nan')):+.4f}")


if __name__ == '__main__':
    main()
