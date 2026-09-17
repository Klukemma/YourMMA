"""What are the features actually contributing?

Before changing how any variable behaves, find out which ones carry signal,
which are dead weight, and which are broken. Reports, for every feature in the
winner model:

  - how often it is missing or constant
  - its spread, so scale bugs show up
  - its correlation with the outcome
  - its permutation importance on held-out data, which unlike a tree's own
    importance cannot be inflated by high cardinality
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
EVAL_START = pd.Timestamp('2024-01-01')


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/audit_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = engine.ufc_valid['date'].reset_index(drop=True)
    print(f"\nwinner model uses {len(X.columns)} features on {len(X):,} fights")

    train = (dates <= BASE_END).values
    evalm = (dates >= EVAL_START).values

    # --- health of each column -------------------------------------------
    rows = []
    for col in X.columns:
        s = X[col]
        rows.append({
            'feature': col,
            'nan_pct': float(s.isna().mean() * 100),
            'zero_pct': float((s.fillna(0) == 0).mean() * 100),
            'std': float(s.std()),
            'unique': int(s.nunique()),
            'corr': float(pd.Series(s).corr(pd.Series(y))) if s.std() > 0 else 0.0,
        })
    health = pd.DataFrame(rows)

    dead = health[(health['std'] == 0) | (health['unique'] <= 1)]
    print(f"\nconstant / dead features: {len(dead)}")
    for _, r in dead.iterrows():
        print(f"  {r['feature']}")

    near_dead = health[(health['unique'] <= 3) & (health['std'] > 0)]
    print(f"\nnear-constant (<=3 distinct values): {len(near_dead)}")
    for _, r in near_dead.iterrows():
        print(f"  {r['feature']:<28} unique={r['unique']}  std={r['std']:.4f}")

    mostly_zero = health[health['zero_pct'] > 90]
    print(f"\nover 90% zero: {len(mostly_zero)}")
    for _, r in mostly_zero.iterrows():
        print(f"  {r['feature']:<28} {r['zero_pct']:.1f}% zero")

    wild = health[health['std'] > 100]
    print(f"\nsuspiciously large spread (std > 100, a scale-bug smell): {len(wild)}")
    for _, r in wild.sort_values('std', ascending=False).iterrows():
        print(f"  {r['feature']:<28} std={r['std']:>10.1f}")

    # --- permutation importance on held-out fights ------------------------
    print("\nfitting base model for permutation importance...")
    models = fit_models(X[train], y[train])
    scaler, lr, rf, xgb, mlp = models

    class Ensemble:
        """Minimal sklearn-shaped wrapper so permutation_importance can score it."""
        def fit(self, *a):
            return self

        def predict(self, Xin):
            return (proba(models, Xin, 3) >= 0.5).astype(float)

        def score(self, Xin, yin):
            return float((self.predict(Xin) == np.asarray(yin, float)).mean())

    result = permutation_importance(
        Ensemble(), X[evalm], y[evalm], n_repeats=5, random_state=42, n_jobs=1)
    imp = pd.DataFrame({'feature': X.columns,
                        'importance': result.importances_mean,
                        'sd': result.importances_std}).sort_values(
        'importance', ascending=False)

    print(f"\n{'top 20 by permutation importance':<34}{'drop in acc':>12}{'sd':>9}")
    print("-" * 56)
    for _, r in imp.head(20).iterrows():
        print(f"  {r['feature']:<32}{r['importance']:>+11.4f}{r['sd']:>9.4f}")

    print(f"\n{'features that HURT when used (negative)':<34}{'drop in acc':>12}")
    print("-" * 56)
    harmful = imp[imp['importance'] < -0.001]
    for _, r in harmful.iterrows():
        print(f"  {r['feature']:<32}{r['importance']:>+11.4f}")
    print(f"  ({len(harmful)} of {len(imp)})")

    useless = imp[imp['importance'].abs() < 0.0005]
    print(f"\ncontributing essentially nothing (|importance| < 0.0005): "
          f"{len(useless)} of {len(imp)}")

    out = ENGINE / 'experiments' / 'feature_audit.csv'
    health.merge(imp, on='feature').sort_values('importance', ascending=False) \
          .to_csv(out, index=False)
    print(f"\nfull table written to {out}")


if __name__ == '__main__':
    main()
