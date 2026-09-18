"""Has the model always ranked fights poorly, or only in 2026?

Accuracy hides this. A model can score 73% by riding favourites while ranking
barely better than chance. AUC measures ranking alone, independent of
calibration, so it separates "the model knows who wins" from "the model knows
which side is favoured".

This matters for what to do next:

  AUC ~0.75 through 2025 and ~0.57 in 2026
      the model does hold information; 2026 is anomalous and the answer is
      probably patience, not new data

  AUC mediocre throughout
      the model never had much discriminative information, accuracy came from
      following favourites, and only better inputs will change that

Walk-forward: for each year, train on everything before it and score that year.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.calibrator_mismatch import fit_models, proba


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/auc_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = pd.to_datetime(engine.ufc_valid['date'].reset_index(drop=True))
    years = dates.dt.year

    print(f"\n{'year':<8}{'n':>6}{'accuracy':>11}{'AUC':>9}{'Brier':>9}"
          f"{'mean p':>9}{'p std':>8}")
    print("-" * 60)

    rows = []
    for year in sorted(years.unique()):
        if year < 2015:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < 500 or test.sum() < 50:
            continue
        models = fit_models(X[train], y[train])
        p = proba(models, X[test], 3)
        yt = y[test]
        if len(np.unique(yt)) < 2:
            continue
        acc = float(((p >= 0.5).astype(float) == yt).mean())
        auc = float(roc_auc_score(yt, p))
        brier = float(np.mean((p - yt) ** 2))
        rows.append({'year': int(year), 'n': int(test.sum()), 'accuracy': acc,
                     'auc': auc, 'brier': brier})
        print(f"{year:<8}{test.sum():>6}{acc:>11.1%}{auc:>9.3f}{brier:>9.4f}"
              f"{p.mean():>9.3f}{p.std():>8.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        return
    before = df[df['year'] < 2026]
    after = df[df['year'] == 2026]
    print("\n" + "-" * 60)
    print(f"  2015-2025 mean AUC : {before['auc'].mean():.3f} "
          f"(range {before['auc'].min():.3f}-{before['auc'].max():.3f})")
    if not after.empty:
        print(f"  2026 AUC           : {after['auc'].iloc[0]:.3f}")
        drop = before['auc'].mean() - after['auc'].iloc[0]
        print(f"  drop               : {drop:+.3f}")
        sd = before['auc'].std()
        print(f"  2015-2025 std      : {sd:.3f}  "
              f"-> 2026 is {drop / sd:.1f} standard deviations below the mean"
              if sd > 0 else "")
    print("\n  For reference, the market scored AUC 0.741 on the 2026 fights")
    print("  where both sides were priced (engine/edge.py).")
    df.to_csv(ENGINE / 'experiments' / 'auc_by_year.csv', index=False)


if __name__ == '__main__':
    main()
