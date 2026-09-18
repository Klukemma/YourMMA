"""Are the rows this project writes distinguishable from the original ones?

Comparing column means across the sync boundary catches a unit error, which is
how the reach and weight bug was found, but it is a weak test: a column can
differ in shape, in its null pattern, or in what it counts while keeping the
same mean.

Adversarial validation is the strong version. Train a classifier to tell 2026
rows from 2025 rows using the features themselves. If it cannot - AUC near 0.5
- the rows are interchangeable and transform() is faithful. If it can, the
feature importances name exactly which columns give the game away.

Two periods written by the ORIGINAL builder run first as a control, because
real year-on-year drift is also learnable and the control measures how much.

Rating columns are excluded: they are computed by this project for every row,
so they carry no information about which builder wrote the raw data.

RESULT, after the unit fix:

    CONTROL  2024 vs 2025 (same builder)   AUC 0.477
    TEST     2025 vs 2026 (transform)      AUC 0.670   excess +0.193

    top distinguishing columns, in order:
      b_losses  b_wins  r_wins  r_losses  b_draws  r_draws

The control confirms same-builder rows are indistinguishable. The remaining
structural difference is entirely the professional win/loss/draw record, which
carry_forward_records leaves blank for a fighter it has never seen. That record
counts bouts outside the UFC, so it cannot be recomputed from this dataset.
"""

import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

df = pd.read_csv('engine/data/UFC_with_mmr_rebuilt_dedup.csv', low_memory=False).copy()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
drop = [c for c in num if 'mmr' in c or 'mu_pre' in c or 'sigma_pre' in c]
num = [c for c in num if c not in drop]

def adversarial(a, b, label):
    X = pd.concat([a[num], b[num]]).fillna(-999).to_numpy()
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                 random_state=0, n_jobs=-1)
    p = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, p)
    clf.fit(X, y)
    imp = sorted(zip(clf.feature_importances_, num), reverse=True)[:8]
    print(f"{label}")
    print(f"   n={len(a)} vs {len(b)}   separability AUC = {auc:.3f}")
    print("   top distinguishing columns:")
    for v, c in imp:
        print(f"     {c:<28}{v:.4f}")
    print()
    return auc

# control: two periods both written by the original builder
ctrl = adversarial(df[(df.date>='2024-01-01')&(df.date<='2024-12-31')],
                   df[(df.date>='2025-01-01')&(df.date<='2025-09-12')],
                   "CONTROL  2024 vs 2025 (both from the original builder)")

# test: original builder vs my transform
test = adversarial(df[(df.date>='2025-01-01')&(df.date<='2025-09-12')],
                   df[df.date>'2025-12-06'],
                   "TEST     2025 vs 2026 (original builder vs transform())")

print("="*60)
print(f"control separability : {ctrl:.3f}   (ordinary year-on-year drift)")
print(f"test separability    : {test:.3f}")
print(f"excess               : {test-ctrl:+.3f}")
