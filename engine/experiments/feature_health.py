"""Which of the model's features carry information, and which are furniture?

The static audit catches constants on the wrong scale. It cannot see a feature
that is well-formed and simply empty. base_prob was the example: perfectly
valid, spanning 0.448-0.565 with a std of 0.0153, a documented headline feature
that was very nearly a constant. Nothing about the source code says so - only
the built matrix does.

Five checks, all on the training rows so nothing here looks at held-out data:

  NEAR CONSTANT      std close to zero, or one value covering most rows. The
                     model cannot split on what does not vary.
  MOSTLY IMPUTED     a large share of rows sitting exactly on the fill value,
                     which means the feature is mostly saying "unknown" while
                     looking like it says something.
  DUPLICATE PAIR     two features correlated above 0.99. One of them is doing
                     nothing, and together they split importance so neither
                     looks important.
  DEGENERATE IN 2026 varies across history but not in the recent period, so it
                     has quietly stopped working.
  NO SIGNAL          |AUC - 0.5| below a threshold on the training rows. Weak
                     alone is not useless in an ensemble, so this is reported
                     last and as information, not a verdict.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

NEAR_CONSTANT_STD = 1e-6
DOMINANT_VALUE_SHARE = 0.95
IMPUTED_SHARE = 0.25
DUPLICATE_CORR = 0.99
WEAK_AUC = 0.02


def near_constant(X):
    out = []
    for col in X.columns:
        values = X[col]
        std = float(values.std())
        counts = values.value_counts(normalize=True)
        top_share = float(counts.iloc[0]) if len(counts) else 1.0
        if std <= NEAR_CONSTANT_STD or top_share >= DOMINANT_VALUE_SHARE:
            out.append({"feature": col, "std": std, "top_share": top_share,
                        "unique": int(values.nunique())})
    return sorted(out, key=lambda r: -r["top_share"])


def mostly_imputed(X, fill=0.0):
    """Rows sitting exactly on the fill value.

    A difference feature is legitimately 0 when two fighters match, so this is
    a flag for reading, not a defect on its own.
    """
    out = []
    for col in X.columns:
        share = float((X[col] == fill).mean())
        if share >= IMPUTED_SHARE:
            out.append({"feature": col, "share": share})
    return sorted(out, key=lambda r: -r["share"])


def duplicate_pairs(X):
    corr = X.corr(numeric_only=True).abs()
    seen, out = set(), []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= DUPLICATE_CORR and (a, b) not in seen:
                seen.add((a, b))
                out.append({"a": a, "b": b, "corr": float(value)})
    return sorted(out, key=lambda r: -r["corr"])


def degenerate_recently(X, dates, year=2026):
    recent = X[(dates.dt.year == year).values]
    if len(recent) < 50:
        return []
    out = []
    for col in X.columns:
        if X[col].std() > NEAR_CONSTANT_STD and recent[col].std() <= NEAR_CONSTANT_STD:
            out.append({"feature": col, "overall_std": float(X[col].std())})
    return out


def univariate_auc(X, y):
    rows = []
    for col in X.columns:
        values = X[col].to_numpy()
        if np.all(values == values[0]):
            rows.append({"feature": col, "auc": 0.5})
            continue
        try:
            rows.append({"feature": col, "auc": float(roc_auc_score(y, values))})
        except ValueError:
            rows.append({"feature": col, "auc": 0.5})
    return sorted(rows, key=lambda r: abs(r["auc"] - 0.5))


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/feature_health_scratch.json")
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = pd.to_datetime(engine.ufc_valid["date"].reset_index(drop=True))

    # Train only, so nothing here is chosen by looking at held-out results.
    train = np.arange(len(X)) < int(len(X) * 0.70)
    Xt, yt, dt = X[train], y[train], dates[train]
    print(f"{Xt.shape[0]:,} training rows x {Xt.shape[1]} features\n")

    flat = near_constant(Xt)
    print("=" * 70)
    print(f"NEAR CONSTANT: {len(flat)}")
    print("  the model cannot split on what does not vary")
    print("=" * 70)
    for r in flat:
        print(f"  {r['feature']:<28} std {r['std']:>10.6f}  "
              f"one value covers {r['top_share']:>6.1%}  ({r['unique']} distinct)")
    if not flat:
        print("  none")

    imputed = mostly_imputed(Xt)
    print(f"\n{'=' * 70}\nMOSTLY THE FILL VALUE: {len(imputed)}")
    print("  a difference is legitimately 0 for an even matchup, so read these")
    print("=" * 70)
    for r in imputed[:15]:
        print(f"  {r['feature']:<28} exactly 0 on {r['share']:>6.1%} of rows")

    dupes = duplicate_pairs(Xt)
    print(f"\n{'=' * 70}\nDUPLICATE PAIRS (|r| >= {DUPLICATE_CORR}): {len(dupes)}")
    print("  one is redundant, and together they split importance")
    print("=" * 70)
    for r in dupes[:15]:
        print(f"  {r['corr']:.4f}  {r['a']:<28} == {r['b']}")
    if not dupes:
        print("  none")

    stale = degenerate_recently(X, dates)
    print(f"\n{'=' * 70}\nVARIES IN HISTORY BUT NOT IN 2026: {len(stale)}")
    print("=" * 70)
    for r in stale:
        print(f"  {r['feature']:<28} overall std {r['overall_std']:.4f}, 2026 std 0")
    if not stale:
        print("  none")

    aucs = univariate_auc(Xt, yt)
    weak = [r for r in aucs if abs(r["auc"] - 0.5) < WEAK_AUC]
    print(f"\n{'=' * 70}\nWEAKEST ALONE: {len(weak)} within {WEAK_AUC} of coin-flip")
    print("  weak alone is not useless in an ensemble - information, not a verdict")
    print("=" * 70)
    for r in weak[:20]:
        print(f"  {r['feature']:<28} AUC {r['auc']:.3f}")

    print(f"\n{'=' * 70}\nSTRONGEST ALONE")
    print("=" * 70)
    for r in sorted(aucs, key=lambda r: -abs(r["auc"] - 0.5))[:12]:
        print(f"  {r['feature']:<28} AUC {r['auc']:.3f}")

    out = ENGINE / "experiments" / "feature_health.csv"
    pd.DataFrame(aucs).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
