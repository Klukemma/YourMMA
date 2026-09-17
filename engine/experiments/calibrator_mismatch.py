"""The production calibrator is fitted on one quantity and applied to another.

predict_fight_prod averages three models:

    p_ens = (LR + RF + XGB) / 3                          predict_card.py:3877

but platt_prod was fitted on an average of four:

    p_ens_cal_full = (LR + RF + XGB + MLP) / 4           predict_card.py:3364

so the calibrator maps from a distribution it never saw. predict_fight(), the
non-production path, uses /4 and is consistent - only the production path,
which makes the actual picks, is mismatched.

This measures three variants on held-out 2026 bouts to decide the fix:

    A  3-model ensemble, calibrator fitted on 3  (consistent)
    B  4-model ensemble, calibrator fitted on 4  (consistent, matches predict_fight)
    C  3-model ensemble, calibrator fitted on 4  (what production does today)

each with the current calibration window and with a recent 6-month window.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.recalibration import CUTOFF, calibration_table, evaluate


def fit_models(X_train, y_train, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=seed).fit(Xs, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                random_state=seed, n_jobs=-1).fit(Xs, y_train)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        random_state=seed, eval_metric='logloss',
                        verbosity=0).fit(Xs, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), alpha=0.001, batch_size=32,
                        learning_rate='adaptive', learning_rate_init=0.001,
                        max_iter=500, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=20, random_state=seed).fit(Xs, y_train)
    return scaler, lr, rf, xgb, mlp


def proba(models, X, n_models):
    """Average of the first n_models - 3 excludes the MLP, 4 includes it."""
    scaler, lr, rf, xgb, mlp = models
    Xs = scaler.transform(X)
    parts = [lr.predict_proba(Xs)[:, 1], rf.predict_proba(Xs)[:, 1],
             xgb.predict_proba(Xs)[:, 1]]
    if n_models == 4:
        parts.append(mlp.predict_proba(Xs)[:, 1])
    return np.mean(parts, axis=0)


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/mismatch_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = engine.ufc_valid['date'].reset_index(drop=True)

    future = (dates > CUTOFF).values
    X_future, y_future = X[future], y[future]
    base_end = pd.Timestamp('2023-12-06')
    train_idx = np.where((dates <= base_end).values)[0]
    print(f"base model: {len(train_idx):,} fights through {base_end.date()}")
    print(f"held-out  : {int(future.sum())} bouts after {CUTOFF.date()}")

    models = fit_models(X.iloc[train_idx], y[train_idx])

    windows = {
        'current (all since base)': (dates > base_end) & (dates <= CUTOFF),
        'last 6 months': (dates > CUTOFF - pd.DateOffset(months=6)) & (dates <= CUTOFF),
    }

    print(f"\n{'variant':<46}{'acc':>8}{'brier':>9}{'logloss':>9}{'ECE':>8}")
    print("-" * 80)
    best = None
    for win_label, win_mask in windows.items():
        cal_idx = np.where(win_mask.values)[0]
        for label, fit_n, apply_n in [
                ('A  fit on 3, apply to 3  (consistent)', 3, 3),
                ('B  fit on 4, apply to 4  (consistent)', 4, 4),
                ('C  fit on 4, apply to 3  (production today)', 4, 3)]:
            p_cal = proba(models, X.iloc[cal_idx], fit_n)
            platt = LogisticRegression(C=1e10, solver='lbfgs',
                                       max_iter=1000).fit(p_cal.reshape(-1, 1), y[cal_idx])
            p_future = platt.predict_proba(
                proba(models, X_future, apply_n).reshape(-1, 1))[:, 1]
            m = evaluate(p_future, y_future)
            print(f"{label + ' | ' + win_label:<46}{m['accuracy']:>8.1%}"
                  f"{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['ece']:>8.3f}")
            if best is None or m['ece'] < best[1]['ece']:
                best = (f"{label} | {win_label}", m, p_future)
        print()

    # what the mismatch actually does to the numbers
    cal_idx = np.where(windows['current (all since base)'].values)[0]
    p3, p4 = proba(models, X.iloc[cal_idx], 3), proba(models, X.iloc[cal_idx], 4)
    print(f"3-model vs 4-model average on the same rows:")
    print(f"  mean absolute difference {np.abs(p3 - p4).mean():.4f}, "
          f"max {np.abs(p3 - p4).max():.4f}")
    print(f"  3-model spread {p3.std():.4f} vs 4-model {p4.std():.4f}")

    print(f"\nlowest ECE: {best[0]}  ({best[1]['ece']:.3f})")
    calibration_table(best[2], y_future, best[0])


if __name__ == '__main__':
    main()
