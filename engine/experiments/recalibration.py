"""Does fitting the calibrator on recent fights fix the overconfidence?

Live picks stated at 80-90% win about 62% of the time. The Platt layer is
currently fitted on the last 10% of history by date. This asks whether fitting
it on a recent window instead - 12, 18 or 24 months - closes that gap on 2026
bouts, and compares isotonic regression as an alternative.

Guard rails, because calibration experiments leak easily:

- The base models are refit for every variant on data strictly before that
  variant's calibration window. A calibrator fitted on rows the model trained
  on sees in-sample probabilities, which are overconfident, and would look
  falsely good.
- Nothing from 2026 is used for fitting anything. It is the held-out future.

    python engine/experiments/recalibration.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

CUTOFF = pd.Timestamp('2025-12-06')     # everything after this is held out


def fit_ensemble(X_train, y_train, seed=42):
    """The production recipe, minus the calibrator."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=seed).fit(Xs, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                random_state=seed, n_jobs=-1).fit(Xs, y_train)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        random_state=seed, eval_metric='logloss',
                        verbosity=0).fit(Xs, y_train)
    return scaler, lr, rf, xgb


def ensemble_proba(models, X):
    scaler, lr, rf, xgb = models
    Xs = scaler.transform(X)
    return (lr.predict_proba(Xs)[:, 1] + rf.predict_proba(Xs)[:, 1]
            + xgb.predict_proba(Xs)[:, 1]) / 3


def evaluate(p, y):
    """Accuracy, Brier and expected calibration error on the stated confidence."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    picked = (p >= 0.5).astype(float)
    won = (picked == y).astype(float)
    stated = np.where(p >= 0.5, p, 1 - p)
    ece, n = 0.0, len(p)
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = (stated >= lo) & (stated < hi)
        if m.sum():
            ece += m.sum() / n * abs(won[m].mean() - stated[m].mean())
    return {
        'accuracy': float(won.mean()),
        'brier': float(np.mean((p - y) ** 2)),
        'log_loss': float(np.mean(-np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-15, None)))),
        'ece': float(ece),
    }


def calibration_table(p, y, label):
    p, y = np.asarray(p, float), np.asarray(y, float)
    picked = (p >= 0.5).astype(float)
    won = (picked == y).astype(float)
    stated = np.where(p >= 0.5, p, 1 - p)
    print(f"\n  {label}")
    print(f"    {'band':<12}{'n':>5}{'stated':>9}{'actual':>9}{'gap':>8}")
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = (stated >= lo) & (stated < hi)
        if not m.sum():
            continue
        print(f"    {lo:.0%}-{min(hi,1):.0%}{'':<5}{m.sum():>5}{stated[m].mean():>9.1%}"
              f"{won[m].mean():>9.1%}{won[m].mean()-stated[m].mean():>+8.1%}")


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/recal_scratch_history.json')
    print("building features (this takes a few minutes)...")
    import predict_card as engine                       # noqa: trains on import

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = engine.ufc_valid['date'].reset_index(drop=True)

    future = dates > CUTOFF
    print(f"\nheld-out 2026 bouts: {int(future.sum())}")
    X_future, y_future = X[future], y[future.values]

    variants = []

    # Current recipe: train on the first 90% of history, calibrate on the last 10%.
    past_idx = np.where(~future.values)[0]
    split = int(len(past_idx) * 0.90)
    variants.append(('current (last 10% of history)',
                     past_idx[:split], past_idx[split:]))

    # Recent-window calibrators. The model stops where the window starts.
    for months in (12, 18, 24):
        start = CUTOFF - pd.DateOffset(months=months)
        cal_mask = (dates > start) & (dates <= CUTOFF)
        train_mask = dates <= start
        variants.append((f'last {months} months',
                         np.where(train_mask.values)[0], np.where(cal_mask.values)[0]))

    print(f"\n{'variant':<34}{'train':>7}{'cal':>7}{'acc':>8}{'brier':>9}{'logloss':>9}{'ECE':>8}")
    print("-" * 82)

    results = {}
    for label, train_idx, cal_idx in variants:
        models = fit_ensemble(X.iloc[train_idx], y[train_idx])
        p_cal = ensemble_proba(models, X.iloc[cal_idx])
        p_future_raw = ensemble_proba(models, X_future)

        # uncalibrated, for reference
        if 'uncalibrated' not in results:
            m = evaluate(p_future_raw, y_future)
            results['uncalibrated'] = (p_future_raw, m)
            print(f"{'raw ensemble (no calibrator)':<34}{len(train_idx):>7}{'-':>7}"
                  f"{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['ece']:>8.3f}")

        platt = LogisticRegression(max_iter=1000).fit(p_cal.reshape(-1, 1), y[cal_idx])
        p_platt = platt.predict_proba(p_future_raw.reshape(-1, 1))[:, 1]
        m = evaluate(p_platt, y_future)
        results[f'platt {label}'] = (p_platt, m)
        print(f"{'Platt: ' + label:<34}{len(train_idx):>7}{len(cal_idx):>7}"
              f"{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['ece']:>8.3f}")

        if len(cal_idx) >= 300:
            iso = IsotonicRegression(out_of_bounds='clip').fit(p_cal, y[cal_idx])
            p_iso = np.clip(iso.predict(p_future_raw), 1e-6, 1 - 1e-6)
            m = evaluate(p_iso, y_future)
            results[f'isotonic {label}'] = (p_iso, m)
            print(f"{'Isotonic: ' + label:<34}{len(train_idx):>7}{len(cal_idx):>7}"
                  f"{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['ece']:>8.3f}")

    print("\n" + "=" * 82)
    print("CALIBRATION ON THE HELD-OUT 2026 BOUTS")
    print("=" * 82)
    best = min((k for k in results if k != 'uncalibrated'),
               key=lambda k: results[k][1]['ece'])
    for key in ('uncalibrated', 'platt current (last 10% of history)', best):
        if key in results:
            calibration_table(results[key][0], y_future, key)
    print(f"\nlowest ECE: {best}  ({results[best][1]['ece']:.3f} "
          f"vs {results['platt current (last 10% of history)'][1]['ece']:.3f} for the current recipe)")

    controlled(X, y, dates, X_future, y_future)


def controlled(X, y, dates, X_future, y_future):
    """Same base model for every variant - only the calibration window moves.

    The comparison above lets the training set move with the window, and its
    accuracy ordering tracks training size almost exactly, so that difference
    cannot be credited to calibration. Here the model is fitted once on
    everything up to a fixed date and every calibrator is fitted on a window
    after it, which isolates the calibrator.
    """
    base_end = pd.Timestamp('2023-12-06')
    train_idx = np.where((dates <= base_end).values)[0]
    print("\n" + "=" * 82)
    print("CONTROLLED: one base model, only the calibration window varies")
    print("=" * 82)
    print(f"base model trained on {len(train_idx):,} fights through {base_end.date()}")

    models = fit_ensemble(X.iloc[train_idx], y[train_idx])
    p_future_raw = ensemble_proba(models, X_future)
    m = evaluate(p_future_raw, y_future)
    print(f"\n{'calibrator':<30}{'cal n':>7}{'acc':>8}{'brier':>9}{'logloss':>9}{'ECE':>8}")
    print("-" * 71)
    print(f"{'none (raw ensemble)':<30}{'-':>7}{m['accuracy']:>8.1%}{m['brier']:>9.4f}"
          f"{m['log_loss']:>9.4f}{m['ece']:>8.3f}")

    rows = []
    for months in (6, 12, 18, 24):
        start = CUTOFF - pd.DateOffset(months=months)
        if start <= base_end:
            start = base_end
        cal_mask = (dates > start) & (dates <= CUTOFF)
        cal_idx = np.where(cal_mask.values)[0]
        if len(cal_idx) < 100:
            continue
        p_cal = ensemble_proba(models, X.iloc[cal_idx])
        platt = LogisticRegression(max_iter=1000).fit(p_cal.reshape(-1, 1), y[cal_idx])
        p = platt.predict_proba(p_future_raw.reshape(-1, 1))[:, 1]
        m = evaluate(p, y_future)
        rows.append((months, len(cal_idx), m, p))
        print(f"{'Platt, last ' + str(months) + ' months':<30}{len(cal_idx):>7}"
              f"{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['log_loss']:>9.4f}{m['ece']:>8.3f}")

    # everything after the base model, i.e. the widest available window
    cal_idx = np.where(((dates > base_end) & (dates <= CUTOFF)).values)[0]
    p_cal = ensemble_proba(models, X.iloc[cal_idx])
    platt = LogisticRegression(max_iter=1000).fit(p_cal.reshape(-1, 1), y[cal_idx])
    p_all = platt.predict_proba(p_future_raw.reshape(-1, 1))[:, 1]
    m_all = evaluate(p_all, y_future)
    print(f"{'Platt, all 24mo since base':<30}{len(cal_idx):>7}{m_all['accuracy']:>8.1%}"
          f"{m_all['brier']:>9.4f}{m_all['log_loss']:>9.4f}{m_all['ece']:>8.3f}")

    if rows:
        best = min(rows, key=lambda r: r[2]['ece'])
        calibration_table(p_future_raw, y_future, 'no calibrator')
        calibration_table(best[3], y_future, f'Platt, last {best[0]} months')


if __name__ == '__main__':
    main()
