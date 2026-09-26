"""Pick a calibration recipe on one period, then confirm it on a later one.

Earlier runs compared calibration windows by their score on 2026, which is the
same data used to judge the final answer - keep doing that and the recipe is
fitted to the test set. So the choice is made on 2025 and only the winner is
carried to 2026, which is then seen exactly once.

Two knobs, both suggested by earlier runs:
  window - how much recent history the calibrator is fitted on
  C      - Platt regularisation. Production uses 1e10, effectively none, on a
           few hundred rows.

    python engine/experiments/calibration_selection.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.calibrator_mismatch import fit_models, proba
from experiments.recalibration import calibration_table, evaluate

BASE_END = pd.Timestamp('2023-12-06')     # base model trains up to here
SELECT_FROM = pd.Timestamp('2025-01-01')  # selection period: calendar 2025
CONFIRM_FROM = pd.Timestamp('2025-12-06')  # confirmation period: after this

WINDOWS = (3, 6, 12, 'all')
CS = (1e10, 1.0, 0.1)


def cal_mask(dates, window, end):
    if window == 'all':
        return (dates > BASE_END) & (dates <= end)
    return (dates > end - pd.DateOffset(months=window)) & (dates <= end)


def main():
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/calsel_scratch.json')
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = engine.ufc_valid['date'].reset_index(drop=True)

    train_idx = np.where((dates <= BASE_END).values)[0]
    models = fit_models(X.iloc[train_idx], y[train_idx])
    print(f"base model: {len(train_idx):,} fights through {BASE_END.date()}")

    select = ((dates >= SELECT_FROM) & (dates < CONFIRM_FROM)).values
    confirm = (dates > CONFIRM_FROM).values
    print(f"selection : {select.sum()} bouts in 2025")
    print(f"confirm   : {confirm.sum()} bouts after {CONFIRM_FROM.date()}\n")

    def build(window, C, cal_end):
        idx = np.where(cal_mask(dates, window, cal_end).values)[0]
        if len(idx) < 80:
            return None, 0
        p_cal = proba(models, X.iloc[idx], 3)
        return LogisticRegression(C=C, solver='lbfgs',
                                  max_iter=1000).fit(p_cal.reshape(-1, 1), y[idx]), len(idx)

    # --- selection, scored on 2025 only ---
    p_select_raw = proba(models, X[select], 3)
    print(f"{'recipe':<26}{'cal n':>7}{'acc':>8}{'brier':>9}{'ECE':>8}   (2025)")
    print("-" * 62)
    m = evaluate(p_select_raw, y[select])
    print(f"{'no calibrator':<26}{'-':>7}{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['ece']:>8.3f}")
    scored = []
    for window in WINDOWS:
        for C in CS:
            platt, n = build(window, C, SELECT_FROM)
            if platt is None:
                continue
            p = platt.predict_proba(p_select_raw.reshape(-1, 1))[:, 1]
            m = evaluate(p, y[select])
            scored.append((m['ece'], window, C, n, m))
            label = f"{window if window=='all' else str(window)+'mo'}, C={C:g}"
            print(f"{label:<26}{n:>7}{m['accuracy']:>8.1%}{m['brier']:>9.4f}{m['ece']:>8.3f}")

    scored.sort()
    _, best_window, best_C, _, best_m = scored[0]
    print(f"\nchosen on 2025: window={best_window}, C={best_C:g}  (ECE {best_m['ece']:.3f})")

    # --- confirmation, this period scored once ---
    print("\n" + "=" * 62)
    print(f"CONFIRMATION on {confirm.sum()} held-out bouts after {CONFIRM_FROM.date()}")
    print("=" * 62)
    p_confirm_raw = proba(models, X[confirm], 3)
    print(f"{'recipe':<34}{'acc':>8}{'brier':>9}{'ECE':>8}")
    print("-" * 59)
    m_raw = evaluate(p_confirm_raw, y[confirm])
    print(f"{'no calibrator':<34}{m_raw['accuracy']:>8.1%}{m_raw['brier']:>9.4f}{m_raw['ece']:>8.3f}")

    platt_prod, n_prod = build('all', 1e10, CONFIRM_FROM)
    p_prod = platt_prod.predict_proba(p_confirm_raw.reshape(-1, 1))[:, 1]
    m_prod = evaluate(p_prod, y[confirm])
    print(f"{'production recipe (all, C=1e10)':<34}{m_prod['accuracy']:>8.1%}"
          f"{m_prod['brier']:>9.4f}{m_prod['ece']:>8.3f}")

    platt_best, n_best = build(best_window, best_C, CONFIRM_FROM)
    p_best = platt_best.predict_proba(p_confirm_raw.reshape(-1, 1))[:, 1]
    m_best = evaluate(p_best, y[confirm])
    label = f"chosen ({best_window if best_window=='all' else str(best_window)+'mo'}, C={best_C:g})"
    print(f"{label:<34}{m_best['accuracy']:>8.1%}{m_best['brier']:>9.4f}{m_best['ece']:>8.3f}")

    print(f"\nECE {m_prod['ece']:.3f} -> {m_best['ece']:.3f}"
          f"   Brier {m_prod['brier']:.4f} -> {m_best['brier']:.4f}"
          f"   accuracy {m_prod['accuracy']:.1%} -> {m_best['accuracy']:.1%}")
    calibration_table(p_prod, y[confirm], 'production recipe')
    calibration_table(p_best, y[confirm], label)


if __name__ == '__main__':
    main()
