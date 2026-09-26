"""Does the production stack beat the raw ensemble, or hurt it?

Live predictions run through three layers:

    raw ensemble (LR+RF+XGB)  ->  Platt calibration  ->  context adjustment

Only the first is present in the walk-forward that reports 69.7%. The live
accuracy is 59.3%. This measures each layer against the same bouts, so the
gap can be attributed rather than guessed at.

The model is trained on data up to the cutoff only. Training on the full
synced file would let it see the very bouts being scored.

    python engine/experiments/adjustment_ablation.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

CUTOFF = '2025-12-06'          # what the live model had when it predicted


def build_truncated_csv(tmpdir):
    """The dataset as it stood when these predictions were made."""
    full = pd.read_csv(ENGINE / 'data' / 'UFC_with_mmr_rebuilt_dedup.csv', low_memory=False)
    full['date'] = pd.to_datetime(full['date'], errors='coerce')
    past = full[full['date'] <= CUTOFF]
    path = Path(tmpdir) / 'truncated.csv'
    past.to_csv(path, index=False)
    print(f"training data: {len(past):,} fights through {past['date'].max().date()}")
    return path


def metrics(probs, correct):
    probs, correct = np.asarray(probs, float), np.asarray(correct, float)
    picked = np.where(probs >= 0.5, 1.0, 0.0)
    hit = (picked == correct).mean()
    # Brier on the stated probability of the pick winning
    stated = np.where(probs >= 0.5, probs, 1 - probs)
    won = np.where(picked == correct, 1.0, 0.0)
    return {
        'accuracy': float(hit),
        'brier': float(np.mean((stated - won) ** 2)),
        'log_loss': float(np.mean(-np.log(np.clip(np.where(won == 1, stated, 1 - stated),
                                                  1e-15, None)))),
        'n': int(len(probs)),
    }


def calibration(probs, correct, label):
    stated = np.where(np.asarray(probs) >= 0.5, probs, 1 - np.asarray(probs))
    picked = np.where(np.asarray(probs) >= 0.5, 1.0, 0.0)
    won = (picked == np.asarray(correct, float)).astype(float)
    print(f"\n  {label}")
    print(f"    {'band':<12}{'n':>5}{'stated':>9}{'actual':>9}{'gap':>8}")
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        mask = (stated >= lo) & (stated < hi)
        if mask.sum() == 0:
            continue
        print(f"    {lo:.0%}-{min(hi,1):.0%}{'':<5}{mask.sum():>5}"
              f"{stated[mask].mean():>9.1%}{won[mask].mean():>9.1%}"
              f"{won[mask].mean()-stated[mask].mean():>+8.1%}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['UFC_CSV'] = str(build_truncated_csv(tmp))
        os.environ['PREDICTIONS_LOG'] = str(Path(tmp) / 'scratch_history.json')

        print("training the model on pre-cutoff data (this takes a few minutes)...")
        import predict_card as engine          # noqa: executing trains everything
        from grade import deduplicate, grade, load_results
        from name_resolution import norm_name

        # Outcomes come from the FULL file - that is the future we are scoring.
        index = load_results(ENGINE / 'data' / 'UFC_with_mmr_rebuilt_dedup.csv')
        history = json.loads((ENGINE / 'data' / 'prediction_history.json').read_text())
        graded, _ = grade(deduplicate(history['predictions']), index)
        print(f"\nscoring {len(graded)} graded bouts")

        rows = []
        for bout in graded:
            pred = engine.predict_fight_prod(
                bout['red_corner'], bout['blue_corner'],
                event_date=str(bout['actual_date'])[:10],
                context=None, verbose=False)
            if pred.get('status') == 'NO_DATA':
                continue
            red_won = norm_name(bout['actual_winner']) == norm_name(bout['red_corner'])
            rows.append({
                'p_ensemble': pred['p_ensemble'],
                'p_platt': pred['p_platt'],
                'logged': (bout['win_probability']
                           if norm_name(bout['predicted_winner']) == norm_name(bout['red_corner'])
                           else 1 - bout['win_probability']),
                'red_won': 1.0 if red_won else 0.0,
            })

        df = pd.DataFrame(rows)
        print(f"re-predicted {len(df)} of {len(graded)}")

        print("\n" + "=" * 62)
        print("EACH LAYER, SAME BOUTS")
        print("=" * 62)
        print(f"  {'variant':<34}{'n':>5}{'acc':>8}{'brier':>9}")
        for label, col in [("raw ensemble (LR+RF+XGB)", 'p_ensemble'),
                           ("+ Platt calibration", 'p_platt'),
                           ("+ context (as logged live)", 'logged')]:
            m = metrics(df[col], df['red_won'])
            print(f"  {label:<34}{m['n']:>5}{m['accuracy']:>8.1%}{m['brier']:>9.4f}")

        print("\n" + "=" * 62)
        print("CALIBRATION")
        print("=" * 62)
        for label, col in [("raw ensemble", 'p_ensemble'),
                           ("Platt calibrated", 'p_platt'),
                           ("as logged live", 'logged')]:
            calibration(df[col].values, df['red_won'].values, label)

        out = Path(tmp) / 'ablation.csv'
        df.to_csv(ENGINE / 'experiments' / 'ablation_results.csv', index=False)
        print(f"\nper-bout probabilities written to engine/experiments/ablation_results.csv")


if __name__ == '__main__':
    main()
