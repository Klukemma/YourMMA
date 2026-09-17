"""Match logged predictions against real results and report how the model did.

    python engine/grade.py            # report only, writes nothing
    python engine/grade.py --write    # also update statuses in the history file

Predictions are logged on every run of predict_card.py, so the same bout can
appear several times. Grading counts each bout once - its most recent
prediction - otherwise a card that happened to be re-run three times would
carry three times the weight in the accuracy figure.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_resolution import norm_name

ENGINE_DIR = Path(__file__).resolve().parent
DATA_DIR = ENGINE_DIR / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
HISTORY = Path(os.environ.get("PREDICTIONS_LOG", DATA_DIR / "prediction_history.json"))

# How far a logged event date may sit from the recorded one and still match.
# Cards get rescheduled; names plus a nearby date is a safer key than the date.
DATE_TOLERANCE = timedelta(days=10)


def categorise_method(method):
    """Dataset method text -> the three classes the model predicts."""
    if not isinstance(method, str):
        return None
    text = method.lower()
    if 'decision' in text:
        return 'Decision'
    if 'sub' in text:
        return 'Submission'
    if 'ko' in text or 'tko' in text:
        return 'KO/TKO'
    return None            # Overturned, Could Not Continue, DQ


def load_results(csv_path):
    """{(norm_a, norm_b): [(date, winner, method)]} for both corner orders."""
    df = pd.read_csv(csv_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        red, blue = row.r_name, row.b_name
        if not isinstance(red, str) or not isinstance(blue, str):
            continue
        entry = (row.date, row.winner, row.method)
        index[(norm_name(red), norm_name(blue))].append(entry)
        index[(norm_name(blue), norm_name(red))].append(entry)
    return index


def find_result(prediction, index):
    """The recorded outcome for a prediction, or None."""
    key = (norm_name(prediction['red_corner']), norm_name(prediction['blue_corner']))
    candidates = index.get(key)
    if not candidates:
        return None
    predicted_date = pd.to_datetime(prediction['event_date'], errors='coerce')
    if pd.isna(predicted_date):
        return candidates[0]
    within = [c for c in candidates if abs(c[0] - predicted_date) <= DATE_TOLERANCE]
    if not within:
        return None
    return min(within, key=lambda c: abs(c[0] - predicted_date))


def deduplicate(predictions):
    """One prediction per bout - the most recent."""
    latest = {}
    for pred in predictions:
        key = (norm_name(pred['red_corner']), norm_name(pred['blue_corner']),
               str(pred.get('event_date'))[:7])          # same bout, same month
        stamp = pred.get('timestamp') or ''
        if key not in latest or stamp > (latest[key].get('timestamp') or ''):
            latest[key] = pred
    return list(latest.values())


def grade(predictions, index):
    """Attach outcomes. Returns (graded, ungraded)."""
    graded, ungraded = [], []
    for pred in predictions:
        result = find_result(pred, index)
        if result is None:
            ungraded.append((pred, 'no matching bout in the dataset'))
            continue
        date, winner, method = result
        if not isinstance(winner, str) or not winner.strip():
            ungraded.append((pred, 'draw or no contest'))
            continue
        row = dict(pred)
        row['actual_winner'] = winner
        row['actual_method'] = categorise_method(method)
        row['actual_date'] = date
        row['correct'] = norm_name(winner) == norm_name(pred['predicted_winner'])
        graded.append(row)
    return graded, ungraded


def _brier(rows):
    return float(np.mean([(r['win_probability'] - (1.0 if r['correct'] else 0.0)) ** 2
                          for r in rows]))


def _log_loss(rows):
    eps = 1e-15
    return float(np.mean([
        -np.log(max(r['win_probability'] if r['correct'] else 1 - r['win_probability'], eps))
        for r in rows]))


def report(graded, ungraded, total_logged):
    n = len(graded)
    if n == 0:
        print("Nothing could be graded.")
        return
    correct = sum(r['correct'] for r in graded)
    acc = correct / n

    print("=" * 66)
    print("PREDICTION ACCURACY")
    print("=" * 66)
    print(f"  logged predictions : {total_logged}")
    print(f"  unique bouts       : {n + len(ungraded)}")
    print(f"  graded             : {n}")
    print(f"  correct            : {correct}")
    print(f"  ACCURACY           : {acc:.1%}")
    print(f"  Brier score        : {_brier(graded):.4f}")
    print(f"  Log loss           : {_log_loss(graded):.4f}")

    # A coin flip and always-take-the-favourite are the bars worth clearing.
    print(f"\n  vs coin flip       : {acc - 0.5:+.1%}")

    print("\n" + "-" * 66)
    print("CALIBRATION - does a stated 70% actually win 70% of the time?")
    print("-" * 66)
    print(f"  {'confidence band':<18}{'n':>5}{'predicted':>12}{'actual':>10}{'gap':>9}")
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bands:
        rows = [r for r in graded if lo <= r['win_probability'] < hi]
        if not rows:
            continue
        stated = float(np.mean([r['win_probability'] for r in rows]))
        actual = float(np.mean([r['correct'] for r in rows]))
        print(f"  {lo:.0%}-{hi if hi <= 1 else 1:.0%}{'':<11}{len(rows):>5}"
              f"{stated:>11.1%}{actual:>10.1%}{actual - stated:>+9.1%}")

    print("\n" + "-" * 66)
    print("METHOD")
    print("-" * 66)
    with_method = [r for r in graded if r['actual_method']]
    if with_method:
        hits = sum(r['method_predicted'] == r['actual_method'] for r in with_method)
        print(f"  method accuracy    : {hits}/{len(with_method)} = {hits/len(with_method):.1%}")
        for cls in ('KO/TKO', 'Submission', 'Decision'):
            rows = [r for r in with_method if r['method_predicted'] == cls]
            if rows:
                h = sum(r['actual_method'] == cls for r in rows)
                print(f"    predicted {cls:<12}{h:>4}/{len(rows):<4} = {h/len(rows):.1%}")

    print("\n" + "-" * 66)
    print("BY EVENT")
    print("-" * 66)
    by_event = defaultdict(list)
    for r in graded:
        by_event[r['event_name']].append(r)
    for event, rows in sorted(by_event.items(), key=lambda kv: -len(kv[1])):
        h = sum(r['correct'] for r in rows)
        print(f"  {h:>2}/{len(rows):<3} {h/len(rows):>6.0%}  {event}")

    if ungraded:
        print("\n" + "-" * 66)
        print(f"NOT GRADED ({len(ungraded)})")
        print("-" * 66)
        reasons = defaultdict(int)
        for _, reason in ungraded:
            reasons[reason] += 1
        for reason, count in reasons.items():
            print(f"  {count:>3}  {reason}")
        for pred, reason in ungraded[:10]:
            print(f"       {pred['event_date']}  {pred['red_corner']} vs {pred['blue_corner']}")


def write_back(graded, ungraded, path):
    """Record outcomes in the history file."""
    history = json.loads(Path(path).read_text())
    outcome = {}
    for r in graded:
        key = (norm_name(r['red_corner']), norm_name(r['blue_corner']))
        outcome[key] = (r['actual_winner'], r['correct'])
    for pred in history['predictions']:
        key = (norm_name(pred['red_corner']), norm_name(pred['blue_corner']))
        if key in outcome:
            winner, correct = outcome[key]
            pred['actual_winner'] = winner
            pred['correct'] = bool(correct)
            pred['status'] = 'correct' if correct else 'incorrect'
    graded_rows = [p for p in history['predictions'] if p.get('status') in ('correct', 'incorrect')]
    history['summary'] = {
        'total': len(history['predictions']),
        'graded': len(graded_rows),
        'correct': sum(p['correct'] for p in graded_rows),
        'pending': len(history['predictions']) - len(graded_rows),
        'unique_bouts_graded': len(graded),
        'accuracy': round(sum(r['correct'] for r in graded) / len(graded), 4) if graded else None,
    }
    Path(path).write_text(json.dumps(history, indent=2))
    print(f"\nWrote outcomes to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='update the history file')
    args = ap.parse_args()

    history = json.loads(HISTORY.read_text())
    predictions = history['predictions']
    unique = deduplicate(predictions)
    index = load_results(UFC_CSV)
    graded, ungraded = grade(unique, index)
    report(graded, ungraded, len(predictions))
    if args.write:
        write_back(graded, ungraded, HISTORY)


if __name__ == '__main__':
    main()
