"""Build the feature matrix once and cache it.

Importing predict_card.py re-runs the entire pipeline - feature engineering,
leakage audits, three model families, a twelve-year walk-forward retrain, the
production refit, the demo predictions and a card - which takes about ten
minutes. Every experiment that only wants X, y and dates was paying that, and
long runs kept dying to container restarts before producing anything.

This pays it once and writes a cache. Experiments then load in under a second.

    python engine/build_features.py          # build or rebuild the cache
    python engine/build_features.py --check  # report what is cached

The cache is keyed on the dataset's size and last date, so it rebuilds by
itself after a data sync rather than silently serving stale features.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

DATA_DIR = ENGINE / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))
CACHE = Path(os.environ.get("FEATURE_CACHE", DATA_DIR / "feature_cache.npz"))


def dataset_fingerprint(csv_path=None):
    """Rows plus last date - enough to notice a sync without hashing 7MB."""
    csv_path = Path(csv_path or UFC_CSV)
    df = pd.read_csv(csv_path, usecols=['date'], low_memory=False)
    dates = pd.to_datetime(df['date'], errors='coerce')
    raw = f"{len(df)}|{dates.max()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build(cache_path=None):
    """Import the pipeline once and save X, y and dates."""
    cache_path = Path(cache_path or CACHE)
    os.environ.setdefault('PREDICTIONS_LOG', '/tmp/feature_cache_scratch.json')
    print("importing predict_card (this is the slow part, once)...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    dates = pd.to_datetime(engine.ufc_valid['date'].reset_index(drop=True))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        values=X.to_numpy(dtype=float),
        columns=np.array(list(X.columns), dtype=object),
        y=y,
        dates=dates.values.astype('datetime64[ns]'),
        fingerprint=np.array([dataset_fingerprint()], dtype=object),
    )
    print(f"cached {X.shape[0]:,} fights x {X.shape[1]} features -> {cache_path}")
    return X, y, dates


def load(cache_path=None, allow_stale=False):
    """Load cached features, rebuilding if the dataset has changed.

    Returns (X, y, dates).
    """
    cache_path = Path(cache_path or CACHE)
    if not cache_path.exists():
        print(f"no cache at {cache_path}; building it")
        return build(cache_path)

    blob = np.load(cache_path, allow_pickle=True)
    cached = str(blob['fingerprint'][0])
    current = dataset_fingerprint()
    if cached != current and not allow_stale:
        print(f"cache is stale (dataset changed: {cached} -> {current}); rebuilding")
        return build(cache_path)

    X = pd.DataFrame(blob['values'], columns=list(blob['columns']))
    y = blob['y']
    dates = pd.Series(pd.to_datetime(blob['dates']))
    return X, y, dates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='report the cache, build nothing')
    args = ap.parse_args()

    if args.check:
        if not CACHE.exists():
            print(f"no cache at {CACHE}")
            return
        blob = np.load(CACHE, allow_pickle=True)
        cached, current = str(blob['fingerprint'][0]), dataset_fingerprint()
        dates = pd.to_datetime(blob['dates'])
        print(f"cache      : {CACHE}")
        print(f"shape      : {blob['values'].shape[0]:,} x {blob['values'].shape[1]}")
        print(f"date range : {dates.min().date()} -> {dates.max().date()}")
        print(f"fingerprint: {cached}  ({'current' if cached == current else 'STALE'})")
        return

    build()


if __name__ == '__main__':
    main()
