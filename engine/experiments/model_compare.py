"""One walk-forward harness for every "does change X beat the model?" idea.

Each earlier experiment rebuilt its own walk-forward loop, imported
predict_card (ten minutes, trains everything, writes app data) and measured
a slightly different thing: historical_backtest.py and residual_harness.py
use calibrator_mismatch.fit_models, which is NOT the production recipe (no
recency weights, no early stopping, no Platt), while the app's picks come
from the production models. So an idea could win against one harness and
lose against the thing that actually ships. This module is the shared
yardstick, and later ideas (symmetry, performance ratings, opponent-adjusted
stats, shrinkage, ...) plug into it by supplying a feature function.

WHAT IT MIRRORS. The production winner model, predict_card.py SECTION 15,
read line by line:

    split          first 90% of valid rows (date order) train, last 10%
                   calibrate                              predict_card.py:1731-1733
    recency        w = 1 + 2 * ((d - d_min) / range) ** 1.5, mean-normalised,
                   on LR, RF and XGB                       predict_card.py:1766-1772
    LR             C=0.1, max_iter=1000, seed 42           predict_card.py:1763,1774
    RF             200 trees, depth 12, min_samples_leaf 10,
                   min_samples_split 10, seed 42           predict_card.py:1778-1782
    XGB            500 trees, depth 5, lr 0.05, subsample 0.8,
                   colsample 0.8, alpha 0.1, lambda 1.0, early stopping
                   50 rounds on the calibration block      predict_card.py:1785-1792
    average        (LR + RF + XGB) / 3                     predict_card.py:1828, 2393-2395
    Platt          LogisticRegression(C=1e10) on that average over the
                   calibration block, applied to the same
                   3-model average                         predict_card.py:1830-1831, 2396
    features       feature_cols_winner only                predict_card.py:408, 1749

The MLP is trained in production but is NOT in the winner average
(predict_card.py:2393-2396), so it is not fitted here.

WALK-FORWARD. Year Y is predicted by models fitted on valid fights of years
< Y only, the 90/10 split and the Platt calibrator inside that block, so no
FITTED parameter used for year Y has seen year Y's outcomes. First predicted
year and minimum training size follow historical_backtest.py (2011, 500).
Every row carries the fight's date so a test can prove the ordering.

Two caveats on that claim, both stated so nobody reads more into it:

  * The models are up to a year staler than production. A December fight
    of year Y is predicted by models trained through 31 December of Y-1,
    whereas predict_card retrains on the whole archive before every card.
    That under-states what a fresh model earns (so a blend fitted here
    leans a little more on the market than a fresh one would), and it is
    the same for every arm, so paired comparisons are fair. Anything that
    takes coefficients from this harness INTO production must either refit
    them on a finer walk-forward (per event or per quarter) or measure the
    fresh-versus-stale shift first - predict_card.py:1820-1828 documents
    fixing exactly that kind of mismatch once already.
  * A few label-free feature constants are computed on the whole archive by
    feature_frame.build, as production does: the mx_* interaction centres
    (feature_spec.interaction, means over all valid rows) and the
    data_sparsity min-max scaling. No outcome enters them, so this is not a
    leak of results, but a 2012 feature does carry a constant that depends
    on 2026 rows. An idea that touches those features should compute its
    centres inside the per-year fit or accept and document the constant.

PRIOR BOUTS. red_prior_bouts / blue_prior_bouts count each corner's UFC
bouts strictly before the fight, in either corner, over the FULL post-2001
archive - draws and no-contests included - so a fighter whose only earlier
bout was a draw is not a debutant. They are computed before the valid-row
filter and carried on the frame.

MARKET. The closing line is matched exactly the way residual_harness.py
does: (normalised red, normalised blue, date) against odds.csv in either
corner order, de-vigged by market_blend.devig. No date tolerance, no
fallback. A fight odds.csv does not price has p_market = NaN. The market is
attached on EVERY call, after the cache lookup, so an odds.csv backfill is
seen without a refit.

COMPARISON. compare() takes two prediction arms on the SAME fights and
reports log loss, Brier and accuracy for each, the paired difference, and a
bootstrap interval for the difference that resamples whole EVENTS (fights on
one card share a referee, a venue, a weigh-in) and, separately, FIGHTERS
(a fighter in the sample twenty times is not twenty observations); the wider
interval is the one reported. `alpha` is the family-wise level: an idea with
k arms passes 0.05 / k, not 0.05.

CACHE. The baseline predictions (no feature function) are the expensive part
and every idea needs them, so they are cached beside this file as
model_compare_baseline.csv, keyed by a hash of the feature matrix, labels,
dates, fight ids, feature names, the protocol string AND the source of the
recipe functions (fit_production, predict_production, recency_weights,
walk_forward, prior_bouts, build_frame); a changed archive or a changed
recipe misses the cache and refits, with or without a PROTOCOL bump. The
cache holds the model columns only - never p_market - so odds.csv is not
part of the key and does not need to be: the market is re-matched on every
call. MODEL_COMPARE_CACHE overrides the directory.

    python engine/experiments/model_compare.py        # baseline + sanity numbers
"""

import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import market_blend  # noqa: E402
from name_resolution import norm_name  # noqa: E402

UFC_CSV = Path(os.environ.get("UFC_CSV", ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"))
ODDS_CSV = Path(os.environ.get("ODDS_CSV", ENGINE / "data" / "odds.csv"))
CACHE_DIR = Path(os.environ.get("MODEL_COMPARE_CACHE", ENGINE / "experiments"))

# historical_backtest.py:38-45 - 2010 is the first priced year, so it trains.
FIRST_PREDICTED_YEAR = 2011
MIN_TRAIN = 500
MIN_TEST = 20
CAL_FRACTION = 0.10          # predict_card.py:1733 - "90% train, 10% calibration"
SEED = 42
DRAWS = 4000
EPS = 1e-3                   # probabilities are clipped before logs are taken

# Bumped whenever what a row MEANS changes (v2: prior bouts counted on the
# full archive, market no longer cached). Changes to the recipe code itself
# are caught by code_fingerprint(), with or without a bump.
PROTOCOL = "production-3model-platt-v2"
MODEL_COLUMNS = ["fight_id", "date", "year", "event", "red", "blue", "red_norm", "blue_norm",
                 "red_prior_bouts", "blue_prior_bouts", "y", "p_ensemble", "p_model",
                 "train_rows"]


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------

def load_archive(path=UFC_CSV):
    return pd.read_csv(path, low_memory=False)


def build_frame(raw=None, features=None, verbose=False):
    """ufc (valid rows), X (winner features), y, as the production model sees
    them - without importing predict_card.

    `features(built) -> DataFrame` lets an idea add or replace columns: it
    receives feature_frame.build's whole result and returns the matrix to
    use (index aligned with built["ufc"]). Default: feature_cols_winner,
    predict_card.py:408.
    """
    import feature_frame
    if raw is None:
        raw = load_archive()
    built = feature_frame.build(raw, verbose=verbose)
    ufc = built["ufc"]
    X = (features(built) if features is not None
         else built["X"][built["feature_cols_winner"]])
    X = pd.DataFrame(X)
    if len(X) != len(ufc):
        raise ValueError(f"feature function returned {len(X)} rows for {len(ufc)} fights")
    # Prior bouts over EVERY post-2001 row, draws and no-contests included,
    # before the valid filter removes them.
    ufc = ufc.copy()
    ufc["red_prior_bouts"], ufc["blue_prior_bouts"] = prior_bouts(ufc)
    valid = ufc["target_win"].notna().to_numpy()        # predict_card.py:406
    ufc = ufc.loc[valid].reset_index(drop=True).copy()
    ufc["date"] = pd.to_datetime(ufc["date"], errors="coerce")
    X = X.loc[valid].reset_index(drop=True).astype(float)
    y = ufc["target_win"].to_numpy(dtype=float)
    return {"ufc": ufc, "X": X, "y": y, "feature_cols": list(X.columns)}


def prior_bouts(ufc):
    """Each corner's UFC bouts STRICTLY before this fight, either corner.

    Point in time by construction: the frame is date-ordered and a fighter
    appears once per date, so a cumulative count over earlier rows is a
    count of earlier fights. Counted over the rows it is given: build_frame
    calls it on the full post-2001 archive (draws and no-contests included)
    before the valid filter, and walk_forward uses those columns when the
    frame carries them, falling back to counting the frame it is given.
    r_career_fights in the feature frame counts red-corner appearances
    only, which is not the same thing.
    """
    dates = pd.to_datetime(ufc["date"])
    long = pd.concat([
        pd.DataFrame({"i": ufc.index, "who": ufc["r_name"].map(norm_name), "date": dates}),
        pd.DataFrame({"i": ufc.index, "who": ufc["b_name"].map(norm_name), "date": dates}),
    ])
    long = long.sort_values(["who", "date", "i"], kind="stable")
    long["n"] = long.groupby("who").cumcount()
    counts = {(i, who): n for i, who, n in zip(long["i"], long["who"], long["n"])}
    red = np.array([counts[(i, w)] for i, w in zip(ufc.index, ufc["r_name"].map(norm_name))])
    blue = np.array([counts[(i, w)] for i, w in zip(ufc.index, ufc["b_name"].map(norm_name))])
    return red, blue


# --------------------------------------------------------------------------
# the production recipe
# --------------------------------------------------------------------------

def recency_weights(dates):
    """predict_card.py:1766-1772, verbatim arithmetic."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    lo, hi = dates.min(), dates.max()
    span = max((hi - lo).days, 1)
    w = np.array([1.0 + 2.0 * ((d - lo).days / span) ** 1.5 for d in dates])
    return w / w.mean()


def fit_production(X, y, dates, seed=SEED):
    """The production winner model fitted on (X, y) exactly as SECTION 15
    does on the whole archive: the LAST CAL_FRACTION of rows, in date order,
    calibrate. Rows must arrive date-sorted (feature_frame.build sorts them;
    predict_card.py:70 and :1733 rely on the same order); this refuses
    anything else rather than calibrating on a random block."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError("fit_production needs date-sorted rows: the calibration "
                         "block is the most recent 10%, not a random 10%")
    n = len(X)
    train_end = int(n * (1.0 - CAL_FRACTION))         # predict_card.py:1733
    X_tr, X_cal = X[:train_end], X[train_end:]
    y_tr, y_cal = y[:train_end], y[train_end:]
    w = recency_weights(dates.iloc[:train_end])

    scaler = StandardScaler()                           # predict_card.py:1752-1754
    X_tr_s = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)

    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
    lr.fit(X_tr_s, y_tr, sample_weight=w)               # predict_card.py:1763,1774
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                min_samples_split=10, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr, sample_weight=w)               # predict_card.py:1778-1782
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        random_state=seed, eval_metric="logloss", verbosity=0,
                        early_stopping_rounds=50)
    xgb.fit(X_tr_s, y_tr, sample_weight=w,
            eval_set=[(X_cal_s, y_cal)], verbose=False)  # predict_card.py:1785-1792

    p_cal = _average(scaler, lr, rf, xgb, X_cal)        # predict_card.py:1816-1828
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(p_cal.reshape(-1, 1), y_cal)              # predict_card.py:1830-1831
    return {"scaler": scaler, "lr": lr, "rf": rf, "xgb": xgb, "platt": platt,
            "train_rows": int(train_end), "cal_rows": int(n - train_end),
            "train_min_date": dates.iloc[0], "train_max_date": dates.iloc[train_end - 1],
            "cal_min_date": dates.iloc[train_end], "cal_max_date": dates.iloc[n - 1],
            "xgb_best_iteration": int(xgb.best_iteration)}


def _average(scaler, lr, rf, xgb, X):
    Xs = scaler.transform(np.asarray(X, float))
    return (lr.predict_proba(Xs)[:, 1] + rf.predict_proba(Xs)[:, 1]
            + xgb.predict_proba(Xs)[:, 1]) / 3           # predict_card.py:2393-2395


def predict_production(models, X):
    """(p_ensemble before Platt, p after Platt) - predict_card.py:2393-2396."""
    p_ens = _average(models["scaler"], models["lr"], models["rf"], models["xgb"], X)
    p = models["platt"].predict_proba(p_ens.reshape(-1, 1))[:, 1]
    return p_ens, p


# --------------------------------------------------------------------------
# the walk-forward
# --------------------------------------------------------------------------

def walk_forward(X, y, ufc, *, fitter=fit_production, predictor=predict_production,
                 first_year=FIRST_PREDICTED_YEAR, min_train=MIN_TRAIN,
                 min_test=MIN_TEST, verbose=True):
    """One row per valid fight of a predicted year, from models fitted on
    years strictly before it. `fitter(X, y, dates)` and
    `predictor(models, X) -> (p_ensemble, p)` are injectable so a test can
    prove what each year's models were shown."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y, float)
    ufc = ufc.reset_index(drop=True)
    dates = pd.to_datetime(ufc["date"])
    years = dates.dt.year.to_numpy()
    if "red_prior_bouts" in ufc and "blue_prior_bouts" in ufc:
        red_prior = ufc["red_prior_bouts"].to_numpy(int)      # from build_frame: full archive
        blue_prior = ufc["blue_prior_bouts"].to_numpy(int)
    else:
        red_prior, blue_prior = prior_bouts(ufc)
    out = []
    for year in sorted(np.unique(years)):
        if year < first_year:
            continue
        train = years < year
        test = years == year
        if train.sum() < min_train or test.sum() < min_test:
            continue
        t0 = time.time()
        models = fitter(X.to_numpy(float)[train], y[train], dates[train])
        p_ens, p = predictor(models, X.to_numpy(float)[test])
        idx = np.flatnonzero(test)
        block = ufc.iloc[idx]
        out.append(pd.DataFrame({
            "fight_id": block["fight_id"].to_numpy() if "fight_id" in block else idx,
            "date": dates.iloc[idx].to_numpy(),
            "year": int(year),
            "event": block.get("event_name", pd.Series("", index=block.index)).to_numpy(),
            "red": block["r_name"].to_numpy(),
            "blue": block["b_name"].to_numpy(),
            "red_norm": block["r_name"].map(norm_name).to_numpy(),
            "blue_norm": block["b_name"].map(norm_name).to_numpy(),
            "red_prior_bouts": red_prior[idx],
            "blue_prior_bouts": blue_prior[idx],
            "y": y[idx],
            "p_ensemble": np.asarray(p_ens, float),
            "p_model": np.asarray(p, float),
            "train_rows": int(train.sum()),
        }))
        if verbose:
            extra = (f", xgb stopped at {models['xgb_best_iteration']}"
                     if isinstance(models, dict) and "xgb_best_iteration" in models else "")
            print(f"  {year}: trained on {train.sum():,}, predicted {test.sum():,} "
                  f"({time.time() - t0:.0f}s{extra})", flush=True)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# the market
# --------------------------------------------------------------------------

def load_prices(path=ODDS_CSV):
    """{(norm a, norm b, date): (odds_a, odds_b)} - residual_harness.py:88-96."""
    odds = pd.read_csv(path)
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    priced = {}
    for a, b, oa, ob, d in zip(odds["fighter_a"], odds["fighter_b"],
                               odds["odds_a"], odds["odds_b"], odds["date"]):
        if pd.isna(d) or pd.isna(oa) or pd.isna(ob):
            continue
        priced[(norm_name(a), norm_name(b), d.date())] = (oa, ob)
    return priced


def market_probabilities(frame, prices):
    """De-vigged probability of the RED corner, NaN where unpriced -
    residual_harness.py:98-106, same keys, same order of lookup."""
    out = np.full(len(frame), np.nan)
    for i, (red, blue, date) in enumerate(zip(frame["red_norm"], frame["blue_norm"],
                                              pd.to_datetime(frame["date"]))):
        day = date.date()
        if (red, blue, day) in prices:
            a, b = prices[(red, blue, day)]
            p = market_blend.devig(a, b)
        elif (blue, red, day) in prices:
            a, b = prices[(blue, red, day)]
            p = market_blend.devig(b, a)
        else:
            p = None
        out[i] = np.nan if p is None else p
    return out


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def code_fingerprint():
    """sha256 of the source of every function that decides what a cached row
    is, so editing the recipe misses the cache even without a PROTOCOL bump."""
    h = hashlib.sha256()
    for fn in (build_frame, prior_bouts, recency_weights, fit_production, _average,
               predict_production, walk_forward):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def inputs_hash(X, y, ufc, feature_cols, protocol=PROTOCOL, code=None):
    """Key for the cached walk-forward: inputs + protocol + recipe source.
    odds.csv is deliberately NOT in it - the market is not cached."""
    h = hashlib.sha256()
    h.update(protocol.encode())
    h.update((code_fingerprint() if code is None else code).encode())
    h.update(json.dumps([str(c) for c in feature_cols]).encode())
    h.update(np.ascontiguousarray(np.asarray(X, float)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, float)).tobytes())
    h.update(pd.to_datetime(ufc["date"]).astype("int64").to_numpy().tobytes())
    if "fight_id" in ufc:
        h.update(json.dumps(list(map(str, ufc["fight_id"]))).encode())
    if "red_prior_bouts" in ufc and "blue_prior_bouts" in ufc:
        h.update(np.asarray(ufc["red_prior_bouts"], int).tobytes())
        h.update(np.asarray(ufc["blue_prior_bouts"], int).tobytes())
    h.update(f"{FIRST_PREDICTED_YEAR}|{MIN_TRAIN}|{MIN_TEST}|{CAL_FRACTION}|{SEED}".encode())
    return h.hexdigest()


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def cache_paths(label, cache_dir=None):
    cache_dir = Path(cache_dir or CACHE_DIR)
    stem = cache_dir / f"model_compare_{label}"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def predictions(raw=None, features=None, *, label="baseline", cache_dir=None,
                odds_path=ODDS_CSV, verbose=True, use_cache=True):
    """Walk-forward predictions with the market attached.

    The walk-forward (model columns only) is cached by input hash; the
    market column is matched against `odds_path` on every call, hit or
    miss, so a changed odds file is always reflected and never refits.
    `features` is passed to build_frame. Anything but the baseline should
    give its own `label`, or it would overwrite the baseline's file (the
    hash would still refuse to serve it as the baseline).
    """
    t0 = time.time()
    frame = build_frame(raw, features, verbose=False)
    X, y, ufc = frame["X"], frame["y"], frame["ufc"]
    key = inputs_hash(X, y, ufc, frame["feature_cols"])
    csv_path, meta_path = cache_paths(label, cache_dir)
    out = None
    if use_cache and csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("hash") == key:
            if verbose:
                print(f"  cache hit {csv_path.name} ({meta.get('fitted', '?')}); "
                      f"matching market from {Path(odds_path).name}")
            out = pd.read_csv(csv_path, parse_dates=["date"])
            out = out.drop(columns=[c for c in out.columns if c not in MODEL_COLUMNS])
        elif verbose:
            print(f"  cache miss {csv_path.name}: inputs or recipe changed, refitting")
    if out is None:
        if verbose:
            print(f"  frame: {len(ufc):,} valid fights x {X.shape[1]} features "
                  f"({time.time() - t0:.0f}s)")
        out = walk_forward(X, y, ufc, verbose=verbose)
        if use_cache:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            out[[c for c in MODEL_COLUMNS if c in out]].to_csv(
                csv_path, index=False, float_format="%.6f")
            meta_path.write_text(json.dumps({
                "hash": key, "protocol": PROTOCOL, "code": code_fingerprint(),
                "label": label, "columns": [c for c in MODEL_COLUMNS if c in out],
                "market_cached": False,
                "fitted": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rows": int(len(out)), "features": int(X.shape[1]),
                "first_year": FIRST_PREDICTED_YEAR, "min_train": MIN_TRAIN,
                "years": sorted(int(v) for v in out["year"].unique()),
            }, indent=1) + "\n")
    out["p_market"] = market_probabilities(out, load_prices(odds_path))
    out.attrs["odds_sha256"] = file_sha256(odds_path)
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1.0 - EPS)


def log_loss_rows(y, p):
    y = np.asarray(y, float)
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_rows(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def correct_rows(y, p):
    return ((np.asarray(p, float) >= 0.5) == (np.asarray(y, float) == 1.0)).astype(float)


def metrics(y, p):
    return {"n": int(len(y)),
            "log_loss": float(log_loss_rows(y, p).mean()),
            "brier": float(brier_rows(y, p).mean()),
            "accuracy": float(correct_rows(y, p).mean())}


def calibration_by_decile(y, p, bins=10):
    """Mean predicted vs observed red-win rate by predicted-probability decile."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    band = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = band == b
        if m.sum() == 0:
            continue
        rows.append({"decile": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int(m.sum()), "predicted": float(p[m].mean()),
                     "observed": float(y[m].mean())})
    return rows


def bootstrap_interval(values, groups, alpha, *, draws=DRAWS, seed=0):
    """Interval for the mean of `values` resampling whole GROUPS with
    replacement (events: a card is one draw)."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        return float("nan"), float("nan")
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    sizes = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        means[i] = (counts * sums).sum() / (counts * sizes).sum()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def bootstrap_interval_fighters(values, red, blue, alpha, *, draws=DRAWS, seed=0):
    """Interval resampling FIGHTERS: each draw picks fighters with
    replacement and a fight is weighted by how many times its two fighters
    were drawn, so a fighter who appears often is one source of noise."""
    values = np.asarray(values, float)
    names, inverse = np.unique(np.concatenate([np.asarray(red), np.asarray(blue)]),
                               return_inverse=True)
    if len(names) < 2:
        return float("nan"), float("nan")
    n = len(values)
    red_i, blue_i = inverse[:n], inverse[n:]
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        counts = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = counts[red_i] + counts[blue_i]
        total = w.sum()
        means[i] = (w * values).sum() / total if total else np.nan
    means = means[np.isfinite(means)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def compare(y, p_base, p_cand, events, *, red=None, blue=None, alpha=0.05,
            draws=DRAWS, seed=0, label_base="baseline", label_cand="candidate"):
    """Paired comparison of two arms on the same fights.

    Returns each arm's metrics, the candidate-minus-baseline difference on
    each, and for each difference the event-resampled interval, the
    fighter-resampled interval (when red/blue are given) and the wider of
    the two at the family-wise `alpha`. Negative log loss / Brier
    differences and positive accuracy differences favour the candidate.
    """
    y = np.asarray(y, float)
    p_base, p_cand = np.asarray(p_base, float), np.asarray(p_cand, float)
    if not (len(y) == len(p_base) == len(p_cand) == len(events)):
        raise ValueError("arms must be scored on the same fights")
    out = {"n": int(len(y)), "alpha": float(alpha),
           "events": int(len(np.unique(np.asarray(events)))),
           label_base: metrics(y, p_base), label_cand: metrics(y, p_cand), "delta": {}}
    for name, rows in (("log_loss", log_loss_rows), ("brier", brier_rows),
                       ("accuracy", correct_rows)):
        d = rows(y, p_cand) - rows(y, p_base)
        by_event = bootstrap_interval(d, events, alpha, draws=draws, seed=seed)
        entry = {"mean": float(d.mean()), "event_low": by_event[0], "event_high": by_event[1]}
        if red is not None and blue is not None:
            by_fighter = bootstrap_interval_fighters(d, red, blue, alpha, draws=draws, seed=seed)
            entry["fighter_low"], entry["fighter_high"] = by_fighter
            entry["low"] = float(np.nanmin([by_event[0], by_fighter[0]]))
            entry["high"] = float(np.nanmax([by_event[1], by_fighter[1]]))
        else:
            entry["low"], entry["high"] = by_event
        out["delta"][name] = entry
    return out


def format_compare(result, label_base="baseline", label_cand="candidate"):
    b, c, d = result[label_base], result[label_cand], result["delta"]
    lines = [f"  {'':<22}{'log loss':>10}{'brier':>9}{'accuracy':>10}   n={result['n']:,}, "
             f"events={result['events']:,}, alpha={result['alpha']:.4f}",
             f"  {label_base:<22}{b['log_loss']:>10.4f}{b['brier']:>9.4f}{b['accuracy']:>10.1%}",
             f"  {label_cand:<22}{c['log_loss']:>10.4f}{c['brier']:>9.4f}{c['accuracy']:>10.1%}"]
    for name in ("log_loss", "brier", "accuracy"):
        e = d[name]
        lines.append(f"  {name + ' delta':<22}{e['mean']:>+10.4f}  "
                     f"[{e['low']:+.4f}, {e['high']:+.4f}]  "
                     f"event [{e['event_low']:+.4f}, {e['event_high']:+.4f}]"
                     + (f"  fighter [{e['fighter_low']:+.4f}, {e['fighter_high']:+.4f}]"
                        if "fighter_low" in e else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sanity
# --------------------------------------------------------------------------

def sanity(frame, since=2020):
    """The project's quoted figures (market_blend.py docstring): on priced
    walk-forward fights from 2020 the model picks 62.0%, the market
    favourite 67.5%, the 0.75 blend 68.2%, over 2,339 fights."""
    priced = frame[np.isfinite(frame["p_market"]) & (frame["year"] >= since)]
    y = priced["y"].to_numpy()
    p_model = priced["p_model"].to_numpy()
    p_market = priced["p_market"].to_numpy()
    p_blend = np.array([market_blend.blend(m, k) for m, k in zip(p_model, p_market)])
    return {"since": since, "priced": int(len(priced)),
            "all_predicted": int((frame["year"] >= since).sum()),
            "model": metrics(y, p_model), "market": metrics(y, p_market),
            "blend_0.75": metrics(y, p_blend),
            "ensemble_uncalibrated": metrics(y, priced["p_ensemble"].to_numpy())}


def main():
    print("walk-forward baseline (production recipe, feature_cols_winner)...")
    frame = predictions()
    priced = np.isfinite(frame["p_market"])
    print(f"\n{len(frame):,} predictions {frame['year'].min()}-{frame['year'].max()}, "
          f"{int(priced.sum()):,} priced")
    print("  by year: " + ", ".join(
        f"{int(y)}:{int(n)}/{int(m)}" for y, n, m in
        frame.groupby("year").agg(n=("p_market", lambda s: np.isfinite(s).sum()),
                                  m=("y", "size")).itertuples()))
    s = sanity(frame)
    print(f"\nsanity, priced fights {s['since']}+ ({s['priced']:,} of "
          f"{s['all_predicted']:,} predicted):")
    print(f"  {'arm':<24}{'accuracy':>10}{'log loss':>10}{'brier':>9}")
    for arm in ("model", "market", "blend_0.75", "ensemble_uncalibrated"):
        m = s[arm]
        print(f"  {arm:<24}{m['accuracy']:>10.1%}{m['log_loss']:>10.4f}{m['brier']:>9.4f}")
    print("  quoted (market_blend.py): model 62.0%, market 67.5%, blend 68.2% on 2,339")
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps({"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               "rows": int(len(frame)), "priced": int(priced.sum()),
                               "sanity": s}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
