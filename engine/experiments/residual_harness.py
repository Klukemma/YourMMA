"""The shared test for "does X move results beyond the model and the market?"

Every soft factor this project gathers - late replacements, injuries, camp
power - is tested the same way, so the answers are comparable and no factor
gets a friendlier test than another:

  - walk-forward model probabilities: each year from a model fitted on
    earlier years only (from FIRST_PREDICTED_YEAR);
  - the blended probability the app prints, three-quarters closing line;
  - the mean residual (actual minus predicted) over the fighters a factor
    marks, with a 95% interval from resampling EVENTS, not fights.

A factor ships only if its interval against the BLEND excludes zero.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.calibrator_mismatch import fit_models, proba
from experiments.short_notice_weight import (FIRST_PREDICTED_YEAR, MIN_TRAIN,
                                             bootstrap_by_event, verdict)

MIN_LABELS = 150


def predictions():
    """(frame, meta): one row per archive fight predicted out of sample,
    with p_model, p_market, p_blend, red, blue, date, event, won."""
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/residual_harness.json")
    import predict_card as engine
    import market_blend as blend
    from name_resolution import norm_name

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    meta = engine.ufc_valid.reset_index(drop=True)
    dates = pd.to_datetime(meta["date"])
    years = dates.dt.year

    rows = []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).to_numpy()
        test = (years == year).to_numpy()
        if train.sum() < MIN_TRAIN or test.sum() < 50:
            continue
        models = fit_models(X[train], y[train])
        p = proba(models, X[test], 3)
        for position, row in enumerate(np.flatnonzero(test)):
            rows.append({"i": int(row), "p_model": float(p[position])})
        print(f"    {year}: {test.sum()} fights", flush=True)

    frame = pd.DataFrame(rows).set_index("i")
    frame["date"] = dates.loc[frame.index].values
    frame["event"] = meta.loc[frame.index, "event_name"].values
    frame["red_raw"] = meta.loc[frame.index, "r_name"].values
    frame["blue_raw"] = meta.loc[frame.index, "b_name"].values
    frame["red"] = [norm_name(n) for n in frame["red_raw"]]
    frame["blue"] = [norm_name(n) for n in frame["blue_raw"]]
    frame["won"] = y[frame.index]

    odds = pd.read_csv(ENGINE / "data" / "odds.csv")
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    priced = {}
    for a, b, oa, ob, d in zip(odds.get("fighter_a", []),
                               odds.get("fighter_b", []),
                               odds.get("odds_a", []), odds.get("odds_b", []),
                               odds.get("date", [])):
        if pd.isna(d) or pd.isna(oa) or pd.isna(ob):
            continue
        priced[(norm_name(a), norm_name(b), d.date())] = (oa, ob)

    def market_for(row):
        day = pd.Timestamp(row.date).date()
        for key, flip in (((row.red, row.blue, day), False),
                          ((row.blue, row.red, day), True)):
            if key in priced:
                a, b = priced[key]
                return blend.devig(b, a) if flip else blend.devig(a, b)
        return np.nan

    frame["p_market"] = [market_for(r) for r in frame.itertuples()]
    frame["p_blend"] = [blend.blend(m, k) if np.isfinite(k) else m
                        for m, k in zip(frame.p_model, frame.p_market)]
    return frame


def residuals(frame, marked, column):
    """Residuals for the marked side of each fight.

    `marked(row)` returns "red", "blue", "both" or None.
    """
    got, events = [], []
    for row in frame.itertuples():
        p = getattr(row, column)
        if not np.isfinite(p):
            continue
        side = marked(row)
        if side in ("red", "both"):
            got.append(row.won - p)
            events.append(row.event)
        if side in ("blue", "both"):
            got.append((1.0 - row.won) - (1.0 - p))
            events.append(row.event)
    return np.asarray(got), np.asarray(events)


def report(label, got, events):
    got = np.asarray(got, dtype=float)
    if len(got) < MIN_LABELS:
        print(f"  {label:<52}{len(got):>6} labels - too few to report")
        return {"label": label, "n": int(len(got)), "too_few": True}
    mean = float(got.mean())
    low, high = bootstrap_by_event(got, events)
    print(f"  {label:<52}{len(got):>6}{mean:>+10.4f}   "
          f"[{low:+.4f}, {high:+.4f}]   {verdict(mean, low, high)}")
    return {"label": label, "n": int(len(got)),
            "events": int(len(np.unique(events))), "mean": mean,
            "low": low, "high": high}
