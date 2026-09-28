"""What is short notice actually worth, and is it worth anything on top of the market?

THE CONSTANT THIS EXISTS TO REPLACE. predict_card carries

    'short_notice': -0.04,       # ~4% penalty for short notice replacement

which nobody fitted. It is a plausible-sounding number someone typed, and the
project's whole position is that a number nobody measured must not move a
probability. Two outcomes are acceptable from this file: a weight with an
interval that excludes zero, or the deletion of the hook. Tuning it toward
something that feels right is not one of them.

TWO QUESTIONS, AND THE SECOND IS THE ONE THAT DECIDES IT.

  1. Do short-notice fighters under-perform the MODEL's probability?
     The model reads records. It cannot know a fighter took the fight nine
     days ago, so if short notice matters at all, this is where it shows.

  2. Do they still under-perform the BLENDED probability - the one the app
     actually prints, three-quarters of which is the closing line?
     By fight night the market has read the same news. A late replacement is
     the most visible thing that can happen to a fight and the price moves
     hard. If the market has already taken the penalty, taking it AGAIN in
     the adjustment layer double-counts it, and the app ends up more wrong
     about exactly the fights it thinks it understands best.

A weight that passes (1) and fails (2) must not ship while the blend is on.
That is the most likely result and it is a real answer, not a null one.

THE ESTIMATOR is the mean residual: actual minus predicted, over the fighters
who took a fight late. It is a difference of the same fighter's outcome from
the same fight's prediction, so everything the model already explains drops
out, which is why this needs hundreds of labels rather than thousands. The
interval comes from a bootstrap over EVENTS, not fights - two replacements on
one card are one piece of news, and treating them as independent would
narrow the interval on a correlation rather than on evidence.

WALK-FORWARD, because a weight fitted on all of history and reported on all
of history is a description, not a prediction. Each year is predicted from
earlier years only, and the weight a year would have shipped with is the one
estimated from the years before it.

    python engine/experiments/short_notice_weight.py

Needs engine/data/fight_changes.csv, which check_fight_changes decides
whether it is possible to build. Without it this prints what it needs and
stops, rather than inventing labels.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.calibrator_mismatch import fit_models, proba

CHANGES = ENGINE / "data" / "fight_changes.csv"
OUT = Path(__file__).with_suffix(".json")

FIRST_PREDICTED_YEAR = 2015
MIN_TRAIN = 500
DRAWS = 4000
# Under this many labelled fighters the interval is wider than the effect
# being looked for, and any number reported would be noise with a decimal
# point on it. check_fight_changes prints the same arithmetic.
MIN_LABELS = 200


def bootstrap_by_event(residuals, events, *, draws=DRAWS, seed=0):
    """95% interval for a mean residual, resampling EVENTS.

    Two late replacements on one card share a cause - a cancelled bout, a
    weight-cut failure, a visa problem - so they are one observation's worth
    of evidence and not two. Resampling fights instead of cards would report
    an interval that is too narrow by exactly the amount of that correlation.
    """
    residuals = np.asarray(residuals, dtype=float)
    events = np.asarray(events)
    unique = np.unique(events)
    if len(unique) < 2:
        return float("nan"), float("nan")
    by_event = {e: residuals[events == e] for e in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(unique), len(unique))
        means[i] = np.concatenate([by_event[unique[j]] for j in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(mean, low, high):
    if not np.isfinite(low):
        return "not enough events to say"
    if low > 0:
        return "REAL, and positive - they OVER-perform"
    if high < 0:
        return "REAL - a penalty this size is earned"
    return "inside the noise - no weight is justified"


def report(label, residuals, events):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < MIN_LABELS:
        print(f"  {label:<34}{len(residuals):>6} labels - too few to report")
        return None
    mean = float(residuals.mean())
    low, high = bootstrap_by_event(residuals, events)
    print(f"  {label:<34}{len(residuals):>6}{mean:>+10.4f}   "
          f"[{low:+.4f}, {high:+.4f}]   {verdict(mean, low, high)}")
    return {"label": label, "n": int(len(residuals)),
            "events": int(len(np.unique(events))), "mean": mean,
            "low": low, "high": high}


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/short_notice.json")
    if not CHANGES.exists():
        print(f"  {CHANGES} does not exist.")
        print("  It needs columns: event_date, stepped_in, replaced,")
        print("  days_notice. Run the check-fight-changes workflow mode first;")
        print("  it reports whether those labels can be gathered at all.")
        print("\n  Nothing was estimated. A weight invented without labels is")
        print("  the exact thing this file exists to replace.")
        return 1

    print("building features...")
    import predict_card as engine
    import market_blend as blend
    from name_resolution import norm_name

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    meta = engine.ufc_valid.reset_index(drop=True)
    dates = pd.to_datetime(meta["date"])
    years = dates.dt.year

    # --- the labels -------------------------------------------------------
    changes = pd.read_csv(CHANGES)
    changes["event_date"] = pd.to_datetime(changes["event_date"],
                                           errors="coerce")
    changes = changes.dropna(subset=["event_date", "stepped_in"])
    late = {(norm_name(n), d.date())
            for n, d in zip(changes.stepped_in, changes.event_date)}
    # THE QUANTIFIED SUBSET, KEPT APART. Wikipedia states a notice period on
    # very few replacements, so this will be small - but it is the only
    # population that is actually SHORT NOTICE rather than merely a
    # replacement, and pooling the two answers neither question. One of the
    # harvested sentences has a man replaced in early March for a fight on 8
    # May; folding his nine-week camp in with a nine-day one is how a null
    # result gets manufactured.
    quick = set()
    if "days_notice" in changes.columns:
        soon = changes[pd.to_numeric(changes.days_notice,
                                     errors="coerce").between(1, 21)]
        quick = {(norm_name(n), d.date())
                 for n, d in zip(soon.stepped_in, soon.event_date)}
    # A fighter whose OPPONENT was swapped but who had a full camp: the other
    # half of the story, and a different mechanism - not tired, just prepared
    # for the wrong person.
    switched = {(norm_name(n), d.date())
                for n, d in zip(changes.get("kept_fighter", []),
                                changes.get("event_date", []))} \
        if "kept_fighter" in changes.columns else set()
    print(f"  {len(late):,} replacements, of which {len(quick):,} state a "
          f"notice period inside three weeks")

    # --- walk-forward predictions ----------------------------------------
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
        index = np.flatnonzero(test)
        for position, row in enumerate(index):
            rows.append({"i": int(row), "p_model": float(p[position])})
        print(f"    {year}: {test.sum()} fights")

    frame = pd.DataFrame(rows).set_index("i")
    frame["date"] = dates.loc[frame.index].dt.date.values
    frame["event"] = meta.loc[frame.index, "event_name"].values
    frame["red"] = [norm_name(n) for n in meta.loc[frame.index, "r_name"]]
    frame["blue"] = [norm_name(n) for n in meta.loc[frame.index, "b_name"]]
    frame["won"] = y[frame.index]

    # --- the market's version of the same probability ---------------------
    odds = pd.read_csv(ENGINE / "data" / "odds.csv")
    odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    priced = {}
    for a, b, oa, ob, d in zip(odds.get("fighter_a", []), odds.get("fighter_b", []),
                               odds.get("odds_a", []), odds.get("odds_b", []),
                               odds.get("date", [])):
        if pd.isna(d) or pd.isna(oa) or pd.isna(ob):
            continue
        priced[(norm_name(a), norm_name(b), d.date())] = (oa, ob)

    def market_for(row):
        for key, flip in (((row.red, row.blue, row.date), False),
                          ((row.blue, row.red, row.date), True)):
            if key in priced:
                a, b = priced[key]
                p = blend.devig(b, a) if flip else blend.devig(a, b)
                return p
        return np.nan

    frame["p_market"] = [market_for(r) for r in frame.itertuples()]
    frame["p_blend"] = [blend.blend(m, k) if np.isfinite(k) else m
                        for m, k in zip(frame.p_model, frame.p_market)]
    print(f"  {int(np.isfinite(frame.p_market).sum()):,} of {len(frame):,} "
          f"fights carry a price")

    # --- residuals, per fighter ------------------------------------------
    # One row per FIGHTER, not per fight: the question is about the person who
    # took the fight late, and their opponent is a different observation with
    # the mirror-image prediction.
    def residuals_for(population, column):
        got, events = [], []
        for row in frame.itertuples():
            p = getattr(row, column)
            if not np.isfinite(p):
                continue
            if (row.red, row.date) in population:
                got.append(row.won - p)
                events.append(row.event)
            if (row.blue, row.date) in population:
                got.append((1.0 - row.won) - (1.0 - p))
                events.append(row.event)
        return np.asarray(got), np.asarray(events)

    print(f"\n  mean residual = what actually happened minus what was "
          f"predicted.\n  A negative number means they under-performed the "
          f"prediction by that much.")
    print(f"\n  {'population':<34}{'n':>6}{'mean':>10}   "
          f"{'95% interval':<22} verdict")
    print("  " + "-" * 96)

    results = []
    for name, population in (("stepped in as a replacement", late),
                             ("stepped in inside three weeks", quick),
                             ("opponent was swapped", switched)):
        if not population:
            continue
        for column, against in (("p_model", "vs the model"),
                                ("p_blend", "vs the blend the app prints")):
            got, events = residuals_for(population, column)
            row = report(f"{name}, {against}", got, events)
            if row:
                results.append(row)

    # --- what it means ----------------------------------------------------
    print()
    model_row = next((r for r in results if "vs the model" in r["label"]
                      and "replacement" in r["label"]), None)
    blend_row = next((r for r in results if "the app prints" in r["label"]
                      and "replacement" in r["label"]), None)
    if model_row and blend_row:
        if blend_row["high"] >= 0 >= blend_row["low"]:
            print("  The hook must not ship. Whatever short notice is worth")
            print("  against the raw model, the market has already taken it:")
            print("  against the probability the app actually prints, the")
            print("  effect is inside the noise. Applying a penalty on top")
            print("  would double-count the single most visible thing that")
            print("  can happen to a fight.")
        elif blend_row["high"] < 0:
            print(f"  A weight of {blend_row['mean']:+.3f} is earned on top of")
            print(f"  the blend, interval [{blend_row['low']:+.3f}, "
                  f"{blend_row['high']:+.3f}].")
            print(f"  The shipped constant is -0.040.")
        else:
            print("  The effect points the wrong way. Do not ship a penalty.")

    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "shipped_constant": -0.04,
        "results": results,
    }, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
