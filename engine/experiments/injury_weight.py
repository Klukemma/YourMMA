"""Does a KNOWN injury history move results, beyond the model and the market?

The question the injury history was gathered to answer, asked the only way
that can be trusted: for each fight, look only at what was public before it
(injury_history.known_injuries), and ask whether fighters carrying that
history under-perform

  1. the MODEL's probability - which reads records and cannot see injuries;
  2. the BLENDED probability the app prints, three-quarters closing line -
     which has read the news. An effect the market already prices must not
     be added again.

Nothing ships from this file unless the second interval excludes zero.

POPULATIONS, each a different mechanism, none pooled with another:
  withdrew_since_last_fight    pulled out of a booking with an injury or
                               illness since their previous bout: the plain
                               "coming back from an injury" case
  hurt_in_last_fight           injured in their previous bout (doctor
                               stoppage, injury TKO, broken limb)
  surgery_or_long_layoff_2y    surgery, or 6+ months out, public in the last
                               two years
  any_known_injury_1y          anything public in the last twelve months
  known_hurt_going_in          reported injured BEFORE this very fight -
                               the rare direct case, and the one a model
                               could least see

The estimator, the walk-forward years and the event-cluster bootstrap are
short_notice_weight.py's, for the same reasons given there. Vague records
(timing not stated, "later admitted") are excluded, as are records whose
fighter could not be matched to the archive.

    python engine/experiments/injury_weight.py
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
from experiments.short_notice_weight import (FIRST_PREDICTED_YEAR, MIN_TRAIN,
                                             bootstrap_by_event, verdict)
import injury_history as ih

OUT = Path(__file__).with_suffix(".json")
MIN_LABELS = 150


def populations(history, fights):
    """{population: set of (fighter, date)} from what was known before each
    fight, and nothing else.

    `fights` is every (fighter, date) with the date of that fighter's
    previous bout, so "since their last fight" is measurable.
    """
    usable = history[history["matched"].astype(bool)
                     & ~history["vague"].astype(bool)].copy()
    usable["known_by"] = pd.to_datetime(usable["known_by"], errors="coerce")
    usable["anchor_date"] = pd.to_datetime(usable["anchor_date"],
                                           errors="coerce")
    usable = usable.dropna(subset=["known_by"])
    by_fighter = {f: g for f, g in usable.groupby("fighter")}

    out = {name: set() for name in ("withdrew_since_last_fight",
                                    "hurt_in_last_fight",
                                    "surgery_or_long_layoff_2y",
                                    "any_known_injury_1y",
                                    "known_hurt_going_in")}
    for fighter, date, previous in fights:
        rows = by_fighter.get(fighter)
        if rows is None:
            continue
        date = pd.Timestamp(date)
        known = rows[rows["known_by"] < date]          # THE RULE
        if known.empty:
            continue
        key = (fighter, date.date())
        if previous is not None and pd.notna(previous):
            previous = pd.Timestamp(previous)
            since = known[(known["kind"] == "withdrawal")
                          & (known["anchor_date"] > previous)
                          & (known["anchor_date"] < date)]
            if not since.empty:
                out["withdrew_since_last_fight"].add(key)
            last = known[(known["kind"] == "in_fight")
                         & (known["anchor_date"] == previous)]
            if not last.empty:
                out["hurt_in_last_fight"].add(key)
        serious = known[(known["surgery"].astype(bool)
                         | (pd.to_numeric(known["months_out"],
                                          errors="coerce") >= 6))
                        & (known["known_by"] >= date - pd.Timedelta(days=730))]
        if not serious.empty:
            out["surgery_or_long_layoff_2y"].add(key)
        if (known["known_by"] >= date - pd.Timedelta(days=365)).any():
            out["any_known_injury_1y"].add(key)
        going_in = known[(known["kind"] == "fought_hurt")
                         & (known["disclosed"] == "before")
                         & (known["anchor_date"] == date)]
        if not going_in.empty:
            out["known_hurt_going_in"].add(key)
    return out


def report(label, residuals, events):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < MIN_LABELS:
        print(f"  {label:<46}{len(residuals):>6} labels - too few to report")
        return {"label": label, "n": int(len(residuals)), "too_few": True}
    mean = float(residuals.mean())
    low, high = bootstrap_by_event(residuals, events)
    print(f"  {label:<46}{len(residuals):>6}{mean:>+10.4f}   "
          f"[{low:+.4f}, {high:+.4f}]   {verdict(mean, low, high)}")
    return {"label": label, "n": int(len(residuals)),
            "events": int(len(np.unique(events))), "mean": mean,
            "low": low, "high": high}


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/injury_weight.json")
    if not ih.HISTORY.exists():
        print(f"  {ih.HISTORY} does not exist - build it first "
              f"(injury_history.py --build). Nothing was estimated.")
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

    history = ih.load()
    history["fighter"] = history["fighter"].map(norm_name)

    # Every appearance with the fighter's previous bout date.
    appearances = []
    for i, row in meta.iterrows():
        for name in (row["r_name"], row["b_name"]):
            appearances.append((norm_name(name), dates[i]))
    appearances.sort(key=lambda a: (a[0], a[1]))
    fights, last = [], {}
    for name, when in appearances:
        fights.append((name, when, last.get(name)))
        last[name] = when
    pops = populations(history, fights)
    for name, members in pops.items():
        print(f"  {name:<32}{len(members):>7} fighter-fights")

    # --- walk-forward predictions ------------------------------------------
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
        print(f"    {year}: {test.sum()} fights")

    frame = pd.DataFrame(rows).set_index("i")
    frame["date"] = dates.loc[frame.index].dt.date.values
    frame["event"] = meta.loc[frame.index, "event_name"].values
    frame["red"] = [norm_name(n) for n in meta.loc[frame.index, "r_name"]]
    frame["blue"] = [norm_name(n) for n in meta.loc[frame.index, "b_name"]]
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
        for key, flip in (((row.red, row.blue, row.date), False),
                          ((row.blue, row.red, row.date), True)):
            if key in priced:
                a, b = priced[key]
                return blend.devig(b, a) if flip else blend.devig(a, b)
        return np.nan

    frame["p_market"] = [market_for(r) for r in frame.itertuples()]
    frame["p_blend"] = [blend.blend(m, k) if np.isfinite(k) else m
                        for m, k in zip(frame.p_model, frame.p_market)]

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

    print("\n  mean residual = actual minus predicted, for the fighter "
          "carrying the history.\n  Negative = they under-perform the "
          "prediction by that much.")
    print(f"\n  {'population':<46}{'n':>6}{'mean':>10}   "
          f"{'95% interval':<22} verdict")
    print("  " + "-" * 108)
    results = []
    for name, members in pops.items():
        for column, against in (("p_model", "vs model"),
                                ("p_blend", "vs blend (what the app prints)")):
            got, events = residuals_for(members, column)
            results.append(report(f"{name}, {against}", got, events))

    shipped = [r for r in results if "blend" in r["label"]
               and not r.get("too_few") and r["high"] < 0]
    print()
    if shipped:
        for r in shipped:
            print(f"  EARNED on top of the blend: {r['label']} "
                  f"{r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]")
    else:
        print("  NOTHING SHIPS: no population under-performs the blended "
              "probability\n  with an interval that excludes zero.")

    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "history_rows": int(len(history)),
        "populations": {k: len(v) for k, v in pops.items()},
        "results": results,
    }, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
