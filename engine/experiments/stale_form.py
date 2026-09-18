"""Did the 2026 collapse come from missing data rather than a regime change?

auc_by_year.py found the model ranked fights at AUC 0.749-0.826 every year from
2015 to 2025 and then fell to 0.646 in 2026 - 5.6 standard deviations below its
own mean. A model that worked for eleven years does not forget how; something
changed in the inputs.

Something did. The dataset has a four-month hole: every per-fight striking,
takedown and control statistic is missing from 2025-09-13 to 2025-12-06 (145
fights, with October and November 2025 entirely blank). Winners and methods
survived, so the ratings are intact - it is the form features that vanished.

That hole reaches into 2026 because form features are built from a fighter's
previous bouts. 56.6% of 2026 fights have at least one fighter who fought in the
blank window, so the model goes in with a stale or empty picture of how that
fighter has recently performed.

This splits 2026 by how many of the two fighters are affected and scores each
group separately. If clean fights still rank near the historical 0.78 while
affected ones collapse, the 2026 drop is a data-repair problem and not evidence
that the model needs new inputs.

A placebo runs the same split on 2025 using the equivalent window a year earlier,
where no data is missing. If the placebo also separates, the grouping is picking
up something about the fighters rather than the missing data, and the result
means nothing.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from experiments.calibrator_mismatch import fit_models, proba

DATA_DIR = ENGINE / "data"
UFC_CSV = Path(os.environ.get("UFC_CSV", DATA_DIR / "UFC_with_mmr_rebuilt_dedup.csv"))

# The observed hole. Derived from the data, not assumed: these are the bounds of
# the rows where r_total_str_landed is null.
BLANK_START = pd.Timestamp("2025-09-13")
BLANK_END = pd.Timestamp("2025-12-06")

# Same span, one year earlier, where the statistics are all present.
PLACEBO_START = BLANK_START - pd.DateOffset(years=1)
PLACEBO_END = BLANK_END - pd.DateOffset(years=1)

STAT_COL = "r_total_str_landed"


def blank_window_bounds(csv_path=None):
    """Re-derive the hole from the dataset so the constants cannot go stale."""
    df = pd.read_csv(csv_path or UFC_CSV, usecols=["date", STAT_COL],
                     low_memory=False)
    dates = pd.to_datetime(df["date"], errors="coerce")
    blank = dates[df[STAT_COL].isna() & dates.notna()]
    if blank.empty:
        return None, None, 0
    return blank.min(), blank.max(), len(blank)


def fighters_active_in(ufc, start, end):
    """Names with at least one bout inside [start, end]."""
    inside = ufc[(ufc["date"] >= start) & (ufc["date"] <= end)]
    return set(inside["r_name"].dropna()) | set(inside["b_name"].dropna())


def affected_count(ufc, affected):
    """How many of the two fighters in each row are in `affected` (0, 1 or 2)."""
    return (ufc["r_name"].isin(affected).astype(int)
            + ufc["b_name"].isin(affected).astype(int))


def score(p, yt):
    if len(np.unique(yt)) < 2:
        return None
    return {
        "n": len(yt),
        "accuracy": float(((p >= 0.5).astype(float) == yt).mean()),
        "auc": float(roc_auc_score(yt, p)),
        "brier": float(np.mean((p - yt) ** 2)),
    }


def report(title, groups):
    print(f"\n{title}")
    print(f"  {'group':<28}{'n':>6}{'accuracy':>11}{'AUC':>9}{'Brier':>9}")
    print("  " + "-" * 61)
    for label, s in groups:
        if s is None:
            print(f"  {label:<28}{'-':>6}{'too few':>11}")
            continue
        print(f"  {label:<28}{s['n']:>6}{s['accuracy']:>11.1%}"
              f"{s['auc']:>9.3f}{s['brier']:>9.4f}")


def split_year(X, y, ufc, years, target_year, affected, cutoff=None):
    """Train on everything before `target_year`, score it split by exposure."""
    train = (years < target_year).values
    test = (years == target_year).values
    if cutoff is not None:
        test &= (ufc["date"] < cutoff).values
    if train.sum() < 500 or test.sum() < 50:
        return None

    models = fit_models(X[train], y[train])
    p = proba(models, X[test], 3)
    yt = y[test]
    exposure = affected_count(ufc[test], affected).to_numpy()

    groups = [(f"all {target_year}", score(p, yt))]
    for k, label in [(0, "neither fighter"), (1, "one fighter"), (2, "both fighters")]:
        m = exposure == k
        groups.append((f"  {label}", score(p[m], yt[m]) if m.sum() >= 30 else None))
    clean = exposure == 0
    dirty = exposure > 0
    if clean.sum() >= 30 and dirty.sum() >= 30:
        groups.append(("  any exposure", score(p[dirty], yt[dirty])))
    return groups


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/stale_form_scratch.json")

    start, end, n_blank = blank_window_bounds()
    print(f"blank-statistics window in the data : "
          f"{start.date() if start is not None else 'none'} -> "
          f"{end.date() if end is not None else 'none'}  ({n_blank} fights)")
    print(f"window used for the split           : "
          f"{BLANK_START.date()} -> {BLANK_END.date()}")
    print(f"placebo window (statistics present) : "
          f"{PLACEBO_START.date()} -> {PLACEBO_END.date()}")

    print("\nbuilding features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True).copy()
    ufc["date"] = pd.to_datetime(ufc["date"], errors="coerce")
    years = ufc["date"].dt.year

    exposed = fighters_active_in(ufc, BLANK_START, BLANK_END)
    placebo = fighters_active_in(ufc, PLACEBO_START, PLACEBO_END)
    print(f"\nfighters in the blank window   : {len(exposed)}")
    print(f"fighters in the placebo window : {len(placebo)}")

    real = split_year(X, y, ufc, years, 2026, exposed)
    if real:
        report("2026, split by exposure to the blank window", real)

    # The placebo scores 2025 only up to the point the hole opens, so no
    # scored fight is itself missing its statistics.
    sham = split_year(X, y, ufc, years, 2025, placebo, cutoff=BLANK_START)
    if sham:
        report(f"PLACEBO: 2025 before {BLANK_START.date()}, "
               f"split by the {PLACEBO_START.year} window", sham)

    print("\n" + "=" * 63)
    if real:
        by = dict(real)
        clean = by.get("  neither fighter")
        dirty = by.get("  any exposure")
        if clean and dirty:
            gap = clean["auc"] - dirty["auc"]
            print(f"2026 AUC, fighters with intact recent form : {clean['auc']:.3f} "
                  f"(n={clean['n']})")
            print(f"2026 AUC, at least one affected fighter    : {dirty['auc']:.3f} "
                  f"(n={dirty['n']})")
            print(f"gap                                        : {gap:+.3f}")
            print("\nHistorical reference: 2015-2025 mean AUC 0.776, "
                  "2026 overall 0.646.")
    if sham:
        byp = dict(sham)
        pc, pd_ = byp.get("  neither fighter"), byp.get("  any exposure")
        if pc and pd_:
            print(f"\nPlacebo gap (should be near zero)          : "
                  f"{pc['auc'] - pd_['auc']:+.3f}")

    rows = []
    for tag, groups in [("2026", real), ("placebo-2025", sham)]:
        for label, s in (groups or []):
            if s:
                rows.append({"period": tag, "group": label.strip(), **s})
    if rows:
        out = ENGINE / "experiments" / "stale_form.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
