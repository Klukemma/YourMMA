"""Fit, test and freeze the METHOD ESTIMATOR (engine/method_price.py) - on prices only.

WHY. A live card has no method prop price (the odds api prices no method
market), so tickets v2 (parlays/engine/sameday.py) price a live prop leg by
the estimator. Shown without its measured error, an estimate reads as a
quote. This script measures it against real prop prices and writes
engine/data/method_price.json, which the card reads both the frozen
coefficients and the error beside each leg from - no number in it is typed.

PRICES ONLY. Every fight it fits or tests on is read through
method_price.fit_rows(), which names its columns (LEG_COLUMNS,
ARCHIVE_COLUMNS) and reads no winner, method or other result: the target is
the six-way prop market (its de-vigged split and its prices), the inputs the
closing moneyline, the division, the scheduled rounds and each corner's UFC
record strictly before the fight (as parlay_legs.py wrote it). Nothing here
can learn how a fight ended.

THE PROTOCOL, in this order:
  DEV     fit 2016-2019, check 2020, over ALPHA_GRID x PRIOR_GRID: the pair
          with the lowest median error over all six props of the check year
          (ties to the earlier grid entry) is the dev choice, used for every
          fit after it. The test years are not read to choose anything.
  TEST    fit 2016-2020 at the dev choice, test 2021-2024: the median
          |estimate - price| / price of every prop, by leg type and side
          (fav: the moneyline's no-vig chance at least 0.5), and over all
          single props; beside it the scouts' ladder (constant split,
          moneyline alone, + division and rounds, + records) on the same
          split, the error by test year, and the loading the real prices
          imply over the estimate by the estimate's price bucket.
  FROZEN  fit 2016-2024 at the dev choice: the coefficients live use reads,
          with the sha256 of their canonical JSON. Its own error on a live
          card is not measured (no live prop price exists): the TEST table
          is the nearest measure, and the card says so.

    python engine/experiments/method_price_fit.py
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import method_price as mp  # noqa: E402

OUT = mp.COEFFS

DEV_FIT = ("2016-01-01", "2019-12-31")
DEV_CHECK = ("2020-01-01", "2020-12-31")
TEST_FIT = ("2016-01-01", "2020-12-31")
TEST = ("2021-01-01", "2024-12-31")
FROZEN_FIT = ("2016-01-01", "2024-12-31")

# Real-price buckets the loading is reported in (decimal, lower bound
# inclusive): the shape of the margin, report-only.
PRICE_BUCKETS = ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 9.0),
                 (9.0, 14.0), (14.0, 25.0), (25.0, math.inf))
# Chances the frozen loading curve is printed at, report-only.
CURVE_AT = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

def caveats(cov):
    """The file's caveats, every number in them read from the run."""
    y = lambda span: f"{span[0][:4]}-{span[1][:4]}"
    return [
        "Fitted and tested on prices only: the target is the six-way prop market, and no "
        "result of any fight it fits or tests on is read.",
        f"The error table is measured fitting on {y(TEST_FIT)} and testing on {y(TEST)}. "
        f"The frozen fit live cards read is fitted on {y(FROZEN_FIT)}, and its error on a "
        "live card cannot be measured (no live prop price exists): the test table is the "
        "nearest measure.",
        "History's moneyline is the closing line (odds.csv); a live card reads its own "
        "pair, a different snapshot, so a live estimate may miss by more than the table.",
        f"The dev choice (ridge penalty, prior weight) was made fitting {y(DEV_FIT)} and "
        f"checking {DEV_CHECK[0][:4]}, and every yearly refit uses it, so a backtest refit "
        f"for a year up to {DEV_CHECK[0][:4]} uses a choice that saw that year's prices "
        "(prices only, never a result).",
        f"History has no prop after {cov['last']}; the fit rows start in {cov['first'][:4]}, "
        f"so there is no refit before {cov['first'][:4]}.",
        "An estimate is what a book would probably charge, not an edge: the six-way market's "
        f"median overround is {_pct(cov['six_way_overround_median'])}% across its six outcomes, "
        f"the moneyline pair's {_pct(cov['moneyline_overround_median'])}%.",
    ]


def _in(rows, span):
    lo, hi = span
    return [r for r in rows if lo <= r["date"] <= hi]


def _pct(x):
    return None if x is None else round(100.0 * x, 1)


def dev_choice(rows):
    fit_rows, check = _in(rows, DEV_FIT), _in(rows, DEV_CHECK)
    grid = []
    for alpha in mp.ALPHA_GRID:
        for prior in mp.PRIOR_GRID:
            m = mp.fit(fit_rows, alpha, prior)
            err = mp.error_table(mp.errors(check, m))["all_single_props"]
            grid.append({"alpha": alpha, "prior_fights": prior, "all_single_props": err})
    best = min(grid, key=lambda g: g["all_single_props"])     # min keeps the first on a tie
    return {"fit": list(DEV_FIT), "check": list(DEV_CHECK),
            "fit_fights": len(mp.usable(fit_rows)), "check_fights": len(mp.usable(check)),
            "alpha": best["alpha"], "prior_fights": best["prior_fights"],
            "all_single_props": best["all_single_props"], "grid": grid}


def loading_by_price(rows, model):
    """The loading the real prices imply over the estimate, 1 / (decimal x c)
    - 1 with c the estimate's chance, by the ESTIMATE's price bucket - the
    curve as it held out of sample (bucketed by the real price instead, the
    estimate's own misses would sort into the buckets and read as margin)."""
    loads = []
    for row in mp.usable(rows):
        est = mp.predict(mp.features_of(row), model)
        for side in mp.SIDES:
            for t in mp.METHOD_TYPES:
                d = float(row[f"{mp.PRICE_COLS[t]}_{side}"])
                e = est[side][t]
                loads.append((e["decimal"], 1.0 / (d * e["market_chance"]) - 1.0))
    out = []
    for lo, hi in PRICE_BUCKETS:
        v = [x for d, x in loads if lo <= d < hi]
        out.append({"from": lo, "to": None if math.isinf(hi) else hi, "props": len(v),
                    "median_loading": float(np.median(v)) if v else None})
    return out


def holdout(rows, alpha, prior):
    fit_rows, test_rows = _in(rows, TEST_FIT), _in(rows, TEST)
    model = mp.fit(fit_rows, alpha, prior)
    table = mp.error_table(mp.errors(test_rows, model))
    ladder = []
    for name, use in mp.LADDER:
        m = mp.fit(fit_rows, alpha, prior, use)
        ladder.append({"inputs": name, "features": len(m["features"]),
                       "all_single_props": mp.error_table(mp.errors(test_rows, m))["all_single_props"]})
    by_year = {}
    for y in sorted({r["date"][:4] for r in test_rows}):
        by_year[y] = mp.error_table(mp.errors([r for r in test_rows if r["date"][:4] == y],
                                              model))["all_single_props"]
    return {"fit": list(TEST_FIT), "test": list(TEST),
            "fit_fights": len(mp.usable(fit_rows)), "test_fights": len(mp.usable(test_rows)),
            "metric": "median of |estimated decimal - real decimal| / real decimal",
            "side": "fav: the corner's no-vig moneyline chance is at least 0.5; dog: below",
            **table, "by_year": by_year, "ladder": ladder,
            "side_slope": model["side_slope"],
            "loading_by_estimate": loading_by_price(test_rows, model)}


def frozen(rows, alpha, prior):
    model = mp.fit(_in(rows, FROZEN_FIT), alpha, prior)
    curve = [{"chance": c, "loading": mp.loading(c, model),
              "decimal": 1.0 / (c * (1.0 + mp.loading(c, model)))} for c in CURVE_AT]
    return {"fit": list(FROZEN_FIT), "model": model, "sha256": mp.coefficient_hash(model),
            "loading_curve": curve}


def coverage(rows):
    usable = mp.usable(rows)
    return {"source": "experiments/parlay_legs.csv.gz (props, closing moneyline, records) "
                      "+ the archive's division and total_rounds",
            "first": usable[0]["date"] if usable else None,
            "last": usable[-1]["date"] if usable else None,
            "fights_by_year": dict(sorted(Counter(r["date"][:4] for r in usable).items())),
            "no_division": sum(mp.normal_division(r.get("division")) is None for r in usable),
            "five_rounds": sum(r.get("rounds_scheduled") == 5 for r in usable),
            "six_way_overround_median": float(np.median([
                sum(1.0 / float(r[f"{mp.PRICE_COLS[t]}_{s}"]) for t in mp.METHOD_TYPES
                    for s in mp.SIDES) - 1.0 for r in usable])) if usable else None,
            "moneyline_overround_median": float(np.median([
                1.0 / float(r["dec_red"]) + 1.0 / float(r["dec_blue"]) - 1.0
                for r in usable])) if usable else None}


def run(rows=None):
    rows = mp.fit_rows() if rows is None else rows
    dev = dev_choice(rows)
    tst = holdout(rows, dev["alpha"], dev["prior_fights"])
    frz = frozen(rows, dev["alpha"], dev["prior_fights"])
    cov = coverage(rows)
    return {"version": 1, "written_by": "engine/experiments/method_price_fit.py",
            "method_types": list(mp.METHOD_TYPES), "features": list(mp.FEATURES),
            "coverage": cov, "dev_choice": dev, "test": tst, "frozen": frz,
            "caveats": caveats(cov)}


def report(doc):
    t = doc["test"]
    lines = [f"METHOD ESTIMATOR - fit {t['fit'][0][:4]}-{t['fit'][1][:4]} "
             f"({t['fit_fights']:,} fights), test {t['test'][0][:4]}-{t['test'][1][:4]} "
             f"({t['test_fights']:,} fights, {t['props']:,} props)",
             f"  dev choice: alpha {doc['dev_choice']['alpha']}, prior "
             f"{doc['dev_choice']['prior_fights']} fights "
             f"(dev check {_pct(doc['dev_choice']['all_single_props'])}%)",
             "  median |estimate - price| / price:",
             f"    {'leg':<12}{'fav':>8}{'dog':>8}"]
    for typ in mp.METHOD_TYPES:
        lines.append(f"    {typ:<12}{_pct(t['by_leg'][typ]['fav']):>7}%{_pct(t['by_leg'][typ]['dog']):>7}%")
    lines.append(f"    all single props {_pct(t['all_single_props'])}%")
    lines.append("  ladder: " + "; ".join(f"{s['inputs']} {_pct(s['all_single_props'])}%"
                                         for s in t["ladder"]))
    lines.append("  by year: " + ", ".join(f"{y} {_pct(v)}%" for y, v in t["by_year"].items()))
    lines.append("  loading by estimated price: " + ", ".join(
        f"{b['from']:g}-{'' if b['to'] is None else format(b['to'], 'g')} "
        f"{_pct(b['median_loading'])}%" for b in t["loading_by_estimate"]))
    lines.append(f"  frozen {doc['frozen']['fit'][0][:4]}-{doc['frozen']['fit'][1][:4]}: "
                 f"{doc['frozen']['model']['fit_fights']:,} fights, sha256 {doc['frozen']['sha256'][:16]}...")
    return "\n".join(lines)


def main():
    doc = run()
    OUT.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    print(report(doc))
    print(f"\nwrote {OUT.relative_to(ENGINE.parent)}")


if __name__ == "__main__":
    main()
