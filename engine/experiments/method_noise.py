"""How much of a method-model "gain" is the random seed?

A gain of a hundredth of a nat was reported for the finish-feature block and
shipped on it. Repeating the same comparison with three seeds showed the
model's own scatter to be the same size, so the number that justified the
change could not support it. This file settles the question with enough
repeats to have an answer rather than an impression.

XGBoost's histogram build is not deterministic across threads, so two runs of
identical code give different trees. Over 3,310 fights that moves log loss by
a couple of thousandths - which is nothing, unless the effect being measured
is also a couple of thousandths.

Eight seeds per variant, walk-forward each time, and the difference reported
against the standard error of the mean rather than against a single run.
A variant has to clear TWO standard errors on both metrics to count.

    python3 -m experiments.method_noise
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.method_model import (CLASSES, CONFIRM_FROM,
                                      FIRST_PREDICTED_YEAR, level_columns,
                                      rate_columns, score, walk_forward)

SEEDS = (42, 7, 2024, 13, 99, 500, 8675309, 31337)


def main():
    import os
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/method_noise.json")
    print("building features...")
    import predict_card as engine

    from feature_inventory import finish_level_names, method_rate_names

    added = set(method_rate_names()) | set(finish_level_names()) \
        | {"division_finish_prior"}
    mask = engine.ufc_valid["target_method"].isin(CLASSES).values
    full = engine.X_valid.reset_index(drop=True)[mask]
    base = full[[c for c in full.columns if c not in added]].reset_index(drop=True)

    meta = engine.ufc_valid.reset_index(drop=True)[mask].copy()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    years = meta["date"].dt.year
    order = {c: i for i, c in enumerate(CLASSES)}
    y = meta["target_method"].map(order).to_numpy()

    valid = engine.ufc["target_win"].notna().to_numpy()

    def aligned(frame):
        return (frame.reset_index(drop=True)[valid]
                .reset_index(drop=True)[mask].reset_index(drop=True))

    levels = aligned(level_columns(engine))
    rates = aligned(rate_columns(engine))
    from experiments.method_model import division_column

    divisions = aligned(division_column(engine))

    variants = {
        "BASE": base,
        "BOTH+W": pd.concat([base, levels, rates], axis=1),
        "+DIVISION": pd.concat([base, levels, rates, divisions], axis=1),
    }
    print(f"  {len(base):,} fights, {len(SEEDS)} seeds, "
          f"{len(variants)} variants")

    results = {}
    for label, X in variants.items():
        lls, f1s = [], []
        arr = X.to_numpy()
        for seed in SEEDS:
            proba, idx = walk_forward(arr, y, years, weighted=True, seed=seed)
            keep = years.to_numpy()[idx] >= CONFIRM_FROM
            row = score(label, proba[keep], y[idx][keep])
            lls.append(row["log_loss"])
            f1s.append(row["macro_f1"])
            print(f"    {label:<12} seed {seed:<8} "
                  f"log loss {row['log_loss']:.4f}  "
                  f"macro-F1 {row['macro_f1']:.4f}", flush=True)
        results[label] = {"log_loss": lls, "macro_f1": f1s}

    def sem(values):
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    print("\n" + "=" * 72)
    print(f"  {'variant':<12}{'log loss':>26}{'macro-F1':>26}")
    print("  " + "-" * 64)
    for label, r in results.items():
        print(f"  {label:<12}"
              f"{np.mean(r['log_loss']):>17.4f} +-{sem(r['log_loss']):<7.4f}"
              f"{np.mean(r['macro_f1']):>17.4f} +-{sem(r['macro_f1']):<7.4f}")
    print("  (+- is the standard error of the mean over the seeds)")

    b = results["BASE"]
    print("\n  against BASE:")
    for label, r in results.items():
        if label == "BASE":
            continue
        d_ll = np.mean(b["log_loss"]) - np.mean(r["log_loss"])
        d_f1 = np.mean(r["macro_f1"]) - np.mean(b["macro_f1"])
        e_ll = np.sqrt(sem(b["log_loss"]) ** 2 + sem(r["log_loss"]) ** 2)
        e_f1 = np.sqrt(sem(b["macro_f1"]) ** 2 + sem(r["macro_f1"]) ** 2)
        real = d_ll > 2 * e_ll and d_f1 > 2 * e_f1
        print(f"    {label:<12} log loss {d_ll:+.4f} (2se {2*e_ll:.4f})  "
              f"macro-F1 {d_f1:+.4f} (2se {2*e_f1:.4f})   "
              f"{'clears the noise' if real else 'INSIDE THE NOISE'}")

    out = ENGINE / "experiments" / "method_noise.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": list(SEEDS),
        "confirm_from": CONFIRM_FROM,
        "results": results,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
