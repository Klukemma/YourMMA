"""What does the stale UNAVAILABLE list cost the winner model?

prediction_row.UNAVAILABLE forces a suffix to NaN at prediction time. Its
career-column entries were written before career_stats.final_stats() existed;
final_stats supplies exactly those values and predict_card merges them into
the stats dict, and then _value throws them away because their names are on
the list.

So the winner model TRAINS on thirteen features it NEVER SEES when it
predicts:

    cd_bouts, cd_ctrl_share, cd_head_share, cd_kd_per15, cd_minutes
    each as a difference, a level and a _known flag

Every accuracy figure this project quotes comes from the walk-forward, which
builds its matrix from the training frame and therefore HAS those values. The
live card does not. If the mismatch costs anything, every number on the phone
is worse than the number in the backtest by an amount nobody has measured.

This measures it the only honest way: train exactly as usual, then blank
those columns on the TEST rows only, which is precisely the arrangement the
live path is in. The difference between that and the normal walk-forward is
what the stale list costs.

    python3 -m experiments.live_mismatch
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

FIRST_PREDICTED_YEAR = 2011
CONFIRM_FROM = 2020
MIN_TRAIN = 500


def blanked_columns(columns):
    """Every feature a live prediction cannot see, from the list itself."""
    from feature_inventory import all_specs
    from prediction_row import UNAVAILABLE

    out = []
    for spec in all_specs():
        red = getattr(spec, "red", None)
        if red and red[2:] in UNAVAILABLE:
            out.extend(spec.emits)
    return [c for c in out if c in columns]


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/live_mismatch.json")
    print("building features...")
    import predict_card as engine

    from experiments.calibrator_mismatch import fit_models, proba

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True).copy()
    years = pd.to_datetime(ufc["date"], errors="coerce").dt.year

    blanked = blanked_columns(X.columns)
    print(f"  {len(X.columns)} winner features; "
          f"{len(blanked)} are NaN at prediction time:")
    for name in blanked:
        print(f"    {name}")
    if not blanked:
        sys.exit("Nothing is being blanked, so there is nothing to measure.")

    rows = []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        models = fit_models(X[train], y[train])

        honest = X[test]
        # The live path's arrangement: trained with, predicted without. 0 is
        # what X carries for a missing value after predict_card's fillna, and
        # is what a live prediction's NaN becomes on the same line.
        crippled = honest.copy()
        crippled[blanked] = 0.0

        rows.append({
            "year": int(year),
            "n": int(test.sum()),
            "p_full": proba(models, honest, 3),
            "p_live": proba(models, crippled, 3),
            "y": y[test],
        })
        print(f"    {year}: {test.sum():,}", flush=True)

    def gather(key, keep):
        return np.concatenate([r[key] for r in rows if keep(r["year"])])

    print(f"\n  {'period':<16}{'full':>10}{'as live':>10}{'cost':>9}")
    print("  " + "-" * 45)
    results = []
    for label, keep in (("2011 onward", lambda v: True),
                        (f"{CONFIRM_FROM} onward", lambda v: v >= CONFIRM_FROM)):
        truth = gather("y", keep)
        full = gather("p_full", keep)
        live = gather("p_live", keep)
        a_full = ((full > 0.5) == (truth == 1)).mean()
        a_live = ((live > 0.5) == (truth == 1)).mean()
        print(f"  {label + ' acc':<16}{a_full:>10.1%}{a_live:>10.1%}"
              f"{a_full - a_live:>+9.1%}")
        print(f"  {label + ' AUC':<16}{roc_auc_score(truth, full):>10.3f}"
              f"{roc_auc_score(truth, live):>10.3f}"
              f"{roc_auc_score(truth, full) - roc_auc_score(truth, live):>+9.3f}")
        results.append({
            "period": label, "n": int(len(truth)),
            "accuracy_full": float(a_full), "accuracy_live": float(a_live),
            "auc_full": float(roc_auc_score(truth, full)),
            "auc_live": float(roc_auc_score(truth, live)),
            "logloss_full": float(log_loss(truth, np.clip(full, 1e-6, 1 - 1e-6))),
            "logloss_live": float(log_loss(truth, np.clip(live, 1e-6, 1 - 1e-6))),
        })

    confirm = results[-1]
    gap = confirm["accuracy_full"] - confirm["accuracy_live"]
    print()
    if gap > 0.005:
        print(f"  THE BACKTEST IS FLATTERING THE PHONE BY {gap:.1%}. Every "
              f"accuracy figure\n  this project quotes was measured with "
              f"features the live card never sees.")
    else:
        print(f"  The gap is {gap:+.1%}. The stale list costs little or "
              f"nothing, so the\n  quoted figures are not flattered by it "
              f"and fixing it is tidying, not a gain.")

    out = ENGINE / "experiments" / "live_mismatch.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "blanked": blanked,
        "results": results,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
