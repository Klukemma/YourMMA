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


# What the list held BEFORE the career columns came off it. Kept literal so
# the cost of the old arrangement stays measurable after the fix - otherwise
# "the fix recovered 1.3 points" becomes a claim nobody can re-run.
BEFORE_THE_FIX = ("cd_bouts", "cd_minutes", "cd_kd_per15", "cd_ctrl_share",
                  "cd_head_share", "cd_opp_ctrl_share", "cd_opp_sub_per15",
                  "cd_wins", "cd_losses", "cd_win_rate",
                  "won_L3", "splm_L3", "str_acc_L3", "td_avg_L3",
                  "data_reliability", "damage_log",
                  "striking_trajectory", "accuracy_trajectory")


def features_for(suffixes, columns):
    """Every emitted feature derived from any of these suffixes."""
    from feature_inventory import all_specs

    out = []
    for spec in all_specs():
        red = getattr(spec, "red", None)
        if red and red[2:] in suffixes:
            out.extend(spec.emits)
    return [c for c in out if c in columns]


def blanked_columns(columns):
    """Every feature a live prediction still cannot see."""
    from prediction_row import UNAVAILABLE

    return features_for(set(UNAVAILABLE), columns)


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/live_mismatch.json")
    print("building features...")
    import predict_card as engine

    from experiments.calibrator_mismatch import fit_models, proba

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True).copy()
    years = pd.to_datetime(ufc["date"], errors="coerce").dt.year

    now = blanked_columns(X.columns)
    before = features_for(set(BEFORE_THE_FIX), X.columns)
    recovered = sorted(set(before) - set(now))
    print(f"  {len(X.columns)} winner features")
    print(f"    {len(before)} were NaN at prediction time before the fix")
    print(f"    {len(now)} still are - genuinely not reconstructable")
    print(f"    {len(recovered)} recovered: {', '.join(recovered)}")
    if not before:
        sys.exit("Nothing is being blanked, so there is nothing to measure.")

    # THREE SEEDS. A single walk-forward of this scatters by about half a
    # point, which is the size of the effect being measured - the method-model
    # work in this repository already believed a one-run number once and had
    # to take it back.
    SEEDS = (42, 7, 2024)
    rows = []
    for seed in SEEDS:
      for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        models = fit_models(X[train], y[train], seed=seed)

        honest = X[test]
        # The live path's arrangement: trained with, predicted without. 0 is
        # what X carries for a missing value after predict_card's fillna, and
        # is what a live prediction's NaN becomes on the same line.
        as_before = honest.copy()
        as_before[before] = 0.0
        as_now = honest.copy()
        as_now[now] = 0.0

        rows.append({
            "seed": seed,
            "year": int(year),
            "n": int(test.sum()),
            "p_full": proba(models, honest, 3),
            "p_before": proba(models, as_before, 3),
            "p_live": proba(models, as_now, 3),
            "y": y[test],
        })
      print(f"    seed {seed} done", flush=True)

    def gather(key, keep, seed):
        return np.concatenate([r[key] for r in rows
                               if keep(r["year"]) and r["seed"] == seed])

    print(f"\n  {'period':<14}{'backtest':>10}{'live before':>13}"
          f"{'live now':>10}{'recovered':>11}")
    print("  " + "-" * 58)
    results = []
    for label, keep in (("2011 onward", lambda v: True),
                        (f"{CONFIRM_FROM} onward", lambda v: v >= CONFIRM_FROM)):
        per_seed = {"p_full": [], "p_before": [], "p_live": []}
        aucs = {"p_full": [], "p_before": [], "p_live": []}
        for seed in SEEDS:
            truth = gather("y", keep, seed)
            for key in per_seed:
                p = gather(key, keep, seed)
                per_seed[key].append(float(((p > 0.5) == (truth == 1)).mean()))
                aucs[key].append(float(roc_auc_score(truth, p)))
        m = {k: float(np.mean(v)) for k, v in per_seed.items()}
        sd = {k: float(np.std(v, ddof=1)) for k, v in per_seed.items()}
        print(f"  {label + ' acc':<14}{m['p_full']:>10.1%}{m['p_before']:>13.1%}"
              f"{m['p_live']:>10.1%}{m['p_live'] - m['p_before']:>+11.1%}")
        print(f"  {'  +- over seeds':<14}{sd['p_full']:>10.1%}{sd['p_before']:>13.1%}"
              f"{sd['p_live']:>10.1%}")
        am = {k: float(np.mean(v)) for k, v in aucs.items()}
        print(f"  {label + ' AUC':<14}{am['p_full']:>10.3f}{am['p_before']:>13.3f}"
              f"{am['p_live']:>10.3f}{am['p_live'] - am['p_before']:>+11.3f}")
        results.append({
            "period": label, "seeds": list(SEEDS),
            "accuracy_backtest": m["p_full"], "accuracy_before": m["p_before"],
            "accuracy_now": m["p_live"], "accuracy_sd": sd,
            "auc_backtest": am["p_full"], "auc_before": am["p_before"],
            "auc_now": am["p_live"],
        })

    c = results[-1]
    gained = c["accuracy_now"] - c["accuracy_before"]
    left = c["accuracy_backtest"] - c["accuracy_now"]

    # THE RIGHT UNCERTAINTY IS NOT THE SEED SPREAD. That measures how much the
    # fitting wobbles; it says nothing about how much a 2,339-fight sample
    # wobbles, which at 60% accuracy is about a point on its own. A first
    # version of this compared a 0.7-point difference against a 0.1-point seed
    # spread and announced that it cleared the noise.
    #
    # The two models are scored on the SAME fights, so the comparison is
    # paired and a bootstrap over fights is what it needs.
    keep = lambda v: v >= CONFIRM_FROM
    truth = gather("y", keep, SEEDS[0])
    before_p = np.mean([gather("p_before", keep, s) for s in SEEDS], axis=0)
    now_p = np.mean([gather("p_live", keep, s) for s in SEEDS], axis=0)
    hit_before = (before_p > 0.5) == (truth == 1)
    hit_now = (now_p > 0.5) == (truth == 1)
    rng_boot = np.random.default_rng(0)
    diffs = [float(hit_now[i].mean() - hit_before[i].mean())
             for i in (rng_boot.integers(0, len(truth), len(truth))
                       for _ in range(4000))]
    low, high = np.percentile(diffs, [2.5, 97.5])
    disagree = int((hit_now != hit_before).sum())

    print(f"\n  The fix moves the confirm period by {gained:+.1%}.")
    print(f"  Paired bootstrap over the same {len(truth):,} fights: "
          f"95% [{low:+.1%}, {high:+.1%}]")
    print(f"  They disagree on {disagree:,} fights of {len(truth):,}.")
    if low > 0 or high < 0:
        print("  The interval excludes zero, so the difference is real.")
    else:
        print("  THE INTERVAL CROSSES ZERO. The career columns were not what "
              "the mismatch\n  was costing, and restoring them neither helps "
              "nor hurts measurably -\n  AUC moves by a thousandth. Keeping "
              "the fix is a correctness argument,\n  that a model should not "
              "train on features it cannot read, and not a\n  performance "
              "one. It must not be sold as a gain.")
    c["paired_interval"] = [float(low), float(high)]
    print(f"\n  {left:.1%} still separates the backtest from the live card. "
          f"That part is real:\n  a snapshot cannot rebuild a window over a "
          f"fighter's last three bouts, and\n  every accuracy figure this "
          f"project quotes is measured with those features.")

    out = ENGINE / "experiments" / "live_mismatch.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "still_unavailable": now,
        "recovered": recovered,
        "results": results,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
