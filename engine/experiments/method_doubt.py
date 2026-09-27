"""Can anything tell us when the finish call is wrong?

The winner question has a wrongness layer that works: its flags score 0.703
where 0.5 is a coin toss. It works because three of its five features are
market-derived - the de-vigged price, the gap between the model and that
price, and whether the pick is an underdog. Strip those and it scores 0.605
against 0.609 for confidence alone, which is to say worse than nothing.

The method question has no market. engine/check_markets.py asked this plan
directly on 2026-09-27: h2h is priced by five bookmakers, totals is accepted
and priced by none, and every method or distance market returns 422
INVALID_MARKET. So the construction that works for the winner cannot be
copied here.

What is left is a SECOND OPINION FROM DIFFERENT MACHINERY. The Monte Carlo
simulator reads the same fighters through rates and hazards rather than a
gradient-boosted ensemble, and where two such systems disagree, one of them
is wrong. That is the same shape as model-against-market, with the simulator
standing in for the market.

The prior is not encouraging and is stated up front rather than discovered
afterwards: experiments/live_flags.py already asked whether simulator
disagreement helps on the WINNER question and found it carried nothing
(0.605). The argument for asking again is that the simulator is weak at
picking winners (AUC 0.559) and strongest at exactly this - duration within
2% and method within a few points - so the disagreement might mean something
here that it did not mean there.

WHAT IS MEASURED. For each fight in the confirm period, walk-forward:

    p_model      the method model's P(finish), trained on earlier years only
    p_sim        the simulator's P(finish) from point-in-time career rates
    |difference| the disagreement
    was_wrong    did the model's finish-or-distance call miss

and then whether a flagger reading those beats one reading confidence alone.
The bar is the same as it was for the winner question: it has to clear
confidence by a real margin or it does not ship, and the card goes on saying
nothing extra.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.method_model import (CLASSES, CONFIRM_FROM,
                                      FIRST_PREDICTED_YEAR, MIN_TRAIN,
                                      class_weights_for, level_columns,
                                      rate_columns, xgb)

SIMULATIONS = 1500      # 1.3pp standard error on P(finish); enough to rank
MIN_FLAG_TRAIN = 300
MAX_IMPUTED_SHARE = 0.5   # the same refusal fight_report makes on a live card


FAILURES = {}


def simulate_finish_probability(row, rounds, rng):
    """The simulator's P(finish) for one fight, from point-in-time rates.

    Failures are counted BY REASON rather than swallowed. The first version
    of this returned None on any exception, every single fight failed, and
    the experiment went on to print a confident conclusion about a simulator
    that had never run.
    """
    import simulate as sim

    try:
        red = sim.rates_from_career_stats(row, "r")
        blue = sim.rates_from_career_stats(row, "b")
    except Exception as err:                            # noqa: BLE001
        FAILURES[f"rates: {type(err).__name__}: {err}"[:120]] = \
            FAILURES.get(f"rates: {type(err).__name__}: {err}"[:120], 0) + 1
        return None
    try:
        # The league rates as defaults, and the same refusal the card makes
        # when more than half a fighter's rates had to be assumed. Calling
        # simulate_fight without defaults - as the first version did - fails
        # on any fighter missing one rate, which is most of them.
        out = sim.simulate_fight(red, blue, rounds=rounds,
                                 n_sims=SIMULATIONS, rng=rng,
                                 defaults=sim.league_rates())
    except Exception as err:                            # noqa: BLE001
        key = f"simulate: {type(err).__name__}: {err}"[:120]
        FAILURES[key] = FAILURES.get(key, 0) + 1
        return None
    # A draw is not a finish and is not a decision the model has a class for;
    # it goes with the decisions, which is where the training target puts it.
    total = len(sim._REQUIRED_RATE_FIELDS)
    for imputed in (out.imputed_a, out.imputed_b):
        if len(imputed) / total > MAX_IMPUTED_SHARE:
            key = "refused: more than half the rates assumed"
            FAILURES[key] = FAILURES.get(key, 0) + 1
            return None

    decision = out.dec_prob_a + out.dec_prob_b + out.draw_prob
    return float(1.0 - decision)


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/method_doubt.json")
    print("building features...")
    import predict_card as engine

    from feature_inventory import finish_level_names, method_rate_names

    added = set(method_rate_names()) | set(finish_level_names())
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

    X = pd.concat([base, aligned(level_columns(engine)),
                   aligned(rate_columns(engine))], axis=1).to_numpy()

    print(f"  {len(meta):,} fights; simulating the confirm period at "
          f"{SIMULATIONS:,} runs each")

    # --- walk forward, keeping P(finish) and whether the call missed -------
    rows = []
    rng = np.random.default_rng(11)
    for year in sorted(years.unique()):
        if year < max(FIRST_PREDICTED_YEAR, CONFIRM_FROM):
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        model = xgb(objective="multi:softprob", eval_metric="mlogloss",
                    num_class=3)
        model.fit(X[train], y[train],
                  sample_weight=class_weights_for(y[train]), verbose=False)
        p_finish = 1.0 - model.predict_proba(X[test])[:, 0]

        block = meta[test]
        for offset, (_, fight) in enumerate(block.iterrows()):
            rounds = 5 if int(fight.get("total_rounds", 3) or 3) >= 5 else 3
            p_sim = simulate_finish_probability(fight, rounds, rng)
            rows.append({
                "year": int(year),
                "p_model": float(p_finish[offset]),
                "p_sim": p_sim,
                "finished": float(y[test][offset] != 0),
            })
        print(f"    {year}: {test.sum():,} fights simulated", flush=True)

    frame = pd.DataFrame(rows)
    frame["confidence"] = (frame["p_model"] - 0.5).abs() * 2
    frame["called_finish"] = frame["p_model"] > 0.5
    frame["was_wrong"] = (frame["called_finish"] != (frame["finished"] == 1)).astype(float)
    usable = frame[frame["p_sim"].notna()].copy()
    usable["disagreement"] = (usable["p_model"] - usable["p_sim"]).abs()

    if FAILURES:
        print("\n  simulations that did not run:")
        for reason, count in sorted(FAILURES.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>6}  {reason}")

    print(f"\n  simulated {len(usable):,} of {len(frame):,}")
    if len(usable) < MIN_FLAG_TRAIN * 2:
        # An earlier version printed "DOES NOT CLEAR THE BAR" over zero rows,
        # which is a conclusion about nothing wearing the clothes of a result.
        sys.exit(f"\nOnly {len(usable):,} fights could be simulated, which is "
                 f"too few to say anything. No verdict is reported. The "
                 f"failure reasons above are the thing to fix.")
    print(f"  the finish call missed {usable['was_wrong'].mean():.1%} of the time")
    print(f"  model says finish {usable['p_model'].mean():.1%}, "
          f"simulator says {usable['p_sim'].mean():.1%}, "
          f"really {usable['finished'].mean():.1%}")

    # --- does anything predict the miss? ----------------------------------
    def quality(columns, label):
        data = usable.dropna(subset=columns)
        if len(data) < MIN_FLAG_TRAIN * 2:
            return label, float("nan"), 0
        cut = len(data) // 2
        fit, held = data.iloc[:cut], data.iloc[cut:]
        if fit["was_wrong"].nunique() < 2 or held["was_wrong"].nunique() < 2:
            return label, float("nan"), 0
        flagger = LogisticRegression(max_iter=1000)
        flagger.fit(fit[columns], fit["was_wrong"])
        scores = flagger.predict_proba(held[columns])[:, 1]
        return label, float(roc_auc_score(held["was_wrong"], scores)), len(held)

    print(f"\n  {'flagger':<34}{'quality':>9}{'n':>8}")
    print("  " + "-" * 51)
    results = []
    for columns, label in (
            (["confidence"], "confidence alone (the bar)"),
            (["disagreement"], "simulator disagreement alone"),
            (["p_model", "confidence"], "model + confidence"),
            (["p_model", "confidence", "disagreement"],
             "model + confidence + disagreement"),
            (["p_model", "confidence", "disagreement", "p_sim"],
             "everything available before the bell")):
        label, auc, n = quality(columns, label)
        results.append({"label": label, "quality": auc, "scored": n})
        print(f"  {label:<34}{auc:>9.3f}{n:>8}")

    # THE BAR IS THE BEST FLAGGER THAT NEEDS NO SIMULATOR, not confidence
    # alone. An earlier version compared against confidence alone and
    # announced that the simulator "clears the bar" at 0.615 against 0.590 -
    # while model+confidence, which costs nothing and runs already, was
    # sitting at 0.608. The simulator was being credited for a gain it had
    # not produced.
    bar = next(r["quality"] for r in results if r["label"] == "model + confidence")
    withsim = next(r["quality"] for r in results
                   if r["label"] == "model + confidence + disagreement")

    data = usable.dropna(subset=["p_model", "confidence", "disagreement"])
    cut = len(data) // 2
    fit, held = data.iloc[:cut], data.iloc[cut:]

    def scores(columns):
        model = LogisticRegression(max_iter=1000)
        model.fit(fit[columns], fit["was_wrong"])
        return model.predict_proba(held[columns])[:, 1]

    s_bar = scores(["p_model", "confidence"])
    s_sim = scores(["p_model", "confidence", "disagreement"])
    truth = held["was_wrong"].to_numpy()
    rng_boot = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        idx = rng_boot.integers(0, len(truth), len(truth))
        if len(np.unique(truth[idx])) < 2:
            continue
        diffs.append(roc_auc_score(truth[idx], s_sim[idx])
                     - roc_auc_score(truth[idx], s_bar[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])

    print(f"\n  against the flagger that needs no simulator "
          f"(model + confidence, {bar:.3f}):")
    print(f"    adding the disagreement gives {withsim:.3f}, "
          f"a difference of {withsim - bar:+.4f}")
    print(f"    95% interval [{low:+.4f}, {high:+.4f}]")
    if low > 0:
        print("  THE SIMULATOR EARNS ITS PLACE.")
    else:
        print("  THE INTERVAL CROSSES ZERO. The simulator's disagreement adds "
              "nothing\n  the model's own probability and confidence do not "
              "already say, so the\n  card keeps saying nothing extra about "
              "how sure it is - the same answer\n  live_flags.py got for the "
              "winner question.")
    results.append({"label": "difference from adding the simulator",
                    "quality": float(withsim - bar),
                    "interval": [float(low), float(high)]})

    out = ENGINE / "experiments" / "method_doubt.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "simulations": SIMULATIONS,
        "confirm_from": CONFIRM_FROM,
        "scored": int(len(usable)),
        "results": results,
    }, indent=1))
    usable.to_csv(ENGINE / "experiments" / "method_doubt.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
