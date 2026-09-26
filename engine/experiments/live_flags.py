"""Can the second layer run on a fight nobody has priced?

The failure model works and is measured: over the confirm period its flags
score around 0.70 where 0.5 is a coin toss. It has never once run on a
prediction the user actually sees, and this file is about why.

Three of its five features are market-derived - the de-vigged price, the gap
between the model and that price, and whether the pick is an underdog. On a
live card there is no price, so features() returns None and the layer is
silently skipped. What is left is the model's own probability and its
confidence, and confidence is close to useless as a guide to correctness:
that is the finding the failure model was built to work around.

So the question is whether something else available BEFORE the bell carries
the same kind of information the market did. The candidate is disagreement
again, but internal: the Monte Carlo simulator reads the same fighters through
completely different machinery - rates and hazards rather than a gradient
boosting ensemble - and when the two disagree, one of them is wrong.

This measures that. It reports, walk-forward and on honest point-in-time
statistics:

    ODDS         the existing five-feature flagger, where a price exists
    NO-ODDS      model, simulator and record features, on every fight
    CONFIDENCE   confidence alone, the baseline both must beat

If NO-ODDS does not clear CONFIDENCE by a real margin it does not ship, and
the live card keeps saying nothing rather than showing a flag that is noise
wearing a percentage.
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

from experiments.calibrator_mismatch import fit_models, proba
from experiments.historical_backtest import (CONFIRM_FROM,
                                             FIRST_PREDICTED_YEAR, MIN_TRAIN)
from failure_model import COLUMNS as ODDS_COLUMNS
from failure_model import MIN_TRAIN as FLAG_MIN_TRAIN
from failure_model import features as odds_features
from roi import load_odds

ODDS = Path(os.environ.get("ODDS_CSV", ENGINE / "data" / "odds.csv"))
SIMULATIONS = 3000        # 0.35pp standard error, and 8,000 fights to get through

# Everything here is known before the opening bell.
NO_ODDS_COLUMNS = [
    "model_p",            # what the ensemble says
    "confidence",         # the baseline, carried so the fit can ignore it
    "sim_p",              # what the simulator says about the same pick
    "sim_disagreement",   # |model - simulator|, the internal analogue of
                          # disagreeing with the market
    "sim_available",      # 0 when the simulator refused, which is itself a
                          # statement about how measurable the fight was
    "thin_record",        # prior bouts of the LESS experienced corner
    "striking_known",     # whether the matchup could be described at all
    "grappling_known",
]


def simulate_row(row):
    """The simulator's probability for the red corner, or NaN if it refuses."""
    import fight_report as fr

    red = {k[2:]: row[k] for k in row.index
           if isinstance(k, str) and k.startswith("r_")}
    blue = {k[2:]: row[k] for k in row.index
            if isinstance(k, str) and k.startswith("b_")}
    rounds = 5 if float(row.get("total_rounds", 3) or 3) == 5 else 3
    summary = fr.summarise(
        fr.simulate_matchup(red, blue, rounds=rounds, n_sims=SIMULATIONS),
        "r", "b")
    if summary is None:
        return float("nan")
    return summary["red_win"] + 0.5 * summary["draw"]


def walk_forward(X, y, ufc):
    """Predict each year from earlier years, carrying the live features."""
    years = ufc["date"].dt.year
    rows = []
    for year in sorted(years.unique()):
        if year < FIRST_PREDICTED_YEAR:
            continue
        train = (years < year).values
        test = (years == year).values
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        models = fit_models(X[train], y[train])
        p = proba(models, X[test], 3)
        block = ufc[test]
        for (_, fight), prob, actual in zip(block.iterrows(), p, y[test]):
            red_wins = prob >= 0.5
            sim_red = simulate_row(fight)
            # The simulator's probability for the SAME pick the model made,
            # which is the only way the two are comparable.
            sim_p = (sim_red if red_wins else 1 - sim_red)
            model_p = float(prob if red_wins else 1 - prob)
            rows.append({
                "date": fight["date"], "year": int(year),
                "pick": fight["r_name"] if red_wins else fight["b_name"],
                "opponent": fight["b_name"] if red_wins else fight["r_name"],
                "win_probability": model_p,
                "confidence": float(abs(2 * prob - 1)),
                "won": bool((actual == 1.0) == red_wins),
                "model_p": model_p,
                "sim_p": 0.5 if sim_p != sim_p else float(sim_p),
                "sim_disagreement": (0.0 if sim_p != sim_p
                                     else abs(model_p - float(sim_p))),
                "sim_available": 0.0 if sim_p != sim_p else 1.0,
                "thin_record": float(min(
                    _number(fight.get("r_cd_bouts")),
                    _number(fight.get("b_cd_bouts")))),
                "striking_known": _number(fight.get("mx_striking_known")),
                "grappling_known": _number(fight.get("mx_grappling_known")),
            })
        print(f"  {year}: trained on {train.sum():,}, "
              f"predicted {test.sum():,}", flush=True)
    return rows


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if value != value else value


def walk_forward_quality(rows, columns, label, odds_index=None):
    """Fit the flagger on settled predictions only, then score it.

    Identical protocol to the odds flagger: at every prediction the model has
    seen only what had already settled, so nothing here is scored by a fit
    that saw its own outcome.
    """
    usable = []
    for row in sorted(rows, key=lambda r: pd.to_datetime(r["date"])):
        if odds_index is not None:
            feat = odds_features(row, odds_index)
            if feat is None:
                continue
        else:
            feat = {c: row[c] for c in columns}
        usable.append((row, feat))

    scored = []
    for i, (row, feat) in enumerate(usable):
        history = usable[:i]
        if len(history) < FLAG_MIN_TRAIN:
            continue
        y = np.array([0.0 if h["won"] else 1.0 for h, _ in history])
        if len(np.unique(y)) < 2:
            continue
        frame = pd.DataFrame([f for _, f in history])[columns]
        model = LogisticRegression(max_iter=1000).fit(frame, y)
        p_fail = float(model.predict_proba(
            pd.DataFrame([feat])[columns])[0, 1])
        scored.append({"date": row["date"], "year": row["year"],
                       "p_fail": p_fail, "failed": not row["won"]})

    return {"label": label, "scored": len(scored),
            "quality": _auc(scored), "rows": scored}


def _auc(scored):
    if len(scored) < 20:
        return float("nan")
    y = np.array([1.0 if s["failed"] else 0.0 for s in scored])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, [s["p_fail"] for s in scored]))


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/live_flags.json")
    print("building features...", flush=True)
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True).copy()
    ufc["date"] = pd.to_datetime(ufc["date"], errors="coerce")

    print("\nwalk-forward predictions (each fight also simulated):", flush=True)
    rows = walk_forward(X, y, ufc)
    print(f"\n{len(rows):,} predictions; simulator available on "
          f"{sum(r['sim_available'] for r in rows):,.0f}")

    confirm = [r for r in rows if r["year"] >= CONFIRM_FROM]
    odds_index = load_odds(ODDS)

    results = [
        walk_forward_quality(confirm, NO_ODDS_COLUMNS, "NO-ODDS (live)"),
        walk_forward_quality(confirm, ["confidence"], "CONFIDENCE only"),
        walk_forward_quality(confirm, ["model_p", "confidence"],
                             "MODEL + CONFIDENCE"),
        walk_forward_quality(confirm, ODDS_COLUMNS, "ODDS (existing)",
                             odds_index=odds_index),
    ]

    print("\n" + "=" * 70)
    print(f"FLAG QUALITY, CONFIRM PERIOD {CONFIRM_FROM} ONWARD")
    print("  0.5 is a coin toss. The live flagger has to beat CONFIDENCE by a")
    print("  real margin or it does not ship.")
    print("=" * 70)
    print(f"  {'flagger':<22}{'scored':>9}{'quality':>10}")
    print("  " + "-" * 41)
    for r in results:
        print(f"  {r['label']:<22}{r['scored']:>9,}{r['quality']:>10.3f}")

    out = ENGINE / "experiments" / "live_flags.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "rows"}
                  for r in results]).to_csv(out, index=False)
    print(f"\nwrote {out}")

    def clean(value):
        # A quality of NaN means "too few scored to say", which JSON cannot
        # carry as a bare NaN and which must not become 0.0 - that would read
        # as a flagger measured to be worthless rather than one not measured.
        if isinstance(value, float) and value != value:
            return None
        return value

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirm_from": CONFIRM_FROM,
        "simulations": SIMULATIONS,
        "results": [{k: clean(v) for k, v in r.items() if k != "rows"}
                    for r in results],
    }
    path = ENGINE / "experiments" / "live_flags.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True,
                               allow_nan=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
