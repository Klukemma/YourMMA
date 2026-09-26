"""Every strategy, over fifteen years of real prices instead of 131 bets.

The backtest on logged predictions could not answer anything. 47 out-of-sample
bets carry an interval of about +/-25%, so a 5% edge and a 5% loss look
identical. That was not a modelling limit: it was odds coverage. The file held
275 prices, all of them 2026.

It now holds 6,565 going back to 2010, and 81.3% of the fights since then match
one by name and date. So the model can be walked forward across the whole
period and every prediction it would have made can be priced.

Walk-forward throughout: each year is predicted by a model trained only on the
years before it, and the flagging model is fitted only on bets that had already
settled. Nothing here sees its own answer.

The split matters more than the headline. Choosing anything while looking at
the whole record is how the first backtest produced +5.6% that vanished on a
holdout, so the confirm period is reported separately and it is the number
worth believing.
"""

import json
import os
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.calibrator_mismatch import fit_models, proba
from failure_model import flag_quality, walk_forward_flags
from parlay import (break_even_hit_rate, build_parlays, report as report_parlays,
                    settle_parlays, summarise_parlays)
from roi import load_odds
from strategies import (report as report_strategies, settle_all,
                        strategy_combined, strategy_fade, strategy_model,
                        summarise)

ODDS = Path(os.environ.get("ODDS_CSV", ENGINE / "data" / "odds.csv"))
FIRST_PREDICTED_YEAR = 2011      # 2010 is the first priced year, so it trains
CONFIRM_FROM = 2020              # tune before this, confirm from it

# Every run() call records itself here so main() can write the confirm-period
# numbers out without recomputing them by a second route that could disagree.
RUNS = {}
MIN_TRAIN = 500


def walk_forward_predictions(X, y, ufc):
    """Predict each year with a model trained only on earlier years."""
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
            rows.append({
                "date": fight["date"],
                "event": fight.get("event_name", ""),
                "pick": fight["r_name"] if red_wins else fight["b_name"],
                "opponent": fight["b_name"] if red_wins else fight["r_name"],
                "win_probability": float(prob if red_wins else 1 - prob),
                "confidence": float(abs(2 * prob - 1)),
                "won": bool((actual == 1.0) == red_wins),
                "year": int(year),
            })
        print(f"  {year}: trained on {train.sum():,}, predicted {test.sum():,}")
    return rows


def run(bets, odds_index, label, min_confidence=0.0):
    flagged, flag_rows = walk_forward_flags(bets, odds_index)
    quality = flag_quality(flag_rows)
    RUNS[label] = {"flag_quality": quality, "flags_scored": len(flag_rows),
                   "flagged": len(flagged)}
    runs = {
        "MODEL": strategy_model(bets, min_confidence),
        "FADE (flagged only)": strategy_fade(bets, flagged),
        "COMBINED": strategy_combined(bets, flagged, min_confidence),
    }
    summaries, priced_by_label = [], {}
    for name, selection in runs.items():
        priced, _ = settle_all(selection, odds_index)
        priced_by_label[name] = priced
        summaries.append(summarise(priced, name))
    RUNS[label]["strategies"] = summaries
    print(f"\n{label}")
    print(f"  flags scored {len(flag_rows):,}, quality {quality:.3f} "
          f"(0.5 is a coin toss), {len(flagged):,} flagged")
    report_strategies(summaries)
    return summaries, priced_by_label


def main():
    os.environ.setdefault("PREDICTIONS_LOG", "/tmp/historical_backtest.json")
    print("building features...")
    import predict_card as engine

    X = engine.X_valid_winner.reset_index(drop=True)
    y = np.asarray(engine.y_win, dtype=float)
    ufc = engine.ufc_valid.reset_index(drop=True).copy()
    ufc["date"] = pd.to_datetime(ufc["date"], errors="coerce")

    print("\nwalk-forward predictions:")
    bets = walk_forward_predictions(X, y, ufc)
    print(f"\n{len(bets):,} predictions across "
          f"{FIRST_PREDICTED_YEAR}-{max(b['year'] for b in bets)}")

    odds_index = load_odds(ODDS)
    priced, unpriced = settle_all(strategy_model(bets), odds_index)
    print(f"priced against real odds: {len(priced):,}   "
          f"no price: {len(unpriced):,}")
    if len(priced) < 200:
        print("Too few priced bets to report anything.")
        return

    print("\n" + "=" * 79)
    print(f"EVERYTHING, {FIRST_PREDICTED_YEAR} ONWARD")
    print("=" * 79)
    run(bets, odds_index, "all priced bets")

    # The number worth believing: nothing about the confirm period informed
    # any choice made here.
    tune = [b for b in bets if b["year"] < CONFIRM_FROM]
    confirm = [b for b in bets if b["year"] >= CONFIRM_FROM]
    print("\n" + "=" * 79)
    print(f"TUNE {FIRST_PREDICTED_YEAR}-{CONFIRM_FROM - 1} "
          f"vs CONFIRM {CONFIRM_FROM} ONWARD")
    print("=" * 79)
    run(tune, odds_index, f"tune period ({len(tune):,} predictions)")
    _, confirm_priced = run(confirm, odds_index,
                            f"confirm period ({len(confirm):,} predictions)")

    print("\n" + "=" * 79)
    print("MODEL STRATEGY BY YEAR")
    print("=" * 79)
    by_year = {}
    for bet in confirm_priced["MODEL"] + \
            settle_all(strategy_model(tune), odds_index)[0]:
        by_year.setdefault(pd.to_datetime(bet["date"]).year, []).append(bet)
    print(f"  {'year':<7}{'bets':>7}{'hit':>8}{'profit':>10}{'ROI':>9}")
    print("  " + "-" * 41)
    rows = []
    for year in sorted(by_year):
        s = summarise(by_year[year], str(year))
        rows.append(s)
        print(f"  {year:<7}{s['bets']:>7}{s['hit_rate']:>8.1%}"
              f"{s['profit']:>+10.2f}{s['roi']:>+9.1%}")

    print("\n" + "=" * 79)
    print("PARLAYS, CONFIRM PERIOD")
    print("=" * 79)
    parlay_rows, settled_by_label = [], {}
    for legs in (2, 3, 4):
        settled = settle_parlays(build_parlays(confirm_priced["MODEL"], legs))
        settled_by_label[f"{legs} legs"] = settled
        parlay_rows.append(summarise_parlays(settled, f"{legs} legs"))
    report_parlays(parlay_rows, settled_by_label)

    out = ENGINE / "experiments" / "historical_backtest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    # The app used to carry these three numbers as hand-typed constants, which
    # meant re-running this experiment left the phone showing the old ones with
    # nothing to say so. Written here instead, from the same objects that were
    # just printed, so the two cannot disagree.
    def clean(value):
        # summarise() returns NaN for a strategy that placed no bets, and
        # json.dumps writes a bare NaN that JSON.parse rejects - a blank screen
        # on a phone. null is the honest carrier: not measured, not zero.
        if isinstance(value, float) and value != value:
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    confirm_label = f"confirm period ({len(confirm):,} predictions)"
    summary = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirm_from": CONFIRM_FROM,
        "first_predicted_year": FIRST_PREDICTED_YEAR,
        "flag_quality": RUNS[confirm_label]["flag_quality"],
        "flags_scored": RUNS[confirm_label]["flags_scored"],
        "flagged": RUNS[confirm_label]["flagged"],
        "strategies": RUNS[confirm_label]["strategies"],
        "parlays": parlay_rows,
    }
    path = ENGINE / "experiments" / "strategies.json"
    path.write_text(json.dumps(clean(summary), indent=1, sort_keys=True,
                               allow_nan=False) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
