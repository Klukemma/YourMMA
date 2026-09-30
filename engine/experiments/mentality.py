"""Does mentality move results beyond the model and the market?

Every signal is read from fights BEFORE the one predicted (mentality.py),
marks the side it applies to when the opponent does not share it, and is
measured with the same harness and the same bar as every other factor
(experiments/residual_harness.py): residual against the model, and against
the blended probability the app prints.

  IN THE FIGHT
    fragile        knocked down 2+ times before, finished in most of them
    resilient      knocked down 2+ times before, survived most, came back
                   to win at least once
    has_quit       a loss by tapping to strikes, retirement, corner
                   stoppage or verbal submission
  HUNGER
    rising         climbing: 3+ straight wins, rating up, active, no title
    rising_vs_champ  a rising fighter facing a UFC title winner
    decorated      2+ UFC title fights won (the "further along" side)
    fading_pace    output in the last two fights down 20%+ on their own
                   earlier average
  CONFIDENCE (kept apart from hunger)
    dominant       5+ straight wins, 3+ of them finishes

Eight populations, two arms each: at a 95% bar, one false "real" result
among them is close to the expected count, so a single hit is a lead to
re-test, not a finding.

    python engine/experiments/mentality.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.residual_harness import predictions, report, residuals
from mentality import MentalityIndex

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
WORLD = ENGINE / "data" / "world_bouts.csv.gz"
OUT = Path(__file__).with_suffix(".json")


def _fragile(m):
    return m["dropped"] >= 2 and m["recovery_rate"] < 0.4


def _resilient(m):
    return m["dropped"] >= 2 and m["recovery_rate"] >= 0.6 and m["comebacks"] >= 1


def _dominant(m):
    return m["dominant_streak"] >= 5 and m["streak_finishes"] >= 3


def _fading(m):
    return m["pace_trend"] is not None and m["pace_trend"] < 0.8


SIGNALS = {
    "fragile": _fragile,
    "resilient": _resilient,
    "has_quit": lambda m: m["quit_losses"] >= 1,
    "rising": lambda m: m["rising"],
    "decorated": lambda m: m["champion_stage"] >= 2,
    "fading_pace": _fading,
    "dominant": _dominant,
}


def marker(index, test, versus=None):
    """side(row): the side where `test` holds and the opponent's doesn't.

    With `versus`, the opponent must instead satisfy `versus` - for
    match-ups like a rising fighter against a title winner.
    """
    cache = {}

    def at(name, date):
        key = (name, pd.Timestamp(date).date())
        if key not in cache:
            cache[key] = index.at(name, date)
        return cache[key]

    def side(row):
        r, b = at(row.red_raw, row.date), at(row.blue_raw, row.date)
        other = versus or (lambda m: not test(m))
        if test(r) and other(b) and not (versus is None and test(b)):
            return "red"
        if test(b) and other(r) and not (versus is None and test(r)):
            return "blue"
        return None
    return side


def main():
    archive = pd.read_csv(ARCHIVE, usecols=[
        "date", "r_name", "b_name", "r_kd", "b_kd", "winner", "method",
        "match_time_sec", "title_fight", "r_sig_str_atmpted",
        "b_sig_str_atmpted", "r_td_atmpted", "b_td_atmpted", "r_mu_pre",
        "b_mu_pre"], low_memory=False)
    world = pd.read_csv(WORLD) if WORLD.exists() else None
    index = MentalityIndex(archive, world)

    print("building predictions...")
    frame = predictions()
    markers = {name: marker(index, test) for name, test in SIGNALS.items()}
    markers["rising_vs_champ"] = marker(index, SIGNALS["rising"],
                                        versus=lambda m: m["champion_stage"] >= 1)

    results = []
    print(f"\n  {'population (the marked side)':<52}{'n':>6}{'mean':>10}   "
          f"{'95% interval':<22} verdict")
    for name, side in markers.items():
        for column, against in (("p_model", "vs model"),
                                ("p_blend", "vs blend")):
            got, events = residuals(frame, side, column)
            results.append(report(f"{name}, {against}", got, events))

    print()
    for r in results:
        if "blend" not in r["label"] or r.get("too_few"):
            continue
        if r["low"] > 0 or r["high"] < 0:
            model = next(x for x in results
                         if x["label"] == r["label"].replace("vs blend",
                                                             "vs model"))
            same_way = not model.get("too_few") and (
                (model["low"] > 0 and r["low"] > 0)
                or (model["high"] < 0 and r["high"] < 0))
            print(f"  LEAD: {r['label']} {r['mean']:+.3f} "
                  f"[{r['low']:+.3f}, {r['high']:+.3f}]"
                  + ("  (and against the model, the same way)" if same_way
                     else "  (not against the model - treat with caution)"))
    print("  With 8 populations tested, a single lead may be chance; it is "
          "re-tested\n  before anything ships.")
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "results": results}, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
