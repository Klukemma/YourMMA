"""Does mentality move results beyond the model and the market?

Every signal is read from fights BEFORE the one predicted (mentality.py)
and measured with the shared harness (experiments/residual_harness.py):
priced fights only, residuals adjusted for win-probability band and age gap,
intervals family-wise across every population here, resampled by event and
by fighter, the wider kept.

A side is marked only against an opponent who could have shown the trait
and did not. "Has never quit" means nothing for a debutant, and counting
debutants as the comparison turned every signal into "long record against
short record" - which is what the first run measured.

  IN THE FIGHT (compared only among fighters equally exposed)
    fragile        knocked down 2+ times before, finished in most of them,
                   against an opponent also dropped 2+ times who was not
    comeback       knocked down 2+ times before and came back to win at
                   least once, against an equally dropped opponent who never
                   did
    has_quit       tapped to strikes, retired or was stopped by the corner
                   before; opponent 5+ UFC fights and never
  HUNGER
    rising         3+ straight wins, rating up, active, no title; opponent
                   3+ UFC fights and not rising
    decorated      2+ UFC title fights won; opponent 5+ UFC fights, none
    fading_pace    output in the last two fights of 5+ minutes down 20%+ on
                   their own earlier average; opponent measured, not fading
  CONFIDENCE (kept apart from hunger)
    dominant       5+ straight wins, 3+ finishes; opponent 5+ fights, not
  THE CONFOUND ITSELF, measured openly
    older_by_3     3+ years older than the opponent (not age-controlled)

    python engine/experiments/mentality.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.residual_harness import header, mark, measure, predictions, sides
from mentality import MentalityIndex

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
WORLD = ENGINE / "data" / "world_bouts.csv.gz"
OUT = Path(__file__).with_suffix(".json")


def _fragile(m):
    return m["dropped"] >= 2 and m["recovery_rate"] < 0.4


def _comeback(m):
    return m["dropped"] >= 2 and m["comebacks"] >= 1


def _fading(m):
    return m["pace_trend"] is not None and m["pace_trend"] < 0.8


def _dominant(m):
    return m["dominant_streak"] >= 5 and m["streak_finishes"] >= 3


# (signal, what the OPPONENT must be for the side to count)
POPULATIONS = {
    "fragile": (_fragile, lambda o: o["dropped"] >= 2 and not _fragile(o)),
    "comeback": (_comeback, lambda o: o["dropped"] >= 2 and not _comeback(o)),
    "has_quit": (lambda m: m["quit_losses"] >= 1,
                 lambda o: o["fights"] >= 5 and o["quit_losses"] == 0),
    "rising": (lambda m: m["rising"],
               lambda o: o["fights"] >= 3 and not o["rising"]),
    "decorated": (lambda m: m["champion_stage"] >= 2,
                  lambda o: o["fights"] >= 5 and o["champion_stage"] == 0),
    "fading_pace": (_fading,
                    lambda o: o["pace_trend"] is not None and not _fading(o)),
    "dominant": (_dominant, lambda o: o["fights"] >= 5 and not _dominant(o)),
}


def marker(index, test, opponent_ok):
    cache = {}

    def at(name, date):
        key = (name, pd.Timestamp(date).date())
        if key not in cache:
            cache[key] = index.at(name, date)
        return cache[key]

    def side(fight):
        r, b = at(fight.red_raw, fight.date), at(fight.blue_raw, fight.date)
        if test(r) and opponent_ok(b):
            return "red"
        if test(b) and opponent_ok(r):
            return "blue"
        return None
    return side


def main():
    archive = pd.read_csv(ARCHIVE, usecols=[
        "date", "r_name", "b_name", "r_kd", "b_kd", "winner", "method",
        "finish_round", "match_time_sec", "title_fight", "r_sig_str_atmpted",
        "b_sig_str_atmpted", "r_td_atmpted", "b_td_atmpted", "r_mu_pre",
        "b_mu_pre"], low_memory=False)
    world = pd.read_csv(WORLD) if WORLD.exists() else None
    index = MentalityIndex(archive, world)

    print("building predictions...")
    frame = predictions()
    rows = sides(frame)
    family = len(POPULATIONS) + 1
    header(family)
    results = []
    for name, (test, opponent_ok) in POPULATIONS.items():
        marked = mark(frame, rows, marker(index, test, opponent_ok))
        results.append(measure(rows, marked, name, family=family))
    older = mark(frame, rows, lambda f: (
        "red" if f.red_age - f.blue_age >= 3 else
        "blue" if f.blue_age - f.red_age >= 3 else None))
    results.append(measure(rows, older, "older_by_3 (age itself)",
                           family=family, control_age=False))

    beyond = [r for r in results if r.get("verdict") == "REAL beyond the market"]
    print()
    if beyond:
        for r in beyond:
            print(f"  BEYOND THE MARKET: {r['label']} {r['blend_adjusted']:+.3f} "
                  f"[{r['blend_low']:+.3f}, {r['blend_high']:+.3f}]")
    else:
        print("  NOTHING clears the family-wise bar against the market.")
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "family": family, "results": results}, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
