"""Does camp power move results beyond the model and the market?

For every archive fight predicted out of sample, each fighter's camp is read
AS OF THAT FIGHT (camp_strength.CampIndex: dated affiliations, the camp's
UFC results in the three years before, title wins, a champion coach). The
side with the stronger camp is marked, and its residual measured against
the model and against the blend - the same harness, the same bar, as every
other factor (experiments/residual_harness.py).

  camp_title_edge      fighter's camp won 2+ UFC title fights in the window,
                       the opponent's none
  camp_rate_edge       fighter's camp win rate 10+ points higher, both camps
                       with 15+ recent UFC bouts
  champion_coach_edge  fighter's listed trainer had won a UFC title by then,
                       the opponent's had not

    python engine/experiments/camp_power.py
"""

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from camp_strength import CampIndex
from experiments.residual_harness import predictions, report, residuals

PROFILES = ENGINE / "data" / "fighter_profiles.jsonl.gz"
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
OUT = Path(__file__).with_suffix(".json")


def edges(camp, frame):
    """{population: function(row) -> side or None}, camps read as of row."""
    cache = {}

    def at(name, date):
        key = (name, pd.Timestamp(date).date())
        if key not in cache:
            cache[key] = camp.at(name, date)
        return cache[key]

    def title_edge(row):
        r, b = at(row.red_raw, row.date), at(row.blue_raw, row.date)
        if r["titles"] >= 2 and b["titles"] == 0 and b["gyms"]:
            return "red"
        if b["titles"] >= 2 and r["titles"] == 0 and r["gyms"]:
            return "blue"
        return None

    def rate_edge(row):
        r, b = at(row.red_raw, row.date), at(row.blue_raw, row.date)
        if min(r["bouts"], b["bouts"]) < 15:
            return None
        if r["rate"] - b["rate"] >= 0.10:
            return "red"
        if b["rate"] - r["rate"] >= 0.10:
            return "blue"
        return None

    def coach_edge(row):
        r, b = at(row.red_raw, row.date), at(row.blue_raw, row.date)
        if r["champion_coach"] and not b["champion_coach"]:
            return "red"
        if b["champion_coach"] and not r["champion_coach"]:
            return "blue"
        return None

    return {"camp_title_edge": title_edge, "camp_rate_edge": rate_edge,
            "champion_coach_edge": coach_edge}


def main():
    if not PROFILES.exists():
        print(f"  {PROFILES.name} does not exist - run harvest-world first.")
        return 1
    with gzip.open(PROFILES, "rt", encoding="utf-8") as handle:
        profiles = [json.loads(line) for line in handle if line.strip()]
    archive = pd.read_csv(ARCHIVE, usecols=["date", "r_name", "b_name",
                                            "winner", "title_fight"],
                          low_memory=False)
    camp = CampIndex(profiles, archive)
    dated = sum(1 for p in profiles if any(a["dated"]
                                           for a in p["affiliations"]))
    print(f"  {len(profiles):,} fighter profiles, {dated:,} with a dated "
          f"team history, {len(camp.members):,} gyms")

    print("building predictions...")
    frame = predictions()
    results = []
    print(f"\n  {'population (the side with the stronger camp)':<52}{'n':>6}"
          f"{'mean':>10}   {'95% interval':<22} verdict")
    for name, marked in edges(camp, frame).items():
        for column, against in (("p_model", "vs model"),
                                ("p_blend", "vs blend")):
            got, events = residuals(frame, marked, column)
            results.append(report(f"{name}, {against}", got, events))

    shipped = [r for r in results if "blend" in r["label"]
               and not r.get("too_few") and r["low"] > 0]
    print()
    print("  EARNED on top of the blend: " + ", ".join(r["label"] for r in shipped)
          if shipped else "  NOTHING SHIPS: no camp edge beats the blended "
          "probability with an interval that excludes zero.")
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "profiles": len(profiles), "dated_profiles": dated,
        "results": results}, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
