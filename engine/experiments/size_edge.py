"""Is the man who is really bigger underrated? True Fight Weight, by proxy.

Everyone at lightweight weighs in at 155, and nobody fights at 155: the
measured fight-night weights regain 7-13% for the middle half, -1% to +19%
at the ends (fight_night_weights.csv). Those measurements are too few - six
events - to fit an estimator that beats the division median
(true_fight_weight.py), and they are taken on fight day and published after
the bout, so they could never be a pre-fight input anyway.

What IS known before every fight is where a fighter has fought before. A
man who has competed at a heavier limit carries a bigger frame down to this
one, and the regain research (CSAC's second-day checks; Peacock et al. on
616 UFC athletes) puts the biggest regains on the fighters cutting the
most. The model has no feature for it: it knows height and reach, not
weight history. So the sides below are marked from the UFC archive, strictly
from bouts before the fight, and measured with the shared harness against
the model and the market:

  fought_heavier_before    has fought at a heavier limit; the opponent has
                           not (natural size carried down)
  first_fight_down         their last bout was at a heavier limit
  first_fight_up           their last bout was at a lighter limit (the
                           smaller man, moving up - expected negative)
  taller_and_longer        2+ inches taller AND 2+ inches longer reach
                           (already model features: checks whether the model
                           prices frame size fully)
  missed_weight            missed weight at the weigh-in the day before
                           (intel_history.csv, read from event pages) - the
                           one weight fact public before the bell. Heavier
                           on the scale, but a failed cut; the market has it
  missed_by_2lb            the same, by two pounds or more

    python engine/experiments/size_edge.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments.residual_harness import header, mark, measure, predictions, sides
from name_resolution import canonical_winners, norm_name
from true_fight_weight import limit_of

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
INTEL = ENGINE / "data" / "intel_history.csv"
OUT = Path(__file__).with_suffix(".json")
INCHES_2 = 5.08


class History:
    """Each fighter's UFC limits by date, read strictly before a date."""

    def __init__(self, archive):
        a = canonical_winners(archive).copy()
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a["limit"] = a["division"].map(limit_of)
        long = pd.concat([
            pd.DataFrame({"fighter": a[f"{s}_name"].map(norm_name),
                          "date": a["date"], "limit": a["limit"],
                          "height": a[f"{s}_height"], "reach": a[f"{s}_reach"]})
            for s in ("r", "b")]).dropna(subset=["date"]).sort_values("date")
        self.by = {k: g for k, g in long.groupby("fighter")}

    def before(self, fighter, date):
        g = self.by.get(norm_name(fighter))
        if g is None:
            return None
        g = g[g["date"] < pd.Timestamp(date)]
        return g if len(g) else None

    def size(self, fighter, date):
        """(heaviest earlier limit, last earlier limit, height, reach)."""
        g = self.by.get(norm_name(fighter))
        height = reach = np.nan
        if g is not None:
            if g["height"].notna().any():
                height = float(g["height"].dropna().iloc[-1])
            if g["reach"].notna().any():
                reach = float(g["reach"].dropna().iloc[-1])
        prior = self.before(fighter, date)
        limits = prior["limit"].dropna() if prior is not None else pd.Series(dtype=float)
        return ((float(limits.max()) if len(limits) else np.nan),
                (float(limits.iloc[-1]) if len(limits) else np.nan),
                height, reach)


def misses(path=None):
    """{(date, fighter key): pounds over or NaN} for every missed weight."""
    intel = pd.read_csv(path or INTEL)
    intel = intel[intel["kind"] == "missed_weight"]
    return {(pd.Timestamp(d).date(), norm_name(f)): float(o) if pd.notna(o) else np.nan
            for d, f, o in zip(intel["event_date"], intel["fighter"],
                               intel["over_by_lbs"])}


def populations(history, missed=None):
    cache = {}
    missed = missed or {}

    def at(name, date):
        key = (norm_name(name), pd.Timestamp(date).date())
        if key not in cache:
            cache[key] = history.size(name, date)
        return cache[key]

    def pair(row):
        limit = limit_of(row.division)
        if not np.isfinite(limit):              # catch weight: no limit
            return None
        return limit, at(row.red_raw, row.date), at(row.blue_raw, row.date)

    def heavier_before(row):
        got = pair(row)
        if not got:
            return None
        limit, r, b = got
        r_up = np.isfinite(r[0]) and r[0] > limit
        b_up = np.isfinite(b[0]) and b[0] > limit
        # Both must have UFC history: "never fought heavier" is only
        # knowable for someone whose earlier bouts are on record.
        if not (np.isfinite(r[0]) and np.isfinite(b[0])) or r_up == b_up:
            return None
        return "red" if r_up else "blue"

    def moved(direction):
        def marked(row):
            got = pair(row)
            if not got:
                return None
            limit, r, b = got
            def did(s):
                if not np.isfinite(s[1]):
                    return False
                return s[1] > limit if direction == "down" else s[1] < limit
            r_did, b_did = did(r), did(b)
            if r_did == b_did:
                return None
            return "red" if r_did else "blue"
        return marked

    def frame_edge(row):
        got = pair(row)
        if not got:
            return None
        _, r, b = got
        dh, dr = r[2] - b[2], r[3] - b[3]
        if not (np.isfinite(dh) and np.isfinite(dr)):
            return None
        if dh >= INCHES_2 and dr >= INCHES_2:
            return "red"
        if dh <= -INCHES_2 and dr <= -INCHES_2:
            return "blue"
        return None

    def missed_by(at_least):
        def marked(row):
            day = pd.Timestamp(row.date).date()
            r = missed.get((day, norm_name(row.red_raw)))
            b = missed.get((day, norm_name(row.blue_raw)))
            def hit(over):
                if over is None:
                    return False
                return at_least == 0 or (np.isfinite(over) and over >= at_least)
            r_hit, b_hit = hit(r), hit(b)
            if r_hit == b_hit:
                return None
            return "red" if r_hit else "blue"
        return marked

    return {"fought_heavier_before": heavier_before,
            "first_fight_down": moved("down"),
            "first_fight_up": moved("up"),
            "taller_and_longer": frame_edge,
            "missed_weight": missed_by(0),
            "missed_by_2lb": missed_by(2)}


def main():
    archive = pd.read_csv(ARCHIVE, usecols=[
        "date", "division", "r_name", "b_name", "winner", "r_height",
        "b_height", "r_reach", "b_reach"], low_memory=False)
    history = History(archive)
    print("building predictions...")
    frame = predictions()
    rows = sides(frame)
    missed = misses()
    print(f"  {len(missed)} missed weights on record")
    pops = populations(history, missed)
    header(len(pops))
    results = [measure(rows, mark(frame, rows, side), name, family=len(pops))
               for name, side in pops.items()]
    beyond = [r for r in results if r.get("verdict") == "REAL beyond the market"]
    print("\n  " + ("BEYOND THE MARKET: " + ", ".join(r["label"] for r in beyond)
                    if beyond else "NOTHING clears the family-wise bar against "
                    "the market."))
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "results": results}, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
