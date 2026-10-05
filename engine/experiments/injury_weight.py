"""Does a KNOWN injury history move results, beyond the model and the market?

The question the injury history was gathered to answer, asked the only way
that can be trusted: for each fight, look only at what was public before it
(injury_history.known_injuries), and ask whether fighters carrying that
history under-perform

  1. the MODEL's probability - which reads records and cannot see injuries;
  2. the BLENDED probability the app prints, three-quarters closing line -
     which has read the news. An effect the market already prices must not
     be added again.

Nothing ships from this file unless the second interval excludes zero.

POPULATIONS, each a different mechanism, none pooled with another:
  withdrew_since_last_fight    pulled out of a booking with an injury or
                               illness since their previous bout: the plain
                               "coming back from an injury" case
  hurt_in_last_fight           injured in their previous bout (doctor
                               stoppage, injury TKO, broken limb)
  surgery_or_long_layoff_2y    surgery, or 6+ months out, public in the last
                               two years
  any_known_injury_1y          anything public in the last twelve months
  known_hurt_going_in          reported injured BEFORE this very fight -
                               the rare direct case, and the one a model
                               could least see

Measured with the shared harness (experiments/residual_harness.py): priced
fights only in both arms, residuals adjusted for win-probability band and
age gap, family-wise intervals resampled by event and by fighter. The first
run of this file let unpriced fights into the blend arm (as the model's own
number) and did not control for age; its -0.031 vs the model is re-measured
here under the stricter test. Vague records (timing not stated, "later
admitted") are excluded, as are records whose fighter could not be matched
to the archive.

    python engine/experiments/injury_weight.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import injury_history as ih

OUT = Path(__file__).with_suffix(".json")


def populations(history, fights):
    """{population: set of (fighter, date)} from what was known before each
    fight, and nothing else.

    `fights` is every (fighter, date) with the date of that fighter's
    previous bout, so "since their last fight" is measurable.
    """
    usable = history[history["matched"].astype(bool)
                     & ~history["vague"].astype(bool)].copy()
    usable["known_by"] = pd.to_datetime(usable["known_by"], errors="coerce")
    usable["anchor_date"] = pd.to_datetime(usable["anchor_date"],
                                           errors="coerce")
    usable = usable.dropna(subset=["known_by"])
    by_fighter = {f: g for f, g in usable.groupby("fighter")}

    out = {name: set() for name in ("withdrew_since_last_fight",
                                    "hurt_in_last_fight",
                                    "surgery_or_long_layoff_2y",
                                    "any_known_injury_1y",
                                    "known_hurt_going_in")}
    for fighter, date, previous in fights:
        rows = by_fighter.get(fighter)
        if rows is None:
            continue
        date = pd.Timestamp(date)
        known = rows[rows["known_by"] < date]          # THE RULE
        if known.empty:
            continue
        key = (fighter, date.date())
        if previous is not None and pd.notna(previous):
            previous = pd.Timestamp(previous)
            since = known[(known["kind"] == "withdrawal")
                          & (known["anchor_date"] > previous)
                          & (known["anchor_date"] < date)]
            if not since.empty:
                out["withdrew_since_last_fight"].add(key)
            last = known[(known["kind"] == "in_fight")
                         & (known["anchor_date"] == previous)]
            if not last.empty:
                out["hurt_in_last_fight"].add(key)
        serious = known[(known["surgery"].astype(bool)
                         | (pd.to_numeric(known["months_out"],
                                          errors="coerce") >= 6))
                        & (known["known_by"] >= date - pd.Timedelta(days=730))]
        if not serious.empty:
            out["surgery_or_long_layoff_2y"].add(key)
        if (known["known_by"] >= date - pd.Timedelta(days=365)).any():
            out["any_known_injury_1y"].add(key)
        going_in = known[(known["kind"] == "fought_hurt")
                         & (known["disclosed"] == "before")
                         & (known["anchor_date"] == date)]
        if not going_in.empty:
            out["known_hurt_going_in"].add(key)
    return out


def main():
    if not ih.HISTORY.exists():
        print(f"  {ih.HISTORY} does not exist - build it first "
              f"(injury_history.py --build). Nothing was estimated.")
        return 1
    from experiments.residual_harness import (header, mark, measure,
                                              predictions, sides)
    from name_resolution import norm_name

    history = ih.load()
    history["fighter"] = history["fighter"].map(norm_name)

    print("building predictions...")
    frame = predictions()
    # Every appearance with the fighter's previous bout date, from the
    # predicted frame's own archive rows.
    import predict_card as engine
    meta = engine.ufc_valid.reset_index(drop=True)
    appearances = sorted((norm_name(n), pd.Timestamp(d))
                         for d, r, b in zip(meta["date"], meta["r_name"],
                                            meta["b_name"])
                         for n in (r, b))
    fights, last = [], {}
    for name, when in appearances:
        fights.append((name, when, last.get(name)))
        last[name] = when
    pops = populations(history, fights)
    for name, members in pops.items():
        print(f"  {name:<32}{len(members):>7} fighter-fights")

    rows = sides(frame)
    header(len(pops))
    results = []
    for name, members in pops.items():
        def side(fight, members=members):
            day = pd.Timestamp(fight.date).date()
            r, b = (fight.red, day) in members, (fight.blue, day) in members
            return "both" if r and b else "red" if r else "blue" if b else None
        results.append(measure(rows, mark(frame, rows, side), name,
                               family=len(pops)))
    beyond = [r for r in results if r.get("verdict") == "REAL beyond the market"]
    print("\n  " + ("BEYOND THE MARKET: " + ", ".join(r["label"] for r in beyond)
                    if beyond else "NOTHING clears the family-wise bar against "
                    "the market."))
    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "history_rows": int(len(history)),
        "populations": {k: len(v) for k, v in pops.items()},
        "results": results,
    }, indent=1))
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
