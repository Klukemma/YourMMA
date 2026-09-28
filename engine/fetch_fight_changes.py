"""Harvest late replacements from Wikipedia into a file that can be fitted.

WHAT THIS LABELS, AND WHAT IT DELIBERATELY DOES NOT. check_fight_changes
established that Wikipedia's event pages carry replacements, that 87% of our
events resolve to the right article, and that essentially none of those
sentences state a notice period. So the label here is

    this fighter stepped in for somebody else

and NOT "this fighter took the fight on short notice", which is the thing
predict_card's unfitted -0.04 is named after. They are not the same. One of
the sentences this harvests reads "Magomedov pulled out of the fight in early
March ... and was replaced by Rustam Khabilov", for a fight on 8 May: a
nine-week camp. `days_notice` is written when the page states one and left
empty when it does not, and a reader who fills the gap with a guess has
destroyed the only reason to gather this.

THE STEPPED-IN FIGHTER IS RESOLVED AGAINST THE CARD, the fighter they
replaced is not. Wikipedia names a fighter in full once and by surname after,
so most extracted names are partial; a surname is very nearly unique among
the twenty-odd people on one card, which is what makes them usable. The
replaced fighter usually never fought, so they are not on the card to match
against, and their name is recorded as written.

    python engine/fetch_fight_changes.py              (needs network)
    python engine/fetch_fight_changes.py --from-year 2011 --limit 50

Writes engine/data/fight_changes.csv. Run it from the workflow; this
container's egress proxy refuses Wikipedia.
"""

import argparse
import sys
import urllib.error
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import ARCHIVE, changes_in, resolve_all
from name_resolution import norm_name, short_name

OUT = ENGINE / "data" / "fight_changes.csv"
COLUMNS = ("event_date", "event_name", "stepped_in", "stepped_in_raw",
           "replaced", "days_notice", "article")
CHUNK = 40          # events resolved per round trip group

# WRITTEN AFTER EVERY CHUNK, not at the end. The first full run was cancelled
# part way and everything it had gathered went with it. A running job's logs
# cannot be read until it finishes, so there was no way to tell whether it was
# progressing or stuck - and, as it turned out, it was progressing, and the
# cancellation was a misjudgement made in the dark. Partial output that
# survives is worth more than tidy output that does not, and a chunk count
# printed as it goes is worth more than both.
FROM_YEAR = 2011    # odds.csv starts here, and an unpriced label cannot be
                    # used by the arm that matters - the one that asks
                    # whether the effect survives the market


def card_names(archive, event_name):
    """Everyone who actually fought on this card."""
    rows = archive[archive.event_name == event_name]
    return sorted({*rows.r_name.dropna(), *rows.b_name.dropna()})


def match_to_card(name, names):
    """The fighter on this card that `name` refers to, or None.

    Three ways, in order of how much they can be trusted: the whole name, the
    surname when exactly one fighter on the card has it, and otherwise
    nothing. AMBIGUITY RESOLVES TO NOTHING rather than to a best guess - a
    label attached to the wrong fighter is worse than a missing one, because
    it is indistinguishable from evidence.
    """
    wanted = norm_name(name)
    for full in names:
        if norm_name(full) == wanted:
            return full
    surname = norm_name(short_name(name))
    if not surname:
        return None
    hits = [full for full in names if norm_name(short_name(full)) == surname]
    return hits[0] if len(hits) == 1 else None


def harvest(events, archive, *, verbose=True, out=None):
    rows, stats = [], {"resolved": 0, "found": 0, "unmatched": 0,
                       "quantified": 0}
    for start in range(0, len(events), CHUNK):
        block = events.iloc[start:start + CHUNK]
        try:
            resolved = resolve_all(block)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            # Keep what has been gathered. A rate limiter that gives up
            # halfway through 2014 should not cost the 2011-2013 labels too.
            print(f"    stopped at event {start}: {err}")
            break
        stats["resolved"] += len(resolved)
        for event in block.itertuples():
            title, text = resolved.get(event.event_name, (None, None))
            if not text:
                continue
            names = card_names(archive, event.event_name)
            for change in changes_in(text):
                stats["found"] += 1
                matched = match_to_card(change["stepped_in"], names)
                if not matched:
                    stats["unmatched"] += 1
                    continue
                if change["days_notice"]:
                    stats["quantified"] += 1
                rows.append({
                    "event_date": event.date.date(),
                    "event_name": event.event_name,
                    "stepped_in": matched,
                    "stepped_in_raw": change["stepped_in"],
                    "replaced": change["replaced"],
                    "days_notice": change["days_notice"],
                    "article": title,
                })
        frame = pd.DataFrame(rows, columns=list(COLUMNS))
        if out is not None and not frame.empty:
            frame.to_csv(out, index=False)
        if verbose:
            print(f"    {min(start + CHUNK, len(events)):>4}/{len(events)} "
                  f"events, {len(rows):,} labels so far", flush=True)
    return pd.DataFrame(rows, columns=list(COLUMNS)), stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N events, for a smoke test")
    parser.add_argument("--from-year", type=int, default=FROM_YEAR,
                        help="skip events before this year")
    args = parser.parse_args(argv)

    archive = pd.read_csv(ARCHIVE, usecols=["event_name", "date", "r_name",
                                            "b_name"], low_memory=False)
    archive["date"] = pd.to_datetime(archive["date"], errors="coerce")
    events = (archive.dropna(subset=["date"])
              .drop_duplicates("event_name")
              .sort_values("date")[["event_name", "date"]]
              .reset_index(drop=True))
    events = events[events.date.dt.year >= args.from_year].reset_index(drop=True)
    if args.limit:
        events = events.head(args.limit)

    print("=" * 74)
    print(f"HARVESTING LATE REPLACEMENTS FROM {len(events):,} EVENTS")
    print("=" * 74)
    frame, stats = harvest(events, archive, out=OUT)

    print(f"\n  {'events resolved':<34}{stats['resolved']:>8}"
          f"  of {len(events):,}")
    print(f"  {'replacements read':<34}{stats['found']:>8}")
    print(f"  {'joined to a fighter on the card':<34}{len(frame):>8}")
    print(f"  {'dropped, no unique match':<34}{stats['unmatched']:>8}")
    print(f"  {'with a notice period stated':<34}{stats['quantified']:>8}")

    if frame.empty:
        print("\n  Nothing to write.")
        return 1
    frame.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(ENGINE.parent)} "
          f"({len(frame):,} rows)")
    print("  These are REPLACEMENTS, not short notice. days_notice is blank")
    print("  wherever the page did not say, and must stay blank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
