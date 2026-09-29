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

from check_fight_changes import ARCHIVE, changes_in, resolve_all, weigh_ins_in
from name_resolution import norm_name, short_name

OUT = ENGINE / "data" / "fight_changes.csv"
# Every kind in one file, one row per fighter per event. fight_changes.csv is
# kept exactly as it was because short_notice_weight.py reads it; this is the
# file the pooled measurement reads, where "adverse news" is estimated across
# kinds rather than eight tiny groups one at a time.
HISTORY = ENGINE / "data" / "intel_history.csv"
HISTORY_COLUMNS = ("event_date", "event_name", "fighter", "kind",
                   "fighter_raw", "withdrawn", "days_notice", "weighed_lbs",
                   "over_by_lbs", "article")
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
FROM_YEAR = 2010    # where the history starts. odds.csv begins in 2011, so
                    # 2010 labels feed the arm measured against the model
                    # and not the one measured against the blend - that is
                    # a cost worth paying for a fifth more labels


def card_names(archive, event_name):
    """Everyone who actually fought on this card."""
    rows = archive[archive.event_name == event_name]
    return sorted({*rows.r_name.dropna(), *rows.b_name.dropna()})


def card_bouts(archive, event_name):
    """Every (red, blue) that actually happened on this card."""
    rows = archive[archive.event_name == event_name]
    return [(r, b) for r, b in zip(rows.r_name, rows.b_name)
            if isinstance(r, str) and isinstance(b, str)]


def opponent_on_card(fighter, bouts):
    """Who `fighter` actually fought on this card, or None.

    This is how the KEPT fighter is found - the one whose opponent was
    switched. "Brown replaced Jones" means somebody had spent a camp
    preparing for Jones and fought Brown; that somebody is Brown's opponent
    on the night, and nothing in the sentence names them.
    """
    found = [b if r == fighter else r for r, b in bouts
             if fighter in (r, b)]
    return found[0] if len(found) == 1 else None


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


def harvest(events, archive, *, verbose=True, out=None, history_out=None):
    """Replacements, switched opponents and missed weights, event by event.

    Returns (fight_changes frame, history frame, stats). Both files are
    written after every chunk, so a run that is cancelled or rate-limited
    halfway keeps everything it had already gathered.
    """
    rows, history = [], []
    stats = {"resolved": 0, "found": 0, "unmatched": 0, "quantified": 0,
             "switched": 0, "weigh_ins": 0, "weigh_unmatched": 0}
    for start in range(0, len(events), CHUNK):
        block = events.iloc[start:start + CHUNK]
        try:
            resolved = resolve_all(block)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            # Keep what has been gathered. A rate limiter that gives up
            # halfway through 2014 should not cost the 2010-2013 labels too.
            print(f"    stopped at event {start}: {err}")
            break
        stats["resolved"] += len(resolved)
        for event in block.itertuples():
            title, text = resolved.get(event.event_name, (None, None))
            if not text:
                continue
            names = card_names(archive, event.event_name)
            bouts = card_bouts(archive, event.event_name)
            when = event.date.date()

            for change in changes_in(text):
                stats["found"] += 1
                matched = match_to_card(change["stepped_in"], names)
                if not matched:
                    stats["unmatched"] += 1
                    continue
                if change["days_notice"]:
                    stats["quantified"] += 1
                rows.append({
                    "event_date": when,
                    "event_name": event.event_name,
                    "stepped_in": matched,
                    "stepped_in_raw": change["stepped_in"],
                    "replaced": change["replaced"],
                    "days_notice": change["days_notice"],
                    "article": title,
                })
                history.append({
                    "event_date": when, "event_name": event.event_name,
                    "fighter": matched, "kind": "stepped_in",
                    "fighter_raw": change["stepped_in"],
                    "withdrawn": change["replaced"],
                    "days_notice": change["days_notice"],
                    "weighed_lbs": None, "over_by_lbs": None,
                    "article": title,
                })
                # The other half of the same change: the fighter who stayed
                # and prepared for someone else. Silva's case, historically.
                kept = opponent_on_card(matched, bouts)
                if kept:
                    stats["switched"] += 1
                    history.append({
                        "event_date": when, "event_name": event.event_name,
                        "fighter": kept, "kind": "opponent_switch",
                        "fighter_raw": kept,
                        "withdrawn": change["replaced"],
                        "days_notice": change["days_notice"],
                        "weighed_lbs": None, "over_by_lbs": None,
                        "article": title,
                    })

            for miss in weigh_ins_in(text):
                matched = match_to_card(miss["fighter"], names)
                if not matched:
                    # Usually a bout that was cancelled after the miss - the
                    # fighter is not on the card because the fight did not
                    # happen, and there is nothing to label.
                    stats["weigh_unmatched"] += 1
                    continue
                stats["weigh_ins"] += 1
                history.append({
                    "event_date": when, "event_name": event.event_name,
                    "fighter": matched, "kind": "missed_weight",
                    "fighter_raw": miss["fighter"], "withdrawn": None,
                    "days_notice": None,
                    "weighed_lbs": miss["weighed_lbs"],
                    "over_by_lbs": miss["over_by_lbs"],
                    "article": title,
                })

        frame = pd.DataFrame(rows, columns=list(COLUMNS))
        hist = pd.DataFrame(history, columns=list(HISTORY_COLUMNS))
        if out is not None and not frame.empty:
            frame.to_csv(out, index=False)
        if history_out is not None and not hist.empty:
            hist.to_csv(history_out, index=False)
        if verbose:
            print(f"    {min(start + CHUNK, len(events)):>4}/{len(events)} "
                  f"events, {len(rows):,} replacements, "
                  f"{stats['weigh_ins']:,} missed weights", flush=True)
    return (pd.DataFrame(rows, columns=list(COLUMNS)),
            pd.DataFrame(history, columns=list(HISTORY_COLUMNS)), stats)


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
    frame, history, stats = harvest(events, archive, out=OUT,
                                    history_out=HISTORY)

    print(f"\n  {'events resolved':<34}{stats['resolved']:>8}"
          f"  of {len(events):,}")
    print(f"  {'replacements read':<34}{stats['found']:>8}")
    print(f"  {'joined to a fighter on the card':<34}{len(frame):>8}")
    print(f"  {'dropped, no unique match':<34}{stats['unmatched']:>8}")
    print(f"  {'with a notice period stated':<34}{stats['quantified']:>8}")
    print(f"  {'opponents switched (kept fighter)':<34}{stats['switched']:>8}")
    print(f"  {'missed weight, fighter on card':<34}{stats['weigh_ins']:>8}")
    print(f"  {'missed weight, bout not on card':<34}"
          f"{stats['weigh_unmatched']:>8}")
    if not history.empty:
        history.to_csv(HISTORY, index=False)
        print(f"\n  wrote {HISTORY.relative_to(ENGINE.parent)} "
              f"({len(history):,} rows)")
        print("  " + ", ".join(f"{k}: {n}" for k, n in
                               history.kind.value_counts().items()))

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
