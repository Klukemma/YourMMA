"""Soft facts about a fight - injuries, short notice, a new head coach.

WHAT THIS IS FOR. The model reads a fighter's record and nothing else. It
does not know that one of them tore a meniscus in camp, took the fight on
nine days' notice, or moved to a gym with a world-class wrestling coach. A
researcher - human or AI - can find those things. This is where they go.

WHAT THIS IS NOT FOR, YET. Moving the probability. predict_card already has
a context hook with `short_notice`, `hard_cut` and the rest, and the numbers
behind it are hand-typed guesses that were never measured:

    'short_notice': -0.04,       # ~4% penalty
    'weight_cut_hard': -0.03,    # ~3% penalty

Nobody fitted those. Feeding better-gathered facts into unmeasured constants
does not make a prediction better; it makes a guess more confident. So an
observation gathered here is SHOWN by default and moves nothing, and the
`applies` flag stays off until somebody fits a weight against outcomes.

THE MARKET HAS MOST OF THIS ALREADY. By fight night the closing line reflects
the injury report, the late replacement and the camp gossip, because people
with money at stake have read the same news. The card now blends 75% market,
so that information is already in the prediction second-hand. An intel layer
earns its place in three narrow places: fights the market prices thinly,
information that arrives after the price was captured, and telling the reader
WHY a number is what it is. That last one is the honest reason to build it
first.

POINT IN TIME OR IT IS WORTHLESS. Every observation carries `gathered` - when
it was known - and anything gathered after a fight started is not evidence, it
is hindsight. Two of this project's worst bugs were a model reading its own
answer; a note written the morning after a knockout saying "he looked slow in
camp" would be a third. `usable_at` enforces it.

SOURCED OR IT DID NOT HAPPEN. Every observation needs a URL. A claim with no
source cannot be checked, cannot be re-gathered, and cannot be told apart from
something a language model imagined - which is the specific failure mode of
having an AI fill this file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
INTEL_PATH = ENGINE / "data" / "fight_intel.json"

SCHEMA = 1

# What a researcher is allowed to record. A closed list, because "any string"
# becomes forty spellings of the same idea and nothing can ever be fitted
# against outcomes.
#
# The value is the SUFFIX of the context key this kind would drive if a weight
# were ever measured for it. predict_card's keys are corner-prefixed -
# red_short_notice, blue_short_notice - because an adjustment has to know
# which fighter it is about, while an observation knows only the fighter's
# name. context_from() does that translation and is the only place that
# should. None means there is no hook at all and the kind is a note.
KINDS = {
    "short_notice": "short_notice",
    "injury": None,
    "hard_weight_cut": "hard_cut",
    "missed_weight": None,
    "new_camp": None,
    "layoff_return": None,
    "personal": None,
    # Who this fighter was ORIGINALLY matched against, before the switch.
    # The interesting half of a late replacement is not the notice, it is
    # that a camp spent eight weeks preparing for a wrestler and walks in
    # against a counter-striker. That is not a judgement a researcher has to
    # make: predict_card already classifies every fighter into one of five
    # archetypes - pressure boxer, counter-striker, wrestler, submission
    # artist, point fighter - so the engine can measure the switch itself
    # from two names. See `replaced_opponent` on the record.
    "opponent_switch": None,
    # The fighter who took somebody else's place. Not "short_notice": the
    # Wikipedia harvest found a notice period stated on 2 of 298, and one of
    # them had a nine-week camp. A replacement is a replacement.
    "stepped_in": None,
    "other": None,
}

# The fields a kind needs beyond the common ones. An opponent_switch with no
# name of the original opponent is just a short-notice note.
EXTRA_FIELDS = {"opponent_switch": ("replaced_opponent",)}

# How sure the researcher is. Kept coarse on purpose: a model asked for a
# number between 0 and 1 will produce 0.85 for everything.
CONFIDENCE = ("reported", "confirmed", "rumoured")


class IntelError(ValueError):
    """A record that cannot be trusted, rejected at the door."""


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def observation(*, fighter, event_date, kind, confidence, note, source,
                gathered=None, applies=False, **extra):
    """One checked fact about one fighter for one fight.

    `applies` is False unless a weight has been measured for this kind. It is
    accepted as an argument so a future, fitted version has somewhere to say
    so - not so that today's caller can switch it on.

    Some kinds need a field of their own; `opponent_switch` needs the name of
    the fighter who was replaced, because the whole value of that observation
    is the comparison between two styles and one name cannot make it.
    """
    if not fighter or not str(fighter).strip():
        raise IntelError("an observation needs a fighter")
    if kind not in KINDS:
        raise IntelError(f"unknown kind {kind!r}; allowed: {sorted(KINDS)}")
    if confidence not in CONFIDENCE:
        raise IntelError(f"confidence must be one of {CONFIDENCE}")
    source = (source or "").strip()
    if not source.startswith(("http://", "https://")):
        raise IntelError(
            "every observation needs a source URL. A claim nobody can check "
            "cannot be told apart from one a language model invented.")
    if not note or len(str(note).strip()) < 8:
        raise IntelError("an observation needs a note a person can read")
    if applies and KINDS[kind] is None:
        raise IntelError(
            f"{kind!r} has no context hook to drive, so it cannot apply to a "
            f"probability; record it as a note")
    required = EXTRA_FIELDS.get(kind, ())
    missing = [f for f in required if not str(extra.get(f) or "").strip()]
    if missing:
        raise IntelError(
            f"{kind!r} needs {', '.join(missing)}: without it this records "
            f"only that something changed, which the engine cannot measure")
    unexpected = set(extra) - set(required)
    if unexpected:
        raise IntelError(f"{kind!r} takes no {sorted(unexpected)}")

    return {
        "fighter": str(fighter).strip(),
        "event_date": str(event_date),
        "kind": kind,
        "confidence": confidence,
        "note": str(note).strip(),
        "source": source,
        "gathered": gathered or _now(),
        "applies": bool(applies),
        **{f: str(extra[f]).strip() for f in required},
    }


def usable_at(record, when):
    """Was this known before `when`?

    Anything gathered after the first bell is hindsight wearing the clothes of
    a prediction. This is the whole reason `gathered` exists.
    """
    gathered = str(record.get("gathered") or "")
    return bool(gathered) and gathered[:10] <= str(when)[:10]


def load(path=INTEL_PATH):
    """Every observation on file, or an empty store."""
    path = Path(path)
    if not path.exists():
        return {"schema": SCHEMA, "observations": []}
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise IntelError(f"fight_intel.json is schema {data.get('schema')}, "
                         f"this code writes {SCHEMA}")
    return data


def add(store, records):
    """Append, refusing exact duplicates. Returns how many were new.

    Append-only: an observation is a record of what was believed at a time,
    and rewriting it later destroys the only property that makes it usable.
    A correction is a NEW observation with a later `gathered`.
    """
    seen = {(o["fighter"], o["event_date"], o["kind"], o["source"])
            for o in store["observations"]}
    added = 0
    for record in records:
        key = (record["fighter"], record["event_date"], record["kind"],
               record["source"])
        if key in seen:
            continue
        store["observations"].append(record)
        seen.add(key)
        added += 1
    return added


def save(store, path=INTEL_PATH):
    Path(path).write_text(json.dumps(store, indent=1, sort_keys=True))
    return path


def for_fight(store, red, blue, event_date, *, known_by=None):
    """Observations about either corner, gathered before the fight.

    `known_by` defaults to the event date, which is the honest cutoff: the
    prediction is made before the fight, so only what was gathered before it
    may be read.
    """
    cutoff = known_by or event_date
    names = {str(red).strip().lower(), str(blue).strip().lower()}
    return [o for o in store["observations"]
            if o["fighter"].lower() in names
            and str(o["event_date"])[:10] == str(event_date)[:10]
            and usable_at(o, cutoff)]


def context_from(records, red, blue):
    """The context dict these records justify - today, always empty.

    Every kind ships with applies=False because no weight has been fitted for
    any of them. The function exists so that the day a weight IS fitted, the
    path from a gathered fact to a moved probability is one place with a test
    on it, rather than something improvised under time pressure on a fight
    night.

    Needs both corners because predict_card's keys are red_/blue_ prefixed. An
    observation about a fighter who is in neither corner is dropped rather
    than guessed at - it is the wrong fight.
    """
    corners = {str(red).strip().lower(): "red",
               str(blue).strip().lower(): "blue"}
    context = {}
    for record in records:
        if not record.get("applies"):
            continue
        suffix = KINDS.get(record["kind"])
        corner = corners.get(record["fighter"].strip().lower())
        if suffix and corner:
            context[f"{corner}_{suffix}"] = True
    return context


def _report(store, event_date=None):
    """What is on file, newest first."""
    rows = store["observations"]
    if event_date:
        rows = [o for o in rows if str(o["event_date"])[:10] == event_date[:10]]
    if not rows:
        print("  nothing on file" + (f" for {event_date}" if event_date else ""))
        return
    for o in sorted(rows, key=lambda o: (o["event_date"], o["gathered"])):
        flag = " [APPLIES]" if o["applies"] else ""
        switched = (f"  (was matched with {o['replaced_opponent']})"
                    if o.get("replaced_opponent") else "")
        print(f"  {o['event_date']}  {o['fighter']:<24} {o['kind']:<16} "
              f"{o['confidence']:<10} gathered {o['gathered'][:10]}{flag}"
              f"{switched}")
        print(f"      {o['note']}")
        print(f"      {o['source']}")


def main(argv=None):
    """Record or list soft intel.

        python3 engine/fight_intel.py --list
        python3 engine/fight_intel.py --list --event 2026-09-26
        python3 engine/fight_intel.py --add \\
            --fighter "Brady Hiestand" --event 2026-09-26 --kind injury \\
            --confidence reported --source https://example.com/story \\
            --note "knee trouble reported in the final week of camp"

    Nothing recorded here moves a prediction. Every kind ships unfitted, and
    --add will not let you claim otherwise.
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--add", action="store_true")
    parser.add_argument("--event")
    parser.add_argument("--fighter")
    parser.add_argument("--kind", choices=sorted(KINDS))
    parser.add_argument("--confidence", choices=CONFIDENCE, default="reported")
    parser.add_argument("--note")
    parser.add_argument("--source")
    parser.add_argument("--replaced-opponent",
                        help="opponent_switch only: who this fighter was "
                             "originally matched against")
    args = parser.parse_args(argv)

    store = load()
    if args.add:
        missing = [n for n in ("fighter", "event", "kind", "note", "source")
                   if not getattr(args, n)]
        if missing:
            parser.error("--add needs " + ", ".join("--" + n for n in missing))
        extra = ({"replaced_opponent": args.replaced_opponent}
                 if args.kind in EXTRA_FIELDS else {})
        try:
            record = observation(fighter=args.fighter, event_date=args.event,
                                 kind=args.kind, confidence=args.confidence,
                                 note=args.note, source=args.source, **extra)
        except IntelError as err:
            # A refusal is the expected outcome of a careless entry, not a
            # crash. Printing a stack trace at somebody trying to record a
            # fact teaches them to stop recording facts.
            parser.error(str(err))
        added = add(store, [record])
        save(store)
        print(f"  {'recorded' if added else 'already on file'}: "
              f"{record['fighter']} / {record['kind']}")
        print("  This does not move any probability. No weight is fitted for "
              "any kind yet.")
    _report(store, args.event if args.list else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
