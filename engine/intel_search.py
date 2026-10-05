"""The scout: turn a card into research questions, and findings into records.

WHAT WAS MISSING. fight_intel.py is a filing cabinet - it validates and stores
an observation, and it has a CLI a person types into by hand. Nothing ever
went and LOOKED for anything. This is the half that does, and it is built for
a researcher that is a language model with a web search, because that is what
is actually available.

THE SHAPE. An agent cannot be trusted to remember what to ask, so the asking
is generated here and is the same every time:

    brief(card)     the exact queries to run, per fighter, and what counts
    ingest(path)    findings back in, validated, or rejected with a reason

Everything hard about this is in the second half. A model handed a search box
and asked for "any news about this fighter" will produce fluent, plausible,
undated, unsourced paragraphs, and a paragraph is indistinguishable from a
fact once it is in a JSON file. So the ingest is adversarial toward its own
researcher:

    a source URL or it does not go in            (fight_intel enforces)
    a kind from a closed list, never free text   (fight_intel enforces)
    gathered BEFORE the fight, or it is rejected - this is the one that
    matters most, because a search run today about a fight last week returns
    the result of that fight dressed as pre-fight news
    nothing it records moves a probability, because no weight is fitted

THE LAST POINT IS NOT A LIMITATION TO BE FIXED LATER, it is the finding.
experiments/short_notice_weight.py measured the one soft factor we could
label at scale - 298 late replacements harvested from Wikipedia - and got
-3.5% against the model and -4.4% against the blend, both intervals spanning
zero, needing about 1,800 labels to resolve. So intel is SHOWN, next to the
number, and the number is unmoved. That is worth having on its own: it tells
a reader why a fight might not go the way the model thinks, which is a
different and more honest service than quietly nudging a percentage by an
amount nobody measured.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import fight_intel as intel

# What to ask about each fighter. Generated rather than improvised so two runs
# a month apart ask the same questions, and so a kind that gets no coverage
# shows up as a gap rather than as something the researcher forgot.
#
# The `kind` on each is the kind a finding would be filed under, so a
# researcher never has to invent one.
QUESTIONS = (
    ("injury", '"{fighter}" injury OR injured camp {year}'),
    ("short_notice", '"{fighter}" short notice replacement {year}'),
    ("new_camp", '"{fighter}" new gym OR coach OR moved camps {year}'),
    ("hard_weight_cut", '"{fighter}" weight cut struggle {year}'),
    ("missed_weight", '"{fighter}" missed weight {year}'),
    ("layoff_return", '"{fighter}" return from layoff OR time off {year}'),
    ("personal", '"{fighter}" personal issues OR outside the cage {year}'),
    ("motivation", '"{fighter}" retirement OR "last fight" OR contract OR '
                   'motivation {year}'),
)

# Sources whose reporting is worth recording. Not a whitelist the ingest
# enforces - a good story can appear anywhere - but the list a researcher
# should reach for first, and the ones whose absence should make a claim
# suspicious.
PREFERRED = ("mmajunkie.usatoday.com", "mmafighting.com", "espn.com",
             "sherdog.com", "bloodyelbow.com", "mmamania.com",
             "bjpenn.com", "tapology.com", "ufc.com")


class IngestError(ValueError):
    """A finding that cannot be trusted, rejected with the reason."""


def brief(fights, event_date):
    """The research plan for one card: every fighter, every question.

    Returns a list of dicts rather than printed text so the same structure
    drives the printed briefing and any future automation.
    """
    year = str(event_date)[:4]
    plan = []
    seen = set()
    for red, blue in fights:
        for fighter in (red, blue):
            name = str(fighter).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            plan.append({
                "fighter": name,
                "event_date": str(event_date)[:10],
                "queries": [{"kind": kind,
                             "query": template.format(fighter=name, year=year)}
                            for kind, template in QUESTIONS],
            })
    return plan


def _before_the_bell(record, event_date):
    """Reject anything gathered after the fight started.

    THE SINGLE MOST DANGEROUS FAILURE in this whole idea. A search run today
    for "Smith injury" about a fight held last Saturday returns coverage of
    how Smith lost, written afterwards, phrased as though it were known
    beforehand. Recorded without this check it becomes a feature that predicts
    the past perfectly and the future not at all - which is exactly the shape
    of the two worst bugs this project has already had.

    WHAT THIS COSTS, said plainly: history cannot be backfilled through here.
    `gathered` is when the RESEARCHER found it, not when the story ran, so a
    piece published the day before a 2019 fight is rejected today along with
    everything else. Fixing that needs a `published` field, and a publication
    date is exactly the kind of detail a language model will supply
    confidently and wrongly - so it would need verifying against the page
    rather than trusting the summary, and direct page fetches are blocked
    here. Until that is solved this gathers for UPCOMING cards only, and the
    label used to fit weights comes from fetch_fight_changes.py, which reads
    dated event articles instead of trusting a researcher.
    """
    gathered = str(record.get("gathered") or "")[:10]
    if not gathered:
        raise IngestError(f"{record.get('fighter')}: no gathered timestamp")
    if gathered > str(event_date)[:10]:
        raise IngestError(
            f"{record.get('fighter')}: gathered {gathered}, after the "
            f"{str(event_date)[:10]} event. This is hindsight, not a "
            f"prediction, and it is the one thing that must never be stored.")


def ingest(findings, *, store=None, path=intel.INTEL_PATH, strict=True):
    """Validated findings onto the pile. Returns (added, rejected).

    Every record goes through fight_intel.observation(), which demands a
    source URL, a kind from the closed list and a readable note, and refuses
    applies=True for any kind with no fitted weight. On top of that this
    refuses anything gathered after its own event.
    """
    store = store if store is not None else intel.load(path)
    accepted, rejected = [], []
    for raw in findings:
        try:
            event_date = raw.get("event_date")
            if not event_date:
                raise IngestError(f"{raw.get('fighter')}: no event_date")
            extra = {f: raw[f] for f in
                     intel.EXTRA_FIELDS.get(raw.get("kind"), ())
                     if f in raw}
            record = intel.observation(
                fighter=raw.get("fighter"), event_date=event_date,
                kind=raw.get("kind"), confidence=raw.get("confidence",
                                                         "reported"),
                note=raw.get("note"), source=raw.get("source"),
                gathered=raw.get("gathered"), **extra)
            _before_the_bell(record, event_date)
        except (intel.IntelError, IngestError) as err:
            rejected.append(str(err))
            if strict:
                continue
            continue
        accepted.append(record)
    added = intel.add(store, accepted)
    return added, rejected


def _print_brief(plan):
    print("=" * 74)
    print(f"SCOUTING BRIEF - {len(plan)} fighters")
    print("=" * 74)
    print("\n  Run each query. Record ONLY what a named source actually says,")
    print("  with its URL. Nothing you record will move a probability; it is")
    print("  shown next to the number so a reader knows what the model cannot")
    print("  see. A claim with no source is worth less than no claim at all.")
    print(f"\n  Reach for these first: {', '.join(PREFERRED[:5])} ...")
    for entry in plan:
        print(f"\n  {entry['fighter']}  (event {entry['event_date']})")
        for q in entry["queries"]:
            print(f"    [{q['kind']:<16}] {q['query']}")
    print("\n" + "=" * 74)
    print("  Write findings to a JSON list and ingest them:")
    print("    python engine/intel_search.py --ingest findings.json")
    print("  Each item: fighter, event_date, kind, confidence, note, source")
    print("  and gathered (ISO8601). Anything gathered after its event is")
    print("  rejected - that is hindsight, not research.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:300])
    parser.add_argument("--brief", action="store_true",
                        help="print the research plan for the current card")
    parser.add_argument("--card", default=str(ENGINE.parent / "app" / "data"
                                              / "card.json"))
    parser.add_argument("--ingest", help="a JSON list of findings")
    args = parser.parse_args(argv)

    if args.brief:
        card = json.loads(Path(args.card).read_text())
        fights = [(f.get("red"), f.get("blue")) for f in card.get("fights", [])]
        _print_brief(brief(fights, card.get("event", {}).get("date", "")))
        return 0

    if args.ingest:
        findings = json.loads(Path(args.ingest).read_text())
        if isinstance(findings, dict):
            findings = findings.get("findings", [])
        store = intel.load()
        added, rejected = ingest(findings, store=store)
        intel.save(store)
        print(f"  recorded {added} of {len(findings)}")
        for reason in rejected:
            print(f"  REJECTED  {reason}")
        print("\n  None of this moves a probability. No weight is fitted for "
              "any kind;")
        print("  experiments/short_notice_weight.py measured the one we could "
              "label\n  at scale and its interval spans zero.")
        return 0

    parser.error("use --brief or --ingest")


if __name__ == "__main__":
    raise SystemExit(main())
