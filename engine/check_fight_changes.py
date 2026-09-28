"""Can a late replacement be labelled across history, and on how many fights?

WHY THIS COMES FIRST. predict_card carries two hand-typed constants nobody
fitted:

    'short_notice': -0.04,       # ~4% penalty
    'weight_cut_hard': -0.03,    # ~3% penalty

Fitting them needs labels: for each historical fight, who stepped in late and
who they replaced. Nothing in the archive carries that - 130 columns and not
one mentions notice, replacement or withdrawal - so it has to come from
outside, and the only free, complete, machine-readable record of UFC fight
changes is Wikipedia's event pages, which carry sentences like

    "Smith was expected to face Jones. However, Jones withdrew due to injury
     and was replaced by Brown."

This asks whether that can actually be harvested, BEFORE anything is built on
the assumption that it can. Three ways it could fail and they want different
answers:

    unreachable      the runner cannot fetch Wikipedia -> the whole approach
                     dies here and the constants should go to zero instead
    unmatchable      our event names do not resolve to article titles ->
                     solvable, but it is a name-matching project first
    too thin         the pages carry too few usable pairs to fit anything ->
                     the honest move is to delete the hook, not tune it

HOW MANY LABELS ARE ENOUGH. The effect is estimated as the mean residual of
short-notice fighters - how much they under-perform the probability the model
gave them - and a residual has a standard deviation near 0.45. So the 95%
interval on the estimate is about +/- 0.9/sqrt(N):

    N = 200    +/- 6.4 points    cannot tell 4% from 0%
    N = 400    +/- 4.5 points    cannot tell 4% from 0%
    N = 1000   +/- 2.8 points    can, barely
    N = 2500   +/- 1.8 points    comfortably

Roughly 6% of UFC bouts involve a late replacement, so 755 events at about 11
bouts each gives perhaps 500 labelled fighters IF coverage is complete. That
is the number this prints, and if it comes back at 200 the answer to item 2
is "this cannot be fitted from Wikipedia", which is a finding, not a failure.

    python engine/check_fight_changes.py            (needs network)
    python engine/check_fight_changes.py --sample 60

Prints counts and a handful of extracted rows to eyeball. Writes nothing.
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"

API = "https://en.wikipedia.org/w/api.php"
LIST_PAGES = ("List of UFC events",)
AGENT = "YourMMA-research/1.0 (fight-change labelling probe)"
TIMEOUT = 30

# The sentence shapes Wikipedia uses for a change. Deliberately several
# patterns rather than one clever regex: these are written by hundreds of
# different editors and a single pattern silently drops the variants it does
# not happen to match, which would show up here as "too thin" and send the
# whole project down the wrong road.
REPLACEMENT_PATTERNS = (
    # "... was replaced by Brown"
    re.compile(r"(?P<out>[A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+){0,3})\s+"
               r"(?:withdrew|pulled out|was (?:forced )?to withdraw|was removed)"
               r"[^.]{0,120}?replaced by\s+"
               r"(?P<in>[A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+){0,3})"),
    # "Brown replaced Jones"
    re.compile(r"(?P<in>[A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+){0,3})\s+"
               r"(?:stepped in for|replaced)\s+"
               r"(?P<out>[A-Z][\w.'’-]+(?: [A-Z][\w.'’-]+){0,3})"),
)

# "on nine days' notice", "on two weeks notice", and bare "on short notice"
# with no period given at all - which is the commonest phrasing of the three
# and was missed by the first version of this regex, because it demanded a
# unit that sentence does not have.
NOTICE = re.compile(r"on\s+(?:(?P<n>\d+|one|two|three|four|five|six|seven|"
                    r"eight|nine|ten|eleven|twelve)\s+)?(?P<unit>day|week)s?'?\s*"
                    r"(?:\s|-)?notice", re.I)
SHORT_NOTICE = re.compile(r"\b(?:short|late)(?:\s|-)notice\b", re.I)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12}


def get(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode())


def wikitext(title):
    body = get({"action": "parse", "page": title, "prop": "wikitext",
                "redirects": 1})
    if "error" in body:
        return None
    return body["parse"]["wikitext"]["*"]


def event_titles():
    """{date: article title} for every UFC event Wikipedia lists.

    Matching on DATE rather than on the event name, because our archive spells
    events differently from Wikipedia in ways that are not worth a fuzzy
    matcher ("UFC 182: Jones vs Cormier" against "UFC 182"). A date is exact,
    and two UFC events on one day is rare enough to notice in the counts.
    """
    found = {}
    for page in LIST_PAGES:
        text = wikitext(page)
        if not text:
            continue
        # Rows look like: | [[UFC 300]] | April 13, 2024 | ...
        for row in text.split("\n|-"):
            link = re.search(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", row)
            date = re.search(r"(\w+ \d{1,2}, \d{4})", row)
            if link and date:
                try:
                    when = pd.Timestamp(date.group(1)).date()
                except ValueError:
                    continue
                found.setdefault(when, link.group(1).strip())
    return found


def _clean(name):
    """Trim the sentence's punctuation off a captured name.

    The name pattern has to allow a full stop so that "Jr." and initials
    survive, which means a name at the end of a sentence swallows the sentence's
    own full stop and comes back as "Carl Brown." - close enough to look right
    in a printout and wrong enough to never join to a fighter. Caught by a test,
    not by reading the output.
    """
    return str(name or "").strip().strip(".,;:!?'\u2019\"").strip()


def changes_in(text):
    """Every (stepped in, replaced, days notice) this page states."""
    # The prose lives outside the bout tables; drop the templates so a
    # fighter's name in a results table is never read as a replacement.
    prose = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    prose = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", prose)
    prose = re.sub(r"<ref[^>]*>.*?</ref>", " ", prose, flags=re.S)
    prose = re.sub(r"<[^>]+>", " ", prose)

    rows = []
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        if "replac" not in sentence and "stepped in" not in sentence:
            continue
        for pattern in REPLACEMENT_PATTERNS:
            hit = pattern.search(sentence)
            if not hit:
                continue
            notice = NOTICE.search(sentence)
            days = None
            if notice:
                raw = notice.group("n")
                count = WORDS.get((raw or "").lower(), None)
                if count is None and raw and raw.isdigit():
                    count = int(raw)
                if count is not None:
                    days = count * (7 if notice.group("unit").lower() == "week"
                                    else 1)
                else:
                    days = 0        # a unit but no number
            elif SHORT_NOTICE.search(sentence):
                days = 0            # said to be short, no period given
            stepped_in, replaced = (_clean(hit.group("in")),
                                    _clean(hit.group("out")))
            if not stepped_in or not replaced:
                continue
            rows.append({"stepped_in": stepped_in,
                         "replaced": replaced,
                         "days_notice": days,
                         "sentence": sentence.strip()[:200]})
            break
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--sample", type=int, default=40,
                        help="how many events to actually fetch")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("CAN LATE REPLACEMENTS BE LABELLED ACROSS HISTORY?")
    print("=" * 74)

    archive = pd.read_csv(ARCHIVE, usecols=["event_name", "date"],
                          low_memory=False)
    archive["date"] = pd.to_datetime(archive["date"], errors="coerce")
    events = (archive.dropna(subset=["date"])
              .drop_duplicates("event_name").sort_values("date"))
    bouts = len(archive)
    print(f"\n  archive: {len(events):,} events, {bouts:,} bouts, "
          f"{events.date.min().date()} to {events.date.max().date()}")

    try:
        titles = event_titles()
    except Exception as err:                      # noqa: BLE001
        print(f"\n  UNREACHABLE: {type(err).__name__}: {err}")
        print("  Wikipedia cannot be read from here. Run this on the Actions")
        print("  runner, which has open network, before concluding anything.")
        return 1
    print(f"  Wikipedia lists {len(titles):,} events by date")

    ours = {d.date(): n for d, n in zip(events.date, events.event_name)}
    matched = {d: titles[d] for d in ours if d in titles}
    print(f"  matched on date: {len(matched):,} of {len(ours):,} "
          f"({len(matched) / max(len(ours), 1):.0%})")
    if not matched:
        print("\n  UNMATCHABLE: no event resolved to an article. This is a")
        print("  name/date matching problem, not a data availability one.")
        return 1

    # Sample across the whole span rather than the most recent N: coverage on
    # a 2024 page says nothing about a 2009 one, and the fit needs both.
    order = sorted(matched)
    step = max(len(order) // args.sample, 1)
    sample = order[::step][:args.sample]
    print(f"\n  fetching {len(sample)} event pages spread over "
          f"{sample[0].year}-{sample[-1].year}...")

    total, with_changes, quantified, failures, surname_only = 0, 0, 0, 0, 0
    examples = []
    for when in sample:
        try:
            text = wikitext(matched[when])
        except Exception as err:                  # noqa: BLE001
            failures += 1
            print(f"    {when}  FETCH FAILED  {type(err).__name__}")
            continue
        if not text:
            failures += 1
            continue
        rows = changes_in(text)
        total += len(rows)
        with_changes += bool(rows)
        quantified += sum(1 for r in rows if r["days_notice"])
        # A one-word name has to be resolved against the card rather than
        # matched directly, so this is the size of the joining job.
        surname_only += sum(1 for r in rows
                            for n in (r["stepped_in"], r["replaced"])
                            if len(n.split()) == 1)
        if rows and len(examples) < 6:
            examples.append((when, matched[when], rows[0]))

    fetched = len(sample) - failures
    if not fetched:
        print("\n  every fetch failed; nothing can be said about coverage.")
        return 1

    per_event = total / fetched
    print(f"\n  {'pages fetched':<34}{fetched:>8}")
    print(f"  {'pages naming a replacement':<34}{with_changes:>8}"
          f"  ({with_changes / fetched:.0%})")
    print(f"  {'replacements found':<34}{total:>8}")
    print(f"  {'with a notice period stated':<34}{quantified:>8}")
    print(f"  {'names given as a surname only':<34}{surname_only:>8}"
          f"  (of {total * 2}, to resolve against the card)")
    print(f"  {'per event':<34}{per_event:>8.2f}")

    projected = per_event * len(matched)
    print(f"\n  projected over all {len(matched):,} matched events: "
          f"{projected:,.0f} labelled fighters")
    interval = 0.9 / max(projected, 1) ** 0.5
    print(f"  which gives a 95% interval of about +/-{interval:.3f} on the "
          f"effect")
    if projected < 400:
        print("  TOO THIN. A hand-typed 4% cannot be told from 0% at this N.")
        print("  The honest move is to delete the hook, not tune it.")
    elif projected < 1000:
        print("  MARGINAL. Enough to rule out a large effect, not enough to")
        print("  confirm a 4% one. Worth gathering; report the interval, and")
        print("  do not ship a weight whose interval spans zero.")
    else:
        print("  ENOUGH. A 4% effect would be distinguishable from zero.")

    if examples:
        print("\n  a few, to eyeball the parser:")
        for when, title, row in examples:
            notice = (f"{row['days_notice']}d" if row["days_notice"]
                      else "unstated")
            print(f"\n    {when}  {title}")
            print(f"      {row['stepped_in']} replaced {row['replaced']}  "
                  f"({notice})")
            print(f"      \"{row['sentence']}\"")
    print("\n  Nothing was written. This only answers whether to build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
