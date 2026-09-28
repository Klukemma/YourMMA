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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"

API = "https://en.wikipedia.org/w/api.php"
AGENT = "YourMMA-research/1.0 (fight-change labelling probe)"
TIMEOUT = 30         # seconds per request
SEARCH_LIMIT = 3     # candidate articles considered per event
PAUSE = 0.15         # seconds between calls, to be a good citizen

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


INFOBOX_DATE = re.compile(r"\|\s*date\s*=\s*([^\n|]+)", re.I)


def infobox_date(text):
    """The date an event article says it happened, or None.

    The article is the authority on its own date. Everything else here -
    search ranking, a title that looks right - is a guess, and this is what
    turns a guess into a match or throws it away.
    """
    hit = INFOBOX_DATE.search(text or "")
    if not hit:
        return None
    raw = re.sub(r"\{\{[^{}]*\|([^{}|]+)\}\}", r"\1", hit.group(1))
    raw = re.sub(r"[\[\]']", "", raw).strip()
    # {{Start date|2024|4|13}} survives the line above as "2024|4|13"
    numbers = re.findall(r"\d+", raw)
    for attempt in (raw, "-".join(numbers[:3]) if len(numbers) >= 3 else ""):
        if not attempt:
            continue
        try:
            return pd.Timestamp(attempt).date()
        except (ValueError, TypeError):
            continue
    return None


def search_titles(name):
    """Article titles Wikipedia thinks match this event name."""
    body = get({"action": "query", "list": "search", "srsearch": name,
                "srlimit": SEARCH_LIMIT})
    return [hit["title"] for hit in
            body.get("query", {}).get("search", [])]


def resolve(name, when):
    """(title, wikitext) for this event, or (None, None).

    SEARCH THEN VERIFY, rather than either alone. An earlier version of this
    built a date -> title map by reading links and dates out of a list page's
    table rows, which paired whichever link came first in a row with whichever
    date did: it resolved 10% of our events and dated UFC 16 - a 1998 card -
    to 2009. Both halves of that failure were invisible in the output; the
    fetches simply 404'd and the coverage number came out low, which would
    have been read as "Wikipedia does not carry this" and killed the work.

    So the title comes from search, which is good at names, and the DATE comes
    from the article's own infobox, which cannot be wrong about when its event
    happened. A candidate whose date disagrees with ours is the wrong article
    and is dropped rather than used.
    """
    for title in search_titles(name):
        try:
            text = wikitext(title)
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
        if not text:
            continue
        theirs = infobox_date(text)
        if theirs and abs((theirs - when).days) <= 1:
            return title, text
        time.sleep(PAUSE)
    return None, None


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

    # Sample across the WHOLE span rather than the most recent N: coverage on
    # a 2024 page says nothing about a 2009 one, and a weight fitted on the
    # last three years would be fitted on the years the market prices best.
    order = events.reset_index(drop=True)
    step = max(len(order) // args.sample, 1)
    sample = order.iloc[::step].head(args.sample)
    print(f"\n  resolving and fetching {len(sample)} events spread over "
          f"{sample.date.min().year}-{sample.date.max().year}...")

    # ONLY a network failure may be reported as unreachable. A bare `except
    # Exception` here turned a NameError in this file into "Wikipedia cannot
    # be read from here", which is a lie that costs a CI round trip to catch.
    try:
        search_titles("UFC 300")
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        print(f"\n  UNREACHABLE: {type(err).__name__}: {err}")
        print("  Wikipedia cannot be read from here. This container's egress")
        print("  proxy refuses it; run the check-fight-changes workflow mode,")
        print("  where the runner has open internet.")
        return 1

    total, with_changes, quantified, failures, surname_only = 0, 0, 0, 0, 0
    examples = []
    unresolved = []
    for event in sample.itertuples():
        when = event.date.date()
        try:
            title, text = resolve(event.event_name, when)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            failures += 1
            print(f"    {when}  FETCH FAILED  {type(err).__name__}")
            continue
        if not text:
            unresolved.append((when, event.event_name))
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
            examples.append((when, title, rows[0]))

    fetched = len(sample) - failures - len(unresolved)
    if not fetched:
        print("\n  every fetch failed; nothing can be said about coverage.")
        return 1

    per_event = total / fetched
    print(f"\n  {'events sampled':<34}{len(sample):>8}")
    print(f"  {'resolved to an article':<34}{fetched:>8}"
          f"  ({fetched / len(sample):.0%})")
    print(f"  {'pages naming a replacement':<34}{with_changes:>8}"
          f"  ({with_changes / fetched:.0%})")
    print(f"  {'replacements found':<34}{total:>8}")
    print(f"  {'with a notice period stated':<34}{quantified:>8}")
    print(f"  {'names given as a surname only':<34}{surname_only:>8}"
          f"  (of {total * 2}, to resolve against the card)")
    print(f"  {'per event':<34}{per_event:>8.2f}")

    # Project on the RESOLUTION RATE as well as the yield: an event whose
    # article cannot be found contributes no labels, and pretending otherwise
    # would overstate what is available by exactly the miss rate.
    resolved_share = fetched / len(sample)
    reachable = resolved_share * len(events)
    projected = per_event * reachable
    print(f"\n  at this resolution rate, {reachable:,.0f} of "
          f"{len(events):,} events are reachable")
    print(f"  projected labels over all of them: {projected:,.0f} fighters")
    interval = 0.9 / max(projected, 1) ** 0.5
    print(f"  which gives a 95% interval of about +/-{interval:.3f} on the "
          f"effect")
    if unresolved:
        print(f"\n  {len(unresolved)} did not resolve, e.g. "
              + "; ".join(f"{d} {n}" for d, n in unresolved[:3]))
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
