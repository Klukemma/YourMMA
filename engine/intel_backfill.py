"""Historical fighter news, 2010 onward - kept only when provably pre-fight.

WHY THIS IS A DIFFERENT PROBLEM FROM THE LIVE SCOUT. For an upcoming card,
"gathered before the bell" is enough, because nothing written after the fight
exists yet. For a 2014 fight, everything exists - and most of what a search
returns about a fighter's injury was written AFTER the fight. "X reveals he
fought with a torn ACL" is published the week after he loses. Losers disclose
far more often than winners, so a history collected without dates would show a
huge "injury" effect in testing and be worthless live, because the information
did not exist when a prediction would have been made.

So the only question that matters for each record is: WHEN WAS THIS
PUBLISHED? And the answer must be verified, never taken on trust - a
publication date is exactly the detail a language model reading a search
result will supply confidently and wrongly. A claimed `published` is kept
for audit and compared, but it decides nothing. Two things can verify:

    the URL      most outlets put the date in the path (/2014/03/08/); read
                 by a regex, not by a model. Month-only paths (/2014/03/)
                 verify only if the whole month is before the event.
    the page     the runner fetches it and reads the page's own
                 published_time metadata (verify_intel_dates.py). This
                 container cannot fetch pages, so these wait as PENDING.

A record is accepted only if the latest date it could have been published is
STRICTLY before the event date. Same-day is rejected: an American fight
night's results stories carry the fight's own date.

THE SAMPLE, AND WHY THE EMPTY SEARCHES ARE RECORDED. There are about 14,000
fighter-appearances since 2010; searching all of them is not happening. So
fights are drawn at RANDOM (seeded, reproducible), and every fighter searched
is written to the log whether or not anything was found. Without that log:
there is no finding rate, no way to see whether 2011 is covered as well as
2024, and no control group of searched-and-clean fighters to compare against.

WHAT THIS CANNOT CATCH, said plainly. An article published before the fight
and EDITED after it ("UPDATE: he lost in the first round") carries a pre-fight
published date and post-fight content. A search summary read from today's
version of the page could fold the update into the note. Notes must state the
pre-fight claim only, and this is the residual risk that remains.
"""

import argparse
import calendar
import json
import random
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import fight_intel as intel

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
SCOUTED = ENGINE / "data" / "intel_history_scouted.csv"
PENDING = ENGINE / "data" / "intel_history_pending.csv"
LOG = ENGINE / "data" / "intel_scout_log.csv"

FROM_YEAR = 2010
RECORD_COLUMNS = ("event_date", "event_name", "fighter", "kind",
                  "confidence", "note", "source", "published_claimed",
                  "published", "verified_by", "withdrawn", "gathered")
LOG_COLUMNS = ("event_date", "event_name", "fighter", "queries", "found",
               "accepted", "pending", "rejected", "searched_at")

# The two questions per fighter. Fewer than the live brief's seven because a
# historical search returns mostly results coverage, and a combined query
# finds the same pre-fight stories at a fraction of the cost.
QUESTIONS = (
    '"{fighter}" "{opponent}" preview OR "fight week" OR camp {year}',
    '"{fighter}" injury OR "short notice" OR "weight cut" OR "new camp" {year}',
)

URL_DAY = re.compile(r"/((?:19|20)\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|-|$)")
URL_MONTH = re.compile(r"/((?:19|20)\d{2})/(\d{1,2})/")


class BackfillError(ValueError):
    """A record that cannot be trusted, with the reason."""


def _as_date(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(str(value)[:10]).date()


def url_date(url):
    """(earliest, latest) the URL says it could have been published, or None.

    A full date in the path pins both ends to one day. A year and month pins
    them to the first and last day of that month - which is why a month-only
    URL can verify a story from March for a fight in May, but not a story from
    March for a fight on the 20th of March.
    """
    text = str(url or "")
    hit = URL_DAY.search(text)
    if hit:
        try:
            day = date(int(hit.group(1)), int(hit.group(2)), int(hit.group(3)))
            return day, day
        except ValueError:
            pass
    hit = URL_MONTH.search(text)
    if hit:
        year, month = int(hit.group(1)), int(hit.group(2))
        if 1 <= month <= 12:
            last = calendar.monthrange(year, month)[1]
            return date(year, month, 1), date(year, month, last)
    return None


def verdict(record, *, verified=None):
    """("accept", verified_by, published) | ("pending", ...) | ("reject", reason).

    `verified` is a date the runner read from the page itself, when it has
    one. Without it, only the URL can verify. The researcher's claimed date
    is compared against whatever verified, and a disagreement of more than a
    day is a rejection: it means the summary got the date wrong, and a
    summary that got the date wrong may have got the content wrong too.
    """
    event = _as_date(record["event_date"])
    claimed = record.get("published_claimed") or record.get("published")
    claimed = _as_date(claimed) if claimed else None

    if verified is not None:
        low = high = _as_date(verified)
        how = "page"
    else:
        bounds = url_date(record.get("source"))
        if bounds is None:
            return ("pending", "no date in the URL; the runner must read the "
                               "page", None)
        low, high = bounds
        how = "url"

    if high >= event:
        if low >= event:
            return ("reject", f"published {low}, on or after the {event} "
                              f"event - written after the fight", None)
        if verified is None:
            # Month-only URL straddling the event: the page must decide.
            return ("pending", f"URL dates it to {low:%Y-%m}, the event's own "
                               f"month; the runner must read the page", None)
        return ("reject", f"published {high}, not before {event}", None)

    if claimed is not None and low == high and abs((claimed - low).days) > 1:
        return ("reject", f"researcher claimed {claimed}, the {how} says "
                          f"{low} - a summary that has the date wrong cannot "
                          f"be trusted on the content", None)
    return ("accept", how, high if low == high else low)


def _validated(raw):
    """Field checks shared with the live scout: source, kind, note, extras."""
    extra = {f: raw[f] for f in intel.EXTRA_FIELDS.get(raw.get("kind"), ())
             if f in raw}
    try:
        intel.observation(fighter=raw.get("fighter"),
                          event_date=raw.get("event_date"),
                          kind=raw.get("kind"),
                          confidence=raw.get("confidence", "reported"),
                          note=raw.get("note"), source=raw.get("source"),
                          gathered=raw.get("gathered"), **extra)
    except intel.IntelError as err:
        raise BackfillError(str(err)) from None
    if not raw.get("event_date"):
        raise BackfillError(f"{raw.get('fighter')}: no event_date")


def classify(findings):
    """Split findings into accepted, pending and rejected (with reasons)."""
    accepted, pending, rejected = [], [], []
    for raw in findings:
        try:
            _validated(raw)
        except BackfillError as err:
            rejected.append(str(err))
            continue
        outcome, detail, published = verdict(raw)
        row = {c: raw.get(c) for c in RECORD_COLUMNS}
        row["published_claimed"] = raw.get("published_claimed") \
            or raw.get("published")
        row["withdrawn"] = raw.get("replaced_opponent") or raw.get("withdrawn")
        row["gathered"] = raw.get("gathered") or _now()
        if outcome == "accept":
            row["published"] = str(published)
            row["verified_by"] = detail
            accepted.append(row)
        elif outcome == "pending":
            row["published"] = None
            row["verified_by"] = None
            pending.append(row)
        else:
            rejected.append(f"{raw.get('fighter')}: {detail}")
    return accepted, pending, rejected


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path, rows, columns, key):
    """Append rows, dropping exact duplicates on `key`. Returns rows added."""
    new = pd.DataFrame(rows, columns=list(columns))
    if new.empty:
        return 0
    old = pd.read_csv(path) if Path(path).exists() else \
        pd.DataFrame(columns=list(columns))
    before = len(old)
    both = pd.concat([old, new], ignore_index=True)
    both = both.drop_duplicates(subset=list(key), keep="first")
    both.to_csv(path, index=False)
    return len(both) - before


def sample_fights(archive, n, *, seed=0, from_year=FROM_YEAR, before=None):
    """n fights drawn at random, reproducibly, from `from_year` onward.

    Random, not the main events. A sample of headliners would over-represent
    exactly the fighters who get the most coverage, and the finding rate - and
    anything fitted on it - would describe famous fighters only.
    """
    frame = archive.dropna(subset=["date", "r_name", "b_name"]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame.date.dt.year >= from_year]
    if before is not None:
        frame = frame[frame.date < pd.Timestamp(before)]
    rows = list(frame[["date", "event_name", "r_name", "b_name"]]
                .itertuples(index=False))
    rows.sort(key=lambda r: (r.date, r.event_name, r.r_name))
    random.Random(seed).shuffle(rows)
    return [{"event_date": str(r.date.date()), "event_name": r.event_name,
             "red": r.r_name, "blue": r.b_name} for r in rows[:n]]


def brief(fights):
    plan = []
    for fight in fights:
        year = fight["event_date"][:4]
        for fighter, opponent in ((fight["red"], fight["blue"]),
                                  (fight["blue"], fight["red"])):
            plan.append({
                "event_date": fight["event_date"],
                "event_name": fight["event_name"],
                "fighter": fighter, "opponent": opponent,
                "queries": [q.format(fighter=fighter, opponent=opponent,
                                     year=year) for q in QUESTIONS],
            })
    return plan


def ingest(payload):
    """Findings plus the search log in; three files updated.

    The payload is {"searched": [...], "findings": [...]}. `searched` lists
    every fighter the researcher ran queries for, INCLUDING those with no
    findings - that list is the denominator, and without it the result is a
    pile of positives with nothing to compare them against.
    """
    findings = payload.get("findings", [])
    searched = payload.get("searched", [])
    accepted, pending, rejected = classify(findings)

    key = ("event_date", "fighter", "kind", "source")
    added = _append(SCOUTED, accepted, RECORD_COLUMNS, key)
    queued = _append(PENDING, pending, RECORD_COLUMNS, key)

    counts = {}
    for row, bucket in [(r, "accepted") for r in accepted] + \
                       [(r, "pending") for r in pending]:
        k = (str(row["event_date"])[:10], row["fighter"])
        counts.setdefault(k, {"accepted": 0, "pending": 0})[bucket] += 1
    log_rows = []
    for entry in searched:
        k = (str(entry["event_date"])[:10], entry["fighter"])
        found = sum(1 for f in findings
                    if (str(f.get("event_date"))[:10], f.get("fighter")) == k)
        c = counts.get(k, {"accepted": 0, "pending": 0})
        log_rows.append({
            "event_date": k[0], "event_name": entry.get("event_name"),
            "fighter": k[1], "queries": entry.get("queries", len(QUESTIONS)),
            "found": found, "accepted": c["accepted"],
            "pending": c["pending"],
            "rejected": found - c["accepted"] - c["pending"],
            "searched_at": entry.get("searched_at") or _now(),
        })
    logged = _append(LOG, log_rows, LOG_COLUMNS, ("event_date", "fighter"))
    return {"accepted": added, "pending": queued, "rejected": rejected,
            "logged": logged}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:300])
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--sample", type=int, default=10,
                        help="how many fights to draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from-year", type=int, default=FROM_YEAR)
    parser.add_argument("--ingest", help="JSON {searched: [], findings: []}")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    if args.brief:
        archive = pd.read_csv(ARCHIVE, usecols=["event_name", "date",
                                                "r_name", "b_name"],
                              low_memory=False)
        fights = sample_fights(archive, args.sample, seed=args.seed,
                               from_year=args.from_year)
        print(json.dumps(brief(fights), indent=1))
        return 0

    if args.ingest:
        result = ingest(json.loads(Path(args.ingest).read_text()))
        print(f"  accepted {result['accepted']}, pending {result['pending']} "
              f"(waiting for the runner to read the page), "
              f"logged {result['logged']} searched fighters")
        for reason in result["rejected"]:
            print(f"  REJECTED  {reason}")
        return 0

    if args.status:
        for name, path in (("accepted", SCOUTED), ("pending", PENDING),
                           ("searched", LOG)):
            n = len(pd.read_csv(path)) if Path(path).exists() else 0
            print(f"  {name:<10}{n:>6}   {path.name}")
        return 0

    parser.error("use --brief, --ingest or --status")


if __name__ == "__main__":
    raise SystemExit(main())
