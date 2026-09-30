"""Every injury and illness sentence Wikipedia holds on UFC fighters since 2010.

NOT A SAMPLE. Every event page since 2010 and every fighter who has fought in
the UFC since 2010 is asked for, and what could not be found is counted and
reported, so coverage is a number and not an impression.

WHAT THIS PRODUCES IS TEXT, NOT LABELS. A sentence mentioning a torn ACL
says nothing yet about WHEN anyone knew it, and that is the only thing that
decides whether it may be used. "He withdrew with a knee injury" is known
before the bout it cancels. "He later revealed he had fought with a broken
hand" is known after the bout it describes. The second is still worth
keeping - it is injury history for every fight AFTER that one - but used as a
feature for the fight it describes, it would hand the model the result, since
losers disclose injuries far more than winners. So each sentence is kept with
everything needed to date it: the paragraph it sits in, the latest date and
event named before it in its section, and for event pages the event's own
date. Structuring and dating is done afterwards, by reading, in
injury_history.py.

Sources, per fighter and per event:
  - prose: every sentence matching INJURY, in every section but the
    reference lists
  - record tables: every row of a fighter's MMA record whose method or notes
    mention an injury, a doctor or a retirement ("TKO (knee injury)"),
    with the row's own date

    python engine/harvest_injuries.py                  (needs network)
    python engine/harvest_injuries.py --limit 20       (smoke test)

Writes engine/data/injury_sentences.jsonl.gz and engine/data/injury_pages.csv.
Run it from the workflow; this container's egress proxy refuses Wikipedia.
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
import urllib.error
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import (ARCHIVE, _BREAK, _prose, get, resolve_all,
                                 search_pages)
from name_resolution import norm_name

SENTENCES = ENGINE / "data" / "injury_sentences.jsonl.gz"
PAGES = ENGINE / "data" / "injury_pages.csv"
FROM_YEAR = 2010
EXISTS_BATCH = 50    # titles per existence check; no content, so cheap
CONTENT_BATCH = 15   # fighter pages are long, and the API caps a response
                     # at about 8 MB - past that it answers in parts, which
                     # fetch_pages follows rather than silently truncating
CONTEXT_BEFORE = 3   # sentences of the same paragraph kept before a hit
CONTEXT_AFTER = 1

# What makes a sentence worth reading. Broad on purpose: a sentence let
# through that is not about an injury costs one reader a second; a sentence
# kept out is an injury nobody will ever count. Body parts alone are NOT
# here, because "a knee to the head" would drown everything - they are
# caught through the injury words, or through "knee problem"-style phrases.
INJURY = re.compile(
    r"\b(?:injur\w*|torn|tore|tearing|rupture\w*|fractur\w*|broke|broken|"
    r"surger\w*|surgical|operat(?:ed|ion)\s+on|"
    r"acl|mcl|pcl|lcl|menisc\w*|labrum|rotator\s+cuff|achilles|"
    r"herniat\w*|hernia|concussion\w*|dislocat\w*|sprain\w*|strain(?:ed)?|"
    r"staph|infection|infected|hospitali[sz]\w*|hospital|illness|ill|sick\w*|"
    r"flu|pneumonia|virus|covid(?:-19)?|coronavirus|tested\s+positive|"
    r"dehydrat\w*|kidney|medically|medical\s+(?:reasons|issue\w*|"
    r"suspension|clearance)|not\s+cleared|doctor\w*|"
    r"could\s+not\s+continue|unable\s+to\s+continue|"
    r"(?:knee|back|shoulder|hand|foot|neck|rib|elbow|ankle|wrist|hip|"
    r"leg|arm|eye|head)\s+(?:problem|issue|ailment|complication)s?)\b",
    re.I)
# In a record table the method cell is terse: "TKO (knee injury)",
# "TKO (doctor stoppage)", "TKO (retirement)". Retirement is a corner or a
# fighter stopping between rounds, very often for an injury, so it is read.
TABLE_HIT = re.compile(INJURY.pattern[:-3] + r"|retirement)\b", re.I)

SKIP_SECTIONS = re.compile(r"references|external links|see also|notes|"
                           r"further reading|bibliography|sources", re.I)
HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
# Dates as prose writes them, and as the date templates are rewritten to
# before the templates are stripped.
DATE = re.compile(
    rf"\b(?:(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}|\d{{1,2}}\s+(?:{MONTHS})\s+"
    rf"\d{{4}}|(?:{MONTHS})\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b")
DATE_TEMPLATE = re.compile(
    r"\{\{\s*(?:dts|[Ss]tart date|[Dd]ate|[Ff]ight date|[Ee]nd date)\s*"
    r"\|\s*(?:format=\w+\s*\|\s*)?(\d{4})\s*\|\s*(\w+)\s*\|\s*(\d{1,2})"
    r"[^{}]*\}\}")
# An event named in prose: "UFC 196", "UFC Fight Night 81", "UFC on Fox 18",
# "UFC Fight Night: Holloway vs. Kattar". A hint for dating, not an identity;
# the extraction step and the archive join decide what it refers to.
EVENT = re.compile(
    r"\b(?:UFC(?:\s+Fight\s+Night|\s+on\s+(?:FOX|Fox|FX|Fuel\s+TV|ESPN\+?|"
    r"ABC|Versus))?(?::?\s+\d+|:\s+[A-Z][\w.'’-]+(?:\s+[A-Z][\w.'’-]+)?"
    r"\s+vs\.?\s+[A-Z][\w.'’-]+)|The\s+Ultimate\s+Fighter(?::\s+[^,.;]+)?)")

MONTH_NUMBER = {m.lower(): i for i, m in enumerate(MONTHS.split("|"), 1)}


def events_in(text):
    """Event names in a sentence, without the full stop a name char took."""
    return [e.rstrip(".,;:") for e in EVENT.findall(text or "")]


def _iso_date(match):
    """{{dts|2012|08|11}} or {{dts|2012|August|11}} as "2012-08-11"."""
    year, month, day = match.group(1), match.group(2), match.group(3)
    if month.isdigit():
        number = int(month)
    else:
        number = MONTH_NUMBER.get(month.lower())
        if not number:
            return match.group(0)
    return f"{int(year):04d}-{number:02d}-{int(day):02d}"


def keep_dates(text):
    """Date templates rewritten to ISO before anything strips templates.

    Stripping first would delete the date from every record-table row and
    from a lot of prose, and a row with no date is a row nobody can place.
    """
    return DATE_TEMPLATE.sub(_iso_date, text or "")


def sections(wikitext):
    """[(heading, body)] in page order; the lead is heading ''."""
    out, last, name = [], 0, ""
    for hit in HEADING.finditer(wikitext):
        out.append((name, wikitext[last:hit.start()]))
        name, last = hit.group(2).strip(), hit.end()
    out.append((name, wikitext[last:]))
    return out


TABLE = re.compile(r"^\{\|.*?^\|\}", re.S | re.M)


def _cell_text(cell):
    cell = re.sub(r"^\s*(?:[a-z]+\s*=\s*\"?[^|\"]*\"?\s*\|)+", "", cell)
    return re.sub(r"\s+", " ", _prose(cell)).strip()


def table_rows(body):
    """Each table row as one "cell | cell | cell" line, and the body without
    its tables.

    A record table written out as prose would give one "sentence" per cell,
    and "TKO (knee injury)" on its own says nothing about who or when. As a
    row it carries the result, the opponent, the event and the date.
    """
    rows = []
    for table in TABLE.findall(body):
        for raw in re.split(r"^\|-.*$", table, flags=re.M)[1:]:
            cells = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line or line.startswith("|}"):
                    continue
                if line[0] in "|!":
                    cells.extend(re.split(r"\|\||!!", line[1:]))
            texts = [t for t in (_cell_text(c) for c in cells) if t]
            if texts:
                rows.append(" | ".join(texts))
    return rows, TABLE.sub("\n\n", body)


def _split(block):
    block = re.sub(r"\s*\n\s*", " ", block)
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", block)
            if s.strip()]


def hits(wikitext):
    """Every injury-bearing sentence and record row on one page.

    Each comes with the heading it sits under, up to CONTEXT_BEFORE sentences
    before it and CONTEXT_AFTER after it from the same paragraph, and the
    latest date and event named earlier in the same section - the things a
    reader needs to say which fight a sentence is about.
    """
    found = []
    for heading, body in sections(keep_dates(wikitext)):
        if SKIP_SECTIONS.fullmatch(heading.strip()) or \
                SKIP_SECTIONS.match(heading.strip()):
            continue
        rows, prose_body = table_rows(body)
        for row in rows:
            if TABLE_HIT.search(row):
                dates = DATE.findall(row)
                found.append({"section": heading, "form": "record_row",
                              "text": row[:600], "context": "",
                              "last_date": dates[-1] if dates else "",
                              "last_event": ""})
        last_date, last_event = "", ""
        for block in _BREAK.split(_prose(prose_body)):
            sentences = _split(block)
            for i, sentence in enumerate(sentences):
                if INJURY.search(sentence):
                    before = sentences[max(0, i - CONTEXT_BEFORE):i]
                    after = sentences[i + 1:i + 1 + CONTEXT_AFTER]
                    found.append({
                        "section": heading, "form": "prose",
                        "text": sentence.strip()[:900],
                        "context": " ".join(before + ["[[" + sentence.strip()
                                                      [:900] + "]]"] + after)
                        [:2400],
                        "last_date": last_date, "last_event": last_event})
                # AFTER the hit, so a sentence's own date is in its text
                # and the hint is what came before it.
                dates = DATE.findall(sentence)
                if dates:
                    last_date = dates[-1]
                events = events_in(sentence)
                if events:
                    last_event = events[-1]
    return found


def _sid(page, text):
    return hashlib.sha1(f"{page}\n{text}".encode()).hexdigest()[:12]


# --- finding each fighter's own page ---------------------------------------

def fighter_candidates(name):
    """Titles a fighter's article may sit at, most specific first.

    "(fighter)" first: a plain "Michael Johnson" is a sprinter, and a plain
    name is often a disambiguation page. Redirects take care of accents.
    """
    name = str(name).strip()
    return [f"{name} (fighter)", f"{name} (mixed martial artist)", name]


def existing(titles):
    """{asked title: resolved title} for the titles that exist. No content."""
    found = {}
    for start in range(0, len(titles), EXISTS_BATCH):
        chunk = [t for t in titles[start:start + EXISTS_BATCH] if t]
        if not chunk:
            continue
        body = get({"action": "query", "prop": "info", "redirects": 1,
                    "titles": "|".join(chunk)})
        query = body.get("query", {})
        hop = {}
        for kind in ("normalized", "redirects"):
            for step in query.get(kind, []):
                hop[step["from"]] = step["to"]
        present = {p["title"] for p in query.get("pages", {}).values()
                   if "missing" not in p and "invalid" not in p}
        for title in chunk:
            target = title
            for _ in range(3):
                if target in hop:
                    target = hop[target]
            if target in present:
                found[title] = target
    return found


def fetch_pages(titles):
    """{title: wikitext}, following the API's continuation.

    A response that would pass the size cap comes back with some pages'
    content missing and a "continue" token. Ignoring it would drop pages
    at random and report them as fighters with no article.
    """
    out = {}
    for start in range(0, len(titles), CONTENT_BATCH):
        chunk = titles[start:start + CONTENT_BATCH]
        params = {"action": "query", "prop": "revisions", "rvslots": "main",
                  "rvprop": "content", "titles": "|".join(chunk)}
        extra = {}
        for _ in range(CONTENT_BATCH + 1):
            body = get({**params, **extra})
            for page in body.get("query", {}).get("pages", {}).values():
                revisions = page.get("revisions")
                if revisions and "slots" in revisions[0]:
                    out[page["title"]] = revisions[0]["slots"]["main"]["*"]
            if "continue" not in body:
                break
            extra = body["continue"]
    return out


def mma_page(text):
    return bool(re.search(r"mixed martial art", text or "", re.I))


def opponent_hits(text, opponents):
    """How many of this fighter's UFC opponents the page names, by surname.

    THE IDENTITY CHECK. A title that exists is not the right person - there
    are two Bruno Silvas and a sprinter called Michael Johnson. A page that
    names the people our fighter actually fought is about our fighter.
    """
    low = norm_name(text)
    count = 0
    for opponent in opponents:
        parts = norm_name(opponent).split()
        if parts and len(parts[-1]) > 2 and re.search(
                rf"\b{re.escape(parts[-1])}\b", low):
            count += 1
    return count


def resolve_fighters(fighters, opponents, verbose=True):
    """{fighter: (title, wikitext)} for every fighter whose page is found.

    Exists-check every candidate title (cheap), fetch the ones that exist,
    keep the candidate naming the most of the fighter's opponents, and search
    for whoever is left.
    """
    candidates = {f: fighter_candidates(f) for f in fighters}
    all_titles = sorted({t for group in candidates.values() for t in group})
    present = existing(all_titles)
    if verbose:
        print(f"  {len(present):,} of {len(all_titles):,} candidate titles "
              f"exist", flush=True)
    texts = fetch_pages(sorted(set(present.values())))

    resolved, missing = {}, []
    for fighter in fighters:
        best, best_score = None, 0
        for title in candidates[fighter]:
            target = present.get(title)
            text = texts.get(target)
            if not text or not mma_page(text):
                continue
            score = opponent_hits(text, opponents.get(fighter, ()))
            if score > best_score:
                best, best_score = (target, text), score
        if best:
            resolved[fighter] = best
        else:
            missing.append(fighter)
    if verbose:
        print(f"  {len(resolved):,} fighters matched by title; searching for "
              f"{len(missing):,}", flush=True)

    for n, fighter in enumerate(missing, 1):
        try:
            pages = search_pages(f"{fighter} mixed martial artist")
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
        best, best_score = None, 0
        for title, text in pages.items():
            if not mma_page(text):
                continue
            score = opponent_hits(text, opponents.get(fighter, ()))
            if score > best_score:
                best, best_score = (title, text), score
        # A search result must name at least two opponents: one shared
        # surname is too easy a coincidence among thousands of fighters.
        if best and best_score >= min(2, len(opponents.get(fighter, ()))):
            resolved[fighter] = best
        if verbose and n % 100 == 0:
            print(f"    searched {n:,}/{len(missing):,}", flush=True)
    return resolved


# --- the run ---------------------------------------------------------------

def records_for(source, subject, subject_date, title, wikitext):
    out = []
    for hit in hits(wikitext):
        out.append({"id": _sid(title, hit["text"]), "source": source,
                    "subject": subject, "subject_date": subject_date,
                    "page": title, **hit})
    return out


def write(records, pages):
    unique, seen = [], set()
    for record in records:
        if record["id"] not in seen:
            seen.add(record["id"])
            unique.append(record)
    with gzip.open(SENTENCES, "wt", encoding="utf-8") as handle:
        for record in unique:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pd.DataFrame(pages, columns=["kind", "subject", "subject_date", "page",
                                 "hits"]).to_csv(PAGES, index=False)
    return len(unique)


def load_archive(from_year):
    archive = pd.read_csv(ARCHIVE, usecols=["event_name", "date", "r_name",
                                            "b_name"], low_memory=False)
    archive["date"] = pd.to_datetime(archive["date"], errors="coerce")
    archive = archive.dropna(subset=["date"])
    return archive[archive.date.dt.year >= from_year]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--limit", type=int, default=0,
                        help="only N events and N fighters, for a smoke test")
    parser.add_argument("--from-year", type=int, default=FROM_YEAR)
    args = parser.parse_args(argv)

    archive = load_archive(args.from_year)
    events = (archive.drop_duplicates("event_name").sort_values("date")
              [["event_name", "date"]].reset_index(drop=True))
    opponents = {}
    for row in archive.itertuples():
        opponents.setdefault(row.r_name, set()).add(row.b_name)
        opponents.setdefault(row.b_name, set()).add(row.r_name)
    fighters = sorted(opponents)
    if args.limit:
        events, fighters = events.head(args.limit), fighters[:args.limit]

    print("=" * 74)
    print(f"INJURY SENTENCES: {len(events):,} EVENTS, {len(fighters):,} "
          f"FIGHTERS, {args.from_year} ON")
    print("=" * 74, flush=True)

    records, pages = [], []

    print("\n  event pages", flush=True)
    resolved_events = {}
    for start in range(0, len(events), 40):
        chunk = events.iloc[start:start + 40]
        try:
            resolved_events.update(resolve_all(chunk))
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            print(f"    chunk {start // 40 + 1} failed: {err}", flush=True)
    for row in events.itertuples():
        found = resolved_events.get(row.event_name)
        if not found:
            pages.append(("event", row.event_name, row.date.date(), "", -1))
            continue
        title, text = found
        new = records_for("event", row.event_name, str(row.date.date()),
                          title, text)
        records.extend(new)
        pages.append(("event", row.event_name, row.date.date(), title,
                      len(new)))
    print(f"    {len(resolved_events):,} of {len(events):,} events resolved, "
          f"{len(records):,} sentences", flush=True)
    write(records, pages)

    print("\n  fighter pages", flush=True)
    resolved = resolve_fighters(fighters, opponents)
    for fighter in fighters:
        found = resolved.get(fighter)
        if not found:
            pages.append(("fighter", fighter, "", "", -1))
            continue
        title, text = found
        new = records_for("fighter", fighter, "", title, text)
        records.extend(new)
        pages.append(("fighter", fighter, "", title, len(new)))
    total = write(records, pages)

    frame = pd.DataFrame(pages, columns=["kind", "subject", "subject_date",
                                         "page", "hits"])
    print(f"\n  {'events with a page':<34}"
          f"{(frame[frame.kind == 'event'].hits >= 0).sum():>8}"
          f"  of {len(events):,}")
    print(f"  {'fighters with a page':<34}"
          f"{(frame[frame.kind == 'fighter'].hits >= 0).sum():>8}"
          f"  of {len(fighters):,}")
    forms = pd.Series([r["form"] for r in records]).value_counts()
    print(f"  {'sentences and rows kept':<34}{total:>8}")
    for form, n in forms.items():
        print(f"    {form:<32}{n:>8}")
    print(f"\n  wrote {SENTENCES.relative_to(ENGINE.parent)} and "
          f"{PAGES.relative_to(ENGINE.parent)}")
    print("  This is text to be read, not labels. Nothing here says when")
    print("  an injury became known - injury_history.py decides that.")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
