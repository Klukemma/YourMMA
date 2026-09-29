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
# Wikipedia asks that a bot identify itself and say where to complain. An
# agent string it does not like is answered with 403, which arrives looking
# exactly like every other HTTPError.
AGENT = ("YourMMA-research/1.0 "
         "(https://github.com/Klukemma/YourMMA; fight-change labelling)")
TIMEOUT = 30         # seconds per request
SEARCH_LIMIT = 3     # candidate articles considered per event
# TWO REQUESTS A SECOND, NOT FIVE. At 0.2s a 40-event probe was answered with
# 429 on most of its calls, and every 429 costs a 2s then a 4s backoff before
# it even raises - so hurrying is slower than going steadily, and a run that
# is being refused looks exactly like a run that is working. A runner's IP is
# shared and may already be hot before this starts.
PAUSE = 0.50         # seconds between calls
MIN_LABELS_USEFUL = 400   # below this a 4% effect cannot be told from zero
BATCH = 40           # titles per query; the API allows 50
RETRIES = 3          # on 429, which is what hammering it looks like

# The sentence shapes Wikipedia uses for a change. Deliberately several
# patterns rather than one clever regex: these are written by hundreds of
# different editors and a single pattern silently drops the variants it does
# not happen to match, which would show up here as "too thin" and send the
# whole project down the wrong road.
# ---------------------------------------------------------------------------
# REPLACEMENTS. Written against real sentences, not guessed. The first version
# of this table held two patterns written from memory, and a dump of what it
# failed to read on 2010-2016 pages turned up 80 plainly-replacement sentences
# in six phrasings it had never seen. Every pattern below is here because of a
# sentence in that dump; the tests quote them.
#
# A name: capitalised words, allowing the particles Brazilian and Dutch names
# carry. Descriptors before a name ("promotional newcomer", "former ice hockey
# player") are handled by reading the LAST name in a clause rather than by
# listing descriptors, which would never be complete.
_NM = (r"[A-Z][\w.'’-]+(?: (?:[A-Z][\w.'’-]+|de|da|dos|do|van|"
       r"von|del|la|le)){0,3}")
WITHDRAW = (r"(?:withdrew|withdraws|pulled out|pull out|had pulled out|"
            r"had to (?:withdraw|pull out)|was forced out|were forced out|"
            r"was forced to (?:withdraw|pull out)|was (?:forced )?to withdraw|"
            r"was removed|was pulled from|was scratched)")
_BUT_OUT = (r"(?:had\s+to\s+)?(?:withdrew|pulled out|pull out|was forced out|"
            r"was forced to (?:withdraw|pull out)|was removed)")
# "replaced by X", "replaced five days later by X"; the tail runs to the end of
# the clause and the replacement is the last name in it.
BY_TAIL = (r"replaced(?:\s+[\w-]+){0,3}?\s+by\s+(?P<tail>[^,.;:()\[\]]+?)"
           r"(?=\s+(?:and|while|who|after|in|at|for|on|as|when|due|but)\b"
           r"|[,.;:()]|$)")

REPLACEMENT_PATTERNS = (
    # "Ortega was replaced by Andre Fili, while Green was replaced by..."
    re.compile(rf"(?P<out>{_NM})\s+(?:himself\s+|herself\s+)?(?:was|were|"
               rf"would be|had been|will be)\s+" + BY_TAIL),
    # "Carlo Prater stepped in as a replacement for Bahadurzada"
    re.compile(rf"(?P<in>{_NM})\s+(?:stepped|steps|step)\s+(?:in|up)\s+"
               rf"(?:as\s+(?:a|the)\s+replacement\s+)?(?:for|to replace)\s+"
               rf"(?P<out>{_NM})"),
    # "Brown replaced Jones"
    re.compile(rf"(?P<in>{_NM})\s+replaced\s+(?P<out>{_NM})"),
    # "Cane was forced out of the bout with an injury and replaced by ..."
    re.compile(rf"(?P<out>{_NM})\s+(?:himself\s+|herself\s+)?{WITHDRAW}"
               rf"[^.]{{0,160}}?" + BY_TAIL),
    # "Ray Borg was expected to face Fredy Serrano at the event, but pulled
    # out on July 21 due to injury and was replaced by Ryan Benoit."
    re.compile(rf"(?P<out>{_NM})\s+was\s+(?:originally\s+|initially\s+)?"
               rf"(?:expected|scheduled|set|booked)\s+to\s+(?:face|take on|"
               rf"fight|meet)\s+{_NM}[^.;]*?,?\s*but\s+{_BUT_OUT}"
               rf"[^.]{{0,160}}?" + BY_TAIL),
    # "Harris was forced out of the bout and Lineker faced promotional
    # newcomer José Maria Tomé."
    re.compile(rf"(?P<out>{_NM})\s+{WITHDRAW}[^.]{{0,120}}?\s+and\s+{_NM}\s+"
               rf"(?:instead\s+)?faced\s+(?P<tail>[^,.;:()]+?)(?=[,.;:()]|$)"),
)

# The replacement named in a LATER sentence than the withdrawal. "Nordine
# Taleb was expected to face Alan Jouban at the event, but pulled out on June
# 7 due to injury. He was replaced by promotional newcomer Belal Muhammad."
# These resolve to the most recent withdrawal on the page; the card join
# then drops anything that does not land on a real fighter.
PRONOUN_PATTERNS = (
    re.compile(r"^(?:He|She)\s+(?:was|had been)\s+" + BY_TAIL),
    re.compile(r"^(?:His|Her)\s+replacement\s+(?:was|is|would be)\s+"
               r"(?P<tail>[^,.;:()]+?)(?=[,.;:()]|$)"),
    re.compile(rf"(?P<in>{_NM})\s+(?:would|will)\s+replace\s+(?:him|her)\b"),
    re.compile(rf"(?P<in>{_NM})\s+(?:agreed\s+to\s+|would\s+|will\s+)?"
               rf"(?:stepped|step)\s+(?:up|in)\s+as\s+(?:a|his|her|the)\s+"
               rf"replacement\b(?!\s+for)"),
)

# Who withdrew, remembered for the pronoun forms above.
WITHDRAWAL = (
    re.compile(rf"(?P<out>{_NM})\s+was\s+(?:originally\s+|initially\s+)?"
               rf"(?:expected|scheduled|set|booked)\s+to\s+(?:face|take on|"
               rf"fight|meet)\s+{_NM}[^.;]*?,?\s*but\s+{_BUT_OUT}"),
    re.compile(rf"(?P<out>{_NM})\s+(?:himself\s+|herself\s+)?{WITHDRAW}"),
)


def _last_name(tail):
    """The last name in a clause: "promotional newcomer Shane Campbell" ->
    "Shane Campbell". None when the clause names nobody."""
    runs = re.findall(_NM, str(tail or ""))
    return runs[-1] if runs else None


# "on nine days' notice", "on two weeks notice", and bare "on short notice"
# with no period given at all - which is the commonest phrasing of the three
# and was missed by the first version of this regex, because it demanded a
# unit that sentence does not have.
# "on just 4 days notice", "on under one week's notice": the qualifier sits
# between "on" and the number and used to hide it. "under" and "less than"
# are upper bounds and are recorded as the bound.
NOTICE = re.compile(r"on\s+(?:(?:just|only|under|less\s+than|about|roughly|"
                    r"barely)\s+)?(?:(?P<n>\d+|one|two|three|four|five|six|seven|"
                    r"eight|nine|ten|eleven|twelve)\s+)?(?P<unit>day|week)(?:'s|s'|s|\u2019s|s\u2019)?\s*"
                    r"(?:\s|-)?notice", re.I)
SHORT_NOTICE = re.compile(r"\b(?:short|late)(?:\s|-)notice\b", re.I)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12}


def get(params):
    """One API call, paced, and retried when the API says to slow down.

    THE PACING IS NOT POLITENESS, IT IS CORRECTNESS. The first version slept
    only when a candidate's date did not match, which meant a run that
    resolved nothing fired 120 requests in four seconds and was answered with
    429 on 35 of 40 events. That came out as "FETCH FAILED HTTPError" and then
    as "TOO THIN", which reads like a fact about Wikipedia and is a fact about
    this function.
    """
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(RETRIES):
        try:
            time.sleep(PAUSE)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as err:
            # A status is diagnosable; "HTTPError" is not. 429 means back off,
            # 403 means the agent string was refused, 404 means wrong title -
            # three completely different problems behind one word.
            if err.code == 429 and attempt < RETRIES - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise urllib.error.HTTPError(
                err.url, err.code, f"HTTP {err.code} from the Wikipedia API",
                err.headers, None)
    raise RuntimeError("unreachable")


def title_candidates(name):
    """Article titles this event might live at, best guess first.

    Most events need no search at all: "UFC 182: Jones vs Cormier" is at
    "UFC 182", and a Fight Night is usually at its own full name give or take
    a full stop after "vs". Guessing first and searching only for the misses
    turns 755 searches into about twenty batched lookups.
    """
    name = str(name).strip()
    out = [name]
    numbered = re.match(r"^(UFC\s+\d+)\b", name)
    if numbered:
        out.append(numbered.group(1))
    if " vs " in name:
        out.append(name.replace(" vs ", " vs. "))
    elif " vs. " in name:
        out.append(name.replace(" vs. ", " vs "))
    seen, unique = set(), []
    for title in out:
        if title and title not in seen:
            seen.add(title)
            unique.append(title)
    return unique


def fetch_many(titles):
    """{title: wikitext} for up to BATCH titles in one call.

    Titles that do not exist come back under "missing" and are simply absent
    here, which is the answer to "is this the right title" for free.
    """
    found = {}
    for start in range(0, len(titles), BATCH):
        chunk = [t for t in titles[start:start + BATCH] if t]
        if not chunk:
            continue
        body = get({"action": "query", "prop": "revisions", "rvslots": "main",
                    "rvprop": "content", "redirects": 1,
                    "titles": "|".join(chunk)})
        query = body.get("query", {})
        # A redirect or a normalisation means the page came back under a
        # different name than we asked for; both maps are needed to put the
        # text back against the title the caller knows.
        alias = {}
        for kind in ("normalized", "redirects"):
            for hop in query.get(kind, []):
                alias[hop["to"]] = alias.get(hop["from"], hop["from"])
        for page in query.get("pages", {}).values():
            if "missing" in page or not page.get("revisions"):
                continue
            text = page["revisions"][0]["slots"]["main"]["*"]
            title = page["title"]
            found[title] = text
            if title in alias:
                found[alias[title]] = text
    return found


def wikitext(title):
    body = get({"action": "parse", "page": title, "prop": "wikitext",
                "redirects": 1})
    if "error" in body:
        return None
    return body["parse"]["wikitext"]["*"]


# To END OF LINE, not to the next pipe. Infobox dates are very often written
# {{Start date|2024|4|13}}, and a pattern that stops at the first pipe reads
# that as "{{Start date" and gives up - silently, on what is probably the
# commonest form there is.
INFOBOX_DATE = re.compile(r"\|\s*date\s*=\s*([^\n]+)", re.I)


def infobox_date(text):
    """The date an event article says it happened, or None.

    The article is the authority on its own date. Everything else here -
    search ranking, a title that looks right - is a guess, and this is what
    turns a guess into a match or throws it away.
    """
    hit = INFOBOX_DATE.search(text or "")
    if not hit:
        return None
    raw = hit.group(1)
    template = re.search(r"\{\{\s*[Ss]tart date[^}]*\}\}", raw)
    if template:
        # {{Start date|2024|4|13|...}} - the first three numbers are the date,
        # anything after them is a time or a flag.
        numbers = re.findall(r"\d+", template.group(0))
        if len(numbers) >= 3:
            try:
                return pd.Timestamp(year=int(numbers[0]), month=int(numbers[1]),
                                    day=int(numbers[2])).date()
            except (ValueError, TypeError):
                return None
    # Plain prose: "April 13, 2024", possibly followed by another field on the
    # same line, and possibly wrapped in link or italic markup.
    raw = re.sub(r"[\[\]']", "", raw).split("|")[0].strip()
    try:
        return pd.Timestamp(raw).date()
    except (ValueError, TypeError):
        return None


def search_pages(name):
    """{title: wikitext} for the articles search thinks match this name.

    ONE call, not two. `generator=search` feeds the search results straight
    into the content query, where the old pair of calls searched for titles
    and then fetched them separately - twice the requests at the exact point
    in the run where requests were already being refused.
    """
    body = get({"action": "query", "generator": "search", "gsrsearch": name,
                "gsrlimit": SEARCH_LIMIT, "prop": "revisions",
                "rvslots": "main", "rvprop": "content", "redirects": 1})
    found = {}
    for page in body.get("query", {}).get("pages", {}).values():
        if "missing" in page or not page.get("revisions"):
            continue
        found[page["title"]] = page["revisions"][0]["slots"]["main"]["*"]
    return found


def resolve_all(sample):
    """{event name: (title, wikitext)} for a frame of events.

    GUESS, BATCH, VERIFY, THEN SEARCH THE REST. The title comes from the event
    name where that works and from search where it does not; the DATE always
    comes from the article's own infobox, which cannot be wrong about when its
    own event happened. A candidate whose date disagrees with ours is the
    wrong article and is dropped rather than used - an earlier version skipped
    that check and confidently dated UFC 16, a 1998 card, to 2009.
    """
    wanted = {row.event_name: row.date.date() for row in sample.itertuples()}
    candidates = {name: title_candidates(name) for name in wanted}
    pages = fetch_many(sorted({t for group in candidates.values()
                               for t in group}))

    resolved, missing = {}, []
    for name, when in wanted.items():
        for title in candidates[name]:
            text = pages.get(title)
            if text and _dated(text, when):
                resolved[name] = (title, text)
                break
        else:
            missing.append(name)

    for name in missing:
        try:
            pages = search_pages(name)
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
        for title, text in pages.items():
            if _dated(text, wanted[name]):
                resolved[name] = (title, text)
                break
    return resolved


def _dated(text, when):
    theirs = infobox_date(text)
    return bool(theirs) and abs((theirs - when).days) <= 1


# Capitalised because they start a sentence, not because they are names.
# They would fall out at the card join anyway, but only after being counted
# as "found, unmatched" - which is a statistic this probe reports and which
# would then overstate how many labels were lost to name matching.
NOT_NAMES = {"he", "she", "they", "it", "his", "her", "their", "however",
             "the", "this", "that", "both", "each", "after", "before",
             "then", "also", "later", "subsequently", "eventually"}


def _is_name(name):
    words = str(name or "").split()
    return bool(words) and words[0].lower() not in NOT_NAMES


def _clean(name):
    """Trim the sentence's punctuation off a captured name.

    The name pattern has to allow a full stop so that "Jr." and initials
    survive, which means a name at the end of a sentence swallows the
    sentence's own full stop and comes back as "Carl Brown." - close enough to
    look right in a printout and wrong enough to never join to a fighter.
    Caught by a test, not by reading the output.
    """
    return str(name or "").strip().strip(".,;:!?'\u2019\"").strip()


# {{convert|159|lb|kg}} - how Wikipedia writes almost every weight. Rewritten
# to "159 pounds" BEFORE templates are stripped, because stripping first would
# delete the number and leave "weighed in at , three pounds over", which no
# pattern can read and which would silently zero the missed-weight count.
CONVERT = re.compile(r"\{\{\s*[Cc]onvert\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*lbs?\b[^{}]*\}\}")


def _prose(text):
    """Readable sentences from wikitext, with tables and references gone.

    The prose lives outside the bout tables; the templates are dropped so a
    fighter's name in a results table is never read as a replacement or a
    missed weight. References go too - a citation's title is somebody else's
    sentence and must not be parsed as this page's.
    """
    prose = CONVERT.sub(r"\1 pounds", text or "")
    prose = re.sub(r"\{\{[^{}]*\}\}", " ", prose)
    prose = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", prose)
    prose = re.sub(r"<ref[^>]*>.*?</ref>", " ", prose, flags=re.S)
    prose = re.sub(r"<ref[^>]*/>", " ", prose)
    prose = re.sub(r"<[^>]+>", " ", prose)
    return prose


# Where a sentence must end even without a full stop: a blank line (a new
# paragraph), or a line that opens with wiki structure - an infobox field, a
# list item, a heading, a table row. A lone newline anywhere else is a hard
# wrap inside a sentence and is read as a space.
_BREAK = re.compile(r"\n\s*\n|\n(?=\s*[|*#=!{}:;])")


def _sentences(text):
    """Sentences, never crossing a paragraph or a line of wiki structure.

    Splitting only on ". Capital" let a clause run on into the next
    paragraph: at UFC 131 "...was replaced by Chris Weidman" has no full
    stop, so "replaced by" read on into "Rani Yahya was scheduled face Dustin
    Poirier" and labelled Poirier the replacement. Five labels were wrong that
    way. Splitting on EVERY newline fixed them and broke three right ones,
    because some pages hard-wrap a sentence ("withdrew due to an ankle
    \ninjury and was replaced by Bobby Green") - so only a paragraph break or
    a structural line ends a sentence. Found by reading harvested labels
    against their evidence.
    """
    out = []
    for block in _BREAK.split(_prose(text)):
        block = re.sub(r"\s*\n\s*", " ", block)
        out.extend(part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", block)
                   if part.strip())
    return out


# --- missed weight -------------------------------------------------------
# "At the weigh-ins, Mackenzie Dern weighed in at 117 pounds, one pound over
# the strawweight non-title fight limit." Pre-fight by construction - the
# weigh-in is the day before - and the market has it too, which is exactly
# why it is worth measuring: it says whether the line prices a bad cut
# correctly.
_NAME = r"(?P<name>[A-Z][\w.'\u2019-]+(?: [A-Z][\w.'\u2019-]+){0,3})"
_LBS = r"(?:[Pp]ounds?|lbs?\.?)"
WEIGH_IN = re.compile(
    _NAME + r"\s+(?:weighed|came)\s+in\s+at\s+"
    r"(?P<weight>\d+(?:\.\d+)?)\s*" + _LBS +
    r"(?:\s*\([^)]*\))?\s*,?\s*"
    r"(?P<over>[\w.\s-]{1,30}?)\s+" + _LBS + r"\s+over\b")
MISSED_WEIGHT = re.compile(_NAME + r"\s+missed\s+weight\b")
# The rest are from the dump of unread 2010-2016 sentences, each one real:
#   "Thiago Silva came in 2 lb over the 206 lb weight limit"
#   "Villefort ... received a portion of Burrell's purse"
#   "Nurmagomedov was given two hours to cut to the lightweight maximum"
#   "Melvin Guillard was the only one of 20 fighters who missed weight"
CAME_IN_OVER = re.compile(_NAME + r"\s+came\s+in\s+(?P<over>[\w.\s-]{1,30}?)"
                          r"\s+" + _LBS + r"\s+over\b")
PURSE = re.compile(r"(?:received|was awarded|was given|went to)\s+"
                   r"(?:a\s+(?:portion|percentage)|\d+(?:\.\d+)?\s*"
                   r"(?:%|percent))\s+of\s+" + _NAME +
                   r"(?:'s|\u2019s)\s+(?:fight\s+)?purse")
HOURS_TO_CUT = re.compile(_NAME + r"\s+was\s+given\s+(?:\w+\s+){0,2}hours?"
                          r"\s+to\s+(?:cut|make)")
ONLY_ONE = re.compile(_NAME + r"\s+was\s+the\s+only\s+(?:one|fighter)\b"
                      r"[^.]*?\bmissed\s+weight")


def _pounds(phrase):
    """"three", "2.5", "half a", "two and a half" -> pounds, or None.

    None rather than a guess: a missed weight with an unreadable margin is
    still a missed weight, and is recorded as one with the margin blank.
    """
    words = str(phrase or "").lower().replace("-", " ").split()
    if not words:
        return None
    text = " ".join(words)
    if text in ("half a", "a half", "half"):
        return 0.5
    half = 0.5 if text.endswith("and a half") else 0.0
    head = text.replace("and a half", "").strip().split()
    if not head:
        return None
    token = head[-1]
    if token == "a":
        return 1.0 + half
    try:
        return float(token) + half
    except ValueError:
        value = WORDS.get(token)
        return value + half if value is not None else None


def weigh_ins_in(text):
    """Every fighter this page says missed weight, and by how much."""
    rows, seen = [], set()

    def add(name, weighed, over, sentence):
        name = _clean(name)
        if _is_name(name) and name not in seen:
            seen.add(name)
            rows.append({"fighter": name, "weighed_lbs": weighed,
                         "over_by_lbs": over, "sentence": sentence[:200]})

    for sentence in _sentences(text):
        clean = sentence.strip()
        for hit in WEIGH_IN.finditer(clean):
            add(hit.group("name"), float(hit.group("weight")),
                _pounds(hit.group("over")), clean)
        for hit in CAME_IN_OVER.finditer(clean):
            add(hit.group("name"), None, _pounds(hit.group("over")), clean)
        for pattern in (MISSED_WEIGHT, PURSE, HOURS_TO_CUT, ONLY_ONE):
            for hit in pattern.finditer(clean):
                add(hit.group("name"), None, None, clean)
    return rows


def _notice(sentence):
    notice = NOTICE.search(sentence)
    if notice:
        raw = notice.group("n")
        count = WORDS.get((raw or "").lower(), None)
        if count is None and raw and raw.isdigit():
            count = int(raw)
        if count is not None:
            return count * (7 if notice.group("unit").lower() == "week" else 1)
        return 0                    # a unit but no number
    if SHORT_NOTICE.search(sentence):
        return 0                    # said to be short, no period given
    return None


def changes_in(text):
    """Every (stepped in, replaced, days notice) this page states."""
    rows, seen = [], set()
    last_out = None
    for sentence in _sentences(text):
        clean = sentence.strip()
        pairs = []
        for pattern in REPLACEMENT_PATTERNS:
            for hit in pattern.finditer(clean):
                groups = hit.groupdict()
                stepped_in = (_clean(groups["in"]) if groups.get("in")
                              else _clean(_last_name(groups.get("tail"))))
                pairs.append((stepped_in, _clean(groups["out"])))
        # Pronoun forms name the replacement here and the withdrawal in an
        # earlier sentence, so they read `last_out` BEFORE it is updated.
        if last_out:
            for pattern in PRONOUN_PATTERNS:
                for hit in pattern.finditer(clean):
                    groups = hit.groupdict()
                    stepped_in = (_clean(groups["in"]) if groups.get("in")
                                  else _clean(_last_name(groups.get("tail"))))
                    pairs.append((stepped_in, last_out))
        for pattern in WITHDRAWAL:
            hit = pattern.search(clean)
            if hit and _is_name(_clean(hit.group("out"))):
                last_out = _clean(hit.group("out"))
                break

        days = _notice(clean)
        for stepped_in, replaced in pairs:
            if not _is_name(stepped_in) or not _is_name(replaced) \
                    or stepped_in == replaced:
                continue
            if (stepped_in, replaced) in seen:
                continue
            seen.add((stepped_in, replaced))
            rows.append({"stepped_in": stepped_in,
                         "replaced": replaced,
                         "days_notice": days,
                         "sentence": clean[:200]})
    return rows


# Words that mark a sentence as PROBABLY about a card change or a weigh-in.
# Deliberately broad: the point is to catch what the patterns miss, so a
# sentence caught here and read by nothing is exactly what gets printed.
CANDIDATE = re.compile(r"replac|withdr|stepped in|step in|forced out|pulled "
                       r"out|pulled from|scratched|missed weight|make weight|"
                       r"made weight|weighed in|came in at|pounds over|"
                       r"lbs? over|catchweight", re.I)


def unparsed(text):
    """Candidate sentences that no pattern read. The parser's blind spots."""
    read = {r["sentence"][:120] for r in changes_in(text)} | \
           {r["sentence"][:120] for r in weigh_ins_in(text)}
    out = []
    for sentence in _sentences(text):
        clean = sentence.strip()
        if CANDIDATE.search(clean) and clean[:120] not in read:
            out.append(clean[:260])
    return out


def dump_unparsed(from_year, to_year, sample, seed=0):
    """Print what the parser missed, from real pages, for a span of years.

    The fix for low recall must be fitted to the sentences Wikipedia actually
    contains. Guessing at phrasings from memory is how the first version
    missed "was forced out of the bout" and "stepped in to replace" - both
    common, both plainly replacements, and both invisible to it.
    """
    import random
    archive = pd.read_csv(ARCHIVE, usecols=["event_name", "date"],
                          low_memory=False)
    archive["date"] = pd.to_datetime(archive["date"], errors="coerce")
    events = (archive.dropna(subset=["date"]).drop_duplicates("event_name"))
    events = events[events.date.dt.year.between(from_year, to_year)]
    rows = list(events.itertuples(index=False))
    random.Random(seed).shuffle(rows)
    block = pd.DataFrame(rows[:sample], columns=events.columns)
    resolved = resolve_all(block)
    print(f"  {len(resolved)} of {len(block)} events resolved, "
          f"{from_year}-{to_year}")
    total = 0
    for name, (title, text) in sorted(resolved.items()):
        misses = unparsed(text)
        parsed = len(changes_in(text)) + len(weigh_ins_in(text))
        print(f"\n  == {title}  (parsed {parsed}, missed {len(misses)})")
        for sentence in misses:
            total += 1
            print(f"     - {sentence}")
    print(f"\n  {total} candidate sentences read by nothing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--sample", type=int, default=40,
                        help="how many events to actually fetch")
    parser.add_argument("--unparsed", action="store_true",
                        help="print candidate sentences the parser missed")
    parser.add_argument("--from-year", type=int, default=2010)
    parser.add_argument("--to-year", type=int, default=2015)
    args = parser.parse_args(argv)

    if args.unparsed:
        dump_unparsed(args.from_year, args.to_year, args.sample)
        return 0

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
        search_pages("UFC 300")
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        print(f"\n  UNREACHABLE: {type(err).__name__}: {err}")
        print("  Wikipedia cannot be read from here. This container's egress")
        print("  proxy refuses it; run the check-fight-changes workflow mode,")
        print("  where the runner has open internet.")
        return 1

    total, with_changes, quantified, failures, surname_only = 0, 0, 0, 0, 0
    examples = []
    try:
        resolved = resolve_all(sample)
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        print(f"\n  FETCHING FAILED: {err}")
        return 1

    unresolved = []
    for event in sample.itertuples():
        when = event.date.date()
        title, text = resolved.get(event.event_name, (None, None))
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
    if projected < MIN_LABELS_USEFUL:
        print("  TOO THIN at this parser's recall. A hand-typed 4% cannot be")
        print("  told from 0% at this N. Note this is a FLOOR: more sentence")
        print("  patterns would find more, so it is a fact about the parser")
        print("  as much as about Wikipedia.")
    elif projected < 1000:
        print("  MARGINAL. Enough to rule out a large effect, not enough to")
        print("  confirm a 4% one. Report the interval, and do not ship a")
        print("  weight whose interval spans zero.")
    else:
        print("  ENOUGH. A 4% effect would be distinguishable from zero.")

    # THE BINDING CONSTRAINT IS NOT THE COUNT. The hook being fitted is
    # `short_notice`, and a replacement is not the same thing as short notice.
    # One of the examples above reads "Magomedov pulled out of the fight in
    # early March ... and was replaced by Rustam Khabilov" - for a fight on
    # 8 May. That replacement had a nine-week camp. Treating him as short
    # notice would be labelling noise and calling it evidence.
    share = quantified / total if total else 0.0
    print(f"\n  OF THOSE, {quantified} of {total} say how much notice was "
          f"given ({share:.0%}).")
    if share < 0.25:
        print("  This is the real blocker, and it is worse than the count. A")
        print("  replacement announced in early March for a fight in May had")
        print("  a full camp; calling that short notice would label noise and")
        print("  call it evidence. What Wikipedia reliably supports is a")
        print("  WEAKER label - 'this fighter was a late replacement' - which")
        print("  is a different question from the one 'short_notice' asks.")
        print("  Fit that one, or fit nothing.")

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
