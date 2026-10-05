"""Measured fight-night weights: the ground truth for True Fight Weight.

The official weight says everyone at 155 fights at 155. They do not: they
cut water to make the limit and put it back before the bout. Where an
athletic commission weighs fighters again on fight night - California's
does - Wikipedia's event pages often carry the numbers:

    ==Fight night weights==
    *'''Usman Nurmagomedov''': 154.8 to 173.2 pounds (12%)
    *'''Renato Moicano''': 154.8 to 180.6 pounds (17%)

This reads every such section on every event and year page the world
harvest discovers, into engine/data/fight_night_weights.csv:

    date, event, fighter, official_lbs, fight_night_lbs, gain_lbs,
    gain_pct, source, published_at

These are MEASUREMENTS, not the variable itself. A fight-night weight is
taken on fight day and usually published after the bout, so it cannot be
used as a pre-fight input for the fight it was taken at. It is used to
FIT an estimator of fight-night weight from pre-fight information (height,
reach, division, previous divisions, age, earlier measured regains), and to
check that estimator on fights it never saw.

published_at is WHEN THE NUMBER BECAME PUBLIC, the only date that matters
for point-in-time use: a measurement may inform a fight only if it was
published strictly before that fight. It is read from the section's own
citations, never earlier than the truth: the earliest cited article dated
on or after the event whose title says it is about fight-night weights
(Bellator 300: event October 7, MMA Fighting's "fight night weights"
article October 9); otherwise the LATEST article the section cites on or
after the event, because a results or salary article cited next to the
weights write-up is dated before the numbers were public; and event date
+ 7 days when the section cites nothing dated on or after the event - a
deliberate lag, not the event date. A citation dated only in prose
("October 9, 2023") is parsed; one that cannot be parsed is ignored.

    python engine/fight_night_weights.py      (needs network)
"""

import re
import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from fighter_profile import infobox
from world_bouts import parse_date, plain

OUT = ENGINE / "data" / "fight_night_weights.csv"

SECTION = re.compile(r"^(={2,4})\s*(?:[Ff]ight[- ][Nn]ight|[Ss]econd[- ]day|"
                     r"[Dd]ay[- ]of[- ](?:the[- ])?[Ff]ight)\s+[Ww]eights?.*?\1\s*$",
                     re.M)
NEXT_HEADING = re.compile(r"^={2,4}[^=].*?={2,4}\s*$", re.M)
NUMBER = r"(\d{2,3}(?:\.\d+)?)"
# "Name: 154.8 to 173.2 pounds (12%)", "Name – 154.8 lb → 173.2 lb",
# "Name: 154.8 lbs / 173.2 lbs"
LINE = re.compile(
    r"^\s*[*#]?\s*(?P<name>.+?)\s*[:–—-]\s*" + NUMBER +
    r"\s*(?:pounds|lbs?\.?)?\s*(?:to|→|->|/|–|—|-)\s*" + NUMBER +
    r"\s*(?:pounds|lbs?\.?)?", re.M)
HEADING = re.compile(r"^(={2,4})\s*(.*?)\s*\1\s*$", re.M)
# A {{Cite ...}} template and, inside one, "|date=" ("|access-date=" and
# "|archive-date=" do not match) and "|title=".
CITE = re.compile(r"\{\{\s*[Cc]ite\b")
CITE_DATE = re.compile(r"\|\s*date\s*=\s*([^|}\n]+)")
CITE_TITLE = re.compile(r"\|\s*title\s*=\s*([^|}\n]+)")
# A title that says the article carries the fight-night numbers.
FIGHT_NIGHT_TITLE = re.compile(r"fight[- ]night|fight[- ]day|second[- ]day|"
                               r"rehydrat|regain|day[- ]of[- ]fight", re.I)
# Without a dated citation, a measurement counts as public this many days
# after the event. CSAC releases the numbers after the bout and media write
# them up over the following days, so the event date itself is too early.
PUBLICATION_LAG_DAYS = 7
COLUMNS = ["date", "event", "fighter", "official_lbs", "fight_night_lbs",
           "gain_lbs", "gain_pct", "source", "published_at"]


def citations(body):
    """(date, title) for every {{Cite ...}} in the text whose date parses.

    The date is read in the forms Wikipedia uses (2023-10-09, October 9,
    2023, 9 October 2023); a date that cannot be parsed, or that is not a
    real day, drops the citation rather than aborting the harvest.
    """
    text = body or ""
    out = []
    for hit in CITE.finditer(text):
        depth, i = 0, hit.start()
        while i < len(text) - 1:                    # walk to the matching }}
            pair = text[i:i + 2]
            if pair == "{{":
                depth += 1
                i += 2
                continue
            if pair == "}}":
                depth -= 1
                i += 2
                if depth == 0:
                    break
                continue
            i += 1
        template = text[hit.start():i]
        date = CITE_DATE.search(template)
        if not date:
            continue
        when = pd.to_datetime(parse_date(date.group(1).strip()), errors="coerce")
        if pd.isna(when):
            continue
        title = CITE_TITLE.search(template)
        out.append((when, plain(title.group(1)) if title else ""))
    return out


def published_at(event_date, body=""):
    """The date the section's numbers became public, as YYYY-MM-DD.

    Among the section's citations dated on or after the event: the
    earliest whose title says fight-night weights; failing that the LATEST
    of them (a results article cited alongside the write-up is dated
    before the numbers were public, so the earliest would be too early);
    failing that event date + PUBLICATION_LAG_DAYS. A citation dated
    before the event (a weigh-in report) cannot have carried fight-night
    numbers and is ignored. None when the event date is unknown.
    """
    if not event_date:
        return None
    day = pd.Timestamp(event_date)
    after = [(d, t) for d, t in citations(body) if d >= day]
    night = [d for d, t in after if FIGHT_NIGHT_TITLE.search(t or "")]
    if night:
        when = min(night)
    elif after:
        when = max(d for d, _ in after)
    else:
        when = day + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    return when.strftime("%Y-%m-%d")


def with_published_at(frame):
    """The same frame with published_at filled by the rule where missing.

    Rows harvested before the column existed carry no citation date, so
    they get the conservative event date + PUBLICATION_LAG_DAYS.
    """
    frame = frame.copy()
    fallback = (pd.to_datetime(frame["date"], errors="coerce")
                + pd.Timedelta(days=PUBLICATION_LAG_DAYS)).dt.strftime("%Y-%m-%d")
    if "published_at" not in frame.columns:
        frame["published_at"] = fallback
    else:
        blank = frame["published_at"].isna() | (frame["published_at"].astype(str).str.strip() == "")
        frame.loc[blank, "published_at"] = fallback[blank]
    return frame


def sections(wikitext):
    """(start, body) for every fight-night-weights section on a page."""
    text = wikitext or ""
    for hit in SECTION.finditer(text):
        nxt = NEXT_HEADING.search(text, hit.end())
        yield hit.start(), text[hit.end(): nxt.start() if nxt else len(text)]


def _anchors(text):
    """(position, event, date) for event sections that carry an infobox."""
    out = []
    for h in HEADING.finditer(text):
        window = text[h.end():h.end() + 3000]
        if re.match(r"\s*\{\{\s*Infobox MMA event", window, re.I):
            out.append((h.start(), plain(h.group(2)),
                        parse_date(infobox(window).get("date", ""))))
    return out


def weights(wikitext, page=""):
    """Every measured official -> fight-night pair on a page."""
    text = wikitext or ""
    box = infobox(text)
    page_event = plain(box.get("name", "")) or page
    page_date = parse_date(box.get("date", ""))
    anchors = _anchors(text)
    rows = []
    for start, body in sections(text):
        event, date = page_event, page_date
        for pos, title, d in anchors:
            if pos < start and d:
                event, date = title, d
        public = published_at(date, body)     # from the refs, before they go
        body = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", body, flags=re.S)
        for m in LINE.finditer(body):
            name = plain(m.group("name")).strip(" *:")
            official, night = float(m.group(2)), float(m.group(3))
            if not name or not (100 <= official <= 280) or \
                    not (official * 0.9 <= night <= official * 1.35):
                continue      # not a weight pair, or a typo past belief
            rows.append({"date": date, "event": event, "fighter": name,
                         "official_lbs": official, "fight_night_lbs": night,
                         "gain_lbs": round(night - official, 1),
                         "gain_pct": round(100 * (night - official) / official, 2),
                         "source": page, "published_at": public})
    return rows


def main(argv=None):
    from harvest_injuries import fetch_pages
    from harvest_world import YEARS, discover

    print("=" * 74)
    print("MEASURED FIGHT-NIGHT WEIGHTS, EVERY EVENT PAGE")
    print("=" * 74, flush=True)
    titles = discover(list(YEARS))
    texts = fetch_pages(titles)
    rows, pages = [], 0
    for title in titles:
        found = weights(texts.get(title, ""), title)
        if found:
            pages += 1
            rows += found
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop_duplicates(["date", "fighter"]).sort_values(["date", "event"])
    frame = with_published_at(frame)
    frame.to_csv(OUT, index=False)
    print(f"\n  {len(frame):,} measured fighter-weights from {pages} pages")
    if len(frame):
        years = pd.to_datetime(frame["date"], errors="coerce").dt.year
        print("  by year: " + ", ".join(f"{int(y)}:{n}" for y, n in
                                        years.value_counts().sort_index().items()
                                        if y == y))
        print(f"  median regain {frame.gain_pct.median():.1f}%, "
              f"90th pct {frame.gain_pct.quantile(.9):.1f}%")
        ufc = frame["event"].astype(str).str.startswith("UFC")
        print(f"  UFC events: {ufc.sum():,} weights, "
              f"{frame.loc[ufc, 'event'].nunique()} events")
    print(f"\n  wrote {OUT.relative_to(ENGINE.parent)}")
    return 0 if len(frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
