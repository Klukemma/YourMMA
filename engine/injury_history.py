"""Injury history, dated by when it was KNOWN, and read only as of a date.

THE ONE RULE. A fight on date D may only see injuries that were public
before D. Everything else in this file exists to make that rule hold:

    withdrew before a bout         known the day before that bout
    hurt in a bout                 known from that bout on - usable for the
                                   fighter's NEXT fight, never that one
    "revealed he fought hurt"      the same: after that bout, not before it
    "later admitted", no date      flagged vague, left out unless asked for

The third is the trap. Post-fight disclosures are mostly made by losers, so
"fought injured" read as though it had been known beforehand would predict
results perfectly in testing and be worthless live. Recorded with its real
date it is still useful: a torn ACL revealed after a March loss is known
before that fighter's October return.

PIPELINE
    harvest_injuries.py        runner: every injury sentence on Wikipedia
    --batch K                  here: one slice of those sentences, as JSON,
                               for a reader to structure
    (readers)                  structured records: who, what, which bout,
                               before/at/after it (a workflow of agents)
    --build extracted.json     here: join names and bouts to the archive,
                               compute known_by, merge duplicates, and
                               write engine/data/injury_history.csv
    known_injuries(...)        the only way a model may read it

    python engine/injury_history.py --count
    python engine/injury_history.py --batch 3 --size 60
    python engine/injury_history.py --build extracted.json
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from name_resolution import norm_name

ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
SENTENCES = ENGINE / "data" / "injury_sentences.jsonl.gz"
HISTORY = ENGINE / "data" / "injury_history.csv"

KINDS = ("withdrawal", "fought_hurt", "in_fight", "surgery", "layoff",
         "other")
DISCLOSED = ("before", "at", "after", "unclear")
BODY_PARTS = ("knee", "leg", "foot_ankle", "hand_wrist", "arm_elbow",
              "shoulder", "back_neck", "head_concussion", "eye_orbital",
              "face", "rib_torso", "hip_groin", "cut", "illness",
              "weight_cut", "unknown", "other")
CERTAINTY = ("stated", "claimed", "speculative")

COLUMNS = ("fighter", "matched", "fighter_raw", "kind", "condition",
           "body_part", "surgery", "months_out", "certainty", "anchor_event",
           "anchor_opponent", "anchor_date", "anchor_precision",
           "anchor_source", "disclosed", "disclosure_date", "known_by",
           "vague", "conflict", "n_sources", "sentence_ids", "source",
           "page", "evidence")


# --- the sentences, for readers ---------------------------------------------

def load_sentences(path=SENTENCES):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


READER_FIELDS = ("id", "source", "subject", "subject_date", "section", "form",
                 "text", "context", "last_date", "last_event")


def batch(sentences, k, size):
    """Slice k of the sentences, carrying only what a reader needs."""
    chunk = sentences[k * size:(k + 1) * size]
    return [{f: s.get(f, "") for f in READER_FIELDS} for s in chunk]


# --- dates --------------------------------------------------------------------

def parse_date(text):
    """(start, end, precision) for "2016-07-09", "2016-07", "July 2016",
    "2016", "July 9, 2016"; or None.

    The WINDOW matters, not a point: "June 2016" could be any day of June,
    and the rule has to hold for the worst one.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        return (pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31), "year")
    month = re.fullmatch(r"(\d{4})-(\d{1,2})", text)
    if month:
        start = pd.Timestamp(int(month.group(1)), int(month.group(2)), 1)
        return (start, start + pd.offsets.MonthEnd(0), "month")
    if re.fullmatch(r"[A-Za-z]+\s+\d{4}", text):
        try:
            start = pd.Timestamp(text).replace(day=1)
        except (ValueError, TypeError):
            return None
        return (start, start + pd.offsets.MonthEnd(0), "month")
    try:
        day = pd.Timestamp(text).normalize()
    except (ValueError, TypeError):
        return None
    return (day, day, "day")


ONE_DAY = pd.Timedelta(days=1)


def known_by(disclosed, window, disclosure=None):
    """The first date the fact may be treated as public, or None.

    Conservative inside every window: a "before" fact dated only to a month
    is assumed known on the LAST day it could have been, never the first.
    A fight on D sees the fact only if known_by < D.
    """
    if window is None:
        return None
    start, end, precision = window
    if disclosed == "before":
        # Before a bout on day A: known by A - 1, so usable AT A. With only a
        # month or year, the latest day it could have been.
        when = end - ONE_DAY if precision == "day" else end
    elif disclosed in ("at", "after"):
        # Known from the bout on, never for it: usable only for fights after.
        when = end
    else:
        when = end
    if disclosure is not None:
        when = max(when, disclosure[1])
    return when


# --- the archive --------------------------------------------------------------

def _event_key(name):
    return re.sub(r"[^a-z0-9]+", " ", norm_name(name)).strip()


class Archive:
    """Names, events and bouts from the fight archive, for joining to."""

    def __init__(self, frame):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        self.frame = frame
        self.event_date = {}
        self.ufc_number = {}
        for name, date in frame.groupby("event_name")["date"].min().items():
            self.event_date[_event_key(name)] = date
            number = re.match(r"\s*UFC\s+(\d+)\b", str(name))
            if number:
                self.ufc_number[int(number.group(1))] = date
        self.by_norm = {}
        self.fights = {}
        for row in frame.itertuples():
            for me, them in ((row.r_name, row.b_name), (row.b_name, row.r_name)):
                self.by_norm.setdefault(norm_name(me), me)
                self.fights.setdefault(me, []).append((row.date, them,
                                                       row.event_name))
        self.card = {k: set(g.r_name) | set(g.b_name)
                     for k, g in frame.groupby("event_name")}
        self.surnames = {}
        for name in self.by_norm.values():
            parts = norm_name(name).split()
            if parts:
                self.surnames.setdefault(parts[-1], set()).add(name)

    def event(self, text):
        """Archive date of an event named in text, or None."""
        if not text:
            return None
        key = _event_key(text)
        if key in self.event_date:
            return self.event_date[key]
        number = re.fullmatch(r"ufc (\d+)", key)
        if number:
            return self.ufc_number.get(int(number.group(1)))
        return None

    def name(self, raw, *, near=None, pool=()):
        """(archive name, matched) for a name as written.

        Full name first. A surname alone is accepted only when exactly one
        person in `pool` (the page's subject and opponents, or the card)
        carries it, or else exactly one archive fighter active within two
        years of `near` does.
        """
        raw = str(raw or "").strip()
        if not raw:
            return raw, False
        hit = self.by_norm.get(norm_name(raw))
        if hit:
            return hit, True
        parts = norm_name(raw).split()
        if not parts:
            return raw, False
        surname = parts[-1]
        if len(parts) > 1:
            # A full name that is not in the archive is somebody else, even
            # when a card-mate shares the surname: "Anthony Smith" is not
            # "Leslie Smith". Only first-and-last agreement (middle names,
            # accents aside) is the same person.
            same = {p for p in pool
                    if norm_name(p).split()[:1] == parts[:1]
                    and norm_name(p).split()[-1:] == [surname]}
            if len(same) == 1:
                return same.pop(), True
            return raw, False
        in_pool = {p for p in pool if norm_name(p).split()[-1:] == [surname]}
        if len(in_pool) == 1:
            return in_pool.pop(), True
        candidates = self.surnames.get(surname, set())
        if near is not None and candidates:
            active = [c for c in candidates
                      if any(abs((d - near).days) <= 730
                             for d, _, _ in self.fights.get(c, ()))]
            if len(active) == 1:
                return active[0], True
        return raw, False

    def bout(self, fighter, opponent, near=None):
        """Date of the fighter's bout against this opponent, nearest `near`."""
        if not opponent or fighter not in self.fights:
            return None
        surname = norm_name(opponent).split()[-1:]
        dates = [d for d, them, _ in self.fights[fighter]
                 if norm_name(them).split()[-1:] == surname]
        if not dates:
            return None
        if near is None:
            return dates[0] if len(dates) == 1 else None
        return min(dates, key=lambda d: abs((d - near).days))


# --- building the history -----------------------------------------------------

def _anchor(record, sentence, archive, fighter):
    """(window, source, conflict) for the bout or date a record hangs on.

    Order of trust: the event page's own date; an event named in the record
    that the archive knows; a full date in the text; for bouts that were
    actually fought, the archive bout against the named opponent; then
    whatever partial date the reader gave. A withdrawn bout never happened,
    so it is never looked up by opponent - the pair often met later, and
    that later date would be wrong by months.
    """
    stated = parse_date(record.get("anchor_date"))
    conflict = False
    if sentence.get("source") == "event" and record.get("anchor_is_page_event",
                                                        True):
        day = parse_date(sentence.get("subject_date"))
        if day:
            if stated and stated[2] == "day" and abs((stated[0] - day[0]).days) > 3:
                conflict = True
            return day, "event_page", conflict
    date = archive.event(record.get("anchor_event"))
    if date is not None:
        if stated and stated[2] == "day" and abs((stated[0] - date).days) > 3:
            conflict = True
        return (date, date, "day"), "archive_event", conflict
    if stated and stated[2] == "day":
        return stated, "text_date", conflict
    if record.get("kind") in ("fought_hurt", "in_fight"):
        near = stated[0] if stated else None
        date = archive.bout(fighter, record.get("anchor_opponent"), near)
        if date is not None:
            return (date, date, "day"), "archive_bout", conflict
    if stated:
        return stated, "reader_date", conflict
    return None, "none", conflict


def build(extracted, sentences, archive):
    """The dated history frame from readers' records.

    `extracted` is a list of records, each carrying the sentence_id it was
    read from. Records that cannot be dated are kept with no known_by: they
    are counted, and never usable.
    """
    by_id = {s["id"]: s for s in sentences}
    rows = []
    for record in extracted:
        sentence = by_id.get(record.get("sentence_id"))
        if sentence is None or record.get("kind") not in KINDS:
            continue
        near = None
        stated = parse_date(record.get("anchor_date"))
        if stated:
            near = stated[0]
        elif parse_date(sentence.get("subject_date")):
            near = parse_date(sentence.get("subject_date"))[0]
        if sentence.get("source") == "event":
            pool = archive.card.get(sentence.get("subject"), set())
        else:
            subject = sentence.get("subject")
            pool = {subject} | {them for _, them, _ in
                                archive.fights.get(subject, ())}
        fighter, matched = archive.name(record.get("fighter"), near=near,
                                        pool=sorted(pool))
        window, source, conflict = _anchor(record, sentence, archive, fighter)
        disclosed = record.get("disclosed") if record.get("disclosed") in \
            DISCLOSED else "unclear"
        disclosure = parse_date(record.get("disclosure_date"))
        when = known_by(disclosed, window, disclosure)
        vague = (disclosed == "unclear" or bool(record.get("later_disclosure"))
                 and disclosure is None) or when is None
        rows.append({
            "fighter": fighter, "matched": matched,
            "fighter_raw": record.get("fighter", ""),
            "kind": record["kind"],
            "condition": record.get("condition", ""),
            "body_part": record.get("body_part") if record.get("body_part")
            in BODY_PARTS else "unknown",
            "surgery": bool(record.get("surgery")),
            "months_out": record.get("months_out"),
            "certainty": record.get("certainty") if record.get("certainty")
            in CERTAINTY else "stated",
            "anchor_event": record.get("anchor_event") or "",
            "anchor_opponent": record.get("anchor_opponent") or "",
            "anchor_date": window[0].date().isoformat() if window else "",
            "anchor_precision": window[2] if window else "",
            "anchor_source": source,
            "disclosed": disclosed,
            "disclosure_date": record.get("disclosure_date") or "",
            "known_by": when.date().isoformat() if when is not None else "",
            "vague": vague, "conflict": conflict,
            "n_sources": 1, "sentence_ids": sentence["id"],
            "source": sentence.get("source", ""),
            "page": sentence.get("page", ""),
            "evidence": sentence.get("text", "")[:600],
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return merge(frame)


MERGE_DAYS = 21


def merge(frame):
    """One row per injury, however many pages tell it.

    UFC 200's page and the fighter's own page both say he withdrew with a
    knee injury; counted twice, one withdrawal would look like two. Same
    fighter, kind and body part within MERGE_DAYS of each other are one
    fact. Its dating comes from the earliest precise source, unless another
    precise source disagrees about before/after - then the later wins.
    """
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_k"] = pd.to_datetime(frame["known_by"], errors="coerce")
    frame = frame.sort_values(["fighter", "kind", "body_part", "_k"])
    keep, current = [], None
    for row in frame.to_dict("records"):
        same = (current is not None and row["matched"] and current["matched"]
                and row["fighter"] == current["fighter"]
                and row["kind"] == current["kind"]
                and row["body_part"] == current["body_part"]
                and pd.notna(row["_k"]) and pd.notna(current["_k"])
                and (row["_k"] - current["_k"]).days <= MERGE_DAYS)
        if same:
            current["n_sources"] += 1
            current["sentence_ids"] += ";" + row["sentence_ids"]
            dating = ("known_by", "anchor_date", "anchor_precision",
                      "anchor_source", "disclosed", "vague", "_k")
            if current["vague"] and not row["vague"]:
                for field in dating:
                    current[field] = row[field]
            elif not row["vague"] and row["disclosed"] != current["disclosed"]:
                # Two sources, two stories about WHEN: one says it was known
                # before the bout, one that it came out after. The later is
                # kept - an injury wrongly treated as known early is exactly
                # the leak this file exists to prevent.
                for field in dating:
                    current[field] = row[field]
                current["conflict"] = True
            current["surgery"] = current["surgery"] or row["surgery"]
            continue
        if current is not None:
            keep.append(current)
        current = dict(row)
    keep.append(current)
    out = pd.DataFrame(keep).drop(columns=["_k"])
    return out[list(COLUMNS)].reset_index(drop=True)


def archive_stoppages(archive_frame, from_year=2010):
    """In-fight injuries the archive already records: doctor stoppages.

    The loser of a doctor's stoppage was stopped for an injury - most often
    a cut. Known from that bout on, like any in-fight injury.
    """
    frame = archive_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[(frame.date.dt.year >= from_year)
                  & frame["method"].astype(str).str.contains("Doctor",
                                                              case=False)]
    rows = []
    for row in frame.itertuples():
        loser = row.b_name if row.winner == row.r_name else row.r_name
        if row.winner not in (row.r_name, row.b_name):
            continue
        rows.append({
            "fighter": loser, "matched": True, "fighter_raw": loser,
            "kind": "in_fight", "condition": "doctor stoppage",
            "body_part": "unknown", "surgery": False, "months_out": None,
            "certainty": "stated", "anchor_event": row.event_name,
            "anchor_opponent": row.winner,
            "anchor_date": row.date.date().isoformat(),
            "anchor_precision": "day", "anchor_source": "archive_method",
            "disclosed": "at", "disclosure_date": "",
            "known_by": row.date.date().isoformat(), "vague": False,
            "conflict": False, "n_sources": 1, "sentence_ids": "",
            "source": "archive", "page": "",
            "evidence": f"{row.event_name}: {row.method}"})
    return pd.DataFrame(rows, columns=COLUMNS)


# --- reading it ---------------------------------------------------------------

def load(path=HISTORY):
    frame = pd.read_csv(path)
    frame["known_by"] = pd.to_datetime(frame["known_by"], errors="coerce")
    return frame


def known_injuries(history, fighter, as_of, *, include_vague=False):
    """Everything about this fighter that was public BEFORE as_of.

    The only accessor a model may use. Strictly before: a fact whose
    known_by is the fight's own date is a fact from that fight.
    """
    as_of = pd.Timestamp(as_of)
    known = pd.to_datetime(history["known_by"], errors="coerce")
    mask = (history["fighter"] == fighter) & history["matched"].astype(bool) \
        & known.notna() & (known < as_of)
    if not include_vague:
        mask &= ~history["vague"].astype(bool)
    return history[mask]


# --- command line -------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--ids", help="comma-separated sentence ids to print")
    parser.add_argument("--build", help="readers' records, as a JSON list")
    args = parser.parse_args(argv)

    sentences = load_sentences()
    if args.count:
        print(len(sentences))
        return 0
    if args.batch is not None:
        print(json.dumps(batch(sentences, args.batch, args.size),
                         ensure_ascii=False, indent=0))
        return 0
    if args.ids:
        wanted = set(args.ids.split(","))
        print(json.dumps([{f: s.get(f, "") for f in READER_FIELDS}
                          for s in sentences if s["id"] in wanted],
                         ensure_ascii=False, indent=0))
        return 0
    if args.build:
        extracted = json.loads(Path(args.build).read_text())
        frame = pd.read_csv(ARCHIVE, usecols=["event_name", "date", "r_name",
                                              "b_name", "method", "winner"],
                            low_memory=False)
        archive = Archive(frame)
        history = pd.concat([build(extracted, sentences, archive),
                             archive_stoppages(frame)], ignore_index=True)
        history.to_csv(HISTORY, index=False)
        usable = history[history.matched & ~history.vague.astype(bool)
                         & (history.known_by != "")]
        print(f"  wrote {HISTORY.relative_to(ENGINE.parent)}: "
              f"{len(history):,} injuries, {len(usable):,} dated and matched")
        print("  " + ", ".join(f"{k}: {n}" for k, n in
                               history.kind.value_counts().items()))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
