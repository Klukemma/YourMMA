"""UFC results the Kaggle archive has not caught up to, from Wikipedia.

    python engine/sync_wiki.py sync            # fetch, bridge, write (network)
    python engine/sync_wiki.py sync --dry-run
    python engine/sync_wiki.py measure         # replay the archive's last events (offline)

The archive is synced from a Kaggle dataset (sync_kaggle.py) scraped from
UFCStats. When that dataset stops updating, the archive stops with it: on
2026-10-22 it ended at 2026-08-08, ten events behind. UFCStats itself answers
a script with a JavaScript challenge (the probe's "fixtures" in
tests/fixtures/ufcstats are that challenge page, not the pages it asked for),
and this project does not get around a bot wall.

Wikipedia's event pages carry every bout's result. This adds the bouts after
the archive's last date as PROVISIONAL rows:

  - What a result row needs and nothing more: the fighters, the winner,
    method, round, time, scheduled rounds, title flag and division. Those
    are pending_rows' CONTRACT columns (the only ones a bout's own row feeds
    into its features) plus the outcome. Per-bout statistics (strikes,
    takedowns, control time) stay empty, as on the 145 bouts of 2025 whose
    statistics never arrived. career_stats counts such a bout toward a
    fighter's record, finishes and cage time, and leaves it out of every
    rate.
  - Each fighter under the name and id the archive already holds, so the
    rating continues their own TrueSkill series (ratings.extend keys on the
    name). NameBridge says how a Wikipedia spelling is matched. A fighter it
    cannot match is a UFC debutant, given a provisional id of their own
    ('wiki_<hash of the name>', the same on every card they fight on).
  - fight_id and event_id are 'wiki_<hash>', marking the rows provisional.
    Every run rebuilds them from scratch, and sync_kaggle.py drops them
    before it merges, so the day Kaggle catches up, its rows (with
    statistics) replace these.

CORNERS. Wikipedia lists the winner first and names no corner. Filed as
listed, every provisional bout would be a red-corner win, in the most recent
(most heavily weighted) rows and inside the calibration window. The red
corner is the matchmaker's favoured or ranked fighter (corner_swap.py), so
red here is the fighter with the higher pre-fight rating (mmr_pre, known
before the bout).

MEASURED (`measure`): the archive cut before its last N events, those events
rebuilt from the Wikipedia snapshot (world_bouts.csv.gz) with aliases learned
only from earlier events, and compared bout by bout with the archive's rows.

  last 8 events, 2026-06-14 to 2026-08-08, 96 bouts
    all 96 found; winner, method, round, time and division 96 of 96;
    scheduled rounds and title flag 95 of 96 (UFC Freedom 250: the snapshot
    keeps no notes, which the live pages carry);
    157 fighters on their own archive id, the other 35 correctly debutants;
    every pre-fight mu and sigma equal to the archive's (max diff 7e-15)
  last 40 events, from 2025-09-13, 458 bouts paired
    winner, method and round 458 of 458; one veteran missed (Wikipedia's
    "Osman Diaz" is the archive's "Ozzy Diaz", and no earlier bout taught
    the alias, so his rating would restart); two Australian cards not
    paired, because Wikipedia dates an event by its local day and UFCStats
    by the American one (a provisional row carries the later, local date);
    Road to UFC cards are not UFC event pages and are left out
  corners
    the higher-rated fighter was the archive's red corner in 65 of 96;
    red won 54% of the rebuilt rows against the archive's 57%
"""

import argparse
import hashlib
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import pending_rows
from fighter_profile import infobox
from name_resolution import canonical_winners, norm_name
from ratings import extend
from transform import carry_forward_records
from world_bouts import _result, parse_date, plain, split_params, templates

DATA = ENGINE / "data"
ARCHIVE = DATA / "UFC_with_mmr_rebuilt_dedup.csv"
WORLD_BOUTS = DATA / "world_bouts.csv.gz"
PROVISIONAL = "wiki_"

# A UFC event page, by title. Contender Series, Road to UFC episodes, The
# Ultimate Fighter and the grappling cards are not in the archive.
EVENT_TITLE = re.compile(r"^UFC\b(?!.*\b(?:BJJ|Fight Pass Invitational)\b)")


def is_provisional(frame):
    """Rows this module wrote, by their fight_id."""
    if "fight_id" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["fight_id"].astype(str).str.startswith(PROVISIONAL)


def debut_id(name):
    """A debutant's provisional id, the same wherever the name appears."""
    return _hash(norm_name(name))


def _hash(*parts):
    return PROVISIONAL + hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def ascii_name(name):
    """'Fabià Sintes (c)' -> 'Fabia Sintes': the archive spells names as
    UFCStats does, without accents or title tags."""
    text = re.sub(r"\s*\((?:c|ic|uc|ac)\)\s*$", "", str(name or ""), flags=re.I)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


# --- reading an event page ---------------------------------------------------

_TITLE_TAG = re.compile(r"\((?:c|ic)\)", re.I)
_BMF = re.compile(r"\bBMF\b|symbolic", re.I)
_TITLE_NOTE = re.compile(r"\bfor the\b.*\bchampionship\b", re.I)


def parse_event(wikitext, title=""):
    """(event, bouts) from one event page.

    event: {name, date, location}. bouts: in page order, the first being
    the main event, each {order, weight, first, second, result, method,
    round, time, notes, card}. A bout not yet fought ("vs.") is left out,
    and so is one under an amateur or exhibition card.
    """
    box = infobox(wikitext or "")
    event = {"name": plain(box.get("name", "")) or title,
             "date": parse_date(box.get("date", "")),
             "promotion": plain(box.get("promotion", "")),
             "location": _location(box)}
    cards = [(s, plain(split_params(inner)[1]) if len(split_params(inner)) > 1 else "")
             for s, inner in templates(wikitext or "", "MMAevent card")]
    bouts = []
    for start, inner in templates(wikitext or "", "MMAevent bout"):
        params = [p.strip() for p in split_params(inner)[1:]]
        if len(params) < 4:
            continue
        params += [""] * (8 - len(params))
        weight, first, word, second, method, rnd, time, notes = params[:8]
        result = _result(plain(word), plain(method))
        card = ""
        for s, label in cards:
            if s < start:
                card = label
        if result is None or re.search(r"amateur|exhibition", card, re.I):
            continue
        bouts.append({"order": len(bouts), "weight": plain(weight),
                      "first": plain(first), "second": plain(second),
                      "result": result, "method": plain(method),
                      "round": plain(rnd), "time": plain(time),
                      "notes": plain(notes), "card": card})
    return event, bouts


def _location(box):
    """'[[Paradise, Nevada]], United States' -> 'Paradise, Nevada, USA'."""
    parts = [plain(box.get(k, "")) for k in ("city",)]
    text = ", ".join(p for p in parts if p)
    return re.sub(r"\bUnited States\b", "USA", text) or np.nan


def method_label(text):
    """Wikipedia's method in the archive's words (UFCStats' labels)."""
    low = (text or "").lower()
    if low.startswith("no contest") or low.startswith("nc"):
        return "Overturned" if "overturn" in low else "Could Not Continue"
    if low.startswith(("dq", "disqualification")):
        return "DQ"
    if "doctor" in low:
        return "TKO - Doctor's Stoppage"
    if low.startswith(("ko", "tko")):
        return "KO/TKO"
    if low.startswith(("submission", "technical submission")):
        return "Submission"
    if low.startswith(("decision", "draw")):
        for kind in ("unanimous", "split", "majority"):
            if kind in low:
                return f"Decision - {kind.capitalize()}"
        return "Decision - Unanimous"
    return text or np.nan


def seconds(time):
    """'4:25' -> 265.0: time into the final round, as UFCStats records it."""
    m = re.match(r"\s*(\d+):(\d{2})", time or "")
    return float(int(m.group(1)) * 60 + int(m.group(2))) if m else np.nan


DIVISION_LB = {"flyweight": 125, "bantamweight": 135, "featherweight": 145,
               "lightweight": 155, "welterweight": 170, "middleweight": 185,
               "light heavyweight": 205, "heavyweight": 265}
WOMEN = ("strawweight", "flyweight", "bantamweight", "featherweight")


def division_of(weight):
    """'Women's Strawweight' -> "women's strawweight". A catchweight returns
    None: see catchweight_division."""
    text = unicodedata.normalize("NFKC", weight or "").replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not text or "catch" in text:
        return None
    return re.sub(r"\s*\(.*?\)", "", text).strip()


def catchweight_division(weight, last_divisions):
    """The division a catchweight bout is filed under.

    UFCStats keeps a bout made at a catchweight (a missed weight, mostly)
    in the division it was booked at. That is the fighters' own division
    when their last bouts agree. Otherwise the heaviest division at or
    under the agreed weight ('Catchweight (147.5 lb)' -> featherweight),
    women's when either fighter last fought in a women's division."""
    known = [d for d in last_divisions if isinstance(d, str) and d]
    if len(known) == 2 and known[0] == known[1]:
        return known[0]
    lb = re.search(r"(\d+(?:\.\d+)?)\s*lb", weight or "")
    if not lb:
        return known[0] if len(set(known)) == 1 else "catch weight"
    women = any(d.startswith("women's") for d in known)
    fits = [d for d, limit in DIVISION_LB.items() if limit <= float(lb.group(1))
            and (not women or d in WOMEN)]
    if not fits:
        return "catch weight"
    best = max(fits, key=DIVISION_LB.get)
    return f"women's {best}" if women else best


def is_title(bout):
    """A championship bout: the page says "For the ... Championship", or a
    fighter carries (c)/(ic) on a bout that is not the symbolic BMF one."""
    notes = bout.get("notes", "")
    if _BMF.search(notes):
        return False
    if _TITLE_NOTE.search(notes):
        return True
    return bool(_TITLE_TAG.search(f"{bout['first']} {bout['second']}"))


def scheduled_rounds(bout):
    """Five for a title bout, the main event, a BMF bout or any bout that
    went past round three; three otherwise."""
    rnd = pd.to_numeric(bout.get("round"), errors="coerce")
    if (is_title(bout) or bout.get("order") == 0 or _BMF.search(bout.get("notes", ""))
            or (rnd == rnd and rnd > 3)):
        return 5.0
    return 3.0


# --- matching names ----------------------------------------------------------

class NameBridge:
    """A Wikipedia spelling -> the archive's (name, id), or None for a fighter
    the archive has never seen.

    In order:
      exact     the normalised name is one fighter in the archive (two
                fighters sharing a name are told apart by the division, as
                pending_rows.resolve_fighter does for a live card);
      learned   an alias read off the events both sources hold. A bout on
                the same date where one fighter's name agrees and the other's
                does not names the second one twice: "Carlos Diego Ferreira"
                on Wikipedia is "Diego Ferreira" in the archive, "Bobby
                Green" is "King Green", "Choi Doo-ho" is "Dooho Choi";
      subset    every word of one spelling is in the other ("Eric McConico
                Jr."), among fighters active in the last three years, when
                exactly one matches.
    """

    def __init__(self, archive, wiki_bouts=None, active_since=None):
        self.archive = archive
        self.candidates = pending_rows.fighter_candidates(archive)
        self.history = pending_rows.division_history(archive)
        self.last_division = {i: h[-1] for i, h in self.history.items()}
        self.names = {}
        long = pd.concat([pd.DataFrame({"id": archive[f"{c}_id"].astype(str),
                                        "name": archive[f"{c}_name"],
                                        "date": pd.to_datetime(archive["date"])})
                          for c in "rb"])
        latest = long.sort_values("date", kind="mergesort").groupby("id").last()
        self.names = latest["name"].to_dict()
        since = active_since or (pd.to_datetime(archive["date"]).max() - pd.DateOffset(years=3))
        active = latest[latest["date"] >= since]
        self.active = {norm_name(n): i for i, n in active["name"].items()}
        self.aliases = self.learn(archive, wiki_bouts) if wiki_bouts is not None else {}

    @staticmethod
    def learn(archive, wiki_bouts):
        """{normalised Wikipedia spelling: archive name}, from bouts both hold."""
        by_date = defaultdict(list)
        for b in wiki_bouts.itertuples(index=False):
            by_date[pd.Timestamp(b.date)].append((norm_name(b.winner), norm_name(b.loser),
                                                  ascii_name(b.winner), ascii_name(b.loser)))
        seen = defaultdict(Counter)
        dates = pd.to_datetime(archive["date"])
        for when, red, blue in zip(dates, archive["r_name"], archive["b_name"]):
            pair = {norm_name(red), norm_name(blue)}
            for w, l, _, _ in by_date.get(when, ()):
                common = pair & {w, l}
                if len(common) != 1 or {w, l} == pair:
                    continue
                agreed = common.pop()
                wiki_other = l if w == agreed else w
                arch_other = blue if norm_name(red) == agreed else red
                seen[wiki_other][arch_other] += 1
                break
        return {k: c.most_common(1)[0][0] for k, c in seen.items()}

    def resolve(self, name, division):
        """(archive name, id, how) or (None, None, 'new')."""
        key = norm_name(name)
        for how, lookup in (("exact", key), ("learned", norm_name(self.aliases.get(key, "")))):
            if lookup and lookup in self.candidates:
                try:
                    fid = pending_rows.resolve_fighter(lookup, division, self.candidates,
                                                       self.history)
                except KeyError:
                    # Two fighters share the name and the division does not
                    # say which: neither is guessed, the bout is left out.
                    return None, None, "ambiguous"
                return self.names.get(fid, name), fid, how
        words = set(key.split())
        hits = [i for n, i in self.active.items()
                if len(words) > 1 and (set(n.split()) <= words or words <= set(n.split()))
                and len(set(n.split())) > 1]
        if len(hits) == 1:
            return self.names.get(hits[0], name), hits[0], "subset"
        return None, None, "new"


# --- provisional rows --------------------------------------------------------

def bridge_rows(archive, events, names):
    """Rated, archive-layout rows for `events` [(event, bouts)], each fighter
    resolved through `names` (a NameBridge). Returns (rows, report)."""
    fights, outcomes, report = [], [], {"new": [], "skipped": [], "how": Counter()}
    for event, bouts in events:
        for b in bouts:
            division = division_of(b["weight"])
            corner, hows = {}, []
            for side in ("first", "second"):
                arch, fid, how = names.resolve(b[side], division)
                hows.append(how)
                if fid is None and how == "new":
                    arch = ascii_name(b[side])
                    fid = debut_id(arch)
                corner[side] = (arch, fid)
            if "ambiguous" in hows:
                report["skipped"].append(f"{event['date']} {b['first']} vs {b['second']}: "
                                         "a name two archive fighters share")
                continue
            report["how"].update(hows)
            report["new"] += [f"{event['date']} {corner[s][0]} ({b['weight']})"
                              for s, how in zip(("first", "second"), hows) if how == "new"]
            if division is None:
                division = catchweight_division(
                    b["weight"], [names.last_division.get(corner[s][1]) for s in ("first", "second")])
            fights.append({"red": corner["first"][0], "blue": corner["second"][0],
                           "red_id": corner["first"][1], "blue_id": corner["second"][1],
                           "date": event["date"], "division": division,
                           "total_rounds": scheduled_rounds(b), "is_title": is_title(b)})
            outcomes.append((event, b, corner))
    if not fights:
        return archive.iloc[0:0].copy(), report

    # build_pending reads the archive as the CSV holds it, dates as text.
    archive = archive.assign(date=pd.to_datetime(archive["date"]))
    text_dates = archive.assign(date=archive["date"].dt.strftime("%Y-%m-%d"))
    rows = pending_rows.build_pending(text_dates, fights)
    cols = defaultdict(list)
    for event, b, corner in outcomes:
        winner = corner["first"] if b["result"] == "win" else (np.nan, np.nan)
        cols["event_name"].append(event["name"])
        cols["event_id"].append(_hash(event["name"], event["date"]))
        cols["location"].append(event["location"])
        cols["fight_id"].append(_hash(event["date"], *sorted([corner["first"][1],
                                                               corner["second"][1]])))
        cols["winner"].append(winner[0])
        cols["winner_id"].append(winner[1])
        cols["method"].append(method_label(b["method"]))
        cols["finish_round"].append(pd.to_numeric(b["round"], errors="coerce"))
        cols["match_time_sec"].append(seconds(b["time"]))
        # build_pending names a fighter by their latest archive spelling; a
        # debutant keeps the Wikipedia one, accents removed.
        cols["r_name"].append(corner["first"][0])
        cols["b_name"].append(corner["second"][0])
    for col, values in cols.items():
        rows[col] = pd.Series(values, index=rows.index, dtype=object if col not in
                              ("finish_round", "match_time_sec") else float)
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.drop(columns=[c for c in rows.columns if c.endswith(("_mu_pre", "_sigma_pre", "_mmr_pre"))])
    rows = carry_forward_records(archive, rows)
    combined = extend(archive, rows)
    new = combined[is_provisional(combined)].copy()
    new = red_is_higher_rated(new)
    return new.reindex(columns=archive.columns), report


def red_is_higher_rated(rows):
    """Swap the corners of each row whose blue corner is rated higher before
    the bout. The winner column names a fighter, so it needs no change."""
    swap = rows["b_mmr_pre"] > rows["r_mmr_pre"]
    pairs = [(c, "b_" + c[2:]) for c in rows.columns
             if c.startswith("r_") and "b_" + c[2:] in rows.columns]
    for r, b in pairs:
        left = rows.loc[swap, r].copy()
        rows.loc[swap, r] = rows.loc[swap, b].values
        rows.loc[swap, b] = left.values
    return rows


def merge(archive, events, names, log=print):
    """The archive without its old provisional rows, plus fresh ones for every
    event after its last real date. Returns (combined, rows, report)."""
    base = archive[~is_provisional(archive)].reset_index(drop=True)
    cutoff = pd.to_datetime(base["date"]).max()
    later = [(e, b) for e, b in events if e["date"] and pd.Timestamp(e["date"]) > cutoff and b]
    rows, report = bridge_rows(base, later, names)
    rows = canonical_winners(rows)
    combined = pd.concat([base, rows], ignore_index=True)
    combined = combined.sort_values("date", kind="mergesort").reset_index(drop=True)
    # A whole-number column stays whole: written as 3.0 it would rewrite every
    # line of the file, and every row would look changed in its history.
    for col in base.columns:
        if pd.api.types.is_integer_dtype(base[col]) and combined[col].notna().all():
            combined[col] = combined[col].astype(base[col].dtype)
    return combined, rows, report


# --- fetching ----------------------------------------------------------------

def discover(years):
    """Titles of UFC event pages: every page in the year's mixed martial arts
    category and its subcategories (what harvest_world reads), and every
    link on the "<year> in UFC" page."""
    from check_fight_changes import get
    from harvest_world import discover as category_pages
    titles = {t for t in category_pages(years, verbose=False) if EVENT_TITLE.search(t)}
    for year in years:
        body = get({"action": "parse", "page": f"{year} in UFC", "prop": "links",
                    "redirects": 1})
        titles |= {link.get("*", "") for link in body.get("parse", {}).get("links", [])
                   if link.get("ns") == 0 and EVENT_TITLE.search(link.get("*", ""))}
    return sorted(titles)


def fetch_events(since, today):
    """Every UFC event page dated after `since` and before `today`, parsed."""
    from harvest_injuries import fetch_pages
    years = list(range(since.year, today.year + 1))
    titles = discover(years)
    texts = fetch_pages(titles)
    events = []
    for title, text in texts.items():
        event, bouts = parse_event(text, title)
        if "ultimate fighting" not in event["promotion"].lower() or not event["date"]:
            continue
        when = pd.Timestamp(event["date"])
        # An event on today's date may still be in progress, its page half
        # filled in; it is taken on the next run.
        if since < when < today and bouts:
            events.append((event, bouts))
    return sorted(events, key=lambda e: e[0]["date"])


def wiki_history():
    """UFC event-page bouts from the committed Wikipedia snapshot, for
    learning aliases."""
    if not WORLD_BOUTS.exists():
        return None
    w = pd.read_csv(WORLD_BOUTS, low_memory=False)
    w = w[~w["source"].astype(str).str.startswith("record:")]
    w = w[w["promotion"].astype(str).str.contains("Ultimate Fighting")]
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    return w.dropna(subset=["date", "winner", "loser"])


def cmd_sync(args):
    # round_trip: the default parser misreads the last digit of a few ratings,
    # and writing them back would change rows this run never touched.
    archive = pd.read_csv(ARCHIVE, low_memory=False, dtype=pending_rows.TEXT_COLUMNS,
                          float_precision="round_trip")
    archive["date"] = pd.to_datetime(archive["date"])
    old = int(is_provisional(archive).sum())
    base = archive[~is_provisional(archive)].reset_index(drop=True)
    cutoff = base["date"].max()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    print(f"Archive: {len(base):,} fights through {cutoff.date()}"
          + (f", plus {old} provisional" if old else ""))
    events = fetch_events(cutoff, today)
    print(f"Wikipedia: {len(events)} UFC events after {cutoff.date()}")
    names = NameBridge(base, wiki_history())
    combined, rows, report = merge(archive, events, names)
    print(f"Provisional fights: {len(rows)}")
    for (name, date), n in rows.groupby(["event_name", rows["date"].dt.date]).size().items():
        print(f"    {n:3d}  {date}  {name}")
    print("  fighters matched: " + ", ".join(f"{k} {v}" for k, v in sorted(report["how"].items())))
    for line in report["new"]:
        print(f"    debut: {line}")
    for line in report["skipped"]:
        print(f"    left out: {line}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if len(rows) == old and len(rows) == 0:
        print("Nothing to add.")
        return
    backup = ARCHIVE.with_suffix(f".backup-{cutoff.date()}.csv")
    if not backup.exists():
        shutil.copy2(ARCHIVE, backup)
    combined.to_csv(ARCHIVE, index=False)
    print(f"Wrote {ARCHIVE}")


# --- measurement -------------------------------------------------------------

def snapshot_events(wiki, start, end):
    """(event, bouts) for UFC events in [start, end] from the snapshot. The
    snapshot keeps page order but not the notes, so a title is read from
    the (c) tags alone."""
    events = []
    part = wiki[(wiki["date"] >= start) & (wiki["date"] <= end)]
    for (date, name), group in part.groupby(["date", "event"], sort=True):
        bouts = [{"order": i, "weight": str(r.weight), "first": str(r.winner),
                  "second": str(r.loser), "result": r.result, "method": str(r.method),
                  "round": str(r.round), "time": str(r.time), "notes": "", "card": str(r.card)}
                 for i, r in enumerate(group.itertuples(index=False))]
        events.append(({"name": name, "date": date.strftime("%Y-%m-%d"),
                        "promotion": "Ultimate Fighting Championship",
                        "location": np.nan}, bouts))
    return events


def _method_class(m):
    m = str(m).upper()
    for key, label in (("KO", "KO/TKO"), ("SUB", "SUB"), ("DEC", "DEC")):
        if m.startswith(key) or key in m.split(" ")[0]:
            return label
    return "OTHER"


def measure(archive, wiki, events_back=8):
    """Cut the archive before its last `events_back` events, rebuild those
    from the Wikipedia snapshot, and compare with the rows the archive has."""
    archive = archive[~is_provisional(archive)].copy()
    archive["date"] = pd.to_datetime(archive["date"])
    dates = sorted(archive["date"].unique())[-events_back:]
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    before = archive[archive["date"] < start].reset_index(drop=True)
    truth = archive[archive["date"] >= start]
    names = NameBridge(before, wiki[wiki["date"] < start])
    rows, report = bridge_rows(before, snapshot_events(wiki, start, end), names)

    # A debutant has a UFCStats id in the archive and a provisional one here, so
    # bouts are paired by date and either fighter's id or name.
    truth_by = defaultdict(list)
    for _, t in truth.iterrows():
        for c in "rb":
            truth_by[(t["date"], t[f"{c}_id"])].append(t)
            truth_by[(t["date"], norm_name(t[f"{c}_name"]))].append(t)
    out = Counter(truth=len(truth), rows=len(rows))
    diffs = []
    for _, r in rows.iterrows():
        found = [t for c in "rb" for k in (r[f"{c}_id"], norm_name(r[f"{c}_name"]))
                 for t in truth_by.get((r["date"], k), [])]
        if not found:
            continue
        t = found[0]
        out["matched"] += 1
        # Which truth corner is this row's red: whichever fighter agrees by
        # id or by name (a debutant agrees by neither, but their opponent does).
        agree = lambda a, b: r[f"{a}_id"] == t[f"{b}_id"] or \
            norm_name(r[f"{a}_name"]) == norm_name(t[f"{b}_name"])
        same = agree("r", "r") or agree("b", "b")
        flip = {"r": "r" if same else "b", "b": "b" if same else "r"}
        win_side = next((c for c in "rb" if r["winner"] == r[f"{c}_name"]), None)
        truth_side = next((c for c in "rb" if t["winner"] == t[f"{c}_name"]), None)
        out["winner"] += (flip[win_side] == truth_side) if win_side else truth_side is None
        out["method"] += _method_class(r["method"]) == _method_class(t["method"])
        out["round"] += r["finish_round"] == t["finish_round"]
        out["time"] += r["match_time_sec"] == t["match_time_sec"]
        out["rounds"] += r["total_rounds"] == t["total_rounds"]
        out["title"] += int(r["title_fight"]) == int(t["title_fight"])
        out["division"] += r["division"] == t["division"]
        out["red_is_archive_red"] += same
        for c in "rb":
            tid = t[f"{flip[c]}_id"]
            if str(r[f"{c}_id"]).startswith(PROVISIONAL):
                # Not matched: right only if the archive had never seen them.
                had = ((before["r_id"] == tid) | (before["b_id"] == tid)).any()
                out["debut_correct" if not had else "debut_wrong"] += 1
            else:
                out["id_correct" if r[f"{c}_id"] == tid else "id_wrong"] += 1
            for k in ("mu_pre", "sigma_pre"):
                diffs.append(abs(float(r[f"{c}_{k}"]) - float(t[f"{flip[c]}_{k}"])))
    out["fighters_resolved"] = 2 * len(rows) - len(report["new"])
    out["rating_max_diff"] = max(diffs) if diffs else np.nan
    out["red_win_rows"] = float((rows["winner"] == rows["r_name"]).mean()) if len(rows) else np.nan
    out["red_win_archive"] = float((truth["winner"] == truth["r_name"]).mean())
    out["window"] = f"{start.date()} to {end.date()}"
    return out, report


def cmd_measure(args):
    archive = pd.read_csv(ARCHIVE, low_memory=False, dtype=pending_rows.TEXT_COLUMNS)
    out, report = measure(archive, wiki_history(), args.events)
    for k, v in out.items():
        print(f"  {k:<22}{v}")
    for line in report["new"]:
        print(f"    not matched: {line}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--dry-run", action="store_true")
    m = sub.add_parser("measure")
    m.add_argument("--events", type=int, default=8)
    args = parser.parse_args(argv)
    return {"sync": cmd_sync, "measure": cmd_measure}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
