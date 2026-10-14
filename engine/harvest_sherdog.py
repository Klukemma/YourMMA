"""Sherdog pro records for both fighters in every Contender Series bout.

The world model has no skill on the Contender Series without prospects'
regional careers (world_model.py). This looks each of them up on Sherdog
and keeps a record only when it contains the Contender Series bout the
fighter was looked up for - a search by name also finds namesakes.

Both fighters of every bout are looked up, winner and loser alike, so the
coverage does not depend on the result. What a lookup returns is today's
page; world_model.py uses each record only from the date its owner first
appeared on the Contender Series (when anyone would have looked them up),
and only the bouts dated before the bout being predicted.

Resumable: fighters already in the output are skipped, and the file is
rewritten every SAVE_EVERY lookups, so a stopped run keeps its work.

    python engine/harvest_sherdog.py [--minutes 300] [--names "A B" ...]
    python engine/harvest_sherdog.py --ufc-debuts     (every UFC fighter,
                    verified against their first UFC bout, newest debuts
                    first; writes sherdog_ufc_records.jsonl.gz - the
                    pre-UFC careers a debut model is built from)
    python engine/harvest_sherdog.py --card           (everyone on the
                    current FIGHT_CARD, looked up again so the app's pro
                    records and last five results are current)
"""

import argparse
import gzip
import json
import sys
import time
import urllib.error
from collections import defaultdict
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import sherdog
from check_fight_changes import AGENT

BOUTS = ENGINE / "data" / "world_bouts.csv.gz"
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
OUT = ENGINE / "data" / "sherdog_records.jsonl.gz"
OUT_UFC = ENGINE / "data" / "sherdog_ufc_records.jsonl.gz"
SAVE_EVERY = 20
MAX_ERRORS = 5          # consecutive failures before stopping: the site
                        # is refusing us, and hammering it helps nobody


def contender_appearances(path=None):
    """{fighter: [(date, opponent), ...]} for every Contender Series bout on
    an event or season page (fighters' own record tables are not used to
    choose who is looked up - that would pick by later success)."""
    frame = pd.read_csv(path or BOUTS)
    frame = frame[~frame["source"].astype(str).str.startswith("record:")]
    dwcs = (frame["event"].astype(str).str.contains("contender series", case=False)
            | frame["source"].astype(str).str.contains("contender series",
                                                       case=False))
    frame = frame[dwcs].dropna(subset=["date", "winner", "loser"])
    out = defaultdict(list)
    for bout in frame.itertuples():
        out[bout.winner].append((str(bout.date)[:10], bout.loser))
        out[bout.loser].append((str(bout.date)[:10], bout.winner))
    return dict(out)


def ufc_debuts(path=None):
    """{fighter: [(date, opponent)]} - every fighter in the UFC archive with
    their FIRST UFC bout, newest debuts first, so a run cut short by its
    time budget has covered the fighters the production model sees."""
    frame = pd.read_csv(path or ARCHIVE, usecols=["date", "r_name", "b_name"],
                        low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna().sort_values("date")
    first = {}
    for bout in frame.itertuples():
        for me, them in ((bout.r_name, bout.b_name), (bout.b_name, bout.r_name)):
            if me not in first:
                first[me] = (str(bout.date.date()), them)
    return {name: [bout] for name, bout in
            sorted(first.items(), key=lambda kv: kv[1][0], reverse=True)}


PREDICT_CARD = ENGINE / "predict_card.py"
CARD_VERIFY_BOUTS = 3

# How Sherdog spells a fighter where it differs from the card: tried in order
# after the card's own spelling. The record is still verified against the
# fighter's UFC bouts when there are any, so a wrong alias finds nothing
# rather than somebody else.
SEARCH_AS = {
    "Cristian Perez": ["Cristian Perez Gonzalez"],
    "SuYoung You": ["Su Young You", "Suyoung You", "Su-Young You"],
}


def find_any(name, agent, bouts):
    """sherdog.find under each SEARCH_AS spelling, then the card's own.

    Sherdog's spelling first: when the card's spelling is a different,
    real fighter's name (a namesake), searching it first would take them."""
    for spelling in SEARCH_AS.get(name, []) + [name]:
        hit = sherdog.find(spelling, agent, bouts)
        if hit:
            return hit
    return None


def card_fighters(source=None):
    """(event_date, [names]) on predict_card's FIGHT_CARD, read from the source
    text (importing predict_card trains every model)."""
    import ast
    tree = ast.parse(Path(source or PREDICT_CARD).read_text())
    values = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("FIGHT_CARD", "EVENT_DATE")):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    names = []
    for fight in values.get("FIGHT_CARD", []):
        for name in fight[:2]:
            if name not in names:
                names.append(name)
    return str(values.get("EVENT_DATE", ""))[:10], names


def card_targets(source=None, archive=None):
    """{fighter: [(date, opponent), ...]} for everyone on the current card,
    with their latest UFC bouts before the card to verify a record against.
    A UFC debutant gets none, and sherdog.find then accepts only a single
    exact-name match, recorded as unverified."""
    event_date, names = card_fighters(source)
    frame = pd.read_csv(archive or ARCHIVE, usecols=["date", "r_name", "b_name"],
                        low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna()
    if event_date:
        frame = frame[frame["date"] < pd.Timestamp(event_date)]
    world = _world_bouts(event_date)
    out = {}
    for name in names:
        rows = []
        for me, them in (("r_name", "b_name"), ("b_name", "r_name")):
            hit = frame[frame[me] == name]
            rows += [(str(d.date()), o) for d, o in zip(hit["date"], hit[them])]
        if not rows:
            # A UFC debutant: their Contender Series or regional bouts on
            # file verify the record instead of a name-only match.
            rows = world.get(name, [])
        out[name] = sorted(rows, reverse=True)[:CARD_VERIFY_BOUTS]
    return out


def _world_bouts(before=None, path=None):
    """{fighter: [(date, opponent), ...]} from world_bouts.csv.gz (Contender
    Series and regional bouts), dated before `before`."""
    path = Path(path or BOUTS)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["date", "winner", "loser"], low_memory=False)
    frame = frame.dropna(subset=["date", "winner", "loser"])
    if before:
        frame = frame[frame["date"].astype(str).str[:10] < str(before)[:10]]
    out = defaultdict(list)
    for bout in frame.itertuples():
        day = str(bout.date)[:10]
        out[bout.winner].append((day, bout.loser))
        out[bout.loser].append((day, bout.winner))
    return dict(out)


def load(path=None):
    path = path or OUT
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["target"]: row for row in rows}


def save(done, path=None):
    path = path or OUT
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as handle:
        for key in sorted(done):
            handle.write(json.dumps(done[key], sort_keys=True) + "\n")
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--minutes", type=float, default=300.0)
    parser.add_argument("--names", nargs="*", default=[],
                        help="also look these up (an upcoming card)")
    parser.add_argument("--retry-missing", action="store_true",
                        help="look again at fighters not found last time")
    parser.add_argument("--ufc-debuts", action="store_true",
                        help="every UFC fighter, verified against their first "
                             "UFC bout, into sherdog_ufc_records.jsonl.gz")
    parser.add_argument("--card", action="store_true",
                        help="look up everyone on predict_card's FIGHT_CARD "
                             "again, so the app's records and last five are "
                             "current; debutants included")
    args = parser.parse_args(argv)

    out = OUT_UFC if (args.ufc_debuts or args.card) else OUT
    targets = (card_targets() if args.card else
               ufc_debuts() if args.ufc_debuts else contender_appearances())
    for name in args.names:
        targets.setdefault(name, [])
    done = load(out)
    order = list(targets) if (args.ufc_debuts or args.card) else sorted(targets)
    # A card is looked up again whatever is on file: a record harvested weeks
    # ago is missing the fighter's latest bouts, and it is the last five that
    # the app shows.
    todo = [n for n in order
            if args.card or n not in done
            or (args.retry_missing and not done[n].get("id"))]
    print(f"  {len(targets)} fighters, {len(done)} already looked up, "
          f"{len(todo)} to do")
    stop = time.time() + args.minutes * 60
    errors = looked = 0
    for name in todo:
        if time.time() > stop:
            print("  time budget spent - the next run carries on")
            break
        try:
            hit = (find_any(name, AGENT, targets[name]) if args.card
                   else sherdog.find(name, AGENT, targets[name]))
            errors = 0
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            errors += 1
            print(f"  {name}: failed ({err})")
            if errors >= MAX_ERRORS:
                print("  too many failures in a row - stopping")
                break
            continue
        first = min((d for d, _ in targets[name]), default=None)
        if args.card:
            previous = done.get(name) or {}
            if hit and previous.get("id") and hit.get("id") != previous["id"]:
                # A different Sherdog fighter than the one on file. Which is
                # right needs a person; neither silently replaces the other.
                print(f"  {name}: CONFLICT - Sherdog {hit.get('id')} found, "
                      f"{previous['id']} on file; keeping the one on file")
                continue
            if not hit and previous.get("id"):
                if previous.get("verified"):
                    # Not found this time (a search hiccup, a renamed page):
                    # a verified record on file is kept rather than erased.
                    print(f"  {name}: not found again - keeping the verified "
                          f"record on file")
                    continue
                # A name-only match that no longer resolves - often a second
                # page with the same name - is dropped: it may be the wrong
                # person, and a wrong record is worse than none.
                print(f"  {name}: name-only record no longer resolves - dropped")
            first = previous.get("first_ufc") or first
        done[name] = {"target": name,
                      "first_ufc" if (args.ufc_debuts or args.card) else "first_contender": first,
                      **(hit or {"id": None}),
                      "looked_up": time.strftime("%Y-%m-%d")}
        looked += 1
        if hit:
            print(f"  {name}: {hit['slug']}-{hit['id']}, {len(hit['record'])} "
                  f"pro bouts{'' if hit['verified'] else ' (NOT verified)'}")
        else:
            print(f"  {name}: no record containing the bout looked up for")
        if looked % SAVE_EVERY == 0:
            save(done, out)
    save(done, out)
    found = sum(1 for r in done.values() if r.get("id"))
    verified = sum(1 for r in done.values() if r.get("verified"))
    print(f"\n  {len(done)} looked up, {found} found, {verified} verified "
          f"against the bout looked up for; wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
