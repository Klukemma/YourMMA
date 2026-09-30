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
OUT = ENGINE / "data" / "sherdog_records.jsonl.gz"
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
    args = parser.parse_args(argv)

    targets = contender_appearances()
    for name in args.names:
        targets.setdefault(name, [])
    done = load()
    todo = [n for n in sorted(targets)
            if n not in done or (args.retry_missing and not done[n].get("id"))]
    print(f"  {len(targets)} fighters, {len(done)} already looked up, "
          f"{len(todo)} to do")
    stop = time.time() + args.minutes * 60
    errors = looked = 0
    for name in todo:
        if time.time() > stop:
            print("  time budget spent - the next run carries on")
            break
        try:
            hit = sherdog.find(name, AGENT, targets[name])
            errors = 0
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            errors += 1
            print(f"  {name}: failed ({err})")
            if errors >= MAX_ERRORS:
                print("  too many failures in a row - stopping")
                break
            continue
        first = min((d for d, _ in targets[name]), default=None)
        done[name] = {"target": name, "first_contender": first,
                      **(hit or {"id": None}),
                      "looked_up": time.strftime("%Y-%m-%d")}
        looked += 1
        if hit:
            print(f"  {name}: {hit['slug']}-{hit['id']}, {len(hit['record'])} "
                  f"pro bouts{'' if hit['verified'] else ' (NOT verified)'}")
        else:
            print(f"  {name}: no record containing the Contender Series bout")
        if looked % SAVE_EVERY == 0:
            save(done)
    save(done)
    found = sum(1 for r in done.values() if r.get("id"))
    verified = sum(1 for r in done.values() if r.get("verified"))
    print(f"\n  {len(done)} looked up, {found} found, {verified} verified "
          f"against their Contender Series bout; wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
