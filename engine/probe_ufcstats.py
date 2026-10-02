"""Does UFCStats hold round-by-round statistics for the archive's fights?

The archive's fight_id, event_id and fighter ids are 16 hex characters -
the form UFCStats uses in its URLs (ufcstats.com/fight-details/<id>). If
they are the same ids, every archived fight's per-round totals (strikes,
takedowns, control time by round) are one page away, which is what a
measured cardio feature needs (how a fighter's output in round 3 compares
with round 1, over a career) instead of the guess the engine carries.

This fetches robots.txt, one event page and the fight-details page of
three archived fights (the newest, one from 2019, one from 2012), slowly,
and saves them as test fixtures so a parser is written against real HTML.
It reports whether each page resolves and whether it carries a
"Per round" section.

    python engine/probe_ufcstats.py        (needs network; 5 requests)
"""

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import AGENT

FIXTURES = ENGINE / "tests" / "fixtures" / "ufcstats"
ARCHIVE = ENGINE / "data" / "UFC_with_mmr_rebuilt_dedup.csv"
BASE = "http://ufcstats.com"
PAUSE = 3.0


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    time.sleep(PAUSE)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8", "replace")


def sample_fights(archive):
    """(label, fight_id, event_id, names) for the newest fight and one each
    from 2019 and 2012."""
    archive = archive.copy()
    archive["date"] = pd.to_datetime(archive["date"], errors="coerce")
    picks = [("newest", archive.sort_values("date").iloc[-1])]
    for year in (2019, 2012):
        part = archive[archive["date"].dt.year == year]
        if len(part):
            picks.append((str(year), part.iloc[len(part) // 2]))
    return [(label, row["fight_id"], row["event_id"],
             f"{row['r_name']} vs {row['b_name']}") for label, row in picks]


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    try:
        status, robots = fetch(f"{BASE}/robots.txt")
        (FIXTURES / "robots.txt").write_text(robots[:20_000])
        print(f"  robots.txt: HTTP {status}, {len(robots)} chars")
        print("    " + "\n    ".join(robots.strip().splitlines()[:12]))
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        print(f"  robots.txt: failed {err}")
    archive = pd.read_csv(ARCHIVE, usecols=["date", "fight_id", "event_id",
                                            "r_name", "b_name"], low_memory=False)
    fights = sample_fights(archive)
    try:
        status, page = fetch(f"{BASE}/event-details/{fights[0][2]}")
        (FIXTURES / f"event_{fights[0][2]}.html").write_text(page[:600_000])
        print(f"  event {fights[0][2]}: HTTP {status}, {len(page):,} chars, "
              f"fight-details links {len(set(re.findall(r'fight-details/([0-9a-f]{16})', page)))}")
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        print(f"  event page: failed {err}")
    for label, fight_id, _, names in fights:
        try:
            status, page = fetch(f"{BASE}/fight-details/{fight_id}")
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            print(f"  {label} {fight_id} ({names}): failed {err}")
            continue
        (FIXTURES / f"fight_{fight_id}.html").write_text(page[:600_000])
        rounds = len(re.findall(r"Round \d", page))
        print(f"  {label} {fight_id} ({names}): HTTP {status}, {len(page):,} "
              f"chars, 'Per round' {'present' if 'Per round' in page else 'ABSENT'}, "
              f"'Round N' mentions {rounds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
