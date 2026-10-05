"""Every bout, every promotion, and every fighter's camp, from Wikipedia.

The UFC model cannot rate a fighter it has never seen fight in the UFC,
which is everyone on a Contender Series card. This gathers what Wikipedia
holds on the rest of the sport:

  1. every page in "Category:YEAR in mixed martial arts" (2005 on) and one
     level of its subcategories - year pages and event pages for Bellator,
     PFL, ONE, KSW, Cage Warriors, LFA, Oktagon, Brave, Rizin, UAE Warriors
     and others - plus every Dana White's Contender Series season page;
  2. every professional bout on them (world_bouts.event_bouts);
  3. the article of every fighter named in those bouts, found by title and
     confirmed by naming at least one of that fighter's actual opponents,
     for their full career record (world_bouts.record_bouts) and their camp,
     coaches and belts (fighter_profile.profile).

Writes
  engine/data/world_bouts.csv.gz       one row per bout, deduplicated across
                                       event pages and fighter records
  engine/data/fighter_profiles.jsonl.gz
  engine/data/world_pages.csv          what was read, for coverage

    python engine/harvest_world.py            (needs network, ~30 minutes)
    python engine/harvest_world.py --limit 3  (smoke test: 3 years)
"""

import argparse
import gzip
import json
import re
import sys
import urllib.error
from collections import defaultdict
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import get
from fighter_profile import profile
from harvest_injuries import (existing, fetch_pages, fighter_candidates,
                              mma_page, opponent_hits)
from name_resolution import norm_name
from world_bouts import event_bouts, parse_date, plain, record_bouts

BOUTS = ENGINE / "data" / "world_bouts.csv.gz"
PROFILES = ENGINE / "data" / "fighter_profiles.jsonl.gz"
PAGES = ENGINE / "data" / "world_pages.csv"
FIXTURES = ENGINE / "tests" / "fixtures" / "wiki"
YEARS = range(2005, 2027)
DWCS_SEASONS = range(1, 13)
MIN_BOUTS_FOR_PAGE = 1        # fighters whose article is looked for


def members(category, kind="page"):
    out, extra = [], {}
    while True:
        body = get({"action": "query", "list": "categorymembers",
                    "cmtitle": category, "cmlimit": 500, "cmtype": kind,
                    **extra})
        out += [m["title"] for m in body.get("query", {})
                .get("categorymembers", [])]
        if "continue" not in body:
            return out
        extra = body["continue"]


def discover(years, verbose=True):
    titles = set()
    for year in years:
        category = f"Category:{year} in mixed martial arts"
        try:
            pages = members(category)
            subcats = members(category, "subcat")
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            print(f"    {year}: failed {err}", flush=True)
            continue
        found = set(pages)
        for sub in subcats:
            try:
                found |= set(members(sub))
            except (urllib.error.URLError, OSError, TimeoutError):
                continue
        titles |= found
        if verbose:
            print(f"    {year}: {len(found)} pages ({len(subcats)} "
                  f"subcategories)", flush=True)
    titles |= {f"Dana White's Contender Series season {n}"
               for n in DWCS_SEASONS}
    return sorted(t for t in titles if not t.startswith(("Category:",
                                                         "Template:")))


# A generic results row - "| Winner ||def.|| Loser | method | ..." - for
# pages that write results as a plain table instead of MMAevent bout
# templates. Used only on pages where the template parser finds nothing, so
# the same bout is never read twice.
def table_bouts(wikitext, page):
    bouts, date = [], None
    for line in (wikitext or "").split("\n"):
        heading = re.match(r"^={2,4}\s*(.*?)\s*={2,4}\s*$", line)
        if heading:
            date = parse_date(heading.group(1)) or date
            continue
        if "def." not in line:
            continue
        cells = [plain(c) for c in re.split(r"\|\||\n\|", line.strip("| "))]
        if "def." not in cells:
            continue
        i = cells.index("def.")
        if i == 0 or i + 1 >= len(cells):
            continue
        bouts.append({"date": date, "event": page, "promotion": "",
                      "weight": cells[i - 2] if i >= 2 else "",
                      "winner": cells[i - 1], "loser": cells[i + 1],
                      "result": "win",
                      "method": cells[i + 2] if i + 2 < len(cells) else "",
                      "round": "", "time": "", "card": "",
                      "source": f"table:{page}"})
    return bouts


def bout_key(b):
    pair = tuple(sorted((norm_name(b["winner"]), norm_name(b["loser"]))))
    return (b.get("date") or "", pair)


def dedupe(bouts):
    """One row per bout; an event page's row beats a fighter record's."""
    best = {}
    for b in bouts:
        if not b.get("date") or not b.get("winner") or not b.get("loser"):
            continue
        key = bout_key(b)
        keep = best.get(key)
        if keep is None or (keep["source"].startswith("record:")
                            and not b["source"].startswith("record:")):
            best[key] = b
    return list(best.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__[:200])
    parser.add_argument("--limit", type=int, default=0,
                        help="only the last N years, for a smoke test")
    args = parser.parse_args(argv)
    years = list(YEARS)[-args.limit:] if args.limit else list(YEARS)

    print("=" * 74)
    print(f"EVERY PROMOTION ON WIKIPEDIA, {years[0]}-{years[-1]}")
    print("=" * 74, flush=True)

    print("\n  discovering pages", flush=True)
    titles = discover(years)
    print(f"  {len(titles):,} candidate pages", flush=True)
    texts = fetch_pages(titles)
    print(f"  {len(texts):,} fetched", flush=True)

    bouts, pages = [], []
    for title in titles:
        text = texts.get(title)
        if not text:
            pages.append(("results", title, 0, "missing"))
            continue
        found = event_bouts(text, title)
        how = "templates"
        if not found:
            found = table_bouts(text, title)
            how = "table" if found else "none"
        bouts += found
        pages.append(("results", title, len(found), how))
    # One real DWCS season page, kept as a fixture for the parser tests.
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for n in (1, 7):
        text = texts.get(f"Dana White's Contender Series season {n}")
        if text:
            (FIXTURES / f"DWCS_season_{n}.wiki").write_text(text[:250_000])
    event_rows = dedupe(bouts)
    print(f"  {len(event_rows):,} bouts from results pages", flush=True)

    # --- every fighter named in them ---------------------------------------
    opponents = defaultdict(set)
    for b in event_rows:
        opponents[b["winner"]].add(b["loser"])
        opponents[b["loser"]].add(b["winner"])
    fighters = sorted(f for f, o in opponents.items()
                      if len(o) >= MIN_BOUTS_FOR_PAGE and len(f) > 3)
    print(f"\n  looking for articles of {len(fighters):,} fighters", flush=True)
    candidates = {f: fighter_candidates(f) for f in fighters}
    present = existing(sorted({t for g in candidates.values() for t in g}))
    print(f"  {len(present):,} candidate titles exist", flush=True)
    fighter_texts = fetch_pages(sorted(set(present.values())))

    profiles, seen_pages = [], set()
    for fighter in fighters:
        best, best_score = None, 0
        for title in candidates[fighter]:
            target = present.get(title)
            text = fighter_texts.get(target)
            if not text or not mma_page(text, target):
                continue
            score = opponent_hits(text, opponents[fighter])
            if score > best_score:
                best, best_score = (target, text), score
        if not best:
            pages.append(("fighter", fighter, 0, "no article"))
            continue
        title, text = best
        if title in seen_pages:
            pages.append(("fighter", fighter, 0, f"same article as another: {title}"))
            continue
        seen_pages.add(title)
        rec = record_bouts(text, fighter)
        bouts += rec
        profiles.append({"fighter": fighter, "page": title,
                         "opponents_named": best_score,
                         **profile(text)})
        pages.append(("fighter", fighter, len(rec), title))

    rows = dedupe(bouts)
    frame = pd.DataFrame(rows, columns=["date", "event", "promotion", "weight",
                                        "winner", "loser", "result", "method",
                                        "round", "time", "card", "source"])
    frame = frame.sort_values(["date", "event"]).reset_index(drop=True)
    frame.to_csv(BOUTS, index=False, compression="gzip")
    with gzip.open(PROFILES, "wt", encoding="utf-8") as handle:
        for p in profiles:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")
    pd.DataFrame(pages, columns=["kind", "title", "bouts", "note"]).to_csv(
        PAGES, index=False)

    years_seen = pd.to_datetime(frame["date"], errors="coerce").dt.year
    print(f"\n  {'bouts, all promotions':<34}{len(frame):>9,}")
    print(f"    {'from results pages':<32}{(~frame.source.str.startswith('record:')).sum():>9,}")
    print(f"    {'from fighter records only':<32}{frame.source.str.startswith('record:').sum():>9,}")
    print(f"  {'distinct fighters':<34}{pd.concat([frame.winner, frame.loser]).nunique():>9,}")
    print(f"  {'fighter articles read':<34}{len(profiles):>9,}")
    print(f"  {'with a dated team history':<34}"
          f"{sum(1 for p in profiles if any(a['dated'] for a in p['affiliations'])):>9,}")
    print("  bouts by year: " + ", ".join(
        f"{int(y)}:{n}" for y, n in years_seen.value_counts().sort_index().items()
        if y == y))
    print(f"\n  wrote {BOUTS.name}, {PROFILES.name}, {PAGES.name}")
    return 0 if len(frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
