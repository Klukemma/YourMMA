"""What does fight data outside the UFC look like on Wikipedia, and what else
is reachable? A probe: it answers questions and saves samples, nothing more.

The model only knows UFC bouts, so it cannot rate a Contender Series debutant
whose whole career is LFA and Cage Warriors. Wikipedia keeps "2024 in Legacy
Fighting Alliance"-style pages with every result of a promotion's year, and
every fighter article carries a full pro record across all promotions. The
parsers for those have to be written against real wikitext, and this
container cannot reach Wikipedia - so the runner saves real pages as test
fixtures, and prints which year pages exist.

It also reads the robots.txt of the two big MMA record sites, because a
DWCS prospect usually has no Wikipedia article at all, and whether those
sites permit fetching fighter pages decides whether they can be a source.

    python engine/probe_world.py        (needs network; writes fixtures)
"""

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import AGENT, get
from harvest_injuries import fetch_pages

FIXTURES = ENGINE / "tests" / "fixtures" / "wiki"
LISTING = ENGINE / "data" / "world_year_pages.json"
MAX_FIXTURE = 250_000          # characters kept per sample page

SAMPLES = [
    "Dana White's Contender Series",
    "2024 in Legacy Fighting Alliance",
    "2023 in Legacy Fighting Alliance",
    "2024 in Cage Warriors",
    "2024 in Bellator MMA",
    "2024 in Professional Fighters League",
    "2016 in Legacy Fighting Alliance",
    "UFC 300",
    "Bellator 300",
    "Jon Jones",
    "Gilbert Burns",
    "Mateusz Gamrot",
]
YEARS = range(2008, 2027)


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


def robots(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except (urllib.error.URLError, OSError, TimeoutError) as err:
        return None, str(err)


def main():
    listing = {}
    for year in YEARS:
        category = f"Category:{year} in mixed martial arts"
        try:
            pages = members(category)
            subcats = members(category, "subcat")
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            print(f"  {year}: failed {err}")
            continue
        listing[year] = {"pages": pages, "subcats": subcats}
        print(f"  {year}: {len(pages)} pages, {len(subcats)} subcategories")
        print("    " + " | ".join(pages[:60]))
        if subcats:
            print("    subcats: " + " | ".join(subcats[:30]))
    LISTING.write_text(json.dumps(listing, indent=1, ensure_ascii=False))

    FIXTURES.mkdir(parents=True, exist_ok=True)
    texts = fetch_pages(SAMPLES)
    for title in SAMPLES:
        text = texts.get(title)
        if not text:
            print(f"  sample missing: {title}")
            continue
        name = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") + ".wiki"
        (FIXTURES / name).write_text(text[:MAX_FIXTURE])
        print(f"  saved {name}: {len(text):,} chars, "
              f"{text.count('def.')} 'def.' cells, "
              f"{len(re.findall(r'MMAevent bout', text))} MMAevent bout templates")

    for site in ("https://www.sherdog.com/robots.txt",
                 "https://www.tapology.com/robots.txt"):
        status, body = robots(site)
        print(f"\n  {site}: HTTP {status}")
        for line in body.splitlines():
            if re.match(r"\s*(user-agent|disallow|allow|crawl-delay)", line, re.I):
                print(f"    {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
