"""Save a few real Sherdog pages, so a record parser can be written against them.

Most Contender Series prospects have no Wikipedia article, and the honest
world model showed that without their regional records it has no skill on
the Contender Series. Sherdog lists every professional bout for everyone,
and its robots.txt allows all agents (Tapology's refuses AI crawlers, so it
is not used). This fetches a handful of pages - a fight-finder search and
three fighter pages - slowly, and saves them as test fixtures.

    python engine/probe_sherdog.py        (needs network; 5 requests)
"""

import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

from check_fight_changes import AGENT

FIXTURES = ENGINE / "tests" / "fixtures" / "sherdog"
BASE = "https://www.sherdog.com"
PAUSE = 3.0
SEARCH = ["Luca Borando", "Camila Reynoso", "George Staines"]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    time.sleep(PAUSE)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode("utf-8", "replace")


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name in SEARCH:
        url = f"{BASE}/stats/fightfinder?SearchTxt={urllib.parse.quote(name)}"
        try:
            status, html = fetch(url)
        except (urllib.error.URLError, OSError, TimeoutError) as err:
            print(f"  search {name}: failed {err}")
            continue
        slug = re.sub(r"\W+", "_", name)
        (FIXTURES / f"search_{slug}.html").write_text(html[:400_000])
        links = sorted(set(re.findall(r'href="(/fighter/[^"]+)"', html)))
        print(f"  search {name}: HTTP {status}, {len(html):,} chars, "
              f"fighter links: {links[:5]}")
        if links:
            try:
                status, page = fetch(BASE + links[0])
            except (urllib.error.URLError, OSError, TimeoutError) as err:
                print(f"    page failed {err}")
                continue
            (FIXTURES / f"fighter_{slug}.html").write_text(page[:600_000])
            print(f"    {links[0]}: HTTP {status}, {len(page):,} chars, "
                  f"'event' mentions {page.count('event')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
