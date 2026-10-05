"""Read each pending story's own publication date, and settle it.

intel_backfill.py accepts a historical record only when its publication date
is provably before the fight. A date in the URL settles most of them in the
container. The rest wait in intel_history_pending.csv for this, which runs on
the Actions runner because this container's egress proxy refuses news sites.

For each pending record it fetches the page and reads the date the PAGE
declares for itself - article:published_time, JSON-LD datePublished and the
like - then hands it to the same verdict() the URL check uses. It never reads
a modified date: an article updated after the fight carries a modified date
after the fight, and choosing the later of the two would reject stories that
were genuinely pre-fight, while choosing the modified date as the publication
date would be simply wrong.

    python engine/verify_intel_dates.py          (needs network)

Outcomes: accepted rows move to intel_history_scouted.csv; rejected rows are
dropped with the reason printed; a page that cannot be fetched stays pending
for the next run, because a timeout says nothing about when a story ran.
A page that loads but declares no date is rejected - there is nothing left
that could verify it.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))

import intel_backfill as backfill

AGENT = ("YourMMA-research/1.0 "
         "(https://github.com/Klukemma/YourMMA; publication-date check)")
TIMEOUT = 20
PAUSE = 1.0

META_TAG = re.compile(r"<meta\b[^>]*>", re.I)
ATTR = re.compile(r'([\w:.-]+)\s*=\s*("([^"]*)"|\'([^\']*)\')', re.I)
# Publication, never modification. Ordered by how reliably each is set by the
# publisher rather than by a template.
PUBLISHED_KEYS = ("article:published_time", "og:published_time",
                  "datepublished", "parsely-pub-date", "sailthru.date",
                  "pubdate", "publish-date", "date")
JSON_LD = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
TIME_TAG = re.compile(r"<time\b[^>]*\bdatetime\s*=\s*[\"']([^\"']+)[\"'][^>]*>",
                      re.I)


def published_from_html(html):
    """The publication date a page declares for itself, or None."""
    html = html or ""
    found = {}
    for tag in META_TAG.findall(html):
        attrs = {m.group(1).lower(): (m.group(3) if m.group(3) is not None
                                      else m.group(4))
                 for m in ATTR.finditer(tag)}
        key = (attrs.get("property") or attrs.get("name")
               or attrs.get("itemprop") or "").lower()
        if key in PUBLISHED_KEYS and attrs.get("content"):
            found.setdefault(key, attrs["content"])
    for key in PUBLISHED_KEYS:
        if key in found:
            parsed = _parse(found[key])
            if parsed:
                return parsed
    for raw in JSON_LD.findall(html):
        parsed = _parse(raw)
        if parsed:
            return parsed
    for tag in re.findall(r"<time\b[^>]*>", html, re.I):
        low = tag.lower()
        if "pubdate" in low or "datepublished" in low:
            hit = TIME_TAG.search(tag)
            if hit and _parse(hit.group(1)):
                return _parse(hit.group(1))
    return None


def _parse(value):
    try:
        stamp = pd.Timestamp(str(value).strip())
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.date()


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read(2_000_000).decode("utf-8", errors="replace")


def settle(pending, fetcher=fetch, pause=PAUSE):
    """(accepted rows, still pending rows, rejection reasons)."""
    accepted, still, rejected = [], [], []
    for row in pending:
        try:
            html = fetcher(row["source"])
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as err:
            still.append(row)
            print(f"    kept pending, could not fetch: {row['source']} "
                  f"({type(err).__name__})")
            time.sleep(pause)
            continue
        declared = published_from_html(html)
        if declared is None:
            rejected.append(f"{row['fighter']}: {row['source']} declares no "
                            f"publication date, so nothing can verify it")
        else:
            outcome, detail, published = backfill.verdict(row,
                                                          verified=declared)
            if outcome == "accept":
                accepted.append({**row, "published": str(published),
                                 "verified_by": "page"})
            else:
                rejected.append(f"{row['fighter']}: {detail}")
        time.sleep(pause)
    return accepted, still, rejected


def main():
    if not backfill.PENDING.exists():
        print("  nothing pending")
        return 0
    frame = pd.read_csv(backfill.PENDING)
    pending = json.loads(frame.to_json(orient="records"))
    print(f"  {len(pending)} pending record(s) to verify")
    accepted, still, rejected = settle(pending)
    added = backfill._append(backfill.SCOUTED, accepted,
                             backfill.RECORD_COLUMNS,
                             ("event_date", "fighter", "kind", "source"))
    pd.DataFrame(still, columns=list(backfill.RECORD_COLUMNS)).to_csv(
        backfill.PENDING, index=False)
    print(f"  accepted {added}, still pending {len(still)}, "
          f"rejected {len(rejected)}")
    for reason in rejected:
        print(f"  REJECTED  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
