"""Probe the fixture feeds the development sandbox cannot reach, and keep
what they return as test fixtures.

football-data.co.uk posts the coming weekend's fixtures on Friday afternoon
and the midweek's on Tuesday, so from Monday to Thursday the app has nothing
to say about the next round. Two free feeds publish the whole schedule with
kickoff times: ESPN's public scoreboard and fixturedownload.com. Which one is
right for the app is a question about what they actually return - kickoff
clocks, postponements, club names - so this saves their real responses
(gzipped JSON) under tests/fixtures/feeds/ for the parser to be written and
tested against. A few dozen requests; run on GitHub's machines:

    python soccer/engine/probe_feeds.py
"""

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tests" / "fixtures" / "feeds"
UA = {"User-Agent": "YourSoccer fixtures probe (github.com/Klukemma/YourMMA)"}

ESPN = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1"}
ESPN_BOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams"
FIXTUREDOWNLOAD = "https://fixturedownload.com/feed/json/{slug}-{year}"
FD_SLUGS = {"E0": ["epl"], "SP1": ["la-liga"], "I1": ["serie-a"], "D1": ["bundesliga"], "F1": ["ligue-1"]}


def get(url):
    import requests
    r = requests.get(url, timeout=60, headers=UA)
    return r.status_code, r.content


def save(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    return path


def fetch(name, url, report):
    """One request, recorded whatever happens; the JSON saved when it parses."""
    try:
        status, body = get(url)
        data = json.loads(body) if status == 200 else None
        note = None if status == 200 else body[:300].decode("utf-8", "replace")
    except Exception as e:                               # noqa: BLE001 - a probe reports, never stops
        status, data, note = repr(e)[:200], None, None
    entry = {"url": url, "status": status}
    if note:
        entry["body"] = note
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
        entry["events"] = len(events)
        entry["statuses"] = sorted({e.get("status", {}).get("type", {}).get("name", "?") for e in events})
        entry["first"] = min((e.get("date", "") for e in events), default=None)
        entry["last"] = max((e.get("date", "") for e in events), default=None)
    report["feeds"][name] = entry
    if data is not None:
        save(name, data)
    print(f"  {name:<34} {status}  {entry.get('events', '')} {entry.get('first') or ''} {entry.get('last') or ''}",
          flush=True)
    return data


def main(now=None):
    now = now or datetime.now(timezone.utc)
    days = [now + timedelta(days=k) for k in range(-4, 13)]
    span = f"{days[0]:%Y%m%d}-{days[-1]:%Y%m%d}"
    report = {"probed": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "span": span, "feeds": {}}
    for code, lg in ESPN.items():
        board = ESPN_BOARD.format(lg=lg)
        fetch(f"espn_{code}_teams", ESPN_TEAMS.format(lg=lg), report)
        # Which query shapes the scoreboard accepts: none (its default
        # window), a date range, and one day at a time.
        fetch(f"espn_{code}_board_default", board, report)
        fetch(f"espn_{code}_board_range", f"{board}?dates={span}", report)
        fetch(f"espn_{code}_board_range_limit", f"{board}?dates={span}&limit=100", report)
        for d in days:
            fetch(f"espn_{code}_board_{d:%Y%m%d}", f"{board}?dates={d:%Y%m%d}", report)
    for code, slugs in FD_SLUGS.items():
        for slug in slugs:
            for year in (now.year, now.year - 1):
                url = FIXTUREDOWNLOAD.format(slug=slug, year=year)
                name = f"fixturedownload_{code}_{year}"
                data = fetch(name, url, report)
                report["feeds"][name]["rows"] = len(data) if isinstance(data, list) else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
