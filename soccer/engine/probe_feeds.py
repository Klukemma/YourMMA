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
ESPN_BOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={span}&limit=500"
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
    with gzip.open(path, "wb", mtime=0) as f:
        f.write(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    return path


def main(now=None):
    now = now or datetime.now(timezone.utc)
    span = f"{(now - timedelta(days=4)):%Y%m%d}-{(now + timedelta(days=12)):%Y%m%d}"
    report = {"probed": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "span": span, "feeds": {}}
    for code, lg in ESPN.items():
        for kind, url in (("board", ESPN_BOARD.format(lg=lg, span=span)), ("teams", ESPN_TEAMS.format(lg=lg))):
            name = f"espn_{code}_{kind}"
            try:
                status, body = get(url)
                data = json.loads(body) if status == 200 else None
            except Exception as e:                       # noqa: BLE001 - a probe reports, never stops
                status, data = repr(e)[:200], None
            report["feeds"][name] = {"url": url, "status": status}
            if data is not None:
                save(name, data)
                if kind == "board":
                    events = data.get("events", [])
                    report["feeds"][name]["events"] = len(events)
                    report["feeds"][name]["statuses"] = sorted({e.get("status", {}).get("type", {}).get("name", "?")
                                                                for e in events})
            print(f"  {name:<22} {status}  {report['feeds'][name].get('events', '')}", flush=True)
    for code, slugs in FD_SLUGS.items():
        for slug in slugs:
            for year in (now.year, now.year - 1):
                url = FIXTUREDOWNLOAD.format(slug=slug, year=year)
                name = f"fixturedownload_{code}_{year}"
                try:
                    status, body = get(url)
                    data = json.loads(body) if status == 200 and body.strip()[:1] in (b"[", b"{") else None
                except Exception as e:                   # noqa: BLE001
                    status, data = repr(e)[:200], None
                report["feeds"][name] = {"url": url, "status": status,
                                         "rows": len(data) if isinstance(data, list) else None}
                if data is not None:
                    save(name, data)
                print(f"  {name:<22} {status}  {report['feeds'][name]['rows']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
