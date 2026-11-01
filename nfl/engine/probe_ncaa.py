"""Probe the sources a college football (NCAA FBS) section of the app could
be built from, and keep what they return under nfl/data/ncaa_probe/ - the
sandbox cannot reach any of them. Run on GitHub's machines:

    python nfl/engine/probe_ncaa.py

A pick ships only if it can be measured, so the questions are: where do past
seasons' results and schedules come from (sportsdataverse's free cfb_schedules
release, built from ESPN), where do past betting lines come from (ESPN's
per-game PickCenter and its core odds API; collegefootballdata.com needs a key),
and what does this week look like (ESPN's FBS scoreboard, with odds).
"""

import gzip
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "ncaa_probe"
UA = {"User-Agent": "YourNFL college probe (github.com/Klukemma/YourMMA)"}
GH = "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases?per_page=100&page={page}"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"


def get(url, headers=None):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:500]
    except Exception as e:                               # noqa: BLE001 - a probe reports, never stops
        return repr(e)[:200], b""


def save(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(OUT / f"{name}.gz", "wb", mtime=0) as f:
        f.write(body)


def fetch(name, url, report, keep=True, headers=None):
    status, body = get(url, headers)
    entry = {"url": url, "status": status, "bytes": len(body)}
    if status != 200:
        entry["body"] = body[:300].decode("utf-8", "replace")
    elif keep:
        save(name, body)
    report[name] = entry
    print(f"  {name:<52} {status} {len(body)}", flush=True)
    return body if status == 200 else None


def main(now=None):
    import os
    now = now or datetime.now(timezone.utc)
    report = {"probed": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "requests": {}}
    req = report["requests"]
    gh = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {}

    # 1. sportsdataverse's releases: which hold college football, and their files.
    releases = []
    for page in (1, 2, 3):
        body = fetch(f"gh_releases_p{page}.json", GH.format(page=page), req, headers=gh)
        if not body:
            break
        batch = json.loads(body)
        releases += batch
        if len(batch) < 100:
            break
    cfb = [r for r in releases if "cfb" in r.get("tag_name", "").lower()]
    report["cfb_releases"] = {r["tag_name"]: sorted(a["name"] for a in r.get("assets", [])) for r in cfb}
    for r in cfb:
        assets = {a["name"]: a["browser_download_url"] for a in r.get("assets", [])}
        # A few seasons of each college release, csv first (the smallest to read).
        for year in ("2010", "2016", "2024", "2025", "2026"):
            pick = [n for n in assets if year in n and n.endswith(".csv.gz")] or \
                   [n for n in assets if year in n and n.endswith(".csv")]
            if pick and r["tag_name"] in ("cfb_schedules",) or (pick and "bet" in r["tag_name"].lower()):
                fetch(f"sdv_{r['tag_name']}_{pick[0]}", assets[pick[0]], req)

    # 2. This week and next on ESPN's FBS scoreboard (groups=80), with odds.
    for k in range(-1, 9):
        d = now + timedelta(days=k)
        fetch(f"espn_board_{d:%Y%m%d}.json", f"{ESPN}/scoreboard?groups=80&limit=300&dates={d:%Y%m%d}", req)
    fetch("espn_teams_fbs.json", f"{ESPN}/teams?groups=80&limit=300", req)
    fetch("espn_board_week.json", f"{ESPN}/scoreboard?groups=80&limit=300", req)

    # 3. Past lines: PickCenter (summary) and the core odds API, on finished
    # games across seasons (ids from the schedules files when present).
    ids = []
    for name in sorted(req):
        if name.startswith("sdv_cfb_schedules_") and req[name]["status"] == 200:
            import csv, io
            text = gzip.decompress((OUT / f"{name}.gz").read_bytes())
            if text[:2] == b"\x1f\x8b":
                text = gzip.decompress(text)
            rows = list(csv.DictReader(io.StringIO(text.decode("utf-8", "replace"))))
            col = next((c for c in ("game_id", "id", "event_id") if rows and c in rows[0]), None)
            if col:
                ids += [(name, r[col]) for r in rows[:: max(1, len(rows) // 4)][:4]]
    for src, gid in ids[:20]:
        fetch(f"espn_summary_{gid}.json", f"{ESPN}/summary?event={gid}", req)
        fetch(f"espn_core_odds_{gid}.json", f"{CORE}/events/{gid}/competitions/{gid}/odds", req)

    # 4. collegefootballdata.com without a key (does it answer at all?).
    fetch("cfbd_lines_nokey.json", "https://api.collegefootballdata.com/lines?year=2024&week=1", req)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
