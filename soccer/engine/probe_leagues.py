"""Probe what every league the app could add looks like in its three
sources, and keep the answers as test fixtures (under
tests/fixtures/feeds/leagues/) - the sandbox cannot reach any of them.

Stake lists some ninety soccer leagues. A league ships in the app only if it
can be measured - results AND pre-match prices over past seasons - so the
first question is which of them football-data.co.uk carries, and in what
shape: its 22 European divisions (one CSV per season, with match statistics
for most) and its "extra" leagues (one CSV for every season, closing prices
only). The second is whether the free feeds (ESPN's scoreboard,
fixturedownload.com) list their fixtures and scores, and under which club
names. A few hundred requests; run on GitHub's machines:

    python soccer/engine/probe_leagues.py
"""

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tests" / "fixtures" / "feeds" / "leagues"
UA = {"User-Agent": "YourSoccer leagues probe (github.com/Klukemma/YourMMA)"}

FD = "https://www.football-data.co.uk"
# football-data.co.uk's European divisions the app does not sync yet, and its
# extra leagues (country codes as on its "new" pages).
MAIN_NEW = ["E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "N1", "B1", "P1", "T1", "G1"]
EXTRA = ["ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS",
         "SWE", "SWZ", "USA"]
FD_PAGES = ["data.php", "all_new_data.php", "fixtures.csv", "new_league_fixtures.csv", "matches.php",
            "matches_new_leagues.php"]

# ESPN league codes: the divisions above, the second tiers the app already
# rates, the European cups, and leagues Stake lists that football-data does
# not carry (schedules only - they could not be measured against prices).
ESPN = ["eng.2", "eng.3", "eng.4", "eng.5", "sco.1", "sco.2", "sco.3", "sco.4", "ned.1", "bel.1", "por.1",
        "tur.1", "gre.1", "ger.2", "ita.2", "esp.2", "fra.2",
        "arg.1", "aut.1", "bra.1", "chn.1", "den.1", "fin.1", "irl.1", "jpn.1", "mex.1", "nor.1", "pol.1",
        "rou.1", "rus.1", "swe.1", "sui.1", "usa.1",
        "uefa.champions", "uefa.europa", "uefa.europa.conf",
        "ksa.1", "aus.1", "kor.1", "ind.1", "col.1", "chi.1", "uru.1", "ecu.1", "per.1", "par.1", "cze.1",
        "cro.1", "srb.1", "ukr.1", "isr.1", "egy.1", "rsa.1", "uae.1", "qat.1", "ned.2", "por.2", "bra.2",
        "arg.2", "usa.usl.1", "eng.fa", "eng.league_cup", "esp.copa_del_rey", "ita.coppa_italia",
        "ger.dfb_pokal", "fra.coupe_de_france", "conmebol.libertadores", "conmebol.sudamericana"]
ESPN_BOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={day}"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams"
ESPN_LEAGUES = "https://sports.core.api.espn.com/v2/sports/soccer/leagues?limit=1000"

FXD = "https://fixturedownload.com/feed/json/{slug}-{year}"
FXD_SLUGS = ["championship", "league-one", "league-two", "national-league", "scottish-premiership",
             "scottish-championship", "eredivisie", "primeira-liga", "belgian-pro-league", "super-lig",
             "super-league-greece", "mls", "a-league", "a-league-men", "j1-league", "j-league", "brasileirao",
             "serie-b", "2-bundesliga", "segunda-division", "ligue-2", "champions-league", "europa-league",
             "conference-league", "saudi-pro-league", "allsvenskan", "eliteserien", "liga-mx"]


def get(url):
    import requests
    r = requests.get(url, timeout=60, headers=UA)
    return r.status_code, r.content


def save(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.gz"
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(body)
    return path


def fetch(name, url, report, keep=True):
    try:
        status, body = get(url)
    except Exception as e:                               # noqa: BLE001 - a probe reports, never stops
        report[name] = {"url": url, "status": repr(e)[:200]}
        print(f"  {name:<44} {repr(e)[:60]}", flush=True)
        return None
    entry = {"url": url, "status": status, "bytes": len(body)}
    if status == 200 and keep:
        save(name, body)
    elif status != 200:
        entry["body"] = body[:200].decode("utf-8", "replace")
    report[name] = entry
    print(f"  {name:<44} {status} {len(body)}", flush=True)
    return body if status == 200 else None


def main(now=None):
    now = now or datetime.now(timezone.utc)
    report = {"probed": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "requests": {}}
    req = report["requests"]
    for page in FD_PAGES:
        fetch(f"fd_{page}", f"{FD}/{page}", req)
    for div in MAIN_NEW:
        for code in ("2627", "2526"):
            fetch(f"fd_{div}_{code}.csv", f"{FD}/mmz4281/{code}/{div}.csv", req)
    for c in EXTRA:
        fetch(f"fd_new_{c}.csv", f"{FD}/new/{c}.csv", req)

    fetch("espn_leagues.json", ESPN_LEAGUES, req)
    days = [now + timedelta(days=k) for k in range(-3, 11)]
    for lg in ESPN:
        teams = fetch(f"espn_{lg}_teams.json", ESPN_TEAMS.format(lg=lg), req)
        if teams is None:
            continue                                     # unknown code: no boards
        events = {}
        for d in days:
            body = fetch(f"espn_{lg}_board_{d:%Y%m%d}", ESPN_BOARD.format(lg=lg, day=f"{d:%Y%m%d}"), req,
                         keep=False)
            if body:
                for e in json.loads(body).get("events", []):
                    events[e["id"]] = e
        save(f"espn_{lg}_boards.json", json.dumps(sorted(events.values(), key=lambda e: (e.get("date", ""), e["id"])),
                                                  separators=(",", ":"), sort_keys=True).encode())
        req[f"espn_{lg}_boards.json"] = {"events": len(events)}
    for slug in FXD_SLUGS:
        for year in (now.year, now.year - 1):
            body = fetch(f"fxd_{slug}_{year}.json", FXD.format(slug=slug, year=year), req, keep=False)
            if body and body.strip()[:1] == b"[":
                save(f"fxd_{slug}_{year}.json", body)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
