"""One-off, local (2026-10-29): the snapshot moves to one file per league and
season, and the leagues the app does not run on yet are seeded from the
league probe's downloads so they can be measured before the runner's own
sync lands. Run once, in the sandbox (no network):

    python soccer/engine/migrate_snapshot.py

1. soccer/data/matches.csv.gz (the ten divisions, every season since
   2005-06, as synced on GitHub's machines up to 2026-10-27) is split into
   soccer/data/matches/{league}/{season}.csv.gz, and load_matches() must
   return exactly the frame the single file gave before it is deleted.
2. From tests/fixtures/feeds/leagues/ (probe_leagues.py, run on GitHub's
   machines 2026-10-29 12:43 UTC), through the same sync code a runner
   uses (a fetch that answers from those files and 404s everything else):
     E2 E3 EC SC0 SC1 SC2 SC3 N1 B1 P1 T1 G1   seasons 2025 and 2026
     ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA
                                               every season in the file
   The runner's next sync replaces each current season with a fresh
   download; its next full sync adds the new divisions' seasons before 2025.
"""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soccer_data as sd  # noqa: E402

PROBE = Path(__file__).resolve().parent / "tests" / "fixtures" / "feeds" / "leagues"
PROBED = "2026-10-29"


def probe_fetch(probe=PROBE):
    """A fetch that answers from the probe's saved downloads, by URL."""
    report = json.loads((probe / "probe_report.json").read_text())["requests"]
    by_url = {r["url"]: probe / f"{name}.gz" for name, r in report.items()
              if r.get("status") == 200 and name.startswith("fd_") and (probe / f"{name}.gz").exists()}

    def fetch(url):
        path = by_url.get(url)
        return gzip.decompress(path.read_bytes()) if path else None
    return fetch


def main(log=print):
    if sd.LEGACY_MATCHES.exists():
        old = sd.load_matches(sd.LEGACY_MATCHES)
        sd.migrate(sd.LEGACY_MATCHES, sd.MATCHES, log=log)
        pd.testing.assert_frame_equal(old, sd.load_matches(sd.MATCHES))
        log(f"   load_matches(): the same {len(old)} rows, {old.shape[1]} columns, dtypes and order as the single file")
    new = [code for code in sd.ALL_LEAGUES if code not in sd.LEAGUES]
    sd.sync_matches(new, full=True, today=PROBED, fetch=probe_fetch(), log=lambda m: None)
    every = sd.load_matches(leagues="all")
    for code, n in every.groupby("league").size().items():
        log(f"   {code:<4} {n:6d} matches, seasons {sorted(set(every.loc[every['league'] == code, 'season']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
