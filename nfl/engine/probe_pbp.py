"""Probe nflverse's play-by-play before the engine reads it for coaching
tendencies (pass rate over expected, red-zone run/pass calling, fourth-down
aggressiveness, pace): which columns exist, how complete they are by season,
and two real games to build and test the aggregation against. Saved under
nfl/data/pbp_probe/ - the sandbox cannot reach nflverse's releases. Run on
GitHub's machines:

    python nfl/engine/probe_pbp.py
"""

import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
OUT = Path(__file__).resolve().parents[1] / "data" / "pbp_probe"
SEASONS = (2006, 2016, 2025)
KEY = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "home_coach",
       "away_coach", "play_type", "pass", "rush", "qb_dropback", "xpass", "pass_oe", "down", "ydstogo",
       "yardline_100", "goal_to_go", "qtr", "game_seconds_remaining", "wp", "vegas_wp", "score_differential",
       "epa", "success", "touchdown", "rush_touchdown", "pass_touchdown", "fourth_down_converted",
       "fourth_down_failed", "shotgun", "no_huddle", "qb_scramble", "two_point_attempt", "penalty",
       "rusher_player_id", "receiver_player_id", "passer_player_id", "td_player_id", "air_yards", "cpoe"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for season in SEASONS:
        req = urllib.request.Request(URL.format(season=season), headers={"User-Agent": "YourNFL pbp probe"})
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
        df = pd.read_csv(io.BytesIO(body), compression="gzip", low_memory=False)
        report[season] = {
            "bytes": len(body), "rows": len(df), "games": int(df["game_id"].nunique()),
            "columns": list(df.columns),
            "present": {k: (k in df.columns) for k in KEY},
            "nonnull": {k: round(float(df[k].notna().mean()), 4) for k in KEY if k in df.columns},
            "play_types": df["play_type"].value_counts(dropna=False).head(12).to_dict() if "play_type" in df else None,
        }
        print(season, len(df), "rows", report[season]["games"], "games", flush=True)
        if season == SEASONS[-1]:
            games = sorted(df["game_id"].unique())[:2]
            sample = df[df["game_id"].isin(games)]
            with gzip.GzipFile(OUT / f"pbp_sample_{season}.csv.gz", "wb", mtime=0) as f:
                f.write(sample.to_csv(index=False).encode())
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
