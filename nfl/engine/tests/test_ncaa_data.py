"""College football data: the sportsdataverse releases and ESPN's FBS board,
read from fixtures cut from the probe of 2026-10-30 (tests/fixtures/ncaa/),
through a stub of the network."""

import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ncaa_data as nc

FIX = Path(__file__).parent / "fixtures" / "ncaa"
SEASONS = (2010, 2016, 2024, 2025, 2026)
PROBED = "2026-10-30T20:43:40Z"
TODAY = date(2026, 10, 30)


class Feed:
    """The network as the probe saw it: GitHub's releases API, the release
    assets and ESPN's teams list and boards. `fail` makes any URL holding
    one of its strings raise; `sizes` lies about an asset's listed size;
    `files` replaces an asset's body; `gone` lists an asset that then
    answers 404; api=False makes the API fail."""

    def __init__(self, fail=(), sizes=None, files=None, api=True, extra=(), gone=()):
        self.files = {}
        for y in SEASONS:
            self.files[f"cfb_schedules_{y}.csv.gz"] = (FIX / f"cfb_schedules_{y}.csv.gz").read_bytes()
            self.files[f"betting_{y}.csv"] = (FIX / f"betting_{y}.csv").read_bytes()
        for name in extra:
            self.files[name] = self.files[name.replace(".csv.gz", ".csv")] if name.startswith("betting") else b""
        self.files.update(files or {})
        self.fail, self.sizes, self.api, self.gone = tuple(fail), dict(sizes or {}), api, set(gone)
        self.urls, self.headers = [], {}

    def release(self, tag):
        prefix = "cfb_schedules_" if tag == nc.SCHEDULE_TAG else "betting_"
        return {"tag_name": tag, "assets": [
            {"name": n, "size": self.sizes.get(n, len(b)),
             "browser_download_url": nc.DOWNLOAD.format(tag=tag, name=n)}
            for n, b in self.files.items() if n.startswith(prefix) and b is not None]}

    def __call__(self, url, headers=None):
        self.urls.append(url)
        self.headers[url] = dict(headers or {})
        if any(f in url for f in self.fail):
            raise ConnectionError("503 Service Unavailable")
        if url.startswith("https://api.github.com/"):
            if not self.api:
                raise ConnectionError("403 rate limit exceeded")
            return json.dumps(self.release(url.rsplit("/", 1)[-1])).encode()
        if url == nc.TEAMS_URL:
            return (FIX / "teams.json").read_bytes()
        if url == nc.BOARD_URL:
            return (FIX / "board_week.json").read_bytes()
        if url.startswith(nc.BOARD_URL + "&dates="):
            day = FIX / f"board_{url[-8:]}.json"
            return (day if day.exists() else FIX / "board_20261101.json").read_bytes()
        name = url.rsplit("/", 1)[-1]
        if name in self.gone:
            return None
        return self.files.get(name)                 # None: a 404


def run(root, feed=None, seasons=SEASONS, **kw):
    logs = []
    feed = feed or Feed()
    result = nc.sync(seasons=seasons, today=TODAY, now=PROBED, fetch=feed, log=logs.append,
                     root=root, **kw)
    return result, logs, feed


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("ncaa")
    result, logs, feed = run(root)
    assert result["problems"] == [], logs
    return root


@pytest.fixture(scope="module")
def games(built):
    return nc.load_games(root=built)


def raw_schedule(season):
    return pd.read_csv(FIX / f"cfb_schedules_{season}.csv.gz", dtype=str, keep_default_na=False)


def game(games, gid):
    rows = games[games["game_id"] == gid]
    assert len(rows) == 1, gid
    return rows.iloc[0]


# ------------------------------------------------------------------ scope

def test_only_games_with_an_fbs_team_are_kept(games):
    assert ((games["home_division"] == "fbs") | (games["away_division"] == "fbs")).all()
    for gid in (302680047, 400872718, 401867873,   # FCS v FCS
                401873229,                         # Division III
                310223147, 400932767):             # all-star games (offseason, no division)
        assert gid not in set(games["game_id"])
    for season in SEASONS:
        raw = raw_schedule(season)
        fbs = (raw["home_division"] == "fbs") | (raw["away_division"] == "fbs")
        assert (games["season"] == season).sum() == fbs.sum()


def test_fcs_opponents_are_kept_with_a_flag(games):
    baylor = game(games, 400868876)                 # Baylor v Northwestern State, 2016
    assert baylor["fcs_game"] == 1 and baylor["fbs_game"] == 0 and baylor["away_division"] == "fcs"
    assert (games["fbs_game"] + games["fcs_game"] == 1).all()
    assert games["fcs_game"].sum() > 0 and games["fbs_game"].sum() > games["fcs_game"].sum()


def test_season_types_postseason_and_neutral_sites(games):
    assert set(games["season_type"]) == {"regular", "postseason"}
    cfp = game(games, 401677176)                    # CFP first round, on Texas' campus
    assert cfp["postseason"] == 1 and cfp["neutral"] == 0 and cfp["playoff_round"] == "First Round"
    bowls = games[(games["postseason"] == 1) & games["playoff_round"].isna()]
    assert bowls["neutral"].mean() > 0.8            # bowls are neutral, as ESPN flags them
    assert game(games, 401677085)["neutral"] == 0   # the Myrtle Beach Bowl is filed as not neutral
    assert game(games, 401859185)["neutral"] == 1   # Notre Dame - Navy in Foxborough


@pytest.mark.parametrize("gid,neutral,home_field", [
    (401677085, 0, -1),     # Myrtle Beach Bowl 2024: UTSA "home", at Coastal Carolina's Brooks Stadium
    (401769076, 1, -1),     # CFP title game, 2025 season: Indiana "home", at Miami's Hard Rock Stadium
    (302450164, 1, 1),      # Rutgers - Norfolk State 2010: filed neutral, at Rutgers' own stadium
    (401859185, 1, 0),      # Notre Dame - Navy in Foxborough: neutral, and neither's stadium
    (401677176, 0, 1),      # CFP first round on Texas' campus
    (401856634, 0, 1),      # Alabama - East Carolina in Tuscaloosa
])
def test_home_field_is_read_from_the_stadium_not_only_the_flag(games, gid, neutral, home_field):
    g = game(games, gid)
    assert (g["neutral"], g["home_field"]) == (neutral, home_field)


def test_home_field_follows_the_flag_unless_a_team_is_at_its_own_stadium(games):
    fixed = games[games["home_field"] != 1 - games["neutral"]]
    assert set(fixed["game_id"]) == {401677085, 401769076, 302450164}
    assert set(games["home_field"]) <= {-1, 0, 1}
    counts = nc.summary(games)["home_field_fixed"]
    assert counts.sum() == 3 and counts[2016] == 0


def test_home_field_rule():
    # Team 1 plays at venue 10 (twice) and once "at home" at 11; team 2 at
    # 20; team 3 splits 30 and 31 evenly; team 4 is only ever away.
    season = [2024] * 6
    venues = nc.home_venues(season, [1, 1, 1, 2, 3, 3], ["10", "10", "11", 20, 30, 31],
                            [0, 0, 0, 0, 0, 0])
    assert venues == {(2024, 1): {10}, (2024, 2): {20}, (2024, 3): {30, 31}}
    cases = [  # home, away, venue, neutral -> home_field
        (1, 2, 10, 0, 1), (1, 2, 10, 1, 1),       # at its own stadium, whatever the flag
        (1, 2, 20, 0, -1), (1, 2, 20, 1, -1),     # at the AWAY team's stadium
        (1, 2, 11, 0, 1),                         # a "home" game elsewhere keeps the flag
        (1, 2, 99, 1, 0), (1, 4, None, 0, 1),     # a neutral site; no venue keeps the flag
        (3, 4, 31, 1, 1),                         # a tie: both of its stadiums count
        (4, 1, 10, 0, -1),
    ]
    got = nc.home_field([2024] * len(cases), [c[0] for c in cases], [c[1] for c in cases],
                        [c[2] for c in cases], [c[3] for c in cases], venues)
    assert got == [c[4] for c in cases]
    # A stadium both teams call home: neither is at home.
    assert nc.home_field([2024], [1], [5], [10], [0], {**venues, (2024, 5): {10}}) == [0]
    # Another season's stadium does not count.
    assert nc.home_field([2025], [2], [1], [20], [1], venues) == [0]


def test_rank_gaps_find_a_ranked_team_with_a_rank_missing():
    t = pd.Timestamp
    g = pd.DataFrame({
        "season": [2016] * 4, "game_id": [1, 2, 3, 4],
        "kickoff": [t("2016-09-03"), t("2016-09-10"), t("2016-09-17"), t("2016-09-24")],
        "home_id": [333, 333, 333, 333], "away_id": [9, 8, 7, 6],
        "home_rank": [1, np.nan, 1, np.nan], "away_rank": [np.nan] * 4,
        "home_division": ["fbs"] * 4, "away_division": ["fcs"] * 4})
    assert nc.rank_gaps(g).tolist() == [False, True, False, False]


def test_kickoffs_are_naive_utc_in_both_date_formats(games):
    got = nc.kickoffs(["2026-09-05T16:00:00.000Z", "2011-01-22T21:00Z", "2026-10-29T23:30Z"])
    assert got.tolist() == [pd.Timestamp("2026-09-05 16:00"), pd.Timestamp("2011-01-22 21:00"),
                            pd.Timestamp("2026-10-29 23:30")]
    assert game(games, 401856634)["kickoff"] == pd.Timestamp("2026-09-05 16:00")
    assert games["kickoff"].dt.tz is None and games["kickoff"].is_monotonic_increasing
    assert games["game_id"].is_unique


def test_a_kickoff_time_not_set_is_a_date_not_a_moment(games):
    tbd = games[games["time_tbd"]]
    assert len(tbd) >= 2
    # ESPN files it as midnight Eastern on the day.
    assert set(tbd["kickoff"].dt.strftime("%H:%M")) <= {"04:00", "05:00"}


def test_played_needs_completed_and_both_scores(games):
    cancelled = game(games, 401640992)              # App State - Liberty 2024: 0-0, cancelled
    assert not cancelled["played"] and np.isnan(cancelled["home_score"])
    future = games[(games["season"] == 2026) & ~games["played"]]
    assert len(future) and future["home_score"].isna().all() and not future["has_line"].any()
    alabama = game(games, 401856634)
    assert alabama["played"] and alabama["result"] == 38 and alabama["total"] == 58


# ------------------------------------------------------------------ the line

def test_the_line_joins_by_game_id_from_the_home_side(games):
    ala = game(games, 401856634)                    # Alabama -28.5 v East Carolina, 52.5
    assert ala["has_line"] and ala["spread_line"] == 28.5 and ala["total_line"] == 52.5
    assert ala["home_favorite"] == 1.0 and ala["line_source"] == "core_odds_api"
    fsu = game(games, 401752665)                    # Florida State +13.5 at home to Alabama
    assert fsu["spread_line"] == -13.5 and fsu["home_favorite"] == 0.0
    lined = games[games["has_line"] & games["played"] & (games["spread_line"] != 0)]
    assert ((lined["spread_line"] > 0) == (lined["result"] > 0)).mean() > 0.6


def test_line_rows_of_games_out_of_scope_are_not_loaded(games):
    lines = pd.read_csv(FIX / "betting_2024.csv")
    out = set(lines["game_id"]) - set(games["game_id"])
    assert 401628579 in out                         # Delaware v Bryant, FCS v FCS, has a line
    assert not set(games["game_id"]) & (out - set(lines["game_id"]))


def test_placeholders_are_missing_lines_not_2_5_and_55_5(games):
    old = games[games["season"] == 2010]            # every 2010 row is "default"
    assert len(old) and not old["has_line"].any()
    assert old[["spread_line", "total_line", "home_favorite"]].isna().all().all()
    for gid in (400869433, 401628456, 401640992):   # "default" rows in 2016 and 2024
        g = game(games, gid)
        assert not g["has_line"] and np.isnan(g["spread_line"]) and np.isnan(g["total_line"])
    duke = game(games, 401754517)                   # 2025 placeholder with a total of 50.5
    assert np.isnan(duke["total_line"]) and np.isnan(duke["home_favorite"])
    assert not games.loc[games["line_source"] == "default", "has_line"].any()
    # The rule is the source, not the numbers: Delaware at WKU really closed -2.5 / 55.5.
    wku = game(games, 401871071)
    assert wku["has_line"] and (wku["spread_line"], wku["total_line"]) == (2.5, 55.5)


def test_pickcenter_rows_keep_who_was_favoured_and_no_number(games):
    buffalo = game(games, 400869301)                # 2016: Kent State favoured at Buffalo
    assert not buffalo["has_line"] and np.isnan(buffalo["spread_line"])
    assert buffalo["home_favorite"] == 0.0 and buffalo["line_source"] == "summary_pickcenter"
    assert game(games, 400869719)["home_favorite"] == 1.0


def test_a_total_without_a_spread_is_not_loaded(games):
    """2016's pickcenter rows with a lone 55.0 total (Rutgers - Howard)."""
    raw = pd.read_csv(FIX / "betting_2016.csv").set_index("game_id")
    assert raw.loc[400869633, "over_under"] == 55.0 and np.isnan(raw.loc[400869633, "home_team_spread"])
    rutgers = game(games, 400869633)
    assert not rutgers["has_line"] and np.isnan(rutgers["total_line"])
    assert rutgers["home_favorite"] == 1.0
    assert not (games["total_line"].notna() & games["spread_line"].isna()).any()


def test_a_game_with_no_line_row_has_no_source(games):
    g = games[(games["season"] == 2026) & ~games["played"]]
    assert g["line_source"].isna().all()


def _core(gid):
    return json.loads((FIX / f"core_odds_{gid}.json").read_text())["items"]


def test_the_release_line_is_espns_close_not_the_open():
    """The evidence for "closing" in the module notes, kept as a test."""
    lines = pd.concat([pd.read_csv(FIX / f"betting_{y}.csv") for y in SEASONS]).set_index("game_id")
    checked = 0
    for gid in (401856634, 401752665, 401628319):
        main = [i for i in _core(gid) if "Live" not in i["provider"]["name"]][0]
        close = float(main["homeTeamOdds"]["close"]["pointSpread"]["american"])
        opened = float(main["homeTeamOdds"]["open"]["pointSpread"]["american"])
        row = lines.loc[gid]
        assert row["home_team_spread"] == close == main["spread"]
        assert row["over_under"] == main["overUnder"]
        moved = opened != close or main["overUnder"] != _total_open(main)
        checked += moved
    assert checked == 3                             # in each, open and close differ
    # 2016: the release took a book's last number, not ESPN's "Opening" or consensus.
    items = {i["provider"]["name"]: i for i in _core(400868876)}
    assert lines.loc[400868876, "home_team_spread"] == items["5Dimes.eu"]["spread"] == -50
    assert items["Opening"]["spread"] == items["consensus"]["spread"] == -48.5
    assert _core(302450005) == []                   # 2010: ESPN has no odds at all


def _total_open(item):
    return float(item["open"]["total"]["american"].lstrip("ou")) if item.get("open", {}).get("total") else None


def test_the_live_board_reads_the_same_provider_as_the_2026_history():
    board = json.loads((FIX / "board_week.json").read_text())
    providers = {o["provider"]["id"] for e in board["events"] for o in e["competitions"][0].get("odds") or []}
    assert providers == {"100"} == {i["provider"]["id"] for i in _core(401856634)}
    pick = json.loads((FIX / "summary_pickcenter_401856634.json").read_text())["pickcenter"]
    assert pick[0]["provider"]["name"] == "Draft Kings" and pick[0]["spread"] == -28.5


# ------------------------------------------------------------------ the live board

def test_the_board_parses_kickoff_status_ranks_and_the_line():
    games, teams = nc.parse_board((FIX / "board_week.json").read_bytes())
    by = {g["game_id"]: g for g in games}
    navy = by[401859185]                            # Notre Dame (5) v Navy, neutral, Saturday
    assert navy["kickoff"] == "2026-10-31T16:00:00Z" and navy["neutral"] and navy["state"] == "pre"
    assert navy["home"] == "Navy Midshipmen" and navy["away_rank"] == 5 and navy["home_rank"] is None
    line = navy["line"]
    assert line["provider"] == "Draft Kings" and line["spread_line"] == -30.5 and line["total_line"] == 56.5
    assert (line["home_ml"], line["away_ml"]) == (2000, -6500)
    assert (line["spread_open"], line["total_open"], line["home_ml_open"]) == (-28.5, 53.5, 2200)
    assert line["home_spread_odds"] == -110 and line["over_odds"] == -108
    final = by[401871071]                           # Delaware at WKU, final, no odds left
    assert final["completed"] and final["line"] is None
    assert (final["home_score"], final["away_score"]) == (21, 35)
    assert navy["home_score"] is None and navy["season_type"] == "regular"
    assert {t["team_id"] for t in teams} >= {87, 2426, 98, 48}


def test_the_board_flags_a_kickoff_time_not_set():
    games, _ = nc.parse_board((FIX / "board_20261107.json").read_bytes())
    tbd = [g for g in games if g["time_tbd"]]
    assert len(tbd) == 3 and {g["kickoff"] for g in tbd} == {"2026-11-07T05:00:00Z"}
    assert nc.parse_board((FIX / "board_20261101.json").read_bytes()) == ([], [])


def test_the_main_provider_is_the_lowest_priority_number_zero_included():
    odds = [{"provider": {"id": "100", "name": "Draft Kings", "priority": 1}, "spread": -3.0},
            {"provider": {"id": "58", "name": "ESPN BET", "priority": 0}, "spread": -4.5},
            {"provider": {"id": "9", "name": "No priority"}, "spread": -1.0}]
    assert nc.parse_line(odds)["provider"] == "ESPN BET" and nc.parse_line(odds)["spread_line"] == 4.5
    assert nc.parse_line([odds[2], odds[0]])["provider"] == "Draft Kings"


@pytest.mark.parametrize("text,value", [("+2000", 2000), ("-110", -110), ("EVEN", 100), ("OFF", None),
                                        (None, None)])
def test_american_odds(text, value):
    assert nc._american(text) == value


@pytest.mark.parametrize("text,value", [("+30.5", 30.5), ("-3", -3), ("o56.5", 56.5), ("u41", 41),
                                        ("PK", 0.0), ("", None), (None, None)])
def test_points(text, value):
    assert nc._points(text) == value


def _board_game(state="pre", line=True, kickoff="2026-10-31T16:00:00Z", tbd=False):
    g = {"game_id": 1, "kickoff": kickoff, "time_tbd": tbd, "state": state, "completed": state == "post"}
    g["line"] = {k: None for k in nc.LINE_FIELDS if k != "line_seen_at"} if line else None
    if line:
        g["line"].update(spread_line=-3.5, total_line=50.5, provider="Draft Kings")
    return g


def test_the_week_keeps_the_last_line_seen_before_kickoff():
    friday = nc.merge_week([_board_game()], [], "2026-10-30T20:00:00Z")
    assert friday[0]["spread_line"] == -3.5 and friday[0]["line_seen_at"] == "2026-10-30T20:00:00Z"
    # Sunday: final, the board has dropped the odds - Friday's line stays, dated Friday.
    sunday = nc.merge_week([_board_game("post", line=False)], friday, "2026-11-01T10:00:00Z")
    assert sunday[0]["spread_line"] == -3.5 and sunday[0]["line_seen_at"] == "2026-10-30T20:00:00Z"
    # Under way with odds on the board (live or not, unknown): never taken.
    live = _board_game("in")
    live["line"]["spread_line"] = -17.5
    during = nc.merge_week([live], friday, "2026-10-31T17:00:00Z")
    assert during[0]["spread_line"] == -3.5
    assert nc.merge_week([live], [], "2026-10-31T17:00:00Z")[0]["spread_line"] is None
    # A line on file seen after kickoff is not a pre-game line.
    late = [{**friday[0], "line_seen_at": "2026-10-31T16:30:00Z"}]
    assert nc.merge_week([_board_game("post", line=False)], late, "2026-11-01T10:00:00Z")[0]["spread_line"] is None
    # A time not set yet: a pre-game line is taken whatever the clock says.
    tbd = nc.merge_week([_board_game(kickoff="2026-11-07T05:00:00Z", tbd=True)], [], "2026-11-07T12:00:00Z")
    assert tbd[0]["spread_line"] == -3.5


def test_load_week(built):
    w = nc.load_week(built)
    assert len(w) == len({*w["game_id"]}) and w["kickoff"].is_monotonic_increasing
    assert (w["fetched_at"] == pd.Timestamp("2026-10-30 20:43:40")).all()
    navy = w[w["game_id"] == 401859185].iloc[0]
    assert navy["line_seen_at"] == pd.Timestamp("2026-10-30 20:43:40") and navy["spread_line"] == -30.5
    assert navy["fbs_game"] == 1 and navy["fcs_game"] == 0 and navy["neutral"] == 1
    assert navy["home_field"] == 0 and navy["venue_id"] == 3738
    assert set(w["home_field"]) <= {-1, 0, 1} and w["venue_id"].notna().all()
    finals = w[w["status"] == "STATUS_FINAL"]
    assert len(finals) == 3 and finals["played"].all() and finals["spread_line"].isna().all()
    lined = w[w["spread_line"].notna()]
    assert (lined["line_seen_at"] < lined["kickoff"]).all() or lined["time_tbd"].any()
    # A team the schedule on file never met: its division is unknown, not guessed.
    merrimack = w[w["game_id"] == 401858290].iloc[0]
    assert np.isnan(merrimack["fbs_game"])
    text = (built / "week.json").read_text()
    assert text.count("\n") == len(w) + 2 and json.loads(text)["week"] == 9


# ------------------------------------------------------------------ teams

def test_teams_parse_with_colours_and_missing_ones():
    teams = {t["team_id"]: t for t in nc.parse_teams((FIX / "teams.json").read_bytes())}
    ala = teams[333]
    assert (ala["name"], ala["abbreviation"], ala["color"], ala["alt_color"]) == \
        ("Alabama Crimson Tide", "ALA", "9e1b32", "ffffff")
    assert ala["logo"].endswith("/500/333.png")
    assert teams[2019]["color"] is None and teams[2001]["alt_color"] is None


def test_teams_merge_keeps_what_a_later_source_lacks():
    merged = nc.merge_teams([{"team_id": 1, "name": "Old", "color": "000000"}],
                            [{"team_id": 1, "name": "New", "color": None}])
    assert merged.loc[0, "name"] == "New" and merged.loc[0, "color"] == "000000"


def test_load_teams_covers_every_team_in_the_schedule(built, games):
    teams = nc.load_teams(built).set_index("team_id")
    ids = set(games["home_id"]) | set(games["away_id"])
    assert ids <= set(teams.index) and teams.loc[list(ids), "name"].notna().all()
    assert teams.loc[333, "color"] == "9e1b32" and teams.loc[333, "fbs"]
    assert teams.loc[2426, "color"] == "00225b"     # Navy: from the board, not the teams list
    # A team only the schedule names: no colours, shown as missing.
    only = teams[teams["color"].isna() & teams["division"].notna()]
    assert len(only) and only["name"].notna().all()


# ------------------------------------------------------------------ sync and the snapshot

def test_sync_writes_one_file_per_season(built):
    for kind in ("schedules", "lines"):
        assert nc.seasons_on_file(built, kind) == list(SEASONS)
    assert (built / "teams.csv.gz").exists() and (built / "week.json").exists()


def test_the_snapshot_is_deterministic(tmp_path, built):
    run(tmp_path)
    for path in sorted(built.rglob("*.csv.gz")):
        twin = tmp_path / path.relative_to(built)
        assert twin.read_bytes() == path.read_bytes(), path.name
        assert path.read_bytes()[4:8] == b"\0\0\0\0"            # gzip mtime 0
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.csv.gz")}
    result, logs, _ = run(tmp_path)
    assert {v for f in result["files"].values() for v in f.values()} == {"unchanged"}
    assert result["teams"] == "unchanged"
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*.csv.gz")} == stamps


def test_stored_text_is_the_release_text(built):
    """Rows are kept as published - no float reformatting - in game id order."""
    lines = gzip.decompress((built / "lines" / "2024.csv.gz").read_bytes()).decode()
    published = (FIX / "betting_2024.csv").read_text().splitlines()
    assert sorted(lines.splitlines()[1:]) == sorted(published[1:])
    ids = [int(l.split(",")[0]) for l in lines.splitlines()[1:]]
    assert ids == sorted(ids)


def _seed(tmp_path):
    run(tmp_path, seasons=[2026])
    return {p: p.read_bytes() for p in tmp_path.rglob("*.csv.gz")}


def test_a_failed_download_keeps_what_is_on_file(tmp_path):
    before = _seed(tmp_path)
    result, logs, _ = run(tmp_path, Feed(fail=["cfb_schedules_2026"]), seasons=[2026])
    assert result["files"][2026] == {"schedule": "not fetched", "lines": "unchanged"}
    assert result["problems"] == ["schedule 2026"]
    assert (tmp_path / "schedules" / "2026.csv.gz").read_bytes() == before[tmp_path / "schedules" / "2026.csv.gz"]
    assert any("kept what is on file" in l for l in logs)
    assert any(l.startswith("   KEPT ON FILE") for l in logs)


@pytest.mark.parametrize("case", ["short", "gzip", "cut", "field", "fewer", "columns", "gone"])
def test_a_truncated_or_unreadable_download_keeps_what_is_on_file(tmp_path, case):
    before = _seed(tmp_path)
    sched = (FIX / "cfb_schedules_2026.csv.gz").read_bytes()
    bet = (FIX / "betting_2026.csv").read_bytes()
    feed = {
        "short": Feed(sizes={"betting_2026.csv": len(bet) + 100}),
        "gzip": Feed(files={"cfb_schedules_2026.csv.gz": sched[: len(sched) // 2]}, api=False),
        "cut": Feed(files={"betting_2026.csv": bet[:-25]}, api=False),
        # cut inside the last field: '...,core_odds' - as wide as the header
        "field": Feed(files={"betting_2026.csv": bet[:-5]}, api=False),
        "fewer": Feed(files={"betting_2026.csv": b"\n".join(bet.split(b"\n")[:4]) + b"\n"}),
        "columns": Feed(files={"betting_2026.csv": bet.replace(b"home_team_spread", b"spread_home")}),
        # listed by the release, then a 404: a problem, not "not published"
        "gone": Feed(gone=["betting_2026.csv"]),
    }[case]
    result, logs, _ = run(tmp_path, feed, seasons=[2026])
    assert result["problems"], logs
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.csv.gz")} == before


def test_a_body_shorter_than_its_content_length_is_refused():
    assert nc.whole(b"abc", {"Content-Length": "3"}) == b"abc"
    with pytest.raises(IOError, match="cut short"):
        nc.whole(b"ab", {"content-length": "3"})
    # Decoded by the HTTP layer: the header counts the compressed bytes.
    assert nc.whole(b"abcdef", {"Content-Length": "3", "Content-Encoding": "gzip"}) == b"abcdef"
    assert nc.whole(b"ab", {}) == b"ab"


def test_cut_off():
    assert nc.cut_off(b"a,b\n1,2") and nc.cut_off(b"a,b\n1,")
    assert not nc.cut_off(b"a,b\n1,2\n") and not nc.cut_off(b"a,b") and not nc.cut_off(b"")


def test_a_season_not_published_is_not_a_problem(tmp_path):
    result, logs, feed = run(tmp_path, seasons=[2027])
    assert result["problems"] == [] and nc.seasons_on_file(tmp_path) == []
    assert any("schedule 2027: not published" in l for l in logs)


def test_without_the_api_the_fixed_urls_are_used_and_the_fresh_csv_first(tmp_path):
    result, logs, feed = run(tmp_path, Feed(api=False), seasons=[2026])
    assert result["problems"] == []
    assert nc.DOWNLOAD.format(tag=nc.BETTING_TAG, name="betting_2026.csv") in feed.urls
    assert not any(u.endswith("betting_2026.csv.gz") for u in feed.urls)
    assert any("GitHub's API did not answer" in l for l in logs)


def test_a_season_only_published_as_csv_gz_is_read_from_it(tmp_path):
    bet = (FIX / "betting_2016.csv").read_bytes()
    feed = Feed(files={"betting_2016.csv": None, "betting_2016.csv.gz": gzip.compress(bet, mtime=0)})
    result, logs, _ = run(tmp_path, feed, seasons=[2016])
    assert result["files"][2016]["lines"] == "written" and result["problems"] == []
    assert gzip.decompress((tmp_path / "lines" / "2016.csv.gz").read_bytes()).count(b"\n") == bet.count(b"\n")


def test_the_board_failing_keeps_the_week_on_file(tmp_path):
    run(tmp_path, seasons=[2026])
    week = (tmp_path / "week.json").read_text()
    result, logs, _ = run(tmp_path, Feed(fail=["scoreboard?groups=80&limit=300"]), seasons=[2026])
    assert "live board" in result["problems"] and (tmp_path / "week.json").read_text() == week


def test_by_default_only_the_current_season_and_full_from_2004(tmp_path):
    _, _, feed = run(tmp_path, seasons=None)
    assert any("cfb_schedules_2026" in u for u in feed.urls)
    assert not any("cfb_schedules_2025" in u for u in feed.urls)
    result, _, _ = run(tmp_path, full=True, seasons=None, board=False)
    assert result["seasons"] == list(range(2004, 2027))
    assert result["problems"] == []                 # seasons the stub lacks are "not published"


@pytest.mark.parametrize("day,fetched", [(date(2027, 2, 10), [2026]), (date(2027, 3, 10), [2026, 2027]),
                                         (date(2027, 4, 1), [2027])])
def test_last_season_is_refreshed_through_march(tmp_path, day, fetched):
    result = nc.sync(today=day, now=PROBED, fetch=Feed(), log=lambda _: None, root=tmp_path, board=False)
    assert result["seasons"] == fetched and result["problems"] == []


@pytest.mark.parametrize("day,season", [(date(2026, 8, 29), 2026), (date(2027, 1, 19), 2026),
                                        (date(2027, 2, 10), 2026), (date(2027, 3, 1), 2027)])
def test_current_season(day, season):
    assert nc.current_season(day) == season


def test_the_token_goes_to_githubs_api_only_and_is_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    _, logs, feed = run(tmp_path, Feed(fail=["cfb_schedules_2026"]), seasons=[2026])
    for url, headers in feed.headers.items():
        has = headers.get("Authorization") == "Bearer secret-token"
        assert has == url.startswith("https://api.github.com/"), url
    assert not any("secret" in l for l in logs)


@pytest.mark.snapshot
def test_the_committed_snapshot_loads():
    """The snapshot as the runner last committed it (reported, never a gate)."""
    g = nc.load_games()
    assert len(g) and g["game_id"].is_unique and g["kickoff"].is_monotonic_increasing
    assert ((g["home_division"] == "fbs") | (g["away_division"] == "fbs")).all()
    lined = g[g["has_line"] & g["played"] & (g["spread_line"] != 0)]
    assert ((lined["spread_line"] > 0) == (lined["result"] > 0)).mean() > 0.6
    assert set(g["home_field"]) <= {-1, 0, 1} and (g["home_field"] == 1 - g["neutral"]).mean() > 0.99
