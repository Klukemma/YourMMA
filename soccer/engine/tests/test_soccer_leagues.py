"""Every league football-data.co.uk carries, on the league probe's real
downloads (tests/fixtures/feeds/leagues/, probe_leagues.py, GitHub's
machines 2026-10-29 12:43 UTC): the registry, both parsers, the clock its
times are on (against ESPN's and fixturedownload.com's UTC kickoffs), the
fixtures, the season-file snapshot and the sync."""

import collections
import difflib
import gzip
import json
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import soccer_data as sd

PROBE = Path(__file__).resolve().parent / "fixtures" / "feeds" / "leagues"
NEW_MAIN = ["E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "N1", "B1", "P1", "T1", "G1"]
# Each league's own clock, for the checks only (the parser never needs it).
ZONES = {"E2": "Europe/London", "E3": "Europe/London", "EC": "Europe/London", "SC0": "Europe/London",
         "SC1": "Europe/London", "SC2": "Europe/London", "SC3": "Europe/London", "N1": "Europe/Amsterdam",
         "B1": "Europe/Brussels", "P1": "Europe/Lisbon", "T1": "Europe/Istanbul", "G1": "Europe/Athens",
         "ARG": "America/Argentina/Buenos_Aires", "AUT": "Europe/Vienna", "BRA": "America/Sao_Paulo",
         "CHN": "Asia/Shanghai", "DNK": "Europe/Copenhagen", "FIN": "Europe/Helsinki", "IRL": "Europe/Dublin",
         "JPN": "Asia/Tokyo", "MEX": "America/Mexico_City", "NOR": "Europe/Oslo", "POL": "Europe/Warsaw",
         "ROU": "Europe/Bucharest", "RUS": "Europe/Moscow", "SWE": "Europe/Stockholm", "SWZ": "Europe/Zurich",
         "USA": "America/New_York"}
SAME_CLOCK_AS_UK = {"E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "P1", "IRL"}


def raw(name):
    return gzip.decompress((PROBE / f"{name}.gz").read_bytes())


@lru_cache(maxsize=None)
def extra(code):
    skipped = collections.Counter()
    rows = sd.parse_extra(raw(f"fd_new_{code}.csv"), league=code, skipped=skipped)
    return sd.frame(rows), dict(skipped)


@lru_cache(maxsize=None)
def main(div, code):
    return sd.frame(sd.parse(raw(f"fd_{div}_{code}.csv"), int("20" + code[:2]), leagues=[div]))


@lru_cache(maxsize=None)
def extra_fixtures():
    return sd.frame(sd.parse_extra(raw("fd_new_league_fixtures.csv")))


# ------------------------------------------------------------- the registry

def test_the_ten_the_pipeline_runs_on_are_untouched():
    assert sd.LEAGUES == {
        "E0": ("Premier League", "England", 1, "soccer_epl"),
        "SP1": ("La Liga", "Spain", 1, "soccer_spain_la_liga"),
        "I1": ("Serie A", "Italy", 1, "soccer_italy_serie_a"),
        "D1": ("Bundesliga", "Germany", 1, "soccer_germany_bundesliga"),
        "F1": ("Ligue 1", "France", 1, "soccer_france_ligue_one"),
        "E1": ("Championship", "England", 2, None),
        "SP2": ("Segunda División", "Spain", 2, None),
        "I2": ("Serie B", "Italy", 2, None),
        "D2": ("2. Bundesliga", "Germany", 2, None),
        "F2": ("Ligue 2", "France", 2, None),
    }
    assert sd.TOP == ["E0", "SP1", "I1", "D1", "F1"]
    assert sd.COUNTRY_TOP == {"England": "E0", "Spain": "SP1", "Italy": "I1", "Germany": "D1", "France": "F1"}
    assert sd.league_codes() == list(sd.LEAGUES) and len(sd.league_codes("all")) == 38


def test_every_league_football_data_carries_is_registered_once():
    assert len(sd.ALL_LEAGUES) == 38 and len(sd.MAIN) == 22 and len(sd.EXTRA) == 16
    for code in sd.LEAGUES:
        assert sd.ALL_LEAGUES[code][:4] == sd.LEAGUES[code]        # LEAGUES' own fields, in place
    report = json.loads((PROBE / "probe_report.json").read_text())["requests"]
    probed_main = {k[3:].rsplit("_", 1)[0] for k in report if k.startswith("fd_") and k.endswith(("_2526.csv", "_2627.csv"))}
    probed_extra = {k[7:-4] for k in report if k.startswith("fd_new_") and k != "fd_new_league_fixtures.csv"}
    assert probed_main == set(NEW_MAIN) and set(sd.MAIN) == set(sd.LEAGUES) | probed_main
    assert probed_extra == set(sd.EXTRA)
    for code, meta in sd.ALL_LEAGUES.items():
        assert meta.kind in ("main", "extra") and meta.calendar in ("aug-may", "calendar-year")
        assert meta.fd_code == code and (meta.kind == "extra") == bool(meta.fd_leagues)
        assert 1 <= meta.tier <= 5 and (meta.odds_key is None) == (code not in sd.TOP)
    assert [sd.ALL_LEAGUES[c].tier for c in ["E0", "E1", "E2", "E3", "EC"]] == [1, 2, 3, 4, 5]
    assert [sd.ALL_LEAGUES[c].tier for c in ["SC0", "SC1", "SC2", "SC3"]] == [1, 2, 3, 4]


def test_the_espn_codes_are_the_ones_espn_answered_in_the_probe():
    report = json.loads((PROBE / "probe_report.json").read_text())["requests"]
    for code, meta in sd.ALL_LEAGUES.items():
        if meta.espn is None:
            assert code == "POL" and report["espn_pol.1_teams.json"]["status"] == 404
        elif code not in sd.LEAGUES or code == "E1" or meta.tier == 2:
            assert report[f"espn_{meta.espn}_teams.json"]["status"] == 200, code
    events = {c: report[f"espn_{m.espn}_boards.json"]["events"] for c, m in sd.ALL_LEAGUES.items()
              if m.espn and f"espn_{m.espn}_boards.json" in report}
    assert sorted(c for c, n in events.items() if n == 0) == ["FIN", "IRL", "ROU", "SC2", "SC3", "SWZ"]


def test_every_league_value_in_the_files_is_placed_or_left_out_on_purpose():
    seen = collections.Counter()
    for code in sd.EXTRA:
        for row in sd._rows(sd._text(raw(f"fd_new_{code}.csv"))):
            seen[(row["Country"], " ".join(row["League"].split()))] += 1
    unplaced = {k: n for k, n in seen.items() if k not in sd.EXTRA_BY_NAME}
    assert unplaced == {("Switzerland", "Challenge League"): 2}


# ------------------------------------------------------------- the extra leagues' files

EXTRA_ROWS = {"ARG": 6444, "AUT": 2698, "BRA": 5639, "CHN": 3040, "DNK": 3024, "FIN": 2714, "IRL": 2738,
              "JPN": 4643, "MEX": 4779, "NOR": 3574, "POL": 4188, "ROU": 4286, "RUS": 3416, "SWE": 3584,
              "SWZ": 2739, "USA": 6282}


@pytest.mark.parametrize("code", sorted(EXTRA_ROWS))
def test_an_extra_leagues_file_parses_every_season_into_the_snapshots_columns(code):
    df, skipped = extra(code)
    assert list(df.columns) == sd.COLUMNS and len(df) == EXTRA_ROWS[code]
    assert skipped == ({"Switzerland Challenge League": 2} if code == "SWZ" else {})
    first = 2014 if code == "CHN" else 2012
    assert sorted(set(df["season"])) == list(range(first, 2027))
    assert (df["league"] == code).all() and df["match_id"].is_unique
    for c in ("home", "away"):
        assert (df[c] == df[c].str.strip()).all() and (df[c] != "").all()
    clubs = df.groupby("season").apply(lambda g: len(set(g["home"]) | set(g["away"])), include_groups=False)
    if code == "ARG":                       # its 2016 holds two tournaments' clubs
        assert clubs.pop(2016) == 31
    assert clubs.between(10, 30).all(), clubs.to_dict()
    # Every row has a kickoff; a result wherever the file has the goals.
    assert df["kickoff"].notna().all() and df["time"].notna().all()
    assert df["result"].isna().sum() == (1 if code == "BRA" else 0)
    # Closing prices only: the average (and Pinnacle's until it went
    # unreliable in July 2025); nothing pre-match, no statistics, no totals.
    by = df.groupby("season")
    avgc = by["avgc_h"].apply(lambda s: s.notna().mean())
    if code == "RUS":
        assert round(avgc[2025], 2) == 0.67
        avgc = avgc.drop(2025)
    assert (avgc >= 0.985).all(), avgc.to_dict()
    psc = by["psc_h"].apply(lambda s: s.notna().mean())
    assert (psc[psc.index <= 2024] >= 0.9).all() and psc[2026] == 0
    for c in ("avg_h", "max_h", "ps_h", "b365_h", "avg_o25", "avgc_o25", "hs", "hst", "hthg"):
        assert df[c].isna().all(), c
    assert ((df["avgc_h"] > 1) | df["avgc_h"].isna()).all()


def test_seasons_take_football_datas_label_by_the_year_they_start():
    assert [sd.season_label(s) for s in ["2012/2013", "2012", " 2026/2027 ", "2012/2014", "12/13", ""]] == \
        [2012, 2012, 2026, None, None, None]
    arg, _ = extra("ARG")
    # "2016" (Feb-May 2016) and "2016/2017" both start in 2016: one season.
    s16 = arg[arg["season"] == 2016]
    assert len(s16) == 692 and (s16["date"].min(), s16["date"].max()) == ("2016-02-05", "2017-06-28")
    jpn, _ = extra("JPN")
    assert jpn.loc[jpn["season"] == 2025, "date"].max() == "2025-12-06"
    assert jpn.loc[jpn["season"] == 2026, "date"].min() == "2026-08-07"      # "2026/2027"
    assert sd.ALL_LEAGUES["JPN"].calendar == "aug-may" and sd.season_for("JPN", "2027-03-01") == 2026
    assert sd.season_for("BRA", "2026-12-01") == 2026 and sd.season_for("BRA", "2027-02-01") == 2027
    assert sd.season_for("E0", "2027-02-01") == 2026 == sd.season_for("AUT", "2026-07-25")


def test_one_match_is_played_after_the_next_season_started():
    # A walk-forward must cut on dates, not on season numbers (module notes).
    late = []
    for code in sd.EXTRA:
        df, _ = extra(code)
        starts = df.groupby("season")["date"].min()
        for season, g in df.groupby("season"):
            later = starts[starts.index > season]
            if len(later):
                late += list(g.loc[g["date"] > later.min(), "match_id"])
    assert late == ["ARG-20150129-bocajuniors-velezsarsfield"]


def test_argentinas_league_cup_is_part_of_its_season_and_switzerlands_play_off_is_not():
    rows = list(sd._rows(sd._text(raw("fd_new_ARG.csv"))))
    copa = [r for r in rows if r["League"].strip() == "Copa De La Liga Profesional"]
    assert len(copa) == 920
    arg, _ = extra("ARG")
    ids = set(arg["match_id"])
    assert all(sd.match_id("ARG", sd.parse_date(r["Date"]), r["Home"], r["Away"]) in ids for r in copa)
    slug = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    for year in ("2021", "2022", "2023", "2024"):
        clubs = lambda name: {slug(r[side]) for r in rows if r["Season"] == year
                              and r["League"].strip() == name for side in ("Home", "Away")}
        assert clubs("Copa De La Liga Profesional") <= clubs("Liga Profesional"), year
    swz = [r for r in sd._rows(sd._text(raw("fd_new_SWZ.csv"))) if r["League"] == "Challenge League"]
    assert [(r["Date"], r["Home"], r["Away"]) for r in swz] == [("27/05/2021", "Thun", "Sion"),
                                                                ("30/05/2021", "Sion", "Thun")]
    df, _ = extra("SWZ")
    assert not df["match_id"].isin([sd.match_id("SWZ", sd.parse_date(r["Date"]), r["Home"], r["Away"])
                                    for r in swz]).any()


def test_the_two_clubs_spelled_two_ways_share_their_ids():
    arg, _ = extra("ARG")
    nor, _ = extra("NOR")
    assert {"Colon Santa FE", "Colon Santa Fe"} <= set(arg["home"])
    assert {"Ham-Kam", "HamKam"} <= set(nor["home"])
    assert sd.match_id("NOR", pd.Timestamp("2026-01-01").date(), "Ham-Kam", "x") == \
        sd.match_id("NOR", pd.Timestamp("2026-01-01").date(), "HamKam", "x")


def test_russias_file_stopped_in_august():
    df, _ = extra("RUS")
    assert df["date"].max() == "2026-08-02" and (df["season"] == 2026).sum() == 16


# ------------------------------------------------------------- the new divisions' season files

MAIN_ROWS = {"E2": (552, 143), "E3": (552, 151), "EC": (552, 177), "SC0": (228, 66), "SC1": (180, 59),
             "SC2": (180, 59), "SC3": (180, 55), "N1": (306, 90), "B1": (311, 80), "P1": (306, 81),
             "T1": (306, 81), "G1": (236, 55)}


@pytest.mark.parametrize("div", NEW_MAIN)
def test_a_new_division_parses_through_the_existing_parser(div):
    for code, n in zip(("2526", "2627"), MAIN_ROWS[div]):
        df = main(div, code)
        assert len(df) == n and (df["league"] == div).all() and df["match_id"].is_unique
        assert (df["season"] == int("20" + code[:2])).all()
        assert df["kickoff"].notna().all() and df["result"].notna().all()
        assert df["avg_h"].notna().mean() >= 0.97 and df["avgc_h"].notna().mean() == 1
        assert df["avg_o25"].notna().mean() >= 0.97
        shots = df["hs"].notna().mean()
        assert shots == (0 if (div, code) == ("EC", "2526") else 1), (div, code, shots)
    # Unchanged by default: a parse without leagues keeps to the ten.
    assert sd.parse(raw(f"fd_{div}_2627.csv"), 2026) == []


# ------------------------------------------------------------- the clock

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def _sim(a, names):
    return max(difflib.SequenceMatcher(None, _norm(a), _norm(n)).ratio() for n in names if n)


def _pair(ours, theirs, floor):
    """Each of our matches -> the one of theirs it is: within 36 hours, the
    same score where both have one, the best pair of names, and clearly so."""
    out = []
    for r in ours.itertuples():
        cands = []
        for t in theirs:
            if abs((t["utc"] - r.kickoff).total_seconds()) > 36 * 3600:
                continue
            if t["score"] is not None and not pd.isna(r.hg) and t["score"] != (int(r.hg), int(r.ag)):
                continue
            cands.append((min(_sim(r.home, t["home"]), _sim(r.away, t["away"])), t["utc"]))
        cands.sort(reverse=True)
        if cands and cands[0][0] >= floor and (len(cands) == 1 or cands[1][0] < cands[0][0] - 0.05):
            out.append((r, cands[0][1]))
    return out


def _espn(lg):
    events = json.loads(raw(f"espn_{lg}_boards.json"))
    out = []
    for e in events:
        c = e["competitions"][0]
        side = {t["homeAway"]: t for t in c["competitors"]}
        done = c["status"]["type"].get("completed")
        names = lambda t: [t["team"].get(k) for k in ("displayName", "shortDisplayName", "name", "location")]
        out.append({"utc": pd.Timestamp(e["date"]).tz_localize(None), "home": names(side["home"]),
                    "away": names(side["away"]),
                    "score": (int(side["home"]["score"]), int(side["away"]["score"])) if done else None})
    return out


def _ours(code):
    """What the probe has of a league in the ESPN window: this season's
    results and the fixtures still to play."""
    if code in NEW_MAIN:
        fx = sd.frame(sd.parse(raw("fd_fixtures.csv"), leagues=[code]))
        df = pd.concat([main(code, "2627"), fx], ignore_index=True)
    else:
        df = pd.concat([extra(code)[0], extra_fixtures()[extra_fixtures()["league"] == code]], ignore_index=True)
    return df.drop_duplicates("match_id", keep="first")


def _local_reading(r, zone):
    """The kickoff had the file's time been the league's own clock."""
    day = pd.Timestamp(r.date).date()
    h, m = map(int, r.time.split(":"))
    return pd.Timestamp(datetime(day.year, day.month, day.day, h, m, tzinfo=ZoneInfo(zone))
                        .astimezone(timezone.utc).replace(tzinfo=None))


# (matches paired with ESPN, of which on football-data's UK clock to the minute)
ESPN_PAIRED = {"ARG": 3, "BRA": 12, "DNK": 1, "MEX": 1, "SWE": 8, "USA": 14,
               "E2": 5, "E3": 2, "EC": 12, "SC0": 6, "SC1": 1, "P1": 1, "T1": 3}


def test_football_datas_times_are_the_uk_clock_by_espns_utc_kickoffs():
    paired_counts = {}
    for code, meta in sd.ALL_LEAGUES.items():
        if code in sd.LEAGUES or not meta.espn or not (PROBE / f"espn_{meta.espn}_boards.json.gz").exists():
            continue
        theirs = _espn(meta.espn)
        if not theirs:
            continue
        lo, hi = min(t["utc"] for t in theirs), max(t["utc"] for t in theirs)
        ours = _ours(code)
        ours = ours[(ours["kickoff"] >= lo - timedelta(hours=12)) & (ours["kickoff"] <= hi + timedelta(hours=12))]
        paired = _pair(ours, theirs, floor=0.45)
        if not paired:
            continue
        paired_counts[code] = len(paired)
        assert all(r.kickoff == utc for r, utc in paired), code
        if code not in SAME_CLOCK_AS_UK:
            assert not any(_local_reading(r, ZONES[code]) == utc for r, utc in paired), code
    assert paired_counts == ESPN_PAIRED
    assert sum(ESPN_PAIRED.values()) == 69


# (file, fixturedownload feeds, paired, on the UK clock, in British Summer Time)
FXD = {"USA": (["fxd_mls_2025", "fxd_mls_2026"], 906, 890),
       "N1": (["fxd_eredivisie_2025", "fxd_eredivisie_2026"], 350, 347),
       "P1": (["fxd_primeira-liga_2025", "fxd_primeira-liga_2026"], 304, 303),
       "T1": (["fxd_super-lig_2025", "fxd_super-lig_2026"], 385, 380),
       "SC0": (["fxd_scottish-premiership_2026"], 60, 60)}


@pytest.mark.parametrize("code", sorted(FXD))
def test_football_datas_times_are_the_uk_clock_with_summer_time_by_fixturedownloads(code):
    feeds, n_paired, n_uk = FXD[code]
    theirs = []
    for name in feeds:
        for e in json.loads(raw(f"{name}.json")):
            if e["HomeTeamScore"] is None:
                continue                                    # a placeholder time, maybe
            theirs.append({"utc": pd.Timestamp(e["DateUtc"]).tz_localize(None), "home": [e["HomeTeam"]],
                           "away": [e["AwayTeam"]], "score": (e["HomeTeamScore"], e["AwayTeamScore"])})
    ours = extra(code)[0] if code in sd.EXTRA else pd.concat([main(code, "2526"), main(code, "2627")])
    ours = ours[ours["season"] >= 2025]
    paired = _pair(ours, theirs, floor=0.35)
    uk = [(r, utc) for r, utc in paired if r.kickoff == utc]
    assert (len(paired), len(uk)) == (n_paired, n_uk)
    summer = [(r, utc) for r, utc in paired
              if datetime.fromisoformat(f"{r.date}T{r.time}").replace(tzinfo=sd.UK).dst()]
    assert len(summer) > len(paired) / 2
    assert sum(r.kickoff == utc for r, utc in summer) >= 0.97 * len(summer)
    # Read as UTC, the summer's would all be an hour out; as the local clock
    # (outside the UK's zone), every one.
    assert sum(pd.Timestamp(f"{r.date} {r.time}") == utc for r, utc in summer) <= 0.02 * len(summer)
    if code not in SAME_CLOCK_AS_UK:
        assert not any(_local_reading(r, ZONES[code]) == utc for r, utc in paired)
    off = [abs((utc - r.kickoff).total_seconds()) / 60 for r, utc in paired if r.kickoff != utc]
    if code == "T1":                    # 14/09/2025: two matches football-data has an hour early
        assert sorted(off)[-2:] == [60, 60] and max(sorted(off)[:-2] or [0]) < 10


def _overlap(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return sum(min(a[k] / na, b[k] / nb) for k in set(a) | set(b))


@pytest.mark.parametrize("code", sd.EXTRA)
def test_over_a_whole_history_the_uk_clock_gives_local_kickoff_times(code):
    """No UTC source covers most of a history: what the kickoffs look like
    in the league's own time. On the UK clock nearly all fall in daytime or
    evening there, and the same times in summer and winter; read as UTC
    the local times shift an hour at the UK's clock changes."""
    df, _ = extra(code)
    zone = ZoneInfo(ZONES[code])
    local = df["kickoff"].dt.tz_localize("UTC").dt.tz_convert(zone)
    assert local.dt.hour.between(11, 22).mean() >= 0.985
    if code in ("ARG", "BRA", "CHN", "JPN", "MEX", "USA"):
        assert pd.to_datetime(df["time"], format="%H:%M").dt.hour.between(11, 22).mean() <= 0.8
    naive = pd.to_datetime(df["date"] + " " + df["time"])
    summer = naive.dt.tz_localize(sd.UK, ambiguous="NaT", nonexistent="NaT").map(
        lambda t: bool(t.dst()) if not pd.isna(t) else None)
    as_utc = naive.dt.tz_localize("UTC").dt.tz_convert(zone)
    hist = lambda s, m: collections.Counter(s[m].dt.strftime("%H:%M"))
    uk = _overlap(hist(local, summer == True), hist(local, summer == False))       # noqa: E712
    utc = _overlap(hist(as_utc, summer == True), hist(as_utc, summer == False))    # noqa: E712
    assert uk > utc, (code, round(uk, 2), round(utc, 2))


# ------------------------------------------------------------- fixtures

def test_the_main_fixtures_keep_to_the_ten_unless_asked():
    body = raw("fd_fixtures.csv")
    assert collections.Counter(r["league"] for r in sd.parse(body)) == {"I1": 10, "I2": 10}
    every = sd.frame(sd.parse(body, leagues="all"))
    assert every.groupby("league").size().to_dict() == {"E2": 5, "E3": 1, "EC": 12, "I1": 10, "I2": 10,
                                                        "SC0": 6, "SC1": 1, "SC2": 1}
    assert every["avg_h"].notna().all() and every["hg"].isna().all()
    # Played by the probe: each fixture's id is its result's.
    new = every[~every["league"].isin(sd.LEAGUES)]
    results = set(pd.concat([main(d, "2627") for d in NEW_MAIN])["match_id"])
    assert len(new) == 26 and new["match_id"].isin(results).all()


def test_the_extra_fixtures_carry_pre_match_prices_into_the_pre_match_columns():
    fx = extra_fixtures()
    assert fx.groupby("league").size().to_dict() == {"ARG": 1, "BRA": 10, "FIN": 3, "MEX": 1, "SWE": 8, "USA": 14}
    assert (fx["season"] == 2026).all() and fx["kickoff"].notna().all() and fx["hg"].isna().all()
    for c in ("avg_h", "avg_d", "avg_a", "max_h", "b365_h"):
        assert fx[c].notna().all(), c
    for c in ("avgc_h", "psc_h", "avg_o25", "ps_h"):           # Pinnacle: none since July 2025
        assert fx[c].isna().all(), c
    for code, g in fx.groupby("league"):
        clubs = set(extra(code)[0].query("season == 2026")[["home", "away"]].stack())
        assert set(g["home"]) | set(g["away"]) <= clubs, code
    usa = fx[fx["league"] == "USA"].iloc[0]
    assert (usa["date"], usa["time"], usa["kickoff"]) == ("2026-10-28", "23:30", pd.Timestamp("2026-10-28 23:30"))


# ------------------------------------------------------------- the snapshot

def test_a_season_file_round_trips_and_an_unchanged_one_is_never_rewritten(tmp_path):
    df = main("E2", "2526")
    assert sd.store_season(tmp_path, "E2", 2025, df.to_dict("records")) == "written"
    path = sd.season_path(tmp_path, "E2", 2025)
    first = path.read_bytes()
    assert first[3] & 0x08 == 0 and first[4:8] == b"\0\0\0\0"          # no file name, mtime 0
    back = sd.load_matches(tmp_path, leagues="E2")
    pd.testing.assert_frame_equal(back, sd.order(df), check_dtype=False)
    assert back["kickoff"].dtype == "datetime64[us]" and back["hg"].dtype == "Int64" and back["home"].dtype == "str"
    # What was loaded, stored again, is the same text: no churn.
    assert sd.store_season(tmp_path, "E2", 2025, back) == "unchanged"
    # Another zlib's bytes for the same text are left alone.
    path.write_bytes(gzip.compress(gzip.decompress(first), 1, mtime=0))
    other = path.read_bytes()
    assert other != first and sd.store_season(tmp_path, "E2", 2025, df) == "unchanged"
    assert path.read_bytes() == other


def test_the_loaders_default_to_the_ten_and_open_up_on_request(tmp_path):
    sd.store_season(tmp_path, "E0", 2006, sd.parse(OLD_E0, 2006))
    sd.store_season(tmp_path, "BRA", 2026, extra("BRA")[0][lambda d: d["season"] == 2026])
    assert set(sd.load_matches(tmp_path)["league"]) == {"E0"}
    every = sd.load_matches(tmp_path, leagues="all")
    assert every.groupby("league").size().to_dict() == {"BRA": 319, "E0": 2}
    assert list(every.columns) == sd.COLUMNS and every["date"].is_monotonic_increasing
    assert len(sd.load_matches(tmp_path, leagues=["BRA"])) == 319 == len(sd.load_matches(tmp_path, leagues="BRA"))
    empty = sd.load_matches(tmp_path / "nothing")
    assert list(empty.columns) == sd.COLUMNS and empty.empty


OLD_E0 = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH\n"
          "E0,19/08/06,Arsenal,Aston Villa,1,1,D,1.24\nE0,19/08/06,Bolton,Tottenham,2,0,H,2.55\n").encode()


# ------------------------------------------------------------- sync

def probe_fetch(url):
    report = json.loads((PROBE / "probe_report.json").read_text())["requests"]
    for name, r in report.items():
        if r.get("url") == url and r.get("status") == 200 and name.startswith("fd_"):
            return raw(name)
    return None


def test_a_sync_from_the_probe_fills_every_league_and_both_fixtures(tmp_path):
    matches, fixtures, asked, lines = tmp_path / "matches", tmp_path / "fixtures.csv", [], []

    def fetch(url):
        asked.append(url)
        return probe_fetch(url)
    sd.sync(today="2026-10-29", fetch=fetch, log=lines.append, matches_path=matches, fixtures_path=fixtures)
    # One request per main division for this season, one per extra league,
    # and both fixtures files.
    assert len(asked) == 22 + 16 + 2 and len(set(asked)) == len(asked)
    assert sd.SEASON_URL.format(code="2627", league="SC3") in asked
    assert sd.EXTRA_URL.format(code="SWZ") in asked and sd.EXTRA_FIXTURES_URL in asked
    every = sd.load_matches(matches, leagues="all")
    assert set(every["league"]) == set(NEW_MAIN) | set(sd.EXTRA)        # nothing of the ten in the probe
    assert (every.loc[every["league"].isin(NEW_MAIN), "season"] == 2026).all()
    assert sorted(set(every.loc[every["league"] == "CHN", "season"])) == list(range(2014, 2027))
    assert sd.load_matches(matches).empty                                # the ten: none on file here
    assert any("SWZ: 2 rows of Switzerland Challenge League left out" in line for line in lines)
    assert not any("KEPT ON FILE" in line for line in lines)
    fx = sd.load_fixtures(fixtures, leagues="all")
    assert fx.groupby("league").size().to_dict() == {"ARG": 1, "BRA": 10, "E2": 5, "E3": 1, "EC": 12, "FIN": 3,
                                                     "I1": 10, "I2": 10, "MEX": 1, "SC0": 6, "SC1": 1,
                                                     "SC2": 1, "SWE": 8, "USA": 14}
    assert sd.load_fixtures(fixtures)["league"].value_counts().to_dict() == {"I1": 10, "I2": 10}
    assert fx["fetched_at"].notna().all()


def test_a_second_sync_rewrites_only_the_seasons_that_changed(tmp_path):
    matches = tmp_path / "matches"
    files = {sd.EXTRA_URL.format(code="AUT"): raw("fd_new_AUT.csv")}
    sd.sync(fetch=files.get, log=lambda *_: None, matches_path=matches, fixtures_path=tmp_path / "f.csv",
            leagues=["AUT"])
    assert sd.seasons_on_file(matches, "AUT") == list(range(2012, 2027))
    stamps = {p.name: p.stat().st_mtime_ns for p in (matches / "AUT").iterdir()}
    # A new result in 2026-27 and a correction to 2013-14: the running
    # season is rewritten, the finished one is not (that takes full=True).
    text = sd._text(raw("fd_new_AUT.csv"))
    lines = text.splitlines(keepends=True)
    changed = [l.replace(",SK Rapid,", ",Rapid Wien,", 1) if ",2013/2014," in l else l for l in lines]
    changed.append(lines[-1].replace("25/10/2026", "01/11/2026"))
    files[sd.EXTRA_URL.format(code="AUT")] = "".join(changed).encode()
    sd.sync(fetch=files.get, log=lambda *_: None, matches_path=matches, fixtures_path=tmp_path / "f.csv",
            leagues=["AUT"])
    now = {p.name: p.stat().st_mtime_ns for p in (matches / "AUT").iterdir()}
    assert [n for n in sorted(now) if now[n] != stamps[n]] == ["2026.csv.gz"]
    assert "Rapid Wien" not in set(sd.load_matches(matches, leagues="AUT")["home"])
    sd.sync(full=True, fetch=files.get, log=lambda *_: None, matches_path=matches,
            fixtures_path=tmp_path / "f.csv", leagues=["AUT"])
    assert "Rapid Wien" in set(sd.load_matches(matches, leagues="AUT").query("season == 2013")["home"])


def test_a_full_sync_survives_404s_empty_files_and_failures_and_keeps_the_snapshot(tmp_path):
    matches, fixtures, lines = tmp_path / "matches", tmp_path / "f.csv", []
    good = {sd.SEASON_URL.format(code="2526", league="G1"): raw("fd_G1_2526.csv"),
            sd.SEASON_URL.format(code="2627", league="G1"): raw("fd_G1_2627.csv"),
            sd.EXTRA_URL.format(code="RUS"): raw("fd_new_RUS.csv")}
    sd.sync(full=True, today="2026-10-29", fetch=good.get, log=lines.append, matches_path=matches,
            fixtures_path=fixtures, leagues=["G1", "RUS"])
    before = sd.load_matches(matches, leagues="all")
    assert sd.seasons_on_file(matches, "G1") == [2025, 2026] and len(before) == 291 + 3416
    assert sum("not published" in line for line in lines) == 2024 - 2005 + 1 + 2   # G1 2005-2024, fixtures x2

    truncated = "\n".join(sd._text(raw("fd_new_RUS.csv")).splitlines()[:-3]) + "\n"

    def bad(url):
        if url.endswith("/2627/G1.csv"):
            raise ConnectionError("503")
        if url.endswith("/2526/G1.csv"):
            return b"<html><body>Service unavailable</body></html>"
        if url.endswith("/0506/G1.csv"):
            return b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"          # a header and no rows
        if url.endswith("/0607/G1.csv"):
            return b""                                                   # an empty body
        if url.endswith("/new/RUS.csv"):
            return truncated.encode()                                    # three matches short
        return None
    lines.clear()
    sd.sync(full=True, today="2026-10-29", fetch=bad, log=lines.append, matches_path=matches,
            fixtures_path=fixtures, leagues=["G1", "RUS"])
    pd.testing.assert_frame_equal(sd.load_matches(matches, leagues="all"), before)
    kept = [line for line in lines if "KEPT ON FILE" in line]
    assert len(kept) == 1 and all(k in kept[0] for k in ("G1 2026", "G1 2025", "G1 2005", "RUS 2026"))
    assert "G1 2006" not in kept[0]                                      # empty is "not published"


def test_one_fixtures_file_failing_keeps_its_leagues_rows_and_the_others_refresh(tmp_path):
    fixtures, lines = tmp_path / "f.csv", []
    files = {sd.FIXTURES_URL: raw("fd_fixtures.csv"), sd.EXTRA_FIXTURES_URL: raw("fd_new_league_fixtures.csv")}
    sd.sync_fixtures(fetch=files.get, log=lambda *_: None, path=fixtures)
    first = sd.load_fixtures(fixtures, leagues="all")
    assert len(first) == 83

    def flaky(url):
        if url == sd.EXTRA_FIXTURES_URL:
            raise ConnectionError("503")
        return sd._text(raw("fd_fixtures.csv")).replace("\nEC,", "\nXX,").encode()   # EC's round is gone
    problems = []
    sd.sync_fixtures(fetch=flaky, log=lines.append, path=fixtures, problems=problems)
    after = sd.load_fixtures(fixtures, leagues="all")
    assert problems == ["extra-league fixtures"]
    assert "EC" not in set(after["league"]) and len(after) == 83 - 12
    same = first[first["league"] != "EC"].reset_index(drop=True)
    pd.testing.assert_frame_equal(after, same)                           # same rows, same first-seen times
    # The ten alone (the pipeline's own call) leave the other leagues' rows be.
    sd.sync_fixtures(leagues=None, fetch=files.get, log=lambda *_: None, path=fixtures)
    assert set(sd.load_fixtures(fixtures, leagues="all")["league"]) == {"I1", "I2"} | set(
        after.loc[after["league"].isin(sd.EXTRA), "league"])


def test_the_hour_the_uk_clock_repeats_reads_as_its_first_pass():
    """25 Oct 2026: 01:00-01:59 on the UK clock happened twice (summer time,
    then GMT), and football-data writes both as 01:xx. It is read as the
    first pass - the earlier moment, never after the real kickoff. The
    skipped hour of a March night reads as GMT."""
    day = pd.Timestamp("2026-10-25").date()
    assert sd.kickoff_utc(day, "01:30") == pd.Timestamp("2026-10-25 00:30")
    assert sd.kickoff_utc(pd.Timestamp("2023-03-26").date(), "01:30") == pd.Timestamp("2023-03-26 01:30")
    in_hour = collections.Counter()
    for code in sd.EXTRA:
        df, _ = extra(code)
        for r in df.itertuples():
            h, m = map(int, r.time.split(":"))
            d = pd.Timestamp(r.date)
            first = datetime(d.year, d.month, d.day, h, m, tzinfo=sd.UK)
            if first.utcoffset() != first.replace(fold=1).utcoffset():
                in_hour[(code, d.month)] += 1
    assert in_hour == {("ARG", 10): 4, ("BRA", 10): 9, ("MEX", 10): 27, ("USA", 10): 12, ("USA", 3): 1}
    # That night's five MLS matches at "01:30": fixturedownload has three at
    # 00:30 UTC (the first pass) and two at 01:30 UTC (the second).
    usa = extra("USA")[0]
    night = usa[(usa["date"] == "2026-10-25") & (usa["time"] == "01:30")]
    theirs = collections.Counter(e["DateUtc"] for e in json.loads(raw("fxd_mls_2026.json"))
                                 if e["DateUtc"] in ("2026-10-25 00:30:00Z", "2026-10-25 01:30:00Z"))
    assert len(night) == 5 and (night["kickoff"] == pd.Timestamp("2026-10-25 00:30")).all()
    assert theirs == {"2026-10-25 00:30:00Z": 3, "2026-10-25 01:30:00Z": 2}
