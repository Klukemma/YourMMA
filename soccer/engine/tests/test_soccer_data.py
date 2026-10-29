"""The soccer data layer: parsing football-data.co.uk's files as they are
actually written, in their old and new shapes."""

import gzip

import pandas as pd
import pytest

import soccer_data as sd

# 2006-07 shape: two-digit years, no kickoff time, "BbAv" averages, a row
# longer than its header, a trailing blank line.
OLD = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,B365H,B365D,B365A,"
    "BbAvH,BbAvD,BbAvA,BbMxH,BbMxD,BbMxA,BbAv>2.5,BbAv<2.5\n"
    "E0,19/08/06,Arsenal,Aston Villa,1,1,D,0,0,D,24,5,12,3,1.25,5,11,1.24,4.9,10.5,1.28,5.2,12,1.7,2.1\n"
    "E0,19/08/06,Bolton,Tottenham,2,0,H,2,0,H,10,12,6,7,2.6,3.2,2.6,2.55,3.15,2.6,2.7,3.3,2.75,2.0,1.8,,,,\n"
    "\n"
)
# 2025-26 shape: four-digit years, a UK kickoff time, "Avg" averages and
# closing prices, a BOM on the header.
NEW = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HC,AC,HY,AY,HR,AR,"
    "B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,"
    "PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
    "SP1,26/10/2025,15:15,Real Madrid,Barcelona,2,1,H,2,1,H,14,12,6,4,5,7,2,3,0,1,"
    "2.3,3.9,2.8,2.35,3.95,2.9,2.4,4.1,3.0,2.3,3.85,2.8,1.45,2.7,2.25,4.0,3.0,2.2,3.9,2.95,1.42,2.8\n"
)


def test_an_old_season_parses_with_its_own_column_names():
    rows = sd.parse(OLD.encode(), 2006)
    assert len(rows) == 2                      # the long row kept, the blank dropped
    a = rows[0]
    assert (a["home"], a["away"], a["hg"], a["ag"], a["result"]) == ("Arsenal", "Aston Villa", 1, 1, "D")
    assert a["date"] == "2006-08-19" and a["season"] == 2006
    assert a["avg_h"] == 1.24 and a["max_a"] == 12 and a["avg_o25"] == 1.7
    assert pd.isna(a["kickoff"])               # no clock time before 2019-20
    assert a["avgc_h"] is None and a["psc_h"] is None


def test_a_new_season_reads_the_uk_clock_as_utc():
    (r,) = sd.parse(NEW.encode("utf-8"), 2025)
    assert r["league"] == "SP1" and r["result"] == "H"
    # 26 Oct 2025 is the day British Summer Time ends: 15:15 UK is 15:15 UTC.
    assert r["kickoff"] == pd.Timestamp("2025-10-26 15:15")
    assert (r["avg_h"], r["ps_h"], r["avgc_h"], r["psc_h"]) == (2.3, 2.35, 2.2, 2.25)
    assert r["avgc_o25"] == 1.42


def test_a_summer_kickoff_is_an_hour_behind_the_uk_clock():
    assert sd.kickoff_utc(pd.Timestamp("2026-08-22").date(), "15:00") == pd.Timestamp("2026-08-22 14:00")


def test_latin_1_files_are_read():
    body = OLD.replace("Aston Villa", "Deportivo La Coruña").encode("latin-1")
    rows = sd.parse(body, 2006)
    assert rows[0]["away"] == "Deportivo La Coruña"


def test_seasons_are_named_by_the_year_they_start():
    assert sd.season_code(2026) == "2627" and sd.season_code(2005) == "0506"
    assert sd.season_of("2026-08-15") == 2026 and sd.season_of("2027-05-20") == 2026
    assert sd.current_season(pd.Timestamp("2026-10-25")) == 2026


def test_a_sync_replaces_only_the_seasons_it_fetched(tmp_path):
    matches, fixtures = tmp_path / "matches", tmp_path / "f.csv"
    files = {sd.SEASON_URL.format(code="0607", league="E0"): OLD.encode()}
    sd.sync(seasons=[2006], fetch=files.get, log=lambda *_: None,
            matches_path=matches, fixtures_path=fixtures)
    first = sd.load_matches(matches)
    assert len(first) == 2

    # A later sync of another season keeps 2006-07 untouched, and a league
    # the site has not published is skipped, not blanked.
    newer = {sd.SEASON_URL.format(code="2526", league="SP1"): NEW.encode(),
             sd.FIXTURES_URL: NEW.replace(",2,1,H,2,1,H,14,12,6,4,5,7,2,3,0,1,", ",,,,,,,,,,,,,,,,,").encode()}
    sd.sync(seasons=[2025], fetch=newer.get, log=lambda *_: None,
            matches_path=matches, fixtures_path=fixtures)
    both = sd.load_matches(matches)
    assert len(both) == 3 and set(both["league"]) == {"E0", "SP1"}
    assert both["hg"].dtype == "Int64"
    fx = sd.load_fixtures(fixtures)
    assert len(fx) == 1 and fx["hg"].isna().all() and fx["fetched_at"].notna().all()


def test_an_unchanged_snapshot_is_byte_identical(tmp_path):
    path = tmp_path / "m.csv.gz"
    df = sd.frame(sd.parse(OLD.encode(), 2006))
    sd._write(df, path)
    first = path.read_bytes()
    sd._write(df, path)
    assert path.read_bytes() == first and gzip.decompress(first).startswith(b"match_id,")


def test_every_top_league_has_an_odds_feed_and_a_second_tier():
    for code in sd.TOP:
        name, country, tier, key = sd.LEAGUES[code]
        assert tier == 1 and key and key.startswith("soccer_")
        assert any(m[1] == country and m[2] == 2 for m in sd.LEAGUES.values())


@pytest.mark.snapshot
def test_every_club_in_a_top_league_since_2016_has_a_name_code_and_colours():
    # A snapshot check, not a gate: a newly promoted club shows under its
    # football-data.co.uk name with a neutral chip until it is added here.
    import soccer_teams as st
    m = sd.load_matches()
    recent = m[m["league"].isin(sd.TOP) & (m["season"] >= 2016)]
    missing = sorted(set(recent["home"]) - set(st.TEAMS))
    assert not missing, f"no display entry for {missing}"
    for (league, season), g in recent.groupby(["league", "season"]):
        codes = [st.TEAMS[t][1] for t in set(g["home"])]
        assert len(codes) == len(set(codes)), f"two clubs share a code in {league} {season}"


def test_club_chips_are_three_letters_and_a_stranger_gets_a_neutral_one():
    import soccer_teams as st
    assert all(len(v[1]) == 3 and v[2].startswith("#") and v[3].startswith("#") for v in st.TEAMS.values())
    stranger = st.display("Nobody FC")
    assert stranger["code"] == "NOB" and stranger["name"] == "Nobody FC" and (stranger["bg"], stranger["fg"]) == st.NEUTRAL


def test_every_club_badge_is_readable():
    import soccer_teams as st
    for key in st.TEAMS:
        d = st.display(key)
        assert st.contrast(d["bg"], d["fg"]) >= st.MIN_CONTRAST, (key, d)
    # A club whose own colours read keeps them.
    assert st.display("Arsenal")["fg"] == "#FFFFFF" and st.display("Dortmund")["fg"] == "#000000"
    assert st.display("Crystal Palace")["fg"] == "#FFFFFF"     # was red on blue


def test_a_failed_or_wrong_download_keeps_what_is_on_file(tmp_path):
    matches, fixtures = tmp_path / "matches", tmp_path / "f.csv"
    good = {sd.SEASON_URL.format(code="2526", league="SP1"): NEW.encode(), sd.FIXTURES_URL: NEW.encode()}
    sd.sync(seasons=[2025], fetch=good.get, log=lambda *_: None, matches_path=matches, fixtures_path=fixtures)
    assert len(sd.load_matches(matches)) == 1 and len(sd.load_fixtures(fixtures)) == 1

    def flaky(url):
        if "SP1" in url:
            raise ConnectionError("503")
        if url == sd.FIXTURES_URL:
            return b"<html><body>Service unavailable</body></html>"
        return None
    lines = []
    sd.sync(seasons=[2025], fetch=flaky, log=lines.append, matches_path=matches, fixtures_path=fixtures)
    assert len(sd.load_matches(matches)) == 1 and len(sd.load_fixtures(fixtures)) == 1
    assert any("KEPT ON FILE" in line for line in lines)


def test_a_season_cut_off_part_way_through_a_row_keeps_what_is_on_file(tmp_path):
    matches, fixtures, url = tmp_path / "matches", tmp_path / "f.csv", sd.SEASON_URL.format(code="2526", league="SP1")
    sd.sync(seasons=[2025], fetch={url: NEW.encode()}.get, log=lambda *_: None,
            matches_path=matches, fixtures_path=fixtures)
    before = sd.load_matches(matches)
    cut = NEW.rstrip("\n").rsplit(",", 8)[0]          # the closing prices gone, the match count the same
    lines = []
    sd.sync(seasons=[2025], fetch={url: cut.encode()}.get, log=lines.append,
            matches_path=matches, fixtures_path=fixtures)
    pd.testing.assert_frame_equal(sd.load_matches(matches), before)
    assert before["avgc_h"].iloc[0] == 2.2
    assert any("SP1 2025: CUT OFF" in line for line in lines)
    assert any("KEPT ON FILE" in line and "SP1 2025" in line for line in lines)


def test_a_price_is_dated_when_it_was_first_seen(tmp_path):
    fixtures = tmp_path / "f.csv"
    fx = sd.frame(sd.parse(NEW.replace(",2,1,H,2,1,H,14,12,6,4,5,7,2,3,0,1,", ",,,,,,,,,,,,,,,,,").encode()))
    fx["fetched_at"] = pd.Timestamp("2026-10-30 16:00")
    fx.to_csv(fixtures, index=False)
    again = sd.frame(sd.parse(NEW.replace(",2,1,H,2,1,H,14,12,6,4,5,7,2,3,0,1,", ",,,,,,,,,,,,,,,,,").encode()))
    assert sd.first_seen(again, sd.load_fixtures(fixtures), pd.Timestamp("2026-11-01 07:17")).iloc[0] == pd.Timestamp("2026-10-30 16:00")
    moved = again.assign(avg_h=2.5)
    assert sd.first_seen(moved, sd.load_fixtures(fixtures), pd.Timestamp("2026-11-01 07:17")).iloc[0] == pd.Timestamp("2026-11-01 07:17")


def test_the_result_is_read_from_the_goals():
    rows = sd.parse(OLD.replace("1,1,D,0,0,D", "1,0,,0,0,D").encode(), 2006)
    assert rows[0]["result"] == "H"
    rows = sd.parse(OLD.replace("1,1,D,0,0,D", ",1,H,0,0,D").encode(), 2006)
    assert rows[0]["result"] is None


# ------------------------------------------------------------- the season files

def test_the_season_files_load_exactly_what_the_single_file_did():
    """The migration of 2026-10-29: run while soccer/data/matches.csv.gz was
    still there (it passed, and the file was then removed - git history has
    it); skipped since. test_migrate_* and the fingerprint below carry on."""
    if not sd.LEGACY_MATCHES.exists():
        pytest.skip("the single matches.csv.gz is gone: migrated on 2026-10-29")
    old = sd.load_matches(sd.LEGACY_MATCHES)
    pd.testing.assert_frame_equal(old, sd.load_matches())
    pd.testing.assert_frame_equal(old, sd.load_matches(leagues=list(sd.LEAGUES)))


@pytest.mark.snapshot
def test_the_finished_seasons_are_the_single_files_to_the_byte():
    # A snapshot check, not a gate: a full sync may take a correction from
    # football-data.co.uk. Seasons 2005-2025 of the ten as the single file
    # held them on 2026-10-29 (82,482 matches).
    import hashlib
    m = sd.load_matches()
    done = m[m["season"] <= 2025].reset_index(drop=True)
    assert len(done) == 82482
    assert hashlib.sha256(done.to_csv(index=False).encode()).hexdigest() == \
        "311bff21af9c0a2e3f11d4a63a4c5d83e078046d801c0b2f31bc1343358ab9a4"


def test_migrate_splits_a_single_file_into_season_files_that_load_the_same(tmp_path):
    rows = sd.parse(OLD.encode(), 2006) + sd.parse(NEW.encode(), 2025)
    legacy = tmp_path / "matches.csv.gz"
    sd._write(sd.order(sd.frame(rows)), legacy)
    old = sd.load_matches(legacy)
    sd.migrate(legacy, tmp_path / "matches", log=lambda *_: None)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "matches").rglob("*.gz")) == \
        ["matches/E0/2006.csv.gz", "matches/SP1/2025.csv.gz"]
    pd.testing.assert_frame_equal(old, sd.load_matches(tmp_path / "matches"))


def test_the_repo_snapshot_is_one_file_per_league_and_season_and_defaults_to_the_ten():
    folders = sorted(p.name for p in sd.MATCHES.iterdir() if p.is_dir())
    assert set(sd.LEAGUES) <= set(folders) <= set(sd.ALL_LEAGUES)
    for code in folders:
        names = sorted(p.name for p in (sd.MATCHES / code).iterdir())
        assert names and all(n.endswith(".csv.gz") and n.split(".")[0].isdigit() for n in names), code
    assert not sd.LEGACY_MATCHES.exists() or sd.LEGACY_MATCHES.is_file()
    m = sd.load_matches()
    assert set(m["league"]) == set(sd.LEAGUES) and m["match_id"].is_unique
