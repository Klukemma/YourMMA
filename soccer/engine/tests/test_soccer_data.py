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
    matches, fixtures = tmp_path / "m.csv.gz", tmp_path / "f.csv"
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
