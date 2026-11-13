"""The league table of the regular season, for every league in the app:
play-off clubs from outside the league, a post-season's knock-outs and a
split's championship and relegation rounds are left out, read from the
results alone - and the five top leagues' tables are the results summed,
as they always were."""

import numpy as np
import pandas as pd
import pytest

import soccer_form as sf

START = pd.Timestamp("2025-08-02")


def rr_rounds(clubs):
    """A single round robin by the circle method: n-1 rounds of n/2 pairs."""
    clubs = list(clubs)
    n = len(clubs)
    out = []
    for r in range(n - 1):
        pairs = [(clubs[i], clubs[n - 1 - i]) for i in range(n // 2)]
        out.append([(a, b) if r % 2 else (b, a) for a, b in pairs])
        clubs = [clubs[0]] + [clubs[-1]] + clubs[1:-1]
    return out


def _rows(league, season, rounds, strength, start=START, step=7, rng=None):
    """One row per pair, a round a week from `start`; the stronger club wins
    2-0 (or, with rng, a random score)."""
    rows = []
    for k, pairs in enumerate(rounds):
        day = (start + pd.Timedelta(days=step * k)).strftime("%Y-%m-%d")
        for h, a in pairs:
            if rng is None:
                hg, ag = (2, 0) if strength[h] > strength[a] else (0, 2)
            else:
                hg, ag = int(rng.poisson(1.4)), int(rng.poisson(1.1))
            rows.append({"match_id": f"{league}{season}{day}{h}{a}", "league": league, "season": season,
                         "date": day, "kickoff": pd.NaT, "home": h, "away": a, "hg": hg, "ag": ag,
                         "result": "H" if hg > ag else "A" if hg < ag else "D"})
    return rows


CLUBS = [f"Club {k:02d}" for k in range(12)]
STRENGTH = {c: -k for k, c in enumerate(CLUBS)}          # Club 00 strongest: the table is CLUBS in order


def _double(clubs=CLUBS):
    first = rr_rounds(clubs)
    return first + [[(a, h) for h, a in pairs] for pairs in first]


def _split_season(league="SC0"):
    """12 clubs, a double round robin (22 rounds), then top six and bottom six
    play a round robin each (5 rounds)."""
    top, bottom = rr_rounds(CLUBS[:6]), rr_rounds(CLUBS[6:])
    rounds = _double() + [t + b for t, b in zip(top, bottom)]
    return pd.DataFrame(_rows(league, 2025, rounds, STRENGTH))


def _summed(m):
    """Every result summed: the table as it was before."""
    pts = {}
    for h, a, hg, ag in zip(m["home"], m["away"], m["hg"], m["ag"]):
        for t, f, g in ((h, hg, ag), (a, ag, hg)):
            p = pts.setdefault(t, [0, 0, 0])
            p[0] += 1
            p[1] += 3 if f > g else 1 if f == g else 0
            p[2] += f - g
    return pts


def test_a_split_after_the_second_round_robin_is_left_out_of_the_table():
    m = _split_season()
    t = sf.table(m, "SC0", 2025)
    assert [r["team"] for r in t] == CLUBS
    assert all(r["played"] == 22 for r in t)                  # the 5 split rounds are not summed
    assert t[0]["points"] == 66 and t[-1]["points"] == 0
    assert (~sf.regular_season(m)).sum() == 30
    # point in time: two rounds into the split the groups show, and it is cut from then on
    days = sorted(m["date"].unique())
    for k, n_split in ((22, 0), (23, 1), (24, 2), (26, 4)):
        before = sf.table(m, "SC0", 2025, before=days[k] if k < len(days) else None)
        played = {r["team"]: r["played"] for r in before}
        if n_split < 2:
            assert set(played.values()) == {22 + n_split}           # one split round in: not told yet
        else:
            assert set(played.values()) == {22}


def test_a_regular_match_postponed_into_the_split_stays_in_the_table():
    m = _split_season()
    days = sorted(m["date"].unique())
    # one of round 21's matches, between a top-six and a bottom-six club, played in the split's third week
    across = m["home"].isin(CLUBS[:6]) != m["away"].isin(CLUBS[:6])
    late = m.index[(m["date"] == days[20]) & across][0]
    m.loc[late, "date"] = (pd.Timestamp(days[24]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    t = sf.table(m, "SC0", 2025)
    assert all(r["played"] == 22 for r in t) and [r["team"] for r in t] == CLUBS
    keep = sf.regular_season(m)
    assert keep[m.index.get_loc(late)] and (~keep).sum() == 30


def test_the_second_halfs_first_rounds_never_read_as_a_split():
    # after the first round robin, two rounds in which the top six and the bottom
    # six happen to meet among themselves: blocks of the table, by chance - not a split
    first = rr_rounds(CLUBS)
    chance = [t + b for t, b in zip(rr_rounds(CLUBS[:6])[:2], rr_rounds(CLUBS[6:])[:2])]
    chance = [[(a, h) for h, a in pairs] for pairs in chance]
    m = pd.DataFrame(_rows("SC0", 2025, first + chance, STRENGTH))
    assert sf.regular_season(m).all()
    assert all(r["played"] == 13 for r in sf.table(m, "SC0", 2025))


def test_a_split_whose_groups_are_not_blocks_of_the_table_is_summed():
    top, bottom = rr_rounds(CLUBS[:5] + [CLUBS[6]]), rr_rounds([CLUBS[5]] + CLUBS[7:])
    m = pd.DataFrame(_rows("SC0", 2025, _double() + [t + b for t, b in zip(top, bottom)], STRENGTH))
    assert sf.regular_season(m).all()                         # 7th in the top group: not told apart


def test_a_play_off_against_a_club_from_outside_the_league_is_not_in_the_table():
    m = pd.DataFrame(_rows("SC1", 2025, _double(CLUBS[:10]) + _double(CLUBS[:10]), STRENGTH))
    po = pd.DataFrame(_rows("SC1", 2025, [[("Outsider FC", CLUBS[9])], [(CLUBS[9], "Outsider FC")]], STRENGTH | {"Outsider FC": 0},
                            start=pd.Timestamp("2026-05-20")))
    m = pd.concat([m, po], ignore_index=True)
    t = sf.table(m, "SC1", 2025)
    assert "Outsider FC" not in [r["team"] for r in t]
    assert all(r["played"] == 36 for r in t)


def _knock_outs(league):
    """16 clubs, a double round robin, then the top eight's knock-outs:
    quarter-finals, a week later the semi-finals, a week later the final."""
    clubs = [f"Club {k:02d}" for k in range(16)]
    strength = {c: -k for k, c in enumerate(clubs)}
    m = _rows(league, 2025, _double(clubs), strength, step=4)
    last = pd.Timestamp(max(r["date"] for r in m))
    ko = [[(clubs[0], clubs[7]), (clubs[1], clubs[6]), (clubs[2], clubs[5]), (clubs[3], clubs[4])],
          [(clubs[0], clubs[3]), (clubs[1], clubs[2])], [(clubs[0], clubs[1])]]
    return pd.DataFrame(m + _rows(league, 2025, ko, strength, start=last + pd.Timedelta(days=10))), clubs


def test_a_post_season_is_left_out_where_the_league_has_one_and_summed_elsewhere():
    usa, clubs = _knock_outs("USA")
    t = sf.table(usa, "USA", 2025)
    assert all(r["played"] == 30 for r in t) and len(t) == 16
    assert (~sf.regular_season(usa)).sum() == 7
    # the same signs in a league with no knock-outs of its own are a cup week: summed
    n1, _ = _knock_outs("N1")
    assert sf.regular_season(n1).all()
    assert sum(r["played"] for r in sf.table(n1, "N1", 2025)) == 2 * len(n1)


def test_a_top_leagues_table_is_every_result_summed_on_every_day():
    clubs = [f"Club {k:02d}" for k in range(20)]
    m = pd.DataFrame(_rows("E0", 2025, _double(clubs), {c: 0 for c in clubs}, rng=np.random.default_rng(4)))
    for day in sorted(m["date"].unique())[::3] + [None]:
        t = sf.table(m, "E0", 2025, before=day)
        seen = m if day is None else m[m["date"] < day]
        want = _summed(seen)
        for r in t:
            assert [r["played"], r["points"], r["gd"]] == want.get(r["team"], [0, 0, 0])


@pytest.mark.snapshot
def test_the_top_fives_tables_on_the_snapshot_are_every_result_summed():
    import soccer_data as sd
    m = sd.load_matches(leagues=list(sd.TOP))
    done = m[m["result"].isin(("H", "D", "A"))]
    for (lg, season), g in done.groupby(["league", "season"]):
        assert sf.regular_season(g).all(), (lg, season)
        want = _summed(g)
        for r in sf.table(m, lg, int(season)):
            assert [r["played"], r["points"], r["gd"]] == want[r["team"]], (lg, season, r["team"])


def _rest(clubs, played):
    """The pairs of `clubs` not in `played`, greedily into days (no club twice a day)."""
    left = [(a, b) for i, a in enumerate(clubs) for b in clubs[i + 1:]
            if (a, b) not in played and (b, a) not in played]
    days = []
    while left:
        day, busy = [], set()
        for p in list(left):
            if not busy & set(p):
                day.append(p)
                busy |= set(p)
                left.remove(p)
        days.append(day)
    return days


def test_a_split_whose_groups_show_only_after_more_rounds_is_still_found():
    # POL 2018-19: top eight and bottom eight, a round robin each; their first two
    # rounds fall into two cycles of four that are not blocks of the table - the
    # third round joins each group up, and from then on it is told
    clubs = [f"Club {k:02d}" for k in range(16)]
    strength = {c: -k for k, c in enumerate(clubs)}

    def opening(g):
        a = [(g[0], g[3]), (g[4], g[7]), (g[1], g[2]), (g[5], g[6])]
        b = [(g[3], g[4]), (g[7], g[0]), (g[2], g[5]), (g[6], g[1])]
        return [a, b] + _rest(g, set(a + b))
    top, bottom = opening(clubs[:8]), opening(clubs[8:])
    split = [t + b for t, b in zip(top, bottom)]
    m = pd.DataFrame(_rows("POL", 2018, _double(clubs) + split, strength))
    assert (~sf.regular_season(m)).sum() == 56
    assert all(r["played"] == 30 for r in sf.table(m, "POL", 2018))
    days = sorted(m["date"].unique())
    two_in = sf.table(m, "POL", 2018, before=days[32])         # two split rounds played: not told yet
    assert {r["played"] for r in two_in} == {32}
    three_in = sf.table(m, "POL", 2018, before=days[33])
    assert {r["played"] for r in three_in} == {30}


def _late_joiner_frame():
    """10 clubs, 9 rounds played and round 10 posted (the median club: 9
    results, 10 with the fixtures); Club 09's last two results filed under a
    second spelling, 'Club 09 FC' - a club with 2 results."""
    clubs = CLUBS[:10]
    m = pd.DataFrame(_rows("ROU", 2026, _double(clubs)[:10], STRENGTH))
    posted = m["date"] == m["date"].max()
    m.loc[posted, ["hg", "ag"]] = np.nan
    m.loc[posted, "result"] = None
    mine = m.index[~posted & ((m["home"] == "Club 09") | (m["away"] == "Club 09"))][-2:]
    for side in ("home", "away"):
        m.loc[mine, side] = m.loc[mine, side].replace("Club 09", "Club 09 FC")
    return m, m.loc[posted, "date"].iloc[0]


def test_a_club_left_out_of_the_table_is_left_out_of_its_sums_on_the_same_rows():
    # Decided twice - on the results alone for the sums, with the fixtures for
    # the listing - 'Club 09 FC' was summed into a table that did not list it:
    # KeyError, and the unattended run stopped. One set now does both.
    m, day = _late_joiner_frame()
    for before in (day, None):
        t = sf.table(m, "ROU", 2026, before=before)
        played = {r["team"]: r["played"] for r in t}
        assert "Club 09 FC" not in played and len(played) == 10
        assert played["Club 09"] == 7 and sum(played.values()) == 2 * (9 * 5 - 2)
    # on its results alone (no fixtures) the median is 9: nobody is from outside
    done = m[m["result"].notna()]
    assert {r["team"] for r in sf.table(done, "ROU", 2026)} == set(CLUBS[:10]) | {"Club 09 FC"}


@pytest.mark.snapshot
def test_g1_2011_12_before_the_replacement_clubs_reach_five_matches_has_a_table():
    # The real case: G1 2011-12's results before 2011-11-27 with that week's
    # fixtures - the median club 9 results, 10 with the fixtures; Doxa Dramas
    # (1 result) and Levadeiakos (2) read as clubs from outside (the notes).
    import soccer_data as sd
    g = sd.load_matches(leagues=["G1"])
    g = g[g["season"] == 2011]
    day = "2011-11-27"
    week = g[(g["date"] >= day) & (g["date"] < "2011-12-04")].copy()
    week[["hg", "ag"]] = np.nan
    week["result"] = None
    frame = pd.concat([g[g["date"] < day], week], ignore_index=True)
    for before in (day, None):
        t = sf.table(frame, "G1", 2011, before=before)
        listed = {r["team"] for r in t}
        assert len(t) == 14 and not listed & {"Doxa Dramas", "Levadeiakos"}
        among = frame[frame["result"].notna() & frame["home"].isin(listed) & frame["away"].isin(listed)]
        assert sum(r["played"] for r in t) == 2 * len(among)


def _last_round_pair(m, clubs):
    """A match of the last regular round between two clubs outside the top
    eight (no knock-outs for either), and the knock-outs' first day."""
    regular = m.iloc[:len(_double(clubs)) * len(clubs) // 2]
    last = regular["date"].max()
    low = set(clubs[8:])
    i = regular.index[(regular["date"] == last) & regular["home"].isin(low) & regular["away"].isin(low)][0]
    return i, last


def test_a_final_regular_match_played_alone_before_the_knock_outs_stays_in_the_table():
    # ARG 2025: on 2025-05-04 every club but River Plate and Velez Sarsfield had
    # played 16; they met on 2025-05-05 and the knock-outs began on 2025-05-10.
    # That match is their last regular one: cut with the knock-outs, both showed
    # 15 played and River three points short.
    usa, clubs = _knock_outs("USA")
    i, last = _last_round_pair(usa, clubs)
    usa.loc[i, "date"] = (pd.Timestamp(last) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    keep = sf.regular_season(usa)
    assert keep[usa.index.get_loc(i)] and (~keep).sum() == 7       # the knock-outs only
    t = sf.table(usa, "USA", 2025)
    assert all(r["played"] == 30 for r in t) and len(t) == 16
    # and the run still starts with two clubs a match short whose match is never
    # played (ARG 2026's Lanus v Estudiantes): the knock-outs are cut all the same
    usa, clubs = _knock_outs("USA")
    i, _ = _last_round_pair(usa, clubs)
    usa = usa.drop(index=i)
    assert (~sf.regular_season(usa)).sum() == 7
    assert sorted(r["played"] for r in sf.table(usa, "USA", 2025)) == [29, 29] + [30] * 14


@pytest.mark.snapshot
def test_arg_2025_keeps_river_v_velez_and_arg_2026_still_cuts_its_knock_outs():
    import soccer_data as sd
    m = sd.load_matches(leagues=["ARG"])
    done = m[m["result"].isin(("H", "D", "A"))]
    g = done[done["season"] == 2025]
    keep = sf.regular_season(g)
    late = ((g["date"] == "2025-05-05") & (g["home"] == "River Plate") & (g["away"] == "Velez Sarsfield")).to_numpy()
    assert late.sum() == 1 and keep[late].all() and (~keep).sum() == 30   # the two halves' knock-outs
    apertura = sf.table(m[m["date"] < "2025-07-01"], "ARG", 2025)
    assert {r["played"] for r in apertura} == {16}
    assert (~sf.regular_season(done[done["season"] == 2026])).sum() == 15
