"""Yardage projections: nothing from a game reaches its own row, a season is
projected only from the seasons before it, the labels are the box score's,
projections are never negative, and the live projection is the backtest's
model reading the same features."""

import json

import numpy as np
import pandas as pd
import pytest

import nfl_features as nf
import nfl_props as npp

SEASON, WEEK = 2023, 9    # Thursday, a Frankfurt morning game, the 1pm slate, Sunday and Monday night


@pytest.fixture(scope="session")
def props_lines():
    return npp.load_lines()


@pytest.fixture(scope="session")
def props_built(games, props_lines):
    return npp.build_with_state(games, props_lines)


def _hide_results_from(games, lines, cutoff):
    """Every result not final at `cutoff` hidden: the score removed, and the
    box score and snap counts of those games scrambled. Who took a snap is
    kept - it is the row itself, the player the projection is about."""
    g = games.copy()
    later = g["kickoff"] + nf.FINAL_AFTER > cutoff
    g.loc[later, ["home_score", "away_score", "result", "total"]] = np.nan
    g.loc[later, "played"] = False
    lines = lines.copy()
    hit = lines["game_id"].isin(set(g.loc[later, "game_id"]))
    rng = np.random.default_rng(0)
    for col in npp.STATS:
        lines.loc[hit, col] = rng.integers(0, 300, hit.sum()).astype(float)
    snapped = hit & lines["in_snaps"]
    lines.loc[snapped, "offense_pct"] = rng.uniform(0.01, 1.0, snapped.sum())
    return g, lines


@pytest.fixture(scope="module")
def hidden(games, props_lines):
    """build_with_state() with results hidden from a kickoff on, cached by kickoff."""
    cache = {}

    def at(kickoff):
        if kickoff not in cache:
            g, lines = _hide_results_from(games, props_lines, kickoff)
            cache[kickoff] = (g, npp.build_with_state(g, lines))
        return cache[kickoff]
    return at


def _kickoffs(frame):
    wk = frame[(frame["season"] == SEASON) & (frame["week"] == WEEK)]
    slots = wk["kickoff"].value_counts()
    return {"first": wk["kickoff"].min(), "busiest": slots.idxmax(), "last": wk["kickoff"].max()}


@pytest.mark.parametrize("slot", ["first", "busiest", "last"])
def test_a_game_reads_the_same_features_with_its_future_hidden(props_built, hidden, slot):
    full = props_built[0]
    kickoff = _kickoffs(full)[slot]
    _, (cut, _) = hidden(kickoff)
    a = full[full["kickoff"] == kickoff].set_index(["game_id", "gsis_id"])[npp.FEATURES]
    b = cut.set_index(["game_id", "gsis_id"]).loc[a.index, npp.FEATURES]
    assert len(a) > 20
    pd.testing.assert_frame_equal(a.astype(float), b.astype(float), check_exact=False, atol=1e-12)
    # And the rows hidden carry no label.
    assert cut.loc[cut["kickoff"] >= kickoff, npp.LABELS].isna().all().all()


def test_the_weeks_after_do_change_once_results_exist(props_built, hidden):
    # The control: the test above would pass vacuously if nothing ever moved.
    full = props_built[0]
    _, (cut, _) = hidden(_kickoffs(full)["first"])
    later = full[(full["season"] == SEASON) & (full["week"] == WEEK + 2)].set_index(["game_id", "gsis_id"])
    b = cut.set_index(["game_id", "gsis_id"]).loc[later.index]
    for col in ("ew_passing_yards", "ew_rushing_yards", "ew_receiving_yards", "sw_targets", "games_season"):
        assert not np.allclose(later[col], b[col]), col


@pytest.fixture(scope="module")
def wf(props_built):
    return npp.walk_forward(props_built[0], start=2015, end=2016)


def test_a_season_is_projected_the_same_without_the_seasons_after_it(props_built, wf):
    frame = props_built[0]
    cut = npp.walk_forward(frame[frame["season"] <= 2015], start=2015)
    rows = frame.index[frame["season"] == 2015]
    pd.testing.assert_frame_equal(wf.loc[rows], cut.loc[rows])


def test_projections_are_yards_and_only_where_they_apply(props_built, wf):
    frame = props_built[0]
    sel = frame["season"].between(2015, 2016)
    for stat in npp.YARDS:
        on = sel & npp.applies(frame, stat)
        for col in [f"{stat}_{k}" for k in npp.KINDS] + [stat]:
            p = wf.loc[on, col]
            assert p.notna().all() and (p >= 0).all() and np.isfinite(p).all(), col
            assert wf.loc[sel & ~npp.applies(frame, stat), col].isna().all(), col
        # Not a constant: the starter, the bell-cow back and the WR1 are
        # projected well above the rest.
        assert wf.loc[on & npp.measured(frame, stat), stat].std() > 10
    assert wf.loc[frame["season"] < 2015].isna().all().all()


def test_the_report_measures_each_stat_on_its_own_players(props_built, wf):
    frame = props_built[0]
    rep = npp.report(frame, wf)
    json.dumps(rep)   # JSON-able
    dev = rep[f"{npp.DEV[0]}-{npp.DEV[1]}"]
    assert set(dev) == set(npp.YARDS)
    for stat, r in dev.items():
        sel = (npp.labelled(frame) & frame["season"].between(2015, 2016) & npp.measured(frame, stat))
        assert r["rows"] == int(sel.sum())
        assert set(r["mae"]) == set(npp.KINDS) == set(r["median_ae"]) == set(r["over"])
        assert all(0 <= v <= 1 for v in r["over"].values())
        assert sum(b["rows"] for b in r["over_by_bin"]) == r["rows"]
        # The model beats the player's own mean, and is a median: he goes
        # over it in roughly half his games.
        assert r["mae"][npp.MODEL[stat]] < r["mae"]["mean"], stat
        assert 0.4 < r["over"][npp.MODEL[stat]] < 0.6, stat
    assert set(rep["dev_choice"]) == set(npp.YARDS)


def test_the_periods_do_not_overlap_and_train_before_both():
    assert npp.TRAIN_FROM < npp.DEV[0] <= npp.DEV[1] < npp.CONFIRM[0]
    assert set(npp.MODEL) == set(npp.YARDS) and set(npp.MODEL.values()) <= set(npp.LEARNERS)


def _row(frame, game_id, name):
    r = frame[(frame["game_id"] == game_id) & (frame["name"] == name)]
    assert len(r) == 1, (game_id, name)
    return r.iloc[0]


def test_labels_are_the_box_score(props_built, props_lines):
    frame = props_built[0]
    # Super Bowl LVII.
    sb = "2022_22_KC_PHI"
    for name, passing, rushing, receiving in (
            ("Patrick Mahomes", 182, 44, 0), ("Jalen Hurts", 304, 70, 0),
            ("Isiah Pacheco", 0, 76, 0), ("Travis Kelce", 0, 0, 81),
            ("DeVonta Smith", 0, 0, 100), ("A.J. Brown", 0, 0, 96)):
        r = _row(frame, sb, name)
        assert (r["passing_yards"], r["rushing_yards"], r["receiving_yards"]) == (passing, rushing, receiving), name
    assert _row(frame, sb, "Patrick Mahomes")["started"] == 1 and _row(frame, sb, "Jalen Hurts")["started"] == 1
    # Super Bowl XLIX, whose snap counts swap the two sides.
    xlix = "2014_21_NE_SEA"
    brady = _row(frame, xlix, "Tom Brady")
    assert brady["team"] == "NE" and brady["passing_yards"] == 328 and brady["started"] == 1
    lynch = _row(frame, xlix, "Marshawn Lynch")
    assert lynch["team"] == "SEA" and lynch["rushing_yards"] == 102
    # Took snaps, no box-score line: a row, and no yards.
    quiet = props_lines[(props_lines["game_id"] == sb) & props_lines["in_snaps"] & ~props_lines["has_stats"]]
    assert len(quiet)
    rows = frame[(frame["game_id"] == sb) & frame["gsis_id"].isin(quiet["gsis_id"])]
    assert len(rows) == len(quiet) and (rows[npp.LABELS] == 0).all().all()


def test_every_row_took_an_offensive_snap_and_each_side_has_one_starter(props_built, props_lines):
    frame = props_built[0]
    assert frame["season"].min() >= npp.TRAIN_FROM
    assert len(frame) == int(props_lines["in_snaps"].sum())
    assert set(frame["position"]) <= set(npp.POSITIONS)
    assert not frame.duplicated(["game_id", "gsis_id"]).any()
    starters = frame.groupby(["game_id", "team"])["started"].sum()
    assert starters.max() == 1 and (starters == 1).mean() > 0.99
    assert (frame.loc[frame["started"] == 1, "position"] == "QB").all()


@pytest.fixture(scope="module")
def live(games, props_built):
    return npp.fit_live(int(props_built[0]["season"].max()), games=games, built=props_built)


def _a_team_that_has_played(frame, season):
    return frame[frame["season"] == season].sort_values("kickoff")["team"].iloc[-1]


def _total(c):
    return sum(c[k] or 0.0 for k in npp.LIVE_KEY.values())


def test_the_live_projection_applies_by_position_and_leaves_out_the_excluded(props_built, live):
    team = _a_team_that_has_played(props_built[0], live.season)
    picks = npp.project_team(live, team, 24.0, 21.0, home=True)
    assert len(picks) >= 5
    assert [_total(c) for c in picks] == sorted((_total(c) for c in picks), reverse=True)
    for c in picks:
        assert set(c) == {"gsis_id", "name", "position", "pass_yds", "rush_yds", "rec_yds"}
        assert (c["rush_yds"] is None) == (c["position"] == "TE")
        assert (c["rec_yds"] is None) == (c["position"] == "QB")
        assert all(c[k] is None or c[k] >= 0 for k in npp.LIVE_KEY.values())
    # One quarterback's passing yards: the one who started the team's last game.
    passers = [c for c in picks if c["pass_yds"] is not None]
    assert len(passers) == 1 and passers[0]["position"] == "QB"
    assert passers[0]["gsis_id"] == live.state["teams"][team].qb
    assert passers[0]["pass_yds"] > 100
    # Excluded players are gone and the rest are unchanged.
    out = {picks[1]["gsis_id"], picks[3]["gsis_id"]}
    rest = npp.project_team(live, team, 24.0, 21.0, home=True, exclude=out)
    kept = [c for c in picks if c["gsis_id"] not in out]
    assert [c["gsis_id"] for c in rest] == [c["gsis_id"] for c in kept]
    for a, b in zip(rest, kept):   # equal to the last bit but for the batch's float sums
        assert all(a[k] == pytest.approx(b[k], abs=1e-9) for k in npp.LIVE_KEY.values())
    # One player, as the team list has him; none for one excluded or unknown.
    for c in picks[:4]:
        assert npp.project_player(live, c["gsis_id"], team, 24.0, 21.0, home=True) == c
    assert npp.project_player(live, picks[1]["gsis_id"], team, 24.0, 21.0, True, exclude=out) is None
    assert npp.project_player(live, "00-not-a-player", team, 24.0, 21.0, True) is None
    # The starter ruled out: his passing yards go to another quarterback, if any.
    rest = npp.project_team(live, team, 24.0, 21.0, home=True, exclude={passers[0]["gsis_id"]})
    assert all(c["pass_yds"] is None or c["position"] == "QB" for c in rest)
    assert sum(c["pass_yds"] is not None for c in rest) == any(c["position"] == "QB" for c in rest)


def test_more_expected_points_more_passing_yards(props_built, live):
    team = _a_team_that_has_played(props_built[0], live.season)
    lean = npp.project_team(live, team, 14.0, 24.0, home=True)
    rich = npp.project_team(live, team, 34.0, 24.0, home=True)
    qb = lambda picks: next(c["pass_yds"] for c in picks if c["pass_yds"] is not None)
    assert qb(rich) > qb(lean)


def test_a_team_yet_to_play_draws_on_last_season(props_built, live):
    frame = props_built[0]
    last = int(frame["season"].max())
    # The models do not matter here, only whom they are asked about.
    nxt = npp.PropsLive(last + 1, live.models, live.state)
    team = _a_team_that_has_played(frame, last)
    picks = npp.project_team(nxt, team, npp.DEFAULT_IMPLIED, npp.DEFAULT_IMPLIED, home=False)
    assert picks and all(live.state["players"][c["gsis_id"]].team == team for c in picks)


def test_live_reads_what_the_backtest_read(props_built, hidden):
    """Project the busiest kickoff's games live, from the state with them
    still to play, and get the backtest's own walk-forward projections:
    same features, same models."""
    full = props_built[0]
    kickoff = _kickoffs(full)["busiest"]
    g, built = hidden(kickoff)
    live = npp.fit_live(SEASON, games=g, built=built)
    # The walk-forward's projections of those rows: trained on the same
    # seasons, so only the earlier seasons and the rows themselves are needed.
    sub = full[(full["season"] < SEASON) | (full["kickoff"] == kickoff)]
    backtest = pd.concat([npp.walk_forward(sub, start=SEASON, end=SEASON, kinds=(npp.MODEL[s],), stats=(s,))[[s]]
                          for s in npp.YARDS], axis=1)
    matched = {s: 0 for s in npp.YARDS}
    for game in g[g["kickoff"] == kickoff].itertuples():
        for team, home in ((game.home, True), (game.away, False)):
            rows = full[(full["game_id"] == game.game_id) & (full["team"] == team)]
            expected = backtest.loc[rows.index].set_index(rows["gsis_id"])
            position = dict(zip(rows["gsis_id"], rows["position"]))
            got = npp.project_team(live, team, npp.implied_points(game.total_line, game.spread_line, home),
                                   npp.implied_points(game.total_line, game.spread_line, not home), home,
                                   roster=rows["gsis_id"], playoff=game.game_type != "REG", min_usage=0.0)
            for c in got:
                if c["position"] != position[c["gsis_id"]]:   # a player relisted this week reads his new position
                    continue
                for s, key in npp.LIVE_KEY.items():
                    if c[key] is not None:
                        assert c[key] == pytest.approx(expected.loc[c["gsis_id"], s], abs=1e-9), (c, s)
                        matched[s] += 1
    assert matched["pass"] >= 10 and matched["rush"] > 80 and matched["rec"] > 100, matched



def test_a_named_starter_without_history_is_projected_not_replaced(live):
    """A rookie's first start: the card names him, so his yards are his -
    projected as a debut - never handed to the veteran behind him."""
    team = next(t for t, st in live.state["teams"].items() if st.qb)
    out = npp.project_team(live, team, 23.0, 20.0, True, starter="00-ROOKIE-QB",
                              starter_name="Rookie Starter")
    passers = [c for c in out if c["pass_yds"] is not None]
    assert [c["gsis_id"] for c in passers] == ["00-ROOKIE-QB"]
    assert passers[0]["name"] == "Rookie Starter" and passers[0]["pass_yds"] > 0
