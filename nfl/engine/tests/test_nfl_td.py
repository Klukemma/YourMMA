"""Anytime touchdown: nothing from a game reaches its own row, a season is
predicted only from the seasons before it, the labels are the box score's,
and the live list is the backtest's model reading the same features."""

import numpy as np
import pandas as pd
import pytest

import nfl_features as nf
import nfl_td as nt

SEASON, WEEK = 2023, 9    # Thursday, a Frankfurt morning game, the 1pm slate, Sunday and Monday night


@pytest.fixture(scope="session")
def td_lines():
    return nt.load_lines()


@pytest.fixture(scope="session")
def td_built(games, td_lines):
    return nt.build_with_state(games, td_lines)


def _hide_results_from(games, lines, cutoff):
    """Every result not final at `cutoff` hidden: the score removed, and the
    box score and snap counts of those games scrambled. Who took a snap is
    kept - it is the row itself, the player the probability is about."""
    g = games.copy()
    later = g["kickoff"] + nf.FINAL_AFTER > cutoff
    g.loc[later, ["home_score", "away_score", "result", "total"]] = np.nan
    g.loc[later, "played"] = False
    lines = lines.copy()
    hit = lines["game_id"].isin(set(g.loc[later, "game_id"]))
    rng = np.random.default_rng(0)
    for col in nt.STATS[:-1]:
        lines.loc[hit, col] = rng.integers(0, 5, hit.sum()).astype(float)
    lines.loc[hit, "tds"] = lines.loc[hit, "rushing_tds"] + lines.loc[hit, "receiving_tds"]
    lines.loc[hit, "scored"] = (lines.loc[hit, "tds"] >= 1).astype(float)
    snapped = hit & lines["in_snaps"]
    lines.loc[snapped, "offense_pct"] = rng.uniform(0.01, 1.0, snapped.sum())
    return g, lines


@pytest.fixture(scope="module")
def hidden(games, td_lines):
    """build_with_state() with results hidden from a kickoff on, cached by kickoff."""
    cache = {}

    def at(kickoff):
        if kickoff not in cache:
            g, lines = _hide_results_from(games, td_lines, kickoff)
            cache[kickoff] = (g, nt.build_with_state(g, lines))
        return cache[kickoff]
    return at


def _kickoffs(frame):
    wk = frame[(frame["season"] == SEASON) & (frame["week"] == WEEK)]
    slots = wk["kickoff"].value_counts()
    return {"first": wk["kickoff"].min(), "busiest": slots.idxmax(), "last": wk["kickoff"].max()}


@pytest.mark.parametrize("slot", ["first", "busiest", "last"])
def test_a_game_reads_the_same_features_with_its_future_hidden(td_built, hidden, slot):
    full = td_built[0]
    kickoff = _kickoffs(full)[slot]
    _, (cut, _) = hidden(kickoff)
    a = full[full["kickoff"] == kickoff].set_index(["game_id", "gsis_id"])[nt.FEATURES]
    b = cut.set_index(["game_id", "gsis_id"]).loc[a.index, nt.FEATURES]
    assert len(a) > 20
    pd.testing.assert_frame_equal(a.astype(float), b.astype(float), check_exact=False, atol=1e-12)
    # And the rows hidden carry no label.
    assert cut.loc[cut["kickoff"] >= kickoff, "scored"].isna().all()


def test_the_weeks_after_do_change_once_results_exist(td_built, hidden):
    # The control: the test above would pass vacuously if nothing ever moved.
    full = td_built[0]
    _, (cut, _) = hidden(_kickoffs(full)["first"])
    later = full[(full["season"] == SEASON) & (full["week"] == WEEK + 2)].set_index(["game_id", "gsis_id"])
    b = cut.set_index(["game_id", "gsis_id"]).loc[later.index]
    assert not np.allclose(later["ew_carries"], b["ew_carries"])
    assert not np.allclose(later["games_season"], b["games_season"])


@pytest.fixture(scope="module")
def wf(td_built):
    return nt.walk_forward(td_built[0], start=2015, end=2016)


def test_a_season_is_predicted_the_same_without_the_seasons_after_it(td_built, wf):
    frame = td_built[0]
    cut = nt.walk_forward(frame[frame["season"] <= 2015], start=2015)
    rows = frame.index[frame["season"] == 2015]
    pd.testing.assert_frame_equal(wf.loc[rows], cut.loc[rows])


def test_probabilities_are_probabilities(td_built, wf):
    frame = td_built[0]
    sel = frame["season"].between(2015, 2016)
    for col in list(nt.LEARNERS + nt.BASELINES) + ["prob"]:
        p = wf.loc[sel, col]
        assert p.notna().all() and ((p > 0) & (p < 1)).all(), col
    assert (wf.loc[frame["season"] < 2015].isna().all()).all()


def test_the_periods_do_not_overlap_and_train_before_both():
    assert nt.TRAIN_FROM < nt.DEV[0] <= nt.DEV[1] < nt.CONFIRM[0]
    assert nt.MODEL in nt.LEARNERS


def _row(frame, game_id, name):
    r = frame[(frame["game_id"] == game_id) & (frame["name"] == name)]
    assert len(r) == 1, (game_id, name)
    return r.iloc[0]


def test_labels_are_the_box_score(td_built, td_lines):
    frame = td_built[0]
    # Super Bowl LVII: Hurts ran in three, A.J. Brown and Kelce caught one
    # each; Mahomes and DeVonta Smith did not score.
    sb = "2022_22_KC_PHI"
    assert _row(frame, sb, "Jalen Hurts")["tds"] == 3 and _row(frame, sb, "Jalen Hurts")["scored"] == 1
    for name in ("A.J. Brown", "Travis Kelce", "Isiah Pacheco"):
        assert _row(frame, sb, name)["scored"] == 1, name
    for name in ("Patrick Mahomes", "DeVonta Smith"):
        assert _row(frame, sb, name)["scored"] == 0, name
    # Super Bowl XLIX, whose snap counts swap the two sides: Lynch scored
    # for Seattle, Brady ran none in for New England.
    xlix = "2014_21_NE_SEA"
    lynch = _row(frame, xlix, "Marshawn Lynch")
    assert lynch["team"] == "SEA" and lynch["scored"] == 1
    assert _row(frame, xlix, "Tom Brady")["team"] == "NE" and _row(frame, xlix, "Tom Brady")["scored"] == 0
    # Took snaps, no box-score line: a row, and no touchdown.
    quiet = td_lines[(td_lines["game_id"] == sb) & td_lines["in_snaps"] & ~td_lines["has_stats"]]
    assert len(quiet)
    rows = frame[(frame["game_id"] == sb) & frame["gsis_id"].isin(quiet["gsis_id"])]
    assert len(rows) == len(quiet) and (rows["scored"] == 0).all()


def test_every_row_took_an_offensive_snap(td_built, td_lines):
    frame = td_built[0]
    assert frame["season"].min() >= nt.TRAIN_FROM
    assert len(frame) == int(td_lines["in_snaps"].sum())
    assert set(frame["position"]) <= set(nt.POSITIONS)
    assert not frame.duplicated(["game_id", "gsis_id"]).any()


def test_implied_points_split_the_total_by_the_home_spread():
    assert nt.implied_points(47.0, 3.0, home=True) == 25.0
    assert nt.implied_points(47.0, 3.0, home=False) == 22.0
    assert nt.implied_points(None, 3.0, home=True) == nt.DEFAULT_IMPLIED
    assert nt.implied_points(float("nan"), 3.0, home=False) == nt.DEFAULT_IMPLIED


@pytest.fixture(scope="module")
def live(games, td_built):
    return nt.fit_live(int(td_built[0]["season"].max()), games=games, built=td_built)


def _a_team_that_has_played(frame, season):
    return frame[frame["season"] == season].sort_values("kickoff")["team"].iloc[-1]


def test_the_live_list_is_sorted_and_leaves_out_the_excluded(td_built, live):
    team = _a_team_that_has_played(td_built[0], live.season)
    picks = nt.predict_team(live, team, 24.0, home=True)
    probs = [c["prob"] for c in picks]
    assert len(picks) >= 5
    assert probs == sorted(probs, reverse=True) and all(0 < p < 1 for p in probs)
    assert all(c["last_team"] == team for c in picks)
    out = {picks[0]["gsis_id"], picks[2]["gsis_id"]}
    rest = nt.predict_team(live, team, 24.0, home=True, exclude=out)
    assert not out & {c["gsis_id"] for c in rest}
    assert [c["gsis_id"] for c in rest] == [c["gsis_id"] for c in picks if c["gsis_id"] not in out]
    # More expected points, more touchdowns.
    richer = nt.predict_team(live, team, 31.0, home=True)
    assert richer[0]["prob"] > picks[0]["prob"]


def test_a_team_yet_to_play_draws_on_last_season(games, td_built):
    frame = td_built[0]
    last = int(frame["season"].max())
    nxt = nt.fit_live(last + 1, games=games, built=td_built)
    team = _a_team_that_has_played(frame, last)
    picks = nt.predict_team(nxt, team, nt.DEFAULT_IMPLIED, home=False)
    assert picks and all(c["last_team"] == team and c["tds_season"] == 0 for c in picks)


def test_live_reads_what_the_backtest_read(td_built, hidden):
    """Predict the busiest kickoff's games live, from the state with them
    still to play, and get the backtest's own probabilities: same features,
    same model."""
    full = td_built[0]
    kickoff = _kickoffs(full)["busiest"]
    g, built = hidden(kickoff)
    live = nt.fit_live(SEASON, games=g, built=built)
    matched = 0
    for game in g[g["kickoff"] == kickoff].itertuples():
        for team, home in ((game.home, True), (game.away, False)):
            rows = full[(full["game_id"] == game.game_id) & (full["team"] == team)]
            expected = dict(zip(rows["gsis_id"], live.model.predict(rows)))
            position = dict(zip(rows["gsis_id"], rows["position"]))
            got = nt.predict_team(live, team, nt.implied_points(game.total_line, game.spread_line, home),
                                  home, roster=rows["gsis_id"], min_usage=0.0)
            for c in got:
                if c["position"] == position[c["gsis_id"]]:   # a player relisted this week reads his new position
                    assert c["prob"] == pytest.approx(expected[c["gsis_id"]], abs=1e-12)
                    matched += 1
    assert matched > 100
