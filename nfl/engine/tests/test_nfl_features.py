"""Pre-game features: nothing a game's features read may be decided after
its kickoff. Tested the direct way - delete the future and rebuild."""

import numpy as np
import pandas as pd
import pytest

import nfl_elo as ne
import nfl_features as nf
import nfl_model as nm

QB_FIELDS = ["home_qb_rating", "away_qb_rating", "home_qb_change", "away_qb_change", "qb_diff"]


def _hide_results_from(games, cutoff):
    g = games.copy()
    later = g["kickoff"] >= cutoff
    g.loc[later, ["home_score", "away_score", "result", "total"]] = np.nan
    g.loc[later, "played"] = False
    return g


@pytest.mark.parametrize("season,week", [(2019, 9), (2024, 12)])
def test_a_game_reads_the_same_features_with_its_future_deleted(games, team_games, qb_games, built, season, week):
    """The exact rule: hide every result from a kickoff onwards, rebuild, and
    the games starting at that kickoff must read identical features. (Earlier
    games the same week - Thursday night before Sunday - are rightly known.)"""
    full, _ = built
    wk = full[(full["season"] == season) & (full["week"] == week)]
    cols = nf.FEATURES + nf.TOTAL_FEATURES + ["elo_prob", "home_elo", "away_elo"]
    for kickoff in sorted(wk["kickoff"].unique()):
        truncated = nf.build(_hide_results_from(games, kickoff), team_games, qb_games, nm.ELO)
        a = wk[wk["kickoff"] == kickoff].set_index("game_id")[cols]
        b = truncated.set_index("game_id").loc[a.index, cols]
        pd.testing.assert_frame_equal(a.astype(float), b.astype(float), check_exact=False, atol=1e-12)


def test_the_week_after_does_change_once_results_exist(games, team_games, qb_games, built):
    # The control: the test above would pass vacuously if nothing ever moved.
    full, _ = built
    first = full[(full["season"] == 2024) & (full["week"] == 12)]["kickoff"].min()
    truncated = nf.build(_hide_results_from(games, first), team_games, qb_games, nm.ELO)
    nxt = full[(full["season"] == 2024) & (full["week"] == 14)].set_index("game_id")
    b = truncated.set_index("game_id").loc[nxt.index]
    assert not np.allclose(nxt["elo_diff"].astype(float), b["elo_diff"].astype(float))


def test_an_unplayed_game_does_not_move_elo(games):
    head = games.head(300).copy()
    frame, ratings = ne.run(head)
    head.loc[head.index[-1], "played"] = False
    _, ratings2 = ne.run(head)
    last = head.iloc[-1]
    # Unplayed, the game leaves its teams exactly where it found them.
    assert ratings2[last["home"]] == pytest.approx(frame.iloc[-1]["home_elo"])
    assert ratings2[last["away"]] == pytest.approx(frame.iloc[-1]["away_elo"])
    assert ratings[last["home"]] != ratings2[last["home"]]


def test_unplayed_games_without_a_projection_assume_the_last_starter(built):
    frame, state = built
    future = frame[~frame["played"]]
    if future.empty:
        pytest.skip("no unplayed games in the snapshot")
    assumed = future[future["home_qb_source"] == "last starter"]
    for row in assumed.itertuples():
        assert row.home_qb_used == state["last_qb"][row.home]


def test_swapping_a_quarterback_recomputes_the_difference(built):
    frame, state = built
    row = frame[frame["played"]].iloc[-1]
    swapped = nf.with_quarterback(row, "home", None, state, "replaced")
    assert swapped["home_qb_rating"] == nf.FeatureParams().qb_prior
    assert swapped["qb_diff"] == pytest.approx(swapped["home_qb_rating"] - swapped["away_qb_rating"])
    assert swapped["home_qb_change"] == 1
