"""The same-game tickets' history (experiments/nfl_sgp_backtest.py) is point
in time: a season's TD and win chances come from models fitted on the
seasons before it, the rookie flag reads only earlier seasons, and a
ticket reads nothing of its own game's result - and the history on file
says why its return is missing."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import nfl_backtest as bt
import nfl_data as nd
import nfl_sgp as sgp
import nfl_td as nt

from conftest import ENGINE

sys.path.insert(0, str(ENGINE / "experiments"))
import nfl_sgp_backtest as bk  # noqa: E402

SEASON = 2016
SHIPS = ("1-3", "3+")


@pytest.fixture(scope="module")
def td_frame(games):
    return nt.build(games, nt.load_lines())


@pytest.fixture(scope="module")
def player_games():
    return nd.load_player_games()


@pytest.fixture(scope="module")
def history(td_frame, player_games):
    return bk.td_history(td_frame, SHIPS, player_games, start=SEASON, end=SEASON)


def _scramble_from(frame, season, cols):
    """Every value of `cols` from `season` on replaced by noise."""
    f = frame.copy()
    later = f["season"] >= season
    rng = np.random.default_rng(0)
    for c in cols:
        f.loc[later, c] = rng.integers(0, 3, later.sum()).astype(float)
    return f


def test_a_seasons_td_chances_come_from_a_model_fitted_on_the_seasons_before_it(td_frame, player_games,
                                                                                 history):
    rows = td_frame[td_frame["season"] == SEASON]
    train = td_frame[nt.labelled(td_frame) & td_frame["season"].between(nt.TRAIN_FROM, SEASON - 1)]
    expected = pd.Series(nt.fit(train, nt.MODEL).predict(rows), index=rows.index)
    ships = nt.frame_groups(rows).isin(SHIPS)
    got = history.loc[rows.index, "prob"]
    assert ships.any() and (~ships).any()
    np.testing.assert_allclose(got[ships].to_numpy(float), expected[ships].to_numpy(float))
    assert got[~ships].isna().all()                  # unshipped groups: no chance, as live
    # The season's own labels, and every later season, reach none of it.
    hidden = _scramble_from(td_frame, SEASON, ["tds", "scored"])
    again = bk.td_history(hidden[hidden["season"] <= SEASON], SHIPS, player_games, start=SEASON, end=SEASON)
    pd.testing.assert_series_equal(again["prob"], history["prob"])


def test_a_seasons_win_chances_come_from_the_seasons_before_it(built):
    frame = built[0].copy()
    frame["market_prob"] = bt.market_prob(frame)
    full = bk.blend_history(frame, end=SEASON + 1)
    hidden = _scramble_from(frame, SEASON, ["result", "home_score", "away_score", "total"])
    cut = bk.blend_history(hidden[hidden["season"] <= SEASON], end=SEASON)
    ids = frame.loc[frame["season"] == SEASON, "game_id"]
    assert full[ids].notna().all()
    np.testing.assert_allclose(full[ids].to_numpy(float), cut[ids].to_numpy(float))


def test_the_rookie_flag_reads_only_earlier_seasons(td_frame, player_games, history):
    earlier = player_games[player_games["season"] < SEASON]
    cut = bk.td_history(td_frame[td_frame["season"] <= SEASON], SHIPS, earlier, start=SEASON, end=SEASON)
    pd.testing.assert_series_equal(cut["rookie"], history["rookie"])
    assert history["rookie"].any() and not history["rookie"].all()


def test_a_ticket_reads_nothing_of_its_own_games_result(games, history):
    week = games[(games["season"] == SEASON) & (games["week"] == 9)]
    blend = 0.6
    for g in week.itertuples(index=False):
        rows = history[history["game_id"] == g.game_id]
        hid = g._replace(home_score=np.nan, away_score=np.nan, played=False)
        a = sgp.build_tickets(bk.game_legs(g, blend, rows))
        b = sgp.build_tickets(bk.game_legs(hid, blend, rows.assign(tds=np.nan)))
        assert a == b and any(t["built"] for t in a)


def test_a_td_price_history_fills_in_the_return(tmp_path, games, history):
    g = next(games[(games["season"] == SEASON) & (games["week"] == 9)].itertuples(index=False))
    rows = history[history["game_id"] == g.game_id]
    assert bk.td_prices(tmp_path / "none.csv.gz") is None
    path = tmp_path / "td_price_history.csv.gz"
    pd.DataFrame({"game_id": g.game_id, "gsis_id": rows["gsis_id"], "decimal": 3.0}).to_csv(
        path, index=False, compression="gzip")
    prices = bk.td_prices(path)
    tickets = [sgp.grade_ticket(t, bk.game_outcome(g, rows))
               for t in sgp.build_tickets(bk.game_legs(g, 0.6, rows, prices))]
    built = [t for t in tickets if t["built"]]
    assert built and all(t["est_payout"] is not None for t in built)
    m = sgp.summarise([dict(t, season=g.season) for t in tickets])
    assert all(m[t["kind"]]["roi"] is not None for t in built)


def test_the_history_on_file_has_every_kind_and_period_and_says_why_return_is_missing():
    path = nd.DATA / f"{bk.OUT_NAME}.json"
    if not path.exists():
        pytest.skip("no history on file yet")
    h = json.loads(path.read_text())
    assert h["version"] == sgp.VERSION and h["known_bias"] == sgp.KNOWN_BIAS
    # The seasons the shipped usage groups were chosen on are nfl_td's own periods, not typed.
    assert f" chosen on {nt.DEV[0]}-{nt.CONFIRM[1]} walk-forward " in h["td_groups"]["note"]
    assert set(h["periods"]) == set(sgp.PERIODS) | {str(h["current_season"]), "all"}
    assert set(h["kinds"]) == set(sgp.KINDS)
    for kind, k in h["kinds"].items():
        for name, m in k["meters"].items():
            assert m["won"] + m["lost"] + m["void"] == m["graded"] <= m["tickets"]
            assert m["tickets"] + m["not_built"] == h["games"][name]
            assert sum(m["not_built_why"].values()) == m["not_built"]
            if not h["td_prices"]["on_file"]:
                assert m["roi"] is None and m["roi_reason"] == sgp.NO_TD_PRICE


def test_the_history_is_replayed_only_when_a_new_final_has_its_box_score_and_snaps(tmp_path):
    """The runner replays before every predict run with --if-stale: a game
    without its box score and snap counts would replay to the same file."""
    path = tmp_path / "h.json"
    games = pd.DataFrame({"game_id": ["a", "b", "c"], "season": [2026, 2026, 2026],
                          "played": [True, True, False],
                          "kickoff": pd.to_datetime(["2026-11-01 18:00", "2026-11-08 18:00", "2026-11-15 18:00"])})
    box, snaps = pd.DataFrame({"game_id": ["a"]}), pd.DataFrame({"game_id": ["a"]})
    assert bk.stale(path, games, box, snaps) == "no history on file"
    path.write_text(json.dumps({"version": "sgp-v0"}))
    assert "sgp-v0" in bk.stale(path, games, box, snaps)
    path.write_text(json.dumps({"version": sgp.VERSION, "through": "2026-11-01T18:00:00Z"}))
    assert bk.stale(path, games, box, snaps) is None                   # b: no box score yet
    box = pd.DataFrame({"game_id": ["a", "b", "c"]})
    assert bk.stale(path, games, box, snaps) is None                   # nor snap counts
    snaps = pd.DataFrame({"game_id": ["a", "b", "c"]})
    assert bk.stale(path, games, box, snaps) == "games final since 2026-11-01T18:00:00Z not in it: 1"
