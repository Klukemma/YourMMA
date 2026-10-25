"""The soccer scoring harness: the yardstick every model is measured with."""

import math

import numpy as np
import pandas as pd
import pytest

import soccer_backtest as bt


def _matches(n=400, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2016-08-13", periods=n // 4, freq="7D").strftime("%Y-%m-%d")
    rows = []
    for i in range(n):
        hg, ag = rng.poisson(1.5), rng.poisson(1.1)
        rows.append({"match_id": f"m{i}", "league": "E0", "season": 2016 + i // 200, "date": days[i // 4],
                     "home": f"T{i % 10}", "away": f"T{(i + 3) % 10}", "hg": hg, "ag": ag,
                     "result": "H" if hg > ag else "A" if hg < ag else "D",
                     "avg_h": 2.1, "avg_d": 3.4, "avg_a": 3.6, "avg_o25": 1.9, "avg_u25": 1.9,
                     "avgc_h": np.nan, "avgc_d": np.nan, "avgc_a": np.nan})
    return pd.DataFrame(rows)


def test_no_vig_removes_the_margin_in_proportion():
    df = pd.DataFrame({"avg_h": [2.0], "avg_d": [3.5], "avg_a": [4.0]})
    p = bt.no_vig(df).iloc[0]
    assert math.isclose(p.sum(), 1.0)
    assert p["p_h"] > p["p_d"] > p["p_a"]
    assert bt.no_vig(pd.DataFrame({"avg_h": [2.0], "avg_d": [np.nan], "avg_a": [4.0]})).isna().all(axis=1).iloc[0]


def test_a_uniform_forecast_scores_log_three():
    res = pd.Series(["H", "D", "A", "H"])
    p = pd.DataFrame({"p_h": [1 / 3] * 4, "p_d": [1 / 3] * 4, "p_a": [1 / 3] * 4})
    assert math.isclose(bt.log_loss(p, res), math.log(3))
    assert math.isclose(bt.brier(p, res), 2 / 3)


def test_pooling_with_all_weight_on_the_market_returns_the_market():
    m = pd.DataFrame({"p_h": [0.6, 0.2], "p_d": [0.25, 0.3], "p_a": [0.15, 0.5]})
    q = pd.DataFrame({"p_h": [0.5, 0.3], "p_d": [0.3, 0.3], "p_a": [0.2, 0.4]})
    out = bt.pool(m, q, (0.0, 1.0, 0.0, 0.0))
    assert np.allclose(out.to_numpy(), q.to_numpy())


def test_the_point_in_time_guard_catches_a_model_that_peeks():
    m = _matches()
    cut = m["date"].iloc[100]

    def honest(df):
        # Each match gets the home win rate of matches on EARLIER days.
        out = []
        for d in df["date"]:
            past = df[df["date"] < d]
            h = (past["result"] == "H").mean() if len(past) else 0.45
            out.append((h, (1 - h) / 2, (1 - h) / 2))
        return pd.DataFrame(out, index=df.index, columns=["p_h", "p_d", "p_a"])

    def peeking(df):
        # ...and this one also counts the same day's results.
        out = []
        for d in df["date"]:
            past = df[df["date"] <= d]
            h = (past["result"] == "H").mean()
            out.append((h, (1 - h) / 2, (1 - h) / 2))
        return pd.DataFrame(out, index=df.index, columns=["p_h", "p_d", "p_a"])

    assert bt.check_point_in_time(honest, m, cut)
    with pytest.raises(AssertionError):
        bt.check_point_in_time(peeking, m, cut)


def test_the_market_scored_against_itself_matches_its_own_numbers():
    m = _matches()
    rep = bt.report(m, bt.no_vig(m), "market", periods=((2016, 2017),))["market"]["2016-2017"]
    assert rep["matches"] == 400 and rep["log_loss"] == rep["market_log_loss"]
    assert "over_2_5" not in rep
