"""The price-dispersion candidate (experiments/price_dispersion.py): with no
extra weights it is today's pick exactly, and its predictions read only
earlier seasons and their own row's pre-match prices. (The scramble test on
the real history runs in the module's own run: python
experiments/price_dispersion.py.)"""

import math

import numpy as np
import pandas as pd
import pytest

import soccer_data as sd
import soccer_model
from experiments import price_dispersion as pdx

TEAMS = [f"Club {c}" for c in "ABCDEFGHIJKLMNOPQRST"]


def _double_round_robin():
    n, fixed, rest = len(TEAMS), TEAMS[0], TEAMS[1:]
    first = []
    for r in range(n - 1):
        order = [fixed] + rest[r:] + rest[:r]
        first += [(order[i], order[n - 1 - i]) for i in range(n // 2)]
    return first + [(a, h) for h, a in first]


def _league(seasons=range(2005, 2009), seed=0):
    """A made-up top league, 380 matches a season: results follow strength,
    the average price follows it with noise, the best price sits above the
    average, Bet365 near it, and an over/under price."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.6, 0.6, len(TEAMS))))
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        for k, (h, a) in enumerate(_double_round_robin()):
            date = day + pd.Timedelta(days=7 * (k // 10))
            ph = 1 / (1 + math.exp(-(0.3 + 2 * (strength[h] - strength[a]))))
            p = np.array([0.74 * ph, 0.26, 0.74 * (1 - ph)])
            r = rng.choice(3, p=p / p.sum())
            hg, ag = [(2, 0), (1, 1), (0, 1)][r]
            hg += int(rng.integers(0, 2))
            ag += int(hg > ag and rng.random() < 0.3)
            res = "H" if hg > ag else "A" if hg < ag else "D"
            noisy = p * np.exp(rng.normal(0, 0.05, 3))
            avg = 0.94 / (noisy / noisy.sum())
            mx = avg * (1 + rng.uniform(0.01, 0.12, 3))
            b365 = avg * np.exp(rng.normal(0, 0.02, 3))
            po = float(np.clip(rng.normal(0.5, 0.06), 0.2, 0.8))
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": date,
                         "home": h, "away": a, "hg": hg, "ag": ag, "result": res,
                         "hs": hg * 4 + 5, "as": ag * 4 + 4, "hst": hg + 2, "ast": ag + 1,
                         "avg_h": avg[0], "avg_d": avg[1], "avg_a": avg[2],
                         "max_h": mx[0], "max_d": mx[1], "max_a": mx[2],
                         "b365_h": b365[0], "b365_d": b365[1], "b365_a": b365[2],
                         "avg_o25": 0.95 / po, "avg_u25": 0.95 / (1 - po),
                         "max_o25": 1.0 / po, "max_u25": 1.0 / (1 - po)})
    return sd.frame(rows)


def test_with_no_extra_weights_it_is_the_pick():
    m = _league()
    base = pdx.predict_walk_forward(m, {"scalars": (), "directional": (), "kappa": 0.0})
    pick = pdx.shipped_pick(m)
    a, b = base[["p_h", "p_d", "p_a"]].to_numpy(float), pick.to_numpy(float)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.isfinite(a).any()
    assert np.nanmax(np.abs(a - b)) < 1e-6
    nan = pd.DataFrame(np.nan, index=m.index, columns=["p_h", "p_d", "p_a"])
    _, weights = soccer_model.pick(m, nan)
    for season, w in weights.items():
        assert np.allclose(base.attrs["weights"][season]["1x2"], w, atol=1e-5)


def test_chosen_prices_every_priced_match_it_can_and_sums_to_one():
    m = _league()
    c = pdx.predict_walk_forward(m)
    assert list(c.columns) == ["p_h", "p_d", "p_a"]           # no over-2.5 of its own (CHOSEN["ou"] is None)
    ok = c["p_h"].notna()
    assert set(m.loc[ok, "season"]) == {2008}                 # 2005-2007 hold 1,140 >= MIN_TRAIN matches
    assert np.allclose(c.loc[ok].sum(axis=1), 1.0)
    fit = pdx.predict_walk_forward(m, {"ou": "fit"})
    assert fit["p_o25"].notna().sum() == ok.sum()


def test_missing_best_price_falls_back_to_zero_feature_not_to_nothing():
    m = _league()
    m.loc[m["season"] == 2008, ["max_h", "max_d", "max_a"]] = np.nan
    c = pdx.predict_walk_forward(m)
    assert c.loc[m["season"] == 2008, "p_h"].notna().all()


def test_scramble_from_a_cut_moves_nothing_before_it():
    m = _league()
    cut = m.loc[(m["season"] == 2008), "date"].sort_values().iloc[150]
    r = pdx.pit_check(m, cut)
    assert r["before_cut_predictions"] > 0 and r["cut_day_predictions"] > 0
    assert r["before_cut_unmoved"] and r["cut_day_unmoved_results_scrambled"]
    assert r["after_cut_moved"]


def test_a_rows_features_are_its_own_prices_only():
    m = _league()
    f0 = pdx.price_features(m)
    s = m.copy()
    s.loc[s.index[1:], ["avg_h", "max_d", "b365_a", "avg_o25"]] *= 1.7
    f1 = pdx.price_features(s)
    for k in ("over", "spread", "b365", "lop", "ou_over", "ou_spread", "ou_dmax"):
        assert f0[k][0] == f1[k][0]
    assert np.array_equal(f0["dmax"][0], f1["dmax"][0])


def test_it_refuses_to_score_confirm():
    m = _league()
    c = pdx.predict_walk_forward(m)
    with pytest.raises(ValueError):
        pdx.compare(m, c, pdx.shipped_pick(m), 2016, 2025)
