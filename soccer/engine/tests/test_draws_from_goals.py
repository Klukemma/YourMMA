"""The draws-from-the-goals-line candidate (experiments/draws_from_goals.py):
with no extra terms it is today's pick exactly (on either training pool), its
goals-implied matrix reproduces the prices it was solved from, and its
predictions read only earlier seasons and their own row's pre-match prices.
(The scramble test on the real history runs in the module's own run:
python experiments/draws_from_goals.py.)"""

import math

import numpy as np
import pandas as pd

import soccer_data as sd
import soccer_model
from experiments import draws_from_goals as dg

TEAMS = [f"Club {c}" for c in "ABCDEFGHIJKLMNOPQRST"]


def _double_round_robin():
    n, fixed, rest = len(TEAMS), TEAMS[0], TEAMS[1:]
    first = []
    for r in range(n - 1):
        order = [fixed] + rest[r:] + rest[:r]
        first += [(order[i], order[n - 1 - i]) for i in range(n // 2)]
    return first + [(a, h) for h, a in first]


def _league(code="E0", seasons=range(2005, 2009), seed=0):
    """A made-up league, 380 matches a season: results follow strength, the
    draw follows the goals line, the average 1X2 and over/under prices follow
    both with noise."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.6, 0.6, len(TEAMS))))
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        for k, (h, a) in enumerate(_double_round_robin()):
            date = day + pd.Timedelta(days=7 * (k // 10))
            po = float(np.clip(rng.normal(0.5, 0.07), 0.25, 0.8))
            draw = 0.26 - 0.3 * (po - 0.5)
            ph = 1 / (1 + math.exp(-(0.3 + 2 * (strength[h] - strength[a]))))
            p = np.array([(1 - draw) * ph, draw, (1 - draw) * (1 - ph)])
            r = rng.choice(3, p=p / p.sum())
            hg, ag = [(2, 0), (1, 1), (0, 1)][r]
            if rng.random() < po:
                hg, ag = hg + 1, ag + 1
            res = "H" if hg > ag else "A" if hg < ag else "D"
            noisy = p * np.exp(rng.normal(0, 0.05, 3))
            avg = 0.94 / (noisy / noisy.sum())
            pon = float(np.clip(po + rng.normal(0, 0.02), 0.1, 0.9))
            rows.append({"match_id": sd.match_id(code, date.date(), h, a), "league": code, "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": date,
                         "home": h, "away": a, "hg": hg, "ag": ag, "result": res,
                         "hs": hg * 4 + 5, "as": ag * 4 + 4, "hst": hg + 2, "ast": ag + 1,
                         "avg_h": avg[0], "avg_d": avg[1], "avg_a": avg[2],
                         "avg_o25": 0.95 / pon, "avg_u25": 0.95 / (1 - pon)})
    return sd.frame(rows)


def _two_tiers():
    return sd.frame(pd.concat([_league("E0"), _league("E1", seed=1)], ignore_index=True).to_dict("records"))


def test_with_no_extra_terms_it_is_the_pick():
    for m, pool in ((_league(), "top5"), (_two_tiers(), "ten")):
        base = dg.predict_walk_forward(m, {"terms": (), "pool": pool})
        pick = dg.shipped_pick(m)
        a, b = base[["p_h", "p_d", "p_a"]].to_numpy(float), pick.to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.isfinite(a).any()
        assert np.nanmax(np.abs(a - b)) < 1e-6
        nan = pd.DataFrame(np.nan, index=m.index, columns=["p_h", "p_d", "p_a"])
        _, weights = soccer_model.pick(m, nan)
        for season, w in weights.items():
            assert np.allclose(base.attrs["weights"][season]["1x2"][:3], w, atol=1e-5)


def test_goals_implied_matrix_reproduces_its_prices():
    q3 = np.array([[0.5, 0.27, 0.23], [0.2, 0.25, 0.55], [0.4, 0.32, 0.28], [np.nan, 0.3, 0.3]])
    q_o = np.array([0.55, 0.62, 0.40, 0.5])
    for rho in (0.0, -0.06):
        K, mu = dg.goals_implied(q3, q_o, rho)
        assert np.isnan(K[3]).all()
        assert np.allclose(K[:3].sum(axis=1), 1)
        share = K[:3, 0] / (K[:3, 0] + K[:3, 2])
        assert np.allclose(share, q3[:3, 0] / (q3[:3, 0] + q3[:3, 2]), atol=1e-8)
        assert np.allclose(1 - np.exp(-mu) * (1 + mu + mu ** 2 / 2), q_o, atol=1e-9)
    # one row solved alone is the same, bit for bit (every row on its own numbers)
    K, _ = dg.goals_implied(q3, q_o, -0.06)
    K1, _ = dg.goals_implied(q3[1:2], q_o[1:2], -0.06)
    assert np.array_equal(K[1], K1[0])
    # a lower goals line implies more draws for the same favourite
    Klo, _ = dg.goals_implied(q3[:1], np.array([0.40]), -0.06)
    Khi, _ = dg.goals_implied(q3[:1], np.array([0.65]), -0.06)
    assert Klo[0, 1] > Khi[0, 1]


def test_chosen_prices_every_priced_top_match_and_sums_to_one():
    m = _two_tiers()
    m.loc[m.index[400], "avg_o25"] = np.nan          # no over/under: the pick's model for that row
    m.loc[m.index[401], "avg_d"] = np.nan            # no 1X2: no candidate
    c = dg.predict_walk_forward(m)
    top = m["league"].isin(sd.TOP).to_numpy()
    fitted = top & (m["season"] >= 2008).to_numpy()  # 2005-2007 top-five matches: 1,140 >= MIN_TRAIN
    p = c[["p_h", "p_d", "p_a"]].to_numpy(float)
    assert np.isnan(p[~top]).all()
    assert "p_o25" not in c
    ok = fitted.copy()
    ok[401] = False
    assert np.isfinite(p[ok]).all()
    assert np.isnan(p[401]).all()
    assert np.allclose(p[ok].sum(axis=1), 1)
    assert len(c.attrs["weights"]) == 1 and len(c.attrs["weights"][2008]["1x2"]) == 7


def test_point_in_time_scramble():
    m = _two_tiers()
    cut = m.loc[(m["season"] == 2008), "date"].sort_values().iloc[150]
    r = dg.pit_check(m, cut)
    assert r["before_cut_predictions"] > 0 and r["cut_day_predictions"] > 0
    assert r["before_cut_unmoved"] and r["cut_day_unmoved_results_scrambled"]
    assert r["max_move_before_cut"] == 0.0


def test_metrics_refuse_confirm():
    m = _league()
    c = dg.predict_walk_forward(m)
    try:
        dg.compare(m, c, dg.shipped_pick(m), 2016, 2025)
    except ValueError:
        return
    raise AssertionError("compare must refuse CONFIRM seasons")
