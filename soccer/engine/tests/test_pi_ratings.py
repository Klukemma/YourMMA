"""Pi-ratings (experiments/pi_ratings.py): the update is Constantinou &
Fenton's, the pooling's alpha-0 control is the shipped recalibration, and
nothing dated before a cut moves when results, statistics and prices from
the cut on are scrambled."""

import math

import numpy as np
import pandas as pd

import soccer_data as sd
import soccer_model as smod
from experiments import pi_ratings as pr

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford"]
SMALL = dict(min_map=40, min_pool=40, warm_from="2000-01-01")


def _league(seasons=range(2005, 2011), seed=0):
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, np.linspace(-0.5, 0.5, len(TEAMS))))
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // 4))
            lam, nu = math.exp(0.3 + strength[h] - strength[a]), math.exp(strength[a] - strength[h])
            hg, ag = rng.poisson(lam), rng.poisson(nu)
            ph = 1 / (1 + math.exp(-(0.4 + 1.6 * (strength[h] - strength[a]))))
            pdr = 0.26
            pa = max(0.05, 1 - ph - pdr)
            ph = 1 - pdr - pa
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "home": h, "away": a, "hg": hg, "ag": ag,
                         "result": "H" if hg > ag else "A" if hg < ag else "D",
                         "hs": hg * 4 + 5, "as": ag * 4 + 4, "hst": hg + 2, "ast": ag + 1,
                         "avg_h": 0.95 / ph, "avg_d": 0.95 / pdr, "avg_a": 0.95 / pa,
                         "avg_o25": 1.9, "avg_u25": 1.9})
    return sd.frame(rows)


def test_one_update_is_the_papers():
    m = _league(seasons=[2005]).iloc[:1].copy()
    m.loc[m.index[0], ["hg", "ag", "result"]] = [3, 0, "H"]
    nxt = m.copy()
    nxt["date"] = "2005-08-20"
    both = pd.concat([m, nxt], ignore_index=True)
    p = dict(lam=0.035, gamma=0.7, c=3.0, lam_new=0.0, h0=0.0)
    f = pr.rate(both, **p)
    # both clubs start at 0: expected goal difference 0, error 3
    psi = 3.0 * math.log10(1 + 3)
    assert np.isclose(f.loc[1, "r_hh"], 0.035 * psi)
    assert np.isclose(f.loc[1, "r_ha"], 0.7 * 0.035 * psi)
    assert np.isclose(f.loc[1, "r_aa"], -0.035 * psi)
    assert np.isclose(f.loc[1, "r_ah"], -0.7 * 0.035 * psi)
    assert f.loc[0, ["r_hh", "r_ha", "r_ah", "r_aa"]].abs().max() == 0


def test_alpha_zero_control_is_the_shipped_recalibration():
    rng = np.random.default_rng(3)
    q = rng.dirichlet([4, 3, 3], 3000)
    idx = np.array([rng.choice(3, p=row) for row in q])
    w = pr.fit_pooled(np.log(q), np.zeros_like(q), idx, fix_alpha=0.0)
    res = pd.Series(np.array(["H", "D", "A"])[idx])
    ws = smod.fit_recalibration(pd.DataFrame(q, columns=["p_h", "p_d", "p_a"]), res)
    assert np.allclose([w[0], w[2], w[3]], ws, atol=1e-5) and w[1] == 0.0


def test_nothing_before_the_cut_moves():
    m = _league()
    cut = "2009-10-05"
    a = pr.predict_walk_forward(m, **SMALL)
    b = pr.predict_walk_forward(pr.scramble(m, cut), **SMALL)
    before = (m["date"] < cut).to_numpy()
    cols = ["p_h", "p_d", "p_a", "pi_h", "pi_d", "pi_a"]
    assert np.isfinite(a.loc[before, "p_h"]).sum() > 50
    assert np.allclose(a.loc[before, cols].to_numpy(float), b.loc[before, cols].to_numpy(float), equal_nan=True)
    on = (m["date"] == cut).to_numpy()
    assert on.any()
    assert np.allclose(a.loc[on, cols[3:]].to_numpy(float), b.loc[on, cols[3:]].to_numpy(float), equal_nan=True)
    after = (m["date"] > cut).to_numpy()
    assert not np.allclose(a.loc[after, "pi_h"].to_numpy(float), b.loc[after, "pi_h"].to_numpy(float))
