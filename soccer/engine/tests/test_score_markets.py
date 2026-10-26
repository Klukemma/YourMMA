"""The score matrix behind the richer match numbers (experiments/score_markets.py):
the matrix, the market-implied fit, what is read off it, the league baseline,
the confidence bands, point in time - and that exposing the model's expected
goals left the model's own numbers exactly as they were."""

import math

import numpy as np
import pandas as pd
import pytest

import soccer_data as sd
import soccer_model as smod
from experiments import score_markets as sm
from experiments import shots_ratings as sr

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford", "Burnley", "Watford", "Leeds", "Stoke"]
SMALL = dict(min_train=60, min_played=2)          # shots_ratings on a small made-up league


def _league(seasons=range(2005, 2011), seed=0):
    """One made-up top league with shots, results and prices that follow
    each club's strength (so the recalibration has something to fit)."""
    rng = np.random.default_rng(seed)
    strength = {t: s for t, s in zip(TEAMS, np.linspace(-0.4, 0.4, len(TEAMS)))}
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // 5))
            lam = math.exp(0.35 + strength[h] - strength[a])
            nu = math.exp(0.05 + strength[a] - strength[h])
            hg, ag = rng.poisson(lam), rng.poisson(nu)
            ph, pd_, pa, po = (float(v[0]) for v in sr.score_probs(np.array([lam]), np.array([nu]), -0.06))
            rows.append({"match_id": sd.match_id("E0", date.date(), h, a), "league": "E0", "season": season,
                         "date": date.strftime("%Y-%m-%d"), "time": "15:00", "kickoff": date + pd.Timedelta(hours=14),
                         "home": h, "away": a, "hg": hg, "ag": ag,
                         "result": "H" if hg > ag else "A" if hg < ag else "D",
                         "hs": hg * 4 + rng.integers(3, 9), "as": ag * 4 + rng.integers(2, 8),
                         "hst": hg + rng.integers(1, 4), "ast": ag + rng.integers(0, 3), "hr": 0, "ar": 0,
                         "avg_h": 0.95 / ph, "avg_d": 0.95 / pd_, "avg_a": 0.95 / pa,
                         "avg_o25": 0.95 / po, "avg_u25": 0.95 / (1 - po)})
    return sd.frame(rows)


# ------------------------------------------------------------- the matrix

def test_the_matrix_is_the_models_own_score_matrix():
    rng = np.random.default_rng(1)
    lam, nu = rng.uniform(0.3, 3.0, 50), rng.uniform(0.3, 2.5, 50)
    for rho in (-0.1, 0.0, 0.05):
        mk = sm.markets(sm.dc_matrix(lam, nu, rho))
        ph, pd_, pa, po = sr.score_probs(lam, nu, rho)
        assert np.allclose(mk["p_h"], ph, atol=1e-13) and np.allclose(mk["p_d"], pd_, atol=1e-13)
        assert np.allclose(mk["p_a"], pa, atol=1e-13) and np.allclose(mk["o25"], po, atol=1e-13)
        assert np.allclose(mk["cells"].sum(axis=1), 1.0)


def test_what_is_read_off_a_matrix_matches_a_hand_count():
    mat = np.zeros((11, 11))
    for (x, y), p in {(0, 0): 0.10, (1, 0): 0.20, (1, 1): 0.30, (2, 1): 0.15, (3, 2): 0.05, (6, 0): 0.20}.items():
        mat[x, y] = p
    s = sm.summary(mat)
    assert math.isclose(s["btts"], 0.30 + 0.15 + 0.05)
    assert math.isclose(s["o15"], 0.30 + 0.15 + 0.05 + 0.20)
    assert math.isclose(s["o25"], 0.15 + 0.05 + 0.20)
    assert math.isclose(s["o35"], 0.05 + 0.20)
    assert [(r["score"], round(r["p"], 10)) for r in s["scores"]] == [("1-1", 0.3), ("1-0", 0.2), ("2-1", 0.15)]
    assert math.isclose(s["other"], 0.20)                 # 6-0 is beyond the 0-5 grid
    mk = sm.markets(mat)
    assert math.isclose(mk["p_h"][0], 0.60) and math.isclose(mk["p_d"][0], 0.40) and mk["p_a"][0] == 0.0
    assert sm.summary(np.full((11, 11), np.nan)) is None


def test_independent_poissons_give_the_textbook_numbers():
    lam, nu = 1.6, 1.1
    mk = sm.markets(sm.dc_matrix(lam, nu, 0.0))
    assert math.isclose(mk["btts"][0], (1 - math.exp(-lam)) * (1 - math.exp(-nu)), abs_tol=1e-6)
    mu = lam + nu
    under = lambda k: sum(math.exp(-mu) * mu ** g / math.factorial(g) for g in range(k + 1))
    assert math.isclose(mk["o15"][0], 1 - under(1), abs_tol=1e-6)
    assert math.isclose(mk["o35"][0], 1 - under(3), abs_tol=1e-6)
    assert math.isclose(mk["cells"][0, 6 * 2 + 1], math.exp(-mu) * lam ** 2 / 2 * nu, abs_tol=1e-7)


def test_score_categories():
    assert list(sm.score_index([0, 2, 5, 6, 1], [0, 1, 5, 0, 7])) == [0, 13, 35, 36, 36]
    assert sm.score_label(13) == "2-1" and sm.score_label(36) == "other"


# ------------------------------------------------------------- the market-implied fit

def test_the_fit_reproduces_the_1x2_and_over_it_was_fitted_to():
    rng = np.random.default_rng(2)
    lam, nu = rng.uniform(0.5, 2.8, 40), rng.uniform(0.4, 2.2, 40)
    rho = rng.uniform(-0.1, -0.03, 40)
    ph, pd_, pa, po = (np.empty(40) for _ in range(4))
    for i in range(40):
        ph[i], pd_[i], pa[i], po[i] = (float(v[0]) for v in sr.score_probs(lam[i:i + 1], nu[i:i + 1], rho[i]))
    mats, info = sm.score_matrix(ph, pd_, pa, po, rho, lam * 0.7, nu * 1.3)   # started off the answer
    mk = sm.markets(mats)
    for got, want in ((mk["p_h"], ph), (mk["p_d"], pd_), (mk["p_a"], pa), (mk["o25"], po)):
        assert np.abs(got - want).max() < 1e-9
    assert np.allclose(info["lam"], lam, atol=1e-6) and np.allclose(info["nu"], nu, atol=1e-6)
    assert info["from_market"].all() and (info["rmse"] < 1e-9).all()

    # no over/under price: the 1X2 alone, still reproduced
    mats, info = sm.score_matrix(ph, pd_, pa, None, rho)
    mk = sm.markets(mats)
    assert np.abs(mk["p_h"] - ph).max() < 1e-8 and np.abs(mk["p_a"] - pa).max() < 1e-8


def test_a_market_no_matrix_can_reach_is_fitted_as_closely_as_it_can_and_says_so():
    # a 70% draw (an end-of-season Serie A price): the fit misses, and reports it
    mats, info = sm.score_matrix(0.2, 0.7, 0.1, 0.42, -0.06)
    assert info["from_market"][0] and info["rmse"][0] > 0.1
    assert np.isfinite(mats).all() and math.isclose(mats.sum(), 1.0)


def test_no_price_falls_back_to_the_models_own_matrix_and_nothing_is_missing_silently():
    mats, info = sm.score_matrix([np.nan, 0.5], [np.nan, 0.27], [np.nan, 0.23], [0.5, np.nan],
                                 [-0.06, -0.06], [1.4, 1.5], [1.0, 1.1])
    assert np.allclose(mats[0], sm.dc_matrix(1.4, 1.0, -0.06)[0])
    assert list(info["from_market"]) == [False, True] and np.isnan(info["rmse"][0])
    mats, info = sm.score_matrix(np.nan, np.nan, np.nan, None, -0.06)       # neither: missing
    assert np.isnan(mats).all() and not info["from_market"][0]


def test_a_matchs_numbers_do_not_depend_on_the_other_matches_fitted_beside_it():
    # Bit for bit: otherwise prices posted AFTER a match could move its numbers
    # in the last decimals (the point-in-time check caught exactly that).
    rng = np.random.default_rng(4)
    n = 300
    lam, nu, rho = rng.uniform(0.3, 3.0, n), rng.uniform(0.2, 2.5, n), rng.uniform(-0.1, -0.03, n)
    t = sm.markets(sm.dc_matrix(lam, nu, rho))
    p3 = np.column_stack([t["p_h"], t["p_d"], t["p_a"]]) * rng.uniform(0.97, 1.03, (n, 3))
    p3 /= p3.sum(axis=1, keepdims=True)
    p3[0] = [0.2, 0.7, 0.1]                                  # one fit no matrix can reach
    po = np.clip(t["o25"] + rng.normal(0, 0.02, n), 0.05, 0.95)
    batch = sm.implied_rates(p3, po, rho, lam, nu)
    for i in (0, 1, 150, n - 1):
        alone = sm.implied_rates(p3[i:i + 1], po[i:i + 1], rho[i:i + 1], lam[i:i + 1], nu[i:i + 1])
        assert all(a[0] == b[i] for a, b in zip(alone, batch))
    whole = sm.markets(sm.dc_matrix(batch[0], batch[1], rho))
    one = sm.markets(sm.dc_matrix(batch[0][:1], batch[1][:1], rho[:1]))
    assert all(np.array_equal(one[k][0], whole[k][0]) for k in whole)


def test_total_rate_inverts_the_poisson_over_25():
    mu = np.array([1.5, 2.6, 3.4])
    over = 1 - np.exp(-mu) * (1 + mu + mu ** 2 / 2)
    assert np.allclose(sm.total_rate(over), mu, atol=1e-9)


# ------------------------------------------------------------- baseline and bands

def test_the_baseline_uses_earlier_seasons_only():
    m = _league(range(2005, 2008))
    b = sm.baseline(m)
    assert b[m["season"] == 2005].isna().all().all()
    y07 = m["season"] == 2007
    early = m[m["season"] < 2007]
    hg, ag = early["hg"].astype(float), early["ag"].astype(float)
    assert np.allclose(b.loc[y07, "btts"], ((hg > 0) & (ag > 0)).mean())
    assert np.allclose(b.loc[y07, "o35"], (hg + ag > 3.5).mean())
    assert np.allclose(b.loc[y07, [f"c{k}" for k in range(sm.N_CELLS)]].sum(axis=1), 1.0)
    changed = m.copy()
    changed.loc[y07, ["hg", "ag"]] = 7                    # this season's own results do not count
    assert sm.baseline(changed)[y07].equals(b[y07])


def test_bands():
    assert [sm.band_of(p) for p in (0.72, 0.60, 0.55, 0.45, 0.399, float("nan"))] == \
        ["STRONG", "STRONG", "SOLID", "LEAN", "TOSS-UP", None]
    edges, raw = sm.band_edges(np.linspace(0.32, 0.82, 101))      # quartiles 0.445, 0.57, 0.695
    assert edges == [0.45, 0.55, 0.70] and raw == pytest.approx([0.445, 0.57, 0.695])


# ------------------------------------------------------------- the model, unchanged

def test_exposing_the_models_expected_goals_changes_none_of_its_numbers():
    m = _league()
    off = sr.predict_walk_forward(m, **SMALL)
    on = sr.predict_walk_forward(m, score_params=True, **SMALL)
    assert list(off.columns) == ["p_h", "p_d", "p_a", "p_o25"]
    assert list(on.columns) == ["p_h", "p_d", "p_a", "p_o25"] + sr.SCORE_PARAMS
    assert off["p_h"].notna().sum() > 200
    assert off.to_numpy().tobytes() == on[off.columns].to_numpy().tobytes()      # bit for bit
    ok = on["p_h"].notna()
    for r in on.loc[ok, "rho"].unique():
        s = ok & (on["rho"] == r)
        again = np.column_stack(sr.score_probs(on.loc[s, "lam"].to_numpy(), on.loc[s, "nu"].to_numpy(), r))
        assert np.allclose(again, on.loc[s, ["p_h", "p_d", "p_a", "p_o25"]].to_numpy(), atol=1e-14)
    assert on.loc[~ok, sr.SCORE_PARAMS].isna().all().all()


# ------------------------------------------------------------- the whole run, point in time

def test_the_walk_forward_is_point_in_time(monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 60)
    m = _league()
    wf = sm.walk_forward(m, model_params=SMALL)
    assert wf["info"]["from_market"].sum() > 200
    k = wf["K"]
    assert np.allclose(k[[f"c{i}" for i in range(sm.N_CELLS)]].sum(axis=1), 1.0)
    cut = m.loc[m["season"] == 2009, "date"].iloc[60]
    r = sm.point_in_time(m, cut, wf, model_params=SMALL)
    assert r["check_point_in_time"] and r["unmoved_on_or_before_cut"] and r["unmoved_before_cut_prices_scrambled"]
    assert r["numbers_compared"] > 1000


def test_the_point_in_time_guard_catches_a_baseline_that_peeks(monkeypatch):
    monkeypatch.setattr(smod, "MIN_TRAIN", 60)
    honest = sm.baseline

    def peeking(matches):
        out = honest(matches)
        hg, ag = matches["hg"].astype(float), matches["ag"].astype(float)
        out["btts"] = float(((hg > 0) & (ag > 0)).mean())     # every season's results, the future's too
        return out
    monkeypatch.setattr(sm, "baseline", peeking)
    m = _league()
    wf = sm.walk_forward(m, model_params=SMALL)
    cut = m.loc[m["season"] == 2009, "date"].iloc[60]
    r = sm.point_in_time(m, cut, wf, model_params=SMALL)
    assert not r["unmoved_on_or_before_cut"] and not r["unmoved_before_cut_prices_scrambled"]
