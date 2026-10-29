"""The numbers beside the pick, as the app gets them (soccer_scores): the
matrix reproduces what it was fitted to, the rule that picks the market's
matrix or the model's own, what a card reads off it, the bands, and the
measured table - which, run by the experiment's rule, is the experiment's
own measurement number for number."""

import math

import numpy as np
import pytest
from test_score_markets import SMALL, _league

import soccer_model as smod
import soccer_scores as ssc
from experiments import score_markets as sm
from experiments import shots_ratings as sr


def _priced(n=30, seed=7):
    """n matches whose 1X2 and over 2.5 come from a known Dixon-Coles matrix."""
    rng = np.random.default_rng(seed)
    lam, nu, rho = rng.uniform(0.5, 2.8, n), rng.uniform(0.4, 2.2, n), rng.uniform(-0.1, -0.03, n)
    t = ssc.markets(ssc.dc_matrix(lam, nu, rho))
    return lam, nu, rho, t["p_h"], t["p_d"], t["p_a"], t["o25"]


# ------------------------------------------------------------- one code, two users

def test_the_experiment_runs_the_apps_code():
    for name in ("dc_matrix", "markets", "summary", "band_of", "implied_rates", "score_matrix", "baseline",
                 "score_index", "score_label"):
        assert getattr(sm, name) is getattr(ssc, name)
    assert sm.BANDS is ssc.BANDS and ssc.MAX_GOALS == sr.MAX_GOALS


# ------------------------------------------------------------- the matrix reproduces its targets

def test_the_market_matrix_reproduces_the_1x2_and_over_it_was_fitted_to():
    lam, nu, rho, ph, pd_, pa, po = _priced()
    mats, info = ssc.shown_matrices(ph, pd_, pa, po, rho, lam * 0.6, nu * 1.4)      # the model's rates are off
    mk = ssc.markets(mats)
    for got, want in ((mk["p_h"], ph), (mk["p_d"], pd_), (mk["p_a"], pa), (mk["o25"], po)):
        assert np.abs(got - want).max() < 1e-9
    assert (info["source"] == "market").all() and all(r is None for r in info["reason"])
    assert (info["miss"] < 1e-9).all() and np.allclose(info["lam"], lam, atol=1e-6)


def test_a_slightly_inconsistent_market_is_fitted_within_the_rule_and_its_miss_is_reported():
    lam, nu, rho, ph, pd_, pa, po = _priced(seed=3)
    pd2 = pd_ + 0.02                                  # the recalibrated draw, priced a little longer
    p3 = np.column_stack([ph, pd2, pa]) / (1 + 0.02)
    mats, info = ssc.shown_matrices(p3[:, 0], p3[:, 1], p3[:, 2], po, rho, lam, nu)
    got = ssc.markets(mats)
    worst = np.abs(np.column_stack([got["p_h"], got["p_d"], got["p_a"], got["o25"]])
                   - np.column_stack([p3, po])).max(axis=1)
    assert np.allclose(info["miss"], worst, atol=1e-15) and (info["miss"] < ssc.FIT_MAX_MISS).all()
    assert (info["source"] == "market").all() and set(info["miss_on"]) <= {0, 1, 2, 3}


# ------------------------------------------------------------- the rule

def test_the_rule_market_where_both_are_priced_the_model_where_not():
    lam, nu, rho, ph, pd_, pa, po = _priced(4, seed=1)
    ph, pd_, pa, po = ph.copy(), pd_.copy(), pa.copy(), po.copy()
    po[1] = np.nan                                    # the 1X2 alone
    ph[2] = pd_[2] = pa[2] = np.nan                   # no price (the over/under alone is not enough)
    mats, info = ssc.shown_matrices(ph, pd_, pa, po, rho, lam * 0.8, nu * 1.2)
    assert list(info["source"]) == ["market", "model", "model", "market"]
    assert list(info["reason"]) == [None, ssc.NO_OVER_UNDER, ssc.NO_PRICE, None]
    own = ssc.dc_matrix(lam * 0.8, nu * 1.2, rho)
    for i in (1, 2):                                  # the model's own matrix, bit for bit
        assert np.array_equal(mats[i], own[i]) and math.isnan(info["miss"][i]) and info["miss_on"][i] == -1


def test_a_fit_that_misses_a_target_badly_is_kept_and_flagged_with_how_badly():
    # A 70% draw beside a 42% over 2.5 (an end-of-season Serie A price): no
    # matrix reaches both, the closest misses by far more than FIT_MAX_MISS.
    # Measured, the model's own matrix misses those prices by more still and
    # scores worse (the notes): the market's is shown, flagged.
    mats, info = ssc.shown_matrices([0.2, 0.5], [0.7, 0.27], [0.1, 0.23], [0.42, 0.52], -0.06, [1.3, 1.5], [0.9, 1.1])
    assert list(info["source"]) == ["market", "market"] and list(info["reason"]) == [None, None]
    assert list(info["flagged"]) == [True, False]
    assert info["miss"][0] > ssc.FIT_MAX_MISS and ssc.TARGET_NAMES[info["miss_on"][0]] in ssc.TARGET_NAMES
    k, _ = ssc.score_matrix(0.2, 0.7, 0.1, 0.42, -0.06, 1.3, 0.9)
    assert np.array_equal(mats[0], k[0])
    c = ssc.card(mats[0], info["source"][0], info["reason"][0], info["miss"][0], info["miss_on"][0])
    assert c["flagged"] and c["miss"] == info["miss"][0]
    assert not ssc.card(mats[1], "market", None, info["miss"][1], info["miss_on"][1])["flagged"]


def test_no_model_rates_means_no_numbers_shown_as_missing():
    mats, info = ssc.shown_matrices([0.5, np.nan], [0.27, np.nan], [0.23, np.nan], [0.5, np.nan],
                                    np.nan, np.nan, np.nan)
    assert np.isnan(mats).all() and list(info["source"]) == [None, None]
    assert list(info["reason"]) == [ssc.NO_MODEL, ssc.NO_MODEL]
    assert ssc.card(mats[0], info["source"][0], info["reason"][0]) is None


def test_the_experiments_rule_is_the_measured_matrix_bit_for_bit():
    lam, nu, rho, ph, pd_, pa, po = _priced(6, seed=5)
    po = po.copy()
    po[2] = np.nan
    want, _ = ssc.score_matrix(ph, pd_, pa, po, rho, lam, nu)
    got, info = ssc.shown_matrices(ph, pd_, pa, po, rho, lam, nu, need_o25=False)
    assert np.array_equal(got, want) and (info["source"] == "market").all()


# ------------------------------------------------------------- what a card reads off it

def test_the_card_reads_the_summary_and_the_matrixs_own_1x2():
    mat = np.zeros((11, 11))
    for (x, y), p in {(0, 0): 0.10, (1, 0): 0.20, (1, 1): 0.30, (2, 1): 0.15, (3, 2): 0.05, (6, 0): 0.20}.items():
        mat[x, y] = p
    c = ssc.card(mat, "market", None, 0.0123, 1)
    assert math.isclose(c["btts"], 0.50) and math.isclose(c["o15"], 0.70) and math.isclose(c["o35"], 0.25)
    assert math.isclose(c["o25"], 0.40) and math.isclose(c["other"], 0.20)
    assert [s["score"] for s in c["scores"]] == ["1-1", "1-0", "2-1"]
    assert c["matrix"] == pytest.approx({"p_h": 0.60, "p_d": 0.40, "p_a": 0.0})
    assert (c["source"], c["reason"], c["miss"], c["miss_on"]) == ("market", None, 0.0123, "draw")
    m = ssc.card(mat, "model", ssc.NO_PRICE, float("nan"), -1)
    assert m["miss"] is None and m["miss_on"] is None and m["reason"] == ssc.NO_PRICE


def test_the_models_own_matrix_is_the_models_own_chances():
    lam, nu = np.array([1.7, 0.9]), np.array([1.0, 1.4])
    mk = ssc.markets(ssc.dc_matrix(lam, nu, -0.07))
    ph, pd_, _pa, po = sr.score_probs(lam, nu, -0.07)
    assert np.allclose(mk["p_h"], ph, atol=1e-13) and np.allclose(mk["p_d"], pd_, atol=1e-13)
    assert np.allclose(mk["o25"], po, atol=1e-13)


# ------------------------------------------------------------- the bands

def test_band_of_at_the_edges():
    assert [ssc.band_of(p) for p in (0.95, 0.60, 0.5999, 0.50, 0.4999, 0.40, 0.3999, 0.2)] == \
        ["STRONG", "STRONG", "SOLID", "SOLID", "LEAN", "LEAN", "TOSS-UP", "TOSS-UP"]
    assert ssc.band_of(None) is None and ssc.band_of(float("nan")) is None
    assert [f for _, f in ssc.BANDS] == [0.60, 0.50, 0.40, 0.0]


# ------------------------------------------------------------- the measured table

@pytest.fixture(scope="module")
def league():
    smod_min = smod.MIN_TRAIN
    smod.MIN_TRAIN = 60
    try:
        m = _league()
        wf = sm.walk_forward(m, model_params=SMALL)
        pick, _ = smod.pick(m, wf["preds"])
        yield m, wf, pick
    finally:
        smod.MIN_TRAIN = smod_min


def test_the_measured_table_is_the_experiments_measurement_by_its_rule(league):
    m, wf, pick = league
    periods = ((2007, 2008), (2009, 2010))
    walked = ssc.walk(m, wf["preds"], pick, need_o25=False)
    got = ssc.measure(m, wf["preds"], pick, walked, periods=periods)
    for lo, hi in periods:
        rep = sm.evaluate(m, wf, (lo, hi))
        mine = got["periods"][f"{lo}-{hi}"]
        assert mine["matches"] == rep["matches"] > 50
        for t in ("btts", "o15", "o35"):
            assert mine[t]["rate"] == rep[t]["rate"]
            for a, b in (("shown", "K"), ("model", "M"), ("base", "B")):
                assert mine[t][a]["log_loss"] == rep[t][b]["log_loss"] and mine[t][a]["mean_p"] == rep[t][b]["mean_p"]
            assert mine[t]["shown"]["vs_base"] == rep[t]["K"]["vs_B"]
        for a, b in (("shown", "K"), ("model", "M"), ("base", "B")):
            for f in ("log_loss", "top1_hit", "top1_mean_p", "top3_hit"):
                assert mine["exact"][a][f] == rep["exact"][b][f]
        sel = m.loc[wf["pick"].index, "season"].between(lo, hi) & m.loc[wf["pick"].index, "result"].isin(("H", "D", "A"))
        bands = sm.band_report(m, wf, wf["pick"].index[sel.to_numpy()])
        for b, _ in ssc.BANDS:
            for f in ("picks", "hit_rate", "mean_prob"):
                assert got["bands"]["pick"][f"{lo}-{hi}"][b][f] == bands[b][f]


def test_the_model_alone_bands_use_the_same_edges_on_the_models_own_picks(league):
    m, wf, pick = league
    got = ssc.measure(m, wf["preds"], pick, periods=((2007, 2010),))
    preds = wf["preds"]
    rows = preds["p_h"].notna() & m["result"].isin(("H", "D", "A")) & m["season"].between(2007, 2010)
    p = preds.loc[rows, ["p_h", "p_d", "p_a"]].to_numpy()
    hit = np.array(["H", "D", "A"])[p.argmax(axis=1)] == m.loc[rows, "result"].to_numpy()
    top = p.max(axis=1)
    table = got["bands"]["model"]["2007-2010"]
    for b, floor in ssc.BANDS:
        s = np.array([ssc.band_of(x) == b for x in top])
        assert table[b]["picks"] == int(s.sum())
        if s.any():
            assert table[b]["hit_rate"] == round(float(hit[s].mean()), 4)
    assert sum(table[b]["picks"] for b, _ in ssc.BANDS) == int(rows.sum())


def test_the_shipped_rule_is_measured_and_counts_its_fallbacks(league):
    m, wf, pick = league
    q = m.copy()
    late = q.index[q["season"] == 2010][:15]
    q.loc[late, ["avg_o25", "avg_u25"]] = np.nan          # fifteen matches with the 1X2 alone
    got = ssc.measure(q, wf["preds"], pick, periods=((2009, 2010),))
    r = got["periods"]["2009-2010"]
    assert r["model"][ssc.NO_OVER_UNDER] == 15 and r["market"] + sum(r["model"].values()) == r["matches"]
    assert r["btts"]["deciles"] >= 5 and 0 <= r["btts"]["deciles_low"] <= r["btts"]["deciles"]
    for t in ("btts", "o15", "o35", "exact"):
        assert all(np.isfinite(r[t][s]["log_loss"]) for s in ("shown", "model", "base"))
