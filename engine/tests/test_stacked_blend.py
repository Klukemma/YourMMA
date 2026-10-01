"""The stacking experiment's bar, walk-forward and pre-registration are
tested, not eyeballed.

The line that decides whether anything ships is `passes_bar`; a one-token
slip there (low < 0 instead of high < 0) would have declared every arm a
pass. These tests pin that line, prove each year's stack and reference
weight are fitted on earlier priced years only, and tie the plan the JSON
reports to the plan that was registered.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments import model_compare as mc  # noqa: E402
from experiments import stacked_blend as sb  # noqa: E402


def priced_frame(n_per_year=150, years=range(2010, 2017), seed=0, market_noise=0.08,
                 model_noise=0.18):
    """Priced walk-forward rows: the market is the sharper forecast, the
    model a noisier one, outcomes drawn from the truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for i in range(n_per_year):
            truth = rng.uniform(0.2, 0.8)
            rows.append({
                "fight_id": f"{year}-{i}", "year": year,
                "date": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(i * 2)),
                "event": f"E{year}-{i // 12}",
                "red_norm": f"r{rng.integers(0, 300)}", "blue_norm": f"b{rng.integers(0, 300)}",
                "red_prior_bouts": int(rng.integers(0, 12)),
                "blue_prior_bouts": int(rng.integers(0, 12)),
                "y": float(rng.uniform() < truth), "truth": truth,
                "p_market": float(np.clip(truth + rng.normal(0, market_noise), 0.02, 0.98)),
                "p_model": float(np.clip(truth + rng.normal(0, model_noise), 0.02, 0.98)),
            })
    return pd.DataFrame(rows)


# --- the design ------------------------------------------------------------

def test_thinness_is_one_for_a_debutant_and_shrinks_with_the_thinner_record():
    frame = pd.DataFrame({"red_prior_bouts": [0, 1, 10, 7], "blue_prior_bouts": [5, 1, 10, 0]})
    assert list(sb.thinness(frame)) == pytest.approx([1.0, 0.5, 1 / 11, 1.0])


def test_designs_have_the_pre_registered_columns_and_clip_before_the_logit():
    frame = priced_frame(n_per_year=5, years=[2012])
    frame.loc[0, "p_market"] = 0.0           # would be -inf without the clip
    assert sb.design_stack(frame).shape == (5, 2)
    assert sb.design_thin(frame).shape == (5, 3)
    assert sb.design_fl(frame).shape == (5, 3)
    assert np.isfinite(sb.design_stack(frame)).all()
    lk = sb.logit(frame["p_market"])
    assert sb.design_fl(frame)[:, 2] == pytest.approx(lk * np.abs(lk))
    assert sb.design_thin(frame)[:, 2] == pytest.approx(sb.logit(frame["p_model"]) * sb.thinness(frame))
    assert set(sb.ARMS) == {"stack", "stack+thin", "stack+FL"} and sb.FAMILY == 3
    assert sb.ALPHA == pytest.approx(0.05 / 3)


# --- point in time ---------------------------------------------------------

def test_each_years_stack_is_fitted_on_earlier_priced_years_only(monkeypatch):
    priced = priced_frame()
    seen = []
    real_fit = sb.fit_arm

    def spy(design, frame):
        seen.append((int(frame["year"].max()), len(frame)))
        return real_fit(design, frame)

    monkeypatch.setattr(sb, "fit_arm", spy)
    monkeypatch.setattr(sb, "MIN_STACK_TRAIN", 200)
    p, coefficients = sb.walk_forward_arm("stack", priced, verbose=False)
    years = priced["year"].to_numpy()
    # 2010 (0 earlier) and 2011 (150 earlier) are below the 200 bar; 2012 on are scored
    assert np.isnan(p[years <= 2011]).all()
    assert np.isfinite(p[years >= 2012]).all()
    assert sorted(coefficients) == list(range(2012, 2017))
    assert len(seen) == 5
    for (max_year, n), year in zip(seen, range(2012, 2017)):
        assert max_year < year, f"the stack for {year} saw year {max_year}"
        assert n == int((years < year).sum()), "the window is every earlier priced fight"
        assert coefficients[year]["fitted_on"] == n


def test_the_reference_weight_is_chosen_on_earlier_years_only_by_log_loss(monkeypatch):
    priced = priced_frame()
    seen = []
    real = sb.best_weight

    def spy(frame, metric="log_loss", grid=sb.WEIGHT_GRID):
        seen.append((int(frame["year"].max()), metric))
        return real(frame, metric, grid)

    monkeypatch.setattr(sb, "best_weight", spy)
    p, weights = sb.walk_forward_weight(priced, "log_loss", verbose=False)
    years = priced["year"].to_numpy()
    assert np.isnan(p[years <= 2011]).all() and np.isfinite(p[years >= 2012]).all()
    assert [m for _, m in seen] == ["log_loss"] * 5
    for (max_year, _), year in zip(seen, sorted(weights)):
        assert max_year < year
    # the market is the sharper forecast here, so the chosen weight leans to it
    assert all(0.5 <= w["weight"] <= 1.0 for w in weights.values())
    for year, w in weights.items():
        test = years == year
        assert p[test] == pytest.approx(sb.weighted_blend(
            w["weight"], priced.loc[test, "p_model"], priced.loc[test, "p_market"]))


def test_best_weight_minimises_the_metric_it_is_asked_for():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.2, 0.8, 4000)
    frame = pd.DataFrame({"y": (rng.uniform(size=4000) < truth).astype(float),
                          "p_market": truth,                                 # exact
                          "p_model": np.clip(truth + rng.normal(0, 0.3, 4000), 0.02, 0.98)})
    assert sb.best_weight(frame, "log_loss") >= 0.9
    frame["p_market"], frame["p_model"] = frame["p_model"], frame["p_market"]
    assert sb.best_weight(frame, "log_loss") <= 0.1


# --- the bar ---------------------------------------------------------------

def test_passes_bar_needs_the_wider_upper_bound_below_zero():
    assert sb.passes_bar({"mean": -0.004, "low": -0.009, "high": -0.001})
    assert not sb.passes_bar({"mean": -0.004, "low": -0.009, "high": 0.002})
    assert not sb.passes_bar({"mean": -0.004, "low": -0.009, "high": 0.0})
    assert not sb.passes_bar({"mean": 0.004, "low": 0.001, "high": 0.009})


def _table_frame(**noise):
    priced = priced_frame(n_per_year=400, years=[2015, 2016], seed=5, **noise)
    priced["p_baseline"] = sb.baseline(priced)
    return priced


def test_table_passes_only_when_both_resampled_intervals_sit_below_zero(monkeypatch, capsys):
    frame = _table_frame()
    frame["p_x"] = frame["p_baseline"]
    monkeypatch.setitem(sb.ARMS, "x", (sb.design_stack, ["a_market", "b_model"]))
    try:
        crafted = {}

        def fake_compare(y, p_base, p_cand, events, **kw):
            e, f = crafted["event"], crafted["fighter"]
            d = {"mean": -0.004, "event_low": e[0], "event_high": e[1],
                 "fighter_low": f[0], "fighter_high": f[1],
                 "low": min(e[0], f[0]), "high": max(e[1], f[1])}
            return {"n": len(y), "alpha": kw.get("alpha"), "events": 1,
                    "baseline": mc.metrics(y, p_base), kw["label_cand"]: mc.metrics(y, p_cand),
                    "delta": {"log_loss": d, "brier": d, "accuracy": d}}

        monkeypatch.setattr(mc, "compare", fake_compare)
        crafted.update(event=(-0.009, -0.001), fighter=(-0.007, +0.001))   # fighter crosses
        assert sb.table(frame, ["x"], "t")["arms"]["x"]["passes"] is False
        crafted.update(event=(-0.009, +0.001), fighter=(-0.007, -0.001))   # event crosses
        assert sb.table(frame, ["x"], "t")["arms"]["x"]["passes"] is False
        crafted.update(event=(-0.009, -0.001), fighter=(-0.007, -0.0005))  # both below
        assert sb.table(frame, ["x"], "t")["arms"]["x"]["passes"] is True
        # the reference baseline never decides anything
        frame["p_pit"] = frame["p_baseline"]
        r = sb.table(frame, ["x"], "t", base_col="p_pit", base_name="pit-weight", decides=False)
        assert r["arms"]["x"]["passes"] is None and r["decides"] is False
    finally:
        pass
    capsys.readouterr()


def test_table_on_real_arms_passes_a_clearly_better_candidate_and_not_an_identical_one(capsys):
    frame = _table_frame(market_noise=0.3, model_noise=0.3)   # a poor baseline
    rng = np.random.default_rng(2)
    truth = frame["truth"].to_numpy()           # an arm that almost knows the truth
    frame["p_better"] = np.clip(truth + rng.normal(0, 0.02, len(frame)), 0.02, 0.98)
    frame["p_same"] = frame["p_baseline"]
    results = sb.table(frame, ["better", "same"], "t", alpha=sb.ALPHA)
    assert results["arms"]["better"]["passes"] is True
    assert results["arms"]["same"]["passes"] is False
    assert results["arms"]["same"]["delta"]["log_loss"]["mean"] == 0.0
    assert results["baseline_arm"] == "baseline 0.75" and results["decides"] is True
    capsys.readouterr()


def test_baseline_arm_is_exactly_market_blend_at_0_75():
    import market_blend
    frame = priced_frame(n_per_year=20, years=[2014])
    assert market_blend.MARKET_WEIGHT == 0.75
    assert sb.baseline(frame) == pytest.approx(
        sb.weighted_blend(0.75, frame["p_model"], frame["p_market"]))


# --- pre-registration ------------------------------------------------------

def test_the_registered_plan_hash_matches_the_docstring_and_the_reported_json():
    registered = json.loads(sb.PLAN_FILE.read_text())
    assert registered["sha256"] == sb.plan_sha256()
    out = sb.OUT
    if out.exists():
        reported = json.loads(out.read_text())
        assert reported["plan"] == sb.__doc__
        assert sb.plan_text(reported["plan"]) == sb.plan_text()
        assert reported.get("plan_sha256") == registered["sha256"]
        # the bar in the JSON is the bar in the plan: 3 arms, alpha 0.05/3
        assert reported["family"] == 3 and reported["alpha"] == pytest.approx(0.05 / 3)
        for name, arm in reported["all_years"]["arms"].items():
            assert arm["passes"] == (arm["delta"]["log_loss"]["high"] < 0)


def test_check_plan_registers_once_and_refuses_a_changed_plan(tmp_path):
    sidecar = tmp_path / "plan.sha256"
    first = sb.check_plan(sidecar)
    assert first["sha256"] == sb.plan_sha256()
    assert sb.check_plan(sidecar)["sha256"] == first["sha256"]
    moved = sb.__doc__.replace("alpha = 0.05 / 3", "alpha = 0.05")
    assert sb.plan_text(moved) != sb.plan_text()
    with pytest.raises(SystemExit):
        sb.check_plan(sidecar, doc=moved)
    # text after the plan is free to change without touching the registration
    annotated = sb.__doc__ + "\nmore notes\n"
    assert sb.check_plan(sidecar, doc=annotated)["sha256"] == first["sha256"]
