"""The shared factor test must not flatter the populations it measures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from experiments import residual_harness as rh


def fights(n=4000, seed=1, age_effect=-0.08):
    """Fights where being 3+ years older costs age_effect, and nothing else
    matters. Half the fights have no price."""
    rng = np.random.default_rng(seed)
    red_age = rng.uniform(22, 38, n)
    blue_age = rng.uniform(22, 38, n)
    p = rng.uniform(0.2, 0.8, n)
    diff = red_age - blue_age
    truth = np.clip(p + age_effect * (diff >= 3) - age_effect * (diff <= -3),
                    0.01, 0.99)
    won = (rng.uniform(size=n) < truth).astype(float)
    frame = pd.DataFrame({
        "date": pd.Timestamp("2020-01-01"), "event": [f"E{i // 10}" for i in range(n)],
        "red": [f"r{i % 300}" for i in range(n)],
        "blue": [f"b{i % 300}" for i in range(n)],
        "red_raw": [f"R{i % 300}" for i in range(n)],
        "blue_raw": [f"B{i % 300}" for i in range(n)],
        "won": won, "p_model": p, "red_age": red_age, "blue_age": blue_age})
    frame["p_market"] = np.where(np.arange(n) % 2 == 0, p, np.nan)
    frame["p_blend"] = frame["p_market"]
    return frame


def test_unpriced_fights_are_in_neither_arm():
    frame = fights(200)
    rows = rh.sides(frame)
    assert len(rows) == 2 * int(np.isfinite(frame["p_blend"]).sum())


def test_an_older_population_shows_the_age_effect_only_when_not_controlled():
    frame = fights()
    rows = rh.sides(frame)
    older = rh.mark(frame, rows, lambda f: "red" if f.red_age - f.blue_age >= 3
                    else ("blue" if f.blue_age - f.red_age >= 3 else None))
    raw = rh.measure(rows, older, "older, raw", control_age=False)
    assert raw["blend_adjusted"] < -0.04 and raw["blend_high"] < 0
    # A signal that is merely CORRELATED with age - here, half of the older
    # fighters, at random - must come out near zero once age is controlled,
    # because the other half of its age band carries the same age effect.
    half = older & (np.arange(len(rows)) % 2 == 0)
    confounded = rh.measure(rows, half, "age-confounded, raw", control_age=False)
    controlled = rh.measure(rows, half, "age-confounded, controlled")
    assert confounded["blend_adjusted"] < -0.04
    assert abs(controlled["blend_adjusted"]) < 0.03


def test_a_population_that_is_exactly_an_age_band_cannot_be_age_controlled():
    frame = fights()
    rows = rh.sides(frame)
    older = rh.mark(frame, rows, lambda f: "red" if f.red_age - f.blue_age >= 3
                    else ("blue" if f.blue_age - f.red_age >= 3 else None))
    got = rh.measure(rows, older, "older, controlled")
    assert got.get("too_few") or got["blend_matched"] < 0.5


def test_a_population_of_few_fighters_is_not_reported():
    frame = fights()
    rows = rh.sides(frame)
    few = rh.mark(frame, rows, lambda f: "red" if f.red in ("r1", "r2") else None)
    assert rh.measure(rows, few, "two fighters").get("too_few")


def test_family_wise_intervals_are_wider():
    frame = fights()
    rows = rh.sides(frame)
    older = rh.mark(frame, rows, lambda f: "red" if f.red_age - f.blue_age >= 3
                    else None)
    one = rh.measure(rows, older, "one", control_age=False)
    eight = rh.measure(rows, older, "eight", family=8, control_age=False)
    assert (eight["blend_high"] - eight["blend_low"]) > \
        (one["blend_high"] - one["blend_low"])
