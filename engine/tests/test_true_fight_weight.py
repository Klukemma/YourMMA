"""True Fight Weight features may only use what was known before the fight."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

import true_fight_weight as tfw

ARCHIVE = pd.DataFrame([
    ("2019-01-01", "welterweight", "Ann", "X", "Ann", 180, 170, 185, 175, "1990-01-01", "1990-01-01"),
    ("2021-01-01", "lightweight", "Ann", "Y", "Ann", 180, 170, 185, 175, "1990-01-01", "1990-01-01"),
    ("2022-01-01", "lightweight", "Bea", "Z", "Bea", 170, 170, 172, 175, "1995-01-01", "1990-01-01"),
], columns=["date", "division", "r_name", "b_name", "winner", "r_height",
            "b_height", "r_reach", "b_reach", "r_dob", "b_dob"])


def test_division_limits_prefer_the_longer_name():
    assert tfw.limit_of("Light Heavyweight") == 205
    assert tfw.limit_of("Heavyweight") == 265
    assert tfw.limit_of("Women's Strawweight") == 115
    assert np.isnan(tfw.limit_of("Catch Weight"))


def test_came_down_counts_only_earlier_divisions():
    a = tfw.Archive(ARCHIVE)
    # Before her 2019 welterweight bout Ann had fought nowhere heavier.
    assert a.features("Ann", "2019-01-01", 170, False)["came_down"] == 0
    # At lightweight in 2021 she had come down 15 lb.
    assert a.features("Ann", "2021-01-01", 155, False)["came_down"] == 15


def test_a_measured_regain_informs_only_later_fights():
    a = tfw.Archive(ARCHIVE)
    measured = pd.DataFrame({"key": ["ann"], "date": [pd.Timestamp("2021-01-01")],
                             "gain_pct": [12.0]})
    same_night = a.features("Ann", "2021-01-01", 155, False, measured)
    later = a.features("Ann", "2021-06-01", 155, False, measured)
    assert same_night["has_prior"] == 0
    assert later["has_prior"] == 1 and later["prior_regain"] == 12.0


def test_a_tall_fighter_for_the_division_is_above_zero():
    a = tfw.Archive(ARCHIVE)
    assert a.features("Ann", "2022-06-01", 155, False)["height_z"] > \
        a.features("Bea", "2022-06-01", 155, False)["height_z"]
